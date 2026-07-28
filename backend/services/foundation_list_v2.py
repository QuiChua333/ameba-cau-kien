"""foundation_list_v2.py — per-foundation 基礎リスト reader (new-template sheets).

Older sheets put every foundation in ONE page-wide schedule with a 備考 column
(handled by table_extractor). Newer sheets drop that table entirely and give each
foundation its OWN block:

    ┌─────────────────────────────────────────────┐
    │  F1A 基礎平面            F1A 基礎断面        │  ← per-type title
    │  ┌────────┬──────┬──────┬──────┬─────────┐  │
    │  │        │      │      │      │ ベース筋 │  │  ← NO 備考 column
    │  │基礎符号│Lx(mm)│Ly(mm)│ D(mm)│  ←  │ ↑ │  │
    │  │  F1A   │4,600 │4,600 │1,100~│46-D16│…  │  │  ← exactly ONE data row
    │  └────────┴──────┴──────┴──────┴─────────┘  │
    │            [cross-section drawing]           │
    │                  1,600                       │  ← bottom dimension stack:
    │                  3,600                       │     3 values → DD
    │                  4,600                       │     2 → D, 1 → TNF
    └─────────────────────────────────────────────┘

Two consequences drive this module:

  • The schedule reader can't help: each mini table has a single data row, and a
    page carries several of them with repeated headers.
  • Classification can't come from 備考 — there is none. It comes from the bottom
    dimension stack instead: a TNF footing is a flat slab and is dimensioned by its
    width alone (1 value), a D footing adds one step (2), a DD footing two (3). The
    stack's LAST (bottom, widest) value is the footing's full width, i.e. Lx — which
    makes it self-validating: a candidate stack is only accepted when its values
    grow downward and end exactly at the Lx read from the mini table.

Returns a table_extractor.TableExtractionResult so the rest of the pipeline (merge
with the Gemini list, preview payload, table crop) is unchanged.
"""

import re
from typing import List, Optional, Set, Tuple

from models import Dimensions, FoundationItem
from services.table_extractor import (
    TableExtractionResult,
    _clean_rebar_spec,
    _normalize,
    _parse_d,
    _parse_number,
    _REBAR_SPEC_RE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Mini-table recognition
# ──────────────────────────────────────────────────────────────────────────────

# Header cells of a per-foundation mini table. _normalize() has already folded
# full-width ASCII, so "Lx(ｍｍ)" arrives as "Lx(mm)".
_H_TYPE = "基礎符号"
_H_REMARKS = "備考"
_H_LX_RE = re.compile(r'^L\s*x', re.IGNORECASE)
_H_LY_RE = re.compile(r'^L\s*y', re.IGNORECASE)
# "D(ｍｍ)" normalises to "D(mm)"; some sheets split the unit into its own word.
_H_D_RE = re.compile(r'^D\s*(?:[(\[]|$)')
_FCODE_RE = re.compile(r'^F\d+[A-Za-z0-9]*$')

# A header row and its single data row sit this far apart at 1/80 scale (~17pt).
_DATA_ROW_MAX_GAP = 26
# Same-line tolerance for grouping header cells.
_LINE_TOL = 3.0
# How far the block owning a mini table reaches sideways from the 基礎符号 cell.
# The section drawing sits under/right of the table, the plan view to its left.
_BLOCK_PAD_LEFT = 155
_BLOCK_PAD_RIGHT = 265
# Bottom-stack members are centred on the same dimension line, so their x-centres
# coincide to within a hair.
_STACK_X_TOL = 4.0


def _cell(words: list, x_center: float, y_top: float, y_bot: float,
          max_dx: float = 22.0) -> List[dict]:
    """Words on the [y_top, y_bot] row whose x-centre is within max_dx of x_center."""
    hits = []
    for w in words:
        if not (y_top <= w["top"] <= y_bot):
            continue
        if abs((w["x0"] + w["x1"]) / 2 - x_center) <= max_dx:
            hits.append(w)
    hits.sort(key=lambda w: w["x0"])
    return hits


def _bottom_stack_count(words: list, lx: float,
                        x_lo: float, x_hi: float,
                        y_lo: float, y_hi: float) -> Optional[int]:
    """Count the values in the block's bottom dimension stack → 1, 2 or 3.

    Candidate stacks are clusters of upright plain numbers sharing an x-centre.
    A cluster is only accepted when, read top-to-bottom, its values STRICTLY
    INCREASE and the last one equals `lx` — the footing's full width. That check
    is what separates a real width stack (1,600 / 3,600 / 4,600) from unrelated
    numbers that happen to sit under the drawing.

    Both the 基礎平面 and 基礎断面 views usually carry the same stack; when they
    disagree the larger count wins (the smaller one was clipped by the window).
    Returns None when no cluster validates.
    """
    if lx <= 0:
        return None

    nums: List[Tuple[float, float, float]] = []   # (x_center, y_top, value)
    for w in words:
        if not w.get("upright"):
            continue
        if not (y_lo < w["top"] < y_hi):
            continue
        xc = (w["x0"] + w["x1"]) / 2
        if not (x_lo <= xc <= x_hi):
            continue
        val = _parse_number(_normalize(w["text"]))
        if val is None or val <= 0:
            continue
        nums.append((xc, w["top"], val))

    if not nums:
        return None

    # Cluster by x-centre.
    nums.sort()
    clusters: List[List[Tuple[float, float, float]]] = []
    for n in nums:
        if clusters and abs(n[0] - clusters[-1][-1][0]) <= _STACK_X_TOL:
            clusters[-1].append(n)
        else:
            clusters.append([n])

    best = None
    for cluster in clusters:
        cluster.sort(key=lambda c: c[1])              # top → bottom
        vals = [c[2] for c in cluster]
        if not (1 <= len(vals) <= 3):
            continue
        if vals[-1] != lx:
            continue
        if any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
            continue
        if best is None or len(vals) > best:
            best = len(vals)
    return best


_COUNT_TO_CLASS = {3: "DD", 2: "D", 1: "TNF"}


def _read_page(page, page_num: int) -> Tuple[List[FoundationItem], Optional[tuple]]:
    """Extract every per-foundation mini table on one page.

    Returns (items, bbox) where bbox is the union of the mini tables found (used
    for the preview crop), or (…, None) when the page has none.
    """
    try:
        words = page.extract_words(extra_attrs=["upright"])
    except Exception as e:
        print(f"[FoundationV2] page {page_num}: extract_words failed: {e}")
        return [], None

    # Every mini table is anchored by its 基礎符号 header cell.
    anchors = [w for w in words
               if w.get("upright") and _H_TYPE in _normalize(w["text"])]
    if not anchors:
        return [], None
    anchors.sort(key=lambda w: (w["x0"], w["top"]))

    items: List[FoundationItem] = []
    boxes: List[tuple] = []

    for anchor in anchors:
        ax = (anchor["x0"] + anchor["x1"]) / 2
        a_top, a_bot = anchor["top"], anchor["bottom"]

        # ── Header columns: the cells sharing the 基礎符号 line ────────────────
        row = [w for w in words
               if w.get("upright") and abs(w["top"] - a_top) < _LINE_TOL
               and w["x0"] >= anchor["x0"] - 2]
        row.sort(key=lambda w: w["x0"])

        # A 備考 cell on the header line means this is the CLASSIC page-wide
        # schedule, not a per-foundation mini table — leave it to table_extractor
        # (its 備考 B-formulas are the authoritative classification source).
        if any(_H_REMARKS in _normalize(w["text"]) for w in row):
            continue

        cols: dict = {}
        for w in row:
            norm = _normalize(w["text"])
            xc = (w["x0"] + w["x1"]) / 2
            if "lx" not in cols and _H_LX_RE.match(norm):
                cols["lx"] = xc
            elif "ly" not in cols and _H_LY_RE.match(norm):
                cols["ly"] = xc
            elif "d" not in cols and _H_D_RE.match(norm):
                cols["d"] = xc
                break            # rebar sub-columns follow; stop scanning headers
        if "d" not in cols or "lx" not in cols:
            continue             # not a foundation mini table

        # ── The single data row directly beneath the header ───────────────────
        # Blocks sit side by side, so stop the row at the NEXT mini table's 基礎符号
        # cell — otherwise a neighbour's rebar specs leak into this foundation.
        right_limit = min(
            [w["x0"] - 4 for w in anchors
             if w["x0"] > anchor["x0"] + 4 and abs(w["top"] - a_top) < _LINE_TOL]
            or [cols["d"] + _BLOCK_PAD_RIGHT]
        )
        band = [w for w in words
                if w.get("upright")
                and a_bot + 1 < w["top"] < a_bot + _DATA_ROW_MAX_GAP
                and anchor["x0"] - 8 <= w["x0"] <= right_limit]
        if not band:
            continue
        d_top = min(w["top"] for w in band)
        d_bot = d_top + 8
        band = [w for w in band if w["top"] <= d_bot]

        type_cells = _cell(band, ax, d_top - 2, d_bot)
        type_val = next((_normalize(w["text"]) for w in type_cells
                         if _FCODE_RE.match(_normalize(w["text"]))), "")
        if not type_val or re.match(r'^F[WG]', type_val):
            continue

        def _num(key: str) -> float:
            for w in _cell(band, cols[key], d_top - 2, d_bot):
                v = _parse_number(_normalize(w["text"]))
                if v is not None:
                    return v
            return 0.0

        lx = _num("lx")
        ly = _num("ly") if "ly" in cols else 0.0

        # D may be a range ("1,100～250"); join whatever sits in the D column.
        d_raw = " ".join(_normalize(w["text"])
                         for w in _cell(band, cols["d"], d_top - 2, d_bot))
        d_str = _parse_d(d_raw)

        # ── Rebar: the ベース筋 sub-columns, right of D, in x order ────────────
        # ← (x direction) is the left sub-column, ↑ (y direction) the right one.
        spec_words = sorted((w for w in band
                             if w["x0"] > cols["d"] + 4
                             and _REBAR_SPEC_RE.search(_normalize(w["text"]))),
                            key=lambda w: w["x0"])
        specs = [s for s in (_clean_rebar_spec(_normalize(w["text"]))
                             for w in spec_words) if s]
        rebar_x = specs[0] if specs else ""
        rebar_y = specs[1] if len(specs) > 1 else ""

        # ── Classification from the block's bottom dimension stack ────────────
        # The block runs from this table down to the next mini table in the same
        # column (blocks are stacked vertically, side by side in 2–3 columns).
        next_tops = [w["top"] for w in anchors
                     if w["top"] > a_bot + _DATA_ROW_MAX_GAP
                     and abs((w["x0"] + w["x1"]) / 2 - ax) < _BLOCK_PAD_RIGHT]
        y_hi = min(next_tops) if next_tops else float(page.height)
        count = _bottom_stack_count(
            words, lx,
            x_lo=ax - _BLOCK_PAD_LEFT, x_hi=ax + _BLOCK_PAD_RIGHT,
            y_lo=d_bot, y_hi=y_hi,
        )
        classification = _COUNT_TO_CLASS.get(count or 0, "TNF")
        if count is None:
            print(f"[FoundationV2] {type_val}: no valid width stack "
                  f"(Lx={lx:.0f}) — defaulting to TNF")

        try:
            items.append(FoundationItem(
                type=type_val,
                dimensions=Dimensions(Lx=lx, Ly=ly, D=d_str),
                top_elevation=None,
                rebar_x=rebar_x,
                rebar_y=rebar_y,
                remarks="",            # this layout has no 備考 column
                classification=classification,
                region=None,
                image_base64=None,
            ))
        except Exception as e:
            print(f"[FoundationV2] {type_val}: build failed: {e}")
            continue

        right = max(w["x1"] for w in band)
        boxes.append((anchor["x0"] - 4, a_top - 12, right + 4, d_bot + 8))
        print(f"[FoundationV2] {type_val}: Lx={lx:.0f} Ly={ly:.0f} D={d_str} "
              f"rebar={rebar_x}/{rebar_y} stack={count} → {classification}")

    if not items:
        return [], None
    bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
    return items, bbox


def extract_per_foundation_list(pdf, pages: Optional[Set[int]] = None) -> TableExtractionResult:
    """Read the per-foundation mini tables from an already-open pdfplumber doc.

    Scans every page and keeps the first occurrence of each type. `page_num`/`bbox`
    come from the page that contributed the most rows, so the preview crop shows
    the densest 基礎リスト sheet.

    pages: 0-based page indices to consider; None = every page. Honour the caller's
    gate — extract_text() on a skipped page would trigger the deep content-stream
    parse the gate exists to avoid (see textlayer_phase1._relevant_page_indices).

    Returns an empty TableExtractionResult when the document uses the classic
    single-schedule layout (or nothing recognisable is found).
    """
    all_items: List[FoundationItem] = []
    seen: set = set()
    best_page, best_bbox, best_n = 0, None, 0

    try:
        scan = [(n, p) for n, p in enumerate(pdf.pages, start=1)
                if pages is None or (n - 1) in pages]
    except Exception as e:
        print(f"[FoundationV2] cannot enumerate pages: {e}")
        return TableExtractionResult()

    for page_num, page in scan:
        try:
            text = page.extract_text() or ""
        except Exception:
            continue
        # Cheap gate: the mini-table header must be on the page.
        if _H_TYPE not in text:
            continue
        items, bbox = _read_page(page, page_num)
        if not items:
            continue
        fresh = [it for it in items if it.type.upper() not in seen]
        for it in fresh:
            seen.add(it.type.upper())
        all_items.extend(fresh)
        if bbox and len(items) > best_n:
            best_page, best_bbox, best_n = page_num, bbox, len(items)

    if not all_items:
        print("[FoundationV2] no per-foundation mini tables found.")
        return TableExtractionResult()

    print(f"[FoundationV2] read {len(all_items)} foundation(s) from "
          f"per-foundation mini tables (anchor page {best_page})")
    return TableExtractionResult(
        items=all_items,
        page_num=best_page,
        bbox=best_bbox,
        pdf_width=float(pdf.pages[best_page - 1].width) if best_page else 0.0,
        pdf_height=float(pdf.pages[best_page - 1].height) if best_page else 0.0,
    )
