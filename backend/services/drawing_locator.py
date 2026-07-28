"""drawing_locator.py — text-layer based locator for drawing titles & section captions.

Uses pdfplumber to read the PDF's vector text layer with accurate visual positions
(pypdfium2 returns glyph-frame positions which can be wrong for rotated/CID fonts
in some Japanese CAD PDFs). pdfplumber's `extract_words` returns positions of the
RENDERED text, which is what we need to crop the right area of the rendered image.

Locates:
  • Foundation cross-section/plan drawing titles  ("F1 基礎断面", "F1, F2 基礎平面", …)
  • FW/FG section captions                        ("外壁基(FW)詳細図", "地中梁リスト")
  • Individual beam labels inside sections        ("FW1 (一般部)", "FW2", "FG1", …)

All output regions are in 0–1000 image-relative coordinates (origin top-left).
"""

import re
import io
from typing import List, Dict, Set, Tuple, Optional
import pdfplumber

from models import ItemRegion
from services.page_gate import relevant_pages

# Text a page must contain to possibly hold a beam label block (FW panels are
# captioned 外壁基…詳細図, FG types live in the 地中梁リスト table).
_BEAM_PAGE_KW = ("外壁基", "地中梁", "リスト", "詳細図")


# ─────────────────────────────────────────────────────────────────────────────
# Regex / search patterns
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_TOKEN_RE = re.compile(r'F\d+[A-Z0-9]*')

# Match a title like "F1,F2,F5 基礎断面", "F1 基礎断面", "F7~F9 基礎断面", and also the
# "基礎"-less variant some firms use in the リスト ("F1,F1A 断面図", "F2B 平面図").
# Note: pdfplumber concatenates words from same line with a space between them.
# 基礎 is optional and a trailing 図 is allowed. The F-code prefix is required, so a
# bare "A-A断面図" or "ピット断面図" (no F\d code) never matches. Separator class
# includes list delimiters (, 、 ，) AND range delimiters (~ ～ ・ /); ranges like
# "F7~F9" are expanded by _expand_types(). The continuation token uses an optional
# F (F?\d+) so "F7~9" works as well as "F7~F9".
_FOUNDATION_SECTION_RE = re.compile(
    r'(F\d+[A-Z0-9]*(?:[\s,、，~～・/]+F?\d+[A-Z0-9]*)*)\s*(?:基礎\s*)?断面図?',
    re.UNICODE,
)
_FOUNDATION_PLAN_RE = re.compile(
    r'(F\d+[A-Z0-9]*(?:[\s,、，~～・/]+F?\d+[A-Z0-9]*)*)\s*(?:基礎\s*)?平面図?',
    re.UNICODE,
)

# FW section caption variants. Captures the FW type number (if present, e.g.,
# "外壁基礎(FW1)詳細図" → "1"; "外壁基(FW)詳細図" → "" generic).
_FW_CAPTION_RE = re.compile(
    r'外壁基(?:[礎\s])*\(?FW(\d*)\)?\s*[礎]?\s*詳細図',
    re.UNICODE,
)

# FG table caption
_FG_CAPTION_RE = re.compile(r'地中梁\s*リスト', re.UNICODE)

# Standalone GL elevation marker (no surrounding parens, no extra text):
# "GL+50", "GL-20", "GL±0", "設計GL+50", "設計GL-1,250", "GL+200mm", "C棟GL±0" etc.
#
# NOTE — this is DELIBERATELY narrower than text_parser._GL_PREFIX, which accepts any
# short prefix (including "SGL"). This regex only pre-filters candidates for the FLOOR
# SLAB oval markers, and broadening it to "SGL" starts detecting slab markers on sheets
# where they were previously ignored — a change to existing projects' floor lists that
# we do not want. So accept only the classic forms plus a building-wing prefix
# ("C棟GL"), which is what the newer sheets need. Keep them in sync only on purpose.
_GL_MARKER_RE = re.compile(
    r'^(?:設計)?(?:.棟)?GL\s*[+\-±]\s*[0-9,]+(?:mm)?\)?$',
    re.UNICODE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _open_pdfplumber(pdf_content: bytes):
    """Open a pdfplumber document from bytes."""
    return pdfplumber.open(io.BytesIO(pdf_content))


def _pdf_to_image_region_topdown(x0: float, y0: float, x1: float, y1: float,
                                  pdf_w: float, pdf_h: float,
                                  page_num: int) -> Optional[ItemRegion]:
    """Convert pdfplumber bbox (origin top-left, y-down) → ItemRegion 0–1000."""
    xmin = int(x0 / pdf_w * 1000)
    ymin = int(y0 / pdf_h * 1000)
    xmax = int(x1 / pdf_w * 1000)
    ymax = int(y1 / pdf_h * 1000)
    xmin = max(0, min(1000, xmin))
    ymin = max(0, min(1000, ymin))
    xmax = max(0, min(1000, xmax))
    ymax = max(0, min(1000, ymax))
    if xmax <= xmin or ymax <= ymin:
        return None
    return ItemRegion(page=page_num, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def _build_lines(words: list) -> List[List[dict]]:
    """Group pdfplumber words into LINES (same top/bottom row, sorted by x)."""
    lines: Dict[Tuple[int, int], List[dict]] = {}
    for w in words:
        # Round top to nearest 2 units to group near-aligned words
        key = (round(w["top"] / 2) * 2, round(w["bottom"] / 2) * 2)
        lines.setdefault(key, []).append(w)
    grouped = []
    for key, ws in lines.items():
        ws.sort(key=lambda w: w["x0"])
        grouped.append(ws)
    return grouped


def _line_text(line_words: List[dict]) -> str:
    """Join words in a line with single spaces (matching what we want to regex-match)."""
    return " ".join(w["text"] for w in line_words)


def _bbox_for_match_span(line_words: List[dict], joined: str,
                          start: int, end: int) -> Optional[Tuple[float, float, float, float]]:
    """Given a regex match span on the joined-line text, return the union bbox
    of the words that fall (at least partially) within that span."""
    cursor = 0
    matched_words = []
    for w in line_words:
        w_start = cursor
        w_end = cursor + len(w["text"])
        # Check overlap of [w_start, w_end) with [start, end)
        if w_start < end and w_end > start:
            matched_words.append(w)
        cursor = w_end + 1  # +1 for the space separator
        if cursor > end:
            break
    if not matched_words:
        return None
    x0 = min(w["x0"] for w in matched_words)
    y0 = min(w["top"] for w in matched_words)
    x1 = max(w["x1"] for w in matched_words)
    y1 = max(w["bottom"] for w in matched_words)
    return (x0, y0, x1, y1)


def _expand_types(types_str: str) -> List[str]:
    """Expand a captured title type-list into individual upper-case types.

    Handles list separators AND numeric ranges:
      "F2,F3"  → [F2, F3]
      "F7~F9"  → [F7, F8, F9]
      "F7~9"   → [F7, F8, F9]
      "F1A"    → [F1A]
    """
    s = types_str.replace('～', '~')
    out: List[str] = []
    # Split on list separators but KEEP '~' so ranges stay in one segment.
    for seg in re.split(r'[\s,、，・/]+', s):
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r'^F(\d+)~F?(\d+)$', seg, re.IGNORECASE)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b and b - a <= 50:
                out.extend(f"F{n}" for n in range(a, b + 1))
                continue
        out.extend(t.upper() for t in _TYPE_TOKEN_RE.findall(seg))
    return out


def _extract_words(page) -> List[dict]:
    """Extract pdfplumber words once with the tolerances used by all scans."""
    return page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)


def _extract_lines(page) -> List[List[dict]]:
    """Words grouped into lines for one page (the unit cached across all scans)."""
    return _build_lines(_extract_words(page))


def build_page_scan(page_idx: int, page) -> dict:
    """Serialisable per-page cache shared across text-layer scans."""
    words = _extract_words(page)
    return {
        "page_idx": page_idx,
        "page_num": page_idx + 1,
        "width": float(page.width),
        "height": float(page.height),
        "lines": _build_lines(words),
        "words": words,
    }


def empty_page_scan(page_idx: int, page) -> dict:
    """Page-scan stub for a page deliberately NOT deep-parsed.

    Same shape as build_page_scan but with no words or lines, so callers that
    index page scans by page number stay aligned. Only width/height are read, and
    those come from the page's MediaBox — no content-stream parsing, which is the
    whole point (see textlayer_phase1._relevant_page_indices).
    """
    return {
        "page_idx": page_idx,
        "page_num": page_idx + 1,
        "width": float(page.width),
        "height": float(page.height),
        "lines": [],
        "words": [],
    }


def _find_pattern_on_lines(lines, regex) -> List[Tuple[Tuple[float, float, float, float], str, List[str]]]:
    """Run a foundation-title regex on pre-built lines; return
    list of (bbox, original_match_text, captured_types)."""
    out = []
    for line_words in lines:
        text = _line_text(line_words)
        for m in regex.finditer(text):
            types_str = m.group(1)
            types = _expand_types(types_str)
            if not types:
                continue
            bbox = _bbox_for_match_span(line_words, text, m.start(), m.end())
            if bbox:
                out.append((bbox, m.group(), types))
    return out


def _find_pattern_on_page(page, regex):
    return _find_pattern_on_lines(_extract_lines(page), regex)


def _find_text_on_lines(lines, regex_or_string) -> Optional[Tuple[float, float, float, float]]:
    """Find the first regex/string match anywhere in pre-built lines.
    Returns bbox of the matched span."""
    for line_words in lines:
        text = _line_text(line_words)
        if hasattr(regex_or_string, "search"):
            m = regex_or_string.search(text)
            if m:
                bbox = _bbox_for_match_span(line_words, text, m.start(), m.end())
                if bbox:
                    return bbox
        else:
            idx = text.find(regex_or_string)
            if idx != -1:
                bbox = _bbox_for_match_span(line_words, text, idx, idx + len(regex_or_string))
                if bbox:
                    return bbox
    return None


def _find_text_on_page(page, regex_or_string):
    return _find_text_on_lines(_extract_lines(page), regex_or_string)


def _compute_foundation_drawing_regions(
    titles_by_page: Dict[int, List[Tuple[ItemRegion, str, List[str]]]],
    type_to_hits: Dict[str, List[Tuple[int, ItemRegion, str]]],
) -> Dict[str, ItemRegion]:
    PAD_TOP_UNITS         = 16
    DEFAULT_HEIGHT_UNITS  = 220
    DEFAULT_HALF_W_UNITS  = 180
    GAP_TO_NEIGHBOUR      = 8

    result: Dict[str, ItemRegion] = {}
    for ftype, hits in type_to_hits.items():
        section_hits = [h for h in hits if h[2] == "section"]
        plan_hits = [h for h in hits if h[2] == "plan"]
        picked = section_hits[0] if section_hits else (plan_hits[0] if plan_hits else None)
        if not picked:
            continue
        page_idx, title_region, _kind = picked

        cx = (title_region.xmin + title_region.xmax) // 2
        ymin0 = max(0, title_region.ymin - PAD_TOP_UNITS)
        ymax0 = min(1000, title_region.ymax + DEFAULT_HEIGHT_UNITS)
        xmin0 = max(0, cx - DEFAULT_HALF_W_UNITS)
        xmax0 = min(1000, cx + DEFAULT_HALF_W_UNITS)

        same_page_titles = titles_by_page.get(page_idx, [])
        our_types: set = set()
        for (r, _k, ts) in same_page_titles:
            if r is title_region:
                our_types = set(ts)
                break

        competing = [
            (r, ts) for (r, _k, ts) in same_page_titles
            if r is not title_region and not (set(ts) & our_types)
        ]

        def _same_col(r: ItemRegion) -> bool:
            their_cx = (r.xmin + r.xmax) // 2
            return (
                abs(cx - their_cx) <= 80 or
                (r.xmin <= title_region.xmax + 6 and r.xmax >= title_region.xmin - 6)
            )

        v_competing = [r for (r, _ts) in competing if _same_col(r)]
        below = [r for r in v_competing if r.ymin > title_region.ymax + 4]
        ymax = min(ymax0, min(r.ymin for r in below) - GAP_TO_NEIGHBOUR) if below else ymax0
        ymax = max(ymax, title_region.ymax + 50)

        above = [r for r in v_competing if r.ymax < title_region.ymin - 4]
        ymin = max(ymin0, max(r.ymax for r in above) + GAP_TO_NEIGHBOUR) if above else ymin0
        ymin = min(ymin, max(0, title_region.ymin - 4))

        def _y_overlaps(r: ItemRegion) -> bool:
            return r.ymin < ymax and r.ymax > ymin

        h_competing = [r for (r, _ts) in competing if _y_overlaps(r)]
        left = [r for r in h_competing if r.xmax < title_region.xmin - 4]
        xmin = max(xmin0, max(r.xmax for r in left) + GAP_TO_NEIGHBOUR) if left else xmin0
        right = [r for r in h_competing if r.xmin > title_region.xmax + 4]
        xmax = min(xmax0, min(r.xmin for r in right) - GAP_TO_NEIGHBOUR) if right else xmax0

        if xmax - xmin < 80:
            xmin = max(0, cx - 50)
            xmax = min(1000, cx + 50)
        if ymax - ymin < 50:
            ymax = min(1000, ymin + 50)

        result[ftype] = ItemRegion(
            page=title_region.page,
            xmin=int(xmin), ymin=int(ymin), xmax=int(xmax), ymax=int(ymax),
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_foundation_drawing_regions(pdf_content: bytes) -> Dict[str, ItemRegion]:
    """For each F-type, return an ItemRegion covering its cross-section drawing.

    Strategy:
      1. Read pdfplumber words on every page.
      2. Per line, run "F? 基礎断面" / "F? 基礎平面" regex.
      3. For each F-type, pick its best title (section preferred over plan).
      4. Compute the crop region anchored on the title and BOUNDED by the
         nearest competing title in each direction. Competing = no shared F-types.
         Vertical bound requires x-overlap with our title (same column).
         Horizontal bound requires y-overlap with our band (same row).
    """
    with _open_pdfplumber(pdf_content) as pdf:
        return find_foundation_drawing_regions_from_pages(
            [build_page_scan(page_idx, page) for page_idx, page in enumerate(pdf.pages)]
        )


def find_foundation_drawing_regions_from_pages(page_scans: List[dict]) -> Dict[str, ItemRegion]:
    """Lines-based foundation-title scan using pre-built per-page caches."""
    titles_by_page: Dict[int, List[Tuple[ItemRegion, str, List[str]]]] = {}
    type_to_hits: Dict[str, List[Tuple[int, ItemRegion, str]]] = {}

    for page_scan in page_scans:
        page_idx = page_scan["page_idx"]
        page_num = page_scan["page_num"]
        page_titles: List[Tuple[ItemRegion, str, List[str]]] = []
        for kind, regex in (("section", _FOUNDATION_SECTION_RE), ("plan", _FOUNDATION_PLAN_RE)):
            for bbox, _raw, types in _find_pattern_on_lines(page_scan["lines"], regex):
                region = _pdf_to_image_region_topdown(
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    page_scan["width"], page_scan["height"], page_num,
                )
                if not region:
                    continue
                page_titles.append((region, kind, types))
                for t in types:
                    type_to_hits.setdefault(t, []).append((page_idx, region, kind))
        if page_titles:
            titles_by_page[page_idx] = page_titles

    return _compute_foundation_drawing_regions(titles_by_page, type_to_hits)


def _find_match_with_groups_on_lines(lines, regex) -> Optional[Tuple[Tuple[float, float, float, float], List[str]]]:
    """Find first regex match on pre-built lines; returns (bbox, groups)."""
    for line_words in lines:
        text = _line_text(line_words)
        m = regex.search(text)
        if m:
            bbox = _bbox_for_match_span(line_words, text, m.start(), m.end())
            if bbox:
                groups = [g for g in m.groups()] if m.groups() else []
                return (bbox, groups)
    return None


def _find_match_with_groups(page, regex) -> Optional[Tuple[Tuple[float, float, float, float], List[str]]]:
    return _find_match_with_groups_on_lines(_extract_lines(page), regex)


def find_beam_section_captions(pdf_content: bytes) -> Dict[int, Dict[str, dict]]:
    """Find FW and FG section CAPTION positions on each page (text-layer).

    Returns:
        {page_idx: {"fw"?: {"region": ItemRegion, "types": ["FW1"] or []},
                    "fg"?: {"region": ItemRegion, "types": []}}}
    For FW, the captured type number (e.g., "1" from "外壁基礎(FW1)詳細図")
    makes the caption TYPE-SPECIFIC: there's no separate variant label and the
    panel is the entire area ABOVE the caption.
    """
    captions: Dict[int, Dict[str, dict]] = {}
    with _open_pdfplumber(pdf_content) as pdf:
        return find_beam_section_captions_from_pages(
            [build_page_scan(page_idx, page) for page_idx, page in enumerate(pdf.pages)]
        )


def find_beam_section_captions_from_pages(page_scans: List[dict]) -> Dict[int, Dict[str, dict]]:
    """Lines-based beam-caption scan using pre-built per-page caches."""
    captions: Dict[int, Dict[str, dict]] = {}
    for page_scan in page_scans:
        page_idx = page_scan["page_idx"]
        page_num = page_scan["page_num"]
        page_caps: Dict[str, dict] = {}

        fw_match = _find_match_with_groups_on_lines(page_scan["lines"], _FW_CAPTION_RE)
        if fw_match:
            bbox_fw, groups = fw_match
            region = _pdf_to_image_region_topdown(
                bbox_fw[0], bbox_fw[1], bbox_fw[2], bbox_fw[3],
                page_scan["width"], page_scan["height"], page_num,
            )
            if region:
                type_num = groups[0].strip() if groups else ""
                page_caps["fw"] = {
                    "region": region,
                    "types": [f"FW{type_num}"] if type_num else [],
                }

        fg_match = _find_match_with_groups_on_lines(page_scan["lines"], _FG_CAPTION_RE)
        if fg_match:
            bbox_fg, _groups = fg_match
            region = _pdf_to_image_region_topdown(
                bbox_fg[0], bbox_fg[1], bbox_fg[2], bbox_fg[3],
                page_scan["width"], page_scan["height"], page_num,
            )
            if region:
                page_caps["fg"] = {"region": region, "types": []}

        if page_caps:
            captions[page_idx] = page_caps
    return captions


def derive_fw_panel_region_from_caption(caption_region: ItemRegion) -> ItemRegion:
    """When the FW caption itself identifies the type (e.g., 外壁基礎(FW1)詳細図),
    the panel is the entire area ABOVE the caption, on the SAME horizontal
    sheet column. Crop a generous box centred horizontally on the caption,
    extending upward by ~25-30% of the page height.
    """
    cap_w = max(1, caption_region.xmax - caption_region.xmin)
    cap_cx = (caption_region.xmin + caption_region.xmax) // 2

    # Width: panels are wider than the caption text — extend ~3x each side
    half_w = max(int(cap_w * 3), 180)
    xmin = max(0, cap_cx - half_w)
    xmax = min(1000, cap_cx + half_w)

    # Height: extend upward; small pad below caption to include it visually
    panel_h = 280
    ymin = max(0, caption_region.ymin - panel_h)
    ymax = min(1000, caption_region.ymax + 8)

    return ItemRegion(page=caption_region.page, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def find_beam_labels(pdf_content: bytes, beam_types: List[str],
                     prefer_pages: Optional[set] = None) -> Dict[str, ItemRegion]:
    """Find positions of individual beam labels.

    `prefer_pages`: if provided, only matches on these 0-indexed page numbers are kept.
    """
    if not beam_types:
        return {}

    queries_per_type: Dict[str, List[str]] = {}
    for t in beam_types:
        u = t.upper()
        if u.startswith("FW"):
            queries_per_type[u] = [f"{u} (一般部)", f"{u}(一般部)", u]
        else:
            queries_per_type[u] = [u]

    # Re-opening the PDF here pays the full deep parse again, so build page scans
    # ONLY for pages that can produce a kept match: `prefer_pages` when the caller
    # narrowed it down, otherwise pages whose text mentions a beam block at all.
    pages = prefer_pages
    if pages is None:
        pages = relevant_pages(pdf_content, _BEAM_PAGE_KW, label="BeamLabels")

    with _open_pdfplumber(pdf_content) as pdf:
        return find_beam_labels_from_pages(
            [build_page_scan(page_idx, page)
             for page_idx, page in enumerate(pdf.pages)
             if pages is None or page_idx in pages],
            beam_types,
            prefer_pages=prefer_pages,
        )


def find_beam_labels_from_pages(page_scans: List[dict], beam_types: List[str],
                                prefer_pages: Optional[set] = None) -> Dict[str, ItemRegion]:
    """Find beam labels on pre-built page caches."""
    if not beam_types:
        return {}

    queries_per_type: Dict[str, List[str]] = {}
    for t in beam_types:
        u = t.upper()
        if u.startswith("FW"):
            queries_per_type[u] = [f"{u} (一般部)", f"{u}(一般部)", u]
        else:
            queries_per_type[u] = [u]

    found: Dict[str, ItemRegion] = {}
    for page_scan in page_scans:
        page_idx = page_scan["page_idx"]
        if prefer_pages is not None and page_idx not in prefer_pages:
            continue
        for btype, queries in queries_per_type.items():
            if btype in found:
                continue
            for q in queries:
                bbox = _find_text_on_lines(page_scan["lines"], q)
                if not bbox:
                    continue
                region = _pdf_to_image_region_topdown(
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    page_scan["width"], page_scan["height"], page_scan["page_num"],
                )
                if region:
                    found[btype] = region
                    break
    return found


def derive_fw_panel_region(label_region: ItemRegion,
                            page_caption: Optional[ItemRegion]) -> ItemRegion:
    """Build a panel region around an FW label.

    Common Japanese layout: variant label ('FW1 (一般部)', 'FW2') sits at the TOP
    of its panel; panel extends DOWNWARD to just above the group caption.
    """
    label_w = max(1, label_region.xmax - label_region.xmin)

    ymin = max(0, label_region.ymin - 5)
    if page_caption and page_caption.ymin > label_region.ymax + 20:
        ymax = max(label_region.ymax + 100, page_caption.ymin - 6)
    else:
        ymax = min(1000, label_region.ymax + 200)

    cx = (label_region.xmin + label_region.xmax) // 2
    half_w = max(label_w * 4, 110)
    xmin = max(0, cx - half_w)
    xmax = min(1000, cx + half_w)

    return ItemRegion(page=label_region.page, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def find_oval_gl_markers(pdf_content: bytes) -> List[dict]:
    """Detect OVAL-shaped GL elevation markers using PDF curve primitives.

    Distinguishing ovals vs rectangles:
      - OVAL/PILL marker (floor elevation indicator): drawn as a single PATH
        of bezier curves; pdfplumber surfaces this as a `curves` element with
        MANY points (>=8 typically 20-50). The bbox tightly hugs the text.
      - RECTANGULAR marker (boundary mark, grid reference, table cell, etc.):
        drawn as 4 straight lines or a `rects` element. pdfplumber surfaces it
        as a `rects` entry with no curve enclosure. These are NOT floor
        elevation markers — they are filtered out.

    For each standalone GL text on each page, we search for a TIGHT enclosing
    curve (area ratio < 5× text area, at least 8 points). If found → keep as
    oval. If only an enclosing rect is found → skip.

    Returns: [{"text": str, "region": ItemRegion}]   (region in 0–1000 image scale)
    """
    with _open_pdfplumber(pdf_content) as pdf:
        page_words = [_extract_words(page) for page in pdf.pages]
        return find_oval_gl_markers_from_cached_words(pdf, page_words)


def find_oval_gl_markers_from_open(pdf) -> List[dict]:
    """Oval GL-marker detection on an already-open pdfplumber document.

    See find_oval_gl_markers for the detection logic. Split out so the caller
    can open the PDF once and share it with the table extractor instead of
    parsing the same (large CAD) document twice.
    """
    page_words = [_extract_words(page) for page in pdf.pages]
    return find_oval_gl_markers_from_cached_words(pdf, page_words)


def find_oval_gl_markers_from_cached_words(pdf, page_words: List[List[dict]],
                                           pages: Optional[Set[int]] = None) -> List[dict]:
    """Oval GL-marker detection reusing caller-supplied word caches.

    pages: 0-based indices to consider; None = every page. Pages outside the set
    are skipped WITHOUT touching page.chars — reading it would trigger the deep
    content-stream parse this gate exists to avoid.
    """
    results: List[dict] = []
    for page_idx, page in enumerate(pdf.pages):
        if pages is not None and page_idx not in pages:
            continue
        # Cheap pre-check: GL markers carry a literal "GL" in the text layer.
        # Pages without it can't hold one — skip extract_words / curve parsing.
        page_text = "".join(c.get("text", "") for c in page.chars)
        if "GL" not in page_text and "ＧＬ" not in page_text:
            continue
        words = page_words[page_idx] if page_idx < len(page_words) else _extract_words(page)
        for w in words:
            if not _GL_MARKER_RE.match(w["text"].strip()):
                continue

            tw = max(1.0, w["x1"] - w["x0"])
            th = max(1.0, w["bottom"] - w["top"])
            text_area = tw * th

            # 1) Look for a TIGHT enclosing curve (oval/pill outline).
            # Accept if either:
            #   (a) area ratio ≤ 5×  (typical tight pill)
            #   (b) ratio ≤ 12× AND pts ≥ 20  (smoother oval, slightly
            #       padded — still clearly the marker shape, NOT a page
            #       border which has thousands of units in dimensions)
            #   (c) Absolute size sanity: curve width < text_width + 40
            #       AND curve height < text_height + 25
            tight_curves = []
            for c in page.curves:
                if not (c["x0"] <= w["x0"] + 2 and c["x1"] >= w["x1"] - 2
                        and c["top"] <= w["top"] + 2 and c["bottom"] >= w["bottom"] - 2):
                    continue
                cw = max(1.0, c["x1"] - c["x0"])
                ch = max(1.0, c["bottom"] - c["top"])
                pts = len(c.get("pts", []))
                if pts < 8:
                    continue  # not a real curved path
                # Absolute-size filter against page borders / building outlines
                if cw > tw + 40 or ch > th + 25:
                    continue
                area_ratio = (cw * ch) / text_area
                if area_ratio > 12:
                    continue
                # Either tight (≤5) or moderately tight with many points
                if area_ratio <= 5 or pts >= 20:
                    tight_curves.append(c)

            if tight_curves:
                # Smallest (tightest) curve is the oval boundary
                c = min(tight_curves,
                        key=lambda c: (c["x1"] - c["x0"]) * (c["bottom"] - c["top"]))
                region = _pdf_to_image_region_topdown(
                    c["x0"], c["top"], c["x1"], c["bottom"],
                    page.width, page.height, page_idx + 1,
                )
                if region:
                    results.append({"text": w["text"].strip(), "region": region})
                continue  # oval handled

            # 2) No tight curve → check for tight rectangle. If a tight rect
            # encloses the text, this is a RECTANGULAR marker → SKIP.
            # (We don't need to do anything; we just don't add to results.)
            # The text might be inside a table cell etc. — irrelevant here.

    return results


def derive_fg_column_region(header_region: ItemRegion,
                             page_caption: ItemRegion) -> ItemRegion:
    """Build a 断面 column-cell region for FG."""
    header_w = max(1, header_region.xmax - header_region.xmin)
    table_h = page_caption.ymin - header_region.ymin
    if table_h <= 0:
        table_h = (1000 - header_region.ymin) // 2

    ymin = max(0, header_region.ymin - 8)
    ymax = min(1000, header_region.ymin + int(table_h * 0.55))

    pad = max(int(header_w * 1.8), 80)
    xmin = max(0, header_region.xmin - pad)
    xmax = min(1000, header_region.xmax + pad)

    return ItemRegion(page=header_region.page, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


# Drawing-caption keywords used to detect neighbouring drawing blocks so a pit
# crop can be bounded against them instead of spilling across the whole sheet.
_PIT_CAPTION_KW = ("詳細図", "断面図", "リスト")


def _line_bbox(line_words) -> Tuple[float, float, float, float]:
    return (min(w["x0"] for w in line_words),
            min(w["top"] for w in line_words),
            max(w["x1"] for w in line_words),
            max(w["bottom"] for w in line_words))


def _collect_caption_anchors(lines) -> List[dict]:
    """All drawing-caption lines on a page (contain 詳細図/断面図/リスト, not 参照).

    Used purely to bound a pit crop horizontally against neighbouring drawing
    blocks (e.g. keep the EVピット① crop from extending into EVピット②).
    """
    anchors: List[dict] = []
    for line_words in lines:
        compact = _line_text(line_words).replace(" ", "").replace("　", "")
        if "参照" in compact or not any(k in compact for k in _PIT_CAPTION_KW):
            continue
        x0, top, x1, bottom = _line_bbox(line_words)
        anchors.append({"x0": x0, "x1": x1, "top": top, "bottom": bottom,
                        "cx": (x0 + x1) / 2, "cy": (top + bottom) / 2})
    return anchors


def _find_pit_caption(lines, type_str: str) -> Tuple[int, Optional[Tuple[float, float, float, float]]]:
    """Best caption for a pit type on one page: (rank, bbox).

    rank 0 = 詳細図 (overview, preferred), 1 = 断面図 (internal section). Floor-plan
    reference labels ("…詳細図参照(底盤天端設計GL-…)") are skipped. Returns
    (99, None) when the type is not captioned on this page.

    When the exact type string is not found in any caption, falls back to matching
    the last 2 characters (e.g. "水槽" from "清水槽" matches "消火水槽詳細図").
    """
    type_compact = type_str.replace(" ", "").replace("　", "")
    if not type_compact:
        return 99, None

    # Build search candidates: exact first, then last-2-char fallback (if >= 3 chars).
    search_candidates = [type_compact]
    if len(type_compact) >= 3:
        search_candidates.append(type_compact[-2:])

    best_rank, best_bbox = 99, None
    for search_s in search_candidates:
        if best_rank == 0:
            break  # already found 詳細図, stop
        for line_words in lines:
            compact = _line_text(line_words).replace(" ", "").replace("　", "")
            if "参照" in compact:
                continue
            ti = compact.find(search_s)
            if ti == -1:
                continue
            for rank, suffix in enumerate(("詳細図", "断面図")):
                if compact.find(suffix, ti + len(search_s)) == -1:
                    continue
                if rank < best_rank:
                    best_rank = rank
                    best_bbox = _line_bbox(line_words)
                break
    return best_rank, best_bbox


def _pit_region_from_caption(cap_bbox, anchors: List[dict],
                             page_w: float, page_h: float, page_num: int) -> Optional[ItemRegion]:
    """Build a crop region for a pit whose caption sits at the BOTTOM of the block.

    Width is bounded at the midpoint to the nearest neighbouring drawing caption
    on the SAME row (so side-by-side blocks like EVピット①/② don't merge); a
    default half-width is used on any side with no neighbour. The drawing extends
    upward from the caption by a fixed fraction of the page height.
    """
    cx0, ctop, cx1, cbot = cap_bbox
    cap_cx = (cx0 + cx1) / 2
    cap_cy = (ctop + cbot) / 2

    row_tol = page_h * 0.02
    same_row = [a for a in anchors
                if abs(a["cy"] - cap_cy) <= row_tol
                and not (a["x0"] - 1 <= cap_cx <= a["x1"] + 1)]  # exclude own line
    left = [a["cx"] for a in same_row if a["cx"] < cap_cx]
    right = [a["cx"] for a in same_row if a["cx"] > cap_cx]

    default_half = page_w * 0.13
    xmin = (max(left) + cap_cx) / 2 if left else cap_cx - default_half
    xmax = (min(right) + cap_cx) / 2 if right else cap_cx + default_half
    # Never clip the caption text itself.
    xmin = min(xmin, cx0 - 4)
    xmax = max(xmax, cx1 + 4)

    ymax = cbot + page_h * 0.006          # tiny pad below caption to include it
    ymin = ctop - page_h * 0.26           # drawing sits above the caption

    return _pdf_to_image_region_topdown(
        max(0.0, xmin), max(0.0, ymin), min(page_w, xmax), min(page_h, ymax),
        page_w, page_h, page_num,
    )


def _pit_block_bounds(page, cap_bbox) -> Optional[Tuple[float, float, float, float]]:
    """Snap a pit crop to the drawing's bordered cell using the PDF's own vector
    borders, so the box hugs exactly one block instead of spilling into the next.

    These detail sheets lay drawings out in bordered cells; the caption sits at
    the bottom. We take the nearest tall VERTICAL edges on each side of the
    caption that rise above it (the cell's left/right borders) and use their
    span for the height. Returns (xmin, ymin, xmax, ymax) in PDF points, or None
    when no plausible borders are found (caller then falls back to the
    caption-neighbour heuristic).
    """
    try:
        pw, ph = float(page.width), float(page.height)
        cx0, ctop, cx1, cbot = cap_bbox
        cap_cx = (cx0 + cx1) / 2
        cap_w = cx1 - cx0

        # Cell side borders: tall verticals that rise above the caption and reach
        # down into its band; short internal ticks are filtered out by min length.
        min_v_len = ph * 0.05
        cands = [e for e in page.vertical_edges
                 if (e["bottom"] - e["top"]) >= min_v_len
                 and e["top"] <= ctop - 2
                 and e["bottom"] >= ctop - ph * 0.32]
        left = [e for e in cands if e["x0"] <= cap_cx]
        right = [e for e in cands if e["x0"] >= cap_cx]
        if not left or not right:
            return None

        le = max(left, key=lambda e: e["x0"])   # nearest border on the left
        re_ = min(right, key=lambda e: e["x0"])  # nearest border on the right
        xmin, xmax = le["x0"], re_["x0"]
        width = xmax - xmin
        # Reject implausible picks: too narrow → we hit an internal line; too wide
        # → the verticals straddle multiple blocks. Fall back to the heuristic.
        if width < max(cap_w * 1.1, pw * 0.04) or width > pw * 0.5:
            return None

        cell_top = min(le["top"], re_["top"])
        cell_bottom = max(le["bottom"], re_["bottom"])
        ymin = cell_top - ph * 0.004
        ymax = max(cell_bottom, cbot) + ph * 0.006   # include caption if just below
        return (xmin, ymin, xmax, ymax)
    except Exception:
        return None


def find_pit_drawing_regions(pdf_content: bytes, pit_types: List[str]) -> Dict[str, ItemRegion]:
    """For each pit type, locate its detail-drawing caption → crop region.

    Pit drawings carry the title as a caption at the BOTTOM of the block
    (e.g. "EVピット①詳細図 S=1/60"), so the drawing is the area ABOVE the caption.
    詳細図 (overview) is preferred over 断面図 (internal section) GLOBALLY — a
    詳細図 on any page beats a 断面図 on any other (some pits, e.g. レールのピット,
    have a 断面図 inset on the floor plan AND a standalone 詳細図 elsewhere; the
    spec wants the 詳細図). Gemini already supplies the clean pit type, so we
    match that string directly.

    The crop is snapped to the drawing's bordered cell (vector borders) when
    possible, otherwise bounded against neighbouring captions. Returns
    {pit_type → ItemRegion}. Types without a discoverable caption are omitted
    (caller keeps whatever region Gemini provided, if any).
    """
    wanted = [t for t in dict.fromkeys(pit_types) if t and t.strip()]
    if not wanted:
        return {}

    # This runs in the MAIN process and opens the PDF a second time, so it pays the
    # full deep-parse cost again — 67s on our worst sample sheet. A pit caption is
    # always "<name>詳細図 / 断面図 / リスト", so pages without one of those words
    # cannot hold a caption and are skipped (see services/page_gate.py).
    pages = relevant_pages(pdf_content, _PIT_CAPTION_KW, label="PitLocator")

    best: Dict[str, tuple] = {}  # type → (rank, ItemRegion); lower rank wins
    with _open_pdfplumber(pdf_content) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            if pages is not None and page_idx not in pages:
                continue
            lines = _build_lines(page.extract_words(
                x_tolerance=3, y_tolerance=3, keep_blank_chars=False))
            anchors = _collect_caption_anchors(lines)
            for ptype in wanted:
                if ptype in best and best[ptype][0] == 0:
                    continue  # already have the preferred 詳細図 for this type
                rank, cap_bbox = _find_pit_caption(lines, ptype)
                if cap_bbox is None or (ptype in best and rank >= best[ptype][0]):
                    continue
                # Region computed here while the page is open (edges/anchors).
                bounds = _pit_block_bounds(page, cap_bbox)
                if bounds:
                    region = _pdf_to_image_region_topdown(
                        bounds[0], bounds[1], bounds[2], bounds[3],
                        page.width, page.height, page_idx + 1)
                else:
                    region = _pit_region_from_caption(
                        cap_bbox, anchors, page.width, page.height, page_idx + 1)
                if region:
                    best[ptype] = (rank, region)

    return {ptype: region for ptype, (_rank, region) in best.items()}
