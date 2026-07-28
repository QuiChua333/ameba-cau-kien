"""table_extractor.py — Extract the foundation schedule table using pdfplumber.

pdfplumber detects table cell boundaries from PDF line geometry, giving reliable
column-aligned extraction for vector/CAD PDFs without any AI calls.

Returns List[FoundationItem] on success, or empty list when:
  - pdfplumber is not installed
  - The PDF is a scanned raster (no table lines)
  - No foundation schedule table is found
  - Any parse error occurs
In all failure cases the caller falls back to Gemini's vision-based extraction.
"""

import io
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False

from models import FoundationItem, Dimensions


# ──────────────────────────────────────────────────────────────────────────────
# Column header keywords
# ──────────────────────────────────────────────────────────────────────────────

_HEADER_KEYWORDS: dict = {
    "type":    ["基礎符号", "符号", "種別"],
    "lx":      ["Lx", "Ｌｘ", "lx", "ＬＸ"],
    "ly":      ["Ly", "Ｌｙ", "ly", "ＬＹ"],
    "d":       ["D", "Ｄ"],
    # Top-of-foundation elevation column (天端高さ / ▽GL). Mapping it explicitly
    # keeps it from being mistaken for a rebar column in the positional fallback.
    "elev":    ["天端", "天端高さ", "底盤天端", "ﾚﾍﾞﾙ", "レベル"],
    # Rebar headers appear as arrow symbols (← ↓) in many Japanese tables,
    # or as "X方向"/"Y方向" text. Include all common variants.
    "rebar_x": ["X方向", "Ｘ方向", "X筋", "ベース筋X", "←", "⟵", "→", "⟶", "➡"],
    "rebar_y": ["Y方向", "Ｙ方向", "Y筋", "ベース筋Y", "↓", "⬇", "↑", "⬆", "⇩"],
    "remarks": ["備考", "特記"],
}

# Minimum confidence: a table is a foundation schedule if it has ≥ this many data rows
_MIN_DATA_ROWS = 2


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(text: Optional[str]) -> str:
    """Strip whitespace and normalize full-width ASCII characters."""
    if text is None:
        return ""
    # Convert full-width ASCII to half-width (Ｆ → F, Ｌ → L, etc.)
    s = text.strip()
    result = []
    for ch in s:
        cp = ord(ch)
        if 0xFF01 <= cp <= 0xFF5E:
            result.append(chr(cp - 0xFEE0))
        else:
            result.append(ch)
    return "".join(result).strip()


def _detect_col_map(rows: list, max_header_rows: int = 3,
                    start_col: int = 0) -> tuple:
    """Scan the first `max_header_rows` rows to build a column → index map.

    start_col: ignore all columns before this index (used to locate the right-half
    header in a side-by-side double-wide table layout).
    Returns (col_map, data_start_row_idx).

    Positional fallback: if rebar columns are not identified by keyword (e.g. the
    header uses graphical arrows that pdfplumber can't extract as text), infer them
    from the gap between the D column and the remarks column.
    """
    col_map: dict = {}
    data_start = 0

    for row_idx, row in enumerate(rows[:max_header_rows]):
        if row is None:
            continue
        for col_idx, cell in enumerate(row):
            if col_idx < start_col:
                continue
            norm = _normalize(str(cell) if cell is not None else "")
            for key, keywords in _HEADER_KEYWORDS.items():
                if any(kw in norm for kw in keywords):
                    if key not in col_map:
                        col_map[key] = col_idx
                        data_start = row_idx + 1

    # Rebar columns when arrow headers (← ↑) aren't extractable as text.
    # PRIMARY (data-driven): the rebar columns are those BETWEEN the D/▽GL anchor
    # and 備考 that actually carry a rebar spec ("N-Dnn") in the data rows. This is
    # robust to the empty spacer columns the word-grid inserts between sub-columns
    # — the first spec-bearing column is rebar_x (←), the second is rebar_y (↑).
    # FALLBACK (positional): if no data rows are available (header-only / arrows),
    # take the two columns immediately before 備考.
    if ("rebar_x" not in col_map or "rebar_y" not in col_map):
        d_idx = col_map.get("d")
        elev_idx = col_map.get("elev")
        rem_idx = col_map.get("remarks")
        left_anchor = max([i for i in (d_idx, elev_idx) if i is not None],
                          default=-1)

        if left_anchor >= 0:
            ncol = max((len(r) for r in rows if r), default=0)
            hi = rem_idx if rem_idx is not None else min(left_anchor + 6, ncol)

            rebar_cols = []
            for col in range(max(left_anchor + 1, start_col), hi):
                hits = sum(
                    1 for row in rows[data_start:]
                    if row and col < len(row)
                    and _REBAR_SPEC_RE.search(str(row[col] or ""))
                )
                if hits:
                    rebar_cols.append(col)

            if rebar_cols:
                col_map.setdefault("rebar_x", rebar_cols[0])
                if len(rebar_cols) >= 2:
                    col_map.setdefault("rebar_y", rebar_cols[1])
            elif rem_idx is not None and rem_idx - 1 > left_anchor:
                n_between = rem_idx - left_anchor - 1
                if n_between >= 2:
                    col_map.setdefault("rebar_x", rem_idx - 2)
                    col_map.setdefault("rebar_y", rem_idx - 1)
                elif n_between == 1:
                    col_map.setdefault("rebar_x", rem_idx - 1)
            elif rem_idx is None:
                col_map.setdefault("rebar_x", left_anchor + 1)
                col_map.setdefault("rebar_y", left_anchor + 2)

    return col_map, data_start


def _detect_all_col_maps(rows: list, max_header_rows: int = 3) -> tuple:
    """Detect column maps for all sub-tables, handling side-by-side layouts.

    Some foundation schedules are too long to fit in one column and are split
    into two (or more) sub-tables placed side by side on the same page.
    pdfplumber sees the whole thing as one wide table with duplicated headers.
    This function detects each sub-table header and returns one col_map per
    sub-table.

    Returns (list[col_map], data_start_row_idx).
    """
    primary_map, data_start = _detect_col_map(rows, max_header_rows, start_col=0)
    if "type" not in primary_map:
        return [primary_map], data_start

    col_maps = [primary_map]
    # The right-half header starts after the last column used by the left half
    search_from = max(primary_map.values()) + 1

    while True:
        next_map, _ = _detect_col_map(rows, max_header_rows, start_col=search_from)
        if "type" not in next_map:
            break
        col_maps.append(next_map)
        print(f"[TableExtract] Detected sub-table #{len(col_maps)} "
              f"starting at col {next_map['type']}")
        search_from = max(next_map.values()) + 1

    return col_maps, data_start


def _extract_item_from_row(row: list, col_map: dict) -> "Optional[FoundationItem]":
    """Extract one FoundationItem from a pdfplumber row using the given col_map.

    Returns None when the row doesn't contain a valid foundation type code.
    """
    def get(key: str, default: str = "") -> str:
        idx = col_map.get(key)
        if idx is None or idx >= len(row) or row[idx] is None:
            return default
        return _normalize(str(row[idx]))

    type_val = get("type")
    if not type_val or not _is_foundation_type(type_val):
        return None
    if re.match(r'^F[WG]', type_val):
        return None

    lx = _parse_number(get("lx")) or 0.0
    ly = _parse_number(get("ly")) or 0.0
    d_raw = get("d", "0")
    d_str = _parse_d(d_raw)

    rebar_x = get("rebar_x")
    rebar_y = get("rebar_y")
    if rebar_x and not rebar_y:
        rebar_x, rebar_y = _split_rebar_combined(rebar_x)

    # Recover a rebar spec that merged into the D cell when the word-grid collapsed
    # the D and ベース筋(←) columns (e.g. D cell "1000~300 16-D16"). _parse_d strips
    # the spec out of D, so put it back as rebar_x when the ← column is itself empty
    # (the merge leaves the separate ← column blank for those rows).
    if not _clean_rebar_spec(rebar_x):
        d_rebar = _clean_rebar_spec(d_raw)
        if d_rebar:
            rebar_x = d_rebar

    remarks = get("remarks")
    remarks = re.sub(r'^\d+\s+', '', remarks).strip()

    # Recovery: if rebar_y is still empty, try to peel a rebar spec from remarks.
    # This occurs when the word-grid collapses the narrow rebar-Y column into the
    # remarks column (columns too close together for the x_gap threshold).
    if not rebar_y and remarks:
        # Case A: rebar spec at the START of remarks, e.g. "22-D16 -" or "48-D13 -"
        m = re.match(r'^(\d+-D\d+(?:@\d+)?)\s*(.*)', remarks, re.DOTALL)
        if m:
            rebar_y = m.group(1)
            remarks = m.group(2).strip()
        else:
            # Case B: rebar spec at the END of remarks, e.g. "B0x x B0y = ... 44-D16"
            m = re.search(r'^(.*\S)\s+(\d+-D\d+(?:@\d+)?)$', remarks, re.DOTALL)
            if m:
                rebar_y = m.group(2)
                remarks = m.group(1).strip()
            else:
                # Case C: rebar spec embedded mid-remarks, e.g. "B0x...1,300 42-D13 B1x...3,200"
                m = re.search(r'^(.*?)\s+(\d+-D\d+(?:@\d+)?)\s+(.*)', remarks, re.DOTALL)
                if m:
                    rebar_y = m.group(2)
                    remarks = (m.group(1) + " " + m.group(3)).strip()

    # Include rebar columns in classification text: pdfplumber sometimes shifts
    # "B...x x" into a rebar cell when column boundaries are slightly off.
    classify_text = " ".join(filter(None, [rebar_x, rebar_y, remarks]))
    classification = _classify_from_remarks(classify_text)

    # Sanitise rebar AFTER classification (which legitimately reads B-formulas
    # that pdfplumber may have shifted into a rebar cell). The stored fields keep
    # only the rebar spec, never captions / F-code lists.
    rebar_x = _clean_rebar_spec(rebar_x)
    rebar_y = _clean_rebar_spec(rebar_y)

    try:
        return FoundationItem(
            type=type_val,
            dimensions=Dimensions(Lx=lx, Ly=ly, D=d_str),
            top_elevation=None,
            rebar_x=rebar_x,
            rebar_y=rebar_y,
            remarks=remarks,
            classification=classification,
            region=None,
            image_base64=None,
        )
    except Exception as e:
        print(f"[TableExtract] Row parse error for type={type_val}: {e}")
        return None


# Drawing-section caption/annotation keywords that mark the end of the schedule
# (the cross-section drawings drawn below it). Never appear in real table cells.
_DRAWING_SECTION_RE = re.compile(
    r'基礎断面|基礎平面|断面図|柱型リスト|柱廻り|はかま筋|補強筋|スタイロ|'
    r'ベース筋|偏心方向|つなぎ筋|捨て[ｺコ]|開口補強|地盤改良'
)


def _merge_continuation_rows(table: list, col_maps: list, data_start: int,
                             row_centers: Optional[list] = None) -> list:
    """Merge multi-line cell bands (no type column) into the row they belong to.

    The word-grid splits a tall multi-line cell (e.g. a two-line 備考 note) into
    several row bands. A band that carries content but no foundation type is an
    "orphan" belonging to a neighbouring data row.

    Crucially, the orphan does NOT always belong to the row ABOVE: when a row's
    type code is vertically centred in a tall row, the cell's FIRST line lands in
    a band ABOVE the type band, so blindly merging "into the previous row" leaks
    that line onto the preceding foundation (e.g. F18's "B0x…" line stolen by
    F140). With `row_centers` we instead attach each orphan to the NEAREST typed
    row by vertical distance; without them we fall back to the previous row.

    Contributions are concatenated in row order, so an orphan above a type keeps
    its content before the type row's own cell text.
    """
    if not table or not col_maps:
        return table

    type_cols = {cm["type"] for cm in col_maps if "type" in cm}
    n = len(table)

    def _is_empty(row) -> bool:
        return not row or all(c is None or str(c).strip() == "" for c in row)

    def _has_type(row) -> bool:
        return any(tidx < len(row) and row[tidx] and str(row[tidx]).strip()
                   for tidx in type_cols)

    # The cross-section drawings sit directly below the schedule. The first row
    # carrying a drawing caption/annotation keyword (基礎断面, 基礎平面, 柱型リスト…)
    # marks the end of tabular data — rows from there down must NOT be merged into
    # the last foundation (otherwise the drawing's F-code captions corrupt its
    # Lx/Ly/D cells, e.g. F18 Lx → "5,400 F11,F12,…" → unparseable → 0).
    # Scan ONLY within the table's own columns (≤ rightmost mapped column); the
    # far-right "ＴＮＦ工法概要" spec block contains "地盤改良" etc. and would
    # otherwise truncate the table prematurely.
    right_edge = max((max(cm.values()) for cm in col_maps if cm), default=n)
    cutoff = n
    for i in range(data_start, n):
        row = table[i]
        if row and any(
            _DRAWING_SECTION_RE.search(str(row[c] or ""))
            for c in range(min(len(row), right_edge + 1))
        ):
            cutoff = i
            break

    real = [i for i in range(data_start, cutoff) if not _is_empty(table[i])]
    typed = [i for i in real if _has_type(table[i])]
    if not typed:
        return table

    def _center(i: int) -> float:
        if row_centers and 0 <= i < len(row_centers):
            return float(row_centers[i])
        return float(i)  # index proxy → ties resolve to the previous row

    # Assign every real row to a target typed row.
    typed_set = set(typed)
    assign: dict = {}
    for i in real:
        if i in typed_set:
            assign[i] = i
            continue
        prev_t = max((t for t in typed if t < i), default=None)
        next_t = min((t for t in typed if t > i), default=None)
        if prev_t is not None and next_t is not None:
            assign[i] = (prev_t if abs(_center(i) - _center(prev_t))
                         <= abs(_center(i) - _center(next_t)) else next_t)
        else:
            assign[i] = prev_t if prev_t is not None else next_t

    result = [list(row) if row else None for row in table]
    ncol = max(len(table[t]) for t in typed)

    from collections import defaultdict
    groups: dict = defaultdict(list)
    for i in real:
        groups[assign[i]].append(i)

    for t, members in groups.items():
        members.sort()  # row order → preserves "orphan above type" ordering
        merged = [None] * ncol
        for col_idx in range(ncol):
            parts = []
            for i in members:
                row = table[i]
                if col_idx < len(row):
                    cell = row[col_idx]
                    if cell is not None and str(cell).strip():
                        parts.append(str(cell).strip())
            merged[col_idx] = " ".join(parts) if parts else None
        result[t] = merged

    # Clear rows whose content was merged into another row.
    for i in real:
        if assign[i] != i and result[i]:
            result[i] = [None] * len(result[i])

    return result


def _gap_cluster(vals: list, gap: float) -> list:
    """Group sorted float values into bands separated by gaps > `gap`.

    Returns list of (lo, hi) band tuples. Words whose x/y centroid falls
    in a band are assigned to the same column/row.
    """
    if not vals:
        return []
    bands, lo, hi = [], float(vals[0]), float(vals[0])
    for v in vals[1:]:
        fv = float(v)
        if fv - hi > gap:
            bands.append((lo, hi))
            lo = fv
        hi = fv
    bands.append((lo, hi))
    return bands


def _join_cell(words: list) -> str:
    """Join the words of one (row, col) bucket back into readable cell text.

    Sort by (rounded top, x0) so a multi-line cell (e.g. a two-line 備考 note
    "B0x…/B1x…") is read line-by-line, top line fully before the bottom line.
    Combined with extract_words(use_text_flow=True) — which rebuilds whole words
    from the PDF's authored character stream instead of geometrically x-sorting
    glyphs — this avoids the char-by-char interleaving ("B B 0 1 x x …") that
    appears on glyph-split CAD PDFs whose two stacked lines sit within y_tolerance.
    """
    if not words:
        return ""
    ws = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    return " ".join(w.get("text", "") for w in ws)


def _build_table_from_words(page, bbox,
                             x_gap: float = 14.0,
                             y_gap: float = 5.0) -> list:
    """Build a 2-D cell table from word centroids instead of line-detected borders.

    Why this is better than find_tables():
      • Immune to column-boundary mis-detection (no dependency on PDF line geometry)
      • Works correctly for double-wide side-by-side layouts
      • Handles graphical (non-text) column headers (arrows ← ↓)
      • No cross-row or cross-column cell contamination

    Each word is cropped to `bbox`, assigned to the nearest column/row band by its
    centroid, then all words in the same (row, col) bucket are joined with a space.

    x_gap: minimum white-space gap between distinct columns (points).
    y_gap: minimum gap between distinct row bands. Keep small (≤6) so that
           two-line remarks cells are merged into one row while separate table
           rows stay distinct.
    """
    try:
        region = page.crop(bbox)
        # use_text_flow=True rebuilds words from the PDF's authored character
        # stream (reading order) instead of re-sorting glyphs by x. On glyph-split
        # CAD PDFs this keeps a two-line cell's stacked lines from interleaving
        # char-by-char ("B B 0 1 x x …") and reassembles proper words ("B0x").
        words = region.extract_words(x_tolerance=4, y_tolerance=4,
                                     keep_blank_chars=False, use_text_flow=True)
    except Exception:
        return []
    if not words:
        return []

    x_mids = sorted({round((w["x0"] + w["x1"]) / 2, 1) for w in words})
    y_mids = sorted({round((w["top"] + w["bottom"]) / 2, 1) for w in words})
    col_bands = _gap_cluster(x_mids, x_gap)
    row_bands = _gap_cluster(y_mids, y_gap)

    half_x = x_gap / 2

    def _col(xc: float) -> int:
        for i, (lo, hi) in enumerate(col_bands):
            if lo - half_x <= xc <= hi + half_x:
                return i
        return -1

    # ── Type-anchored row bands ────────────────────────────────────────────────
    # Every foundation row carries a 基礎符号 code (F1, F1A, "F2,F2A"…) in the type
    # column, regularly spaced. Gap-clustering on ALL word y-centroids chains
    # tightly-packed rows together (their 2-line ベース筋/備考 cells fill the gaps),
    # merging e.g. F1+F1A+F1B into one row with Lx=0 and a run-on D. Banding on the
    # type-code anchors instead keeps each foundation on its own row. Falls back to
    # the gap-cluster bands when there is no clean column of codes.
    type_words = [w for w in words if re.match(r'^F\d', (w["text"] or "").strip())]
    col_code_count: dict = {}
    for w in type_words:
        c = _col((w["x0"] + w["x1"]) / 2)
        if c >= 0:
            col_code_count[c] = col_code_count.get(c, 0) + 1
    # Schedule type columns carry MANY codes (≥3); sparse columns are section-drawing
    # captions ("F2B 平面図") and must not act as anchors.
    anchor_cols = {c for c, n in col_code_count.items() if n >= 3}
    anchor_ys = sorted((w["top"] + w["bottom"]) / 2 for w in type_words
                       if _col((w["x0"] + w["x1"]) / 2) in anchor_cols)

    if len(anchor_ys) >= 3:
        gaps = sorted(anchor_ys[i + 1] - anchor_ys[i]
                      for i in range(len(anchor_ys) - 1))
        spacing = gaps[len(gaps) // 2] or y_gap
        # Collapse anchors that share a row (side-by-side sub-tables aligned in y).
        groups: list = []
        for y in anchor_ys:
            if groups and y - groups[-1][-1] < spacing * 0.5:
                groups[-1].append(y)
            else:
                groups.append([y])
        anchors = [sum(g) / len(g) for g in groups]

        # Only override the gap-cluster bands when they actually MERGE rows: i.e.
        # several anchors fall into one gap-band. When gap-clustering already gives a
        # distinct band per anchor (the common case), keep it untouched so files that
        # extract correctly today don't shift. Anchor banding is the repair path only.
        def _gap_row(yc: float) -> int:
            for i, (lo, hi) in enumerate(row_bands):
                if lo - y_gap <= yc <= hi + y_gap:
                    return i
            return -1
        occupied = [_gap_row(a) for a in anchors]
        merged = len(anchors) - len({i for i in occupied if i >= 0})

        if len(anchors) >= 3 and merged >= 2:
            ys_all = [(w["top"] + w["bottom"]) / 2 for w in words]
            top_all, bot_all = min(ys_all), max(ys_all)
            mids = [(anchors[i] + anchors[i + 1]) / 2 for i in range(len(anchors) - 1)]
            bands: list = [(top_all - 1, anchors[0] - spacing * 0.5)]  # header band
            prev = anchors[0] - spacing * 0.5
            for i, a in enumerate(anchors):
                hi = mids[i] if i < len(mids) else max(bot_all + 1, a + spacing * 0.5)
                bands.append((prev, hi))
                prev = hi
            row_bands = bands
            print(f"[TableExtract] row-merge detected ({merged}); "
                  f"using {len(anchors)} type-code anchors for row bands")

    # Anchor bands are contiguous (boundaries at anchor midpoints); gap-cluster bands
    # have gaps between them. A small tolerance handles points exactly on a boundary.
    def _row(yc: float) -> int:
        for i, (lo, hi) in enumerate(row_bands):
            if lo - y_gap <= yc <= hi + y_gap:
                return i
        return -1

    buckets: dict = {}
    for w in words:
        c = _col((w["x0"] + w["x1"]) / 2)
        r = _row((w["top"] + w["bottom"]) / 2)
        if r >= 0 and c >= 0:
            buckets.setdefault((r, c), []).append(w)

    nc, nr = len(col_bands), len(row_bands)
    table = [
        [_join_cell(buckets.get((r, c), [])) for c in range(nc)]
        for r in range(nr)
    ]
    # Vertical centre of each row band — lets the continuation merge attach an
    # orphan (type-less) band to the nearest row by geometry, not just "previous".
    row_centers = [round((lo + hi) / 2, 1) for (lo, hi) in row_bands]
    print(f"[TableExtract] word-grid: {nr} rows × {nc} cols "
          f"from bbox {tuple(round(v) for v in bbox)}")
    return table, row_centers


def _parse_number(text: Optional[str]) -> Optional[float]:
    """Parse a Japanese-format number '2,000' or '2000' → 2000.0"""
    if not text:
        return None
    cleaned = _normalize(text).replace(',', '').replace('，', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_d(text: Optional[str]) -> str:
    """Parse D field into a canonical "D" or "D~D2" string.

    Handles multiple source formats:
      '700'          → '700'
      '700~300'      → '700~300'
      '700\n300'     → '700~300'   (multi-line cell)
      '900(~250)'    → '900~250'   (parenthetical range from PDF)
      '900（~250）'   → '900~250'   (full-width parentheses)
    """
    if not text:
        return "0"
    s = _normalize(text).replace('\r', '').strip()

    # Drop any ▽GL/▽SGL elevation annotation that leaked into the D cell
    # (e.g. "1050~300 ▽SGL 564" → "1050~300"). D never contains a ▽ marker.
    s = re.sub(r'▽.*$', '', s, flags=re.DOTALL).strip()
    # Drop a rebar spec that merged into the D cell when the word-grid collapsed
    # the D and ベース筋(←) columns into one (e.g. "1000 18-D16" → "1000",
    # "1000~300 34-D13" → "1000~300"). D never contains an "N-Dnn" rebar spec.
    s = re.sub(r'\s*\d+\s*-\s*D\d+(?:@[\d,]+)?.*$', '', s, flags=re.DOTALL).strip()
    if not s:
        return "0"

    # Parenthetical format: "900(~250)" or "900（~250）"
    m = re.search(r'^([\d,，]+)\s*[（(]~?\s*([\d,，]+)[)）]', s)
    if m:
        d_main = m.group(1).replace(',', '').replace('，', '').strip()
        d2     = m.group(2).replace(',', '').replace('，', '').strip()
        if d_main and d2:
            return f"{d_main}~{d2}"

    # Newline-separated or tilde-separated range
    s = s.replace('\n', '~')
    parts = s.split('~')
    cleaned_parts = []
    for p in parts:
        p2 = p.replace(',', '').replace('，', '').strip()
        if p2:
            cleaned_parts.append(p2)
    result = '~'.join(cleaned_parts) if cleaned_parts else "0"
    return result or "0"


def _classify_from_remarks(text: str) -> str:
    """Derive DD/D/TNF classification from remarks (and optionally rebar) text.

    Primary pattern: "B0x x B0y = ..." / "Bx x By = ..."
    Fallback pattern: "B0y = NNN" — occurs when pdfplumber's column boundary
    misaligns and drops the "B...x x" prefix into an adjacent rebar cell.
    """
    # Primary: full "B...x x B...y" formula
    full_re = re.compile(r'B\w*[xX]\s*[×x]\s*B\w*[yY]')
    count = len(full_re.findall(text))

    if count == 0:
        # Fallback: match "B\w*y = <number>" patterns (column-split artifacts)
        trunc_re = re.compile(r'\bB\w*[yY]\s*=\s*[\d,]')
        count = len(trunc_re.findall(text))

    if count >= 2:
        return "DD"
    elif count == 1:
        return "D"
    return "TNF"


def _is_foundation_type(value: str) -> bool:
    """Return True for a single foundation code (F1, F2A, F10…) OR a combined cell
    listing several of them ("F11,F23", "F132、F21"). Combined cells are kept here
    and split downstream by split_combined_types so the word-grid's reliable
    tabular data (Lx/Ly/D/rebar) isn't discarded for grouped rows."""
    v = value.strip()
    if re.match(r'^F\d+[A-Za-z0-9]*$', v):
        return True
    parts = [p.strip() for p in re.split(r'[,、，\s]+', v) if p.strip()]
    return len(parts) >= 2 and all(re.match(r'^F\d+[A-Za-z0-9]*$', p) for p in parts)


_REBAR_SPEC_RE = re.compile(r'\d+-D\d+(?:@\d+)?')


def _split_rebar_combined(text: str) -> tuple:
    """Split a combined rebar cell into (rebar_x, rebar_y).

    Handles three formats:
      1. Explicit Y marker: "X方向: 34-D13\nY方向: 34-D13"
      2. Two specs in one cell: "34-D13 34-D13" or "34-D13\n34-D13"
         (happens when pdfplumber merges the two rebar sub-columns)
      3. Single value: return (value, "")
    """
    # Pattern 1: explicit Y direction marker
    y_match = re.search(r'Y[方向筋：:]\s*(.+)', text, re.DOTALL)
    if y_match:
        rebar_y = y_match.group(1).strip()
        x_match = re.search(r'^(.+?)(?=Y[方向筋：:]|\Z)', text, re.DOTALL)
        rebar_x = x_match.group(1).strip() if x_match else text.strip()
        rebar_x = re.sub(r'^X[方向筋：:]\s*', '', rebar_x).strip()
        return rebar_x, rebar_y

    # Pattern 2: two rebar specs in the same cell (merged sub-columns)
    specs = _REBAR_SPEC_RE.findall(text)
    if len(specs) >= 2:
        return specs[0], specs[1]
    return text.strip(), ""


def _clean_rebar_spec(text: str) -> str:
    """Keep only a valid rebar spec (e.g. '18-D16'); drop everything else.

    Wide detail rows (e.g. F18's shared 基礎断面) push drawing captions and
    foundation-code lists ("F11,F12,F20,…") or B-formulas into a rebar column.
    A real foundation rebar cell is always an "N-Dnn" spec, so extract that and
    discard any non-spec text. Returns "" when no spec is present.
    """
    if not text:
        return ""
    m = _REBAR_SPEC_RE.search(text)
    return m.group(0) if m else ""


# ──────────────────────────────────────────────────────────────────────────────
# Table parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_table(table: list, row_centers: Optional[list] = None) -> Tuple[List[FoundationItem], int]:
    """Parse a single pdfplumber table into (items, last_data_row_idx).

    Handles both single-column and side-by-side (double/multi-wide) foundation
    schedules — the latter have duplicate header rows placed horizontally.

    row_centers: optional per-row vertical centres (from the word-grid) used to
    attach multi-line orphan bands to the nearest row by geometry.

    last_data_row_idx: index into `table` of the last row that contained valid
    foundation data (used to crop the image). Returns ([], -1) when the table
    doesn't look like a foundation schedule.
    """
    if not table or len(table) < _MIN_DATA_ROWS + 1:
        return [], -1

    col_maps, data_start = _detect_all_col_maps(table)

    # Must have at least type and one dimension column in the primary sub-table
    primary = col_maps[0]
    if "type" not in primary or ("lx" not in primary and "d" not in primary):
        return [], -1

    # Merge continuation rows (multi-line notes cells split across row bands)
    table = _merge_continuation_rows(table, col_maps, data_start, row_centers)

    items: List[FoundationItem] = []
    last_data_row_idx = -1
    seen_types: set = set()  # guard against duplicates across sub-tables

    for row_idx, row in enumerate(table):
        if row_idx < data_start:
            continue
        if not row or all(c is None or str(c).strip() == "" for c in row):
            continue

        row_had_item = False
        for col_map in col_maps:
            item = _extract_item_from_row(row, col_map)
            if item is None or item.type in seen_types:
                continue
            items.append(item)
            seen_types.add(item.type)
            row_had_item = True

        if row_had_item:
            last_data_row_idx = row_idx

    return items, last_data_row_idx


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TableExtractionResult:
    """Foundation table extracted by pdfplumber, with page location for image cropping."""
    items: List[FoundationItem] = field(default_factory=list)
    page_num: int = 0                  # 1-based page number where table was found
    bbox: Optional[Tuple] = None       # (x0, top, x1, bottom) in pdfplumber coords (top-left origin, points)
    pdf_width: float = 0.0             # page width in PDF points
    pdf_height: float = 0.0            # page height in PDF points

    @property
    def found(self) -> bool:
        return len(self.items) >= _MIN_DATA_ROWS


_EMPTY_RESULT = TableExtractionResult()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

# Keywords that mark a page as a *candidate* for holding the foundation
# schedule. find_tables() (expensive edge detection on dense CAD pages) only
# runs on pages whose text layer contains one of these — the foundation table
# always carries a 符号/種別 header. Cheap char-scan first, heavy table
# detection only where it can pay off.
_TABLE_PAGE_KEYWORDS = ("符号", "種別", "基礎リスト")


def _page_chars_text(page) -> str:
    """Concatenate a page's raw chars into a string for a cheap keyword scan.

    Reads `page.chars` (parsed once and cached on the page object by pdfplumber)
    rather than `extract_text()` so we skip the layout pass and reuse the same
    cached objects that `extract_words()` needs later in the shared-document flow.
    """
    try:
        return "".join(c.get("text", "") for c in page.chars)
    except Exception:
        return ""


def extract_foundation_table(pdf_bytes: bytes) -> TableExtractionResult:
    """Extract the foundation schedule from a PDF using pdfplumber line detection.

    Thin wrapper that opens the document; see extract_foundation_table_from_open
    for the actual logic (also reused when the document is opened once and shared
    with oval-marker detection).
    """
    if not _PDFPLUMBER_AVAILABLE:
        print("[TableExtract] pdfplumber not installed — skipping table extraction.")
        return _EMPTY_RESULT
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return extract_foundation_table_from_open(pdf)
    except Exception as e:
        print(f"[TableExtract] Error: {e}")
        return _EMPTY_RESULT


def extract_foundation_table_from_open(pdf, pages: Optional[set] = None) -> TableExtractionResult:
    """Extract the foundation schedule from an already-open pdfplumber document.

    Uses find_tables() (not extract_tables()) so we also get the table bounding
    box for accurate image cropping later.

    pages: 0-based page indices to consider; None = every page. Honour the caller's
    gate — page.chars / find_tables() on a skipped page would trigger the deep
    content-stream parse the gate exists to avoid (see
    textlayer_phase1._relevant_page_indices).

    Returns a TableExtractionResult. On any failure, result.found is False
    and the caller falls back to Gemini's vision-based extraction.
    """
    try:
        candidate_pages = [(n, p) for n, p in enumerate(pdf.pages, start=1)
                           if pages is None or (n - 1) in pages]

        # Pre-filter: only run the expensive find_tables() on pages whose text
        # layer mentions a foundation-table header keyword. Fall back to every
        # page if none match (e.g. outlined/vectorised header text), so we never
        # regress to "found nothing" on an unusual sheet.
        candidates = [
            (n, p) for (n, p) in candidate_pages
            if any(k in _page_chars_text(p) for k in _TABLE_PAGE_KEYWORDS)
        ]
        scan = candidates if candidates else candidate_pages

        # A page's find_tables() often returns several overlapping regions — the
        # tight schedule box AND a page-wide box that also swallows the cross-section
        # drawings below it. The wide box pollutes the word-grid and merges adjacent
        # foundation rows (e.g. F1A+F1B+F2 collapse, losing their Lx/Ly). So DON'T
        # return the first table that parses; evaluate every candidate and keep the
        # best one — most rows with real dimensions (Lx>0), then most rows overall.
        best: Optional[TableExtractionResult] = None
        best_score: tuple = (-1, -1)

        def _score(items: list) -> tuple:
            clean = sum(1 for it in items
                        if it.dimensions.Lx > 0 and " " not in it.type)
            return (clean, len(items))

        for page_num, page in scan:
            table_objects = page.find_tables()
            if not table_objects:
                continue
            for tbl in table_objects:
                # PRIMARY: word-centroid extraction — immune to cell-boundary issues.
                # find_tables() is used only to locate the table bbox on the page.
                word_grid = _build_table_from_words(page, tbl.bbox)
                word_table, word_row_centers = word_grid if word_grid else (None, None)
                items, last_row_idx = (
                    _parse_table(word_table, word_row_centers) if word_table else ([], -1)
                )

                # FALLBACK: native pdfplumber cell extraction (original approach).
                if len(items) < _MIN_DATA_ROWS:
                    raw_data = tbl.extract()
                    items, last_row_idx = _parse_table(raw_data)
                    word_table = None

                if len(items) < _MIN_DATA_ROWS:
                    continue

                # Crop image bbox to just the data section.
                crop_bottom = tbl.bbox[3]
                if word_table and 0 <= last_row_idx < len(word_table):
                    # Approximate the bottom of the last data row from the
                    # word-grid row index; fall back to full table bottom.
                    try:
                        crop_bottom = tbl.rows[last_row_idx].bbox[3] + 4
                    except (IndexError, AttributeError):
                        pass
                elif 0 <= last_row_idx < len(tbl.rows):
                    crop_bottom = tbl.rows[last_row_idx].bbox[3] + 4
                data_bbox = (tbl.bbox[0], tbl.bbox[1], tbl.bbox[2], crop_bottom)

                score = _score(items)
                print(f"[TableExtract] candidate: {len(items)} rows "
                      f"(score={score}) on page {page_num}, "
                      f"bbox={tuple(round(v) for v in tbl.bbox)}")
                if score > best_score:
                    best_score = score
                    best = TableExtractionResult(
                        items=items,
                        page_num=page_num,
                        bbox=data_bbox,
                        pdf_width=page.width,
                        pdf_height=page.height,
                    )

        if best is not None:
            print(f"[TableExtract] Selected best table: {len(best.items)} rows "
                  f"on page {best.page_num}, data_bbox={best.bbox}")
            return best
    except Exception as e:
        print(f"[TableExtract] Error: {e}")

    # FALLBACK — newer sheets have no page-wide schedule at all: each foundation
    # carries its own single-row mini table (基礎符号/Lx/Ly/D/ベース筋, no 備考)
    # under its 基礎断面. Only reached when the classic layout yielded nothing, so
    # documents that DO have a schedule keep their existing behaviour exactly.
    print("[TableExtract] No page-wide schedule — trying per-foundation mini tables.")
    try:
        from services.foundation_list_v2 import extract_per_foundation_list
        v2 = extract_per_foundation_list(pdf, pages=pages)
        if v2.items:
            return v2
    except Exception as e:
        print(f"[TableExtract] per-foundation fallback failed: {e}")

    print("[TableExtract] No foundation schedule found via pdfplumber.")
    return _EMPTY_RESULT
