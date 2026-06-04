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

    # Positional fallback for rebar columns when arrow headers aren't extractable.
    # The ベース筋 sub-columns use graphical arrows (← ↑) that pdfplumber can't read
    # as text, so anchor on the 備考 (remarks) column instead: the two columns
    # immediately to its LEFT are rebar_x | rebar_y. Anchoring on remarks (not on
    # D) is robust to an intervening 天端高さ/▽GL elevation column between D and the
    # rebar columns — the old D-anchored gap==3 rule placed rebar onto that
    # elevation column and left the real rebar columns unmapped (→ blank rebar).
    if ("rebar_x" not in col_map or "rebar_y" not in col_map):
        d_idx = col_map.get("d")
        elev_idx = col_map.get("elev")
        rem_idx = col_map.get("remarks")
        # Left edge of the rebar block = rightmost known column before it.
        left_anchor = max([i for i in (d_idx, elev_idx) if i is not None],
                          default=-1)

        if rem_idx is not None and rem_idx - 1 > left_anchor:
            n_between = rem_idx - left_anchor - 1  # cols strictly between anchor & remarks
            if n_between >= 2:
                # … | rebar_x | rebar_y | 備考  (skips any elevation col on the left)
                col_map.setdefault("rebar_x", rem_idx - 2)
                col_map.setdefault("rebar_y", rem_idx - 1)
            elif n_between == 1:
                # … | rebar (combined) | 備考
                col_map.setdefault("rebar_x", rem_idx - 1)
        elif rem_idx is None and left_anchor >= 0:
            # No remarks column — assume the next two columns are rebar_x/rebar_y.
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
    d_str = _parse_d(get("d", "0"))

    rebar_x = get("rebar_x")
    rebar_y = get("rebar_y")
    if rebar_x and not rebar_y:
        rebar_x, rebar_y = _split_rebar_combined(rebar_x)

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


def _merge_continuation_rows(table: list, col_maps: list, data_start: int) -> list:
    """Pre-merge continuation rows (empty type column) into the previous data row.

    When pdfplumber's word-grid uses a small y_gap, multi-line note cells are
    split into separate row bands. The continuation row has an empty type column
    but non-empty remarks. This function merges such rows back into the preceding
    data row so that multi-line notes are concatenated correctly.
    """
    if not table or not col_maps:
        return table

    type_cols = {cm["type"] for cm in col_maps if "type" in cm}
    result = [list(row) if row else None for row in table]
    last_data_idx = data_start - 1

    for i in range(data_start, len(result)):
        row = result[i]
        if not row or all(c is None or str(c).strip() == "" for c in row):
            continue

        has_type = any(
            tidx < len(row) and row[tidx] and str(row[tidx]).strip()
            for tidx in type_cols
        )

        if has_type:
            last_data_idx = i
        elif last_data_idx >= data_start:
            # Continuation row: merge every non-empty cell into the previous data row
            prev = result[last_data_idx]
            for col_idx in range(min(len(row), len(prev))):
                cell = row[col_idx]
                if cell is None or str(cell).strip() == "":
                    continue
                prev_val = str(prev[col_idx] or "").strip()
                cell_str = str(cell).strip()
                prev[col_idx] = (prev_val + " " + cell_str).strip() if prev_val else cell_str
                result[i][col_idx] = None  # clear to avoid double-use

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
        words = region.extract_words(x_tolerance=4, y_tolerance=4,
                                     keep_blank_chars=False)
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
            buckets.setdefault((r, c), []).append(w["text"])

    nc, nr = len(col_bands), len(row_bands)
    table = [
        [" ".join(buckets.get((r, c), [])) for c in range(nc)]
        for r in range(nr)
    ]
    print(f"[TableExtract] word-grid: {nr} rows × {nc} cols "
          f"from bbox {tuple(round(v) for v in bbox)}")
    return table


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
    """Return True if value looks like a regular foundation type (F1, F2A, F10...)."""
    return bool(re.match(r'^F\d+[A-Za-z0-9]*$', value.strip()))


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

def _parse_table(table: list) -> Tuple[List[FoundationItem], int]:
    """Parse a single pdfplumber table into (items, last_data_row_idx).

    Handles both single-column and side-by-side (double/multi-wide) foundation
    schedules — the latter have duplicate header rows placed horizontally.

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
    table = _merge_continuation_rows(table, col_maps, data_start)

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


def extract_foundation_table_from_open(pdf) -> TableExtractionResult:
    """Extract the foundation schedule from an already-open pdfplumber document.

    Uses find_tables() (not extract_tables()) so we also get the table bounding
    box for accurate image cropping later.

    Returns a TableExtractionResult. On any failure, result.found is False
    and the caller falls back to Gemini's vision-based extraction.
    """
    try:
        pages = list(enumerate(pdf.pages, start=1))

        # Pre-filter: only run the expensive find_tables() on pages whose text
        # layer mentions a foundation-table header keyword. Fall back to every
        # page if none match (e.g. outlined/vectorised header text), so we never
        # regress to "found nothing" on an unusual sheet.
        candidates = [
            (n, p) for (n, p) in pages
            if any(k in _page_chars_text(p) for k in _TABLE_PAGE_KEYWORDS)
        ]
        scan = candidates if candidates else pages

        for page_num, page in scan:
            table_objects = page.find_tables()
            if not table_objects:
                continue
            for tbl in table_objects:
                # PRIMARY: word-centroid extraction — immune to cell-boundary issues.
                # find_tables() is used only to locate the table bbox on the page.
                word_table = _build_table_from_words(page, tbl.bbox)
                items, last_row_idx = _parse_table(word_table) if word_table else ([], -1)

                # FALLBACK: native pdfplumber cell extraction (original approach).
                if len(items) < _MIN_DATA_ROWS:
                    print("[TableExtract] word-grid yielded insufficient rows "
                          f"({len(items)}); falling back to find_tables() cells")
                    raw_data = tbl.extract()
                    items, last_row_idx = _parse_table(raw_data)

                if len(items) >= _MIN_DATA_ROWS:
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
                    print(f"[TableExtract] Found {len(items)} rows on page {page_num}, "
                          f"data_bbox={data_bbox}")
                    return TableExtractionResult(
                        items=items,
                        page_num=page_num,
                        bbox=data_bbox,
                        pdf_width=page.width,
                        pdf_height=page.height,
                    )
    except Exception as e:
        print(f"[TableExtract] Error: {e}")

    print("[TableExtract] No foundation schedule found via pdfplumber.")
    return _EMPTY_RESULT
