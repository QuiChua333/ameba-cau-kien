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


def _detect_col_map(rows: list, max_header_rows: int = 3) -> tuple:
    """Scan the first `max_header_rows` rows to build a column → index map.

    Returns (col_map, data_start_row_idx).
    col_map e.g. {"type": 0, "lx": 1, "ly": 2, "d": 3, "rebar_x": 4, "rebar_y": 5}

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
            norm = _normalize(str(cell) if cell is not None else "")
            for key, keywords in _HEADER_KEYWORDS.items():
                if any(kw in norm for kw in keywords):
                    if key not in col_map:
                        col_map[key] = col_idx
                        data_start = row_idx + 1  # data starts after this header row

    # Positional fallback for rebar columns when arrow headers aren't extractable
    if ("rebar_x" not in col_map or "rebar_y" not in col_map):
        d_idx = col_map.get("d")
        rem_idx = col_map.get("remarks")
        if d_idx is not None and rem_idx is not None and rem_idx > d_idx:
            gap = rem_idx - d_idx  # number of columns between D and remarks
            if gap == 3:
                # Standard layout: D | rebar_x | rebar_y | remarks
                if "rebar_x" not in col_map:
                    col_map["rebar_x"] = d_idx + 1
                if "rebar_y" not in col_map:
                    col_map["rebar_y"] = d_idx + 2
            elif gap == 2:
                # Compressed layout: D | rebar (combined) | remarks
                if "rebar_x" not in col_map:
                    col_map["rebar_x"] = d_idx + 1
        elif d_idx is not None and rem_idx is None:
            # No remarks column detected — assume rebar_x/rebar_y are the next two cols
            if "rebar_x" not in col_map:
                col_map["rebar_x"] = d_idx + 1
            if "rebar_y" not in col_map:
                col_map["rebar_y"] = d_idx + 2

    return col_map, data_start


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


def _classify_from_remarks(remarks: str) -> str:
    """Derive DD/D/TNF classification from remarks column content."""
    pattern = re.compile(r'B\w*[xX]\s*[×x]\s*B\w*[yY]')
    count = len(pattern.findall(remarks))
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


# ──────────────────────────────────────────────────────────────────────────────
# Table parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_table(table: list) -> Tuple[List[FoundationItem], int]:
    """Parse a single pdfplumber table into (items, last_data_row_idx).

    last_data_row_idx: index into `table` of the last row that contained valid
    foundation data. Used to crop the image only up to that row, excluding
    cross-section drawings that may occupy rows below the data section.
    Returns ([], -1) if the table doesn't look like a foundation schedule.
    """
    if not table or len(table) < _MIN_DATA_ROWS + 1:
        return [], -1

    col_map, data_start = _detect_col_map(table)

    # Must have at least type and one dimension column to be a foundation schedule
    if "type" not in col_map or ("lx" not in col_map and "d" not in col_map):
        return [], -1

    items: List[FoundationItem] = []
    last_data_row_idx = -1

    for row_idx, row in enumerate(table):
        if row_idx < data_start:
            continue
        if not row or all(c is None or str(c).strip() == "" for c in row):
            continue

        def get(key: str, default: str = "") -> str:
            idx = col_map.get(key)
            if idx is None or idx >= len(row) or row[idx] is None:
                return default
            return _normalize(str(row[idx]))

        type_val = get("type")
        if not type_val or not _is_foundation_type(type_val):
            continue

        # Skip beams — their geometry comes from cross-section drawings (Gemini)
        if re.match(r'^F[WG]', type_val):
            continue

        lx = _parse_number(get("lx")) or 0.0
        ly = _parse_number(get("ly")) or 0.0
        d_str = _parse_d(get("d", "0"))

        rebar_x = get("rebar_x")
        rebar_y = get("rebar_y")

        # Handle combined rebar column
        if rebar_x and not rebar_y:
            rebar_x, rebar_y = _split_rebar_combined(rebar_x)

        remarks = get("remarks")
        # Strip leading digit+space artifacts from pdfplumber (e.g. "6 -" → "-")
        remarks = re.sub(r'^\d+\s+', '', remarks).strip()
        classification = _classify_from_remarks(remarks)

        try:
            item = FoundationItem(
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
            items.append(item)
            last_data_row_idx = row_idx  # track the last row that had real data
        except Exception as e:
            print(f"[TableExtract] Row parse error for type={type_val}: {e}")

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
                raw_data = tbl.extract()
                items, last_row_idx = _parse_table(raw_data)
                if len(items) >= _MIN_DATA_ROWS:
                    # Crop bbox to just the data section (header + data rows).
                    # Rows after the last data row are cross-section drawings — exclude them.
                    crop_bottom = tbl.bbox[3]  # full table bottom as fallback
                    if 0 <= last_row_idx < len(tbl.rows):
                        # Use the bottom of the last data row + small border margin
                        crop_bottom = tbl.rows[last_row_idx].bbox[3] + 4
                    data_bbox = (tbl.bbox[0], tbl.bbox[1], tbl.bbox[2], crop_bottom)
                    print(f"[TableExtract] Found {len(items)} rows on page {page_num}, "
                          f"data_bbox={data_bbox}")
                    return TableExtractionResult(
                        items=items,
                        page_num=page_num,
                        bbox=data_bbox,         # tight: header + data rows only
                        pdf_width=page.width,
                        pdf_height=page.height,
                    )
    except Exception as e:
        print(f"[TableExtract] Error: {e}")

    print("[TableExtract] No foundation schedule found via pdfplumber.")
    return _EMPTY_RESULT
