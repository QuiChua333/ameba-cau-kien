"""
excel_service.py — Fill the 掘削深度_Templete sheet in the Excel template
with extracted foundation/beam data and return the file as bytes.
"""
import io
import re
from pathlib import Path
from typing import List

import openpyxl

from models import FoundationItem, PitHoleItem

TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "計算書(施工) - DD.xlsx"

# Mapping: Gemini classification → Excel "Loại Móng" string
CLASSIFICATION_MAP = {
    "DD":    "TNF-DD",
    "D":     "TNF-D",
    "TNF":   "TNF",
    "FW/FG": "FW/FG",
}

# The data table starts at this row (header is row 9)
DATA_START_ROW = 10


def _parse_d(d_value: str) -> tuple[float | None, float | None]:
    """
    Parse D dimension string into (d_main, d2).

    Supported formats:
      "700~300"     → (700, 300)
      "900(~250)"   → (900, 250)   ← parenthetical format from PDF/pdfplumber
      "900（~250）"  → (900, 250)   ← full-width parentheses variant
      "400"         → (400, None)
      "0"           → (0,   None)
    """
    def _to_float(t: str) -> float:
        return float(t.strip().replace(',', '').replace('，', ''))

    s = str(d_value).strip()

    # Format: "900(~250)" or "900（~250）" — parenthetical range
    m = re.search(r'^([\d,，]+)\s*[（(]~?\s*([\d,，]+)[)）]', s)
    if m:
        try:
            return _to_float(m.group(1)), _to_float(m.group(2))
        except ValueError:
            pass

    # Format: "700~300" — tilde-separated range
    if "~" in s:
        parts = s.split("~", 1)
        try:
            return _to_float(parts[0]), _to_float(parts[1])
        except ValueError:
            pass

    # Plain number
    try:
        return _to_float(s), None
    except ValueError:
        return None, None


def fill_excel(
    foundation_list: List[FoundationItem],
    pit_list: List[PitHoleItem] | None = None,
) -> bytes:
    """
    Load the Excel template, fill the 掘削深度_Templete sheet with
    foundation, beam data, and readable pit holes, then return as bytes.
    """
    wb = openpyxl.load_workbook(str(TEMPLATE_PATH))
    ws = wb["掘削深度_Templete"]

    # Filter: only foundations and beams (not empty rows)
    items = [item for item in foundation_list if item.type]

    for i, item in enumerate(items):
        row = DATA_START_ROW + i

        # A — Zone (default 1)
        ws.cell(row=row, column=1).value = 1

        # B — STT (sequential number)
        ws.cell(row=row, column=2).value = i + 1

        # C — Tên Móng (foundation/beam name)
        ws.cell(row=row, column=3).value = item.type

        # D — Loại Móng (mapped classification)
        loai_mong = CLASSIFICATION_MAP.get(item.classification, item.classification)
        ws.cell(row=row, column=4).value = loai_mong

        # F — Cao độ mặt trên cấu kiện H1 (top_elevation, mm)
        if item.top_elevation is not None:
            ws.cell(row=row, column=6).value = item.top_elevation

        # G — H2 (for DD type, same as H1 unless we have specific data)
        if item.top_elevation is not None:
            ws.cell(row=row, column=7).value = item.top_elevation

        # H — Chiều cao cấu kiện D (main height)
        # I — Chiều cao D2 Daike (only for D / DD)
        d_main, d2 = _parse_d(item.dimensions.D)
        if d_main is not None and d_main > 0:
            ws.cell(row=row, column=8).value = d_main   # H
        if d2 is not None and item.classification in ("D", "DD"):
            ws.cell(row=row, column=9).value = d2        # I

    # ── Readable pit holes (appended after foundation rows) ──────────────────
    readable_pits = [p for p in (pit_list or []) if p.readable and p.type]
    pit_offset = len(items)
    for j, pit in enumerate(readable_pits):
        row = DATA_START_ROW + pit_offset + j

        ws.cell(row=row, column=1).value = 1                      # A — Zone
        ws.cell(row=row, column=2).value = pit_offset + j + 1     # B — STT
        ws.cell(row=row, column=3).value = pit.type               # C — Name
        ws.cell(row=row, column=4).value = "ピット"               # D — Type

        if pit.top_elevation is not None:
            ws.cell(row=row, column=6).value = pit.top_elevation  # F — H1
            ws.cell(row=row, column=7).value = pit.top_elevation  # G — H2

        if pit.D is not None and pit.D > 0:
            ws.cell(row=row, column=8).value = pit.D              # H — D

    # Save to bytes buffer
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()
