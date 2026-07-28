"""textlayer_phase1.py — Phase 1a text-layer extraction and page-line caching.

Bundles the Phase 1a pdfplumber scans that run before/alongside the Gemini
first pass so the PDF is opened and parsed ONCE. The shared document/page
objects provide the char/curve caches for table + oval detection, while the
same extracted word lists are reused to build line caches and deterministic
drawing/caption locators.

`extract_table_and_ovals` is a top-level function so it can be dispatched to a
ProcessPoolExecutor: the work is CPU-bound pure Python, and running it in a
separate process sidesteps the GIL, letting it overlap with the Gemini
network wait. Keep this module's imports light (no google-genai) so the spawned
child starts quickly.
"""

import io
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False

from services.table_extractor import (
    extract_foundation_table_from_open,
    TableExtractionResult,
)
from services.text_parser import (
    parse_elevations,
    parse_section_elevations,
    parse_pit_slab_thickness,
)
from models import ItemRegion
from services.drawing_locator import (
    build_page_scan,
    find_beam_section_captions_from_pages,
    find_foundation_drawing_regions_from_pages,
    find_oval_gl_markers_from_cached_words,
)


@dataclass
class TextLayerScanResult:
    """Serializable text-layer scan bundle reused after Gemini returns."""
    table_result: TableExtractionResult = field(default_factory=TableExtractionResult)
    ovals: List[dict] = field(default_factory=list)
    foundation_regions: Dict[str, ItemRegion] = field(default_factory=dict)
    beam_captions: Dict[int, Dict[str, dict]] = field(default_factory=dict)
    page_scans: List[dict] = field(default_factory=list)
    # {CODE: signed_mm} 天端 elevations read from 基礎断面 cross-sections — negative
    # below ▽GL, positive when the foundation top stands above it (plain
    # dimensions and "[ ]内の数値は" bracket notes). Computed while the PDF is open once.
    section_elevations: Dict[str, int] = field(default_factory=dict)
    # {pit_name: D_mm} floor-slab thickness read from each pit's 詳細図. Fills pits
    # whose D the Gemini vision pass leaves null (vision value wins when present).
    pit_d_map: Dict[str, int] = field(default_factory=dict)


def scan_text_layer(pdf_bytes: bytes) -> TextLayerScanResult:
    """Run all Phase 1a text-layer scans while the PDF is open once.

    Returns page caches (pre-built lines + page geometry) so downstream locator
    steps can reuse them without reopening the PDF in the main process.
    """
    result = TextLayerScanResult()
    if not _PDFPLUMBER_AVAILABLE:
        print("[Phase1a] pdfplumber not installed — skipping text-layer extraction.")
        return result
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_scans_full = [build_page_scan(page_idx, page) for page_idx, page in enumerate(pdf.pages)]
            result.table_result = extract_foundation_table_from_open(pdf)
            result.ovals = find_oval_gl_markers_from_cached_words(
                pdf,
                [page_scan["words"] for page_scan in page_scans_full],
            )
            result.foundation_regions = find_foundation_drawing_regions_from_pages(page_scans_full)
            result.beam_captions = find_beam_section_captions_from_pages(page_scans_full)

            # Cross-section 天端 elevations. Reuse THIS open document (pages already
            # parsed above) — reopening to read word geometry would re-parse the whole
            # dense PDF and roughly double latency. Gate on 基礎断面 so PDFs without
            # cross-sections pay nothing beyond the cheap (warm) text scan.
            try:
                full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                if '基礎断面' in full_text:
                    elev = parse_elevations(full_text)
                    result.section_elevations = parse_section_elevations(
                        pdf, elev.explicit
                    )
                if '詳細図' in full_text:
                    result.pit_d_map = parse_pit_slab_thickness(pdf)
            except Exception as e:
                print(f"[Phase1a] section-elevation scan failed: {e}")
            result.page_scans = [
                {
                    "page_idx": page_scan["page_idx"],
                    "page_num": page_scan["page_num"],
                    "width": page_scan["width"],
                    "height": page_scan["height"],
                    "lines": page_scan["lines"],
                }
                for page_scan in page_scans_full
            ]
    except Exception as e:
        print(f"[Phase1a] scan_text_layer failed: {e}")
    return result


def extract_table_and_ovals(pdf_bytes: bytes) -> Tuple[TableExtractionResult, List[dict]]:
    """Run foundation-table extraction + oval GL detection on one open document.

    Returns (TableExtractionResult, ovals). Never raises — on any failure it
    returns whatever was gathered so the caller can fall back to Gemini's
    vision-based extraction.
    """
    result = scan_text_layer(pdf_bytes)
    return result.table_result, result.ovals
