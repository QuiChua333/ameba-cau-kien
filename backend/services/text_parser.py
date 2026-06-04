"""text_parser.py — Extract elevation data from the PDF text layer (no AI required).

Three annotation patterns cover the common formats in Japanese CAD construction drawings:
  1. Floor-plan per-type   F1A(設計GL-450)  → explicit per-type elevation
  2. Cross-section per-type (F5,F6:1,100)  → elevation inside combined cross-section title
  3. Project-wide default  特記無き基礎天端高さは、設計GL-250とする

Priority when merging: floor-plan > cross-section > project default > Gemini value.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Compiled patterns
# ──────────────────────────────────────────────────────────────────────────────

# Pattern 1 — Floor plan per-type annotations (highest priority)
# Matches ▽*GL variants: GL, SGL, 設計GL, 設計SGL  ((?:設計)?S?GL)
# Examples: F1A(設計GL-450)  F1(GL-1,000)  F15A(SGL-1,250)
_FLOOR_PLAN_RE = re.compile(
    r'(F\d+[A-Z0-9]*)\s*[（(](?:設計)?S?GL[-－]([0-9,，]+)[）)]',
    re.UNICODE,
)

# Pattern 2 — Cross-section per-type annotations (dimension labels inside cross-section drawings)
# Matches: (F5,F6:1,100)  (F8A:800)  (F5、F6：1,100)  （F5,F6：1,100）
_CROSS_SECTION_RE = re.compile(
    r'[（(]\s*(F\d+[A-Za-z0-9]*(?:[,、，\s]+F\d+[A-Za-z0-9]*)*)\s*[:：]\s*([0-9,，]+)\s*[）)]',
    re.UNICODE,
)

# Pattern 3 — Project-wide default
# Matches ▽*GL variants: GL, SGL, 設計GL, 設計SGL  ((?:設計)?S?GL)
# Examples: 特記無き基礎天端高さは、設計GL-250とする
#           特記無き基礎天端高さは、GL−200とする
#           特記無き基礎天端高さは、SGL-465とする
_DEFAULT_RE = re.compile(
    r'特記無き基礎天端高さは[、,，\s]*(?:設計)?S?GL[-－−](\d+)',
    re.UNICODE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ElevationData:
    """Parsed elevation information from the PDF text layer."""
    explicit: dict = field(default_factory=dict)   # {"F1A": -450, "F3": -1250}
    default: Optional[int] = None                  # -250 or None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def parse_elevations(pdf_text: str) -> ElevationData:
    """Extract all elevation data from the PDF text layer.

    Floor-plan annotations take precedence over cross-section annotations for
    the same type (floor plan is always the authoritative source).
    Returns an ElevationData instance (explicit map + optional project default).
    """
    data = ElevationData()

    # Pattern 1 — floor plan annotations (highest priority, process first)
    for m in _FLOOR_PLAN_RE.finditer(pdf_text):
        code = m.group(1).strip().upper()
        num = int(m.group(2).replace(',', '').replace('，', ''))
        if code not in data.explicit:
            data.explicit[code] = -num

    # Pattern 2 — cross-section per-type annotations
    for m in _CROSS_SECTION_RE.finditer(pdf_text):
        types_raw = m.group(1)
        num_str = m.group(2).replace(',', '').replace('，', '')
        try:
            value = -int(num_str)
        except ValueError:
            continue
        for part in re.split(r'[,、，\s]+', types_raw):
            code = part.strip().upper()
            # Only accept F-type codes; skip if already set by floor-plan annotation
            if code and re.match(r'^F\d+', code) and code not in data.explicit:
                data.explicit[code] = value

    # Pattern 3 — project default
    dm = _DEFAULT_RE.search(pdf_text)
    if dm:
        data.default = -int(dm.group(1))

    print(f"[TextParse] Explicit ({len(data.explicit)}): {data.explicit}")
    print(f"[TextParse] Default: {data.default}")
    return data


# Combined-type splitter — a single F-code token (F11, F23A, F132 …)
_FCODE_RE = re.compile(r'^F\d+[A-Za-z0-9]*$')


def split_combined_types(foundation_list: list) -> list:
    """Expand combined foundation rows into one item per type.

    A combined row like "F11, F23" / "F132、F21" lists several foundation codes
    that share one table row (and often one drawing). The UI table and the Excel
    export expect one row PER foundation, so split such rows into individual
    FoundationItems, copying ALL fields (dimensions, rebar, remarks,
    classification, region, image, top_elevation) verbatim to each part.

    Defensive rules:
      - FW/FG beams are never split.
      - A row is only split when EVERY comma/space-separated part looks like an
        F-code; otherwise it is left untouched (avoids mangling odd type strings).
      - Exact duplicate F-types that splitting may surface are de-duplicated
        (keep first occurrence), so the table never shows the same code twice.
    """
    result = []
    for item in foundation_list:
        if item.classification == "FW/FG":
            result.append(item)
            continue

        parts = [p.strip() for p in re.split(r'[,、，\s]+', item.type) if p.strip()]
        if len(parts) <= 1 or not all(_FCODE_RE.match(p.upper()) for p in parts):
            result.append(item)
            continue

        print(f"[Split] '{item.type}' → {parts}")
        for p in parts:
            new_item = item.model_copy(deep=True)
            new_item.type = p
            result.append(new_item)

    # Drop exact duplicate F-types (keep first); leave beams untouched.
    seen: set = set()
    deduped = []
    for it in result:
        if it.classification == "FW/FG":
            deduped.append(it)
            continue
        key = it.type.strip().upper().replace(" ", "").replace("　", "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped


# Final-pass field sanitiser
_REBAR_SPEC_RE = re.compile(r'\d+\s*-\s*D\d+(?:@[\d,]+)?')

# Drawing-section captions/annotations that leak into 備考 when the table bbox
# overlaps the cross-section drawings below it. The real remarks (B-formulas,
# "-", parenthetical notes) always precede these, so truncate at the first hit.
_DRAWING_NOISE_RE = re.compile(
    r'(基礎断面|基礎平面|断面図|柱廻り|柱型リスト|はかま筋|補強筋|スタイロ|'
    r'地盤改良|ベース筋|偏心方向|つなぎ筋|捨て[ｺコ]|砕石|埋戻|開口補強|立上げ)'
)


def _strip_drawing_text(remarks: str) -> str:
    """Truncate 備考 at the first drawing-caption/annotation keyword."""
    if not remarks:
        return remarks
    m = _DRAWING_NOISE_RE.search(remarks)
    return remarks[:m.start()].strip() if m else remarks


def sanitize_fields(foundation_list: list) -> list:
    """Final cleanup of D, rebar and remarks across all sources (plumber + Gemini).

    - D: drop any ▽GL/▽SGL elevation annotation that leaked into the cell, e.g.
      "1050~300 ▽SGL 564" → "1050~300". D never contains a ▽ marker.
    - remarks: truncate drawing captions/annotations (基礎断面, 柱廻り, 補強筋, …)
      that bled in from the cross-section drawings below the table, e.g.
      "… B1x x B1y = 4,400 x 4,400 基礎断面 柱廻り(スタイロ t=20) …" → "… 4,400".
    - rebar_x/rebar_y (non-beam only): keep only a valid rebar spec (e.g.
      "18-D16"); strip drawing captions / foundation-code lists ("F11,F12,…")
      that bled into the column on wide detail rows. Beams (FW/FG) keep their
      rebar verbatim (they use other formats like 上端筋/下端筋 specs).
    """
    for item in foundation_list:
        d = re.sub(r'\s*▽.*$', '', str(item.dimensions.D), flags=re.DOTALL).strip()
        if d and d != item.dimensions.D:
            item.dimensions.D = d

        cleaned_remarks = _strip_drawing_text(item.remarks or "")
        if cleaned_remarks != (item.remarks or ""):
            item.remarks = cleaned_remarks

        if item.classification == "FW/FG":
            continue
        for attr in ("rebar_x", "rebar_y"):
            val = getattr(item, attr) or ""
            m = _REBAR_SPEC_RE.search(val)
            cleaned = m.group(0).replace(" ", "") if m else ""
            if cleaned != val:
                setattr(item, attr, cleaned)
    return foundation_list


def resolve_elevations_for_list(foundation_list: list, elev_data: ElevationData) -> list:
    """Apply text-layer elevations to foundation items, splitting combined rows when needed.

    Priority:
      1. Explicit annotation (floor-plan or cross-section)
      2. Project-wide default
      3. Keep existing Gemini value (no override)

    FW/FG beams are skipped — they never have floor-plan annotations.
    Combined rows like "F4, F4A" are SPLIT when parts resolve to different elevations.
    Returns a (possibly expanded) list.
    """
    result = []
    for item in foundation_list:
        if item.classification == "FW/FG":
            result.append(item)
            continue

        parts = [p.strip().upper() for p in re.split(r'[,、，\s]+', item.type) if p.strip()]

        # Compute effective elevation per part
        part_elevs: dict = {}
        for p in parts:
            if p in elev_data.explicit:
                part_elevs[p] = elev_data.explicit[p]
            elif elev_data.default is not None:
                part_elevs[p] = elev_data.default
            else:
                part_elevs[p] = None  # no text-layer override; keep Gemini value

        known_vals = [v for v in part_elevs.values() if v is not None]

        if not known_vals:
            # No text-layer info for any part — keep Gemini value
            result.append(item)
            continue

        distinct = set(known_vals)
        all_resolved = all(v is not None for v in part_elevs.values())

        if len(parts) <= 1 or (len(distinct) == 1 and all_resolved):
            # Single part, or every part resolves to the same value
            new_val = next(iter(distinct))
            src = "explicit" if (parts and parts[0] in elev_data.explicit) else "default"
            if item.top_elevation != new_val:
                print(f"[TextParse] [{item.type}] {item.top_elevation} → {new_val}  ({src})")
            item.top_elevation = new_val
            result.append(item)
        else:
            # Parts resolve to different elevations (or some have no override) → SPLIT
            print(f"[TextParse] [{item.type}] Split — elevations differ: {part_elevs}")
            for p in parts:
                new_item = item.model_copy(deep=True)
                new_item.type = p
                elev = part_elevs[p]
                if elev is not None and new_item.top_elevation != elev:
                    src = "explicit" if p in elev_data.explicit else "default"
                    print(f"[TextParse]   [{p}] {new_item.top_elevation} → {elev}  ({src})")
                    new_item.top_elevation = elev
                result.append(new_item)

    return result
