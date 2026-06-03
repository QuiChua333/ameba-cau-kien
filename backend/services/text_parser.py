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
# Matches: F1A(設計GL-450)  F3(設計GL-1,250)  F8A（設計GL-800）  F1(GL-1,000)  F2B(GL-1,250)
_FLOOR_PLAN_RE = re.compile(
    r'(F\d+[A-Z0-9]*)\s*[（(](?:設計)?GL[-－]([0-9,，]+)[）)]',
    re.UNICODE,
)

# Pattern 2 — Cross-section per-type annotations (dimension labels inside cross-section drawings)
# Matches: (F5,F6:1,100)  (F8A:800)  (F5、F6：1,100)  （F5,F6：1,100）
_CROSS_SECTION_RE = re.compile(
    r'[（(]\s*(F\d+[A-Za-z0-9]*(?:[,、，\s]+F\d+[A-Za-z0-9]*)*)\s*[:：]\s*([0-9,，]+)\s*[）)]',
    re.UNICODE,
)

# Pattern 3 — Project-wide default
# Matches: 特記無き基礎天端高さは、設計GL-250とする
#           特記無き基礎天端高さは、GL−200とする  (no 設計, U+2212 minus)
_DEFAULT_RE = re.compile(
    r'特記無き基礎天端高さは[、,，\s]*(?:設計)?GL[-－−](\d+)',
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
