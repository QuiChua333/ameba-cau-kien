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


# Pit elevation from the floor-plan reference annotation. The 基礎伏図 labels each
# pit with its slab-top elevation, e.g.:
#   "消火水槽詳細図参照(底盤天端設計GL-1,000)"  → {"消火水槽": -1000}
#   "EV2ピット詳細図参照(底盤天端SGL-1,150)"   → {"EV2ピット": -1150}
#   "ES1ピット詳細図参照(底盤天端GL-1,250)"     → {"ES1ピット": -1250}
# 底盤天端 = top surface of the bottom slab = exactly the pit top_elevation, so this
# is authoritative and removes the vision guesswork (1,000 vs 1,200 vs 1,495).
_PIT_REF_RE = re.compile(
    r'([0-9A-Za-z①-⑳、,]*[一-龥ァ-ヶー]+)詳細図参照\s*[（(]\s*'
    r'底盤天端(?:設計)?S?GL[-－－]([0-9,，]+)',
    re.UNICODE,
)


def parse_pit_elevations(pdf_text: str) -> dict:
    """Extract {pit_type: top_elevation} from floor-plan '…詳細図参照(底盤天端…GL-XXX)'.

    Returns a dict of pit name → negative mm. Leading numeric noise from an
    adjacent annotation (e.g. "...GL-1,000消火水槽") is stripped from the name.
    """
    result: dict = {}
    for m in _PIT_REF_RE.finditer(pdf_text):
        # The capture may include preceding noise from an adjacent annotation
        # ("…F27FW1…消火水槽"). The real pit name is the trailing Japanese run plus
        # any latin/number prefix glued directly to it (e.g. "EV2" in "EV2ピット").
        m2 = re.search(r'([A-Za-z]{1,4}\d{0,2})?([一-龥ァ-ヶー]+)$', m.group(1))
        if not m2:
            continue
        name = (m2.group(1) or "") + m2.group(2)
        num = int(m.group(2).replace(',', '').replace('，', ''))
        result.setdefault(name, -num)
    if result:
        print(f"[PitText] floor-plan pit elevations: {result}")
    return result


# Pit-detail title, e.g. "消火水槽詳細図", "EVピット①詳細図", "ESCピット詳細図".
_PIT_DETAIL_TITLE_RE = re.compile(
    r'(消火水槽|EV[ピﾋ]ット[①-⑳0-9]*|ES[CＣ]?[ピﾋ]ット[①-⑳0-9]*'
    r'|[ピﾋ]ット[①-⑳0-9]*|側溝|水槽)\s*詳細図',
    re.UNICODE,
)


def parse_pit_slab_thickness(pdf) -> dict:
    """Read each pit's floor-slab thickness D from its 詳細図 cross-section.

    pdf: an ALREADY-OPEN pdfplumber document (reuse the pipeline's single open;
    reopening a dense CAD PDF costs tens of seconds).

    In a pit detail the left dimension chain runs ▽GL → slab top → slab bottom →
    leveling concrete (捨てコン). Read by RELATION rather than absolute magic ranges:
      • depth = the largest dim in the box = ▽GL → slab top (the top_elevation span).
      • D = the segment DIRECTLY BELOW the slab top = slab bottom − slab top. It is
        the nearest dim under `depth` that is SMALL RELATIVE TO depth (value < depth/2)
        — this skips other GL-anchored totals (e.g. a GL→improvement-bottom dim that
        is nearly as large as depth) yet still accepts a thin OR thick slab, because
        the cut-off scales with the pit's own depth instead of a fixed 100–400 mm.
    The dimension text is rotated 90°, so it reads reversed ("200" → "002").

    Returns {pit_name: D_mm}. Used only to FILL pits whose D the Gemini vision pass
    left null — the vision value wins when present.
    """
    result: dict = {}
    try:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if '詳細図' not in page_text:
                continue
            words = page.extract_words(extra_attrs=['upright'])
            for w in words:
                m = _PIT_DETAIL_TITLE_RE.search(w['text'])
                if not m or '参照' in w['text']:   # "…詳細図参照" is a floor-plan leader
                    continue
                name = m.group(1)
                tx, ty = w['x0'], w['top']
                # The detail box sits ABOVE its title; collect its rotated dims.
                dims = []
                for v in words:
                    if v.get('upright') or not re.search(r'[0-9]', v['text']):
                        continue
                    s = v['text'][::-1].replace(',', '')
                    if not re.fullmatch(r'[0-9]+', s):
                        continue
                    xc = (v['x0'] + v['x1']) / 2
                    if abs(xc - tx) < 260 and (ty - 270) < v['top'] < (ty + 5):
                        dims.append((int(s), (v['top'] + v['bottom']) / 2))
                if not dims:
                    continue
                depth = max(dims, key=lambda d: d[0])           # ▽GL → slab top
                # D = the segment right below the slab top (slab bottom − slab top),
                # small relative to depth so other GL-anchored totals are skipped.
                below = [d for d in dims
                         if d[1] > depth[1] + 2 and d[0] < depth[0] / 2]
                if below:
                    below.sort(key=lambda d: d[1])              # closest below the depth
                    result.setdefault(name, below[0][0])
    except Exception as e:
        print(f"[PitSlabD] failed: {e}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Cross-section 天端 elevations
# ──────────────────────────────────────────────────────────────────────────────
#
# Many foundations have NO floor-plan elevation annotation; their 基礎天端高さ is
# shown only inside the 基礎断面 cross-section as a dimension chain ▽GL → foundation
# top → base. We read it directly instead of falling back to the project default
# (which would be wrong for any non-default foundation, e.g. a 1,000-deep one).
#
# The 天端 = the chain segments between ▽GL and the base region (cover 30/70/100,
# はかま 250): a labelled total if one is drawn (a value V = a + b of two segments,
# e.g. 1,000 = 675+325), else the sum of the segments (e.g. 200+300 = 500).
# Section text is rotated 90°, so pdfplumber returns it REVERSED ("[1,000]" →
# "]000,1["); we reverse each token. A bracket note
# 「[ ]内の数値は、F1Aの数値とする」 gives that owner the SAME-style ([ ] vs ( ))
# chain value; the other types in the title take the plain value.

_BRACKET_NOTE_RE = re.compile(
    r'([\[\]（）()])\s*内の数値は[\s、,]*(F\d+[A-Za-z0-9]*)\s*の数値とする',
    re.UNICODE,
)
_SECTION_TITLE_RE = re.compile(r'(F\d+[A-Za-z0-9,、，\s]*?)\s*基礎断面', re.UNICODE)
_CODE_SPLIT_RE = re.compile(r'[,、，\s]+')


def _bracket_style(ch: str) -> str:
    return 'paren' if ch in '（）()' else 'square'


def _reversed_dim_values(reversed_text: str) -> list:
    """From a reversed rotated dim word, return [(value:int, style:str|None), …].

    style is 'square' for [..], 'paren' for (..), None for a plain number.
    """
    res = []
    for m in re.finditer(r'(\[|\()\s*([0-9,]+)\s*(\]|\))', reversed_text):
        try:
            res.append((int(m.group(2).replace(',', '')), _bracket_style(m.group(1))))
        except ValueError:
            pass
    if not res:
        s = reversed_text.replace(',', '')
        if re.fullmatch(r'[0-9]+', s):
            res.append((int(s), None))
    return res


# Cover/blinding/はかま dimension values that mark the start of the base region —
# everything ABOVE them in the chain is the 天端 depth.
_BASE_DIM_VALUES = {30, 70, 100, 250}


def _section_tengan(region: list, sum_if_no_total: bool) -> Optional[int]:
    """region: [(value, y), …] same-style dims ABOVE the base. Return 天端 mm or None.

    The 天端 is the GL-to-foundation-top distance. How it is read from the chain:
      • A labelled total — a value V equal to the sum of two other segments stacked on
        it (e.g. 1,000 = 675+325, 650 = 200+450) — IS the whole 天端 span → use V.
      • No labelled total:
          - PLAIN segments (sum_if_no_total=False): take only the TOP segment from ▽GL.
            F1C's chain is 200 then 300 → 天端 = 200; the 300 is the はかま/body BELOW
            the foundation top, not part of 天端. Summing would over-count (→500).
          - BRACKET-NOTE segments (sum_if_no_total=True): the note's bracketed numbers
            ARE the 天端 sub-segments, so SUM them. F6A has only "(325) (325)" with no
            total drawn → 天端 = 325+325 = 650 (= the plain siblings' 650 total).
    Values outside a plausible range are dropped as parse garbage ("200700", etc.).
    """
    items = [(v, y) for v, y in region if 30 <= v <= 1600]
    if not items:
        return None
    vals = [v for v, _y in items]
    totals = [V for V in vals
              if any(a + b == V and a != V and b != V for a in vals for b in vals)]
    if totals:
        result = max(totals)
    elif sum_if_no_total:
        result = sum(vals)                            # bracket note: sum the sub-segments
    else:
        result = min(items, key=lambda t: t[1])[0]    # plain: topmost segment from ▽GL
    return result if 30 <= result <= 2000 else None


def parse_section_elevations(pdf, explicit: dict) -> dict:
    """Per-type 基礎天端高さ read deterministically from 基礎断面 cross-sections.

    pdf: an ALREADY-OPEN pdfplumber document. Pass the open doc (not bytes) so this
    reuses pages the pipeline has already parsed — reopening + re-parsing a dense CAD
    PDF here costs tens of seconds (the page.chars/layout pass), doubling latency.

    explicit: the {CODE: negative_mm} floor-plan map — kept authoritative; this only
    FILLS types it does not already cover (never overrides a 伏図 annotation).

    Each "<codes> 基礎断面" has a left dimension chain ▽GL → foundation top → base.
    The 天端 (see _section_tengan) is read from the chain segments above the base
    region (cover 30/70/100, はかま 250). Every non-note type in the title gets the
    plain value; a bracket-note owner gets the same-style ([ ] vs ( )) value.
    Returns {CODE: negative_mm}.
    """
    result: dict = {}
    try:
        for page in pdf.pages:
            # Pages are already warm (parsed earlier), so extract_text is cheap here.
            page_text = page.extract_text() or ""
            if '基礎断面' not in page_text:
                continue
            owner_style = {
                m.group(2).upper(): _bracket_style(m.group(1))
                for m in _BRACKET_NOTE_RE.finditer(page_text)
            }

            words = page.extract_words(extra_attrs=['upright'])

            # Section titles: "<codes> 基礎断面"
            sections = []  # (codes:list[str], title_x0, title_top)
            for w in words:
                if '基礎断面' not in w['text']:
                    continue
                m = _SECTION_TITLE_RE.search(w['text'])
                codes_txt = m.group(1) if m else None
                if not codes_txt:  # code list is a separate word to the left
                    left = [v for v in words
                            if abs(v['top'] - w['top']) < 4 and v['x1'] <= w['x0'] + 2
                            and re.match(r'^F\d', v['text'])]
                    if left:
                        codes_txt = max(left, key=lambda v: v['x0'])['text']
                if codes_txt:
                    codes = [c.upper() for c in _CODE_SPLIT_RE.split(codes_txt)
                             if re.match(r'^F\d', c)]
                    if codes:
                        sections.append((codes, w['x0'], w['top']))

            for codes, tx, ty in sections:
                # The left dimension chain sits below-and-left of the section title.
                chain = []  # (value, style, y_centroid)
                for w in words:
                    xc = (w['x0'] + w['x1']) / 2
                    if not (tx - 160 <= xc <= tx - 50) or not (ty < w['top'] < ty + 160):
                        continue
                    if not w.get('upright') and re.search(r'[0-9]', w['text']):
                        for val, style in _reversed_dim_values(w['text'][::-1]):
                            chain.append((val, style, (w['top'] + w['bottom']) / 2))
                if not chain:
                    continue

                # Base region begins at the first cover/はかま plain dim; the 天端
                # depth is the chain above it.
                base_ys = [y for v, s, y in chain
                           if s is None and v in _BASE_DIM_VALUES]
                base_y = min(base_ys) if base_ys else float('inf')

                def _region(style):
                    return [(v, y) for v, s, y in chain if s == style and y < base_y]

                plain_t = _section_tengan(_region(None), sum_if_no_total=False)
                if plain_t is not None:
                    # Emit for ALL non-note codes (even those with a 伏図 value) so the
                    # caller can compare and flag floor-plan vs section conflicts. The
                    # caller keeps the 伏図 value as authoritative (setdefault).
                    for c in codes:
                        if c not in owner_style and c not in result:
                            result[c] = -plain_t
                            print(f"[SectionElev] [{c}] {codes} 天端=GL-{plain_t}")

                for owner, style in owner_style.items():
                    if owner not in codes:
                        continue
                    bt = _section_tengan(_region(style), sum_if_no_total=True)
                    if bt is not None:
                        result[owner] = -bt
                        print(f"[SectionElev] [{owner}] {codes} {style} 天端=GL-{bt}")
    except Exception as e:
        print(f"[SectionElev] failed: {e}")
    return result


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
        # Drop a rebar spec merged into D ("1000 18-D16" → "1000") and the ▽ leak.
        d = re.sub(r'\s*\d+\s*-\s*D\d+(?:@[\d,]+)?.*$', '', d, flags=re.DOTALL).strip()
        if d and d != str(item.dimensions.D):
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
      1. Explicit value — floor-plan annotation, cross-section text, OR a 断面
         dimension read by parse_section_elevations (merged into elev_data.explicit).
      2. Project-wide default (特記無き…GL-XXX) — the drawing's own rule for any
         foundation not otherwise specified.
      3. Keep existing value (no override) only when neither exists.

    Order matters: section dimensions are merged into `explicit` BEFORE this runs, so
    a deep foundation read from its 断面 (e.g. F3D = GL-1,000) wins over the default,
    while genuinely-unspecified shallow foundations still get the default (≈ their
    real near-GL top) — every foundation ends up with a value, none left blank.

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

        # Effective elevation per part: explicit (incl. 断面) > project default > keep.
        part_elevs: dict = {}
        for p in parts:
            if p in elev_data.explicit:
                part_elevs[p] = elev_data.explicit[p]
            elif elev_data.default is not None:
                part_elevs[p] = elev_data.default
            else:
                part_elevs[p] = None

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
