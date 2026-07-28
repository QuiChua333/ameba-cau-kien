import google.genai as genai
from google.genai import types
from pydantic import BaseModel as _PydanticBase
from typing import List, Optional
from PIL import Image, ImageDraw
import json
import io
import base64
import time
import random
import re
import threading as _threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import config

# ── Singleton Gemini client (shared across all requests) ──────────────────────
_gemini_client: Optional[genai.Client] = None
_gemini_client_lock = _threading.Lock()

def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        with _gemini_client_lock:
            if _gemini_client is None:
                _gemini_client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _gemini_client

# ── Concurrency limiter: max concurrent Gemini API calls across all requests ──
_GEMINI_MAX_CONCURRENT = 5
_gemini_sem = _threading.Semaphore(_GEMINI_MAX_CONCURRENT)
from models import (
    ExtractionResponse,
    FoundationItem,
    Dimensions,
    ItemRegion,
    ProjectInfo,
    PitHoleItem,
)

# ─────────────────────────────────────────────────────────────────────────────
# Text layer and table extraction (delegated to dedicated modules)
# ─────────────────────────────────────────────────────────────────────────────

from services.text_parser import (
    parse_elevations,
    resolve_elevations_for_list,
    split_combined_types,
    sanitize_fields,
    parse_pit_elevations,
)
from services.table_extractor import TableExtractionResult
from services.textlayer_phase1 import TextLayerScanResult, scan_text_layer
from services.drawing_locator import (
    find_foundation_drawing_regions,
    find_beam_section_captions,
    find_beam_labels,
    find_beam_labels_from_pages,
    find_pit_drawing_regions,
    derive_fw_panel_region,
    derive_fw_panel_region_from_caption,
    derive_fg_column_region,
)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

PROMPT = """
Analyze the provided images of Japanese construction drawings carefully.

STEP 1: FIND THE FOUNDATION DATA — TWO SHEET LAYOUTS EXIST.

⚠️ BOTH layouts are in active use. FIRST decide which one this drawing set uses,
then follow the matching branch in STEP 2 and STEP 3. The test is simple:
  → Is there ONE page-wide 基礎リスト table with a 備考 (Remarks) column?
      YES → LAYOUT A.   NO → LAYOUT B.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYOUT A — ONE PAGE-WIDE FOUNDATION SCHEDULE (the classic sheet)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A single table titled "基礎リスト" / "基礎符号" / "Foundation List" lists EVERY
foundation, one row each, with a 備考 (Remarks) column on the right:

  ┌────────┬────────┬────────┬──────────┬─────────────────┬──────────────────────┐
  │基礎符号│Lx(mm)  │Ly(mm)  │ D(mm)    │    ベース筋      │        備考          │
  │        │        │        │          │   ←    │    ↑   │                      │
  ├────────┼────────┼────────┼──────────┼────────┼────────┼──────────────────────┤
  │  F1    │ 2,400  │ 2,400  │ 900~350  │ 13-D13 │ 13-D13 │ B0x x B0y = 700x700  │
  │        │        │        │          │        │        │ B1x x B1y = 1,800x…  │
  │  F1A   │ 2,400  │ 2,400  │ 900~450  │ 13-D13 │ 13-D13 │ Bx x By = 900 x 900  │
  │  F1B   │ 2,400  │ 2,400  │ 500      │ 13-D13 │ 13-D13 │          -           │
  └────────┴────────┴────────┴──────────┴────────┴────────┴──────────────────────┘

Ignore other schedules (Column List, Beam List) and floor plans.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYOUT B — PER-FOUNDATION MINI TABLES (the newer sheet)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
There is NO page-wide schedule. Each foundation has its OWN block on the
基礎リスト sheet: a title, a one-row mini table, and its cross-section drawing.
The mini table has the SAME columns as Layout A **except there is NO 備考 column**:

    F1A 基礎平面              F1A 基礎断面          ← per-type titles
    ┌────────┬────────┬────────┬──────────┬─────────────────┐
    │基礎符号│Lx(mm)  │Ly(mm)  │ D(mm)    │    ベース筋      │   ← no 備考!
    │        │        │        │          │   ←    │   ↑    │
    ├────────┼────────┼────────┼──────────┼────────┼────────┤
    │  F1A   │ 4,600  │ 4,600  │1,100~250 │ 46-D16 │ 46-D16 │   ← ONE row only
    └────────┴────────┴────────┴──────────┴────────┴────────┘
                  [cross-section drawing]
                        1,600
                        3,600                ← bottom dimension stack
                        4,600                  (see STEP 3, LAYOUT B)

Several such blocks sit on one sheet, stacked vertically and side by side
(e.g. F1 | F3 on the top row, F1A | F3A below, F2 | F4 below that).

STEP 2: EXTRACT DATA.

▶ LAYOUT A — read the schedule ROW by ROW:
  - Focus ONLY on the "Foundation List" you found.
  - Do not copy values from previous rows.
  - Be very careful with dimensions (Lx, Ly) — they differ for each type.
  - Be very careful with Rebar counts (e.g., 11-D13 vs 13-D16).

▶ LAYOUT B — emit ONE FoundationItem per MINI TABLE:
  - Scan the WHOLE sheet and find EVERY mini table. Do not stop after the first —
    a sheet typically holds 4–8 of them, and there may be more than one such sheet.
  - Read each mini table's single data row: type, Lx, Ly, D, and the two ベース筋
    specs (← = rebar_x, ↑ = rebar_y).
  - D is often a taper range written "1,100～250" — keep BOTH numbers as "1100~250".
    A flat footing has a single D ("800").
  - Set remarks to "" (this layout has no 備考 column). Do NOT invent remarks.
  - Read each block's values ONLY from inside that block. The neighbouring block's
    mini table and drawing are a DIFFERENT foundation — never mix their numbers.

STEP 3: CLASSIFY FOUNDATION TYPE (CRITICAL).

The classification counts how many stepped tiers the footing has:
  TNF = a flat slab (no step) · D = one step · DD = two steps.
How you COUNT depends on the layout.

▶ LAYOUT A — count "B...x x B...y" formulas in the 備考 (Remarks) column:
  - Rule 1 (DD): count >= 2 (e.g. both "B0x x B0y = …" AND "B1x x B1y = …") → "DD".
  - Rule 2 (D):  count == 1 (e.g. only "Bx x By = …")                       → "D".
  - Rule 3 (TNF): count == 0 (empty, "-", or text without formulas)         → "TNF".
  ⚠️ When a 備考 column exists it is AUTHORITATIVE — use it, not the drawing.

▶ LAYOUT B — there is no 備考 column, so COUNT THE VALUES IN THE BOTTOM
  DIMENSION STACK beneath that foundation's own drawing:

  Under each cross-section (and under its 基礎平面 plan view) sits a stack of
  horizontal width dimensions, each on its own dimension line, growing wider
  downward. The BOTTOM (widest) one equals that foundation's Lx.

      1,600     ← innermost tier width      ┐
      3,600     ← middle tier width          │ 3 values → "DD"
      4,600     ← full footing width (= Lx)  ┘

  - 3 values  → "DD"    (two steps: e.g. 1,600 / 3,600 / 4,600)
  - 2 values  → "D"     (one step:  e.g. 1,600 / 4,600)
  - 1 value   → "TNF"   (flat slab: e.g. just 5,000)

  ✅ SELF-CHECK — the LAST (bottom, largest) value in the stack MUST equal the Lx
     you read from that block's mini table. If it does not, you are looking at the
     wrong stack (a neighbouring block's, or a rebar/grid dimension). Find the
     right one before counting.
  ✅ CROSS-CHECK with D: a tapered D ("1,100～250") means the footing is stepped →
     expect "D" or "DD". A single flat D ("800") usually means "TNF".
  ⛔ Do NOT count vertical dimensions, the left-side depth chain, rebar spacing
     numbers, or the 地盤改良 (ground improvement) extents.
  ⛔ Do NOT count the same stack twice: the 基礎平面 and 基礎断面 views of ONE
     foundation show the SAME stack. Count the values in one view, not both.

  WORKED EXAMPLES — LAYOUT B:
    F1A: mini table Lx=4,600 · stack 1,600 / 3,600 / 4,600 (bottom = 4,600 = Lx ✓)
         → 3 values → classification = "DD"  ✓
    F3A: mini table Lx=5,000 · stack 5,000 only (bottom = 5,000 = Lx ✓), D = 800 flat
         → 1 value → classification = "TNF"  ✓
    ⛔ WRONG: classifying F3A as "DD" because its neighbour F3 has three values.
    ⛔ WRONG: classifying every Layout-B foundation "TNF" because there is no 備考.

STEP 4: EXTRACT TOP ELEVATION FROM CROSS-SECTION DRAWINGS (CRITICAL NEW STEP).

For EVERY foundation type (F1, F2, ...) AND every beam (FW1, FW2, FG1, ...) found,
you MUST search for its corresponding cross-section drawing and extract the top_elevation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRE-STEP 4A: SCAN FOUNDATION FLOOR PLAN FOR EXPLICIT ANNOTATIONS (HIGHEST PRIORITY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The 基礎伏図 (Foundation Floor Plan) is THE MOST RELIABLE source for top_elevation.
ALWAYS scan it before reading any cross-section drawing.

━━━ THE ▽*GL PREFIX — READ THIS FIRST, IT APPLIES EVERYWHERE BELOW ━━━
Every firm brands its ground-line datum differently. "GL" is the only stable part;
whatever is glued in front of it is just a label and carries NO meaning for you:
     GL      設計GL      SGL      設計SGL      C棟GL      A棟GL      B棟GL
Pattern: ▽*GL — "*" is any short prefix (nothing, S, 設計, 設計S, or a building-wing
name such as C棟). TREAT THEM ALL AS THE SAME ▽GL DATUM, in floor-plan annotations,
in default notes, and on cross-section datum lines.
⚠️ But NEVER confuse ▽*GL with ▽*FL. "*FL" (e.g. ▽C棟FL, 設計FL) is the FINISHED
   FLOOR line, which sits ABOVE ▽*GL. Only ▽*GL is the elevation datum.

1) Find the floor plan note (any format is valid):
     "基礎符号(設計GL-***)の(　)内数値は、基礎天端高さを示す"
     "基礎符号(GL-***)の(　)内数値は、基礎天端高さを示す"
     "基礎符号(SGL-***)の(　)内数値は、基礎天端高さを示す"
     "基礎符号(C棟GL-***)の(　)内数値は、基礎天端高さを示す"
   This confirms that the "(*GL-XXX)" value next to each foundation symbol
   = the foundation SLAB TOP elevation (底盤天端高さ).

2) Collect ALL explicit per-foundation annotations from the floor plan.
   These appear as "(*GL-XXX)" immediately after the foundation code.
   Accept any GL prefix variant (GL, SGL, 設計GL, 設計SGL, C棟GL, …).
   Examples:
     "F1(GL-1,000)"       → F1   top_elevation = -1,000
     "F2B(GL-1,250)"      → F2B  top_elevation = -1,250
     "F2C(GL-1,500)"      → F2C  top_elevation = -1,500
     "F1A(設計GL-450)"    → F1A  top_elevation = -450
     "F3(設計GL-1,250)"   → F3   top_elevation = -1,250
     "F2A(設計GL-1,700)"  → F2A  top_elevation = -1,700
     "F15A(SGL-1,250)"    → F15A top_elevation = -1,250   ← SGL = 設計GL
     "F115(SGL-1,150)"    → F115 top_elevation = -1,150
     "F22B(SGL-1,250)"    → F22B top_elevation = -1,250
     "F1A(C棟GL-500)"     → F1A  top_elevation = -500     ← C棟GL = per-wing GL
     "F3A(C棟GL-1,100)"   → F3A  top_elevation = -1,100
   These values are AUTHORITATIVE. Use them directly.

   ⚠️ KEEP THE SIGN THAT IS WRITTEN. The annotation is usually "-" (the foundation top
   is buried below ▽GL), but a shallow foundation can stand ABOVE the ground line and
   is then annotated "+" — that value stays POSITIVE:
     "F1A(設計GL+60)"     → F1A  top_elevation = +60
     "F2(GL±0)"           → F2   top_elevation = 0
   Never flip a "+" to "-" just because most foundations are negative.

3) Find the DEFAULT elevation note (used ONLY as a last resort):
     Format A: "特記無き基礎天端高さは、設計GL-250とする" → project default = -250
     Format B: "特記無き基礎天端高さは、GL−200とする"    → project default = -200
     Format C: "特記無き基礎天端高さは、SGL-465とする"    → project default = -465
     Format D: "特記無き基礎天端高さは、設計GL+60とする"  → project default = +60
     Format E: "特記無き基礎天端高さは、C棟GL+650とする" → project default = +650
   (Any *GL prefix variant is valid — GL, SGL, 設計GL, 設計SGL, C棟GL; keep the sign)
   ⚠️ A POSITIVE default is real, not a typo: on some sites the foundation top sits
      ABOVE the ground line. Keep the "+" exactly as written.
   ⚠️ Apply this default ONLY to foundations that have NO explicit annotation on the
   floor plan AND whose cross-section drawing cannot be found or read.
   If a foundation has an explicit floor plan annotation, ALWAYS use that, never the default.

4) PRIORITY ORDER:
   ① Floor plan explicit annotation "(GL-XXX)" or "(設計GL-XXX)"  ← USE THIS IF PRESENT
   ② Cross-section drawing reading (STEP B below)                  ← FALLBACK ONLY
   ③ Set top_elevation = null (N/A)                               ← WHEN NEITHER ① NOR ② WORKS
   ⛔ Do NOT fill in the project default value when ① or ② should have worked but failed.
      If you cannot confidently read the value, use null — do not guess.

⛔ CRITICAL: DO NOT confuse FLOOR SLAB elevation markers with FOUNDATION top elevations.
   The floor plan contains oval or rectangular markers labeled "GL+100", "GL+45", "GL±0",
   "(GL+100)", "(GL±0)" etc. These indicate the HEIGHT OF THE FLOOR SLAB, NOT the
   foundation top elevation. They look like:
     (GL+100) ←── this is slab height: floor surface is 100mm above GL
     (GL+45)  ←── this is slab height: floor surface is 45mm above GL
     (GL±0)   ←── this is slab height: floor surface is at GL
   ⛔ NEVER use these oval/bubble GL markers as top_elevation for foundations.
   ✅ Foundation top elevations come ONLY from:
      - "(設計GL-XXX)" annotations tied to a foundation symbol code (F1A, F3, etc.)
      - "特記無き基礎天端高さは、GL−XXX" default note
      - Cross-section drawings (▽GL dimension chain reading)

⛔ CRITICAL ANTI-PATTERN: If the top_elevation you read from a cross-section matches
   the D value from the foundation table (e.g., D=700 → top_elevation=-700), this is
   ALMOST CERTAINLY WRONG. You extracted the footing body depth, not the soil gap.
   Go back and check the floor plan annotation or use the project default.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRE-STEP 4B: DEEP-PIT FOUNDATIONS — SPECIAL READING RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Some foundations are installed at the BOTTOM OF A DEEP PIT:
  EVピット① (EV shaft pit, e.g., 1,250mm deep)
  EVピット② (EV shaft pit, e.g., 1,650mm deep)
  ESCピット  (escalator pit, e.g., 1,100mm deep)
  消火水槽   (fire suppression tank, e.g., 1,700mm deep)

For these foundations the cross-section shows footing details WITHIN the pit:
  "30 D 70"  or  "250 D 30"  ← These are intra-pit clearances, NOT distances from ▽GL!

⛔ The "30" or "250" visible next to "D 70" in these drawings is the clearance INSIDE the
   excavated pit, NOT the soil gap from ▽GL. DO NOT use it as top_elevation.

✅ The correct top_elevation = -(pit depth from ▽GL), e.g.:
     EVピット① depth 1,250mm → top_elevation = -1,250
     ESCピット  depth 1,100mm → top_elevation = -1,100
     消火水槽   depth 1,700mm → top_elevation = -1,700

The pit depth label (e.g., "1,250 EVピット①") usually appears ON THE CROSS-SECTION
as a large dimension spanning from ▽GL to the pit floor level.
Use that large dimension number as the top_elevation (as a negative value).

WORKED EXAMPLES — deep pit foundations:
  F3 基礎断面:  pit label "EVピット① 1,250" → top_elevation = -1,250  ✓
  F2A 基礎断面: pit label "消火水槽 1,700"   → top_elevation = -1,700  ✓
  F2B 基礎断面: pit label "1,100"             → top_elevation = -1,100  ✓
  ⛔ WRONG: reading "30" from "30 D 70" inside the pit → top_elevation = -30  ✗
  ⛔ WRONG: reading D=700 as top_elevation → top_elevation = -700  ✗

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

--- HOW TO FIND THE CROSS-SECTION DRAWING ---
Scan all images for drawings labeled (usually placed as a title above or below the drawing):
  "F1 基礎断面", "F2 基礎断面", "FW1 断面", "FW1 基礎断面", "FG1 断面", etc.
The label format is: [Type] + "基礎断面" or "断面図" or "断面".
These are side-view (cross-section) drawings showing the foundation in the ground.

--- HOW TO READ THE TOP ELEVATION — STRICT ALGORITHM ---

⚠️ THE SINGLE MOST IMPORTANT RULE:
   top_elevation = the dimension in the SOIL/FILL ZONE (空隙部) between ▽GL and the first concrete surface.
   It is NEVER the dimension inside the concrete body.
   It is NEVER equal to D (the footing body depth).

ALGORITHM — follow exactly in this order:

STEP A0) ★ SHORTCUT — IS THERE A "▼基礎天端" LEVEL MARKER? (CHECK THIS FIRST)
   Newer cross-sections label the levels outright with small filled triangles:
       ▽C棟FL     ← finished floor line   (ABOVE the ground line)
       ▽C棟GL     ← THE GROUND-LINE DATUM (zero)
       ▼柱型天端  ← top of the COLUMN STUB   ⛔ NOT the foundation top
       ▼基礎天端  ← top of the FOUNDATION SLAB  ✅ THIS is what top_elevation means
   When "▼基礎天端" is present you do not have to reason about which dimension is
   which — the drawing states the level for you:
     a) SIGN: is ▼基礎天端 drawn BELOW ▽*GL or ABOVE it?
          BELOW → top_elevation is NEGATIVE (buried — the usual case)
          ABOVE → top_elevation is POSITIVE (the concrete stands proud of the ground)
     b) MAGNITUDE, when ▼基礎天端 is BELOW ▽*GL: read the dimension that spans
        exactly from the ▽*GL line down to the ▼基礎天端 line. If nested chains
        both cover that gap, take the OUTER (leftmost) total, not its subdivisions.
          e.g. outer "500" with inner "350" + "150" (350+150=500) → use 500 → -500
     c) MAGNITUDE, when ▼基礎天端 is ABOVE ▽*GL: these sheets do NOT dimension the
        ▽GL→天端 gap directly — they dimension both levels from ▽*FL instead. So
        SUBTRACT:  top_elevation = (▽FL→▽GL) − (▽FL→▼基礎天端)
          e.g. ▽FL→▽GL = 1,000 and ▽FL→▼基礎天端 = 350 → top_elevation = +650
        If you cannot read both of those dimensions, set top_elevation = null and
        let the project default note supply the value. Do NOT guess.
   ⛔ NEVER use ▼柱型天端 (column-stub top) as top_elevation — it sits ABOVE
      ▼基礎天端 and would give too small a depth.

STEP A) Find the ▽GL horizontal line in the cross-section drawing (any prefix —
   ▽GL, ▽SGL, ▽設計GL, ▽C棟GL … are all the same datum; see the ▽*GL note above).
   ⚠️ Do NOT mistake a ▽*FL (finished floor) line for it — ▽*FL is higher up.

STEP A2) DECIDE THE SIGN FROM WHICH SIDE OF THE ▽GL LINE THE DIMENSION IS DRAWN.
   The ▽GL line is the zero datum, and the chain can run either way from it:
   - Foundation top BELOW the ▽GL line (buried — the usual case) → NEGATIVE (e.g. -450).
   - Foundation top ABOVE the ▽GL line (the concrete stands proud of the ground; the
     gap is dimensioned ABOVE the line, e.g. a small "60" between ▽設計GL and the top
     surface) → POSITIVE (top_elevation = +60).
   ⚠️ Do NOT assume every value is negative. Look at where the dimension sits relative
      to the ▽GL line and sign it accordingly.
   ⚠️ CAUTION when BOTH sides are dimensioned: if the number above ▽GL is the 土間/slab
      top (土間天端) and a labelled total spans from it down past ▽GL to the concrete
      (e.g. above 60, total 560, leg 500 → 560 = 60 + 500), then the FOUNDATION top is
      the buried leg → top_elevation = -500, not +60.

STEP B) On the LEFT side, find the dimension from ▽GL to the FOUNDATION SLAB TOP (底盤天端).
   The "foundation slab top" is the TOP SURFACE of the horizontal footing (底盤) at the
   BOTTOM of the foundation — NOT the top of the column stub (柱型) or pedestal.

   For TNF-style deep foundations:
   - The cross-section shows a column stub (柱型) sitting ON TOP of the foundation slab.
   - The OUTER (leftmost) dimension chain spans: ▽GL → [large number: 800–1,500mm] → foundation slab top.
   - The INNER dimension chain shows: ▽GL → [small number: 200mm] → column stub top,
     then [medium number: 600–1,300mm] → down to the foundation slab top.
   - The OUTER chain's single large number = top_elevation. Read it. Write as NEGATIVE
     (it is below ▽GL; only sign it positive when it is drawn ABOVE the line — STEP A2).
   - DO NOT use the small inner chain value (e.g. 200 = column stub top depth) as top_elevation.

   ⚠️ top_elevation can be any value from -2,000mm up to a small POSITIVE number
      (e.g. +60) depending on the project. There is NO "typical small value" bias and
      no "always negative" bias. Read whatever the outer chain shows, and sign it by
      which side of the ▽GL line it is drawn on (STEP A2).

STEP B2) SPECIAL RULE FOR STEPPED/TIERED FOUNDATIONS (段付基礎):
   Some foundation cross-sections have a stepped shape — a wider flat footing at the
   bottom and a narrower raised pedestal in the center (like the shape of an upside-down
   wedding cake). These drawings show TWO separate dimension chains on the left side:

   Chain L1 — OUTERMOST LEFT (aligned with the outer edge of the wide footing):
     The vertical dimension stack furthest to the LEFT of the drawing.
     The topmost number here = distance from ▽GL to the top surface of the OUTER footing.
     This is the TRUE top_elevation.

   Chain L2 — INNER LEFT (aligned with the pedestal or step):
     A second vertical dimension stack closer to the center of the drawing.
     It describes the height of the pedestal/step above the outer footing surface.
     ⛔ Do NOT use numbers from this inner chain as top_elevation.

   HOW TO DISTINGUISH:
   - If you see two separate stacks of dimension numbers on the left side, identify the
     one that is FURTHEST FROM THE CENTER (outermost/leftmost) — that is Chain L1.
   - The outermost chain's first number below ▽GL = top_elevation.

   WORKED EXAMPLE — Stepped foundation F1A:
     Chain L1 (outermost left): 450 → D → 100
       [1] 450 = soil gap from ▽GL to outer footing top → top_elevation = -450  ✓
     Chain L2 (inner, for pedestal): 250 → 200
       [1] 250 = pedestal height above outer footing top → IGNORE
     ✅ ANSWER: top_elevation = -450  (NOT -250)

STEP C) VERIFY your answer using this spatial test:
   - Draw an imaginary line at your extracted top_elevation depth below ▽GL.
   - This line should land exactly on the TOP SURFACE of the concrete block (where soil meets concrete).
   - If that line falls INSIDE the concrete block → you extracted D, not top_elevation. Go back to STEP B and pick the smaller, higher number.
   - For stepped foundations: this line should land on the OUTER flat footing surface (the widest part), NOT on the pedestal top.

STEP D) SIGN RULES:
   * Top surface is BELOW ▽GL  → NEGATIVE  (e.g., 200 mm below → -200)
   * Top surface is AT ▽GL     → ZERO       (→ 0)
   * Top surface is ABOVE ▽GL  → POSITIVE   (e.g., 100 mm above → +100)

STEP E) If no cross-section drawing exists for that type → set top_elevation to null.

--- DIMENSION LAYOUT IN TYPICAL JAPANESE FOUNDATION DRAWINGS ---

Layout A — Simple foundation (flat top):
The left side of a cross-section has exactly 3 stacked dimensions from top to bottom:
   [1] soil gap     ← ▽GL down to concrete top surface  → THIS is top_elevation (e.g. 200)
   [2] "D" bracket  ← spans the full concrete body depth → THIS is D (e.g. 250, 300, 500, 700)
   [3] leveling     ← thin leveling concrete at base     → IGNORE (e.g. 30, 50, 100)
You ALWAYS want dimension [1]. The "D" label always marks dimension [2].

Layout B — Stepped/tiered foundation (段付基礎, 段差基礎):
The drawing shows a stepped profile. The left side has TWO dimension chains:
   Outermost chain (furthest from center):
     [1] soil gap to outer footing top → THIS is top_elevation (e.g. 450)
     [2] D bracket for the outer footing depth → THIS is D
     [3] leveling → IGNORE
   Inner chain (for pedestal/step, closer to center):
     Numbers like 250, 200 describing the pedestal height → IGNORE for top_elevation
⚠️ ALWAYS use the OUTERMOST dimension chain's [1] for top_elevation in stepped foundations.

Layout C — Combined cross-section with PER-TYPE elevation annotations:
Some drawings group multiple foundation types under one title
(e.g., "F5, F6, F8A 基礎断面") but show DIFFERENT depths for different
types using parenthetical labels on dimension lines, for example:
  "(F5, F6 : 1,100)" ← 1,100 mm from ▽GL applies to F5 and F6 only
  "(F8A : 800)"       ← 800 mm from ▽GL applies to F8A only
⚠️ CRITICAL: When you see this layout, you MUST emit a SEPARATE FoundationItem
   for each individual type with its own top_elevation.
   Do NOT emit a single combined entry "F5, F6, F8A" with one top_elevation.

--- WORKED EXAMPLES (memorize these) ---

Example 1 — "F1 基礎断面":
   Left-side scan top→bottom: 200 ... D/250 ... 100
   [1] 200  = soil gap → top_elevation = -200  ✓
   [2] 250  = concrete body (D) → IGNORE
   [3] 100  = leveling → IGNORE
   ✅ ANSWER: top_elevation = -200  (NOT -250)

Example 2 — "F2, F3, F4 基礎断面":
   Left-side scan top→bottom: 200 ... 300 ... 100
   [1] 200  = soil gap → top_elevation = -200  ✓
   [2] 300  = concrete body (D) → IGNORE
   [3] 100  = leveling → IGNORE
   ✅ ANSWER: top_elevation = -200  (NOT -300)

Example 3 — "F3, F5 基礎断面":
   Left-side scan top→bottom: 200 ... 70 ... 30
   [1] 200  = soil gap → top_elevation = -200  ✓
   [2] 70   = concrete body (D) → IGNORE
   [3] 30   = leveling → IGNORE
   ✅ ANSWER: top_elevation = -200  (NOT -70, NOT -30)

Example 4 — TNF deep foundation "F2B 基礎断面":
   Floor plan shows "F2B(GL-1,250)" → USE THIS: top_elevation = -1,250  ✓ (highest priority)
   Cross-section shows two chains on the left:
   Outer chain (leftmost): 1,250 ... D ... 30   ← GL to foundation SLAB TOP
   Inner chain (column stub): 200 ... 1,050      ← GL to stub top / stub height
   [1] outer chain 1,250 = depth to foundation slab top → top_elevation = -1,250  ✓
   Inner 200 = column stub top (GL-200) → IGNORE for top_elevation
   ✅ ANSWER: top_elevation = -1,250  (NOT -200 — 200 is only the column stub top depth)

Example 5 — TNF deep foundation "F1 基礎断面":
   Floor plan shows "F1(GL-1,000)" → USE THIS: top_elevation = -1,000  ✓ (highest priority)
   Cross-section shows:
   Outer chain: 1,000 ... D/550 ... 30   ← GL to foundation SLAB TOP
   Inner chain: 200 ... 800              ← GL to stub top / stub height (200+800=1,000)
   ✅ ANSWER: top_elevation = -1,000  (NOT -200)

Example 6 — Stepped foundation "F1A 基礎断面" (non-TNF style):
   Two dimension chains on the left:
   Outermost chain: 450 ... D/? ... 100
   Inner chain (pedestal): 250 ... 200
   [1] outermost 450 = soil gap → top_elevation = -450  ✓
   Inner 250 = pedestal height above footing → IGNORE
   ✅ ANSWER: top_elevation = -450  (NOT -250 — that is the inner pedestal dimension)

Example 5 — Combined cross-section "F5, F6, F8A 基礎断面" with per-type annotations:
   Left side shows multiple labeled dimension brackets:
     "(F5, F6 : 1,100)" → 1,100 mm gap from ▽GL → F5, F6 top_elevation = -1,100
     "(F8A : 800)"       → 800 mm gap from ▽GL   → F8A top_elevation = -800
   ⚠️ These are DIFFERENT values for different types in the SAME drawing.
   ✅ ANSWER: emit THREE separate FoundationItems:
     F5:  top_elevation = -1,100  ✓
     F6:  top_elevation = -1,100  ✓
     F8A: top_elevation = -800    ✓
   ⛔ WRONG: one combined entry "F5, F6, F8A" with top_elevation = -1,100  ✗

⛔ FORBIDDEN PATTERNS — these are ALWAYS wrong:
   - Using the "D" labeled number as top_elevation
   - Using the bottom leveling number (30, 50, 70mm at very bottom) as top_elevation
   - Summing multiple dimensions (e.g. 200+50=250) → NEVER sum
   - Using the INNER chain small value (e.g. "200" = column stub top depth) instead of the
     OUTER chain large value (e.g. "1,250" = foundation slab top depth) for TNF foundations

G) SPECIAL RULE FOR FW / FG BEAM TYPES (VERY IMPORTANT):
   FW and FG cross-section drawings (e.g., "FW1 断面図") are different from regular
   foundation cross-sections. They show a WALL element (PCパネル / PCPanel) that
   EXTENDS UPWARD above ▽GL AND a horizontal concrete BEAM that sits BELOW ▽GL.

   THERE ARE TWO GROUPS OF DIMENSIONS IN THESE DRAWINGS — DO NOT MIX THEM UP:

   Group 1 — ABOVE ▽GL (wall/panel dimensions — DO NOT USE for top_elevation):
     - Numbers like "108", "12", "975" appearing above the ▽GL line.
     - These describe the height of the PC wall panel or wall elements.
     - Example in FW1: "108" is the PCパネル height above ▽GL → IGNORE THIS.

   Group 2 — BELOW ▽GL (beam/footing dimensions — USE THIS for top_elevation):
     - The ▽GL line is drawn horizontally.
     - Immediately BELOW ▽GL there is a small vertical dimension number
       (e.g., "200") indicating how far the top of the horizontal concrete
       beam/footing is from ▽GL.
     - This number gives top_elevation as a NEGATIVE value.
     - Example in FW1: "200" below ▽GL → top_elevation = -200.

   ⚠️ VALUES VARY BY PROJECT. Never assume a fixed number. Always read the actual value from the drawing.

   WORKED EXAMPLES FOR FW BEAMS (illustrative — actual numbers differ per project):
     Any FW/FG 断面図:
       - Numbers ABOVE ▽GL (e.g., "108", "975", "500", "400以下") = wall/panel heights → IGNORE
       - The FIRST number below ▽GL on the LEFT SIDE = soil gap → top_elevation (as negative)
       ✅ If the first number below ▽GL is "250" → top_elevation = -250
       ✅ If the first number below ▽GL is "200" → top_elevation = -200
       ✅ If the first number below ▽GL is "300" → top_elevation = -300
       ✅ If the first number below ▽GL is "150" → top_elevation = -150

   RULE: For FW/FG beams, the top_elevation is ALWAYS the dimension shown
   BELOW the ▽GL line, measuring down to the top horizontal surface of the beam.
   READ THE NUMBER FROM THE DRAWING. Do not infer it from any other source.

STEP 5: SCAN FOR BEAMS — TWO FORMATS EXIST (FW and FG).

There are TWO completely different beam drawing formats. Detect BOTH:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT A — FW BEAMS (外壁基礎 / PC wall footing)
  Standalone detail drawings, title like "外壁基礎(FW1)詳細図 S=1/60"
  One drawing per variant, may appear side by side.
  D and top_elevation read from the LEFT-SIDE dimension chain below ▽GL.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT B — FG BEAMS (地中梁 / underground beams)
  Appear in a TRANSPOSED TABLE called "地中梁リスト" (underground beam schedule).

  TABLE LAYOUT (types are COLUMNS, fields are ROWS):
    ┌───────┬───────────────────────────┬──────────────┬──────────────┐
    │ 符号  │          FG1              │     FG2      │     FG3      │
    ├───────┼─────────────┬─────────────┼──────────────┼──────────────┤
    │ 位置  │ Y6通り端、中央│ Y5通り端   │   全断面     │   全断面     │
    │       │ X1通り端、中央│ X2通り端  │              │              │
    ├───────┼─────────────┴─────────────┼──────────────┼──────────────┤
    │ 断面  │  [cross-section drawing]  │  [drawing]   │  [drawing]   │
    ├───────┼───────────────────────────┼──────────────┼──────────────┤
    │ B×D  │       900×850             │  400×1,450   │   500×850    │
    ├───────┼───────────────────────────┼──────────────┼──────────────┤
    │上端筋 │ 13-D25  │  9-D25         │  7-D25       │  8-D29       │
    ├───────┼─────────┴──────────────────┼──────────────┼──────────────┤
    │下端筋 │         9-D25             │  4-D25       │  4-D29       │
    │ S t. │      ⊞ - D13@100          │ □ - D13@100  │ ⊞ - D16@100 │
    │ 腹筋 │         2-D13             │  4-D13       │  2-D13       │
    │巾止筋 │      D10@1,000以内        │D10@1,000以内 │D10@1,000以内 │
    └───────┴───────────────────────────┴──────────────┴──────────────┘

  HOW TO EXTRACT FG BEAMS FROM THIS TABLE:
  1. SCAN ALL COLUMNS — The 符号 row has one header per FG type (FG1, FG2, FG3, …).
     You MUST scan EVERY column, not just the first one.
     Extract a SEPARATE FoundationItem for EACH FG type found.
  2. EXTRACT D from the "B×D" row:
     Format is "BxD" e.g. "900×850", "400×1,450".
     D = the SECOND number (after the × sign). Example: 900×850 → D = 850.
     ⚠️ B (first number) is beam WIDTH — do NOT confuse with D.
  3. EXTRACT top_elevation from the 断面 cross-section drawing in the 断面 row:
     Apply the same ▽GL reading rule as STEP 4 / FORMAT A beams.
     The FIRST dimension number below ▽GL on the left side = soil gap → top_elevation (negative).
  4. EXTRACT REBAR:
     rebar_x = the 上端筋 (top reinforcement) spec for this column.
     rebar_y = the 下端筋 (bottom reinforcement) spec for this column.
  5. Some FG types span MULTIPLE POSITION SUB-COLUMNS (e.g. FG1 has two 位置 sub-columns).
     This is the SAME beam FG1 — do NOT create two entries. Create ONE entry for FG1.
     Read D and top_elevation from the first (left) sub-column cross-section.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMON RULES FOR ALL BEAMS (FORMAT A and FORMAT B):
- Extract the Type using ONLY the BASE code (e.g., "FW1", "FW2", "FG1") — no suffixes.
- Set Classification to "FW/FG".
- Set Lx and Ly to 0 (plan dimensions are not relevant for beams).

⚠️ VARIANT SECTIONS — ONE ENTRY PER BEAM TYPE (CRITICAL):
  Some beam types have MULTIPLE cross-section panels or sub-columns showing different
  structural conditions. Common examples:
    "FW1 (一般部)"   = FW1 at a typical wall section (general case)
    "FW1 (間柱部)"   = FW1 at a column/stud location
    FG1 with "Y6通り端" and "Y5通り端" as two position sub-columns
  THESE ARE NOT SEPARATE BEAM TYPES. They are variants of the SAME beam.
  Rules:
  1. Extract only ONE FoundationItem per base beam code (FW1, FW2, FG1, …).
  2. Set type = base code only ("FW1", "FG1"), never include position suffixes.
  3. Read D and top_elevation from the first/general panel or sub-column.
  4. Do NOT create two entries for the same base type.

- EXTRACT BEAM DEPTH D:
  ▶ FORMAT A (FW beams): D is NOT in any table. Read from the cross-section drawing.
  ▶ FORMAT B (FG beams): D is explicitly in the "B×D" table row. Use that value.
    The cross-section drawing confirms it but the table value is authoritative.

  ⚠️ D VARIES BY PROJECT. Never assume a fixed value. Always read from the source.

  ━━━ THE THREE LEFT-SIDE NUMBERS BELOW ▽GL ━━━

  On the LEFT side of the cross-section, below ▽GL, you will see 2–3 dimension numbers
  stacked vertically in a chain going downward. They always represent in order:

    № 1 (topmost, closest to ▽GL) = top_elevation gap  ← already found in STEP 4
    № 2 (directly below №1)       = D  ← THIS IS WHAT YOU WANT
    № 3 (bottommost, smallest)    = leveling concrete (30–50mm) ← IGNORE

  D = № 2.  Always.  It is the number sitting directly below the top_elevation number.

  ━━━ WHAT EACH NUMBER REPRESENTS ━━━

  №1 top_elevation (e.g. 200):
    Spans from ▽GL line DOWN to the TOP SURFACE of the concrete.
    It is in the SOIL zone — the gap between ground level and concrete.
    ⛔ This is NOT D. Do not use it as D.

  №2 D (e.g. 350 or 100):
    Spans from the TOP SURFACE of the concrete DOWN to the BOTTOM of the concrete body.
    It is the structural HEIGHT of the footing/beam itself.
    ✅ This IS D.
    For thin beams: the label may be pushed BELOW the thin body but it is still №2.
    D can be smaller OR larger than top_elevation — both are possible.

  №3 leveling (e.g. 30):
    The smallest number at the very bottom (30–50mm).
    ⛔ IGNORE. Not D.

  ━━━ BOUNDARIES — WHAT TO IGNORE ━━━

  ABOVE ▽GL: Numbers like "400以下", "150以上", "975", "108", "12" describe wall/panel heights.
  ALL of these are ABOVE the ▽GL line → IRRELEVANT for D. Never use them.

  RIGHT SIDE: Numbers like "200", "250", "150" with label 土間コンクリート (t=***) = slab thickness.
  → IGNORE for D.

  BOTTOM: Numbers like "300", "500" = plan widths (horizontal).
  → IGNORE for D.

  ━━━ WORKED EXAMPLES ━━━

  Example A — Left side: 200 → 350 → 30  (and right side shows 200 or 250):
    №1 = 200 → top_elevation = -200
    №2 = 350 → D = 350  ✓
    №3 = 30  → leveling, IGNORE
    Right side "200" or "250" = slab, IGNORE
    ✅ D = 350  (NOT 200 — 200 is №1, the soil gap, NOT D)

  Example B — Left side: 200 → 100  (thin beam, no leveling shown separately):
    №1 = 200 → top_elevation = -200
    №2 = 100 → D = 100  ✓  (label pushed below thin body)
    No №3 visible.
    ✅ D = 100  (NOT 200 — 200 is №1, the soil gap, NOT D)

  SIGN / UNIT: D is always a POSITIVE integer in mm.
  If no cross-section drawing exists or dimension is unclear → set D to 0.

STEP 6: EXTRACT BOUNDING BOXES (CRITICAL).
- For EVERY item you extract (Foundation or Beam), you MUST identify its Bounding Box on the page.
- For Foundations (F1, F2, etc.):
    Box enclosing the CROSS-SECTION DRAWING (基礎断面) for this foundation type —
    the labeled side-view drawing such as "F1 基礎断面", "F1A 基礎断面", etc.
    ⚠️ Do NOT use the table row as the bounding box.
    - If multiple types share one drawing (e.g., "F4, F4A 基礎断面"), ALL those types
      share the same bounding box covering the entire combined drawing.
    - Include the title label (e.g., "F1 基礎断面") and the full cross-section area.
    - LAYOUT B: the block's mini table sits between the title and the drawing —
      include it, and extend ymax down past the bottom dimension stack (the
      1,600 / 3,600 / 4,600 lines) so the classification evidence is visible.
      Stop the box before the NEXT block's title; never let one box cover two
      foundations (F1A and F3A sit side by side and are separate items).
    - If no cross-section drawing exists for a type, set region to null.
- For FW Beams (FORMAT A — standalone detail drawings):
    In many Japanese drawing sheets, the beam detail layout is:
      ┌─────────────────────────────────┐  ← top of bordered region (high on page)
      │   [ACTUAL CROSS-SECTION drawing]│  ← rebar, dimensions, ▽GL shown here
      │   [possibly multiple panels]    │
      └─────────────────────────────────┘
         外壁基礎(FW1)詳細図  S=1/60      ← caption/label at the BOTTOM

    The LABEL (e.g. "外壁基礎(FW1)詳細図") is at the BOTTOM, the drawings are ABOVE.
    DO NOT set ymin to the label location — that gives an empty box!
    The bounding box MUST cover the entire bordered drawing block ABOVE the label.

    Rules:
    - ymin = top edge of the bordered drawing block (NOT the label line)
    - ymax = bottom edge including the label caption
    - xmin/xmax = left/right edges of the drawing block
    - If multiple panels are side-by-side (e.g. FW1-left and FW1-right), include ALL of them.
    - The box must capture: rebar lines, ▽GL marker, dimension numbers, concrete hatching.

- For FG Beams (FORMAT B — 地中梁リスト table):
    Each FG type occupies one (or more) columns in the table.
    The bounding box for FG1, FG2, FG3 etc. should cover THAT COLUMN of the table:
    - ymin = top of the table (including the 符号 / header row)
    - ymax = bottom of the table (including 巾止筋 row)
    - xmin/xmax = left/right edge of that FG type's column (or merged columns if multi-position)
    This allows the second-pass crop to see the 断面 drawing, B×D value, and rebar specs.
    If a type spans multiple position sub-columns (e.g. FG1 with 2 sub-columns), the box
    must cover ALL sub-columns of that FG type.
- Provide: page number (1-based), ymin, xmin, ymax, xmax (0-1000 scale).

STEP 7: EXTRACT MAIN TABLE REGION (CRITICAL FOR FOUNDATION TABLE IMAGE).
- LAYOUT A: provide a bounding box for the ENTIRE "Foundation List" table in the
  `table_region` field.
  CRITICAL: The table region MUST extend DOWN to include the VERY LAST ROW (e.g., F7,
  F8, or whatever is the final foundation item).
  Do NOT cut off the bottom row. Ensure ymax captures the bottom border of the last row.
- LAYOUT B: there is no single schedule. Set `table_region` to a box covering ALL the
  per-foundation mini tables on the 基礎リスト sheet (their combined extent), so the
  preview shows every type/Lx/Ly/D/ベース筋 row at once. If the mini tables are spread
  over several sheets, use the sheet holding the most of them.
- Either way this is separate from individual item regions.

STEP 8: DETECT AND HIGHLIGHT OVAL GL MARKERS IN FLOOR AREA.

━━━ RULE 0 — TEXT CONTENT FILTER (CHECK FIRST, BEFORE SHAPE) ━━━
The text inside the shape MUST literally start with "GL" followed by a sign and a number:
  Valid:   "GL+120", "GL-50", "GL±0", "GL+200", "GL+45"
  INVALID: "33,000"  ← this is a grid dimension. NEVER a GL marker.
  INVALID: "8,020"   ← grid dimension. IGNORE.
  INVALID: "18,000"  ← grid dimension. IGNORE.
  INVALID: any pure number without "GL" prefix → NOT a GL marker, IGNORE.

⛔ DIMENSION BOXES AT THE BOTTOM OF THE DRAWING:
   Japanese floor plans always have a row of dimension strings below the plan
   (e.g., 8,020 / 8,250 / 33,000 / 84,000) in thin or dashed rectangles.
   These are structural grid dimensions. They NEVER contain GL text.
   Even if they look like a box → IGNORE ALL OF THEM COMPLETELY.

━━━ RULE 1 — SHAPE FILTER (CHECK SECOND) ━━━
Accept GL markers with EITHER oval/pill OR rectangular borders.
The ONLY requirement is the text must start with "GL".

✅ YES - EXTRACT (any border shape + GL text):
   ( GL+200 )       ← oval/pill border ✓
   ( GL±0 )         ← oval/pill border ✓
   [ GL-20  ]       ← RECTANGLE border — ALSO VALID ✓
   [ GL+120 ]       ← RECTANGLE border — ALSO VALID ✓
   ( GL+120 )-->    ← oval with leader arrow → VALID, extract ONLY the bubble

❌ NO - IGNORE (no GL text, regardless of border):
   [ 33,000 ]       ← number-only rectangle → IGNORE
   [ 8,020  ]       ← dimension box → IGNORE

━━━ RULE 2 — LOCATION ━━━
Only detect GL markers INSIDE the main floor plan area (the hatched/column grid region).
Ignore any GL text that appears in legends, title blocks, or dimension rows at the edges.

⛔ FOUNDATION FOOTPRINT SYMBOLS — NOT GL MARKERS:
   The floor plan contains dashed square or rectangular outlines showing the plan view
   of column foundations (footings). These appear as nested dashed squares/rectangles
   grouped at column grid intersections. Labels nearby: F1, F2, F3, F7, F8, FW1, etc.
   THESE ARE NOT GL MARKERS. Do NOT detect them.
   They contain no "GL" text. They are structural footprint outlines.
   Even if they look like a box in the plan → IGNORE.

━━━ RULE 2b — COMPLETENESS ━━━
Scan the ENTIRE interior of the floor plan area. GL markers can appear anywhere —
center, corners, near walls, near foundations. A typical plan has 3–10 GL markers.
Do NOT stop after finding 1–2. Find ALL of them.

━━━ RULE 3 — BOUNDING BOX: OVAL BUBBLE ONLY ━━━
Many GL markers have a leader line (arrow) extending from the oval to point at a slab area.
When setting the bounding box coordinates:
  - INCLUDE: only the oval/pill bubble itself
  - EXCLUDE: the leader line, the arrow, any extending line
  - The box should tightly wrap the oval text bubble, not the full annotation

COORDINATE SYSTEM:
- 1000×1000 grid. Top-left = (0,0). Bottom-right = (1000,1000).
- xmin/xmax = left/right edges of the OVAL BUBBLE only
- ymin/ymax = top/bottom edges of the OVAL BUBBLE only

FOR EACH GL MARKER FOUND:
- Extract the text content (e.g., "GL+200", "GL±0", "GL+120")
- Provide region: Page, ymin, xmin, ymax, xmax (0-1000 scale, oval bubble only)

STEP 9: IDENTIFY THE MAIN FLOOR PLAN REGION.
- Find the main floor plan drawing — the large hatched view showing the building layout
  with column grid lines, foundation footprint symbols (dashed squares), and GL markers.
- This is the 基礎伏図 or 平面図 — NOT the foundation table, NOT beam cross-sections.
- Provide its bounding box in the `floor_plan_region` field.
- Include the full extent of the hatched slab area with all GL markers inside.
- If no clear floor plan exists, set floor_plan_region to null.

STEP 10: DETECT PIT HOLE DRAWINGS (ピット詳細図 / hố pit).

Scan ALL pages for cross-section drawings showing underground pits or channels.
These are identified by drawing titles such as:
  - "側溝詳細図"           (side drain / drainage channel pit)
  - "レールのピット詳細図"  (rail pit)
  - "EVピット①詳細図"      (elevator pit №1)
  - "EVピット②詳細図"      (elevator pit №2)
  - Any title containing "ピット" (pit) and "詳細図" or "断面図"

DO NOT confuse these with regular foundation drawings (F1, F2) or beam drawings (FW, FG).
Pit drawings specifically show a U-shaped or rectangular cavity dug into the ground,
often labeled 地盤改良 (ground improvement) at the bottom, with the pit floor slab inside.

⛔ CRITICAL — DO NOT EXTRACT PITS FROM THE 基礎伏図 (FLOOR PLAN):
The 基礎伏図 page sometimes contains a small embedded detail box in a corner with a
pointer/leader line from the floor plan to the box. This box may be labeled
"レールのピット断面図" or similar and shows an internal construction detail (rail anchor,
base plate, rebar configuration). This embedded box is a SUPPLEMENTARY DETAIL embedded
inside the floor plan — it is NOT a standalone pit drawing.
⛔ NEVER create a pit entry from an embedded box inside the 基礎伏図.
✅ ONLY create pit entries from standalone pit drawings that appear as dedicated sections
   OUTSIDE the main floor plan boundary (i.e., they have their own bordered cell and title
   that stands alone, not connected by a leader line to the floor plan grid).

⛔ CRITICAL — ONE ENTRY PER PIT TYPE (NO DUPLICATES FROM 詳細図 + 断面図):
A pit type may appear in TWO drawings on different pages or locations:
  "レールのピット詳細図"  ← standalone detail (use THIS for the entry)
  "レールのピット断面図"  ← embedded in floor plan or supplementary section (IGNORE for entry)
Create exactly ONE pit entry per unique pit name. If both exist, use only the 詳細図.
Do NOT create two entries for the same pit name.

⚠️ DO NOT classify 土間段差 (floor step / slab level change) as a pit.
土間段差 is a detail showing a step-down transition between two floor slab levels — it is a
SLAB detail, NOT an underground pit. It looks similar (has ▽GL, 地盤改良, vertical dimension ~350mm,
horizontal dimensions ~150mm) but its title is "土間段差" with NO "ピット" in the name.
Never extract top_elevation or D from the 土間段差 drawing, and never use its dimensions
(150, 350, 100, etc.) for any pit entry.

⚠️ DO NOT use 地盤改良 (ground improvement) depth values as pit depths.
Specs like "二次改良 GL-850・1,650mm" or "一次改良 GL-3,050mm" describe the depth of
ground treatment zones — they are NOT the pit floor elevation. Ignore these numbers entirely
when reading pit top_elevation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOR EACH PIT DRAWING FOUND, EXTRACT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. type: The pit name from the drawing title.
   Strip scale notation. Copy the name EXACTLY as written — do NOT translate,
   paraphrase, or substitute synonyms. If the title says "消火水槽", type = "消火水槽".
   If it says "清水槽", type = "清水槽". Reproduce verbatim.
   Examples:
   "側溝詳細図 S=1/50"          → type = "側溝"
   "レールのピット詳細図 S=1/50"  → type = "レールのピット"
   "EVピット①詳細図 S=1/60"     → type = "EVピット①"
   "EVピット②詳細図 S=1/60"     → type = "EVピット②"
   "EVピット①②詳細図 S=1/60"   → type = "EVピット①②"
   "EVピット①、②詳細図 S=1/60" → type = "EVピット①、②"
   "消火水槽詳細図 S=1/60"       → type = "消火水槽"
   "清水槽詳細図 S=1/60"         → type = "清水槽"

   ⚠️ COMBINED PIT LABELS: If the title contains combined numbers such as "EVピット①②"
   or "EVピット①、②", that means BOTH pits are described in ONE drawing.
   Create ONE entry with the combined type name exactly as written in the title.
   Do NOT split into separate "EVピット①" and "EVピット②" entries.

2. readable + top_elevation:

   DECISION FLOW — follow in order:

   STEP A — Is this a STEPPED PIT (two floor levels visible in the cross-section)?
   A stepped pit has one floor level on the right side and a deeper stepped zone on the left side.
   The deeper left zone often has ※ marks ("※は、意匠図参照のこと。").

   If YES (stepped pit):
   - Read the depth to the HIGHER (shallower) floor = the side with the SHORTER distance from ▽GL.
     This is almost always the RIGHT side floor, and its dimension IS explicit (no ※).
   - Set readable=true and top_elevation = -(that explicit dimension).
   - The ※ on the LEFT/DEEPER side does NOT make the pit unreadable —
     it only means the deeper step's dimension is deferred to architectural drawings.
   Example: right floor = 1,250mm below ▽GL (explicit), left deeper step = ※
   → readable=true, top_elevation=-1250.

   STEP B — Is this a SIMPLE PIT (one floor level, no step)?
   If YES (simple pit):
   - readable=true ONLY if there is a CLEAR, EXPLICIT numeric dimension from ▽GL straight
     down to the floor top surface, AND that dimension is NOT marked ※.
   - readable=false in ANY of these cases:
     ① The vertical dimension from ▽GL to the floor top is marked ※ (even if other numbers
        like "30" or "150" also appear — those are cover/slab-thickness annotations, NOT the depth).
     ② The pit floor top surface is at ▽GL level or ABOVE ▽GL (no soil zone below GL).
     ③ The sanity check below fails AND no valid large dimension can be found.

   DIMENSION IDENTIFICATION CHECK:
   Before using any number as top_elevation, confirm it is the VERTICAL GL-to-floor dimension:
   - It must be drawn as a VERTICAL arrow/line from ▽GL straight DOWN to the floor top surface.
   - Small numbers like "30", "50", "70" = cover/leveling concrete → NOT top_elevation.
   - Numbers labeled "D" = slab body thickness → NOT top_elevation.
   - Horizontal numbers (step widths, wall thicknesses, channel widths) → NOT top_elevation.
   - If the only vertical GL-to-floor link is ※ → readable=false (regardless of size).
   - ⚠️ TWO ENDPOINTS define top_elevation — get BOTH right (do NOT just take the
     largest number when several vertical dimensions are stacked side by side):
       • TOP endpoint  = the ▽GL line (measure FROM ▽GL, never from a lower datum).
       • BOTTOM endpoint = the TOP surface of the bottom slab (where the fill/soil
         meets the top of the concrete) — NOT the slab's underside.
     Two common traps:
       (a) An inner chain may START below ▽GL (at a step/slab top). Ignore it; use
           the chain whose top touches ▽GL.   e.g. inner 1,400 vs 1,495-from-▽GL → 1,495.
       (b) A dimension may run PAST the slab top down to the slab UNDERSIDE, i.e. it
           equals (depth-to-slab-top + slab thickness D). That is NOT top_elevation.
           e.g. 1,000 to the slab top vs 1,200 to the slab bottom (D=200) → use 1,000.
     ⛔ Pick the dimension running from ▽GL to the slab TOP — not the largest, not the
        one to the slab bottom, not one starting below ▽GL.
   - top_elevation can be any value (100mm, 300mm, 1,000mm, etc.) — there is no minimum threshold.

3. D: Thickness of the pit floor slab.
   - D sits in the SAME vertical dimension chain as top_elevation, IMMEDIATELY BELOW it:
     top_elevation = ▽GL → slab top, and the very next segment down (slab top → slab
     bottom) IS D. So read D right next to / just below where you measured ▽GL.
   - ⛔ Do NOT read D from a different part of the drawing — a step/段 dimension, the
     土間コンクリート thickness, or any horizontal width. e.g. if a "150" step appears on
     the far side but the slab in the ▽GL chain is "200", D = 200 (NOT 150).
   - Exclude the leveling concrete (捨てコンクリート t=50) beneath the slab.
   - Typical range: 150–300mm. If the value exceeds 400mm, re-check — you are likely
     reading a foundation (F-type) slab or beam dimension, not the pit slab.
   - Do NOT return null just because the drawing is busy or a deeper step is marked ※.
   - If readable=false or dimension is genuinely unclear, set D=null.

   ━━━ CRITICAL — STAY INSIDE THE PIT'S BORDERED CELL ━━━
   Detail sheets often place MULTIPLE drawing blocks on the same page:
     • Pit drawings (ピット詳細図)
     • Wall/foundation details (FW1, FW1A — 外壁基礎詳細図)
     • Floor step details (土間段差)
     • Column type lists (柱型リスト)

   When extracting top_elevation and D for a pit, read ONLY from within the
   bordered cell that carries the pit name. Values from neighboring cells
   (FW1: "350", 土間段差: "150", 柱型リスト, etc.) MUST NOT be used.
   If you are uncertain which cell a dimension belongs to, choose the largest
   continuous vertical dimension that starts at ▽GL inside the pit's own cell.

4. region: Bounding box covering the pit drawing.

   ⚠️ REGION IS MANDATORY: You MUST always provide a non-null region for EVERY pit
   you detect. Even if you are uncertain about the exact boundaries, provide your best
   estimate based on the drawing's visible extent on the page. A null region means
   no image will be shown for this pit, which is always wrong.

   ━━━ CRITICAL — 詳細図 vs 断面図 ━━━
   A pit may have TWO drawings with very similar names:
     "レールのピット詳細図"  ← OVERVIEW detail drawing (show the full pit shape + ▽GL)
     "レールのピット断面図"  ← INTERNAL cross-section often embedded in the 基礎伏図

   RULE: Create exactly ONE pit entry per unique pit type name.
   - If a standalone 詳細図 exists → use it; completely ignore any 断面図 for that type.
   - If only a standalone 断面図 exists (no 詳細図 anywhere) → use the 断面図.
   - If the 断面図 is embedded inside the 基礎伏図 (connected by a leader line to the plan)
     → it is NOT a standalone drawing; treat it as non-existent for pit detection purposes.

   ALWAYS set region to the standalone 詳細図 drawing block, NOT the 断面図.
   The 詳細図 is the primary pit overview. The 断面図 shows internal construction
   details only — it must NOT be used as the pit's bounding box.

   Rules:
   - ymin = top of the 詳細図 drawing block
   - ymax = bottom of the title caption text ("詳細図 S=1/50") — the caption MUST be
     INSIDE the bounding box; do not clip the caption
   - xmin/xmax = left/right edges of the bordered cell

   ━━━ CRITICAL — ONE BORDERED RECTANGLE = ONE PIT ━━━
   Some pit drawings show TWO side-by-side cross-section views inside ONE bordered rectangle.
   For example, a rectangle may contain a left-side view (without access opening) and a
   right-side view (with access opening 人通口) of the same pit type.
   In this case, create ONLY ONE pit entry — NOT two separate entries.
   Use the dimensions from whichever view shows ▽GL and the full depth dimension clearly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example A — UNREADABLE (vertical dimension from ▽GL to floor is ※):
  Drawing title: "側溝詳細図 S=1/50"
  Left-side dimension chain from ▽GL going DOWN:
    ▽GL → ※ (the VERTICAL depth to the floor top is marked ※, not a number)
    Below that: "30" (bottom cover), "150" (slab thickness D)
    Horizontal dimensions at bottom: "150 ※ 50"
    Note: "※は、意匠図参照のこと。"

  The "30" and "150" visible on the left are the bottom cover and slab thickness D.
  They are dimensions WITHIN the concrete slab, NOT the depth from ▽GL to the floor top.
  The VERTICAL distance from ▽GL down to the floor top surface is marked ※ — unreadable.
  → type="側溝", readable=false, top_elevation=null, D=null

  ⛔ WRONG: reading "150" as top_elevation = -150 (150 is slab thickness D, not GL-to-floor depth).
  ✅ RIGHT: the ※ on the vertical GL-to-floor dimension = unreadable.

  NOTE: This differs from a STEPPED pit where ※ appears on the DEEPER side only —
  in that case the pit IS readable using the EXPLICIT dimension on the shallower side.

Example B — UNREADABLE (floor top surface at or above ▽GL):
  Drawing title: "レールのピット詳細図 S=1/50"
  The rail pit is a shallow groove cut INTO the floor slab. Its bottom is ABOVE ▽GL
  (the groove lives inside the slab thickness, not in the soil below).
  The drawing shows: ▽GL dashed line at top, then numbers like "80", "130", "250"
  which are ALL horizontal widths or slab-internal clearances — there is NO downward
  vertical soil zone between ▽GL and the concrete. The concrete starts AT or ABOVE ▽GL.
  Even if a number like "130" appears to the left of the ▽GL line in the 断面図 embedded
  in the 基礎伏図, that "130" is a HORIZONTAL or SLAB-INTERNAL dimension, NOT a vertical
  depth from ▽GL downward into soil.
  → type="レールのピット", readable=false, top_elevation=null, D=null

  ⛔ WRONG: reading "130" or "200" as top_elevation = -130 or -200.
  ⛔ WRONG: using the "200" from the floor plan's default note (特記無き基礎天端高さGL-200)
            as the pit top_elevation. That note applies to FOUNDATIONS, not pits.
  ✅ RIGHT: no soil zone below ▽GL before hitting concrete = floor is at/above GL = unreadable.

Example C — READABLE:
  Drawing title: "EVピット①詳細図 S=1/60"
  Dimension from ▽設計GL down to pit floor top: 1,250mm (not marked ※)
  Pit floor slab thickness: 200mm
  → type="EVピット①", readable=true, top_elevation=-1250, D=200

Example D — READABLE (multiple cross-sections for the same pit):
  Drawing title: "EVピット②詳細図 S=1/60" (with A-A and B-B cross-sections)
  Both A-A and B-B show: 1,650mm from ▽GL to pit floor top, D=200mm
  → Create ONE entry: type="EVピット②", readable=true, top_elevation=-1650, D=200
  (Do NOT create two entries for A-A and B-B — they are sections of the same pit.)

⛔ WRONG: Creating separate entries for "EVピット②(A-A)" and "EVピット②(B-B)".
✅ RIGHT: One entry for "EVピット②" using the values from the first cross-section.

Example E — COMBINED LABEL (both pits described in one drawing):
  Drawing title: "EVピット①、②詳細図 S=1/60"
  Dimension from ▽GL down to pit floor top: 1,250mm (not marked ※)
  Pit floor slab thickness: 200mm
  → Create ONE entry: type="EVピット①、②", readable=true, top_elevation=-1250, D=200

⛔ WRONG: Creating separate "EVピット①" and "EVピット②" entries.
✅ RIGHT: One entry for "EVピット①、②" (preserve the combined label exactly as written).

Example F — STEPPED PIT (two floor levels, one side ※):
  Drawing title: "水盤ピット詳細図 S=1/60"
  Left side of drawing: shows 1,150 (a structural partial dimension) and ※ marks
  Right side of drawing: ▽GL to right floor top = 1,250mm (explicit vertical dimension, not ※)
  Pit floor slab thickness: 200mm
  Same page also contains: 土間段差 (floor step detail with 150mm) and FW1 (wall detail with 350mm)
  → type="水盤ピット", readable=true, top_elevation=-1250, D=200
  (Use 1,250 — the large vertical dimension from ▽GL to the concrete floor.
   Ignore 150 from 土間段差 and 350 from FW1; those are from different drawing cells.)

⛔ WRONG: top_elevation=-150 (reading 150mm step from neighboring 土間段差 cell).
⛔ WRONG: D=350 (reading FW1 wall height from neighboring cell).
✅ RIGHT: top_elevation=-1250, D=200 (reading from INSIDE the 水盤ピット bordered cell).

Example G — TWO DRAWINGS IN ONE BORDERED RECTANGLE:
  One bordered rectangle contains: LEFT view (without 人通口) + RIGHT view (with 人通口 口-600×600)
  Both views have title "水盤ピット詳細図 S=1/60" or share the same title below the rectangle.
  → Create ONE entry for "水盤ピット". Use the view with the clearest ▽GL and dimensions.

⛔ WRONG: Creating two separate "水盤ピット" entries for the left and right views.
✅ RIGHT: One entry for "水盤ピット" using dimensions from the primary view.

Example H — top_elevation runs from ▽GL to the SLAB TOP (two traps):
  TRAP A — an inner chain starts BELOW ▽GL:
    Two side-by-side verticals: inner 1,400 (starts at a step/slab top, below ▽GL)
    and outer 1,495 (from ▽GL down to the floor top). Use the one starting at ▽GL.
    → top_elevation = -1495   (NOT -1400)
  TRAP B — a dimension runs to the slab UNDERSIDE:
    A "1,000" vertical from ▽GL to the TOP of the bottom slab, and a "1,200" that
    continues to the slab BOTTOM (1,000 + 200 slab thickness D).
    → top_elevation = -1000   (NOT -1200; 1,200 = slab-top depth + D)
  ⛔ Never just pick the largest number — pick the one from ▽GL to the slab TOP.

Extract the following information for each item:
1. Type (基礎符号)
2. Dimensions: Lx, Ly, D. (Fill 0 for beam Lx/Ly if N/A; read D from drawing for beams).
3. Top Elevation (top_elevation): Distance from ▽*GL to top surface in mm, negative if
   below the ground line, POSITIVE if above it.
4. Rebar Info: X, Y.
5. Remarks (備考) — "" when the sheet has no 備考 column (LAYOUT B).
6. Classification: "DD", "D", "TNF", or "FW/FG" for beams.
   LAYOUT A → from the 備考 B-formula count.  LAYOUT B → from the bottom
   dimension-stack count (3/2/1). See STEP 3.
7. Region: Page, ymin, xmin, ymax, xmax.
8. Oval GL List: List of detected GL markers with text and region.

⚠️ FINAL REMINDER — the two sheet layouts:
   • A page-wide 基礎リスト WITH a 備考 column  → LAYOUT A rules.
   • Per-foundation mini tables, NO 備考 column → LAYOUT B rules: one item per mini
     table, remarks="", classification from the bottom dimension stack.
   Never emit an empty foundation_list just because there is no page-wide schedule —
   look for the per-foundation mini tables instead.

IMPORTANT: Ensure numeric values for Lx and Ly are returned as plain numbers (e.g., 2000), NOT as strings with commas (e.g., "2,000").
"""

SYSTEM_INSTRUCTION = (
    "You are an expert Quantity Surveyor for Japanese construction projects. "
    "Foundation data comes in TWO sheet layouts and you must handle both: (A) one "
    "page-wide 基礎リスト with a 備考 column, or (B) a per-foundation mini table under "
    "each 基礎断面 with no 備考 column — in layout B the classification comes from the "
    "number of width dimensions stacked under the drawing (3=DD, 2=D, 1=TNF). "
    "Any ▽*GL prefix (GL, SGL, 設計GL, C棟GL, …) is the same ground-line datum; ▽*FL is not. "
    "Behavior: 1. Extract Foundations from tables. 2. Extract Beams (FW/FG) from detail drawings. "
    "3. Detect Floor Slabs and GL markers. 4. Detect Regular Floors (same elevation GL markers). "
    "5. Detect Sloped Floors (GL markers connected by arrows). "
    "6. Detect Pit Hole drawings (ピット詳細図) and classify as readable/unreadable. "
    "7. STRICTLY provide bounding boxes for cropping."
)

BEAM_PAGE_PROMPT = """
You are a structural engineering drawing expert reading a Japanese construction drawing page.
Task: extract D (beam depth) and top_elevation (soil gap) for these beam types: {beam_list}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — LOCATE THE CORRECT DRAWING SECTION (DO THIS FIRST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  This page likely contains MULTIPLE types of structural drawings:
  • Regular foundation cross-sections (F1, F2, F5B, F6 etc.) — DO NOT read these
  • FW beam panels — look for the block labeled "外壁基礎詳細図" or "外壁基(FW)詳細図"
  • FG beam table — look for the block labeled "地中梁リスト"

For FW beams: find the grouped panel area (usually right or lower portion of the sheet)
  that contains multiple panels side by side: [FW1 (一般部)] [FW1 (間柱部)] [FW2]
  The entire group is captioned "外壁基礎(FW)詳細図 S=1/60" or similar AT THE BOTTOM.

For FG beams: find the table whose top-left cell says "符号" and has FG1/FG2/FG3 as column
  headers. This table is captioned "地中梁リスト S=1/60" or similar.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — READ VALUES INSIDE EACH FW PANEL (NOT FROM THE FULL PAGE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each FW panel is a self-contained bordered rectangle. Read ONLY within that panel.

INSIDE the panel, look at its OWN left edge for a vertical chain of dimension numbers:

    ┌─────────────────────────────────────────┐
    │ [wall/PC panel above ground level]      │  ← wall height (e.g. 500, 1200): IGNORE
    │                                         │
    ▽設計GL ─────────────────────────────────── ← dashed horizontal line with ▽ triangle
    │ 250 ↕ [empty soil zone — no hatching]  │  ← soil gap = top_elevation = -250
    ├─────────────────────────────────────────┤  ← top surface of concrete beam
    │ 350 ↕ [hatched concrete + × rebar]     │  ← beam depth = D = 350
    ├─────────────────────────────────────────┤  ← bottom of beam
    │  30 ↕ [leveling concrete]              │  ← leveling ≈ 30 mm: IGNORE
    └─────────────────────────────────────────┘
         300 or 500 (horizontal width)          ← plan width: IGNORE

  • top_elevation = the vertical dimension in the EMPTY zone (between ▽GL and concrete top)
    → return as NEGATIVE integer  (e.g. 250mm gap → -250)
  • D = the vertical dimension spanning the HATCHED concrete body
    → return as POSITIVE integer  (e.g. 350)

⚠️ CRITICAL — OUTER TOTAL DIMENSION IS NEITHER top_elevation NOR D:
  A panel's left chain often shows THREE stacked numbers where the OUTERMOST equals
  the sum of the inner two — the soil gap and the beam body, e.g.:
        325  ← soil gap  (top_elevation = -325)
      1,000  ← OUTER TOTAL = 325 + 675  ← ❌ NOT D, NOT top_elevation — IGNORE
        675  ← hatched beam body  (D = 675)
  Rule: if value_A + value_B == value_C, then C is the overall embedment total —
  discard C. top_elevation = the upper sub (A, the soil gap, → negative); D = the
  lower sub (B, the concrete body). NEVER report the total (1,000) as D.

If a panel type has two variants (FW1 一般部 and FW1 間柱部), use the 一般部 panel.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — READ VALUES FROM FG TABLE (地中梁リスト)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• D: read from the "B×D" table row.  Format "WIDTHxDEPTH" → D = DEPTH (second number).
  Example: "400×1,450" → D = 1450
• top_elevation: read from the cross-section drawing in the 断面 table row using the same
  empty-zone / hatched-zone method described above.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUMBERS TO ABSOLUTELY IGNORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Rebar annotations (INSIDE hatched concrete zone):
   "D10@200", "D13@200 (シングル)", "1-D13", "2-D13", "3-D13"
   These are steel bar specifications — the number after "@" is bar spacing, not D.
❌ Dimensions from REGULAR FOUNDATION drawings (F1, F2, F5B etc.) on the same page.
   Those foundations have their own ▽GL lines and dimension chains — do NOT use them.
❌ The OUTER total dimension that spans BOTH the soil gap and the beam body (it equals
   their sum, e.g. 1,000 = 325 soil + 675 beam). It is the overall embedment depth, NOT
   D. D is ONLY the hatched beam body (the lower sub, 675). See the CRITICAL note above.
❌ Wall/panel height above ▽GL: numbers like 500, 1,200 are above ground, not underground.
❌ Wall cap width: a small "150" shown as a horizontal dimension at the very top of the wall.
❌ Slab thickness: "150" or "t=150" appearing on the RIGHT side of the ▽GL line.
❌ Horizontal plan widths: "300", "500", "550" at the bottom of the panel.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — REPORT BOUNDING BOX OF EACH BEAM'S CROSS-SECTION DRAWING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each beam type, also report the bounding box (ymin, xmin, ymax, xmax) of its
cross-section drawing in 0–1000 scale relative to the image sent to you:
• FW beams: the bbox of the entire FW panel rectangle (top border to bottom including
  the dimension numbers, but NOT the caption text below it)
• FG beams: the bbox of the cross-section drawing cell (断面 row) for that beam's
  column in the 地中梁リスト table (include the cell borders and drawing inside)
Set all four to 0 if you cannot locate the cross-section drawing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return JSON with a "beams" array. Include all target types plus any extra FW/FG found:
{{"beams": [
  {{"type": "FW1", "D": 350, "top_elevation": -250, "ymin": 320, "xmin": 450, "ymax": 750, "xmax": 650}},
  {{"type": "FW2", "D": 350, "top_elevation": -250, "ymin": 320, "xmin": 660, "ymax": 750, "xmax": 870}},
  {{"type": "FG1", "D": 850, "top_elevation": -250, "ymin": 100, "xmin": 200, "ymax": 400, "xmax": 400}}
]}}
Use D=0 or top_elevation=0 only when truly unreadable. Use all four bbox values = 0 when
the cross-section drawing cannot be located.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Gemini call with retry
# ─────────────────────────────────────────────────────────────────────────────

FIRST_PASS_MAX_LONG_SIDE = 1800  # px — Gemini reads numerals fine at this size


def _call_gemini_with_retry(images: List[Image.Image], on_partial=None) -> ExtractionResponse:
    """Single non-streaming Gemini call with overload-aware retry.

    Returns the parsed ExtractionResponse. We deliberately avoid
    `generate_content_stream`: incremental JSON parsing is fragile, eats CPU
    on every chunk, and saves no wall-clock time because the network/model
    cost is dominated by the single request, not the streaming overhead.
    """
    max_retries = 5
    base_delay = 2

    def _emit(phase: str, data: dict):
        if on_partial:
            try:
                on_partial(phase, data)
            except Exception as e:
                print(f"[Stream] Gemini callback {phase} failed: {e}")

    # Resize once outside the retry loop. Coordinates returned are normalised
    # to 0-1000 so downstream remapping is unaffected by the resize.
    resized = [_resize_for_api(im, FIRST_PASS_MAX_LONG_SIDE) for im in images]

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            print(f"[Gemini] Sending {len(resized)} image(s) to {config.GEMINI_MODEL_NAME} "
                  f"(attempt {attempt+1}/{max_retries})...")
            t_call = time.time()
            _emit("status", {
                "message": "Đang gọi Gemini để phân tích bản vẽ...",
                "stage": "gemini_request_started",
                "attempt": attempt + 1,
            })
            with _gemini_sem:
                response = _get_gemini_client().models.generate_content(
                    model=config.GEMINI_MODEL_NAME,
                    contents=[PROMPT, *resized],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.0,
                        max_output_tokens=65536,
                        response_mime_type="application/json",
                        response_schema=ExtractionResponse,
                        thinking_config=types.ThinkingConfig(include_thoughts=False),
                    ),
                )
            print(f"[Gemini] Response received in {time.time()-t_call:.1f}s")

            # Log finish reason to detect truncation early.
            try:
                finish_reason = response.candidates[0].finish_reason if response.candidates else None
                if finish_reason and str(finish_reason) not in ("FinishReason.STOP", "STOP", "1"):
                    print(f"[Gemini] Non-STOP finish_reason: {finish_reason}")
            except Exception:
                pass

            parsed = getattr(response, "parsed", None)
            if parsed is not None:
                if isinstance(parsed, ExtractionResponse):
                    return parsed
                return ExtractionResponse.model_validate(parsed)

            text = (response.text or "").strip()
            if text:
                return ExtractionResponse(**json.loads(text))
            raise ValueError("Empty response from Gemini.")
        except Exception as e:
            last_err = e
            error_str = str(e)
            is_overloaded = "503" in error_str or "overloaded" in error_str or "UNAVAILABLE" in error_str
            is_rate_limited = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower()
            is_json_truncated = isinstance(e, json.JSONDecodeError)
            if (is_overloaded or is_json_truncated or is_rate_limited) and attempt < max_retries - 1:
                if is_rate_limited:
                    # 429: back off longer — quota windows are typically 60s
                    delay = 15 * (2 ** attempt) + random.uniform(0, 5)
                    reason = "Rate limited (429)"
                else:
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    reason = "Overloaded (503)" if is_overloaded else "Truncated JSON"
                print(f"[Gemini] {reason}. Retry in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"[Gemini] Error (attempt {attempt+1}/{max_retries}): {e}")
                raise

    # Defensive — loop only exits via return or raise above.
    raise last_err if last_err else RuntimeError("Gemini call failed without exception")


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crop_beam_images(foundation_list: list, images: List[Image.Image]):
    """Crop and attach base64 images for FW/FG beam items.

    With the caption-anchored locator producing tight section-relative bboxes,
    only light padding is required. We keep a safety expansion (the legacy
    label-anchored heuristic) for the rare case where an item's region was
    inherited from the first pass and never replaced.
    """
    for item in foundation_list:
        if not (item.classification == "FW/FG" and item.region):
            continue
        region = item.region
        page_idx = region.page - 1
        if not (0 <= page_idx < len(images)):
            continue
        target_image = images[page_idx]
        width, height = target_image.size

        raw_top    = (region.ymin / 1000) * height
        raw_bottom = (region.ymax / 1000) * height
        raw_left   = (region.xmin / 1000) * width
        raw_right  = (region.xmax / 1000) * width

        box_w = raw_right - raw_left
        box_h = raw_bottom - raw_top

        # Skip degenerate boxes
        if box_w <= 2 or box_h <= 2:
            continue

        # Padding strategy:
        # - FG: very light pad (locator already returns the 断面 cell precisely)
        # - FW: light pad on all sides; if locator failed and region is suspiciously
        #   short (legacy label-anchored bbox), apply a recovery upward expansion
        is_fg = item.type.upper().startswith("FG")
        if is_fg:
            pad_top    = box_h * 0.05
            pad_bottom = box_h * 0.05
            pad_x      = max(12, box_w * 0.03)
        else:
            # FW
            if box_h < height * 0.10:
                # Legacy fallback: locator never replaced first-pass bbox
                pad_top = height * 0.30
            else:
                pad_top = box_h * 0.05
            pad_bottom = box_h * 0.05
            pad_x = max(20, box_w * 0.05)

        top    = max(0,      int(raw_top    - pad_top))
        left   = max(0,      int(raw_left   - pad_x))
        bottom = min(height, int(raw_bottom + pad_bottom))
        right  = min(width,  int(raw_right  + pad_x))

        if right > left and bottom > top:
            cropped = target_image.crop((left, top, right, bottom))
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=88)
            item.image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")


def _resize_for_api(img: Image.Image, max_long_side: int) -> Image.Image:
    """Downscale image so its longest side ≤ max_long_side, preserving aspect ratio.

    Coordinates returned by Gemini are in 0-1000 scale normalised to the image
    size, so resizing does NOT affect coordinate accuracy after remapping.
    """
    w, h = img.size
    if max(w, h) <= max_long_side:
        return img
    scale = max_long_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


class _BeamPageEntry(_PydanticBase):
    type: str
    D: int
    top_elevation: int
    ymin: int = 0   # cross-section bbox in 0-1000 relative to the sent image (0 = not provided)
    xmin: int = 0
    ymax: int = 0
    xmax: int = 0

class _BeamPageResult(_PydanticBase):
    beams: List[_BeamPageEntry]


def _crop_beam_page_area(items: list, page_image: Image.Image):
    """Crop the page to the union of all beam regions, expanded generously.

    Using the union of all beam bounding boxes as a hint, we crop the portion
    of the page that contains the FW/FG drawing section.  This removes
    unrelated drawings (e.g. regular foundations on the opposite side of the
    sheet) while keeping enough context for the AI to read dimensions correctly.

    Falls back to the full page when no valid regions exist.

    Returns (cropped_image, crop_left, crop_top, crop_w, crop_h) so callers
    can remap Gemini's 0-1000 bbox coordinates back to the original page space.
    """
    width, height = page_image.size
    regions = [item.region for item in items if item.region]
    if not regions:
        return page_image, 0, 0, width, height  # full page, no offset

    # Union of all beam regions (0-1000 scale → pixels)
    xmin_f = min(r.xmin for r in regions) / 1000 * width
    xmax_f = max(r.xmax for r in regions) / 1000 * width
    ymin_f = min(r.ymin for r in regions) / 1000 * height
    ymax_f = max(r.ymax for r in regions) / 1000 * height

    # Expand generously: 20% horizontally, 25% vertically
    pad_x = (xmax_f - xmin_f) * 0.20 + 80   # at least 80px extra
    pad_y = (ymax_f - ymin_f) * 0.25 + 80

    left   = max(0,      int(xmin_f - pad_x))
    top    = max(0,      int(ymin_f - pad_y))
    right  = min(width,  int(xmax_f + pad_x))
    bottom = min(height, int(ymax_f + pad_y))

    # Safety: crop must be at least 30% of the page in each dimension
    # to avoid a tiny clip that misses dimension labels
    min_w = int(width  * 0.30)
    min_h = int(height * 0.30)
    if (right - left) < min_w:
        # Extend symmetrically
        cx = (left + right) // 2
        left  = max(0,     cx - min_w // 2)
        right = min(width, cx + min_w // 2)
    if (bottom - top) < min_h:
        cy = (top + bottom) // 2
        top    = max(0,      cy - min_h // 2)
        bottom = min(height, cy + min_h // 2)

    crop_w = right - left
    crop_h = bottom - top
    print(f"[BeamPage] Cropping to ({left},{top})–({right},{bottom}) "
          f"from {width}×{height} page")
    return page_image.crop((left, top, right, bottom)), left, top, crop_w, crop_h


def _run_beam_page_second_pass(foundation_list: list, images: List[Image.Image]):
    """Second-pass beam value extraction (D, top_elevation) — parallel per page.

    Groups FW/FG items by their first-pass page (only pages that actually have
    a known FW/FG drawing are scanned — we no longer add region-less items to
    every page, which used to N×M-fan-out the calls). Pages run concurrently
    via a small thread pool because each call spends almost all its time on
    Gemini I/O.
    """
    from collections import defaultdict
    import threading

    fw_fg_items = [item for item in foundation_list if item.classification == "FW/FG"]
    if not fw_fg_items:
        return

    by_page: dict = defaultdict(list)
    no_region_items = []
    for item in fw_fg_items:
        if item.region:
            by_page[item.region.page - 1].append(item)
        else:
            no_region_items.append(item)

    # Region-less items attach to every page that already has a beam call.
    # If no page has any FW/FG yet, fall back to scanning all pages once.
    if no_region_items:
        if not by_page:
            for pg in range(len(images)):
                by_page[pg] = []
        for item in no_region_items:
            for pg in by_page:
                if item not in by_page[pg]:
                    by_page[pg].append(item)
        print(f"[BeamPage] {len(no_region_items)} region-less item(s) "
              f"attached to pages {sorted(p+1 for p in by_page)}")

    discovery_lock = threading.Lock()

    def _process_page(page_idx: int, items: list):
        if page_idx >= len(images):
            return
        page_image = images[page_idx]
        orig_w, orig_h = page_image.size
        beam_list_str = ", ".join(item.type for item in items)
        prompt = BEAM_PAGE_PROMPT.format(beam_list=beam_list_str)

        send_image, crop_left, crop_top, crop_w, crop_h = _crop_beam_page_area(items, page_image)
        print(f"[BeamPage] Page {page_idx+1}: extracting {beam_list_str}, "
              f"image={send_image.size[0]}×{send_image.size[1]}...")
        t_call = time.time()
        try:
            with _gemini_sem:
                response = _get_gemini_client().models.generate_content(
                    model=config.GEMINI_MODEL_NAME,
                    contents=[prompt, _resize_for_api(send_image, max_long_side=1600)],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=_BeamPageResult,
                        thinking_config=types.ThinkingConfig(include_thoughts=False),
                    ),
                )
            print(f"[BeamPage] Page {page_idx+1} response in {time.time()-t_call:.1f}s")

            if not response.parsed:
                print(f"[BeamPage] No parsed response for page {page_idx+1}")
                return

            result_map = {entry.type: entry for entry in response.parsed.beams}

            def _remap_entry_bbox(entry) -> Optional[ItemRegion]:
                if entry.xmax <= entry.xmin or entry.ymax <= entry.ymin:
                    return None
                return ItemRegion(
                    page=page_idx + 1,
                    xmin=int((crop_left + (entry.xmin / 1000) * crop_w) / orig_w * 1000),
                    ymin=int((crop_top  + (entry.ymin / 1000) * crop_h) / orig_h * 1000),
                    xmax=int((crop_left + (entry.xmax / 1000) * crop_w) / orig_w * 1000),
                    ymax=int((crop_top  + (entry.ymax / 1000) * crop_h) / orig_h * 1000),
                )

            for item in items:
                entry = result_map.get(item.type)
                if not entry:
                    continue
                if entry.D > 0 and item.dimensions:
                    item.dimensions.D = str(entry.D)
                if entry.top_elevation != 0:
                    item.top_elevation = entry.top_elevation
                if item.region is None:
                    remapped = _remap_entry_bbox(entry)
                    if remapped:
                        item.region = remapped

            # Discover beam types the first pass missed (thread-safe append).
            for entry in response.parsed.beams:
                norm = _normalize_type(entry.type)
                if not re.match(r'^F[WG]\d', entry.type.strip(), re.IGNORECASE):
                    continue
                if entry.D == 0 and entry.top_elevation == 0:
                    continue
                with discovery_lock:
                    known_types = {_normalize_type(i.type) for i in foundation_list}
                    if norm in known_types:
                        continue
                    foundation_list.append(FoundationItem(
                        type=entry.type,
                        dimensions=Dimensions(Lx=0, Ly=0, D=str(entry.D)),
                        top_elevation=entry.top_elevation if entry.top_elevation != 0 else None,
                        rebar_x="",
                        rebar_y="",
                        remarks="",
                        classification="FW/FG",
                        region=_remap_entry_bbox(entry),
                        image_base64=None,
                    ))
                    print(f"[BeamPage] Discovered new beam '{entry.type}' on page {page_idx+1} "
                          f"(D={entry.D}, elev={entry.top_elevation})")

        except Exception as e:
            print(f"[BeamPage] Error on page {page_idx+1}: {e}")

    pages = [(pg, items) for pg, items in by_page.items() if pg < len(images)]
    if not pages:
        return
    with ThreadPoolExecutor(max_workers=max(1, min(4, len(pages)))) as ex:
        list(ex.map(lambda p: _process_page(*p), pages))


# ─────────────────────────────────────────────────────────────────────────────
# Pit second pass — focused high-resolution re-read of top_elevation / D
# ─────────────────────────────────────────────────────────────────────────────

class _PitPassResult(_PydanticBase):
    top_elevation: int  # negative mm (gap from ▽GL down to pit floor top); 0 if unreadable
    D: int              # positive mm (pit floor slab thickness); 0 if unreadable
    readable: bool


PIT_SECOND_PASS_PROMPT = """
You are reading ONE Japanese pit detail drawing (ピット詳細図), cropped from the
sheet and enlarged. Pit type: {pit_type}

Read TWO values FROM THIS DRAWING ONLY (ignore anything outside its border):

1) top_elevation — the vertical distance from the ▽GL (or ▽SGL) line straight DOWN
   to the TOP surface of the pit floor slab, in mm, returned as a NEGATIVE integer.

   ⚠️ CRITICAL — measure from ▽GL to the SLAB TOP (do NOT just take the largest
   number when several vertical dimensions are stacked):
     • TOP endpoint  = the ▽GL line (measure FROM ▽GL, not from a lower datum).
     • BOTTOM endpoint = the TOP surface of the bottom slab (where the fill meets the
       top of the concrete) — NOT the slab's underside.
   Two traps:
     (a) An inner chain may START below ▽GL (at a step/slab top); ignore it and use
         the one whose top touches ▽GL.   e.g. inner 1,400 vs 1,495-from-▽GL → -1495.
     (b) A dimension may run PAST the slab top down to the slab UNDERSIDE, i.e. it
         equals (depth-to-slab-top + slab thickness D). That is NOT top_elevation.
         e.g. 1,000 to the slab top vs 1,200 to the slab bottom (D=200) → -1000, NOT -1200.
   ⛔ Pick the dimension from ▽GL to the slab TOP — not the largest, not to the slab bottom.

   STEPPED PIT (deeper zone marked ※ on one side): use the explicit readable floor
   side (no ※), still applying the outer-chain rule above.

2) D — thickness of the pit floor slab. Read it in the SAME vertical dimension chain
   you used for top_elevation, IMMEDIATELY BELOW it: the top_elevation segment runs
   ▽GL → slab top; the very next segment down (slab top → slab bottom) IS D. So D sits
   right next to / just below where you measured ▽GL, on the same side.
   ⛔ Do NOT read D from a different part of the drawing (a step/段 dimension, the 土間
      コンクリート thickness, or a horizontal width). e.g. if a "150" step appears on the
      far side but the slab in the ▽GL chain is "200", D = 200 (NOT 150).
   ⛔ Exclude the 捨てコンクリート (t=50) leveling layer beneath the slab.
   Typical 150–300mm.

If a value genuinely cannot be read, set it to 0. Set readable=false only when the
vertical GL-to-floor dimension is marked ※ or the floor is at/above ▽GL.

Return JSON only, e.g.: {{"top_elevation": -1495, "D": 200, "readable": true}}
"""


def _crop_pit_area(region, page_image: Image.Image) -> Image.Image:
    """Crop the pit cell generously so the full dimension chains (incl. the outer
    vertical dimension on the right that reaches ▽GL) stay inside the image."""
    w, h = page_image.size
    x0 = (region.xmin / 1000) * w
    x1 = (region.xmax / 1000) * w
    y0 = (region.ymin / 1000) * h
    y1 = (region.ymax / 1000) * h

    pad_x = (x1 - x0) * 0.18 + 60
    pad_y = (y1 - y0) * 0.18 + 60
    left   = max(0, int(x0 - pad_x))
    right  = min(w, int(x1 + pad_x))
    top    = max(0, int(y0 - pad_y))
    bottom = min(h, int(y1 + pad_y))

    # Keep at least ~25% of the page in each dimension so a tight text-layer box
    # never clips the outer dimension chain.
    min_w, min_h = int(w * 0.25), int(h * 0.25)
    if right - left < min_w:
        cx = (left + right) // 2
        left, right = max(0, cx - min_w // 2), min(w, cx + min_w // 2)
    if bottom - top < min_h:
        cy = (top + bottom) // 2
        top, bottom = max(0, cy - min_h // 2), min(h, cy + min_h // 2)
    return page_image.crop((left, top, right, bottom))


def _run_pit_second_pass(pit_list: list, images: List[Image.Image]):
    """Re-read each pit's top_elevation / D from a tight, high-resolution crop.

    The first pass reads whole pages at low resolution, where two adjacent vertical
    dimensions (e.g. 1,400 vs 1,495) are easy to confuse. This pass crops the pit
    cell, enlarges it, and asks a focused question with the outer-chain rule, then
    overrides the first-pass values when the re-read succeeds.
    """
    pits = [p for p in pit_list
            if p.region and 0 <= (p.region.page - 1) < len(images)]
    if not pits:
        return

    def _process(pit):
        page_image = images[pit.region.page - 1]
        send = _resize_for_api(_crop_pit_area(pit.region, page_image),
                               max_long_side=1600)
        prompt = PIT_SECOND_PASS_PROMPT.format(pit_type=pit.type)
        try:
            with _gemini_sem:
                response = _get_gemini_client().models.generate_content(
                    model=config.GEMINI_MODEL_NAME,
                    contents=[prompt, send],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=_PitPassResult,
                        thinking_config=types.ThinkingConfig(include_thoughts=False),
                    ),
                )
            r = response.parsed
            if not r:
                return
            if r.readable and r.top_elevation != 0:
                old = pit.top_elevation
                pit.top_elevation = float(r.top_elevation)
                pit.readable = True
                if old != pit.top_elevation:
                    print(f"[PitPass] {pit.type}: top_elevation {old} → {pit.top_elevation}")
            if r.D and r.D > 0:
                pit.D = float(r.D)
        except Exception as e:
            print(f"[PitPass] Error for {pit.type}: {e}")

    print(f"[PitPass] Re-reading {len(pits)} pit(s) at high resolution...")
    with ThreadPoolExecutor(max_workers=max(1, min(4, len(pits)))) as ex:
        list(ex.map(_process, pits))


def _crop_table_image(table_region, images: List[Image.Image]) -> Optional[str]:
    """Crop and return base64 image of the foundation table."""
    if not table_region:
        return None
    region = table_region
    page_idx = region.page - 1
    if 0 <= page_idx < len(images):
        target_image = images[page_idx]
        width, height = target_image.size
        margin = 30
        top    = max(0,      (region.ymin / 1000) * height - margin)
        left   = max(0,      (region.xmin / 1000) * width  - margin)
        bottom = min(height, (region.ymax / 1000) * height + margin)
        right  = min(width,  (region.xmax / 1000) * width  + margin)
        if right > left and bottom > top:
            cropped = target_image.crop((left, top, right, bottom))
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    return None


def _generate_slab_overview(slab_list: list, images: List[Image.Image]) -> Optional[str]:
    """Generate a highlighted overview image covering all detected slabs."""
    slabs_by_page: dict = {}
    for item in slab_list:
        if item.region:
            slabs_by_page.setdefault(item.region.page, []).append(item)

    if not slabs_by_page:
        return None

    target_page_num = max(slabs_by_page, key=lambda k: len(slabs_by_page[k]))
    page_idx = target_page_num - 1
    if page_idx < 0 or page_idx >= len(images):
        return None

    target_image = images[page_idx].copy()
    draw = ImageDraw.Draw(target_image)
    width, height = target_image.size
    ux, uy = float("inf"), float("inf")
    ux2, uy2 = float("-inf"), float("-inf")

    for slab in slabs_by_page[target_page_num]:
        if slab.region:
            bl = (slab.region.xmin / 1000) * width
            bt = (slab.region.ymin / 1000) * height
            br = (slab.region.xmax / 1000) * width
            bb = (slab.region.ymax / 1000) * height
            draw.rectangle([bl, bt, br, bb], outline="red", width=5)
            ux, uy = min(ux, bl), min(uy, bt)
            ux2, uy2 = max(ux2, br), max(uy2, bb)

    om = 100
    c_left   = max(0,      ux  - om)
    c_top    = max(0,      uy  - om)
    c_right  = min(width,  ux2 + om)
    c_bottom = min(height, uy2 + om)

    if c_right > c_left and c_bottom > c_top:
        cropped = target_image.crop((c_left, c_top, c_right, c_bottom))
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    return None


def _annotate_gl_pages(oval_gl_list: list, images: List[Image.Image]) -> List[str]:
    """Draw GL marker rectangles on page copies and return as base64 list."""
    annotated = [img.copy() for img in images]
    for item in oval_gl_list:
        if item.region:
            region = item.region
            page_idx = region.page - 1
            if 0 <= page_idx < len(annotated):
                target = annotated[page_idx]
                draw = ImageDraw.Draw(target)
                w, h = target.size
                left   = (region.xmin / 1000) * w
                top    = (region.ymin / 1000) * h
                right  = (region.xmax / 1000) * w
                bottom = (region.ymax / 1000) * h
                draw.rectangle([left, top, right, bottom], outline="red", width=5)

    result = []
    for img in annotated:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return result


def _build_regular_floors(oval_gl_list: list) -> list:
    """Group GL markers by elevation to create RegularFloor entries."""
    from models import RegularFloor
    elevation_map: dict = defaultdict(list)
    for gl_item in oval_gl_list:
        elevation_map[gl_item.text].append(gl_item.region)

    floors = []
    for elevation, regions in elevation_map.items():
        floors.append(RegularFloor(elevation=elevation, count=len(regions), regions=regions))
    return floors


def _generate_floor_overview(oval_gl_list: list, images: List[Image.Image]) -> Optional[str]:
    """Generate a cropped overview image of all GL markers on the main floor plan."""
    gl_by_page: dict = {}
    for item in oval_gl_list:
        if item.region:
            gl_by_page.setdefault(item.region.page, []).append(item)

    if not gl_by_page:
        return None

    target_page_num = max(gl_by_page, key=lambda k: len(gl_by_page[k]))
    page_idx = target_page_num - 1
    if page_idx < 0 or page_idx >= len(images):
        return None

    target_image = images[page_idx].copy()
    draw = ImageDraw.Draw(target_image)
    width, height = target_image.size
    ux, uy = float("inf"), float("inf")
    ux2, uy2 = float("-inf"), float("-inf")

    for gl_item in gl_by_page[target_page_num]:
        if gl_item.region:
            print(f"DEBUG GL Marker: {gl_item.text}")
            print(f"  Raw coords (0-1000): xmin={gl_item.region.xmin}, ymin={gl_item.region.ymin}, xmax={gl_item.region.xmax}, ymax={gl_item.region.ymax}")
            print(f"  Image size: {width}x{height}")

            bl = (gl_item.region.xmin / 1000) * width
            bt = (gl_item.region.ymin / 1000) * height
            br = (gl_item.region.xmax / 1000) * width
            bb = (gl_item.region.ymax / 1000) * height

            print(f"  Pixel coords: left={bl:.0f}, top={bt:.0f}, right={br:.0f}, bottom={bb:.0f}")
            draw.rectangle([bl, bt, br, bb], outline="red", width=5)
            ux, uy = min(ux, bl), min(uy, bt)
            ux2, uy2 = max(ux2, br), max(uy2, bb)

    om = 100
    c_left   = max(0,      ux  - om)
    c_top    = max(0,      uy  - om)
    c_right  = min(width,  ux2 + om)
    c_bottom = min(height, uy2 + om)

    if c_right > c_left and c_bottom > c_top:
        cropped = target_image.crop((c_left, c_top, c_right, c_bottom))
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Table image crop from pdfplumber bbox
# ─────────────────────────────────────────────────────────────────────────────

def _crop_table_from_plumber(result: TableExtractionResult,
                             images: List[Image.Image]) -> Optional[str]:
    """Crop the foundation table from the rendered page image using pdfplumber's bbox.

    pdfplumber's bbox uses top-left origin in PDF points; the rendered PIL images
    use top-left origin in pixels. We scale by (img_pixels / pdf_points).
    This gives a reliable crop regardless of what Gemini said the table_region was.
    """
    if not result.found or not result.bbox or result.page_num < 1:
        return None

    page_idx = result.page_num - 1
    if page_idx >= len(images):
        return None

    img = images[page_idx]
    img_w, img_h = img.size

    if result.pdf_width <= 0 or result.pdf_height <= 0:
        return None

    scale_x = img_w / result.pdf_width
    scale_y = img_h / result.pdf_height

    x0, top, x1, bottom = result.bbox
    margin = 20  # px padding around the table

    left   = max(0,     x0     * scale_x - margin)
    top_px = max(0,     top    * scale_y - margin)
    right  = min(img_w, x1     * scale_x + margin)
    bot_px = min(img_h, bottom * scale_y + margin)

    if right <= left or bot_px <= top_px:
        return None

    cropped = img.crop((left, top_px, right, bot_px))
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_plumber_preview_payload(plumber_result: TableExtractionResult,
                                   images: List[Image.Image]) -> dict:
    """Build an early partial payload from pdfplumber rows only.

    This lets the UI render the table immediately, before Gemini finishes
    project metadata, elevations, beam passes, and image crops.
    """
    foundations = list(plumber_result.items or [])
    foundations.sort(key=_foundation_sort_key)
    table_image_base64 = _crop_table_from_plumber(plumber_result, images)

    preview = ExtractionResponse(
        project_info=ProjectInfo(
            project_name="Đang trích xuất...",
            drawing_date="N/A",
            drawing_scale="N/A",
        ),
        foundation_list=foundations,
        table_image_base64=table_image_base64,
    )
    payload = preview.model_dump(mode="json")
    payload["is_partial"] = True
    payload["partial_stage"] = "pdfplumber"
    payload["partial_message"] = "Bảng tạm từ pdfplumber. Gemini đang bổ sung cao độ, dầm và ảnh crop."
    payload["excel_ready"] = False
    payload["images_pending"] = True
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# pdfplumber / Gemini result merge
# ─────────────────────────────────────────────────────────────────────────────

def _strip_beam_variant_suffix(t: str) -> str:
    """Remove Japanese variant suffixes from beam type names.

    "FW1 (一般部)" → "FW1"
    "FW1（間柱部）" → "FW1"
    "FW1 一般部"   → "FW1"
    "FW2"          → "FW2"  (unchanged)
    """
    # Strip full-width or half-width parenthetical suffixes
    cleaned = re.sub(r'[\s　]*[（(][^)）]+[)）]', '', t)
    # Strip bare Japanese suffix words after a space (一般部, 間柱部, 一般, 柱部, etc.)
    cleaned = re.sub(r'[\s　]+[぀-ヿ一-鿿㐀-䶿]+.*$', '', cleaned)
    return cleaned.strip()


def _deduplicate_beams(foundation_list: list) -> list:
    """Normalize beam type names and deduplicate variant cross-sections.

    When Gemini returns both "FW1 (一般部)" and "FW1 (間柱部)", these are the
    same physical beam "FW1" shown in two contexts.  Keep only the first
    occurrence (一般部 is preferred because it appears first in the drawing)
    and set its type to the stripped base code.
    """
    seen_beam_keys: dict = {}
    result: list = []

    for item in foundation_list:
        if item.classification != "FW/FG":
            result.append(item)
            continue

        base_type = _strip_beam_variant_suffix(item.type)
        # Normalize for comparison (upper-case, half-width, no spaces)
        key = re.sub(r'\s+', '', base_type.upper())
        for ch_i, ch in enumerate(key):
            cp = ord(ch)
            if 0xFF01 <= cp <= 0xFF5E:
                key = key[:ch_i] + chr(cp - 0xFEE0) + key[ch_i+1:]

        if key not in seen_beam_keys:
            item.type = base_type  # overwrite with clean name
            seen_beam_keys[key] = item
            result.append(item)
        else:
            print(f"[BeamDedup] Dropping duplicate beam '{item.type}' (already have '{base_type}')")

    return result


def _normalize_type(t: str) -> str:
    """Normalize a foundation type string for comparison (strip, uppercase, half-width)."""
    s = t.strip().upper()
    result = []
    for ch in s:
        cp = ord(ch)
        if 0xFF01 <= cp <= 0xFF5E:
            result.append(chr(cp - 0xFEE0))
        else:
            result.append(ch)
    return "".join(result).replace(" ", "").replace("　", "")


def _merge_with_plumber_data(gemini_list: list, plumber_list: list) -> list:
    """Merge pdfplumber's foundation rows with Gemini's foundation list.

    pdfplumber provides reliable tabular fields (Lx, Ly, D, rebar, remarks, classification).
    Gemini provides visual fields (region, image_base64) and top_elevation from cross-sections.

    Strategy:
      - For types present in both: use pdfplumber's tabular data + Gemini's visual data.
      - For types only in plumber: include them without region/image (Gemini missed them).
      - For types only in Gemini (beams, or types plumber missed): keep Gemini's data.
    """
    if not plumber_list:
        return gemini_list

    gemini_map = {_normalize_type(item.type): item for item in gemini_list}
    plumber_types = {_normalize_type(item.type) for item in plumber_list}

    result = []
    for plumber_item in plumber_list:
        norm = _normalize_type(plumber_item.type)
        g_item = gemini_map.get(norm)
        if g_item:
            merged = plumber_item.model_copy(deep=True)
            merged.region = g_item.region
            merged.image_base64 = g_item.image_base64
            # Keep Gemini's cross-section top_elevation when plumber has none
            if merged.top_elevation is None and g_item.top_elevation is not None:
                merged.top_elevation = g_item.top_elevation
            # Don't let pdfplumber's blank rebar erase a value Gemini read from the
            # ベース筋 arrows column: prefer plumber, fall back to Gemini when blank.
            if not (merged.rebar_x or "").strip() and (g_item.rebar_x or "").strip():
                merged.rebar_x = g_item.rebar_x
            if not (merged.rebar_y or "").strip() and (g_item.rebar_y or "").strip():
                merged.rebar_y = g_item.rebar_y
            result.append(merged)
        else:
            result.append(plumber_item)

    # Append Gemini items not found in pdfplumber (beams, or missed types)
    for g_item in gemini_list:
        if _normalize_type(g_item.type) not in plumber_types:
            result.append(g_item)

    print(f"[Merge] plumber={len(plumber_list)} + gemini-only={len(result)-len(plumber_list)} → total={len(result)}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sort and row-image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _foundation_sort_key(item) -> tuple:
    """Natural sort key: F1 < F1A < F1B < F2 < F10 < FW1 < FG1.

    Combined rows like "F4, F4A" sort by the first listed type.
    """
    first = re.split(r'[,、，\s]+', item.type.strip())[0].strip().upper()
    # Normalize full-width to ASCII
    norm = []
    for ch in first:
        cp = ord(ch)
        norm.append(chr(cp - 0xFEE0) if 0xFF01 <= cp <= 0xFF5E else ch)
    first = "".join(norm)

    if re.match(r'^FG', first):
        cat = 2
    elif re.match(r'^FW', first):
        cat = 1
    else:
        cat = 0

    m = re.match(r'^F(?:[WG])?(\d+)([A-Za-z]*)', first)
    if m:
        return (cat, int(m.group(1)), m.group(2).upper())
    return (cat, 9999, first)


def _crop_foundation_row_images(foundation_list: list, images: List[Image.Image]):
    """Crop the cross-section drawing from the page for each regular F-type item.

    With the text-layer locator, `item.region` is anchored on the drawing
    title (e.g., "F1 基礎断面"), so the bbox is already tight and accurate.
    Light, symmetric padding is applied to ensure the title and any margin
    dimension labels are fully visible.

    Skips FW/FG beams (they go through `_crop_beam_images`).
    """
    for item in foundation_list:
        if item.classification == "FW/FG" or item.image_base64 or not item.region:
            continue

        region = item.region
        page_idx = region.page - 1
        if not (0 <= page_idx < len(images)):
            continue

        target = images[page_idx]
        w, h = target.size

        raw_top    = (region.ymin / 1000) * h
        raw_bottom = (region.ymax / 1000) * h
        raw_left   = (region.xmin / 1000) * w
        raw_right  = (region.xmax / 1000) * w
        box_h      = raw_bottom - raw_top

        # Skip degenerate boxes
        if (raw_right - raw_left) <= 4 or box_h <= 4:
            continue

        # Light, symmetric padding (~2% on each side). Slight upward bias to
        # always keep the title fully visible in case ymin grazes the text.
        pad_y_top    = max(8, h * 0.01)
        pad_y_bottom = max(8, h * 0.01)
        pad_x        = max(12, w * 0.01)

        top    = max(0, int(raw_top    - pad_y_top))
        bottom = min(h, int(raw_bottom + pad_y_bottom))
        left   = max(0, int(raw_left   - pad_x))
        right  = min(w, int(raw_right  + pad_x))

        if right > left and bottom > top:
            cropped = target.crop((left, top, right, bottom))
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=90)
            item.image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")


def _normalize_pit_type(t: str) -> str:
    """Normalize a pit type name for dedup comparison.

    Upper-case, half-width, and strip spaces + comma variants so that
    "ES1, ES2ピット" / "ES1、ES2ピット" / "ES1，ES2ピット" all compare equal.
    """
    s = (t or "").strip().upper()
    out = []
    for ch in s:
        cp = ord(ch)
        out.append(chr(cp - 0xFEE0) if 0xFF01 <= cp <= 0xFF5E else ch)
    norm = "".join(out)
    for sep in (" ", "　", ",", "、", "，"):
        norm = norm.replace(sep, "")
    return norm


def _apply_pit_text_elevations(pit_list: list, elev_map: dict) -> list:
    """Merge floor-plan text-layer pit elevations into the pit list.

    For each pit named in `elev_map` (from "…詳細図参照(底盤天端…GL-XXX)"): set its
    top_elevation — authoritative, since 底盤天端 IS the slab-top elevation — and
    mark readable=True. Pits that Gemini missed entirely are appended as new
    entries (D filled later by the second pass / left null). Idempotent: safe to
    call before the pit second pass (to detect/seed) and again after it (to
    re-assert the authoritative value over any vision re-read).
    """
    for name, elev in elev_map.items():
        cs = _pit_code_set(name)
        # Match by code-set so an individual floor-plan name (e.g. "EV2ピット") updates
        # the combined "EV1, EV2ピット" instead of being appended as a duplicate.
        existing = next((p for p in pit_list if _pit_code_set(p.type) & cs), None)
        if existing is not None:
            existing.top_elevation = float(elev)
            existing.readable = True
        else:
            pit_list.append(PitHoleItem(
                type=name, top_elevation=float(elev), D=None,
                readable=True, region=None, image_base64=None,
            ))
            print(f"[PitText] added pit '{name}' (GL{elev:+.0f}) missing from vision pass")
    return pit_list


_PIT_COMBINED_RE = re.compile(r'^(.+?)(ピット|ﾋ゚ｯﾄ|水槽|側溝)\s*([①-⑳0-9]*)\s*$')


def _pit_code_set(name: str) -> set:
    """The set of individual pit codes a name covers, for dedup comparison.

    "EV1, EV2ピット" → {"EV1ピット", "EV2ピット"}; "EV1ピット" → {"EV1ピット"};
    "消火水槽" → {"消火水槽"}. Lets a combined entry and its individual entries be
    recognised as the SAME pit(s) without rewriting the displayed name.
    """
    m = _PIT_COMBINED_RE.match((name or "").strip())
    if m:
        prefix, suffix, tail = m.group(1), m.group(2), m.group(3)
        codes = [c.strip() for c in re.split(r'[,、，\s]+', prefix) if c.strip()]
        if len(codes) > 1:
            return {_normalize_pit_type(f"{c}{suffix}{tail}") for c in codes}
    return {_normalize_pit_type(name)}


def _merge_subsumed_pits(pit_list: list) -> list:
    """Collapse pit entries that refer to the same pit(s), KEEPING the name as drawn.

    The vision pass often emits BOTH the combined detail-title "EV1, EV2ピット" AND the
    individual "EV1ピット"/"EV2ピット" (the floor-plan text layer also uses individual
    names). These are the same pits, so we group entries whose code-sets overlap and
    keep ONE survivor — the entry with the LARGEST code-set, i.e. the combined name
    exactly as the drawing writes it (no name splitting). The survivor back-fills any
    missing top_elevation / D / region / image from the entries merged into it.
    """
    groups: list = []   # each: [code_set, [members…]]
    for pit in pit_list:
        cs = _pit_code_set(pit.type)
        hit = next((g for g in groups if g[0] & cs), None)
        if hit:
            hit[0] |= cs
            hit[1].append(pit)
        else:
            groups.append([set(cs), [pit]])

    result: list = []
    for _cset, members in groups:
        survivor = max(members, key=lambda p: len(_pit_code_set(p.type)))
        for p in members:
            if p is survivor:
                continue
            if p.readable and not survivor.readable:
                survivor.readable = True
            if survivor.top_elevation is None and p.top_elevation is not None:
                survivor.top_elevation = p.top_elevation
            if survivor.D is None and p.D is not None:
                survivor.D = p.D
            if survivor.region is None and p.region is not None:
                survivor.region = p.region
            if not survivor.image_base64 and p.image_base64:
                survivor.image_base64 = p.image_base64
        result.append(survivor)

    removed = len(pit_list) - len(result)
    if removed:
        print(f"[PitDedup] merged {removed} duplicate/subsumed pit(s): "
              f"{len(pit_list)} → {len(result)}")
    return result


def _deduplicate_pits(pit_list: list) -> list:
    """Collapse duplicate pit entries that share the same type name.

    A pit type commonly appears in two drawings on the sheet (詳細図 + 断面図, or
    A-A / B-B sections), and Gemini may emit one entry per drawing. Keep ONE entry
    per normalized type name, merging fields so the survivor carries the best data:
    prefer readable=true, and back-fill top_elevation / D / region / image from any
    duplicate when the first occurrence is missing them.
    """
    seen: dict = {}      # normalized type -> index into result
    result: list = []
    for pit in pit_list:
        if not pit.type:
            result.append(pit)
            continue
        key = _normalize_pit_type(pit.type)
        if key not in seen:
            seen[key] = len(result)
            result.append(pit)
            continue
        existing = result[seen[key]]
        if pit.readable and not existing.readable:
            existing.readable = True
        if existing.top_elevation is None and pit.top_elevation is not None:
            existing.top_elevation = pit.top_elevation
        if existing.D is None and pit.D is not None:
            existing.D = pit.D
        if existing.region is None and pit.region is not None:
            existing.region = pit.region
        if not existing.image_base64 and pit.image_base64:
            existing.image_base64 = pit.image_base64

    removed = len(pit_list) - len(result)
    if removed:
        print(f"[PitDedup] removed {removed} duplicate pit(s): "
              f"{len(pit_list)} → {len(result)}")
    return result


def _crop_pit_images(pit_list: list, images: List[Image.Image]):
    """Crop and attach base64 images for pit hole items."""
    for item in pit_list:
        if not item.region or item.image_base64:
            continue
        region = item.region
        page_idx = region.page - 1
        if not (0 <= page_idx < len(images)):
            continue
        target = images[page_idx]
        w, h = target.size

        raw_top    = (region.ymin / 1000) * h
        raw_bottom = (region.ymax / 1000) * h
        raw_left   = (region.xmin / 1000) * w
        raw_right  = (region.xmax / 1000) * w

        if (raw_right - raw_left) <= 4 or (raw_bottom - raw_top) <= 4:
            continue

        pad = max(10, h * 0.01)
        top    = max(0, int(raw_top    - pad))
        bottom = min(h, int(raw_bottom + pad))
        left   = max(0, int(raw_left   - pad))
        right  = min(w, int(raw_right  + pad))

        if right > left and bottom > top:
            cropped = target.crop((left, top, right, bottom))
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=90)
            item.image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Text-layer drawing locator (replaces Gemini caption/section calls)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_text_layer_locator(
    foundation_list: list,
    pdf_bytes: bytes | None = None,
    text_layer_scan: Optional[TextLayerScanResult] = None,
) -> dict:
    """Use the PDF text layer (pypdfium2) to deterministically locate drawings.

    Sets `item.region` for:
      • Regular F-type foundations          (from "F? 基礎断面" / "F? 基礎平面" titles)
      • FW beams                            (from "FW? (一般部)" / "FW?" labels)
      • FG beams                            (from "FG?" column headers + caption)

    Returns a `text_layer_summary` dict the caller can use to decide whether
    to skip subsequent Gemini-based locator passes.
    """
    summary = {
        "foundation_found": 0,
        "fw_found": 0,
        "fg_found": 0,
        "fw_caption_pages": [],
        "fg_caption_pages": [],
        # type(upper) → ItemRegion for every foundation cross-section title found.
        # Returned so the caller can re-apply these to plumber-only types that
        # Gemini omitted from its list (they aren't present here to receive one).
        "foundation_regions": {},
    }
    if not pdf_bytes or not foundation_list:
        if not foundation_list:
            return summary
        if text_layer_scan is None:
            return summary

    page_scans = text_layer_scan.page_scans if text_layer_scan else []

    # ── Regular foundations ──────────────────────────────────────────────────
    if text_layer_scan is not None:
        f_regions = text_layer_scan.foundation_regions or {}
    elif pdf_bytes:
        try:
            f_regions = find_foundation_drawing_regions(pdf_bytes)
        except Exception as e:
            print(f"[TextLocator] Foundation title scan failed: {e}")
            f_regions = {}
    else:
        f_regions = {}
    summary["foundation_regions"] = f_regions

    for item in foundation_list:
        if item.classification == "FW/FG":
            continue
        key = item.type.upper()
        if key in f_regions:
            item.region = f_regions[key]
            summary["foundation_found"] += 1

    # ── FW/FG section captions (with captured type info) ────────────────────
    if text_layer_scan is not None:
        captions = text_layer_scan.beam_captions or {}
    elif pdf_bytes:
        try:
            captions = find_beam_section_captions(pdf_bytes)
        except Exception as e:
            print(f"[TextLocator] Caption scan failed: {e}")
            captions = {}
    else:
        captions = {}

    for page_idx, caps in captions.items():
        if "fw" in caps:
            summary["fw_caption_pages"].append(page_idx + 1)
        if "fg" in caps:
            summary["fg_caption_pages"].append(page_idx + 1)

    # Build {FW type → (page_idx, caption_region)} for type-specific captions
    type_specific_fw: dict = {}
    for page_idx, caps in captions.items():
        fw_info = caps.get("fw")
        if not fw_info:
            continue
        for t in fw_info.get("types", []):
            type_specific_fw[t.upper()] = (page_idx, fw_info["region"])

    # ── Individual FW/FG labels → derived panel/column regions ──────────────
    fw_pages = {pg for pg, caps in captions.items() if "fw" in caps}
    fg_pages = {pg for pg, caps in captions.items() if "fg" in caps}
    beam_types_all = [it.type for it in foundation_list if it.classification == "FW/FG"]
    fw_types = [t for t in beam_types_all if t.upper().startswith("FW")]
    fg_types = [t for t in beam_types_all if t.upper().startswith("FG")]

    fw_types_needing_labels = [t for t in fw_types if t.upper() not in type_specific_fw]

    labels: dict = {}
    if page_scans:
        if fw_types_needing_labels and fw_pages:
            try:
                labels.update(find_beam_labels_from_pages(
                    page_scans,
                    fw_types_needing_labels,
                    prefer_pages=fw_pages,
                ))
            except Exception as e:
                print(f"[TextLocator] FW label scan failed: {e}")
        if fg_types and fg_pages:
            try:
                labels.update(find_beam_labels_from_pages(
                    page_scans,
                    fg_types,
                    prefer_pages=fg_pages,
                ))
            except Exception as e:
                print(f"[TextLocator] FG label scan failed: {e}")
    elif pdf_bytes:
        if fw_types_needing_labels and fw_pages:
            try:
                labels.update(find_beam_labels(pdf_bytes, fw_types_needing_labels, prefer_pages=fw_pages))
            except Exception as e:
                print(f"[TextLocator] FW label scan failed: {e}")
        if fg_types and fg_pages:
            try:
                labels.update(find_beam_labels(pdf_bytes, fg_types, prefer_pages=fg_pages))
            except Exception as e:
                print(f"[TextLocator] FG label scan failed: {e}")

    for item in foundation_list:
        if item.classification != "FW/FG":
            continue
        key = item.type.upper()

        if key in type_specific_fw:
            _pg, cap_region = type_specific_fw[key]
            item.region = derive_fw_panel_region_from_caption(cap_region)
            summary["fw_found"] += 1
            continue

        label_region = labels.get(key)
        if not label_region:
            continue
        page_idx = label_region.page - 1
        page_caps = captions.get(page_idx, {})

        if key.startswith("FW"):
            fw_info = page_caps.get("fw")
            fw_cap_region = fw_info["region"] if fw_info else None
            item.region = derive_fw_panel_region(label_region, fw_cap_region)
            summary["fw_found"] += 1
        elif key.startswith("FG"):
            fg_info = page_caps.get("fg")
            if fg_info:
                item.region = derive_fg_column_region(label_region, fg_info["region"])
                summary["fg_found"] += 1

    print(f"[TextLocator] {summary['foundation_found']} foundations, "
          f"{summary['fw_found']} FW, {summary['fg_found']} FG located from text layer")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_data_from_images(images: List[Image.Image], pdf_text: str = "",
                             pdf_bytes: bytes = None,
                             on_partial=None) -> ExtractionResponse:
    """Extract foundation data from page images using Gemini + deterministic post-processing.

    Pipeline (stage events only — no incremental content streaming):
      1. pdfplumber table  + oval GL detection (text layer)         — ~2-3s
         → partial_table_data emitted as soon as plumber rows are ready.
      2. Gemini first pass (single non-streaming call, resized images) — ~10-18s
      3. Text-layer locator + plumber merge + elevation override     — ~0.5s
      4. Second beam pass (parallel per page) overlapped with crops  — ~6-10s
      5. partial_data emitted (Excel-ready).
      6. All crops finalised, return → complete emitted by caller.
    """
    if not config.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set.")

    def _emit(phase: str, data: dict):
        if on_partial:
            try:
                on_partial(phase, data)
            except Exception as e:
                print(f"[Stream] on_partial({phase}) callback raised: {e}")

    # ── Phase 1a (text layer) overlapped with Phase 1b (Gemini) ─────────────
    # Text-layer work (table, oval GL, line caches, captions, foundation title
    # regions) is CPU-bound pure Python and feeds only post-Gemini merge steps.
    # Run it in a separate process so it overlaps the Gemini request without
    # fighting for the GIL in the main process.
    from models import OvalGLItem
    t_phase1 = time.time()
    text_layer_scan = TextLayerScanResult()
    _preview_state = {"emitted": False}

    def _emit_table_preview(tr: TableExtractionResult):
        if _preview_state["emitted"]:
            return
        _preview_state["emitted"] = True
        elapsed = time.time() - t_phase1
        print(f"[Phase1a] table {len(tr.items)} item(s) ({elapsed:.2f}s, overlapped)")
        _emit("status", {
            "message": f"Đã đọc bảng móng bằng pdfplumber: {len(tr.items)} dòng ({elapsed:.1f}s)",
            "stage": "table_extract_done",
        })
        if tr.items:
            _emit("partial_table_data", _build_plumber_preview_payload(tr, images))

    def _on_phase1a_done(fut):
        try:
            scan = fut.result()
        except Exception as e:
            print(f"[Phase1a] worker failed: {e}")
            return
        print(f"[OvalGL] text-layer detected {len(scan.ovals)} oval(s)")
        _emit_table_preview(scan.table_result)

    phase1a_future = None
    phase1a_pool = None
    if pdf_bytes:
        try:
            phase1a_pool = ProcessPoolExecutor(max_workers=1)
            phase1a_future = phase1a_pool.submit(scan_text_layer, pdf_bytes)
            phase1a_future.add_done_callback(_on_phase1a_done)
        except Exception as e:
            print(f"[Phase1a] process pool unavailable ({e}); will run inline.")
            phase1a_future = None
            if phase1a_pool is not None:
                phase1a_pool.shutdown(wait=False)
                phase1a_pool = None

    # ── Phase 1b: Gemini first pass (runs concurrently with the process) ────
    t_gemini = time.time()
    parsed_response = _call_gemini_with_retry(images, on_partial=on_partial)
    print(f"[Gemini] First pass total: {time.time()-t_gemini:.1f}s")

    # ── Collect Phase 1a (already done in the overlap, or recover inline) ───
    if phase1a_future is not None:
        try:
            text_layer_scan = phase1a_future.result()
        except Exception as e:
            print(f"[Phase1a] process result error ({e}); recomputing inline.")
            try:
                text_layer_scan = scan_text_layer(pdf_bytes)
                _emit_table_preview(text_layer_scan.table_result)
            except Exception as e2:
                print(f"[Phase1a] inline fallback failed: {e2}")
        finally:
            if phase1a_pool is not None:
                phase1a_pool.shutdown(wait=False)
    elif pdf_bytes:
        # Pool never started — run inline (sequential fallback).
        try:
            text_layer_scan = scan_text_layer(pdf_bytes)
            _emit_table_preview(text_layer_scan.table_result)
        except Exception as e:
            print(f"[Phase1a] inline run failed: {e}")
    plumber_result = text_layer_scan.table_result
    ovals = text_layer_scan.ovals
    plumber_list = plumber_result.items

    # Sanitise first-pass GL list and replace with text-layer detections.
    if parsed_response.oval_gl_list:
        parsed_response.oval_gl_list = [
            m for m in parsed_response.oval_gl_list
            if m.text.upper().startswith("GL")
        ]
    if ovals:
        parsed_response.oval_gl_list = [
            OvalGLItem(text=o["text"], region=o["region"]) for o in ovals
        ]

    # ── Deduplicate beam variants (FW1 一般部 + FW1 間柱部 → one FW1) ─────────
    if parsed_response.foundation_list:
        parsed_response.foundation_list = _deduplicate_beams(parsed_response.foundation_list)

    # ── Deduplicate pits (詳細図 + 断面図 / A-A + B-B → one entry per type) ───
    if parsed_response.pit_list:
        # Collapse combined + individual entries for the same pit(s), keeping the name
        # exactly as drawn ("EV1, EV2ピット" stays combined — no name splitting).
        parsed_response.pit_list = _merge_subsumed_pits(parsed_response.pit_list)

    # ── Pit detection + authoritative elevation from the floor-plan text layer ──
    # "…詳細図参照(底盤天端…GL-XXX)" gives the exact slab-top elevation and reliably
    # surfaces pits the vision pass missed. Run BEFORE the Phase-2 pool so the
    # text-layer region locator and second pass include any pit added here.
    pit_elev_map = parse_pit_elevations(pdf_text) if pdf_text else {}
    if pit_elev_map:
        parsed_response.pit_list = _apply_pit_text_elevations(
            parsed_response.pit_list, pit_elev_map
        )

    # ── Text-layer locator (fast, deterministic, no API) ────────────────────
    text_layer_summary: dict = {}
    if parsed_response.foundation_list and (pdf_bytes or text_layer_scan.page_scans):
        t_txt = time.time()
        text_layer_summary = _apply_text_layer_locator(
            parsed_response.foundation_list,
            pdf_bytes=pdf_bytes,
            text_layer_scan=text_layer_scan,
        )
        print(f"[TextLocator] done in {time.time()-t_txt:.2f}s")

    # ── Phase 2: second beam pass IN PARALLEL with image crops ──────────────
    # The second pass (Gemini reading FW/FG dimension chains) and the CPU/IO
    # crops are independent — overlap them in a single thread pool.
    _emit("status", {
        "message": "Đang bổ sung giá trị dầm và cắt ảnh chi tiết...",
        "stage": "second_pass_and_crops_started",
    })
    t_phase2 = time.time()
    overlap_futures: dict = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        if parsed_response.foundation_list:
            overlap_futures["beam_pass"] = pool.submit(
                _run_beam_page_second_pass,
                parsed_response.foundation_list, images,
            )
        if parsed_response.slab_list:
            overlap_futures["slab_overview"] = pool.submit(
                _generate_slab_overview, parsed_response.slab_list, images
            )
        if parsed_response.oval_gl_list:
            overlap_futures["annotated"] = pool.submit(
                _annotate_gl_pages, parsed_response.oval_gl_list, images
            )
            overlap_futures["floor_overview"] = pool.submit(
                _generate_floor_overview, parsed_response.oval_gl_list, images
            )
        # pdfplumber-derived table crop is independent of Gemini results.
        if plumber_list:
            overlap_futures["table_plumber"] = pool.submit(
                _crop_table_from_plumber, plumber_result, images
            )
        elif parsed_response.table_region:
            overlap_futures["table"] = pool.submit(
                _crop_table_image, parsed_response.table_region, images
            )
        # Pit drawing locator (text layer) + focused high-res re-read.
        # Both depend only on pdf_bytes + pit type names — NOT on the beam pass,
        # the plumber merge, or any crop — so run the whole pit Gemini round-trip
        # concurrently with the beam pass instead of serially after the pool.
        # (_gemini_sem still caps total concurrent Gemini calls.)
        def _pit_locate_and_reread():
            applied = 0
            try:
                regions = find_pit_drawing_regions(
                    pdf_bytes, [p.type for p in parsed_response.pit_list]
                )
                for p in parsed_response.pit_list:
                    r = regions.get(p.type)
                    if r:
                        p.region = r
                        applied += 1
                print(f"[PitLocator] located {applied} / {len(parsed_response.pit_list)} pit drawing(s) from text layer")
            except Exception as e:
                print(f"[PitLocator] failed: {e}")
            # Focused high-res re-read now that regions are set (only processes
            # pits whose region was located above).
            _run_pit_second_pass(parsed_response.pit_list, images)
            return applied

        if parsed_response.pit_list and pdf_bytes:
            overlap_futures["pit_pass"] = pool.submit(_pit_locate_and_reread)

        # Wait for the beam pass first — downstream merging needs its results.
        if "beam_pass" in overlap_futures:
            overlap_futures["beam_pass"].result()

        # ── Post-beam steps that depend on the final foundation_list ────────
        if plumber_list:
            parsed_response.foundation_list = _merge_with_plumber_data(
                parsed_response.foundation_list, plumber_list
            )

        # Re-apply text-layer cross-section regions to any foundation still
        # missing one. The merge appends plumber-only types (table rows Gemini
        # omitted from its list); those never passed through the locator's
        # in-list update, so they'd otherwise have no region (→ no crop).
        f_regions = text_layer_summary.get("foundation_regions") or {}
        if f_regions:
            recovered = 0
            for item in parsed_response.foundation_list:
                if item.classification == "FW/FG" or item.region:
                    continue
                r = f_regions.get(item.type.upper())
                if r:
                    item.region = r
                    recovered += 1
            if recovered:
                print(f"[TextLocator] re-applied region to {recovered} plumber-only foundation(s)")

        # Split combined rows ("F11, F23" → F11 + F23) so the UI table and Excel
        # show one foundation per row. Runs unconditionally (independent of the
        # text layer); all fields are copied identically to each split part.
        if parsed_response.foundation_list:
            before = len(parsed_response.foundation_list)
            parsed_response.foundation_list = split_combined_types(
                parsed_response.foundation_list
            )
            if len(parsed_response.foundation_list) != before:
                print(f"[Split] combined rows expanded: {before} → "
                      f"{len(parsed_response.foundation_list)}")

        # Final field cleanup: strip ▽SGL/▽GL leaks from D and non-rebar captions
        # (F-code lists, B-formulas) from the rebar columns — covers both plumber
        # and Gemini-sourced rows.
        if parsed_response.foundation_list:
            parsed_response.foundation_list = sanitize_fields(
                parsed_response.foundation_list
            )

        if pdf_text and parsed_response.foundation_list:
            elev_data = parse_elevations(pdf_text)
            floor_plan_elev = dict(elev_data.explicit)   # 伏図 values, before section fill
            section_elev = dict(text_layer_scan.section_elevations)
            # Cross-section 天端 elevations (plain dims + "[ ]内の数値は" bracket notes)
            # read deterministically during the Phase-1a scan. Fill types the floor
            # plan doesn't cover; setdefault keeps a 伏図 annotation authoritative.
            for code, val in section_elev.items():
                elev_data.explicit.setdefault(code, val)
            if elev_data.explicit or elev_data.default is not None:
                parsed_response.foundation_list = resolve_elevations_for_list(
                    parsed_response.foundation_list, elev_data
                )
            # Flag a CONFLICT: floor-plan (伏図) 天端 ≠ the section (断面) 天端 for the
            # same type. We keep the 伏図 value as top_elevation (authoritative) but
            # record the section value in top_elevation_alt so the UI/Excel can show
            # it beside the cell and highlight it red for manual checking.
            for item in parsed_response.foundation_list:
                if item.classification == "FW/FG":
                    continue
                for part in re.split(r'[,、，\s]+', (item.type or "").upper()):
                    part = part.strip()
                    fp, sec = floor_plan_elev.get(part), section_elev.get(part)
                    if fp is not None and sec is not None and fp != sec:
                        item.top_elevation_alt = float(sec)
                        print(f"[ElevConflict] {item.type}: 伏図 {fp} vs 断面 {sec} "
                              f"→ keep {fp}, flag {sec}")
                        break

        if parsed_response.foundation_list:
            parsed_response.foundation_list.sort(key=_foundation_sort_key)

        if parsed_response.oval_gl_list:
            parsed_response.floor_regular_list = _build_regular_floors(parsed_response.oval_gl_list)

        # Beam image crops depend on the final foundation_list (regions may
        # have just been updated by the second pass), so submit them now and
        # let them run while we collect the other futures.
        if parsed_response.foundation_list:
            overlap_futures["beam_crops"] = pool.submit(
                _crop_beam_images, parsed_response.foundation_list, images
            )

        # Drain the rest.
        if "slab_overview" in overlap_futures:
            parsed_response.slab_overview_base64 = overlap_futures["slab_overview"].result()
        if "annotated" in overlap_futures:
            parsed_response.annotated_pages = overlap_futures["annotated"].result()
        if "floor_overview" in overlap_futures:
            parsed_response.floor_overview_base64 = overlap_futures["floor_overview"].result()
        if "table_plumber" in overlap_futures:
            img = overlap_futures["table_plumber"].result()
            if img:
                parsed_response.table_image_base64 = img
        elif "table" in overlap_futures:
            parsed_response.table_image_base64 = overlap_futures["table"].result()
        if "beam_crops" in overlap_futures:
            overlap_futures["beam_crops"].result()

    print(f"[Phase2] Beam pass + crops overlap done in {time.time()-t_phase2:.1f}s")

    # ── Foundation row image crops (text-layer regions — quick CPU/IO) ──────
    if parsed_response.foundation_list:
        _crop_foundation_row_images(parsed_response.foundation_list, images)

    # ── Pit hole image crops ──────────────────────────────────────────────────
    if parsed_response.pit_list:
        # Text-layer region location + focused high-res re-read already ran
        # concurrently with the beam pass inside the Phase 2 pool
        # (_pit_locate_and_reread). Drain it here to surface any exception; the
        # pool's exit already joined it, so result() returns immediately.
        if "pit_pass" in overlap_futures:
            try:
                overlap_futures["pit_pass"].result()
            except Exception as e:
                print(f"[PitPass] pit locate/re-read failed: {e}")
        # Fill pit slab thickness D from the deterministic 詳細図 read ONLY where the
        # vision pass left it null — the Gemini value (it sees the image) wins when
        # present; this is the fallback so no pit is left without a D.
        pit_d_map = getattr(text_layer_scan, "pit_d_map", {}) or {}
        if pit_d_map:
            by_norm = {_normalize_pit_type(k): v for k, v in pit_d_map.items()}
            for pit in parsed_response.pit_list:
                if pit.D is None:
                    d = by_norm.get(_normalize_pit_type(pit.type))
                    if d is not None:
                        pit.D = float(d)
                        print(f"[PitSlabD] filled {pit.type} D={d} (vision null)")
        # Re-assert floor-plan text-layer elevations AFTER the vision second pass:
        # 底盤天端 is authoritative, so it overrides any vision top_elevation.
        if pit_elev_map:
            _apply_pit_text_elevations(parsed_response.pit_list, pit_elev_map)
        _crop_pit_images(parsed_response.pit_list, images)
        print(f"[PitCrop] Cropped {sum(1 for p in parsed_response.pit_list if p.image_base64)} / {len(parsed_response.pit_list)} pit image(s)")

    # ★ Emit "partial_data" right before returning. Caller layers "complete"
    # on top with the same payload; the partial event lets the client render
    # data immediately even if the transport flush of the final event lags.
    partial_payload = parsed_response.model_dump(mode="json")
    partial_payload["excel_ready"] = True
    partial_payload["images_pending"] = False
    _emit("partial_data", partial_payload)

    return parsed_response
