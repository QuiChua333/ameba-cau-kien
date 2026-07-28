"""page_gate.py — cheap page-relevance pre-scan for the pdfplumber passes.

WHY THIS EXISTS

pdfplumber's `page.chars` (reached by `extract_words`, `extract_text`, `find_tables`
and `page.curves`) parses a page's ENTIRE content stream — every line, curve and
rect included. On a dense CAD sheet that costs SECONDS, and the cost tracks drawing
complexity rather than text: a 382-character 配筋標準図 sheet in our sample set takes
48s on its own. Once a page is parsed everything else is nearly free — extract_words
on an already-parsed page is ~0.01s.

So the way to make the text-layer passes fast is not to optimise them; it is to stop
handing them pages that cannot possibly contribute. Boring logs (ボーリング柱状図),
revision sheets and generic rebar standard sheets (配筋標準図) carry no foundation
schedule, no section/plan drawing, no beam list and no pit detail.

pypdfium2 (already a dependency, used for rendering and text extraction) reads a
whole document's text layer in C in ~0.2s. That is cheap enough to run first and use
purely as a filter.

SAFETY

The gate is meant to be GENEROUS — err towards keeping a page — and it FAILS OPEN:
  • a page with no text layer at all (a scanned sheet) is always kept;
  • a page whose text cannot be read is always kept;
  • if pypdfium2 is missing or errors, the gate is dropped and every page is scanned;
  • if the filter would keep every page (or none), it returns None = "no gate", so
    the caller's existing code path runs untouched.
Callers must treat `None` as "scan everything".
"""

import re
from typing import Iterable, List, Optional, Pattern, Set


def page_texts(pdf_bytes: bytes) -> Optional[List[str]]:
    """Per-page text layer via pypdfium2, or None when it cannot be read."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    try:
        doc = pdfium.PdfDocument(pdf_bytes)
        out: List[str] = []
        for i in range(len(doc)):
            try:
                out.append(doc[i].get_textpage().get_text_range() or "")
            except Exception:
                out.append("")          # unreadable → treated as "keep" upstream
        return out
    except Exception as e:
        print(f"[PageGate] pre-scan unavailable ({e}); all pages will be scanned.")
        return None


def relevant_pages(pdf_bytes: bytes,
                   keywords: Iterable[str],
                   marker_re: Optional[Pattern] = None,
                   label: str = "PageGate") -> Optional[Set[int]]:
    """0-based indices of pages worth deep-parsing, or None to scan every page.

    A page is kept when its text contains any of `keywords`, or matches
    `marker_re`, or has no readable text at all. See the module docstring for the
    fail-open guarantees.
    """
    texts = page_texts(pdf_bytes)
    if not texts:
        return None

    keep: Set[int] = set()
    for i, text in enumerate(texts):
        if not text.strip():
            keep.add(i)                 # no text layer (scanned) → don't gamble
        elif any(k in text for k in keywords):
            keep.add(i)
        elif marker_re is not None and marker_re.search(text):
            keep.add(i)

    n = len(texts)
    if not keep or len(keep) >= n:
        return None                     # nothing to gain — run as before
    skipped = sorted(i + 1 for i in range(n) if i not in keep)
    print(f"[{label}] page gate: deep-parsing {len(keep)}/{n} page(s); "
          f"skipping {skipped}")
    return keep


# A GL *marker* ("GL+50", "C棟GL±0", "設計GL-1,250") — deliberately NOT the bare
# letters "GL", which also appear on soil-column sheets carrying nothing else we
# need (e.g. "▽設計GL=10.00" on a ボーリング柱状図).
GL_MARKER_HINT = re.compile(r'(?:設計)?(?:.棟)?GL\s*[+\-±]\s*[0-9,]', re.UNICODE)
