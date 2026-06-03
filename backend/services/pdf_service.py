import pypdfium2 as pdfium
from typing import List
from PIL import Image

# Render scale: 2.5 keeps Japanese small numerals crisp while keeping the
# raw bitmap under ~4096 px on A1 sheets. Going higher only wastes CPU
# because we resize to ≤1800 px before sending to Gemini anyway.
RENDER_SCALE = 2.5
MAX_DIMENSION = 4096


def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract all text content from a PDF using pypdfium2's text layer.

    Works for vector/CAD-generated PDFs (not scanned images).
    Returns concatenated text from all pages, separated by newlines.
    """
    texts: List[str] = []
    try:
        pdf = pdfium.PdfDocument(pdf_content)
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            if text:
                texts.append(text)
    except Exception as e:
        print(f"[TextExtract] PDF text extraction error: {e}")
    return "\n".join(texts)


def convert_pdf_to_images(pdf_content: bytes, max_pages: int = 100) -> List[Image.Image]:
    """Convert pages of a PDF to RGB images using pypdfium2.

    Rendered sequentially because the pdfium C library is NOT thread-safe —
    parallel rendering caused segfaults that crashed the uvicorn worker.
    Sequential rendering at scale=2.5 is still fast enough for our use case.
    """
    images: List[Image.Image] = []
    try:
        pdf = pdfium.PdfDocument(pdf_content)
        n_pages = min(len(pdf), max_pages)
        for i in range(n_pages):
            page = pdf[i]
            bitmap = page.render(scale=RENDER_SCALE)
            pil_image = bitmap.to_pil()

            if max(pil_image.size) > MAX_DIMENSION:
                pil_image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)

            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            images.append(pil_image)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        raise
