from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from services.pdf_service import convert_pdf_to_images, extract_text_from_pdf
from services.gemini_service import extract_data_from_images
from services.excel_service import fill_excel
from models import FoundationItem, PitHoleItem
from pydantic import BaseModel
from typing import List
import asyncio
import io
import json
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Japanese Construction Drawing Extractor",
    description="API to extract structured foundation data from Japanese PDF drawings using Gemini.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/extract-foundation-data-stream")
async def extract_foundation_data_stream(file: UploadFile = File(...)):
    """SSE stream that surfaces extraction **stages**, not partial content.

    Events emitted (in order):
      • connected            — handshake (immediate)
      • status               — stage label updates (PDF convert, table extract, Gemini, crops…)
      • started              — total_pages once PDF→images is done
      • partial_table_data   — pdfplumber preview snapshot (table renderable early)
      • partial_data         — full snapshot once Gemini + post-processing done (Excel ready, images may still be pending)
      • complete             — final snapshot with all image crops embedded
      • error                — fatal error with `detail`
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    content = await file.read()
    loop = asyncio.get_running_loop()

    async def event_generator():
        q: asyncio.Queue = asyncio.Queue()
        SENTINEL: object = object()

        # Immediate handshake so the client knows the stream is alive even
        # before the worker thread spins up. Some proxies/browsers otherwise
        # delay the response until the first real bytes arrive.
        yield ":ping\n\n"
        yield 'event: connected\ndata: {}\n\n'

        def schedule_put(item):
            """Thread-safe put onto the asyncio queue from the worker thread."""
            try:
                asyncio.run_coroutine_threadsafe(q.put(item), loop)
            except Exception:
                pass

        def on_partial(phase: str, data: dict):
            schedule_put((phase, data))

        def worker():
            t_w = time.time()
            try:
                schedule_put(("status", {
                    "message": "Đang chuẩn bị ảnh từ PDF...",
                    "stage": "pdf_convert_started",
                }))
                images = convert_pdf_to_images(content)
                if not images:
                    schedule_put(("error", {"detail": "Could not convert PDF to images."}))
                    return
                schedule_put(("started", {
                    "total_pages": len(images),
                    "elapsed": round(time.time() - t_w, 2),
                }))

                schedule_put(("status", {
                    "message": "Đang đọc text layer của PDF...",
                    "stage": "pdf_text_started",
                }))
                pdf_text = extract_text_from_pdf(content)
                result = extract_data_from_images(
                    images, pdf_text=pdf_text, pdf_bytes=content,
                    on_partial=on_partial,
                )
                complete_payload = result.model_dump(mode="json")
                complete_payload["excel_ready"] = True
                complete_payload["images_pending"] = False
                schedule_put(("complete", complete_payload))
            except Exception as e:
                logger.exception("Stream worker failed")
                schedule_put(("error", {"detail": str(e)}))
            finally:
                schedule_put(SENTINEL)

        loop.run_in_executor(None, worker)

        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                yield ":ping\n\n"
                continue
            if item is SENTINEL:
                break
            phase, data = item
            payload = json.dumps(data, ensure_ascii=False)
            yield f"event: {phase}\ndata: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
        },
    )


class GenerateExcelRequest(BaseModel):
    foundation_list: List[FoundationItem]
    pit_list: List[PitHoleItem] = []


@app.post("/generate-excel")
async def generate_excel(request: GenerateExcelRequest):
    """Generate a filled Excel template from extracted foundation/beam data."""
    try:
        from urllib.parse import quote
        excel_bytes = fill_excel(request.foundation_list, request.pit_list)
        encoded_name = quote("計算書(施工) - DD.xlsx", safe="")
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=\"keisansho.xlsx\"; filename*=UTF-8''{encoded_name}"},
        )
    except Exception as e:
        logger.error(f"Excel generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Excel generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
