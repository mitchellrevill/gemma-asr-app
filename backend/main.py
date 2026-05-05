"""
main.py — FastAPI server for the Gemma 4 E2B-it police situation report ASR app.

Endpoints:
  POST /api/transcribe   — Upload audio → receive structured JSON police report
  GET  /api/status       — Health check / model load status
  GET  /                 — Serves the front-end HTML
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from asr import get_asr
from audio_utils import (
    CHUNK_DURATION_S,
    cleanup_temp_files,
    chunk_audio,
    get_audio_duration,
    load_audio,
    save_chunks_to_temp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Supported audio MIME types ───────────────────────────────────────────────
SUPPORTED_AUDIO_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/m4a", "audio/x-m4a",
    "audio/ogg", "audio/webm",
    "audio/flac", "audio/x-flac",
    "application/octet-stream",  # browsers sometimes send this for recordings
}


def normalize_media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


MAX_FILE_SIZE_MB = 500
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


# ── Lifespan: pre-load model on startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-loading Gemma 4 E2B-it model…")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_asr().load)
    logger.info("Model ready — server accepting requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Gemma ASR Police Report API",
    description="Transcribes audio recordings and generates structured police situation reports.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the police report template statically
if TEMPLATES_DIR.exists():
    app.mount("/templates", StaticFiles(directory=str(TEMPLATES_DIR)), name="templates")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index))


@app.get("/api/status")
async def status() -> Dict[str, Any]:
    asr = get_asr()
    return {
        "status": "ready" if asr._model is not None else "loading",
        "model": "google/gemma-4-E2B-it",
        "chunk_duration_s": CHUNK_DURATION_S,
    }


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept an audio file, chunk it into <=28s segments, transcribe each with
    Gemma 4 E2B-it, then extract a structured police report JSON.
    """
    # ── Validate ──────────────────────────────────────────────────────────
    content_type = normalize_media_type(file.content_type or "")
    if content_type not in SUPPORTED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type: {file.content_type}. "
                f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_TYPES))}"
            ),
        )

    tmp_dir = tempfile.mkdtemp(prefix="gemma_asr_")
    raw_path = os.path.join(tmp_dir, f"upload_{int(time.time())}{Path(file.filename or 'audio.wav').suffix}")
    chunk_paths = []

    try:
        # ── Save upload ───────────────────────────────────────────────────
        with open(raw_path, "wb") as f:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.",
                )
            f.write(content)

        # ── Get duration ──────────────────────────────────────────────────
        audio_duration_s = get_audio_duration(raw_path)
        logger.info(f"Received audio: {file.filename} ({audio_duration_s:.1f}s)")

        # ── Load & chunk ──────────────────────────────────────────────────
        loop = asyncio.get_event_loop()

        def _load_and_chunk():
            waveform, sr = load_audio(raw_path)
            chunks = chunk_audio(waveform, sr)
            paths = save_chunks_to_temp(chunks, sr, temp_dir=tmp_dir)
            return paths

        chunk_paths = await loop.run_in_executor(None, _load_and_chunk)
        n_chunks = len(chunk_paths)
        logger.info(f"Processing {n_chunks} chunks...")

        # ── Transcribe all chunks ─────────────────────────────────────────
        def _transcribe():
            asr = get_asr()
            return asr.transcribe_chunks(chunk_paths)

        transcription = await loop.run_in_executor(None, _transcribe)
        logger.info(f"Transcription complete ({len(transcription)} chars)")

        # ── Extract structured report ─────────────────────────────────────
        def _extract():
            asr = get_asr()
            return asr.extract_report(transcription)

        report = await loop.run_in_executor(None, _extract)

        # ── Augment with pipeline metadata ───────────────────────────────
        report["audio_duration_s"] = round(audio_duration_s, 1)
        report["chunks_processed"] = n_chunks
        report["confidence"] = "high" if n_chunks == 1 else "medium"
        report["raw_transcription"] = transcription
        report.setdefault("report_date", time.strftime("%Y-%m-%d"))
        report.setdefault("status", "open")
        report.setdefault("classification", "UNCLASSIFIED")

        return JSONResponse(content=report)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        cleanup_temp_files(chunk_paths)
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # Single worker — model is loaded once in memory
    )
