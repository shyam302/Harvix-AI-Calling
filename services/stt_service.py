"""
Speech-to-text container: faster-whisper on the GPU (ari_app.stt).

    POST /transcribe   multipart: audio=<wav>, model, language, initial_prompt
                       -> {"text", "language", "probability"}
    GET  /health       -> {"ready", "info"}
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile

import _common  # noqa: F401  (logging + no service forwarding)
from _common import save_upload
from ari_app import stt
from ari_app.inference_pool import configure

MODEL = os.getenv("WHISPER_MODEL", "medium")
state = {"ready": False}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure(
        whisper_max_concurrent=int(os.getenv("WHISPER_MAX_CONCURRENT", "3")),
        tts_max_concurrent=1,
    )
    await asyncio.to_thread(stt.warmup_whisper, model_name=MODEL)
    state["ready"] = True
    yield


app = FastAPI(title="stt", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"ready": state["ready"], "info": f"whisper {MODEL} on {stt.whisper_runtime_info()}"}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    model: str = Form(MODEL),
    language: str | None = Form(None),
    initial_prompt: str | None = Form(None),
) -> dict:
    path = await save_upload(audio)
    try:
        text, lang, prob = await stt.transcribe_wav(
            path, model or MODEL, language=language or None, initial_prompt=initial_prompt or None
        )
    finally:
        path.unlink(missing_ok=True)
    return {"text": text, "language": lang, "probability": prob}
