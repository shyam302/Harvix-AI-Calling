"""
Text-to-speech container: Supertonic on the GPU + ffmpeg phone conversion (ari_app.tts).

    POST /synthesize   JSON: same arguments as ari_app.tts.synthesize_to_wav
                       -> {"wav": <base64 8 kHz WAV>, "wav16": <base64 16 kHz WAV or null>}
    GET  /health       -> {"ready", "info"}
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI

import _common  # noqa: F401
from ari_app import tts
from ari_app.inference_pool import configure

state = {"ready": False}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure(whisper_max_concurrent=1, tts_max_concurrent=int(os.getenv("TTS_MAX_CONCURRENT", "2")))
    from ari_app.supertonic_onnx import ensure_cuda_library_path

    ensure_cuda_library_path()
    await asyncio.to_thread(
        tts.warmup_supertonic,
        voice_en=os.getenv("SUPERTONIC_VOICE", "F3"),
        voice_hi=os.getenv("SUPERTONIC_VOICE_HI", ""),
        voice_female=os.getenv("SUPERTONIC_VOICE_FEMALE", "F3"),
        voice_male=os.getenv("SUPERTONIC_VOICE_MALE", "M1"),
    )
    state["ready"] = True
    yield


app = FastAPI(title="tts", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"ready": state["ready"], "info": "supertonic"}


@app.post("/synthesize")
async def synthesize(body: dict[str, Any]) -> dict:
    text = str(body.pop("text", ""))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "tts.wav"
        await tts.synthesize_to_wav(text, out, **body)
        twin = tts.wideband_path(out)
        return {
            "wav": base64.b64encode(out.read_bytes()).decode(),
            "wav16": base64.b64encode(twin.read_bytes()).decode() if twin.is_file() else None,
        }
