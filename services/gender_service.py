"""
Caller gender container: Wav2Vec2 gender model (ari_app.gender_model).

    POST /classify   multipart: audio=<wav>, min_confidence
                     -> {"gender": "female"|"male"|null, "p_female", "p_male"}
    GET  /health     -> {"ready", "info"}
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile

import _common  # noqa: F401
from _common import save_upload
from ari_app import gender_model

state = {"ready": False}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["ready"] = await asyncio.to_thread(gender_model.warmup_gender_model, enabled=True)
    yield


app = FastAPI(title="gender", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"ready": state["ready"], "info": gender_model._MODEL_NAME}


@app.post("/classify")
async def classify(audio: UploadFile = File(...), min_confidence: float = Form(0.55)) -> dict:
    path = await save_upload(audio)
    try:
        result = await asyncio.to_thread(
            gender_model.classify_caller_gender, path, min_confidence=min_confidence
        )
    finally:
        path.unlink(missing_ok=True)
    if result is None:
        return {"gender": None, "p_female": None, "p_male": None}
    g, pf, pm = result
    return {"gender": g, "p_female": pf, "p_male": pm}
