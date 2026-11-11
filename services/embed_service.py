"""
Embedding container: sentence-transformers (multilingual-e5-small) for the RAG index.

    POST /embed    JSON {"texts": [...], "kind": "query"|"passage", "model": "..."}
                   -> {"vectors": [[...], ...]}   (L2-normalised float32)
    GET  /health   -> {"ready", "info"}
"""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

import _common  # noqa: F401

DEFAULT_MODEL = os.getenv("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")
_models: dict[str, Any] = {}
_lock = threading.Lock()
state = {"ready": False, "device": "?"}


def _model(name: str):
    with _lock:
        if name not in _models:
            import torch
            from sentence_transformers import SentenceTransformer

            state["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            _models[name] = SentenceTransformer(name, device=state["device"])
        return _models[name]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asyncio.to_thread(_model, DEFAULT_MODEL)
    state["ready"] = True
    yield


app = FastAPI(title="embed", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"ready": state["ready"], "info": f"{DEFAULT_MODEL} on {state['device']}"}


@app.post("/embed")
async def embed(body: dict[str, Any]) -> dict:
    texts = [str(t) for t in body.get("texts", [])]
    kind = body.get("kind", "query")
    name = body.get("model") or DEFAULT_MODEL

    def run():
        # e5 models expect "query: " / "passage: " prefixes (same as ari_app.rag)
        vecs = _model(name).encode(
            [f"{kind}: {t}" for t in texts],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    return {"vectors": await asyncio.to_thread(run)}
