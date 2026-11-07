"""ASGI app: uvicorn serves HTTP (health) while the ARI bot runs in the background."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ari_app.ari_loop import get_active_call_registry, run_ari_forever
from ari_app.config import load_settings
from ari_app.llm import check_vllm_reachable
from ari_app.inference_pool import configure as configure_inference_pool
from ari_app.inference_pool import pool_status, shutdown as shutdown_inference_pool
from ari_app.gender_model import warmup_gender_model
from ari_app.live_vad import warmup_vad
from ari_app.stt import warmup_whisper
from ari_app.tts import warmup_supertonic

log = logging.getLogger("ari_callbot")


# transformers prints harmless "LOAD REPORT … UNEXPECTED" tables; show errors only.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def _configure_logging() -> None:
    """Uvicorn does not enable our package loggers; mirror `python -m ari_app` so calls print to the terminal."""
    level = getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    try:
        logging.basicConfig(level=level, format=fmt, stream=sys.stderr, force=True)
    except TypeError:
        # Python < 3.8: no force=
        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(level=level, format=fmt, stream=sys.stderr)

    # Optional copy of the console feed on disk (LOG_FILE=logs/callbot.log).
    log_file = (os.getenv("LOG_FILE") or "").strip()
    if log_file:
        from logging.handlers import RotatingFileHandler

        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(handler)

    # Quiet third-party noise that doesn't belong in the per-call console feed.
    for noisy in (
        "httpx",
        "httpcore",
        "ddgs",
        "ddgs.ddgs",
        "primp",
        "duckduckgo_search",
        "crawl4ai",
        "openai",
        "urllib3",
        "asyncio",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _configure_logging()
    settings = load_settings()
    if not settings.ari_password:
        log.error("Set ARI_PASSWORD in the environment or .env file.")
        raise RuntimeError("ARI_PASSWORD is not set")

    workers = settings.inference_thread_workers or None
    configure_inference_pool(
        whisper_max_concurrent=settings.whisper_max_concurrent,
        tts_max_concurrent=settings.tts_max_concurrent,
        thread_workers=workers,
    )

    loop = asyncio.get_running_loop()
    def _warmup_models() -> None:
        if settings.tts_backend != "edge":
            from ari_app.supertonic_onnx import ensure_cuda_library_path

            ensure_cuda_library_path()
            warmup_supertonic(
                voice_en=settings.tts_voice,
                voice_hi=settings.tts_voice_hi,
                voice_female=settings.tts_voice_female,
                voice_male=settings.tts_voice_male,
            )
        else:
            log.info("TTS_BACKEND=edge — skipping Supertonic warmup")
        warmup_whisper(model_name=settings.whisper_model)
        warmup_gender_model(enabled=settings.gender_model_enabled)
        if settings.vad_enabled:
            warmup_vad()
        if settings.rag_enabled:
            # Load the embedding model and index the knowledge folder before calls.
            from ari_app.rag import get_knowledge_base

            kb = get_knowledge_base(settings.rag_docs_dir, model_name=settings.rag_embed_model)
            kb.warmup()
            log.info("Knowledge base ready: %s chunks from %s", len(kb.chunks), kb.docs_dir)

    await loop.run_in_executor(None, _warmup_models)

    if await check_vllm_reachable(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        connect_timeout_seconds=settings.vllm_connect_timeout_seconds,
    ):
        log.info("vLLM reachable at %s (model %s)", settings.vllm_base_url, settings.vllm_model)
    else:
        log.error(
            "vLLM is not reachable at %s — fix network/Tailscale/firewall or VLLM_BASE_URL "
            "before calls can get AI replies.",
            settings.vllm_base_url,
        )

    task = asyncio.create_task(run_ari_forever(), name="ari_forever")
    yield
    # Shutdown (Ctrl+C): every step is time-boxed so one Ctrl+C is enough.
    log.info("Shutting down: stopping ARI loop and closing the database")
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
        await asyncio.wait_for(task, timeout=3)
    shutdown_inference_pool()
    from ari_app import db

    await db.close()


app = FastAPI(title="Asterisk Callbot", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    registry = get_active_call_registry()
    return JSONResponse(
        {
            "status": "ok",
            "active_calls": registry.active_count(),
            "calls": registry.list_active(),
            "inference_pool": pool_status(),
        }
    )


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {"service": "asterisk-callbot", "health": "/health", "recordings": "/recordings", "customers": "/customers"},
    )


# ---- call recordings page + API (data from PostgreSQL, audio from RECORDINGS_DIR) ----

_STATIC = Path(__file__).resolve().parent / "static"


def _jsonable(row: dict) -> dict:
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}


@app.get("/recordings", response_class=HTMLResponse)
async def recordings_page() -> HTMLResponse:
    return HTMLResponse((_STATIC / "recordings.html").read_text(encoding="utf-8"))


@app.get("/api/calls")
async def api_calls(
    q: str = "", limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)
) -> JSONResponse:
    from ari_app import db

    if not db.available():
        raise HTTPException(503, "Database not connected (DATABASE_URL / docker compose up -d postgres)")
    rows = await db.list_calls(limit=limit, offset=offset, search=q)
    return JSONResponse([_jsonable(r) for r in rows])


@app.get("/api/calls/{call_id}")
async def api_call(call_id: int) -> JSONResponse:
    from ari_app import db

    call = await db.get_call(call_id)
    if call is None:
        raise HTTPException(404, "No such call")
    call["messages"] = [_jsonable(m) for m in call["messages"]]
    call["complaints"] = [_jsonable(c) for c in call.get("complaints", [])]
    call.pop("recording_path", None)  # the browser uses /audio, never a disk path
    return JSONResponse(_jsonable(call))


# ---- customers + complaints admin page ----

def _db_or_503():
    from ari_app import db

    if not db.available():
        raise HTTPException(503, "Database not connected (DATABASE_URL / docker compose up -d postgres)")
    return db


def _deep_jsonable(v):
    if isinstance(v, dict):
        return {k: _deep_jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_deep_jsonable(x) for x in v]
    return v.isoformat() if hasattr(v, "isoformat") else v


@app.get("/customers", response_class=HTMLResponse)
async def customers_page() -> HTMLResponse:
    return HTMLResponse((_STATIC / "customers.html").read_text(encoding="utf-8"))


@app.get("/api/customers")
async def api_customers(
    q: str = "", limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)
) -> JSONResponse:
    db = _db_or_503()
    return JSONResponse(_deep_jsonable(await db.list_customers(search=q, limit=limit, offset=offset)))


@app.get("/api/customers/{user_id}")
async def api_customer(user_id: int) -> JSONResponse:
    db = _db_or_503()
    cust = await db.get_customer(user_id)
    if cust is None:
        raise HTTPException(404, "No such customer")
    return JSONResponse(_deep_jsonable(cust))


@app.get("/api/complaints")
async def api_complaints(
    status: str = "",
    q: str = "",
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> JSONResponse:
    db = _db_or_503()
    rows = await db.list_complaints(status=status, search=q, limit=limit, offset=offset)
    return JSONResponse({"items": _deep_jsonable(rows), "counts": await db.complaint_counts()})


@app.post("/api/complaints/{ticket_no}/status")
async def api_complaint_status(ticket_no: str, body: dict) -> JSONResponse:
    db = _db_or_503()
    status = str(body.get("status", ""))
    if status not in db.COMPLAINT_STATUSES:
        raise HTTPException(400, f"status must be one of {', '.join(db.COMPLAINT_STATUSES)}")
    if not await db.set_complaint_status(ticket_no, status):
        raise HTTPException(404, "No such ticket")
    log.info("complaint %s status -> %s (admin page)", ticket_no, status)
    return JSONResponse({"ticket_no": ticket_no, "status": status})


@app.post("/api/complaints/{ticket_no}/resolution")
async def api_complaint_resolution(ticket_no: str, body: dict) -> JSONResponse:
    db = _db_or_503()
    text = str(body.get("resolution", ""))
    if not await db.set_complaint_resolution(ticket_no, text):
        raise HTTPException(404, "No such ticket")
    log.info("complaint %s resolution updated (admin page, %s chars)", ticket_no, len(text.strip()))
    return JSONResponse({"ticket_no": ticket_no, "resolution": text.strip()[:1000]})


@app.get("/api/calls/{call_id}/audio")
async def api_call_audio(call_id: int, download: bool = False) -> FileResponse:
    from ari_app import call_recording, db

    call = await db.get_call(call_id)
    if call is None or not call.get("recording_path"):
        raise HTTPException(404, "No recording for this call")
    root = call_recording.recordings_root(load_settings().recordings_dir)
    path = call_recording.resolve_inside(root, call["recording_path"])
    if path is None:
        raise HTTPException(404, "Recording file missing on disk")
    return FileResponse(
        path,
        media_type="audio/ogg",
        filename=path.name if download else None,
    )


def main() -> None:
    import uvicorn

    host = os.getenv("UVICORN_HOST", "127.0.0.1")
    port = int(os.getenv("UVICORN_PORT", "8765"))
    uvicorn.run(
        "ari_app.server:app",
        host=host,
        port=port,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
        # Don't wait forever for browsers (recordings page keeps audio connections
        # open); after 3 s open connections are closed and shutdown continues.
        timeout_graceful_shutdown=3,
    )


if __name__ == "__main__":
    main()
