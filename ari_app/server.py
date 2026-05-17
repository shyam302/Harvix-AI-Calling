"""ASGI app: uvicorn serves HTTP (health) while the ARI bot runs in the background."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ari_app.ari_loop import get_active_call_registry, run_ari_forever
from ari_app.config import load_settings
from ari_app.llm import check_vllm_reachable
from ari_app.stt import warmup_whisper
from ari_app.tts import warmup_supertonic

log = logging.getLogger("ari_callbot")


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


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _configure_logging()
    settings = load_settings()
    if not settings.ari_password:
        log.error("Set ARI_PASSWORD in the environment or .env file.")
        raise RuntimeError("ARI_PASSWORD is not set")

    loop = asyncio.get_running_loop()
    def _warmup_models() -> None:
        warmup_supertonic(
            voice_en=settings.tts_voice,
            voice_hi=settings.tts_voice_hi,
            voice_female=settings.tts_voice_female,
            voice_male=settings.tts_voice_male,
        )
        warmup_whisper(model_name=settings.whisper_model)

    await loop.run_in_executor(None, _warmup_models)

    if not await check_vllm_reachable(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        connect_timeout_seconds=settings.vllm_connect_timeout_seconds,
    ):
        log.error(
            "vLLM is not reachable at %s — fix network/Tailscale/firewall or VLLM_BASE_URL "
            "before calls can get AI replies.",
            settings.vllm_base_url,
        )

    task = asyncio.create_task(run_ari_forever(), name="ari_forever")
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="Asterisk Callbot", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    registry = get_active_call_registry()
    return JSONResponse(
        {
            "status": "ok",
            "active_calls": registry.active_count(),
            "calls": registry.list_active(),
        }
    )


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {"service": "asterisk-callbot", "health": "/health"},
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
    )


if __name__ == "__main__":
    main()
