"""
Optional remote AI services (Docker: one container per function).

When one of these is set, the matching module sends its work to that service
over HTTP instead of loading the model in this process:

    STT_SERVICE_URL     faster-whisper        (stt.py)          e.g. http://127.0.0.1:8101
    TTS_SERVICE_URL     Supertonic + ffmpeg   (tts.py)          e.g. http://127.0.0.1:8102
    EMBED_SERVICE_URL   e5 embeddings         (rag.py)          e.g. http://127.0.0.1:8103
    GENDER_SERVICE_URL  caller gender model   (gender_model.py) e.g. http://127.0.0.1:8104

Unset (the default) = everything runs in-process, exactly as before.
The services themselves live in services/ and reuse the same ari_app code.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
_async_client: httpx.AsyncClient | None = None


def url(name: str) -> str:
    """Base URL of a service ("stt", "tts", "embed", "gender"), or "" when not used."""
    return (os.getenv(f"{name.upper()}_SERVICE_URL") or "").strip().rstrip("/")


def enabled(name: str) -> bool:
    return bool(url(name))


def _client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _async_client


async def post_file(
    name: str, path: str, file_path: Path, data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """POST a WAV as multipart form to <service><path>; returns the JSON body."""
    with open(file_path, "rb") as fh:
        files = {"audio": (file_path.name, fh.read(), "audio/wav")}
    form = {k: str(v) for k, v in (data or {}).items() if v is not None}
    r = await _client().post(f"{url(name)}{path}", files=files, data=form)
    r.raise_for_status()
    return r.json()


async def post_json(name: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    r = await _client().post(f"{url(name)}{path}", json=payload)
    r.raise_for_status()
    return r.json()


def post_json_sync(name: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """For code that runs in worker threads (RAG indexing, gender model)."""
    r = httpx.post(f"{url(name)}{path}", json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def post_file_sync(
    name: str, path: str, file_path: Path, data: dict[str, Any] | None = None
) -> dict[str, Any]:
    with open(file_path, "rb") as fh:
        files = {"audio": (file_path.name, fh.read(), "audio/wav")}
    form = {k: str(v) for k, v in (data or {}).items() if v is not None}
    r = httpx.post(f"{url(name)}{path}", files=files, data=form, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def wait_ready(name: str, timeout_s: float = 300.0) -> bool:
    """Block until <service>/health answers ok (models loaded). Used at bot startup."""
    import time

    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url(name)}/health", timeout=5.0)
            if r.status_code == 200 and r.json().get("ready"):
                log.info("%s service ready at %s (%s)", name.upper(), url(name), r.json().get("info", ""))
                return True
            last = r.text[:120]
        except httpx.HTTPError as exc:
            last = str(exc)
        time.sleep(2)
    log.error("%s service at %s not ready after %.0fs: %s", name.upper(), url(name), timeout_s, last)
    return False
