"""Shared helpers for the AI service containers (stt, tts, embed, gender)."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

from fastapi import UploadFile

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
# These containers run the model themselves: never forward to another service.
for _name in ("STT", "TTS", "EMBED", "GENDER"):
    os.environ.pop(f"{_name}_SERVICE_URL", None)


async def save_upload(audio: UploadFile) -> Path:
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        return Path(tmp.name)
