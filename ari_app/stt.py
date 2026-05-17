"""Transcribe WAV using faster-whisper (GPU large-v3 recommended for Hindi/English)."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import httpx

_model_lock = threading.Lock()
_models: dict[tuple[str, str, str], Any] = {}
# Shared WhisperModel is not safe for concurrent transcribe(); serialize inference.
_transcribe_lock = threading.Lock()
_warned_en_model_for_multilingual = False

log = logging.getLogger(__name__)


def _maybe_warn_en_model(whisper_model: str, language: str | None) -> None:
    global _warned_en_model_for_multilingual
    if not whisper_model.endswith(".en"):
        return
    if language == "en":
        return
    if _warned_en_model_for_multilingual:
        return
    _warned_en_model_for_multilingual = True
    log.warning(
        "WHISPER_MODEL=%s is English-only. Hindi or auto language need a multilingual "
        "model (e.g. tiny, base, small — without the .en suffix).",
        whisper_model,
    )


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _whisper_device() -> str:
    raw = os.getenv("WHISPER_DEVICE", "cuda").strip().lower() or "cuda"
    if raw in ("auto", "gpu"):
        return "cuda" if _cuda_available() else "cpu"
    if raw == "cuda" and not _cuda_available():
        log.warning("WHISPER_DEVICE=cuda but no CUDA device; using cpu")
        return "cpu"
    return raw


def _whisper_compute_type(device: str) -> str:
    raw = os.getenv("WHISPER_COMPUTE_TYPE", "").strip()
    if raw:
        return raw
    return "int8" if device == "cpu" else "float16"


def whisper_runtime_info() -> str:
    """Human-readable device/compute for startup logs."""
    device = _whisper_device()
    return f"{device}/{_whisper_compute_type(device)}"


def _clear_model_cache(model_name: str, device: str, compute_type: str) -> None:
    key = (model_name, device, compute_type)
    with _model_lock:
        _models.pop(key, None)


def _cuda_runtime_broken(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "libcublas" in msg or "cudnn" in msg or "cuda" in msg and "not found" in msg


def _get_model(model_name: str, device: str, compute_type: str) -> Any:
    key = (model_name, device, compute_type)
    with _model_lock:
        if key not in _models:
            from faster_whisper import WhisperModel

            log.info(
                "Loading faster-whisper model=%s device=%s compute_type=%s",
                model_name,
                device,
                compute_type,
            )
            _models[key] = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
            log.info("Whisper model %s ready", model_name)
        return _models[key]


def warmup_whisper(*, model_name: str) -> None:
    """Preload Whisper weights at startup (avoids multi-second delay on first call)."""
    device = _whisper_device()
    compute_type = _whisper_compute_type(device)
    _get_model(model_name, device, compute_type)


async def _ffmpeg_to_16k_mono_wav(src: Path, dst: Path) -> None:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(dst),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    code = await proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {code}")


def _transcribe_sync_on_device(
    wav_16k: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
) -> tuple[str, str | None]:
    model = _get_model(model_name, device, compute_type)
    with _transcribe_lock:
        kwargs: dict[str, Any] = {"beam_size": 1}
        if language:
            kwargs["language"] = language
        segments, info = model.transcribe(str(wav_16k), **kwargs)
        parts: list[str] = []
        for seg in segments:
            t = seg.text.strip()
            if t:
                parts.append(t)
        detected: str | None = None
        if language is None and info is not None:
            raw = getattr(info, "language", None)
            if raw:
                detected = str(raw).strip().lower()[:2]
    return " ".join(parts).strip(), detected


def _transcribe_sync(
    wav_16k: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
) -> tuple[str, str | None]:
    _maybe_warn_en_model(model_name, language)
    try:
        return _transcribe_sync_on_device(
            wav_16k, model_name, device, compute_type, language
        )
    except RuntimeError as exc:
        if device != "cuda" or not _cuda_runtime_broken(exc):
            raise
        _clear_model_cache(model_name, device, compute_type)
        log.warning(
            "Whisper CUDA failed (%s); falling back to cpu/int8. "
            "Fix GPU libs: conda install -c conda-forge libcublas=12",
            exc,
        )
        return _transcribe_sync_on_device(
            wav_16k, model_name, "cpu", "int8", language
        )


async def transcribe_wav(
    wav_path: Path,
    whisper_model: str,
    *,
    language: str | None = None,
) -> tuple[str, str | None]:
    """Returns (transcript, whisper_detected_lang or None when language is forced)."""
    if not wav_path.is_file():
        return "", None

    device = _whisper_device()
    compute_type = _whisper_compute_type(device)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        await _ffmpeg_to_16k_mono_wav(wav_path, tmp_path)
        loop = asyncio.get_running_loop()
        text, detected = await loop.run_in_executor(
            None,
            lambda: _transcribe_sync(
                tmp_path, whisper_model, device, compute_type, language
            ),
        )
        if language:
            detected = language[:2].lower()
        return text, detected
    finally:
        tmp_path.unlink(missing_ok=True)


async def download_recording_file(
    *,
    url: str,
    dest: Path,
    auth: httpx.Auth,
) -> bool:
    """Download stored recording WAV. Returns False if Asterisk has no file (e.g. 404, duration 0)."""
    async with httpx.AsyncClient(auth=auth, timeout=120.0) as client:
        r = await client.get(url)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
