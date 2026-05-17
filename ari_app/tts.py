"""Text-to-speech: Supertonic (local) -> 8 kHz mono WAV for Asterisk."""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from ari_app.lang import resolve_tts_lang, text_contains_devanagari

log = logging.getLogger(__name__)

_DEFAULT_VOICE_EN = "F3"
_DEFAULT_VOICE_HI = "F3"

_tts_lock = threading.Lock()
_tts_instance: Any | None = None
_voice_styles: dict[str, Any] = {}


def pick_supertonic_lang(text: str, *, session_lang: str | None = None) -> str:
    """Hindi when reply or call session is Hindi (Devanagari, session, or Romanized)."""
    return resolve_tts_lang(text, session_lang=session_lang)


def pick_supertonic_voice(
    text: str,
    *,
    voice_en: str,
    voice_hi: str,
    session_lang: str | None = None,
) -> str:
    if pick_supertonic_lang(text, session_lang=session_lang) == "hi":
        return (voice_hi or voice_en or _DEFAULT_VOICE_HI).strip() or _DEFAULT_VOICE_HI
    return (voice_en or _DEFAULT_VOICE_EN).strip() or _DEFAULT_VOICE_EN


def _tts_plain_text(text: str, *, max_chars: int = 800) -> str:
    t = (text or "").strip()
    t = re.sub(r"[*_#`]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_chars:
        t = t[: max_chars - 3].rstrip() + "..."
    return t


def _ensure_engine() -> Any:
    """Load Supertonic once (thread-safe). Do not call other helpers that take _tts_lock."""
    global _tts_instance
    with _tts_lock:
        if _tts_instance is None:
            from supertonic import TTS

            log.info("Loading Supertonic TTS (auto_download=True)...")
            _tts_instance = TTS(auto_download=True)
            log.info("Supertonic TTS ready")
        return _tts_instance


def _get_voice_style(engine: Any, voice_name: str) -> Any:
    with _tts_lock:
        if voice_name not in _voice_styles:
            log.info("Loading Supertonic voice style %s", voice_name)
            _voice_styles[voice_name] = engine.get_voice_style(voice_name=voice_name)
        return _voice_styles[voice_name]


def warmup_supertonic(
    *,
    voice_en: str,
    voice_hi: str = "",
    voice_female: str = "",
    voice_male: str = "",
) -> None:
    """Preload model + voice styles at startup (optional; runs in a thread)."""
    engine = _ensure_engine()
    names = {voice_en, voice_hi or voice_en, voice_female, voice_male}
    for name in names:
        if name:
            _get_voice_style(engine, name.strip())


def _synthesize_sync(
    text: str,
    out_wav: Path,
    *,
    voice_name: str,
    lang: str,
    speed: float,
    silence_duration: float,
) -> None:
    plain = _tts_plain_text(text)
    if not plain:
        raise ValueError("TTS text is empty after cleanup")

    engine = _ensure_engine()
    style = _get_voice_style(engine, voice_name)

    log.info(
        "Supertonic synthesizing lang=%s voice=%s chars=%s preview=%r",
        lang,
        voice_name,
        len(plain),
        plain[:60],
    )
    t0 = time.perf_counter()
    wav, _duration = engine.synthesize(
        plain,
        voice_style=style,
        lang=lang,
        speed=speed,
        silence_duration=silence_duration,
    )
    log.info("Supertonic synthesize finished in %.1fs", time.perf_counter() - t0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        raw_wav = Path(tmp.name)
    try:
        engine.save_audio(wav, str(raw_wav))
        _ffmpeg_to_asterisk_wav(raw_wav, out_wav)
        log.info("Supertonic WAV ready for Asterisk: %s", out_wav)
    finally:
        raw_wav.unlink(missing_ok=True)


def _ffmpeg_to_asterisk_wav(src: Path, dst: Path) -> None:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ar",
            "8000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg TTS post-process failed: {proc.returncode}")


def speed_for_lang(
    lang_code: str,
    *,
    speed_en: float,
    speed_hi: float,
    speed_default: float,
) -> float:
    if lang_code == "en":
        return speed_en
    if lang_code == "hi":
        return speed_hi
    return speed_default


async def synthesize_to_wav(
    text: str,
    out_wav: Path,
    *,
    voice_en: str,
    voice_hi: str = "",
    speed: float = 1.05,
    speed_en: float | None = None,
    speed_hi: float | None = None,
    silence_duration: float = 0.12,
    session_lang: str | None = None,
    lang: str | None = None,
    voice_name: str | None = None,
) -> None:
    """Synthesize with Supertonic; lang hi/en from text + call session."""
    resolved_lang = lang or pick_supertonic_lang(text, session_lang=session_lang)
    resolved_speed = speed_for_lang(
        resolved_lang,
        speed_en=speed_en if speed_en is not None else speed,
        speed_hi=speed_hi if speed_hi is not None else speed,
        speed_default=speed,
    )
    resolved_voice = voice_name or pick_supertonic_voice(
        text,
        voice_en=voice_en,
        voice_hi=voice_hi,
        session_lang=resolved_lang,
    )

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: _synthesize_sync(
            text,
            out_wav,
            voice_name=resolved_voice,
            lang=resolved_lang,
            speed=resolved_speed,
            silence_duration=silence_duration,
        ),
    )
