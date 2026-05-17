"""Text-to-speech: Supertonic (local) -> 8 kHz mono WAV for Asterisk."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
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


def _probe_sample_rate(wav_path: Path) -> int | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "csv=p=0",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip().isdigit():
            return int(proc.stdout.strip())
    except OSError:
        pass
    return None


def _clamp_speed(speed: float, *, lang: str | None = None) -> float:
    """Allow modest speed-up via SUPERTONIC_SPEED_* (capped to avoid chipmunk on 8 kHz)."""
    s = float(speed)
    if lang == "hi":
        return max(0.90, min(s, 1.12))
    return max(0.90, min(s, 1.15))


def _telephony_af_chain(*, tail_pad_s: float = 0.0) -> str:
    """
    Clean 8 kHz mono for Asterisk.

    - Band-limit like a phone line (reduces harsh / alien highs).
    - soxr resample without async= (async time-stretch causes fast chipmunk tails).
  """
    parts = [
        "highpass=f=100",
        "lowpass=f=3400",
        "aresample=8000:resampler=soxr:precision=28:cutoff=0.96",
    ]
    if tail_pad_s > 0:
        parts.append(f"apad=pad_dur={tail_pad_s:.2f}")
    return ",".join(parts)


def _run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode(errors="replace")[-800:]
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err}")


def _ffmpeg_to_asterisk_wav(
    src: Path, dst: Path, *, tail_pad_s: float = 0.0
) -> None:
    """Convert Supertonic WAV to standard 8 kHz 16-bit mono PCM for Asterisk."""
    sr = _probe_sample_rate(src)
    if sr:
        log.debug("TTS source sample_rate=%s Hz -> 8000 Hz", sr)

    af = _telephony_af_chain(tail_pad_s=tail_pad_s)
    try:
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-af",
                af,
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(dst),
            ]
        )
    except RuntimeError:
        # Fallback without soxr / band-limit
        _run_ffmpeg(
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
            ]
        )


def _synthesize_sync(
    text: str,
    out_wav: Path,
    *,
    voice_name: str,
    lang: str,
    speed: float,
    silence_duration: float,
    tail_pad_s: float = 0.0,
) -> None:
    plain = _tts_plain_text(text)
    if not plain:
        raise ValueError("TTS text is empty after cleanup")

    engine = _ensure_engine()
    style = _get_voice_style(engine, voice_name)

    use_speed = _clamp_speed(speed, lang=lang)
    use_silence = max(0.12, min(float(silence_duration), 0.22))

    log.info(
        "Supertonic synthesizing lang=%s voice=%s speed=%.2f chars=%s preview=%r",
        lang,
        voice_name,
        use_speed,
        len(plain),
        plain[:60],
    )
    t0 = time.perf_counter()
    wav, _duration = engine.synthesize(
        plain,
        voice_style=style,
        lang=lang,
        speed=use_speed,
        silence_duration=use_silence,
    )
    log.info(
        "Supertonic done in %.1fs speed=%.2f silence=%.2fs",
        time.perf_counter() - t0,
        use_speed,
        use_silence,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        raw_wav = Path(tmp.name)
    try:
        engine.save_audio(wav, str(raw_wav))
        _ffmpeg_to_asterisk_wav(raw_wav, out_wav, tail_pad_s=tail_pad_s)
        log.info("Supertonic WAV ready for Asterisk: %s", out_wav)
    finally:
        raw_wav.unlink(missing_ok=True)


def concat_asterisk_wavs(
    parts: list[Path], dst: Path, *, tail_pad_s: float = 0.0
) -> None:
    """Merge chunks and apply one telephony pass (smooth, no boundary artifacts)."""
    parts = [p for p in parts if p.is_file()]
    if not parts:
        raise ValueError("no WAV parts to concat")
    if len(parts) == 1:
        _ffmpeg_to_asterisk_wav(parts[0], dst, tail_pad_s=tail_pad_s)
        return

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as manifest:
        for p in parts:
            manifest.write(f"file '{p.resolve()}'\n")
        manifest_path = Path(manifest.name)

    af = _telephony_af_chain(tail_pad_s=tail_pad_s)
    try:
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_path),
                "-af",
                af,
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(dst),
            ]
        )
    finally:
        manifest_path.unlink(missing_ok=True)


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
    speed: float = 1.0,
    speed_en: float | None = None,
    speed_hi: float | None = None,
    silence_duration: float = 0.18,
    tail_pad_s: float = 0.0,
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
            tail_pad_s=tail_pad_s,
        ),
    )
