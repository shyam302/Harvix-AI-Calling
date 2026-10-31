"""Text-to-speech: Supertonic (local) -> 8 kHz mono WAV for Asterisk."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import wave
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


def _tts_backend() -> str:
    return (os.getenv("TTS_BACKEND") or "supertonic").strip().lower()


def _tts_plain_text(
    text: str, *, max_chars: int = 800, session_lang: str | None = None
) -> str:
    from ari_app.conversation import prepare_text_for_tts

    lang = session_lang or pick_supertonic_lang(text, session_lang=session_lang)
    t = prepare_text_for_tts(text, lang=lang)
    t = re.sub(r"[*_#`]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_chars:
        t = t[: max_chars - 3].rstrip() + "..."
    return t


def _speed_to_edge_rate(speed: float) -> str:
    """Map Supertonic-style speed (0.9–1.12) to edge-tts rate string."""
    pct = int(round((float(speed) - 1.0) * 100))
    pct = max(-15, min(pct, 12))
    return f"{pct:+d}%" if pct != 0 else "+0%"


def _ensure_engine() -> Any:
    """Load Supertonic once (thread-safe). Do not call other helpers that take _tts_lock."""
    global _tts_instance
    with _tts_lock:
        if _tts_instance is None:
            from ari_app.supertonic_onnx import (
                configure_supertonic_onnx,
                ensure_cuda_library_path,
                log_supertonic_session_providers,
                session_uses_cuda,
            )

            # CUDA pip libs must be on LD_LIBRARY_PATH before onnxruntime / supertonic load.
            lib_dirs = ensure_cuda_library_path()
            if lib_dirs:
                log.info("Supertonic CUDA libs: %s", lib_dirs)
            configure_supertonic_onnx()

            from supertonic import TTS

            log.info("Loading Supertonic TTS (auto_download=True)...")
            _tts_instance = TTS(auto_download=True)
            if session_uses_cuda(_tts_instance):
                log.info("Supertonic TTS ready (ONNX GPU)")
            else:
                log.warning(
                    "Supertonic TTS ready but ONNX is on CPU — "
                    "pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 "
                    "nvidia-cudnn-cu12 nvidia-cufft-cu12"
                )
            log_supertonic_session_providers(_tts_instance)
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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def wideband_enabled() -> bool:
    """Also write a 16 kHz ``.wav16`` next to each 8 kHz ``.wav`` (HD audio on G.722 calls)."""
    return _bool_env("TTS_WIDEBAND", True)


def wideband_path(wav_path: Path) -> Path:
    return wav_path.with_suffix(".wav16")


def remove_tts_outputs(wav_path: Path) -> None:
    """Delete a played TTS file and its wideband twin."""
    wav_path.unlink(missing_ok=True)
    wideband_path(wav_path).unlink(missing_ok=True)


def _telephony_pre_chain(*, tail_pad_s: float = 0.0) -> str:
    """
    Rate-independent cleanup before band-limiting.

    - Trim only the *trailing* near-silence (mid-sentence pauses are kept so speech
      does not sound rushed), then a short fade-out to avoid a click.
    - Normalize loudness to a phone-friendly level (Supertonic output is ~-27 dB).
    """
    fade_ms = _int_env("TTS_TAIL_FADE_MS", 45)
    fade_s = max(0.02, min(fade_ms / 1000.0, 0.10))

    parts = ["highpass=f=80"]
    if _bool_env("TTS_TRIM_TRAILING_SILENCE", True):
        parts.append(
            "areverse,"
            "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.08,"
            f"afade=t=in:st=0:d={fade_s:.3f},"
            "areverse"
        )
    else:
        parts.append(f"areverse,afade=t=in:st=0:d={fade_s:.3f},areverse")
    if _bool_env("TTS_LOUDNORM", True):
        target = _int_env("TTS_LOUDNESS_LUFS", -18)
        parts.append(f"loudnorm=I={target}:TP=-1.5:LRA=11")
    if tail_pad_s > 0:
        parts.append(f"apad=pad_dur={tail_pad_s:.2f}")
    return ",".join(parts)


_soxr_available: bool | None = None


def _ffmpeg_has_soxr() -> bool:
    """Conda's ffmpeg is often built without libsoxr; aresample=resampler=soxr then fails."""
    global _soxr_available
    if _soxr_available is None:
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-buildconf"],
                capture_output=True,
                text=True,
                check=False,
            )
            _soxr_available = "libsoxr" in (proc.stdout or "")
        except OSError:
            _soxr_available = False
        if not _soxr_available:
            log.info("ffmpeg has no libsoxr; using built-in high-quality resampler")
    return _soxr_available


def _band_chain(rate: int, *, light: bool) -> str:
    lowpass = 3400 if rate <= 8000 else 7200
    if _ffmpeg_has_soxr():
        precision = 20 if light else 28
        resample = f"aresample={rate}:resampler=soxr:precision={precision}:cutoff=0.96"
    else:
        filter_size = 32 if light else 64
        resample = (
            f"aresample={rate}:filter_size={filter_size}:phase_shift=10:cutoff=0.96"
        )
    return f"lowpass=f={lowpass},{resample}"


def _telephony_af_chain(*, tail_pad_s: float = 0.0, light: bool = False) -> str:
    """Clean 8 kHz mono for Asterisk (never aresample=async=, it rushes tails)."""
    return _telephony_pre_chain(tail_pad_s=tail_pad_s) + "," + _band_chain(8000, light=light)


def _wav_pcm_info(path: Path) -> tuple[int, int, int] | None:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
    except wave.Error:
        return None


def _run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode(errors="replace")[-800:]
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err}")


def _ffmpeg_to_asterisk_wav(
    src: Path, dst: Path, *, tail_pad_s: float = 0.0
) -> None:
    """
    Convert Supertonic WAV to 8 kHz 16-bit mono PCM for Asterisk.

    With TTS_WIDEBAND, also writes a 16 kHz ``.wav16`` twin; Asterisk plays it
    instead of the 8 kHz file when the call uses a wideband codec (G.722).
    """
    light = _bool_env("TTS_FFMPEG_LIGHT", True)
    sr = _probe_sample_rate(src) or (_wav_pcm_info(src) or (None,))[0]
    if sr:
        log.debug("TTS postprocess %s Hz -> telephony (light=%s)", sr, light)

    base_cmd = ["ffmpeg", "-y", "-nostdin", "-threads", "2", "-i", str(src)]
    pcm = ["-ac", "1", "-c:a", "pcm_s16le"]
    try:
        if wideband_enabled():
            graph = (
                f"[0:a]{_telephony_pre_chain(tail_pad_s=tail_pad_s)},asplit=2[nb][wb];"
                f"[nb]{_band_chain(8000, light=light)}[o8];"
                f"[wb]{_band_chain(16000, light=light)}[o16]"
            )
            _run_ffmpeg(
                base_cmd
                + ["-filter_complex", graph]
                + ["-map", "[o8]", *pcm, str(dst)]
                + ["-map", "[o16]", *pcm, "-f", "wav", str(wideband_path(dst))]
            )
        else:
            af = _telephony_af_chain(tail_pad_s=tail_pad_s, light=light)
            _run_ffmpeg(base_cmd + ["-af", af, *pcm, str(dst)])
    except RuntimeError:
        wideband_path(dst).unlink(missing_ok=True)
        _run_ffmpeg(
            base_cmd
            + [
                "-ar",
                "8000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(dst),
            ]
        )


async def _edge_synthesize_async(
    text: str,
    out_wav: Path,
    *,
    voice_name: str,
    rate: str,
    lang: str,
    tail_pad_s: float = 0.0,
) -> None:
    import edge_tts

    plain = _tts_plain_text(text, session_lang=lang)
    if not plain:
        raise ValueError("TTS text is empty after cleanup")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        raw_wav = Path(tmp.name)
    try:
        log.info(
            "Edge TTS synthesizing voice=%s rate=%s chars=%s preview=%r",
            voice_name,
            rate,
            len(plain),
            plain[:60],
        )
        t0 = time.perf_counter()
        comm = edge_tts.Communicate(plain, voice_name, rate=rate)
        await comm.save(str(raw_wav))
        log.info("Edge TTS done in %.1fs", time.perf_counter() - t0)
        _ffmpeg_to_asterisk_wav(raw_wav, out_wav, tail_pad_s=tail_pad_s)
        log.info("Edge TTS WAV ready for Asterisk: %s", out_wav)
    finally:
        raw_wav.unlink(missing_ok=True)


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
    plain = _tts_plain_text(text, session_lang=lang)
    if not plain:
        raise ValueError("TTS text is empty after cleanup")

    from ari_app.inference_pool import run_tts

    def _do_synth() -> None:
        engine = _ensure_engine()
        style = _get_voice_style(engine, voice_name)

        use_speed = _clamp_speed(speed, lang=lang)
        use_silence = max(0.08, min(float(silence_duration), 0.40))

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

    run_tts(_do_synth)


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
    edge_voice_en: str | None = None,
    edge_voice_hi: str | None = None,
    edge_rate: str | None = None,
) -> None:
    """Synthesize with Supertonic or Edge TTS; lang hi/en from text + call session."""
    resolved_lang = lang or pick_supertonic_lang(text, session_lang=session_lang)
    resolved_speed = speed_for_lang(
        resolved_lang,
        speed_en=speed_en if speed_en is not None else speed,
        speed_hi=speed_hi if speed_hi is not None else speed,
        speed_default=speed,
    )

    backend = _tts_backend()
    if backend == "edge":
        ev_en = (edge_voice_en or os.getenv("EDGE_TTS_VOICE_EN") or "en-IN-NeerjaNeural").strip()
        ev_hi = (edge_voice_hi or os.getenv("EDGE_TTS_VOICE_HI") or "hi-IN-SwaraNeural").strip()
        edge_v = ev_hi if resolved_lang == "hi" else ev_en
        rate = (edge_rate or os.getenv("EDGE_TTS_RATE") or _speed_to_edge_rate(resolved_speed)).strip()
        await _edge_synthesize_async(
            text,
            out_wav,
            voice_name=edge_v,
            rate=rate,
            lang=resolved_lang,
            tail_pad_s=tail_pad_s,
        )
        return

    resolved_voice = voice_name or pick_supertonic_voice(
        text,
        voice_en=voice_en,
        voice_hi=voice_hi,
        session_lang=resolved_lang,
    )

    from ari_app.inference_pool import get_executor

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        get_executor(),
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
