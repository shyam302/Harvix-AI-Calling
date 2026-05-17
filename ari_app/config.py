"""Load settings from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ari_host: str
    ari_port: int
    ari_username: str
    ari_password: str
    stasis_app: str
    vllm_base_url: str
    vllm_api_key: str
    vllm_model: str
    vllm_timeout_seconds: float
    vllm_connect_timeout_seconds: float
    whisper_model: str
    whisper_language: str | None  # None = auto-detect; "en" / "hi" / etc.
    tts_sound_subdir: str
    tts_voice: str  # fallback Supertonic voice if gender voices unset
    tts_voice_hi: str
    tts_voice_female: str
    tts_voice_male: str
    gender_female_min_hz: float
    gender_male_max_hz: float
    gender_min_pitch_samples: int
    opening_greeting: str
    record_max_silence_seconds: float
    record_max_duration_seconds: int
    pause_after_tts_ms: int
    pause_before_listen_ms: int
    pause_before_response_ms: int
    record_min_duration_seconds: float
    llm_max_tokens: int
    reply_max_sentences: int
    reply_max_chars: int
    supertonic_speed: float
    supertonic_speed_en: float
    supertonic_speed_hi: float
    supertonic_silence_duration: float
    tts_chunk_max_chars: int
    session_max_messages: int
    call_primary_lang: str  # hi | en — default when turn is ambiguous
    lang_switch_min_en_words: int
    lang_switch_consecutive_en: int


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _call_primary_lang_from_env() -> str:
    raw = (os.getenv("CALL_PRIMARY_LANG", "hi") or "hi").strip().lower()
    return raw if raw in ("hi", "en") else "hi"


def _whisper_language_from_env() -> str | None:
    raw = os.getenv("WHISPER_LANGUAGE", "auto").strip().lower()
    if raw in ("", "auto", "none"):
        return None
    return raw


def load_settings() -> Settings:
    return Settings(
        ari_host=os.getenv("ARI_HOST", "127.0.0.1"),
        ari_port=int(os.getenv("ARI_PORT", "8088")),
        ari_username=os.getenv("ARI_USERNAME", "callbot"),
        ari_password=os.getenv("ARI_PASSWORD", ""),
        stasis_app=os.getenv("STASIS_APP", "callbot"),
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        vllm_api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        vllm_model=os.getenv("VLLM_MODEL", "google/gemma-4-E2B-it"),
        vllm_timeout_seconds=_float_env("VLLM_TIMEOUT_SECONDS", 120.0),
        vllm_connect_timeout_seconds=_float_env("VLLM_CONNECT_TIMEOUT_SECONDS", 15.0),
        whisper_model=os.getenv("WHISPER_MODEL", "large-v3"),
        whisper_language=_whisper_language_from_env(),
        tts_sound_subdir=os.getenv("TTS_SOUND_SUBDIR", "custom"),
        tts_voice=(
            os.getenv("SUPERTONIC_VOICE_EN")
            or os.getenv("EDGE_TTS_VOICE")
            or "F3"
        ).strip()
        or "F3",
        tts_voice_hi=(
            os.getenv("SUPERTONIC_VOICE_HI")
            or os.getenv("EDGE_TTS_VOICE_HI")
            or ""
        ).strip(),
        tts_voice_female=(
            os.getenv("SUPERTONIC_VOICE_FEMALE")
            or os.getenv("SUPERTONIC_VOICE_EN")
            or "F3"
        ).strip()
        or "F3",
        tts_voice_male=(
            os.getenv("SUPERTONIC_VOICE_MALE") or "M1"
        ).strip()
        or "M1",
        gender_female_min_hz=_float_env("GENDER_PITCH_FEMALE_MIN_HZ", 165.0),
        gender_male_max_hz=_float_env("GENDER_PITCH_MALE_MAX_HZ", 155.0),
        gender_min_pitch_samples=_int_env("GENDER_MIN_PITCH_SAMPLES", 2),
        opening_greeting=(
            os.getenv("OPENING_GREETING") or "Hi, how can I help?"
        ).strip()
        or "Hi, how can I help?",
        record_max_silence_seconds=_float_env("RECORD_MAX_SILENCE_SECONDS", 1.0),
        record_max_duration_seconds=_int_env("RECORD_MAX_DURATION_SECONDS", 45),
        pause_after_tts_ms=_int_env("PAUSE_AFTER_TTS_MS", 350),
        pause_before_listen_ms=_int_env("PAUSE_BEFORE_LISTEN_MS", 450),
        pause_before_response_ms=_int_env("PAUSE_BEFORE_RESPONSE_MS", 400),
        record_min_duration_seconds=_float_env("RECORD_MIN_DURATION_SECONDS", 0.35),
        llm_max_tokens=_int_env("LLM_MAX_TOKENS", 96),
        reply_max_sentences=_int_env("REPLY_MAX_SENTENCES", 2),
        reply_max_chars=_int_env("REPLY_MAX_CHARS", 220),
        supertonic_speed=_float_env("SUPERTONIC_SPEED", 1.08),
        supertonic_speed_en=_float_env("SUPERTONIC_SPEED_EN", 1.22),
        supertonic_speed_hi=_float_env("SUPERTONIC_SPEED_HI", 1.08),
        supertonic_silence_duration=_float_env("SUPERTONIC_SILENCE_DURATION", 0.12),
        tts_chunk_max_chars=_int_env("TTS_CHUNK_MAX_CHARS", 140),
        session_max_messages=_int_env("SESSION_MAX_MESSAGES", 20),
        call_primary_lang=_call_primary_lang_from_env(),
        lang_switch_min_en_words=_int_env("LANG_SWITCH_MIN_EN_WORDS", 4),
        lang_switch_consecutive_en=_int_env("LANG_SWITCH_CONSECUTIVE_EN", 1),
    )
