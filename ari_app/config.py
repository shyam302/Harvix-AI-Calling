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
    gender_detect_caller: bool  # estimate pitch each turn
    gender_adapt_when_confident: bool  # tailor replies to caller only when pitch is confident
    gender_mirror_caller: bool  # also switch TTS voice to M1/F3 when confident
    gender_mirror_grammar: bool  # deprecated alias for gender_adapt_when_confident
    gender_min_confident_samples: int
    gender_confident_fraction: float
    agent_gender: str  # neutral | female | male — bot grammar (default neutral)
    agent_voice_gender: str  # female | male — TTS voice (AGENT_VOICE_GENDER)
    grammar_match_voice: bool  # if true and AGENT_GENDER=neutral, match voice (usually false)
    gender_grammar_until_confident: bool  # Hindi neutral until caller pitch is confident
    gender_grammar_min_turns: int  # user turns before persona grammar unlocks
    llm_temperature: float
    opening_greeting: str
    record_max_silence_seconds: float
    prompt_on_silence: bool  # if false, stay on mic (no "say again?") on empty/silent turns
    record_max_duration_seconds: int
    pause_after_tts_ms: int
    pause_before_listen_ms: int
    pause_before_response_ms: int
    record_min_duration_seconds: float
    llm_max_tokens: int
    reply_max_sentences: int
    reply_max_chars: int
    supertonic_speed: float
    supertonic_speed_greeting: float  # opening line only (often English)
    supertonic_speed_en: float
    supertonic_speed_hi: float  # conversation replies in Hindi
    supertonic_silence_duration: float
    tts_tail_pad_seconds: float
    tts_post_play_settle_ms: int  # wait after PlaybackFinished (echo / ghost on mic)
    tts_chunk_max_chars: int
    session_max_messages: int
    call_primary_lang: str  # hi | en — default when turn is ambiguous
    lang_switch_min_en_words: int
    lang_switch_consecutive_en: int
    continuous_conversation: bool  # streaming LLM + short gaps (less walkie-talkie)
    llm_streaming: bool  # stream tokens from vLLM when continuous mode on
    continuous_record_max_silence_seconds: float
    continuous_pause_after_tts_ms: int
    continuous_pause_before_listen_ms: int
    continuous_pause_before_response_ms: int
    continuous_tts_tail_pad_seconds: float
    whisper_max_concurrent: int
    tts_max_concurrent: int
    inference_thread_workers: int  # 0 = auto from slots


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


def _agent_gender_from_env() -> str:
    explicit = (os.getenv("AGENT_GENDER") or os.getenv("AGENT_DEFAULT_GENDER") or "").strip()
    if explicit:
        raw = explicit.lower()
        if raw in ("male", "female", "neutral"):
            return raw
    return "neutral"


def _agent_voice_gender_from_env() -> str:
    raw = (os.getenv("AGENT_VOICE_GENDER") or "female").strip().lower()
    return raw if raw in ("male", "female") else "female"


def _whisper_language_from_env() -> str | None:
    raw = os.getenv("WHISPER_LANGUAGE", "auto").strip().lower()
    if raw in ("", "auto", "none"):
        return None
    return raw


def _default_whisper_max_concurrent() -> int:
    raw = os.getenv("WHISPER_MAX_CONCURRENT", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    from ari_app.stt import _whisper_device

    model = (os.getenv("WHISPER_MODEL") or "medium").strip().lower()
    if "large" in model:
        return 2 if _whisper_device() == "cuda" else 1
    return 3 if _whisper_device() == "cuda" else 2


def load_settings() -> Settings:
    whisper_slots = _default_whisper_max_concurrent()
    tts_slots = max(1, _int_env("TTS_MAX_CONCURRENT", 2))
    thread_workers = _int_env("INFERENCE_THREAD_WORKERS", 0)
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
        whisper_model=os.getenv("WHISPER_MODEL", "medium"),
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
        gender_female_min_hz=_float_env("GENDER_PITCH_FEMALE_MIN_HZ", 170.0),
        gender_male_max_hz=_float_env("GENDER_PITCH_MALE_MAX_HZ", 145.0),
        gender_min_pitch_samples=_int_env("GENDER_MIN_PITCH_SAMPLES", 3),
        gender_detect_caller=os.getenv("GENDER_DETECT_CALLER", "true").strip().lower()
        not in ("0", "false", "no", "off"),
        gender_adapt_when_confident=os.getenv(
            "GENDER_ADAPT_WHEN_CONFIDENT", "true"
        ).strip().lower()
        not in ("0", "false", "no", "off"),
        gender_mirror_caller=os.getenv("GENDER_MIRROR_CALLER", "false").strip().lower()
        in ("1", "true", "yes", "on"),
        gender_mirror_grammar=os.getenv("GENDER_MIRROR_GRAMMAR", "false").strip().lower()
        in ("1", "true", "yes", "on"),
        gender_min_confident_samples=_int_env("GENDER_MIN_CONFIDENT_SAMPLES", 5),
        gender_confident_fraction=_float_env("GENDER_CONFIDENT_FRACTION", 0.8),
        agent_gender=_agent_gender_from_env(),
        agent_voice_gender=_agent_voice_gender_from_env(),
        grammar_match_voice=os.getenv("GRAMMAR_MATCH_VOICE", "false").strip().lower()
        in ("1", "true", "yes", "on"),
        gender_grammar_until_confident=os.getenv(
            "GENDER_GRAMMAR_UNTIL_CONFIDENT", "true"
        ).strip().lower()
        not in ("0", "false", "no", "off"),
        gender_grammar_min_turns=_int_env("GENDER_GRAMMAR_MIN_TURNS", 2),
        llm_temperature=_float_env("LLM_TEMPERATURE", 0.82),
        opening_greeting=(
            os.getenv("OPENING_GREETING") or "Hi, how can I help?"
        ).strip()
        or "Hi, how can I help?",
        record_max_silence_seconds=_float_env("RECORD_MAX_SILENCE_SECONDS", 1.4),
        prompt_on_silence=os.getenv("PROMPT_ON_SILENCE", "false").strip().lower()
        in ("1", "true", "yes", "on"),
        record_max_duration_seconds=_int_env("RECORD_MAX_DURATION_SECONDS", 45),
        pause_after_tts_ms=_int_env("PAUSE_AFTER_TTS_MS", 200),
        pause_before_listen_ms=_int_env("PAUSE_BEFORE_LISTEN_MS", 250),
        pause_before_response_ms=_int_env("PAUSE_BEFORE_RESPONSE_MS", 120),
        record_min_duration_seconds=_float_env("RECORD_MIN_DURATION_SECONDS", 0.35),
        llm_max_tokens=_int_env("LLM_MAX_TOKENS", 120),
        reply_max_sentences=_int_env("REPLY_MAX_SENTENCES", 2),
        reply_max_chars=_int_env("REPLY_MAX_CHARS", 260),
        supertonic_speed=_float_env("SUPERTONIC_SPEED", 1.08),
        supertonic_speed_greeting=_float_env(
            "SUPERTONIC_SPEED_GREETING",
            _float_env("SUPERTONIC_SPEED_EN", 1.06),
        ),
        supertonic_speed_en=_float_env("SUPERTONIC_SPEED_EN", 1.06),
        supertonic_speed_hi=_float_env("SUPERTONIC_SPEED_HI", 1.08),
        supertonic_silence_duration=_float_env("SUPERTONIC_SILENCE_DURATION", 0.12),
        tts_tail_pad_seconds=_float_env("TTS_TAIL_PAD_SECONDS", 0.0),
        tts_post_play_settle_ms=_int_env("TTS_POST_PLAY_SETTLE_MS", 150),
        tts_chunk_max_chars=_int_env("TTS_CHUNK_MAX_CHARS", 280),
        session_max_messages=_int_env("SESSION_MAX_MESSAGES", 20),
        call_primary_lang=_call_primary_lang_from_env(),
        lang_switch_min_en_words=_int_env("LANG_SWITCH_MIN_EN_WORDS", 4),
        lang_switch_consecutive_en=_int_env("LANG_SWITCH_CONSECUTIVE_EN", 1),
        continuous_conversation=os.getenv(
            "CONTINUOUS_CONVERSATION", "false"
        ).strip().lower()
        in ("1", "true", "yes", "on"),
        llm_streaming=os.getenv("LLM_STREAMING", "false").strip().lower()
        in ("1", "true", "yes", "on"),
        continuous_record_max_silence_seconds=_float_env(
            "CONTINUOUS_RECORD_MAX_SILENCE_SECONDS", 0.85
        ),
        continuous_pause_after_tts_ms=_int_env("CONTINUOUS_PAUSE_AFTER_TTS_MS", 40),
        continuous_pause_before_listen_ms=_int_env(
            "CONTINUOUS_PAUSE_BEFORE_LISTEN_MS", 40
        ),
        continuous_pause_before_response_ms=_int_env(
            "CONTINUOUS_PAUSE_BEFORE_RESPONSE_MS", 0
        ),
        continuous_tts_tail_pad_seconds=_float_env(
            "CONTINUOUS_TTS_TAIL_PAD_SECONDS", 0.06
        ),
        whisper_max_concurrent=whisper_slots,
        tts_max_concurrent=tts_slots,
        inference_thread_workers=thread_workers,
    )
