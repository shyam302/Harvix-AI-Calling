"""Pick Supertonic voice + opening line from agent gender (mirrors caller)."""

from __future__ import annotations

from ari_app.config import Settings


def supertonic_voice_for_agent(settings: Settings, agent_gender: str) -> tuple[str, str]:
    """Return (voice_en, voice_hi) for male or female agent."""
    if agent_gender == "male":
        v = settings.tts_voice_male
    else:
        v = settings.tts_voice_female
    return v, (settings.tts_voice_hi or v)


def opening_greeting_for_agent(
    settings: Settings,
    agent_gender: str,
    lang: str,
) -> str:
    if lang == "en":
        if agent_gender == "male":
            return "Hi, how can I help you?"
        return settings.opening_greeting if _is_latin(settings.opening_greeting) else (
            "Hi, how can I help you?"
        )
    if agent_gender == "male":
        return "नमस्ते, बताइए मैं क्या मदद कर सकता हूँ?"
    custom = settings.opening_greeting
    if _has_devanagari(custom):
        return custom
    return "नमस्ते, बताइए मैं क्या मदद कर सकती हूँ?"


def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def _is_latin(text: str) -> bool:
    return bool(text) and not _has_devanagari(text)
