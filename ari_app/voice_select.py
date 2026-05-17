"""Pick Supertonic voice + opening line from AGENT_GENDER in config."""

from __future__ import annotations

from ari_app.config import Settings


def supertonic_voice_for_agent(settings: Settings, agent_gender: str) -> tuple[str, str]:
    """Return (voice_en, voice_hi). Default female (F3); male (M1) only if AGENT_GENDER=male."""
    if agent_gender == "male":
        v = settings.tts_voice_male
    else:
        v = settings.tts_voice_female
    return v, (settings.tts_voice_hi or v)


def opening_greeting_for_agent(
    settings: Settings,
    grammar_gender: str,
    lang: str,
) -> str:
    """Opening line text. Use grammar_gender=neutral at call start for unbiased Hindi."""
    custom = (settings.opening_greeting or "").strip()
    if custom:
        return custom
    if lang == "en":
        return "Hi, how can I help you today?"
    if grammar_gender == "male":
        return "नमस्ते, बताइए मैं क्या मदद कर सकता हूँ?"
    if grammar_gender == "female":
        return "नमस्ते, बताइए मैं क्या मदद कर सकती हूँ?"
    return "नमस्ते, बताइए—आज क्या मदद चाहिए?"


def opening_greeting_for_call(
    settings: Settings,
    lang: str,
    *,
    neutral_until_confident: bool,
) -> str:
    """Opening: neutral Hindi when policy requires; voice gender does not affect wording."""
    grammar = "neutral" if neutral_until_confident else (
        settings.agent_gender
        if settings.agent_gender in ("male", "female", "neutral")
        else "neutral"
    )
    return opening_greeting_for_agent(settings, grammar, lang)
