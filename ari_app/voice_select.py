"""Pick Supertonic voice + opening line from AGENT_GENDER in config."""

from __future__ import annotations

import random
from datetime import datetime

from ari_app.config import Settings

_OPENING_EN: tuple[str, ...] = (
    "Hi, how can I help you today?",
    "Hello, welcome. What can I do for you?",
    "Hi there. How may I help you?",
    "Hello. Tell me what you need.",
)

_OPENING_HI: tuple[str, ...] = (
    "नमस्ते, बताइए मैं आपकी क्या मदद कर सकती हूँ?",
    "नमस्ते, स्वागत है। आज क्या जानना चाहेंगे?",
    "हैलो, मैं आपकी सहायता के लिए यहाँ हूँ।",
    "नमस्ते, आपकी क्या सेवा करूँ?",
)

_OPENING_HI_NEUTRAL: tuple[str, ...] = (
    "नमस्ते, स्वागत है। बताइए, क्या मदद चाहिए?",
    "नमस्ते। आज क्या जानना चाहेंगे?",
    "हैलो, स्वागत है।",
)


def supertonic_voice_for_agent(settings: Settings, agent_gender: str) -> tuple[str, str]:
    """Return (voice_en, voice_hi). Default female (F3); male (M1) only if AGENT_GENDER=male."""
    if agent_gender == "male":
        v = settings.tts_voice_male
    else:
        v = settings.tts_voice_female
    return v, (settings.tts_voice_hi or v)


def _time_of_day_bucket() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    return "evening"


def _dynamic_opening_en(*, caller_name: str | None) -> str:
    name = (caller_name or "").strip()
    if name and name not in ("-", "unknown"):
        return random.choice(
            (
                f"Hi {name}, how can I help you today?",
                f"Hello {name}, what can I do for you?",
            )
        )
    bucket = _time_of_day_bucket()
    if bucket == "morning":
        return random.choice(
            (
                "Good morning. How can I help you today?",
                "Hi, good morning. What can I do for you?",
            )
        )
    if bucket == "evening":
        return random.choice(
            (
                "Good evening. How may I help you?",
                "Hi, good evening. What do you need?",
            )
        )
    return random.choice(_OPENING_EN)


def _dynamic_opening_hi(grammar_gender: str, *, caller_name: str | None) -> str:
    name = (caller_name or "").strip()
    if name and name not in ("-", "unknown"):
        return random.choice(
            (
                f"नमस्ते {name}, बताइए मैं क्या मदद कर सकती हूँ?",
                f"नमस्ते {name}, आपकी क्या सेवा करूँ?",
            )
        )
    if grammar_gender == "male":
        return random.choice(
            (
                "नमस्ते, बताइए मैं आपकी क्या मदद कर सकता हूँ?",
                "नमस्ते, स्वागत है। आज क्या जानना चाहेंगे?",
            )
        )
    if grammar_gender == "female":
        return random.choice(_OPENING_HI)
    return random.choice(_OPENING_HI_NEUTRAL)


def opening_greeting_for_agent(
    settings: Settings,
    grammar_gender: str,
    lang: str,
    *,
    caller_name: str | None = None,
) -> str:
    """Opening line: fixed OPENING_GREETING from env, or a varied line each call."""
    custom = (settings.opening_greeting or "").strip()
    if custom and not settings.opening_greeting_dynamic:
        return custom
    if custom and settings.opening_greeting_dynamic:
        return custom
    if lang == "en":
        return _dynamic_opening_en(caller_name=caller_name)
    return _dynamic_opening_hi(grammar_gender, caller_name=caller_name)


def opening_greeting_for_call(
    settings: Settings,
    lang: str,
    *,
    neutral_until_confident: bool,
    caller_name: str | None = None,
) -> str:
    """Opening: neutral Hindi when policy requires; voice gender does not affect wording."""
    grammar = "neutral" if neutral_until_confident else (
        settings.agent_gender
        if settings.agent_gender in ("male", "female", "neutral")
        else "neutral"
    )
    return opening_greeting_for_agent(
        settings,
        grammar,
        lang,
        caller_name=caller_name,
    )
