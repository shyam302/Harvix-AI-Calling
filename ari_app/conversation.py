"""Trim and chunk text so phone replies sound natural and TTS starts sooner."""

from __future__ import annotations

import random
import re

_RETRY_PHRASES_EN = (
    "Sorry, I didn't quite catch that. Could you say it again?",
    "I missed that—mind repeating?",
    "Still there? Say that once more for me.",
)
_RETRY_PHRASES_HI = (
    "माफ़ कीजिए, साफ़ नहीं सुनाई दिया। एक बार फिर बोलिए?",
    "जी, दोबारा बोलिएगा?",
    "आपकी आवाज़ कट गई—फिर से बताइए?",
)
_LLM_UNAVAILABLE_EN = (
    "Hang on, I'm having trouble connecting for a second. Try again?",
    "One moment—please say that again.",
)
_LLM_UNAVAILABLE_HI = (
    "एक पल रुकिए, कनेक्शन में दिक्कत है। दोबारा बोलिए?",
    "जी, थोड़ी देर बाद फिर से बताइए?",
)


def pick_retry_phrase(lang: str = "hi") -> str:
    if lang == "hi":
        return random.choice(_RETRY_PHRASES_HI)
    return random.choice(_RETRY_PHRASES_EN)


def pick_llm_unavailable_phrase(lang: str = "hi") -> str:
    if lang == "hi":
        return random.choice(_LLM_UNAVAILABLE_HI)
    return random.choice(_LLM_UNAVAILABLE_EN)


_SENTENCE_END = re.compile(r'(?<=[.!?।])\s+')
# Clause breaks when we must shorten without cutting mid-word.
_CLAUSE_BREAK = re.compile(r'(?<=[,;:—–-])\s+')


def trim_reply_for_phone(
    text: str,
    *,
    max_sentences: int = 2,
    max_chars: int = 220,
) -> str:
    """Keep LLM output short enough for a natural phone turn."""
    t = (text or "").strip()
    t = re.sub(r"[*_#`]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return t

    parts = [p.strip() for p in _SENTENCE_END.split(t) if p.strip()]
    if not parts:
        parts = [t]
    if max_sentences > 0 and len(parts) > max_sentences:
        parts = parts[:max_sentences]
    t = " ".join(parts)
    if len(t) <= max_chars:
        return t

    # Prefer full sentences; otherwise break at commas.
    shortened: list[str] = []
    n = 0
    for sent in parts:
        if n + len(sent) + (1 if shortened else 0) > max_chars:
            break
        shortened.append(sent)
        n += len(sent) + (1 if shortened else 0)
    if shortened:
        return " ".join(shortened)[:max_chars].rstrip()

    for piece in _CLAUSE_BREAK.split(t):
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= max_chars:
            return piece
    return t[: max_chars - 3].rstrip() + "..."


def split_tts_phrases(text: str, *, max_chars: int = 140) -> list[str]:
    """Split at sentence boundaries for chunked TTS (overlap playback with synthesis)."""
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    sentences = [p.strip() for p in _SENTENCE_END.split(t) if p.strip()]
    if not sentences:
        return [t[:max_chars]]

    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        candidate = f"{buf} {sent}".strip() if buf else sent
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        buf = sent if len(sent) <= max_chars else sent[:max_chars]
    if buf:
        chunks.append(buf)
    return chunks or [t[:max_chars]]
