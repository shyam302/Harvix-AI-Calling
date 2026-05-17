"""Trim and chunk text so phone replies sound natural and TTS starts sooner."""

from __future__ import annotations

import random
import re

_RETRY_PHRASES_EN = (
    "Sorry, I didn't catch that—say it again?",
    "Hmm, I missed that. One more time?",
    "Still there? Say that again for me.",
)
_RETRY_PHRASES_HI = (
    "माफ़ कीजिए, साफ़ नहीं सुनाई दिया—फिर से बोलिए?",
    "जी, एक बार और बोलिएगा?",
    "आवाज़ कट गई—दोबारा बताइए?",
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

# LLM slip: masculine forms while female voice (F3) — safe phrase-level fixes.
_HI_FEMALE_GRAMMAR_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"कर\s*सकता\s*हूँ", re.I), "कर सकती हूँ"),
    (re.compile(r"कर\s*सकता\s*हूं", re.I), "कर सकती हूं"),
    (re.compile(r"सकता\s*हूँ", re.I), "सकती हूँ"),
    (re.compile(r"सकता\s*हूं", re.I), "सकती हूं"),
    (re.compile(r"करता\s*हूँ", re.I), "करती हूँ"),
    (re.compile(r"करता\s*हूं", re.I), "करती हूं"),
    (re.compile(r"रहता\s*हूँ", re.I), "रहती हूँ"),
    (re.compile(r"गया\s*हूँ", re.I), "गई हूँ"),
    (re.compile(r"समझ\s*गया", re.I), "समझ गई"),
    (re.compile(r"बता\s*दूँगा", re.I), "बता दूँगी"),
    (re.compile(r"बताऊँगा", re.I), "बताऊँगी"),
)

_HI_MALE_GRAMMAR_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"कर\s*सकती\s*हूँ", re.I), "कर सकता हूँ"),
    (re.compile(r"सकती\s*हूँ", re.I), "सकता हूँ"),
    (re.compile(r"करती\s*हूँ", re.I), "करता हूँ"),
    (re.compile(r"समझ\s*गई", re.I), "समझ गया"),
)


_HI_NEUTRAL_GRAMMAR_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"मैं\s+आपकी\s+क्या\s+मदद\s+कर\s*सकत[ाई]\s*हूँ", re.I), "बताइए, क्या मदद चाहिए"),
    (re.compile(r"मैं\s+.*?\s+कर\s*सकत[ाई]\s*हूँ", re.I), "बताइए, क्या मदद चाहिए"),
    (re.compile(r"कर\s*सकत[ाई]\s*हूँ", re.I), "मदद कर सकते हैं"),
    (re.compile(r"करत[ाई]\s*हूँ", re.I), "बताइए"),
    (re.compile(r"रहत[ाई]\s*हूँ", re.I), "ठीक है"),
    (re.compile(r"गय[ाई]\s*हूँ", re.I), "ठीक है"),
    (re.compile(r"समझ\s*गय[ाई]", re.I), "ठीक है"),
    (re.compile(r"बताऊँग[ाई]", re.I), "बताइए"),
)


def align_hindi_bot_grammar(text: str, *, grammar_gender: str) -> str:
    """Fix LLM gender slips to match AGENT_GENDER (female/male/neutral)."""
    if not text:
        return text
    if grammar_gender == "female":
        out = text
        for pattern, repl in _HI_FEMALE_GRAMMAR_FIXES:
            out = pattern.sub(repl, out)
        return out
    if grammar_gender == "male":
        out = text
        for pattern, repl in _HI_MALE_GRAMMAR_FIXES:
            out = pattern.sub(repl, out)
        return out
    if grammar_gender == "neutral":
        out = text
        for pattern, repl in _HI_NEUTRAL_GRAMMAR_FIXES:
            out = pattern.sub(repl, out)
        return out
    return text


_LEADING_FILLER_EN = re.compile(
    r"^(?:(?:"
    r"(?:oh|ah|uh|um|umm|hmm|hm|mmm|yeah|yep)(?:\s*[,!?.\-—]\s*)|"
    r"(?:ok|okay|sure|well|so|right|alright)(?:\s*,\s*)|"
    r"(?:got it|i see|i understand)(?:\s*,\s*)"
    r"))+",
    re.IGNORECASE,
)
_LEADING_FILLER_HI = re.compile(
    r"^(?:(?:ओह|अह|अरे|ह्म्म|हम्म|हां|हाँ|अच्छा|ठीक)(?:\s*[,!?।\-—]\s*))+",
)


def strip_leading_filler(text: str) -> str:
    """Remove oh/ah/hmm-style openers so TTS does not start with grunts."""
    t = (text or "").strip()
    if not t:
        return t
    for _ in range(4):
        prev = t
        t = _LEADING_FILLER_EN.sub("", t).lstrip()
        t = _LEADING_FILLER_HI.sub("", t).lstrip()
        if t == prev:
            break
    return t


def trim_reply_for_phone(
    text: str,
    *,
    max_sentences: int = 2,
    max_chars: int = 220,
) -> str:
    """Keep LLM output short enough for a natural phone turn."""
    t = strip_leading_filler(text or "")
    t = re.sub(r"[*_#`]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return t

    parts = [
        strip_leading_filler(p.strip())
        for p in _SENTENCE_END.split(t)
        if p.strip()
    ]
    parts = [p for p in parts if p]
    if not parts:
        parts = [t] if t else []
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


def pop_complete_sentences(
    buffer: str, *, min_chars: int = 8
) -> tuple[list[str], str]:
    """
    Pull finished sentences from a growing LLM stream buffer.
    Leaves the final incomplete fragment in the remainder.
    """
    text = buffer or ""
    if not text.strip():
        return [], text

    parts = _SENTENCE_END.split(text)
    if len(parts) <= 1:
        if len(text) > 100:
            cut = text.rfind(" ", 0, 90)
            if cut > min_chars:
                return [text[:cut].strip()], text[cut:].lstrip()
        return [], text

    complete = [p.strip() for p in parts[:-1] if p.strip()]
    remainder = parts[-1]
    ready = [s for s in complete if len(s) >= min_chars]
    return ready, remainder


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
