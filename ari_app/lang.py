"""Detect Hindi vs English for LLM replies and Supertonic TTS lang=hi/en."""

from __future__ import annotations

import re

# Common Romanized Hindi on phone calls (Whisper often outputs this without Devanagari).
_ROMAN_HI_WORDS = frozenset(
    {
        "aap",
        "aapka",
        "aapki",
        "aapke",
        "mujhe",
        "mujhse",
        "tumhe",
        "tumhara",
        "kya",
        "kyun",
        "kyon",
        "kaise",
        "kab",
        "kahan",
        "nahin",
        "nahi",
        "haan",
        "han",
        "ji",
        "theek",
        "thik",
        "achha",
        "accha",
        "batao",
        "bataiye",
        "boliye",
        "bolo",
        "suniye",
        "suno",
        "samajh",
        "samjha",
        "samjhi",
        "madad",
        "dhanyavad",
        "shukriya",
        "namaste",
        "kaun",
        "kuch",
        "bahut",
        "abhi",
        "kal",
        "aaj",
        "ghar",
        "paise",
        "hain",
        "hai",
        "ho",
        "hoon",
        "hun",
        "tha",
        "thi",
        "the",
        "mera",
        "meri",
        "mere",
        "apka",
        "apki",
        "apke",
        "yeh",
        "ye",
        "woh",
        "wo",
        "karo",
        "kariye",
        "chahiye",
        "sakti",
        "sakta",
        "sakte",
        "bhai",
        "didi",
        "sir",
        "madam",
        "problem",
    }
)

# User clearly wants English (switch immediately, even on a Hindi-primary call).
_EXPLICIT_EN_PHRASES = frozenset(
    {
        "continue",
        "continued",
        "go on",
        "go ahead",
        "carry on",
        "keep going",
        "tell me more",
        "say more",
        "in english",
        "speak english",
        "talk in english",
        "talking in english",
        "talk english",
        "talking english",
        "speak in english",
        "speaking english",
        "speaking in english",
        "english please",
        "please speak english",
        "please talk in english",
        "use english",
        "switch to english",
        "can you speak english",
        "can you talk in english",
        "i want english",
        "want english",
        "only english",
        "english mein",
        "english me",
        "angrezi mein",
        "angrezi me",
    }
)

# User clearly wants Hindi (even if they ask in English words).
_EXPLICIT_HI_PHRASES = frozenset(
    {
        "in hindi",
        "speak hindi",
        "talk in hindi",
        "talking in hindi",
        "talk hindi",
        "talking hindi",
        "speak in hindi",
        "speaking hindi",
        "speaking in hindi",
        "hindi please",
        "please speak hindi",
        "please talk in hindi",
        "use hindi",
        "switch to hindi",
        "can you speak hindi",
        "can you talk in hindi",
        "can we talk in hindi",
        "i want hindi",
        "want hindi",
        "only hindi",
        "hindi mein",
        "hindi me",
        "in hindhi",
        "talk in hindhi",
    }
)

# Hindi script: "speak in English", etc.
_DEVANAGARI_ENGLISH_RE = re.compile(
    r"अंग्रेज[ीज़]|इंग्लिश|इंग्लीश", re.IGNORECASE
)
_DEVANAGARI_HINDI_RE = re.compile(r"हिंदी|हिन्दी", re.IGNORECASE)
_DEVANAGARI_SPEAK_RE = re.compile(r"बोल|बात|कर|सुन", re.IGNORECASE)

# Short English — do not flip a Hindi call on these alone (not include "continue").
_SHORT_EN_KEEP_HI = frozenset(
    {
        "yes",
        "no",
        "ok",
        "okay",
        "yeah",
        "yep",
        "nope",
        "hi",
        "hello",
        "bye",
        "thanks",
        "thank",
        "you",
        "please",
        "sorry",
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
    }
)


def text_contains_devanagari(text: str) -> bool:
    for ch in text or "":
        if "\u0900" <= ch <= "\u097f":
            return True
    return False


def _latin_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", (text or "").lower())


def _roman_hindi_score(text: str) -> int:
    words = _latin_words(text)
    if not words:
        return 0
    return sum(1 for w in words if w in _ROMAN_HI_WORDS)


def _devanagari_english_intent(text: str) -> bool:
    if not text_contains_devanagari(text):
        return False
    return bool(_DEVANAGARI_ENGLISH_RE.search(text)) and bool(
        _DEVANAGARI_SPEAK_RE.search(text)
    )


def _devanagari_hindi_intent(text: str) -> bool:
    if not text_contains_devanagari(text):
        return False
    if _devanagari_english_intent(text):
        return False
    return bool(_DEVANAGARI_HINDI_RE.search(text)) and bool(
        _DEVANAGARI_SPEAK_RE.search(text)
    )


def _latin_hindi_intent(phrase: str) -> bool:
    """'talk in hindi', 'can we speak hindi', etc."""
    if not phrase or "hindi" not in phrase and "hindhi" not in phrase:
        return False
    if re.search(
        r"\b(talk|speak|switch|use|want|only|bolo|boliye|baat|karo|kariye)\b",
        phrase,
    ):
        return True
    if re.search(r"\b(in|to)\s+hind?hi\b", phrase):
        return True
    if re.search(r"\bhind?hi\s+(please|mein|me|only)\b", phrase):
        return True
    if re.search(r"\b(can we|could we|i just want|i want to)\b", phrase):
        return True
    return False


def _latin_english_intent(phrase: str) -> bool:
    """'talk english', 'please use english', etc."""
    if not phrase or "english" not in phrase and "angrezi" not in phrase:
        return False
    if re.search(
        r"\b(talk|speak|switch|use|continue|want|only|bolo|boliye|baat|karo|kariye)\b",
        phrase,
    ):
        return True
    if re.search(r"\b(in|to)\s+english\b", phrase):
        return True
    if re.search(r"\benglish\s+(please|mein|me|only)\b", phrase):
        return True
    return False


def looks_like_hindi(text: str, *, whisper_lang: str | None = None) -> bool:
    if is_explicit_english_intent(text):
        return False
    if text_contains_devanagari(text):
        if _devanagari_english_intent(text):
            return False
        return True
    wl = (whisper_lang or "").strip().lower()[:2]
    if wl == "en":
        return False
    hits = _roman_hindi_score(text)
    word_count = len(re.findall(r"\w+", text or ""))
    if hits >= 2:
        return True
    if hits >= 1 and word_count <= 10:
        return True
    # Whisper forced to hi can tag English as hi — do not lock session on tag alone.
    if wl == "hi" and hits >= 1:
        return True
    return False


def _normalized_phrase(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_explicit_hindi_intent(text: str) -> bool:
    """e.g. 'talk in hindi', 'hindi mein bolo' → switch to Hindi this turn."""
    if _devanagari_hindi_intent(text):
        return True
    phrase = _normalized_phrase(text)
    if not phrase:
        return False
    if phrase in _EXPLICIT_HI_PHRASES:
        return True
    for key in _EXPLICIT_HI_PHRASES:
        if len(key) > 6 and key in phrase:
            return True
    return _latin_hindi_intent(phrase)


def is_explicit_english_intent(text: str) -> bool:
    """e.g. 'continue', 'go on', 'speak in english' → switch to English this turn."""
    if is_explicit_hindi_intent(text):
        return False
    if _devanagari_english_intent(text):
        return True
    phrase = _normalized_phrase(text)
    if not phrase:
        return False
    if phrase in _EXPLICIT_EN_PHRASES:
        return True
    for key in _EXPLICIT_EN_PHRASES:
        if len(key) > 6 and key in phrase:
            return True
    return _latin_english_intent(phrase)


def _is_short_english_utterance(text: str) -> bool:
    if is_explicit_english_intent(text):
        return False
    words = _latin_words(text)
    if not words or len(words) > 4:
        return False
    return all(w in _SHORT_EN_KEEP_HI for w in words)


def clear_english_switch(
    text: str,
    *,
    whisper_lang: str | None = None,
    min_en_words: int = 4,
) -> bool:
    """
    Strong signal the caller switched to English (not Whisper noise on a Hindi call).
    Requires enough English words; ignores short 'ok/yes/thanks' and Romanized Hindi.
    """
    if is_explicit_hindi_intent(text):
        return False
    if text_contains_devanagari(text):
        return False
    if looks_like_hindi(text, whisper_lang=whisper_lang):
        return False
    if _is_short_english_utterance(text):
        return False
    words = _latin_words(text)
    if len(words) < min_en_words:
        return False
    if _roman_hindi_score(text) > 0:
        return False
    wl = (whisper_lang or "").strip().lower()[:2]
    if wl == "en" and len(words) >= max(2, min_en_words - 1):
        return True
    if wl and wl not in ("en",):
        return False
    return True


def detect_turn_lang(
    text: str,
    *,
    whisper_lang: str | None = None,
    primary_lang: str = "hi",
) -> str:
    if looks_like_hindi(text, whisper_lang=whisper_lang):
        return "hi"
    if clear_english_switch(text, whisper_lang=whisper_lang):
        return "en"
    return primary_lang if primary_lang in ("hi", "en") else "hi"


def update_preferred_lang(
    current: str,
    user_text: str,
    *,
    whisper_lang: str | None = None,
    primary_lang: str = "hi",
    consecutive_en_signals: int = 0,
    min_en_words: int = 4,
    consecutive_en_required: int = 2,
) -> tuple[str, int]:
    """
    Sticky session language with hysteresis.

    - Hindi call (primary hi): stay on hi unless English is clear for several turns.
    - One Hindi/Devanagari/Roman-Hindi utterance immediately returns to hi.
    - Explicit English ("continue", "speak english") switches immediately.
    - Explicit Hindi ("talk in hindi") switches immediately, even if spoken in English.
    """
    if is_explicit_hindi_intent(user_text):
        return "hi", 0

    if is_explicit_english_intent(user_text):
        return "en", 0

    if looks_like_hindi(user_text, whisper_lang=whisper_lang):
        return "hi", 0

    if clear_english_switch(
        user_text, whisper_lang=whisper_lang, min_en_words=min_en_words
    ):
        consecutive_en_signals += 1
    else:
        consecutive_en_signals = 0

    if primary_lang == "hi":
        if current == "en":
            # Stay in English through short replies until Hindi is detected (explicit or content).
            return "en", consecutive_en_signals
        if consecutive_en_signals >= consecutive_en_required:
            return "en", consecutive_en_signals
        return "hi", consecutive_en_signals

    # Primary English: switch to hi immediately when Hindi detected (above).
    if consecutive_en_signals > 0 or clear_english_switch(
        user_text, whisper_lang=whisper_lang, min_en_words=min_en_words
    ):
        return "en", 0
    return current or "en", 0


def whisper_language_hint(
    session_lang: str,
    *,
    configured: str | None,
) -> str | None:
    """
    When WHISPER_LANGUAGE=auto, return None so Whisper detects each utterance.
    Session language (hi/en) still drives LLM + TTS; locking Whisper to hi
    blocked phrases like "talk in english" (wrong transcript + looks_like_hindi).
    Set WHISPER_LANGUAGE=hi or =en to force a fixed STT language.
    """
    if configured:
        return configured
    return None


def resolve_tts_lang(
    text: str,
    *,
    session_lang: str | None = None,
) -> str:
    if text_contains_devanagari(text):
        return "hi"
    if session_lang == "en":
        return "en"
    if session_lang == "hi":
        # Latin-only text (e.g. English opening) uses English TTS; Devanagari uses hi.
        if _latin_words(text) and _roman_hindi_score(text) == 0:
            return "en"
        return "hi"
    return "hi" if looks_like_hindi(text) else "en"


def reply_should_be_devanagari(reply: str, reply_lang: str) -> bool:
    """True if LLM slipped to English/Latin while session expects Hindi."""
    if reply_lang != "hi":
        return True
    if text_contains_devanagari(reply):
        return True
    latin = _latin_words(reply)
    if not latin:
        return True
    return len(latin) < 3
