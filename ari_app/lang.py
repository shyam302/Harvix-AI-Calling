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
        "change to english",
        "switch language to english",
        "language english",
        "english language",
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
        "change to hindi",
        "switch language to hindi",
        "language hindi",
        "hindi language",
        "bolo hindi",
        "hindi bolo",
        "hindi me baat karo",
        "hindi mein baat karo",
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


# Latin words that are normal English — do not count as Romanized Hindi.
_ENGLISH_COMMON = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "not", "this", "that", "what", "when", "where",
        "who", "how", "why", "which", "with", "from", "by", "as", "it", "its", "i",
        "you", "he", "she", "we", "they", "my", "your", "his", "her", "our", "their",
        "me", "him", "us", "them", "do", "does", "did", "have", "has", "had", "will",
        "would", "can", "could", "should", "may", "might", "be", "been", "being", "am",
        "please", "thanks", "thank", "hello", "hi", "bye", "today", "tomorrow",
        "yesterday", "gold", "silver", "rate", "price", "india", "live", "search",
        "tell", "know", "want", "need", "help", "about", "per", "gram", "grams",
    }
)


def _roman_hindi_score(text: str) -> int:
    words = _latin_words(text)
    if not words:
        return 0
    return sum(
        1
        for w in words
        if w in _ROMAN_HI_WORDS and w not in _ENGLISH_COMMON
    )


def _devanagari_english_intent(text: str) -> bool:
    """Devanagari request for English — no calls to looks_like_* (avoids recursion)."""
    if not text_contains_devanagari(text):
        return False
    return bool(_DEVANAGARI_ENGLISH_RE.search(text))


def _devanagari_hindi_intent(text: str) -> bool:
    """Devanagari request for Hindi — no calls to looks_like_* (avoids recursion)."""
    if not text_contains_devanagari(text):
        return False
    if _DEVANAGARI_ENGLISH_RE.search(text) and not _DEVANAGARI_HINDI_RE.search(text):
        return False
    if _DEVANAGARI_HINDI_RE.search(text):
        return True
    return bool(_DEVANAGARI_SPEAK_RE.search(text))


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


def is_latin_english_content(
    text: str,
    *,
    min_words: int = 2,
) -> bool:
    """Latin script, no Romanized-Hindi vocabulary — real English (not Whisper lang tag)."""
    if is_explicit_hindi_intent(text):
        return False
    if text_contains_devanagari(text):
        return False
    words = _latin_words(text)
    if len(words) < min_words:
        return False
    if _roman_hindi_score(text) > 0:
        return False
    return True


# Languages Supertonic-3 can speak; anything else uses its language-agnostic "na" mode.
SUPERTONIC_LANGS = frozenset(
    "en ko ja ar bg cs da de el es et fi fr hi hr hu id it lt lv nl pl pt ro ru sk sl sv "
    "tr uk vi".split()
)

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi",
    "ur": "Urdu", "ne": "Nepali", "or": "Odia", "as": "Assamese", "sa": "Sanskrit",
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "ru": "Russian", "uk": "Ukrainian", "pl": "Polish", "cs": "Czech",
    "sk": "Slovak", "sl": "Slovenian", "hr": "Croatian", "bg": "Bulgarian", "ro": "Romanian",
    "hu": "Hungarian", "el": "Greek", "tr": "Turkish", "ar": "Arabic", "fa": "Persian",
    "he": "Hebrew", "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish",
    "et": "Estonian", "lv": "Latvian", "lt": "Lithuanian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "ms": "Malay",
    "tl": "Tagalog", "sw": "Swahili",
}

# Whisper often tags spoken Hindi as Urdu (same spoken language); reply in Hindi.
_WHISPER_LANG_ALIASES = {"ur": "hi"}

# "speak in Spanish", "can you talk French", "switch to German please"
_LANG_REQUEST_VERBS = r"(speak|talk|reply|respond|answer|switch|continue|change)"


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def _explicit_language_request(text: str) -> str | None:
    """Caller asks (in Latin script) for a specific language by name."""
    phrase = _normalized_phrase(text)
    if not phrase or not re.search(rf"\b{_LANG_REQUEST_VERBS}\b", phrase):
        return None
    for code, name in LANGUAGE_NAMES.items():
        if re.search(rf"\b{name.lower()}\b", phrase):
            return code
    return None


def detect_reply_lang(
    text: str,
    *,
    whisper_lang: str | None,
    whisper_prob: float | None,
    current: str,
    min_words: int = 2,
    min_prob: float = 0.6,
) -> str:
    """
    Language the bot should answer this turn in: whatever the caller is speaking.

    Sticky on weak evidence (one-word acks like "ok" / "haan", low Whisper
    confidence) so a single misdetection does not flip the call's language.
    """
    if is_explicit_hindi_intent(text):
        return "hi"
    if is_explicit_english_intent(text):
        return "en"
    requested = _explicit_language_request(text)
    if requested:
        return requested

    wl = (whisper_lang or "").strip().lower()[:2] or None
    wl = _WHISPER_LANG_ALIASES.get(wl, wl) if wl else None

    if text_contains_devanagari(text):
        # Marathi / Nepali also use Devanagari; otherwise Hindi.
        return wl if wl in ("mr", "ne") else "hi"
    if wl in (None, "hi") and looks_like_hindi(text, whisper_lang=wl):
        return "hi"  # Romanized Hindi / Hinglish
    if not wl or wl == current:
        return current

    words = len(re.findall(r"\w+", text or ""))
    if words < min_words:
        return current
    if whisper_prob is not None and whisper_prob < min_prob:
        return current
    if wl == "en" and _is_short_english_utterance(text):
        return current
    return wl


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
    """Supertonic language code for this text (any call language, not only hi/en)."""
    if text_contains_devanagari(text):
        return "hi"
    if session_lang in (None, "", "hi"):
        # Latin-only text (e.g. English fallback line) uses English TTS; else Hindi.
        if _latin_words(text) and _roman_hindi_score(text) == 0:
            return "en"
        return "hi" if looks_like_hindi(text) or session_lang == "hi" else "en"
    if session_lang in SUPERTONIC_LANGS:
        return session_lang
    # Language Supertonic was not trained on (Tamil, Bengali, ...).
    if re.fullmatch(r"[\sA-Za-z0-9'\".,!?;:()\-—]+", text or ""):
        return "en"
    return "na"


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
