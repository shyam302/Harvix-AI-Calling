"""
Caller registration over the phone. Registration is NOT asked at the start of a
call: callers are only looked up (registered -> addressed by name). When an
unregistered caller wants to lodge a complaint, the bot asks name + PIN first.

Helpers here: complaint intent, understanding a spoken name and PIN, and the lines
the bot says while registering or verifying.

The PIN is never spoken back, logged, kept in chat history or sent to the LLM.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# PIN

_DIGIT_WORDS = {
    # English
    "zero": "0", "oh": "0", "o": "0", "nil": "0",
    "one": "1", "won": "1",
    "two": "2", "to": "2", "too": "2",
    "three": "3", "tree": "3",
    "four": "4", "for": "4", "fore": "4",
    "five": "5",
    "six": "6", "sex": "6",
    "seven": "7",
    "eight": "8", "ate": "8",
    "nine": "9", "nein": "9",
    # Hindi (romanised)
    "shunya": "0", "sunya": "0",
    "ek": "1", "do": "2", "teen": "3", "tin": "3", "char": "4", "chaar": "4",
    "paanch": "5", "panch": "5", "chhe": "6", "che": "6", "chah": "6",
    "saat": "7", "sat": "7", "aath": "8", "aat": "8", "nau": "9",
    # Hindi (Devanagari)
    "शून्य": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4",
    "पांच": "5", "पाँच": "5", "छह": "6", "छः": "6", "छे": "6",
    "सात": "7", "आठ": "8", "नौ": "9",
}
_REPEAT = {"double": 2, "triple": 3}


def parse_pin(text: str, length: int = 4) -> str | None:
    """Digits from a spoken PIN ("1 2 3 4", "one two three four", "double five six seven",
    "ek do teen char"). Returns the PIN only if exactly ``length`` digits were heard."""
    if not text:
        return None
    # Devanagari digits -> ASCII
    text = text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    # split on spaces/punctuation (not \w: Devanagari vowel signs are not word chars)
    tokens = re.findall(r"\d+|[^\s\d.,!?;:।\-]+", text.lower())
    digits: list[str] = []
    repeat = 1
    for tok in tokens:
        if tok in _REPEAT:
            repeat = _REPEAT[tok]
            continue
        if tok.isdigit():
            digits.extend(tok[0] * repeat + tok[1:])
        elif tok in _DIGIT_WORDS:
            digits.extend(_DIGIT_WORDS[tok] * repeat)
        repeat = 1
    pin = "".join(digits)
    return pin if len(pin) == length else None


# name

_NAME_PATTERNS = [
    r"\bmy name is\s+(.+)",
    r"\bmy name's\s+(.+)",
    r"\bname is\s+(.+)",
    r"\bthis is\s+(.+)",
    r"\bi am\s+(.+)",
    r"\bi'm\s+(.+)",
    r"\bit's\s+(.+)",
    r"\bcall me\s+(.+)",
    r"मेरा नाम\s+(.+?)(?:\s+है)?$",
    r"\bmera naam\s+(.+?)(?:\s+hai)?$",
]
_NOT_NAMES = {"yes", "no", "okay", "ok", "hello", "hi", "sorry", "what", "hmm", "um", "uh"}


def name_heuristic(text: str) -> str | None:
    t = re.sub(r"[.!?,]+", " ", (text or "")).strip()
    if not t:
        return None
    for pat in _NAME_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            t = m.group(1).strip()
            break
    words = [w for w in t.split() if w.lower() not in ("here", "speaking", "hai")]
    if not words or len(words) > 3:
        return None
    if all(w.lower() in _NOT_NAMES for w in words):
        return None
    return " ".join(w[:1].upper() + w[1:] for w in words)


async def extract_name(
    text: str, *, base_url: str, api_key: str, model: str, timeout_s: float = 4.0
) -> str | None:
    """The caller's name from what they said; LLM first, simple patterns as fallback."""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout_s)
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=40,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "A phone caller was asked for their name. From the speech-recognition "
                        'text, return JSON {"name": "<the person\'s name>"} written in Latin '
                        'letters with normal capitalisation, or {"name": null} if they did '
                        "not say a name (refused, asked a question, noise)."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        name = (data.get("name") or "").strip()
        if name and len(name) <= 40 and name.lower() not in _NOT_NAMES:
            return name
        if data.get("name") is None:
            return None
    except Exception as exc:
        log.info("name extraction via LLM failed (%s); using patterns", exc)
    return name_heuristic(text)


# complaint intent (registration is only needed to lodge a complaint)

_COMPLAINT = re.compile(
    r"\b(complain\w*|grievance|file an? (issue|report)|raise an? (issue|ticket)|"
    r"report an? (problem|issue)|lodge|shikayat|shikaayat)\b|शिकायत|कंप्लेंट|कम्प्लेन",
    re.IGNORECASE,
)


def wants_complaint(text: str) -> bool:
    return bool(_COMPLAINT.search(text or ""))


# extra registration questions (registration_questions.json)

_QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "registration_questions.json"
_SKIP = re.compile(
    r"\b(skip|next|no|nope|none|don'?t have|do not have|nahi|nahin|nahi hai)\b|नहीं|छोड़",
    re.IGNORECASE,
)


@dataclass
class Question:
    key: str
    type: str  # text | digits | choice
    ask: dict[str, str]
    required: bool = False
    min_len: int = 1
    max_len: int = 15
    options: list[str] = field(default_factory=list)

    def prompt(self, lang: str) -> str:
        return self.ask.get(lang) or self.ask["en"]


def load_questions(path: Path | None = None) -> list[Question]:
    """Questions from registration_questions.json; [] if the file is missing or invalid."""
    p = path or Path(os.getenv("REGISTRATION_QUESTIONS_FILE") or _QUESTIONS_FILE)
    if not p.is_absolute():
        p = _QUESTIONS_FILE.parent / p
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        log.error("registration questions: cannot read %s (%s); asking none", p, exc)
        return []
    out = []
    for q in raw.get("questions", []):
        try:
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", q["key"]) or "en" not in q["ask"]:
                raise ValueError("key must be lowercase_snake and ask.en is required")
            if q.get("type", "text") not in ("text", "digits", "choice"):
                raise ValueError("type must be text, digits or choice")
            out.append(
                Question(
                    key=q["key"],
                    type=q.get("type", "text"),
                    ask=q["ask"],
                    required=bool(q.get("required", False)),
                    min_len=int(q.get("min_len", 1)),
                    max_len=int(q.get("max_len", 15)),
                    options=list(q.get("options", [])),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.error("registration questions: skipping %r (%s)", q, exc)
    return out


def is_skip(text: str) -> bool:
    return bool(_SKIP.search(text or ""))


def parse_digits(text: str) -> str:
    """All digits in a spoken number ("nine eight one 2 3", "double five")."""
    text = (text or "").translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    tokens = re.findall(r"\d+|[^\s\d.,!?;:।\-]+", text.lower())
    digits: list[str] = []
    repeat = 1
    for tok in tokens:
        if tok in _REPEAT:
            repeat = _REPEAT[tok]
            continue
        if tok.isdigit():
            digits.append(tok[0] * repeat + tok[1:])
        elif tok in _DIGIT_WORDS:
            digits.append(_DIGIT_WORDS[tok] * repeat)
        repeat = 1
    return "".join(digits)


async def extract_answer(
    question: Question,
    text: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout_s: float = 5.0,
) -> str | None:
    """The answer to a text/choice question from what the caller said (None = no answer)."""
    if question.type == "choice":
        rule = f'"value" must be exactly one of {json.dumps(question.options)}, or null'
    else:
        rule = '"value" is the answer in a few words, in Latin letters, or null'
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout_s)
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=60,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "A phone caller was asked a registration question. From the speech-"
                        f'recognition text of their reply, return JSON {{"value": ...}}. {rule}. '
                        "Use null if they did not answer, refused, or asked something else."
                    ),
                },
                {"role": "user", "content": f"Question: {question.ask['en']}\nReply: {text}"},
            ],
        )
        value = json.loads(resp.choices[0].message.content or "{}").get("value")
    except Exception as exc:
        log.info("answer extraction failed (%s); keeping the raw reply", exc)
        return text.strip()[:100] or None
    if value is None:
        return None
    value = str(value).strip()[:200]
    if question.type == "choice" and value not in question.options:
        return None
    if value.lower() in ("skip", "none", "no", "nothing", "n/a", "null"):
        return None
    return value or None


# what the bot says

_LINES = {
    "en": {
        "ask_name": "I can help you lodge a complaint. First I need to register you, it only takes a moment. What's your name?",
        "name_again": "Sorry, I didn't catch your name. Could you say it again?",
        "ask_pin": "Thanks, {name}. Now choose a {n} digit secret PIN. Say it one digit at a time, or type it on your keypad.",
        "pin_again": "I need exactly {n} digits. Please say your PIN again, or type it on the keypad.",
        "confirm_pin": "Please say or type the same PIN once more to confirm.",
        "pin_mismatch": "Those didn't match. Let's try again. Say a {n} digit PIN.",
        "registered": "Thank you, {name}, you're registered. Please remember your PIN.",
        "pin_saved": "Thank you, your PIN is saved. Just {k} quick questions more.",
        "answer_again": "Sorry, I didn't catch that. Could you say it again?",
        "digits_again": "I need between {lo} and {hi} digits. Please say it again, or type it and press hash.",
        "reg_skipped": "No problem, we can finish the registration later.",
        "welcome_back": "Welcome back, {name}! How can I help you today?",
        "verify_ask": "Sure, {name}. Before I take your complaint, please say or type your {n} digit PIN.",
        "verify_wrong": "That PIN didn't match. Please try again.",
        "verified": "Thank you, {name}, you're verified.",
        "verify_failed": "Sorry, I couldn't verify your PIN, so I can't take the complaint right now.",
    },
    "hi": {
        "ask_name": "मैं आपकी शिकायत दर्ज करने में मदद करूँगी। उससे पहले आपका रजिस्ट्रेशन कर लेते हैं, बस एक मिनट। आपका नाम क्या है?",
        "name_again": "माफ़ कीजिए, नाम ठीक से सुनाई नहीं दिया। क्या आप दोबारा बता सकते हैं?",
        "ask_pin": "धन्यवाद, {name}। अब {n} अंकों का एक गुप्त पिन चुनिए। एक-एक अंक बोलिए, या कीपैड पर टाइप कीजिए।",
        "pin_again": "मुझे पूरे {n} अंक चाहिए। कृपया पिन दोबारा बोलिए, या कीपैड पर टाइप कीजिए।",
        "confirm_pin": "पुष्टि के लिए वही पिन एक बार फिर बोलिए या टाइप कीजिए।",
        "pin_mismatch": "दोनों पिन मेल नहीं खाए। फिर से कोशिश करते हैं। {n} अंकों का पिन बोलिए।",
        "registered": "धन्यवाद, {name}, आपका रजिस्ट्रेशन हो गया। अपना पिन याद रखिए।",
        "pin_saved": "धन्यवाद, आपका पिन सेव हो गया। बस {k} छोटे सवाल और।",
        "answer_again": "माफ़ कीजिए, समझ नहीं पाई। क्या आप दोबारा बता सकते हैं?",
        "digits_again": "मुझे {lo} से {hi} अंक चाहिए। कृपया दोबारा बोलिए, या टाइप करके हैश दबाइए।",
        "reg_skipped": "कोई बात नहीं, रजिस्ट्रेशन बाद में कर लेंगे।",
        "welcome_back": "फिर से स्वागत है, {name}! बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
        "verify_ask": "ज़रूर, {name}। शिकायत लेने से पहले अपना {n} अंकों का पिन बोलिए या टाइप कीजिए।",
        "verify_wrong": "पिन मेल नहीं खाया। कृपया फिर से कोशिश कीजिए।",
        "verified": "धन्यवाद, {name}, आपकी पहचान हो गई।",
        "verify_failed": "माफ़ कीजिए, पिन की पुष्टि नहीं हो पाई, इसलिए अभी शिकायत दर्ज नहीं कर सकती।",
    },
}


def line(key: str, lang: str, **kw: object) -> str:
    table = _LINES.get(lang) or _LINES["en"]
    return table[key].format(**kw)
