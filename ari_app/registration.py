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


# Asking ABOUT complaints (not lodging one): "how many complaints do I have",
# "status of my complaint", "I already registered a complaint", "track my ticket".
_STATUS_ASK = re.compile(
    r"\b(how many|status|check|track|tracking|update|progress|existing|already|"
    r"earlier|previous|old|my (complaints|tickets)|what happened|any news|"
    r"kitni|kitne|kya hua|purani|pichli)\b|कितनी|कितने|स्टेटस|पुरानी|पिछली|क्या हुआ",
    re.IGNORECASE,
)
_TICKET = re.compile(r"\b(complain\w*|ticket|grievance|shikayat|shikaayat)\b|शिकायत|टिकट|कंप्लेंट", re.IGNORECASE)
_LODGE = re.compile(
    r"\b(lodge|file|register|raise|make|log|new|another|report)\b|\bwant to complain|"
    r"\bi have a complaint|darj|naya|nayi|दर्ज|नई|नया",
    re.IGNORECASE,
)


def complaint_intent(text: str) -> str | None:
    """
    'status' (asking about existing complaints), 'lodge' (wants to make one), or None.
    "Can you tell me how many complaints I have?" -> status, not a new complaint.
    """
    t = text or ""
    if not (_TICKET.search(t) or _COMPLAINT.search(t)):
        return None
    asks_status = bool(_STATUS_ASK.search(t))
    lodges = bool(_LODGE.search(t))
    if asks_status and not lodges:
        return "status"
    if asks_status and lodges:
        # "I already lodged one, what's the status" vs "lodge a new one, not the old"
        return "lodge" if _NEW_WORD.search(t) else "status"
    if not lodges:
        return None  # just mentions complaints ("how does the complaint system work?")
    if _HOW_QUESTION.search(t) and not _WANTS.search(t):
        return None  # "how do I lodge a complaint?" is a question for the LLM, not a request
    return "lodge"


# Questions ABOUT complaints vs requests to make one.
_HOW_QUESTION = re.compile(
    r"\b(how|what|why|when|where|which|explain|tell me about|tell me how|kaise|kya hota)\b",
    re.IGNORECASE,
)
_WANTS = re.compile(
    r"\b(i want|i wanna|i'?d like|i would like|i need|i have a complaint|i want to complain|"
    r"please|help me|can you (lodge|file|register|raise|take|log)|"
    r"karni hai|karna hai|chahta|chahti)\b|करनी है|करना है|चाहता|चाहती",
    re.IGNORECASE,
)


# Backing out of a complaint that has started: "I don't want to lodge a complaint",
# "cancel", "never mind", "rehne do". "I don't want this product" (a description) is
# NOT a cancel: a negated want must be about the complaint itself.
_NEG_WANT = re.compile(
    r"\b(don'?t|do not|dont|no need to|not going to|won'?t)\s+(want|need|wish|wanna)?\s*(to\s+)?"
    r"(lodge|file|register|raise|make|give|lose|log)?",
    re.IGNORECASE,
)
_ABOUT_COMPLAINT = re.compile(
    r"\b(complain\w*|ticket|lodge|register|file)\b|शिकायत|shikayat",
    re.IGNORECASE,
)
_CANCEL_PHRASE = re.compile(
    r"\b(cancel|never ?mind|forget it|leave it|stop it|stop this|not now|no thanks|"
    r"rehne do|rahne do|nahi chahiye|mat karo|band karo|chhodo|chodo)\b|रहने दो|नहीं चाहिए|मत करो|छोड़ो",
    re.IGNORECASE,
)


def wants_cancel(text: str) -> bool:
    t = text or ""
    if _NEG_WANT.search(t) and _ABOUT_COMPLAINT.search(t):
        return True
    return len(t.split()) <= 6 and bool(_CANCEL_PHRASE.search(t))


def wants_complaint(text: str) -> bool:
    """True only when the caller wants to LODGE a complaint."""
    return complaint_intent(text) == "lodge"


_NEW_WORD = re.compile(r"\b(new|another|naya|nayi|one more)\b|नई|नया", re.IGNORECASE)


# Reply to "update on your existing complaint, or a new one?"
_EXISTING = re.compile(
    r"\b(existing|old|older|previous|earlier|last|that one|same|status|update|"
    r"progress|track|purani|purana|pichli|pichla|wali|wahi)\b|पुरानी|पुराना|पिछली|पिछला|स्टेटस|अपडेट",
    re.IGNORECASE,
)
_NEW = re.compile(
    r"\b(new|another|different|fresh|one more|naya|nayi|nai|dusri|doosri|aur ek)\b|नई|नया|दूसरी|दूसरा",
    re.IGNORECASE,
)


def complaint_choice(text: str) -> str | None:
    """'existing', 'new', or None when the reply is about something else / unclear."""
    t = text or ""
    is_new, is_old = bool(_NEW.search(t)), bool(_EXISTING.search(t))
    if is_new and not is_old:
        return "new"
    if is_old and not is_new:
        return "existing"
    if is_new and is_old:
        # "not the old one, a new one" -> whichever is mentioned last wins
        last_new = max(m.end() for m in _NEW.finditer(t))
        last_old = max(m.end() for m in _EXISTING.finditer(t))
        return "new" if last_new > last_old else "existing"
    return None


# extra registration questions (registration_questions.json)

_QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "registration_questions.json"
_SKIP = re.compile(
    r"\b(skip|next|no|nope|none|don'?t have|do not have|nahi|nahin|nahi hai)\b|नहीं|छोड़",
    re.IGNORECASE,
)


@dataclass
class Question:
    key: str
    type: str  # free | text | digits | choice
    ask: dict[str, str]
    required: bool = False
    min_len: int = 1
    max_len: int = 15
    options: list[str] = field(default_factory=list)

    def prompt(self, lang: str) -> str:
        return self.ask.get(lang) or self.ask["en"]


def load_complaint_questions() -> list[Question]:
    """Questions from complaint_questions.json (COMPLAINT_QUESTIONS_FILE)."""
    return load_questions(
        Path(os.getenv("COMPLAINT_QUESTIONS_FILE") or _QUESTIONS_FILE.parent / "complaint_questions.json")
    )


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
            if q.get("type", "text") not in ("free", "text", "digits", "choice"):
                raise ValueError("type must be free, text, digits or choice")
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


# Phone audio often turns "skip" into "keep", "kip", "ship", "let's keep"…
_SKIP_SOUNDALIKE = re.compile(r"\b(skip+(ed|ping)?|s?keep|kip|ship|skeep|skib|scape|escape)\b", re.IGNORECASE)


def is_skip(text: str) -> bool:
    t = text or ""
    if _SKIP.search(t):
        return True
    # sound-alikes only count in a very short reply ("Let's keep." = 2 words)
    return len(t.split()) <= 3 and bool(_SKIP_SOUNDALIKE.search(t))


_YES = re.compile(
    r"\b(yes|yeah|yep|yup|sure|correct|right|ok|okay|fine|confirm|go ahead|please do|"
    r"register it|that's it|haan|han|ha|ji|theek|thik|sahi|bilkul)\b|हाँ|हां|जी|ठीक|सही|बिल्कुल",
    re.IGNORECASE,
)
_NO = re.compile(
    r"\b(no|nope|nah|not|wrong|incorrect|change|cancel|don'?t|nahi|nahin|galat|mat)\b|नहीं|गलत|मत",
    re.IGNORECASE,
)


def yes_no(text: str) -> str | None:
    """'yes', 'no', or None when unclear. "No, that's right" is unclear on purpose."""
    t = text or ""
    y, n = bool(_YES.search(t)), bool(_NO.search(t))
    if y and not n:
        return "yes"
    if n and not y:
        return "no"
    return None


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
        "complaint_start": "Okay, let me take down your complaint.",
        "complaint_cancelled": "Okay, I won't register a complaint. What else can I help you with?",
        "confirm_intro": "Let me read that back.",
        "confirm_field": "{label}: {value}.",
        "confirm_said": "You said: {value}.",
        "confirm_ask": "Shall I register this complaint? Please say yes or no.",
        "confirm_again": "Sorry, please say yes to register it, or no to change it.",
        "confirm_fix": "Okay, let's correct it.",
        "confirm_cancel": "Okay, I haven't registered it. Is there anything else I can help you with?",
        "required_answer": "I need this to register the complaint. {question}",
        "no_open_complaints": "You don't have any open complaints right now. Would you like to lodge one?",
        "status_unknown_caller": "I couldn't find any complaints for this number. Would you like to lodge one?",
        "offer_one": "Welcome back, {name}! I see you have an open complaint, ticket {ticket}, about {topic}. Would you like an update on it, or do you want to lodge a new complaint?",
        "offer_many": "Welcome back, {name}! I see you have {count} open complaints, the latest is ticket {ticket}, about {topic}. Would you like an update on them, or do you want to lodge a new complaint?",
        "status_item": "Ticket {ticket}, about {topic}, raised on {date}, is {status}.",
        "status_order": "The order number on it is {order}.",
        "status_end": "Our team is working on it. Is there anything else I can help you with?",
        "status_end_done": "Is there anything else I can help you with?",
        "end_ask": "Shall I end the call now? Please say yes, or tell me what else you need.",
        "verify_start": "Welcome back, {name}! For your security, please say or type your {n} digit PIN.",
        "verify_start_failed": "Sorry, I couldn't verify your PIN, so I can't share your account details on this call. I can still help with general questions. What can I do for you?",
        "welcome_verified": "Thank you, you're verified. How can I help you today?",
        "offer_one_v": "Thank you, you're verified. I see you have an open complaint, ticket {ticket}, about {topic}. Would you like an update on it, or do you want to lodge a new complaint?",
        "offer_many_v": "Thank you, you're verified. I see you have {count} open complaints, the latest is ticket {ticket}, about {topic}. Would you like an update on them, or do you want to lodge a new complaint?",
        "verify_needed": "For your security, I can't share or take complaints on this call because the PIN wasn't verified. Please call again and enter your PIN.",
        "end_stay": "Sure, what else can I help you with?",
        "goodbye": "Thank you for calling{name}. Have a great day, goodbye!",
        "silence_goodbye": "I can't hear you, so I'll end the call now. Please call again any time. Goodbye!",
        "status_resolution": "Here is the update from our team: {text}",
        "status_open": "still open",
        "status_in_progress": "being worked on",
        "status_resolved": "resolved",
        "complaint_saved": "Your complaint is registered. Your ticket number is {ticket}. Is there anything else I can help you with?",
        "complaint_failed": "Sorry, I couldn't save the complaint right now. Please call again a little later.",
        "complaint_no_pin": "Without verifying your PIN I can't take a complaint. Is there anything else I can help with?",
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
        "complaint_start": "ठीक है, मैं आपकी शिकायत दर्ज करती हूँ।",
        "complaint_cancelled": "ठीक है, मैं शिकायत दर्ज नहीं करूँगी। बताइए, और क्या मदद कर सकती हूँ?",
        "confirm_intro": "मैं एक बार दोहरा देती हूँ।",
        "confirm_field": "{label}: {value}।",
        "confirm_said": "आपने कहा: {value}।",
        "confirm_ask": "क्या मैं यह शिकायत दर्ज कर दूँ? कृपया हाँ या नहीं बोलिए।",
        "confirm_again": "माफ़ कीजिए, दर्ज करने के लिए हाँ बोलिए, या बदलने के लिए नहीं।",
        "confirm_fix": "ठीक है, इसे सही कर लेते हैं।",
        "confirm_cancel": "ठीक है, मैंने इसे दर्ज नहीं किया। क्या मैं और कुछ मदद कर सकती हूँ?",
        "required_answer": "शिकायत दर्ज करने के लिए यह ज़रूरी है। {question}",
        "no_open_complaints": "अभी आपकी कोई खुली शिकायत नहीं है। क्या आप नई शिकायत दर्ज करना चाहेंगे?",
        "status_unknown_caller": "इस नंबर पर कोई शिकायत नहीं मिली। क्या आप शिकायत दर्ज करना चाहेंगे?",
        "offer_one": "फिर से स्वागत है, {name}! आपकी एक शिकायत खुली है, टिकट {ticket}, {topic} के बारे में। क्या आप उसका अपडेट जानना चाहेंगे, या नई शिकायत दर्ज करनी है?",
        "offer_many": "फिर से स्वागत है, {name}! आपकी {count} शिकायतें खुली हैं, सबसे नई टिकट {ticket} है, {topic} के बारे में। क्या आप उनका अपडेट जानना चाहेंगे, या नई शिकायत दर्ज करनी है?",
        "status_item": "टिकट {ticket}, {topic} के बारे में, {date} को दर्ज हुई थी, और अभी {status} है।",
        "status_order": "इस पर ऑर्डर नंबर {order} है।",
        "status_end": "हमारी टीम इस पर काम कर रही है। क्या मैं और कुछ मदद कर सकती हूँ?",
        "status_end_done": "क्या मैं और कुछ मदद कर सकती हूँ?",
        "end_ask": "क्या मैं अब कॉल समाप्त कर दूँ? हाँ बोलिए, या बताइए और क्या चाहिए।",
        "verify_start": "फिर से स्वागत है, {name}! आपकी सुरक्षा के लिए, कृपया अपना {n} अंकों का पिन बोलिए या टाइप कीजिए।",
        "verify_start_failed": "माफ़ कीजिए, पिन की पुष्टि नहीं हो पाई, इसलिए इस कॉल पर मैं आपके खाते की जानकारी नहीं बता सकती। सामान्य सवालों में मदद कर सकती हूँ। बताइए?",
        "welcome_verified": "धन्यवाद, आपकी पहचान हो गई। बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
        "offer_one_v": "धन्यवाद, आपकी पहचान हो गई। आपकी एक शिकायत खुली है, टिकट {ticket}, {topic} के बारे में। क्या आप उसका अपडेट जानना चाहेंगे, या नई शिकायत दर्ज करनी है?",
        "offer_many_v": "धन्यवाद, आपकी पहचान हो गई। आपकी {count} शिकायतें खुली हैं, सबसे नई टिकट {ticket} है, {topic} के बारे में। क्या आप उनका अपडेट जानना चाहेंगे, या नई शिकायत दर्ज करनी है?",
        "verify_needed": "आपकी सुरक्षा के लिए, पिन की पुष्टि न होने के कारण इस कॉल पर मैं शिकायत की जानकारी नहीं दे सकती और न ही नई शिकायत ले सकती। कृपया दोबारा कॉल करके पिन डालें।",
        "end_stay": "ज़रूर, बताइए और क्या मदद कर सकती हूँ?",
        "goodbye": "कॉल करने के लिए धन्यवाद{name}। आपका दिन शुभ हो, नमस्ते!",
        "silence_goodbye": "मुझे आपकी आवाज़ नहीं आ रही, इसलिए मैं कॉल समाप्त कर रही हूँ। कभी भी दोबारा कॉल करें। नमस्ते!",
        "status_resolution": "हमारी टीम का अपडेट यह है: {text}",
        "status_open": "खुली",
        "status_in_progress": "प्रगति पर",
        "status_resolved": "हल हो चुकी",
        "complaint_saved": "आपकी शिकायत दर्ज हो गई है। आपका टिकट नंबर है {ticket}। क्या मैं और कुछ मदद कर सकती हूँ?",
        "complaint_failed": "माफ़ कीजिए, अभी शिकायत सेव नहीं हो पाई। कृपया थोड़ी देर बाद फिर कॉल करें।",
        "complaint_no_pin": "पिन की पुष्टि के बिना मैं शिकायत नहीं ले सकती। क्या मैं और कुछ मदद कर सकती हूँ?",
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


def speak_ticket(ticket: str) -> str:
    """'100012' -> '1 0 0 0 1 2' so TTS reads it digit by digit."""
    return " ".join(ticket)


_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def _norm_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def strip_echo(text: str, prompt: str, answers: list[str] | None = None) -> str:
    """
    Remove the bot's own question echoed back at the start of the caller's answer.

    "sentences."                 after "...in a few sentences."  -> ""        (pure echo)
    "otherwise say skip. Skip."  after "...otherwise say skip."  -> "Skip."
    "Skip."                      after "...otherwise say skip."  -> "Skip."   (real answer)
    """
    words = text.split()
    if not words or not prompt:
        return text.strip()
    p = _norm_words(prompt)
    t = [(_norm_words(w) or [""])[0] for w in words]
    # An expected answer ("customer service", "delivery") is never echo.
    flat = " ".join(t)
    for a in answers or []:
        if " ".join(_norm_words(a)) and " ".join(_norm_words(a)) in flat:
            return text.strip()
    # Longest prefix of the answer that is echo: the END of the prompt
    # ("…sentences", "…otherwise say skip"), or 3+ consecutive prompt words.
    cut = 0
    for n in range(len(t), 0, -1):
        piece = t[:n]
        if p[-n:] == piece or (
            n >= 3 and any(p[i : i + n] == piece for i in range(len(p) - n + 1))
        ):
            cut = n
            break
    if cut == 0:
        return text.strip()
    rest = " ".join(words[cut:]).strip()
    if rest:
        return rest
    # The whole reply also occurs in the prompt. A real one-word answer (skip, yes,
    # no, a number, a category) is kept; anything else is treated as echo.
    if len(t) <= 2 and (is_skip(text) or yes_no(text) or parse_digits(text)):
        return text.strip()
    return ""


# ---- ending the call ------------------------------------------------------------

# Caller signals they are done: "that's all", "bye", "no thanks", "bas", "dhanyavaad"…
_CLOSING = re.compile(
    r"\b(bye|bye bye|goodbye|good bye|that'?s all|that is all|that'?s it|nothing else|"
    r"nothing more|no thanks|no thank you|i'?m done|im done|i am done|all good|"
    r"no more questions|no other questions|no questions|hang up|end the call|cut the call|"
    r"disconnect|alvida|bas itna|bas|shukriya|dhanyavad|dhanyawad|dhanyavaad)\b|"
    r"बस|धन्यवाद|शुक्रिया|अलविदा|फ़ोन रख|फोन रख|कुछ नहीं",
    re.IGNORECASE,
)
_THANKS = re.compile(r"\b(thank you|thanks|thankyou)\b", re.IGNORECASE)
_PLAIN_NO = re.compile(r"^\W*(no|nope|nah|nahi|nahin|no no|नहीं|ना)\W*$", re.IGNORECASE)
_ASKED_ANYTHING_ELSE = re.compile(r"anything else|what else|और कुछ|कुछ और", re.IGNORECASE)
_END_YES = re.compile(
    r"\b(end|hang up|disconnect|cut|bye|goodbye|that'?s all|done|kar do|rakh do)\b|"
    r"रख दो|काट दो|अलविदा",
    re.IGNORECASE,
)


def wants_to_end(text: str, last_bot_line: str = "") -> bool:
    """
    The caller is wrapping up. "No" only counts right after the bot asked
    "anything else?"; "thank you" only when that's (almost) all they said.
    """
    t = (text or "").strip()
    if not t:
        return False
    if _CLOSING.search(t):
        return True
    if _THANKS.search(t) and len(t.split()) <= 4:
        return True
    return bool(_PLAIN_NO.match(t)) and bool(_ASKED_ANYTHING_ELSE.search(last_bot_line or ""))


def end_reply(text: str) -> str:
    """Answer to "Shall I end the call now?": 'end', 'stay', or 'other' (a new request)."""
    t = (text or "").strip()
    if not t:
        return "stay"
    yn = yes_no(t)
    if _END_YES.search(t) or _CLOSING.search(t) or yn == "yes":
        return "end"
    if yn == "no" and len(t.split()) <= 4:
        return "stay"
    return "other"


# Questions about the caller's OWN data never need a web search (the internet can't
# know "who changed my name" or "where is my parcel"); the LLM answers from caller info.
_ABOUT_SELF = re.compile(
    r"\b(my|mine|me)\b[^.?!]{0,25}\b(name|account|pin|password|complain\w*|ticket|order|"
    r"number|phone|registration|profile|details|data|refund|payment|parcel|delivery|"
    r"subscription|plan|bill|balance|status|call)\b|"
    r"\b(mera|meri|mere|mujhe)\b|मेरा|मेरी|मेरे|मुझे",
    re.IGNORECASE,
)


def about_caller(text: str) -> bool:
    return bool(_ABOUT_SELF.search(text or ""))


# Questions about THIS business ("what courses do you offer", "your prices", "your
# opening hours") are answered from the knowledge base / LLM: the web can't know them,
# and searching only costs ~3 s and returns unrelated pages.
_ABOUT_BUSINESS = re.compile(
    r"\b(do|does|can|could|will)\s+(you|u|y'?all)\s+(guys\s+)?"
    r"(offer|provide|sell|have|teach|do|support|deliver|ship|accept|run)\b|"
    r"\byour\s+(\w+\s+)?(services?|courses?|products?|plans?|prices?|pricing|fees?|company|"
    r"office|hours|timings?|policy|policies|team|address|location|branches?|offers?|support)\b|"
    r"\b(aapke|aapki|aapka)\b|आपके|आपकी|आपका",       # "your" (not plain "aap" = polite "you")
    re.IGNORECASE,
)


def about_business(text: str) -> bool:
    return bool(_ABOUT_BUSINESS.search(text or ""))


# "What's today's date / what time is it / which day": the prompt carries the exact local
# date and time; a web search returns pages from other time zones (it answered "the 11th"
# on the 12th). Never search these.
_DATE_TIME = re.compile(
    r"\b(today'?s (?:date|day)|(?:date|day|time) (?:is it )?(?:today|now|right now)|"
    r"what(?:'s| is)? (?:the )?(?:date|day|time)(?: is it)? ?(?:today|now|right now)?\s*\??$|"
    r"what (?:time|day) is it|current (?:date|time|day)|which day is (?:it|today)|"
    r"aaj (?:kya|kaun si|konsi|kaunsa) (?:tarikh|tareekh|date|din)|kitne baje|abhi kya time)\b|"
    r"आज (?:क्या|कौन सी|कौनसी) (?:तारीख|तारिख)|कितने बजे|अभी क्या समय|आज कौन सा दिन|आज क्या दिन",
    re.IGNORECASE,
)


def asks_date_time(text: str) -> bool:
    return bool(_DATE_TIME.search(text or ""))


# A bot reply that ends by offering to take a complaint / request ("Would you like to
# lodge one?", "Shall I register it as a complaint?"); a "yes" then starts the flow.
_OFFER = re.compile(
    r"(would you like|do you want|shall i|should i|can i|may i|want me to)[^.?!]{0,50}"
    r"(lodge|register|file|raise|log|record|create|take)[^.?!]{0,30}(complaint|request|ticket)|"
    r"(would you like|do you want|shall i|should i|want me to)[^.?!]{0,30}(lodge|file|register|raise) (one|it|this|that)\b|"
    r"(शिकायत|अनुरोध)[^।?]{0,40}(दर्ज)[^।?]{0,30}\?",
    re.IGNORECASE,
)


def offers_complaint(reply: str) -> bool:
    return bool(_OFFER.search(reply or ""))
