"""Speakable number words for phone TTS (correct Hindi compounds / English)."""

from __future__ import annotations

import re

# Correct spoken Hindi 0–99 (चौबीस not बीस चार).
_HI_0_99: tuple[str, ...] = (
    "शून्य",
    "एक",
    "दो",
    "तीन",
    "चार",
    "पांच",
    "छह",
    "सात",
    "आठ",
    "नौ",
    "दस",
    "ग्यारह",
    "बारह",
    "तेरह",
    "चौदह",
    "पंद्रह",
    "सोलह",
    "सत्रह",
    "अठारह",
    "उन्नीस",
    "बीस",
    "इक्कीस",
    "बाईस",
    "तेईस",
    "चौबीस",
    "पच्चीस",
    "छब्बीस",
    "सत्ताईस",
    "अट्ठाईस",
    "उनतीस",
    "तीस",
    "इकतीस",
    "बत्तीस",
    "तैंतीस",
    "चौंतीस",
    "पैंतीस",
    "छत्तीस",
    "सैंतीस",
    "अड़तीस",
    "उनतालीस",
    "चालीस",
    "इकतालीस",
    "बयालीस",
    "तैंतालीस",
    "चवालीस",
    "पैंतालीस",
    "छियालीस",
    "सैंतालीस",
    "अड़तालीस",
    "उनचास",
    "पचास",
    "इक्यावन",
    "बावन",
    "तिरपन",
    "चौवन",
    "पचपन",
    "छप्पन",
    "सत्तावन",
    "अट्ठावन",
    "उनसठ",
    "साठ",
    "इकसठ",
    "बासठ",
    "तिरसठ",
    "चौंसठ",
    "पैंसठ",
    "छियासठ",
    "सड़सठ",
    "अड़सठ",
    "उनहत्तर",
    "सत्तर",
    "इकहत्तर",
    "बहत्तर",
    "तिहत्तर",
    "चौहत्तर",
    "पचहत्तर",
    "छिहत्तर",
    "सतहत्तर",
    "अठहत्तर",
    "उन्यासी",
    "अस्सी",
    "इक्यासी",
    "बयासी",
    "तिरासी",
    "चौरासी",
    "पचासी",
    "छियासी",
    "सत्तासी",
    "अठासी",
    "नवासी",
    "नब्बे",
    "इक्यानवे",
    "बानवे",
    "तिरानवे",
    "चौरानवे",
    "पचानवे",
    "छियानवे",
    "सत्तानवे",
    "अट्ठानवे",
    "निन्यानवे",
)

_ONES_EN = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS_EN = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)

_DEVANAGARI_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_digits_in_text(text: str) -> str:
    """Devanagari digits → ASCII so downstream regex matches."""
    return (text or "").translate(_DEVANAGARI_DIGIT_MAP)


def _two_digits_hi(n: int) -> str:
    if 0 <= n < len(_HI_0_99):
        return _HI_0_99[n]
    return str(n)


def int_to_hindi_words(n: int) -> str:
    if n == 0:
        return _HI_0_99[0]
    if n < 0:
        return f"माइनस {int_to_hindi_words(-n)}"
    if n < 100:
        return _two_digits_hi(n)
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        head = _two_digits_hi(hundreds) + " सौ"
        if rem:
            return f"{head} {_two_digits_hi(rem)}"
        return head
    if n < 100_000:
        thousands, rem = divmod(n, 1000)
        head = (
            f"{int_to_hindi_words(thousands)} हज़ार"
            if thousands > 1
            else "हज़ार"
        )
        if rem:
            return f"{head} {int_to_hindi_words(rem)}"
        return head
    if n < 10_000_000:
        lakhs, rem = divmod(n, 100_000)
        head = (
            f"{int_to_hindi_words(lakhs)} लाख"
            if lakhs > 1
            else "एक लाख"
        )
        if rem:
            return f"{head} {int_to_hindi_words(rem)}"
        return head
    crores, rem = divmod(n, 10_000_000)
    head = (
        f"{int_to_hindi_words(crores)} करोड़"
        if crores > 1
        else "एक करोड़"
    )
    if rem:
        return f"{head} {int_to_hindi_words(rem)}"
    return head


def _three_digits_en(n: int) -> str:
    if n == 0:
        return ""
    if n < 20:
        return _ONES_EN[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return _TENS_EN[tens]
        return f"{_TENS_EN[tens]} {_ONES_EN[ones]}"
    hundreds, rem = divmod(n, 100)
    head = f"{_ONES_EN[hundreds]} hundred"
    if rem:
        return f"{head} {_three_digits_en(rem)}"
    return head


def int_to_english_words(n: int) -> str:
    if n == 0:
        return "zero"
    if n < 0:
        return f"minus {int_to_english_words(-n)}"
    if n < 1000:
        return _three_digits_en(n)
    if n < 1_000_000:
        thousands, rem = divmod(n, 1000)
        head = f"{int_to_english_words(thousands)} thousand"
        if rem:
            return f"{head} {int_to_english_words(rem)}"
        return head
    if n < 1_000_000_000:
        millions, rem = divmod(n, 1_000_000)
        head = f"{int_to_english_words(millions)} million"
        if rem:
            return f"{head} {int_to_english_words(rem)}"
        return head
    billions, rem = divmod(n, 1_000_000_000)
    head = f"{int_to_english_words(billions)} billion"
    if rem:
        return f"{head} {int_to_english_words(rem)}"
    return head


def speak_number(value: str, *, lang: str) -> str:
    """Turn a numeric string (with optional decimals) into spoken words."""
    raw = normalize_digits_in_text(value).replace(",", "").strip()
    if not raw:
        return value
    int_fn = int_to_hindi_words if lang == "hi" else int_to_english_words
    point_word = "दशमलव" if lang == "hi" else "point"

    if "." in raw:
        whole, frac = raw.split(".", 1)
        frac = frac.rstrip("0")
        parts: list[str] = []
        if whole.lstrip("-").isdigit() or whole in ("", "-"):
            w = int(whole) if whole not in ("", "-") else 0
            parts.append(int_fn(w))
        if frac and frac.isdigit():
            if lang == "hi":
                digits = " ".join(_HI_0_99[int(d)] for d in frac if d.isdigit())
            else:
                digits = " ".join(_ONES_EN[int(d)] for d in frac if d.isdigit())
            parts.append(f"{point_word} {digits}")
        return " ".join(parts)

    if raw.lstrip("-").isdigit():
        return int_fn(int(raw))
    return value


_CARAT_RE = re.compile(
    r"(?<!\d)(24|22|18)\s*(?:[kK]|carat|कैरट|करेट|कैरेट)\b",
    re.IGNORECASE,
)
_CARAT_HI_NAMES = {"24": "चौबीस कैरट", "22": "बाईस कैरट", "18": "अठारह कैरट"}
_CARAT_EN_NAMES = {"24": "24 carat", "22": "22 carat", "18": "18 carat"}


def replace_carat_markers(text: str, *, lang: str) -> str:
    names = _CARAT_HI_NAMES if lang == "hi" else _CARAT_EN_NAMES

    def _sub(m: re.Match[str]) -> str:
        return names.get(m.group(1), m.group(0))

    return _CARAT_RE.sub(_sub, text)
