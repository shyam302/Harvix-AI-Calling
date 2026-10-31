"""Web search for real-time caller questions (Google CSE or DuckDuckGo fallback)."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

_GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"

# Free search engines rate-limit bursts. After repeated failures, pause searching
# briefly instead of adding seconds of dead air to every caller turn.
_FAILURE_WINDOW_S = 120.0
_FAILURES_TO_PAUSE = 2
_PAUSE_S = 60.0
_recent_failures: list[float] = []
_paused_until = 0.0


def search_available() -> bool:
    return time.monotonic() >= _paused_until


def record_search_result(ok: bool) -> None:
    global _paused_until
    now = time.monotonic()
    if ok:
        _recent_failures.clear()
        return
    _recent_failures[:] = [t for t in _recent_failures if now - t < _FAILURE_WINDOW_S]
    _recent_failures.append(now)
    if len(_recent_failures) >= _FAILURES_TO_PAUSE:
        _paused_until = now + _PAUSE_S
        _recent_failures.clear()
        log.warning("Web search failing repeatedly; pausing searches for %.0fs", _PAUSE_S)

# Topics that need fresh facts. Time words alone ("today", "आज", "who is",
# "how much") are deliberately not triggers: "How are you today?" or
# "आज मैं खुश हूँ" must not cost the caller a multi-second web search.
_REALTIME_EN = re.compile(
    r"\b("
    r"news|headlines?|weather|temperature|forecast|rain|raining|"
    r"prices?|rates?|stocks?|share price|exchange rate|sensex|nifty|"
    r"score|scores|match result|who won|live score|live updates?|live news|"
    r"real[- ]?time|latest|"
    r"google|search online|on the web|internet|look up|check online|"
    r"gold|silver|petrol|diesel|crypto|bitcoin"
    r")\b",
    re.IGNORECASE,
)
_REALTIME_HI = re.compile(
    r"(ताज़ा|ताजा|खबर|समाचार|न्यूज़|न्यूज|मौसम|बारिश|तापमान|कीमत|दाम|रेट|"
    r"स्कोर|कौन जीता|गूगल|इंटरनेट|"
    r"गोल्ड|सोने\s*(?:का|की|के)|चाँदी|चांदी|पेट्रोल|डीजल|लाइव)",
    re.IGNORECASE,
)
_EXPLICIT_SEARCH = re.compile(
    r"\b(search|google|look up|find online|web (?:search|result)|live search)\b",
    re.IGNORECASE,
)
_EXPLICIT_SEARCH_HI = re.compile(
    r"(सर्च\s*कर|खोज\s*कर|ऑनलाइन\s*बत|वेब\s*पर|गूगल\s*कर|"
    r"लाइव\s*सर्च|लाइफ\s*सर्च|लाइब\s*साज|साज\s*कर)",
    re.IGNORECASE,
)
_GOLD = re.compile(r"(gold|गोल्ड|सोन)", re.IGNORECASE)
_SILVER = re.compile(r"\b(silver|चाँदी|चांदी)\b", re.IGNORECASE)
_WEATHER = re.compile(r"\b(weather|मौसम|बारिश|temperature)\b", re.IGNORECASE)
_PETROL = re.compile(r"\b(petrol|diesel|पेट्रोल|डीजल)\b", re.IGNORECASE)
_STOCK = re.compile(r"\b(stock|share|nifty|sensex|शेयर)\b", re.IGNORECASE)
_CRICKET = re.compile(
    r"\b("
    r"cricket|ipl|wpl|match|score|cricbuzz|espncricinfo|bcci|icc|odi|t20|test|"
    r"qualifier|eliminator|playoff|final|semi-?final|"
    # Team names (English) so callers can name a team without saying "cricket".
    r"gujarat titans?|mumbai indians?|chennai super kings?|royal challengers?|"
    r"rajasthan royals?|sunrisers hyderabad|punjab kings?|delhi capitals?|"
    r"kolkata knight riders?|lucknow super giants?|"
    r"rcb|csk|mi|kkr|gt|srh|rr|dc|pbks|lsg"
    r")\b|"
    r"(क्रिकेट|आईपीएल|आइपिल|आईपिल|मैच|मेच|स्कोर|"
    r"क्वालीफायर|एलिमिनेटर|फाइनल|सेमी\s*फाइनल|"
    # Team names in Hindi spelling Whisper produces.
    r"गुजरा[तर]?\s*टाइट[नर]्?स?|मुंबई\s*इंडियन्स|चेन्नई\s*सुपर\s*किंग्स|"
    r"रॉयल\s*चैलेंजर्स|राजस्थान\s*रॉयल्स|सनराइजर्स|पंजाब\s*किंग्स|"
    r"दिल्ली\s*कैपिटल्स|कोलकाता\s*नाइट|लखनऊ\s*सुपर)",
    re.IGNORECASE,
)
_CRICKET_LIVE_INTENT = re.compile(
    r"\b("
    r"live score|score|next match|upcoming match|schedule|fixtures?|table|points table|"
    r"ipl|wpl|cricket|cricbuzz|who won|match result|"
    r"qualifier|eliminator|playoff|final|semi-?final|"
    r"gujarat titans?|mumbai indians?|chennai super kings?|royal challengers?|"
    r"rajasthan royals?|sunrisers hyderabad|punjab kings?|delhi capitals?|"
    r"kolkata knight riders?|lucknow super giants?|"
    r"rcb|csk|mi|kkr|gt|srh|rr|dc|pbks|lsg"
    r")\b|"
    r"(लाइव|स्कोर|अगला\s*मैच|नेक्स्ट\s*मैच|अगला\s*मेच|नेक्स्ट\s*मेच|शेड्यूल|फिक्स्चर|"
    r"पॉइंट्स?\s*टेबल|आईपीएल|आइपिल|आईपिल|क्रिकेट|मैच|मेच|कौन\s*जीता|"
    r"क्वालीफायर|एलिमिनेटर|फाइनल|सेमी\s*फाइनल|"
    r"गुजरा[तर]?\s*टाइट[नर]्?स?|मुंबई\s*इंडियन्स|चेन्नई|रॉयल\s*चैलेंजर्स|"
    r"राजस्थान\s*रॉयल्स|सनराइजर्स|पंजाब\s*किंग्स|दिल्ली\s*कैपिटल्स|"
    r"कोलकाता\s*नाइट|लखनऊ\s*सुपर)",
    re.IGNORECASE,
)

# Domains unsuitable for live rates / current values.
_BLOCKED_DOMAINS = (
    "wikipedia.org",
    "wikimedia.org",
    "fandom.com",
    "quora.com",
    "reddit.com",
    "pinterest.",
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "amazon.",
    "flipkart.",
)

_TOPIC_GOOD_DOMAINS: dict[str, tuple[str, ...]] = {
    "gold": (
        "goodreturns.in",
        "livepriceofgold.com",
        "moneycontrol.com",
        "mcxlive.com",
        "goldprice.org",
        "angelone.in",
        "economictimes",
        "businesstoday",
        "livemint.com",
        "indiatoday",
        "bankbazaar",
        "policybazaar",
    ),
    "silver": (
        "goodreturns.in",
        "livepriceofgold.com",
        "moneycontrol.com",
        "mcxlive.com",
    ),
    "weather": (
        "weather.com",
        "accuweather.com",
        "imd.gov.in",
        "skymetweather.com",
    ),
    "petrol": (
        "goodreturns.in",
        "moneycontrol.com",
        "ppac.gov.in",
        "indianoil.com",
    ),
    "stock": (
        "moneycontrol.com",
        "economictimes",
        "livemint.com",
        "bseindia.com",
        "nseindia.com",
    ),
    "news": (
        "reuters.com",
        "bbc.com",
        "ndtv.com",
        "indiatoday",
        "thehindu.com",
        "livemint.com",
    ),
    "cricket": (
        "cricbuzz.com",
        "espncricinfo.com",
        "icc-cricket.com",
        "bcci.tv",
    ),
}

_GOLD_QUERIES = (
    "24 carat gold price per 10 gram india today INR",
    "22 carat gold rate today india 10 gram site:goodreturns.in",
    "gold rate today india 10 gram site:livepriceofgold.com",
)
_SILVER_QUERIES = (
    "silver price india today per kg INR site:goodreturns.in",
    "silver rate today india site:moneycontrol.com",
)
_WEATHER_QUERIES = (
    "weather forecast today india",
    "today weather site:weather.com",
)
_PETROL_QUERIES = (
    "petrol diesel price today india city site:goodreturns.in",
    "fuel price today india site:moneycontrol.com",
)
_CRICKET_QUERIES = (
    "cricbuzz live cricket score today",
    "cricbuzz latest cricket news india",
    "live cricket match score IPL today cricbuzz",
)


@dataclass(frozen=True)
class SearchHit:
    title: str
    snippet: str
    link: str


def is_google_configured(*, api_key: str, cx: str) -> bool:
    return bool((api_key or "").strip() and (cx or "").strip())


def is_serpapi_configured(*, api_key: str) -> bool:
    return bool((api_key or "").strip())


def needs_realtime_search(user_text: str) -> bool:
    """True when the caller likely needs up-to-date web facts."""
    t = (user_text or "").strip()
    if len(t) < 4:
        return False
    if _EXPLICIT_SEARCH.search(t) or _EXPLICIT_SEARCH_HI.search(t):
        return True
    # Cricket callers often ask "next match / schedule" without saying "live/current".
    if _CRICKET_LIVE_INTENT.search(t):
        return True
    if _REALTIME_EN.search(t) or _REALTIME_HI.search(t):
        return True
    return False


def build_search_query(user_text: str, *, max_len: int = 120) -> str:
    """Turn conversational STT into a focused web query."""
    raw = re.sub(r"\s+", " ", (user_text or "").strip())
    raw = re.sub(r"[^\w\s\u0900-\u097F.,?!'-]", " ", raw)
    lower = raw.lower()

    # Drop common Hindi/English filler so search engines get the topic.
    drop = (
        r"\b(please|kindly|tell me|batao|batado|bata do|bataiye|bolo|boliye|"
        r"search|google|online|live|web|internet|karke|kar ke|nahi|nahin|no)\b",
        r"(कृपया|बताओ|बता दो|बताइए|सर्च|खोज|ऑनलाइन|लाइव|लाइफ|वेब|गूगल|करके|नहीं|मत)",
    )
    q = lower
    for pat in drop:
        q = re.sub(pat, " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()

    if _mentions_gold(raw):
        return "24 carat gold price per 10 gram india today INR live"[:max_len]
    if _CRICKET.search(raw):
        when_match = re.search(
            r"\b(today|tonight|tomorrow|day after tomorrow|this week|weekend)\b|"
            r"(आज|आज\s*रात|कल|परसों|इस\s*हफ्ते|वीकेंड)",
            raw,
            re.I,
        )
        if when_match:
            when_word = when_match.group(0).strip().lower()
            return f"IPL 2026 schedule {when_word} cricbuzz"[:max_len]
        if re.search(
            r"(next|upcoming|schedule|fixture|नेक्स्ट|अगला|मेच|मैच|शेड्यूल|फिक्स्चर)",
            raw,
            re.I,
        ):
            return "IPL 2026 next match schedule cricbuzz"[:max_len]
        if re.search(r"(score|live|result|स्कोर|लाइव|रिजल्ट)", raw, re.I):
            return "IPL 2026 live score cricbuzz"[:max_len]
        # Team-name only queries: include the team in the search.
        team_match = re.search(
            r"(gujarat titans?|mumbai indians?|chennai super kings?|"
            r"royal challengers?|rajasthan royals?|sunrisers hyderabad|"
            r"punjab kings?|delhi capitals?|kolkata knight riders?|"
            r"lucknow super giants?|rcb|csk|mi|kkr|gt|srh|rr|dc|pbks|lsg)",
            raw,
            re.I,
        )
        if team_match:
            return f"{team_match.group(0)} next match cricbuzz"[:max_len]
        return "IPL 2026 cricbuzz latest match update"[:max_len]
    if _SILVER.search(raw):
        return "silver price india today per kg INR live"[:max_len]
    if _PETROL.search(raw):
        return "petrol diesel price today india INR"[:max_len]
    if _STOCK.search(raw):
        return "nifty sensex today live india"[:max_len]
    if _WEATHER.search(raw):
        # Strip weather/conversational words, keep what's likely a city name.
        city_part = q
        # Drop preamble fillers Whisper often emits ("okay thank you so how can i…").
        # We strip these wherever they occur — for weather lookups they're never
        # part of the city name.
        city_part = re.sub(
            r"\b(okay|ok|um|uh|so|well|hey|hi|hello|"
            r"thank\s*you|thanks|please|alright|right)\b",
            " ",
            city_part,
            flags=re.IGNORECASE,
        )
        city_part = re.sub(
            r"\bhow\s+can\s+(i|you)[a-z\s,]*\b",
            " ",
            city_part,
            flags=re.IGNORECASE,
        )
        city_part = re.sub(
            r"\b(weather|today|tomorrow|tonight|forecast|rain|temperature|"
            r"now|right now|currently|please|tell me|what is|whats|the|in|at|"
            r"can you|will you|how|is|it|me|about|going to be|so|just|then)\b",
            " ",
            city_part,
            flags=re.IGNORECASE,
        )
        city_part = re.sub(
            r"(मौसम|बारिश|आज|कल|अभी|बताइए|बताओ|कैसा|कैसी|क्या|है|में|का|की)",
            " ",
            city_part,
        )
        city_part = re.sub(r"[,.!?'\"]+", " ", city_part)
        city_part = re.sub(r"\s+", " ", city_part).strip()
        # If only filler is left, default to a major Indian city for the demo.
        if (
            city_part
            and 2 <= len(city_part) <= 40
            and re.search(r"[A-Za-z\u0900-\u097F]{3,}", city_part)
        ):
            return f"weather today {city_part} india"[:max_len]
        return "weather today noida india"[:max_len]

    if not q:
        q = raw[:max_len]
    return (q or "news india today")[:max_len].strip()


def _mentions_gold(text: str) -> bool:
    t = text or ""
    if _GOLD.search(t):
        return True
    return any(x in t for x in ("सोना", "सोने", "सोन", "गोल्ड"))


def detect_search_topic(user_text: str) -> str:
    raw = user_text or ""
    if _mentions_gold(raw):
        return "gold"
    if "रेट" in raw and _mentions_gold(raw):
        return "gold"
    if "रेट" in raw and re.search(r"gold|गोल्ड", raw, re.I):
        return "gold"
    if _SILVER.search(raw):
        return "silver"
    if _WEATHER.search(raw):
        return "weather"
    if _PETROL.search(raw):
        return "petrol"
    if _STOCK.search(raw):
        return "stock"
    if _CRICKET.search(raw):
        return "cricket"
    if _REALTIME_EN.search(raw) and re.search(r"news|खबर", raw, re.I):
        return "news"
    return "general"


def _topic_queries(user_text: str, topic: str) -> tuple[str, ...]:
    if topic == "gold":
        return _GOLD_QUERIES
    if topic == "silver":
        return _SILVER_QUERIES
    if topic == "weather":
        city = build_search_query(user_text, max_len=40)
        return (f"weather forecast today {city}",) + _WEATHER_QUERIES[:1]
    if topic == "petrol":
        return _PETROL_QUERIES
    if topic == "cricket":
        return _CRICKET_QUERIES
    return (build_search_query(user_text),)


def _is_blocked_domain(url: str) -> bool:
    u = (url or "").lower()
    return any(b in u for b in _BLOCKED_DOMAINS)


def _score_hit(hit: SearchHit, *, topic: str) -> int:
    if _is_blocked_domain(hit.link):
        return -1000
    blob = f"{hit.link} {hit.title} {hit.snippet}".lower()
    score = 0
    good = _TOPIC_GOOD_DOMAINS.get(topic) or _TOPIC_GOOD_DOMAINS.get("news", ())
    for dom in good:
        if dom in blob:
            score += 50
    if topic in ("gold", "silver", "petrol", "stock"):
        if re.search(r"\d", hit.snippet):
            score += 20
        if any(w in blob for w in ("today", "live", "per gram", "10 gram", "inr", "rupee", "₹")):
            score += 15
        if "wikipedia" in blob:
            score -= 500
    if topic == "weather" and any(w in blob for w in ("°", "celsius", "rain", "forecast")):
        score += 15
    if topic == "cricket" and any(
        w in blob for w in ("live", "score", "match", "cricket", "ipl", "commentary")
    ):
        score += 20
    return score


def _rank_hits(hits: list[SearchHit], *, topic: str, limit: int) -> list[SearchHit]:
    seen: set[str] = set()
    scored: list[tuple[int, SearchHit]] = []
    for h in hits:
        key = (h.link or h.title).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        s = _score_hit(h, topic=topic)
        if s > -500:
            scored.append((s, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [h for _, h in scored[:limit]]
    if out:
        log.info(
            "Ranked %s web hits for topic=%s (top domain=%s)",
            len(out),
            topic,
            (out[0].link or "")[:60],
        )
    return out


_INR_IN_TEXT = re.compile(
    r"₹\s*([\d,]+(?:\.\d+)?)|"
    r"(?:Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)|"
    r"([\d,]+)\s*/\s*gram",
    re.IGNORECASE,
)


def extract_price_facts(hits: list[SearchHit], *, topic: str) -> list[str]:
    """Pull ₹ amounts from snippets so the LLM states the same figure."""
    facts: list[str] = []
    seen: set[str] = set()
    for h in hits:
        blob = f"{h.title} {h.snippet}"
        for m in _INR_IN_TEXT.finditer(blob):
            val = next(g for g in m.groups() if g)
            if not val:
                continue
            key = val.replace(",", "").strip()
            if not key.isdigit():
                continue
            amount = int(key)
            if topic == "gold" and not (8_000 <= amount <= 30_000):
                continue
            if key in seen:
                continue
            seen.add(key)
            ctx = blob[max(0, m.start() - 20) : m.end() + 50].strip()
            unit = "per gram" if re.search(r"per\s*gram|/gram|प्रति\s*ग्राम", ctx, re.I) else ""
            facts.append(f"₹{amount:,} {unit}".replace(",", ","))
        if len(facts) >= 3:
            break
    return facts[:3]


def _cricket_summary_from_hits(hits: list[SearchHit]) -> tuple[list[str], list[str]]:
    """Pull `LIVE:` / `UPCOMING:` lists out of the structured Cricbuzz hits."""
    live: list[str] = []
    upcoming: list[str] = []
    for h in hits:
        snip = (h.snippet or "").strip()
        if not snip:
            continue
        body = snip
        bucket: list[str] | None = None
        if body.upper().startswith("LIVE:"):
            bucket = live
            body = body[len("LIVE:"):].strip()
        elif body.upper().startswith("UPCOMING:"):
            bucket = upcoming
            body = body[len("UPCOMING:"):].strip()
        if bucket is None:
            continue
        for item in body.split("|"):
            item = item.strip(" -—|")
            if not item or len(item) < 4:
                continue
            if item.lower() in {x.lower() for x in bucket}:
                continue
            bucket.append(item)
    return live[:5], upcoming[:6]


def format_context_for_llm(hits: list[SearchHit], *, query: str, topic: str = "general") -> str:
    if not hits:
        return (
            " A web search returned no useful live snippets. Apologize briefly in one "
            "sentence; do not cite Wikipedia or guess a number."
        )

    if topic == "cricket":
        live, upcoming = _cricket_summary_from_hits(hits)
        q_lower = (query or "").lower()
        is_ipl_query = bool(re.search(r"\bipl\b|आईपीएल|आइपिल|आईपिल", q_lower))
        any_ipl_in_data = any(
            re.search(r"\bipl\b|गुजरात|मुंबई|चेन्नई|राजस्थान|कोलकाता", item, re.IGNORECASE)
            for item in (live + upcoming)
        )

        if live or upcoming:
            cricket_lines: list[str] = [
                " Live cricket data from Cricbuzz. Answer in ONE short spoken sentence "
                "using the lines below. If caller asks 'next match' or 'schedule', use "
                "UPCOMING. If they ask 'live/score', use LIVE. Speak team names, status, "
                "and time/date when present. Never read URLs."
            ]
            if is_ipl_query and not any_ipl_in_data:
                cricket_lines.append(
                    " IMPORTANT: caller asked about IPL specifically but no IPL match "
                    "is in the lists below — the IPL season is between matches right now. "
                    "Tell the caller in ONE sentence: 'IPL isn't running right now; the "
                    "current cricket on Cricbuzz is <name one upcoming or live match>.'"
                )
            if live:
                cricket_lines.append(" LIVE: " + " | ".join(live))
            if upcoming:
                cricket_lines.append(" UPCOMING: " + " | ".join(upcoming))
            cricket_lines.append(f" (query: {query[:80]})")
            return " " + " ".join(cricket_lines)
        # No structured lines — fall through to generic format with cricket framing.
        lines = [
            " Live cricket results context. Give one short spoken update with "
            "teams/score/status. Do not invent details and do not read URLs:"
        ]
    elif topic == "weather":
        lines = [
            " Live weather data from accuweather/weather.com. Answer in ONE short "
            "spoken sentence with temperature, condition (rain/clear/cloudy), and the "
            "city. If the caller didn't name a city, assume Noida/Delhi NCR from "
            "the snippet below and say the city you used. Do not say 'which city' "
            "— that wastes a turn. Do not read URLs."
        ]
    elif topic in ("gold", "silver"):
        lines = [
            " Live results from gold/price sites (NOT Wikipedia). "
            "Use ONLY the rupee amounts below; one short spoken sentence with carat + per 10 gram "
            "or per gram; do not invent numbers; do not read URLs:"
        ]
    elif topic in ("petrol", "stock"):
        lines = [
            " Live price results. Use ONLY the amounts below in one short spoken sentence; "
            "do not invent numbers; do not read URLs:"
        ]
    elif topic == "news":
        lines = [
            " Latest news results. Sum up the most relevant one or two items in short "
            "spoken sentences; do not invent details; do not read URLs:"
        ]
    else:
        lines = [
            " Fresh web results for the caller's question. Answer it in one or two short "
            "spoken sentences using only facts from these results. If they do not contain "
            "the answer, say you couldn't find that. Do not invent numbers; do not read URLs:"
        ]

    prices = extract_price_facts(hits, topic=topic)
    if prices:
        lines.append(" Extracted prices: " + " | ".join(prices))
    for i, h in enumerate(hits, 1):
        bit = f"{i}. {h.title}: {h.snippet}"
        lines.append(bit[:280])
    lines.append(f" (query: {query[:80]})")
    return " " + " ".join(lines)


def _hits_from_ddg_rows(rows: list[dict[str, Any]], *, limit: int) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        title = (row.get("title") or "").strip()
        snippet = (row.get("body") or row.get("snippet") or "").strip()
        link = (row.get("href") or row.get("link") or "").strip()
        if title or snippet:
            hits.append(SearchHit(title=title, snippet=snippet, link=link))
    return hits


def _ddg_text_sync(
    q: str,
    *,
    max_results: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore[no-redef]
        except ImportError:
            log.warning("Install ddgs: pip install ddgs")
            return []
    try:
        with DDGS(timeout=timeout_seconds) as ddgs:
            return list(
                ddgs.text(
                    q,
                    max_results=max(1, min(max_results, 10)),
                    region="in-in",
                    # Several engines so one rate-limited engine does not empty the
                    # result; skip wikipedia/grokipedia ("auto" adds them) — useless
                    # for live facts. (ddgs no longer ships a bing backend.)
                    backend="duckduckgo,brave,mojeek,yahoo,startpage,google",
                )
            )
    except Exception as exc:
        log.warning("DuckDuckGo search failed for %r: %s", q[:60], exc)
        return []


def _search_duckduckgo_sync(
    query: str,
    *,
    num_results: int,
    timeout_seconds: float,
    topic: str = "general",
) -> list[SearchHit]:
    t0 = time.perf_counter()
    if topic == "general" and _mentions_gold(query):
        topic = "gold"
    queries = _topic_queries(query, topic) if topic != "general" else (build_search_query(query),)
    per_query = max(4, num_results + 2)
    all_rows: list[dict[str, Any]] = []
    for q in queries[:3]:
        all_rows.extend(
            _ddg_text_sync(q, max_results=per_query, timeout_seconds=timeout_seconds)
        )

    raw_hits = _hits_from_ddg_rows(all_rows, limit=per_query * 3)
    hits = _rank_hits(raw_hits, topic=topic, limit=num_results)
    log.info(
        "DuckDuckGo topic=%s queries=%s raw=%s ranked=%s in %.2fs",
        topic,
        len(queries),
        len(raw_hits),
        len(hits),
        time.perf_counter() - t0,
    )
    return hits


async def search_duckduckgo(
    query: str,
    *,
    num_results: int = 3,
    timeout_seconds: float = 10.0,
    topic: str = "general",
) -> list[SearchHit]:
    return await asyncio.to_thread(
        _search_duckduckgo_sync,
        query,
        num_results=num_results,
        timeout_seconds=timeout_seconds,
        topic=topic,
    )


def _hits_from_serpapi_dict(data: dict[str, Any], *, limit: int) -> list[SearchHit]:
    raw: list[SearchHit] = []
    answer_box = data.get("answer_box")
    if isinstance(answer_box, dict):
        title = (answer_box.get("title") or "Google answer").strip()
        snippet = (
            answer_box.get("snippet")
            or answer_box.get("answer")
            or answer_box.get("price")
            or ""
        )
        if isinstance(snippet, (int, float)):
            snippet = str(snippet)
        snippet = str(snippet).strip()
        link = (answer_box.get("link") or "").strip()
        if snippet:
            raw.append(SearchHit(title=title, snippet=snippet, link=link))

    for item in data.get("organic_results") or []:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        link = (item.get("link") or "").strip()
        if title or snippet:
            raw.append(SearchHit(title=title, snippet=snippet, link=link))
        if len(raw) >= limit * 2:
            break
    return raw


def _search_serpapi_sync(
    query: str,
    *,
    api_key: str,
    num_results: int,
    topic: str,
) -> list[SearchHit]:
    """Google results via SerpAPI (https://serpapi.com)."""
    key = (api_key or "").strip()
    if not key:
        return []

    q = build_search_query(query)
    t0 = time.perf_counter()
    try:
        from serpapi import GoogleSearch
    except ImportError:
        log.warning("Install SerpAPI: pip install google-search-results")
        return []

    params: dict[str, Any] = {
        "engine": "google",
        "q": q,
        "api_key": key,
        "google_domain": "google.co.in",
        "gl": "in",
        "hl": "en",
        "num": max(num_results, 5),
    }
    try:
        data = GoogleSearch(params).get_dict()
    except Exception as exc:
        log.warning("SerpAPI search failed for %r: %s", q[:60], exc)
        return []

    raw = _hits_from_serpapi_dict(data, limit=num_results)
    hits = _rank_hits(raw, topic=topic, limit=num_results)
    log.info(
        "SerpAPI search %r topic=%s -> %s hits in %.2fs",
        q[:50],
        topic,
        len(hits),
        time.perf_counter() - t0,
    )
    return hits


async def search_serpapi(
    query: str,
    *,
    api_key: str,
    num_results: int = 3,
    topic: str = "general",
) -> list[SearchHit]:
    return await asyncio.to_thread(
        _search_serpapi_sync,
        query,
        api_key=api_key,
        num_results=num_results,
        topic=topic,
    )


async def search_google(
    query: str,
    *,
    api_key: str,
    cx: str,
    num_results: int = 3,
    timeout_seconds: float = 8.0,
) -> list[SearchHit]:
    """Google Programmable Search Engine (Custom Search JSON API)."""
    key = (api_key or "").strip()
    engine_id = (cx or "").strip()
    if not key or not engine_id:
        return []

    q = build_search_query(query)
    params = {
        "key": key,
        "cx": engine_id,
        "q": q,
        "num": max(1, min(num_results, 5)),
        "safe": "active",
    }
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            r = await client.get(_GOOGLE_CSE_URL, params=params)
            r.raise_for_status()
            data: dict[str, Any] = r.json()
    except httpx.HTTPError as exc:
        log.warning("Google search failed for %r: %s", q[:60], exc)
        return []

    items = data.get("items") or []
    raw: list[SearchHit] = []
    for item in items[: num_results * 2]:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        link = (item.get("link") or "").strip()
        if title or snippet:
            raw.append(SearchHit(title=title, snippet=snippet, link=link))

    topic = detect_search_topic(query)
    hits = _rank_hits(raw, topic=topic, limit=num_results)
    log.info(
        "Google search %r topic=%s -> %s hits in %.2fs",
        q[:50],
        topic,
        len(hits),
        time.perf_counter() - t0,
    )
    return hits


async def search_web(
    query: str,
    *,
    api_key: str = "",
    cx: str = "",
    serpapi_api_key: str = "",
    provider: str = "auto",
    num_results: int = 3,
    timeout_seconds: float = 10.0,
    crawl_max_pages: int = 1,
) -> tuple[list[SearchHit], str]:
    """
    Run web search.

    provider: fast | auto | crawl | serpapi | google | duckduckgo
    fast = DuckDuckGo/multi-engine snippets only (~1 s).
    crawl = duckduckgo-search + crawl4ai (Playwright) + BeautifulSoup + LangChain
    auto = crawl first, then SerpAPI / Google CSE / DuckDuckGo snippets.
    """
    prov = (provider or "auto").strip().lower()
    topic = detect_search_topic(query)
    if topic == "general" and _mentions_gold(query):
        topic = "gold"
    built_q = build_search_query(query)

    if prov == "fast":
        # Snippets only: the page crawl also starts from these engines, so when
        # they are rate-limited it just adds seconds of silence on the call.
        hits = await search_duckduckgo(
            query,
            num_results=num_results,
            timeout_seconds=timeout_seconds,
            topic=topic,
        )
        return hits, built_q

    use_crawl = prov in ("crawl", "auto")
    if use_crawl:
        try:
            from ari_app.live_search_crawl import search_live_crawl

            hits = await search_live_crawl(
                query,
                num_results=num_results,
                max_pages_to_crawl=crawl_max_pages,
                timeout_seconds=timeout_seconds,
            )
            if hits:
                return hits, built_q
            log.info("Live crawl returned no hits; trying API/snippet fallback")
        except Exception as exc:
            log.warning("Live crawl search failed: %s", exc)
        if prov == "crawl":
            hits = await search_duckduckgo(
                query,
                num_results=num_results,
                timeout_seconds=timeout_seconds,
                topic=topic,
            )
            return hits, built_q

    use_serpapi = prov == "serpapi" or (
        prov == "auto" and is_serpapi_configured(api_key=serpapi_api_key)
    )
    use_google = prov == "google" or (
        prov == "auto"
        and not use_serpapi
        and is_google_configured(api_key=api_key, cx=cx)
    )

    if use_serpapi and is_serpapi_configured(api_key=serpapi_api_key):
        hits = await search_serpapi(
            query,
            api_key=serpapi_api_key,
            num_results=num_results,
            topic=topic,
        )
        if hits:
            return hits, built_q
        log.info("SerpAPI returned no hits; trying fallback search")

    if use_google and is_google_configured(api_key=api_key, cx=cx):
        hits = await search_google(
            query,
            api_key=api_key,
            cx=cx,
            num_results=num_results,
            timeout_seconds=timeout_seconds,
        )
        if hits:
            return hits, built_q
        log.info("Google CSE returned no hits; trying DuckDuckGo fallback")

    hits = await search_duckduckgo(
        query,
        num_results=num_results,
        timeout_seconds=timeout_seconds,
        topic=topic,
    )
    return hits, built_q
