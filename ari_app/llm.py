"""Chat completion via vLLM OpenAI-compatible server."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

log = logging.getLogger(__name__)

# Real-time phone agent: everything the model writes is spoken aloud by TTS.
# Positive, concrete guidance works better than long "never do X" lists on small models.
_PHONE_AGENT_CORE_EN = (
    "You are a warm, capable customer-service agent on a live phone call. Everything you "
    "write is spoken aloud, so write exactly what you would say.\n"
    "How to talk:\n"
    "- Answer what the caller asked first, directly, in one or two short sentences (usually "
    "under 30 words). A brief natural acknowledgement ('Sure', 'Got it', 'I understand') is "
    "fine, but don't open every reply with one.\n"
    "- Remember everything said earlier in this call; use what the caller already told you "
    "and never ask for it again. Use their name only if you know it, and only now and then.\n"
    "- Ask a question only when you need it to help (to clarify, or to offer a next step). "
    "Ask 'Is there anything else I can help with?' only when their request is fully done, "
    "never twice in a row.\n"
    "- Simple everyday words and contractions. Calm and kind when the caller is upset, "
    "simpler when they are confused.\n"
    "Customer care:\n"
    "- If the caller is upset, frustrated or rude, acknowledge their feeling once, in your "
    "own words, never argue or get defensive, don't over-apologise, and move to a concrete "
    "next step: answer, register a complaint, or say the team will follow up.\n"
    "- Company facts (policies, refunds, prices, products, delivery times, hours, offers) may "
    "come ONLY from 'About the company' or the reference information you are given. If the "
    "answer isn't there, say plainly that you don't have that information and offer to "
    "register it as a request or complaint so the team can get back to them. Never guess or "
    "make up a policy, price, date, name, order detail or ticket status.\n"
    "- The same for the caller's own account, orders or changes made to it: if it's not in "
    "the caller info, you don't know it. Don't promise to check, look into or fix something "
    "you cannot do on this call; offer to register it for the team instead.\n"
    "- Never send the caller to other websites, phone numbers, apps or organisations "
    "(government portals, other companies, helplines) unless they appear in 'About the "
    "company'. Complaints and requests are handled here, on this call.\n"
    "- If asked what you can help with, say it briefly: questions about the company, "
    "registering a complaint, and the status of their complaints.\n"
    "- You can't change account details (name, number, PIN) yourself; say the team or the "
    "account settings handle that, and offer to raise it.\n"
    "- If you didn't catch what they said, say so briefly and ask them to repeat it.\n"
    "- If they thank you or say goodbye, close warmly in one line.\n"
    "You can look things up online: when fresh facts are needed, the system searches the web "
    "and gives you the results, so never say you can't search or have no internet access.\n"
    "Never use lists, markdown, emojis, or placeholders like [Your Name]. Don't say 'How may "
    "I assist you' or 'Thank you for calling'. If asked who you are, give your name and say "
    "you're the company's phone assistant; mention being an AI only if asked directly.\n"
    "Complaints: you cannot change, close, resolve or escalate a complaint yourself. If "
    "asked to, say you have noted it and the team will update the ticket. Only state a "
    "complaint's status or resolution if it appears in the caller info; never guess one. "
    "Never ask the caller for a ticket or complaint number: their complaints are already "
    "in the caller info once they are verified."
)

_PHONE_AGENT_CORE_HI = (
    "आप लाइव फ़ोन कॉल पर एक गर्मजोशी भरे, समझदार कस्टमर-सर्विस एजेंट हैं। "
    "आप जो लिखेंगे वह सीधे बोला जाएगा, इसलिए वही लिखें जो फ़ोन पर बोलते हैं।\n"
    "कैसे बात करें:\n"
    "- कॉलर ने जो पूछा, पहले उसका सीधा जवाब दें: एक या दो छोटे वाक्य, आमतौर पर 25 शब्दों से कम। "
    "'जी', 'ठीक है' जैसी छोटी प्रतिक्रिया ठीक है, पर हर जवाब ऐसे शुरू न करें।\n"
    "- इस कॉल में पहले कही गई हर बात याद रखें; जो कॉलर बता चुके हैं वह दोबारा न पूछें। "
    "नाम तभी लें जब पता हो, और कभी-कभार।\n"
    "- सवाल तभी पूछें जब मदद के लिए ज़रूरी हो। 'और कुछ मदद कर सकती/सकता हूँ?' तभी पूछें जब "
    "उनका काम पूरा हो जाए, लगातार दो बार कभी नहीं।\n"
    "- रोज़मर्रा की आसान हिंदी। कॉलर परेशान हो तो नरमी से, उलझन में हो तो और आसान शब्दों में।\n"
    "ग्राहक सेवा:\n"
    "- कॉलर नाराज़, परेशान या रूखे हों तो अपने शब्दों में एक बार उनकी भावना मानें, "
    "कभी बहस न करें, बार-बार माफ़ी न माँगें, और ठोस अगला कदम बताएँ: जवाब, शिकायत दर्ज करना, या टीम का फ़ॉलो-अप।\n"
    "- कंपनी की बातें (नीति, रिफ़ंड, कीमत, प्रोडक्ट, डिलीवरी का समय, समय-सारणी, ऑफ़र) सिर्फ़ 'About the company' "
    "या दी गई संदर्भ जानकारी से बताएँ। वहाँ जवाब नहीं है तो साफ़ कहें कि यह जानकारी आपके पास नहीं है और "
    "टीम के लिए अनुरोध या शिकायत दर्ज करने की पेशकश करें। कोई नीति, कीमत, तारीख, नाम, ऑर्डर या टिकट स्टेटस कभी न बनाएँ।\n"
    "- कॉलर के खाते या ऑर्डर की बात कॉलर की जानकारी में नहीं है तो आपको नहीं पता। जो इस कॉल पर नहीं कर सकते, "
    "उसे चेक करने या ठीक करने का वादा न करें; टीम के लिए दर्ज करने की पेशकश करें।\n"
    "- कॉलर को कभी दूसरी वेबसाइट, फ़ोन नंबर, ऐप या संस्था (सरकारी पोर्टल, दूसरी कंपनी, हेल्पलाइन) पर न भेजें, "
    "जब तक वह 'About the company' में न हो। शिकायत और अनुरोध यहीं, इसी कॉल पर लिए जाते हैं।\n"
    "- कोई पूछे आप किसमें मदद कर सकती हैं, तो छोटे में बताएँ: कंपनी के बारे में सवाल, शिकायत दर्ज करना, और उनकी शिकायत का स्टेटस।\n"
    "- खाते की जानकारी (नाम, नंबर, पिन) आप खुद नहीं बदल सकते; बताएँ कि टीम यह करती है और इसे दर्ज करने की पेशकश करें।\n"
    "- बात समझ न आए तो छोटे में बताएँ और दोबारा बोलने को कहें।\n"
    "- धन्यवाद या अलविदा पर एक लाइन में प्यार से बात खत्म करें।\n"
    "आप ऑनलाइन जानकारी देख सकती/सकते हैं: ज़रूरत होने पर सिस्टम वेब पर खोजकर नतीजे देता है, "
    "इसलिए कभी न कहें कि आप सर्च नहीं कर सकते या इंटरनेट नहीं है।\n"
    "सूची, मार्कडाउन, इमोजी या [नाम] जैसे खाली निशान न लिखें। कोई पूछे आप कौन हैं तो अपना नाम और "
    "बताएँ कि आप कंपनी की फ़ोन सहायक हैं; AI का ज़िक्र तभी करें जब सीधे पूछा जाए। "
    "केवल देवनागरी में बोलचाल की हिंदी लिखें; अंग्रेज़ी के आम शब्द देवनागरी में चल सकते हैं।\n"
    "शिकायतें: आप खुद किसी शिकायत को बदल, बंद या हल नहीं कर सकते। ऐसा कहा जाए तो बताएँ कि "
    "आपने नोट कर लिया है और टीम टिकट अपडेट करेगी। शिकायत का स्टेटस या समाधान तभी बताएँ "
    "जब वह कॉलर की जानकारी में दिया हो; कभी अंदाज़ा न लगाएँ। कॉलर से कभी टिकट या "
    "शिकायत नंबर न माँगें; पहचान होने के बाद उनकी शिकायतें कॉलर की जानकारी में होती हैं।"
)

_PERSONA_EN = {
    "female": " Your voice is a woman's.",
    "male": " Your voice is a man's.",
    "neutral": " Keep wording about yourself gender-neutral.",
}

_PERSONA_HI = {
    "female": (
        " आप महिला हैं—अपने लिए हमेशा स्त्रीलिंग: समझ गई, कर सकती हूँ, बताती हूँ, देखती हूँ।"
    ),
    "male": (
        " आप पुरुष हैं—अपने लिए हमेशा पुल्लिंग: समझ गया, कर सकता हूँ, बताता हूँ, देखता हूँ।"
    ),
    "neutral": (
        " अपने लिए लिंग-तटस्थ वाक्य बोलें (जैसे 'बताइए', 'देखते हैं', 'हो सकता है')।"
    ),
}

_PERSONA_OTHER = {
    "female": " You are a woman; where the language marks gender, use feminine forms for yourself.",
    "male": " You are a man; where the language marks gender, use masculine forms for yourself.",
    "neutral": " Keep wording about yourself gender-neutral.",
}


def _reply_language_rule(reply_lang: str) -> str:
    from ari_app.lang import language_name

    name = language_name(reply_lang)
    return (
        f" The caller is speaking {name}. Reply only in natural, spoken {name}, "
        f"in its usual script. If the caller switches to another language, switch with them."
    )


# Never speak a stray "SEARCH: ..." line if the model writes one.
_SEARCH_MARKER = re.compile(r"SEARCH\s*:\s*[^\n]*", re.I)


def strip_search_lines(reply: str) -> str:
    return _SEARCH_MARKER.sub("", reply or "").strip()


_CALLER_GENDER_HI = {
    "male": " कॉलर पुरुष लगते हैं; उन्हें सम्मान से 'आप' कहें।",
    "female": " कॉलर महिला लगती हैं; उन्हें सम्मान से 'आप' कहें।",
}


def _client(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    connect_timeout_seconds: float,
) -> AsyncOpenAI:
    timeout = httpx.Timeout(
        timeout_seconds,
        connect=connect_timeout_seconds,
    )
    return AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout)


def _now_local():
    """Current time in CALL_TIMEZONE (default Asia/Kolkata)."""
    import os
    from datetime import datetime, timedelta, timezone

    name = os.getenv("CALL_TIMEZONE", "Asia/Kolkata")
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(name)), name
    except Exception:  # no tz database: fall back to India time
        return datetime.now(timezone(timedelta(hours=5, minutes=30))), "Asia/Kolkata"


def _business_and_time() -> str:
    """Who the bot works for, what it can do, and today's date/time (it can't know these)."""
    import os

    company = (os.getenv("COMPANY_NAME") or "").strip()
    agent = (os.getenv("AGENT_NAME") or "").strip()
    info = (os.getenv("COMPANY_INFO") or "").strip()
    now, tz = _now_local()
    parts = []
    if company or agent:
        who = f"Your name is {agent}" if agent else "You are the phone assistant"
        parts.append(f" {who}{f', the phone assistant of {company}' if company and agent else (f' of {company}' if company else '')}.")
    if info:
        parts.append(f" About the company: {info}")
    parts.append(
        " On this call you can: answer questions (from the company information and reference "
        "documents you are given, or general knowledge), register a complaint, tell the caller "
        "the status of their complaints, and look up current facts online."
    )
    parts.append(
        f" Today is {now:%A}, {now.day} {now:%B %Y}, and the time is {now:%I:%M %p} "
        f"({tz}). Use this for any question about today's date, day or time."
    )
    return "".join(parts)


def build_system_prompt(
    *,
    reply_lang: str,
    agent_gender: str = "neutral",
    caller_grammar_gender: str | None = None,
    caller_context: str | None = None,
    caller_number: str | None = None,
    caller_name: str | None = None,
    web_context: str | None = None,
) -> str:
    g = agent_gender if agent_gender in ("male", "female", "neutral") else "neutral"
    if reply_lang == "hi":
        system = _PHONE_AGENT_CORE_HI + _PERSONA_HI[g]
        cg = caller_grammar_gender if caller_grammar_gender in ("male", "female") else None
        if cg:
            system += _CALLER_GENDER_HI[cg]
    elif reply_lang == "en":
        system = _PHONE_AGENT_CORE_EN + _PERSONA_EN[g] + _reply_language_rule("en")
    else:
        # Any other caller language: English instructions, reply in their language.
        system = _PHONE_AGENT_CORE_EN + _PERSONA_OTHER[g] + _reply_language_rule(reply_lang)

    system += _business_and_time()

    if caller_name and caller_name not in ("-", "unknown"):
        if caller_context and "registered" in caller_context:
            # Name comes from our user database, so it is reliable.
            system += (
                f" The caller is {caller_name}, a registered customer. Use their name "
                "naturally now and then (not in every reply)."
            )
        else:
            system += f" The caller's name may be {caller_name}."
    if caller_context:
        system += f" Caller info (do not read out): {caller_context}."
    elif caller_number:
        system += f" Caller id {caller_number} (do not read digits aloud unless asked)."
    if web_context:
        system += (
            " Use the fresh facts below as if you simply know them: answer directly, "
            "and do not say you searched, looked it up, or checked online."
        )
        system += web_context
    return system


_WEB_LOOKUP_SYSTEM = (
    "You decide whether a phone assistant must search the web before answering the caller's "
    "latest message. Search if the caller wants a specific fact that is time-sensitive or that "
    "a small assistant could easily get wrong: prices, exchange rates, opening hours, schedules, "
    "flight or travel times, distances, populations, statistics, news, sports results, election "
    "results, business or contact details, product specifications, recent events, or who "
    "currently holds a position. Do not search for greetings, small talk, opinions, advice, "
    "how-to help, creative writing, simple common knowledge, or anything about the caller "
    "themselves. Only search when the caller is asking for information now; a statement "
    "about their plans or life (\"I'm going to Goa next week\") is not a request.\n"
    "Reply with only NO, or with one line: SEARCH: <short English web search query>. "
    "Use the conversation to make the query complete."
)
_LOOKUP_QUERY = re.compile(r"SEARCH\s*:\s*([^\n]+)", re.I)


async def web_lookup_query(
    *,
    base_url: str,
    api_key: str,
    model: str,
    user_text: str,
    history: list[dict],
    timeout_seconds: float = 5.0,
    context_messages: int = 6,
    force: bool = False,
) -> str | None:
    """
    Ask the model whether this turn needs a web lookup (cheap: a few tokens, ~0.15 s).

    Returns a context-complete search query (e.g. "flight time from Dar es Salaam to
    India" for "how long will the flight take?"), or None when no search is needed.
    """
    lines = []
    for msg in history[-context_messages:]:
        role = msg.get("role")
        content = re.sub(r"\s+", " ", (msg.get("content") or "").strip())
        if role in ("user", "assistant") and content:
            lines.append(f"{'Caller' if role == 'user' else 'Assistant'}: {content[:300]}")
    prompt = f"Caller's latest message: {user_text}"
    if force:
        prompt += (
            "\nThe caller is asking you to look something up, so reply with a SEARCH "
            "line. Work out from the conversation what they want found."
        )
    if lines:
        prompt = "Conversation so far:\n" + "\n".join(lines) + "\n\n" + prompt
    client = _client(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=min(3.0, timeout_seconds),
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _WEB_LOOKUP_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=32,
    )
    out = (resp.choices[0].message.content or "").strip() if resp.choices else ""
    m = _LOOKUP_QUERY.search(out)
    if not m:
        return None
    query = m.group(1).strip().strip("<>\"'`*.").strip()
    return query[:120] or None


async def check_vllm_reachable(
    *,
    base_url: str,
    api_key: str,
    connect_timeout_seconds: float = 10.0,
) -> bool:
    client = _client(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=connect_timeout_seconds + 5.0,
        connect_timeout_seconds=connect_timeout_seconds,
    )
    try:
        await client.models.list()
        return True
    except (APIConnectionError, APITimeoutError, TimeoutError) as exc:
        log.warning("vLLM not reachable at %s: %s", base_url, exc)
        return False
    except Exception as exc:
        log.warning("vLLM check failed at %s: %s", base_url, exc)
        return False


def _alternating_history(history: list[dict], cap: int) -> list[dict]:
    """
    Chat templates (Gemma especially) expect user/assistant turns to alternate and to
    start with the user. A call starts with the bot greeting and can have back-to-back
    bot lines (greeting + language question, retry prompts); without this the model
    loses the thread and "restarts" the call.
    """
    merged: list[dict] = []
    for msg in history[-cap:]:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1] = {"role": role, "content": merged[-1]["content"] + " " + content}
        else:
            merged.append({"role": role, "content": content})
    if merged and merged[0]["role"] == "assistant":
        merged.insert(0, {"role": "user", "content": "(call connected)"})
    return merged


def _chat_messages(
    *,
    reply_lang: str,
    agent_gender: str,
    caller_grammar_gender: str | None,
    caller_context: str | None,
    caller_number: str | None,
    caller_name: str | None,
    web_context: str | None,
    history: list[dict],
    user_text: str,
    max_history_messages: int,
) -> tuple[str, list[dict]]:
    lang = (reply_lang or "hi").strip().lower() or "hi"
    gender = agent_gender if agent_gender in ("male", "female", "neutral") else "neutral"
    system = build_system_prompt(
        reply_lang=lang,
        agent_gender=gender,
        caller_grammar_gender=caller_grammar_gender,
        caller_context=caller_context,
        caller_number=caller_number,
        caller_name=caller_name,
        web_context=web_context,
    )
    turns = _alternating_history(history, max(2, max_history_messages))
    if turns and turns[-1]["role"] == "user":
        # Previous caller turn got no reply (e.g. error); fold it into this one.
        user_text = turns.pop()["content"] + " " + user_text
    messages = [{"role": "system", "content": system}, *turns]
    messages.append({"role": "user", "content": user_text})
    return lang, messages


def _empty_reply_fallback(*, lang: str, gender: str) -> str:
    if lang == "hi":
        if gender == "male":
            return "माफ़ कीजिए, साफ़ नहीं सुनाई दिया। एक बार फिर बोलिए?"
        return "माफ़ कीजिए, साफ़ नहीं सुनाई दिया—एक बार फिर बोलिए?"
    return "Sorry, I didn't catch that—say it again?"


async def reply_text(
    *,
    base_url: str,
    api_key: str,
    model: str,
    user_text: str,
    history: list[dict],
    reply_lang: str = "hi",
    agent_gender: str = "neutral",
    caller_grammar_gender: str | None = None,
    caller_number: str | None = None,
    caller_name: str | None = None,
    caller_context: str | None = None,
    web_context: str | None = None,
    max_history_messages: int = 20,
    timeout_seconds: float = 120.0,
    connect_timeout_seconds: float = 15.0,
    max_tokens: int = 96,
    temperature: float = 0.82,
) -> str:
    client = _client(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
    )
    lang, messages = _chat_messages(
        reply_lang=reply_lang,
        agent_gender=agent_gender,
        caller_grammar_gender=caller_grammar_gender,
        caller_context=caller_context,
        caller_number=caller_number,
        caller_name=caller_name,
        web_context=web_context,
        history=history,
        user_text=user_text,
        max_history_messages=max_history_messages,
    )
    gender = agent_gender if agent_gender in ("male", "female", "neutral") else "neutral"
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    if not choice.message or not choice.message.content:
        return _empty_reply_fallback(lang=lang, gender=gender)
    return choice.message.content.strip()


async def reply_text_stream(
    *,
    base_url: str,
    api_key: str,
    model: str,
    user_text: str,
    history: list[dict],
    reply_lang: str = "hi",
    agent_gender: str = "neutral",
    caller_grammar_gender: str | None = None,
    caller_number: str | None = None,
    caller_name: str | None = None,
    caller_context: str | None = None,
    web_context: str | None = None,
    max_history_messages: int = 20,
    timeout_seconds: float = 120.0,
    connect_timeout_seconds: float = 15.0,
    max_tokens: int = 96,
    temperature: float = 0.82,
    max_sentences: int = 0,
) -> AsyncIterator[str]:
    """
    Yield each sentence as the model generates it (for speak-as-you-go TTS).
    """
    from ari_app.conversation import pop_complete_sentences

    client = _client(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
    )
    lang, messages = _chat_messages(
        reply_lang=reply_lang,
        agent_gender=agent_gender,
        caller_grammar_gender=caller_grammar_gender,
        caller_context=caller_context,
        caller_number=caller_number,
        caller_name=caller_name,
        web_context=web_context,
        history=history,
        user_text=user_text,
        max_history_messages=max_history_messages,
    )
    gender = agent_gender if agent_gender in ("male", "female", "neutral") else "neutral"

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    buffer = ""
    yielded = 0
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = (delta.content or "") if delta else ""
        if not piece:
            continue
        buffer += piece
        ready, buffer = pop_complete_sentences(buffer)
        for sent in ready:
            yielded += 1
            yield sent
            if max_sentences > 0 and yielded >= max_sentences:
                return

    tail = buffer.strip()
    if tail and (max_sentences <= 0 or yielded < max_sentences):
        yield tail
    elif yielded == 0:
        yield _empty_reply_fallback(lang=lang, gender=gender)
