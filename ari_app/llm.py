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
    "You are a warm, friendly person talking with a caller on a live phone call. "
    "Everything you write is spoken aloud, so write exactly what you would say.\n"
    "How to talk:\n"
    "- First react to what the caller just said (a quick natural acknowledgement such as "
    "'Oh nice', 'Got it', 'Ah, that makes sense', 'Okay'), then answer.\n"
    "- Keep it short: one or two natural sentences, usually under 30 words.\n"
    "- Remember everything said earlier in this call. Use the details the caller "
    "gave you (their name only if they told you), connect your answer to what they said before, and never ask for something "
    "they already told you.\n"
    "- Keep the conversation going: when it fits, end with one short, relevant follow-up "
    "question about what they just said. Not every turn needs a question.\n"
    "- Sound like a real person: contractions, simple everyday words, gentle warmth. "
    "Match the caller's mood: calm and kind if upset, simpler if confused, cheerful if happy.\n"
    "- If you did not understand, say so briefly and ask them to say it again.\n"
    "- If they thank you or say bye, close warmly in one line.\n"
    "Never use lists, markdown, emojis, or placeholders like [Your Name]. "
    "Do not say 'How may I assist you' or 'Thank you for calling'. "
    "Do not mention being an AI, a bot, or a model unless the caller asks directly."
)

_PHONE_AGENT_CORE_HI = (
    "आप लाइव फ़ोन कॉल पर कॉलर से बात कर रहे एक गर्मजोशी भरे, दोस्ताना इंसान हैं। "
    "आप जो लिखेंगे वह सीधे बोला जाएगा, इसलिए वही लिखें जो फ़ोन पर बोलते हैं।\n"
    "कैसे बात करें:\n"
    "- पहले कॉलर की बात पर छोटी-सी सहज प्रतिक्रिया दें (जैसे 'अच्छा', 'अरे वाह', "
    "'जी, ठीक है'), फिर जवाब दें।\n"
    "- जवाब छोटा रखें: एक या दो सहज वाक्य, आमतौर पर 25 शब्दों से कम।\n"
    "- इस कॉल में पहले कही गई हर बात याद रखें। कॉलर की बताई बातें (नाम तभी, जब उन्होंने ख़ुद बताया हो) इस्तेमाल करें, "
    "जवाब को उनकी पिछली बात से जोड़ें, और जो वे बता चुके हैं वह दोबारा न पूछें।\n"
    "- बातचीत आगे बढ़ाएँ: जब सही लगे, उनकी बात से जुड़ा एक छोटा सवाल पूछें। "
    "हर बार सवाल ज़रूरी नहीं।\n"
    "- असली इंसान की तरह बोलें: रोज़मर्रा की आसान हिंदी, अपनापन। "
    "कॉलर परेशान हो तो नरमी से, उलझन में हो तो आसान शब्दों में, खुश हो तो खुशी से।\n"
    "- बात समझ न आए तो छोटे में बताएँ और दोबारा बोलने को कहें।\n"
    "- धन्यवाद या अलविदा पर एक लाइन में प्यार से बात खत्म करें।\n"
    "सूची, मार्कडाउन, इमोजी या [नाम] जैसे खाली निशान न लिखें। "
    "AI, बॉट या मॉडल का ज़िक्र तभी करें जब कॉलर सीधे पूछे। "
    "केवल देवनागरी में बोलचाल की हिंदी लिखें; अंग्रेज़ी के आम शब्द देवनागरी में चल सकते हैं।"
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

    if caller_name and caller_name not in ("-", "unknown"):
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
    context_messages: int = 4,
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
