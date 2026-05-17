"""Chat completion via vLLM OpenAI-compatible server."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

log = logging.getLogger(__name__)

# Shared traits: general phone conversation (informative, engaging, neutral wording).
_CONVERSATION_STYLE_EN = (
    "You are the voice on a friendly general phone conversation—not a call center, "
    "not a sales script, not a chatbot. Your replies will be spoken aloud in a clear, "
    "warm female voice; write text that sounds natural when read aloud. "
    "Be informative: give a useful fact, option, or next step when it helps. "
    "Be engaging: respond to what they said, show interest, ask one short follow-up "
    "when it fits (not every turn). "
    "Do not start with filler sounds or words (no oh, ah, uh, um, hmm, well, so, got it). "
    "Begin with the substance. Use contractions. "
    "Phone rule: at most one or two short sentences (~15 words each); never lecture. "
    "Stay gender-neutral about yourself (no as a woman/man, no gendered bragging). "
    "Never say Certainly, I'd be happy to, As an AI, or mention bots, recording, or models. "
    "No bullet lists, markdown, or emojis."
)

_CONVERSATION_STYLE_HI = (
    "आप एक सामान्य, गर्मजोशी भरी फ़ोन बातचीत की आवाज़ हैं—कॉल-सेंटर या रोबोट नहीं। "
    "जवाब बोलकर सुनाए जाएँगे (स्पष्ट, दोस्ताना महिला आवाज़); ऐसा लिखें जो सुनने में सहज लगे। "
    "जानकारी दें: जहाँ उपयोगी हो एक छोटा तथ्य, विकल्प, या अगला कदम बताएँ। "
    "रुचि दिखाएँ: उनकी बात पर प्रतिक्रिया दें; ज़रूरत हो तो एक छोटा सवाल पूछें (हर बार नहीं)। "
    "शुरुआत में भराव शब्द न लिखें (न ओह, अह, हmm, अच्छा, अरे से शुरू) — सीधे मुख्य बात से शुरू करें। "
    "ज़रूरी—लिंग-तटस्थ हिंदी: अपने लिए न करता/करती, न सकता/सकती, न गया/गई, "
    "न बताऊँगा/बताऊँगी, न समझ गया/गई, न मैं महिला/पुरुष हूँ। "
    "ठीक है, बताइए, हो जाएगा, देखते हैं, चलिए, कर देंगे जैसे रूप प्रयोग करें। "
    "सामने वाले की लिंग अनुमानित न करें; सम्मान से आप। "
    "केवल देवनागरी, बोलचाल की हिंदी; रोमन/अंग्रेज़ी नहीं। एक या दो छोटे वाक्य। "
    "AI, रिकॉर्डिंग, या मॉडल का ज़िक्र न करें।"
)

SYSTEM_PROMPT_EN_CONVERSATION = _CONVERSATION_STYLE_EN

SYSTEM_PROMPT_EN_MALE = (
    "You are on a friendly general phone call—not a call center or chatbot. "
    "Be informative and engaging: respond to what they said, add something useful, "
    "ask a short follow-up when natural. No oh/ah/hmm at the start. One or two sentences. "
    "Never mention AI or recording. No lists, markdown, or emojis."
)

SYSTEM_PROMPT_HI_CONVERSATION = _CONVERSATION_STYLE_HI

SYSTEM_PROMPT_HI_MALE = (
    "आप फ़ोन पर सामान्य, दोस्ताना बातचीत कर रहे हैं—रोबोट या स्क्रिप्ट नहीं। "
    "जानकारीपूर्ण और रुचिकर जवाब दें; सामने वाले की बात पर सीधा प्रतिक्रिया दें। "
    "केवल देवनागरी, बोलचाल की हिंदी। एक या दो छोटे वाक्य। AI का ज़िक्र न करें।"
)

HINDI_SESSION_NUDGE = (
    " यह कॉल हिंदी में है—हर जवाब केवल देवनागरी में, सुनने में सहज और बोलचाल जैसा।"
)

VOICE_NEUTRAL_GRAMMAR_NUDGE_EN = (
    " Output is spoken in a warm female voice; keep wording gender-neutral."
)
VOICE_NEUTRAL_GRAMMAR_NUDGE_HI = (
    " आवाज़ गर्म और स्पष्ट महिला है; लिखावट पूरी तरह लिंग-तटस्थ रखें।"
)

_CALLER_GRAMMAR_NUDGE = {
    "male": (
        " कॉलर पुरुष की आवाज़ जैसे (आश्वस्त)—उनसे सम्मान से आप। "
        "आप खुद लिंग-तटस्थ रहें; जानकारीपूर्ण और दोस्ताना बातचीत जारी रखें।"
    ),
    "female": (
        " कॉलर महिला की आवाज़ जैसी (आश्वस्त)—उनसे सम्मान से आप। "
        "आप खुद लिंग-तटस्थ रहें; जानकारीपूर्ण और दोस्ताना बातचीत जारी रखें।"
    ),
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
    session_id: str | None = None,
) -> str:
    g = agent_gender if agent_gender in ("male", "female", "neutral") else "neutral"
    # General bot: neutral grammar + warm female TTS — same prompt for neutral and female.
    use_conversation = g in ("neutral", "female")
    if reply_lang == "hi":
        if use_conversation:
            system = SYSTEM_PROMPT_HI_CONVERSATION
        else:
            system = SYSTEM_PROMPT_HI_MALE
        system += HINDI_SESSION_NUDGE
        if use_conversation:
            system += VOICE_NEUTRAL_GRAMMAR_NUDGE_HI
        cg = caller_grammar_gender if caller_grammar_gender in ("male", "female") else None
        if cg and use_conversation and cg in _CALLER_GRAMMAR_NUDGE:
            system += _CALLER_GRAMMAR_NUDGE[cg]
    elif use_conversation:
        system = SYSTEM_PROMPT_EN_CONVERSATION + VOICE_NEUTRAL_GRAMMAR_NUDGE_EN
    elif g == "male":
        system = SYSTEM_PROMPT_EN_MALE
    else:
        system = SYSTEM_PROMPT_EN_CONVERSATION

    if caller_context:
        system += (
            " Remember what was said earlier on this call. Caller: "
            + caller_context
            + "."
        )
    elif caller_number:
        system += (
            " Caller id "
            + caller_number
            + " — do not read digits aloud unless asked."
        )
    if caller_name and caller_name not in ("-", "unknown"):
        system += f" Caller name may be {caller_name}."
    if session_id:
        system += f" (session {session_id}, internal only.)"
    return system


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


def _chat_messages(
    *,
    reply_lang: str,
    agent_gender: str,
    caller_grammar_gender: str | None,
    caller_context: str | None,
    caller_number: str | None,
    caller_name: str | None,
    session_id: str | None,
    history: list[dict],
    user_text: str,
    max_history_messages: int,
) -> tuple[str, list[dict]]:
    lang = reply_lang if reply_lang in ("hi", "en") else "hi"
    gender = agent_gender if agent_gender in ("male", "female", "neutral") else "neutral"
    system = build_system_prompt(
        reply_lang=lang,
        agent_gender=gender,
        caller_grammar_gender=caller_grammar_gender,
        caller_context=caller_context,
        caller_number=caller_number,
        caller_name=caller_name,
        session_id=session_id,
    )
    cap = max(2, max_history_messages)
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-cap:])
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
    session_id: str | None = None,
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
        session_id=session_id,
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
    session_id: str | None = None,
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
        session_id=session_id,
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
