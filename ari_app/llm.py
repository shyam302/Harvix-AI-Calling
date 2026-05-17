"""Chat completion via vLLM OpenAI-compatible server."""

from __future__ import annotations

import logging

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

log = logging.getLogger(__name__)

SYSTEM_PROMPT_EN_FEMALE = (
    "You are a warm woman on a live phone call—natural and relaxed. "
    "Default to ONE short sentence. Use contractions. No lists, markdown, or emojis. "
    "Never mention AI, bots, or recording."
)

SYSTEM_PROMPT_EN_MALE = (
    "You are a warm man on a live phone call—natural and relaxed. "
    "Default to ONE short sentence. Use contractions. No lists, markdown, or emojis. "
    "Never mention AI, bots, or recording."
)

SYSTEM_PROMPT_HI_FEMALE = (
    "आप एक भारतीय महिला की तरह फ़ोन पर बात कर रही हैं—गर्मजोशी भरी, सहज और संक्षिप्त। "
    "जवाब हमेशा आम बोलचाल की हिंदी में देवनागरी लिपि में लिखें। "
    "आमतौर पर एक छोटा वाक्य; ज़रूरत हो तभी दूसरा। जी, ठीक है, बिल्कुल स्वाभाविक रूप से। "
    "AI या रिकॉर्डिंग का ज़िक्र न करें।"
)

SYSTEM_PROMPT_HI_MALE = (
    "आप एक भारतीय पुरुष की तरह फ़ोन पर बात कर रहे हैं—गर्मजोशी भरे, सहज और संक्षिप्त। "
    "जवाब हमेशा आम बोलचाल की हिंदी में देवनागरी लिपि में लिखें। "
    "आमतौर पर एक छोटा वाक्य; ज़रूरत हो तभी दूसरा। जी, ठीक है, बिल्कुल स्वाभाविक रूप से। "
    "AI या रिकॉर्डिंग का ज़िक्र न करें।"
)

HINDI_SESSION_NUDGE = (
    " यह कॉल हिंदी में है—हर जवाब केवल देवनागरी में। "
    "कॉलर की तरह ही लिंग (मैं करता/करती हूँ) बनाए रखें।"
)


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
    agent_gender: str = "female",
    caller_context: str | None = None,
    caller_number: str | None = None,
    caller_name: str | None = None,
    session_id: str | None = None,
) -> str:
    male = agent_gender == "male"
    if reply_lang == "hi":
        system = (SYSTEM_PROMPT_HI_MALE if male else SYSTEM_PROMPT_HI_FEMALE) + (
            HINDI_SESSION_NUDGE
        )
    elif male:
        system = SYSTEM_PROMPT_EN_MALE
    else:
        system = SYSTEM_PROMPT_EN_FEMALE

    if caller_context:
        system += (
            " इसी कॉल में पहले जो बात हुई याद रखें। Caller: " + caller_context + "."
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


async def reply_text(
    *,
    base_url: str,
    api_key: str,
    model: str,
    user_text: str,
    history: list[dict],
    reply_lang: str = "hi",
    agent_gender: str = "female",
    caller_number: str | None = None,
    caller_name: str | None = None,
    caller_context: str | None = None,
    session_id: str | None = None,
    max_history_messages: int = 20,
    timeout_seconds: float = 120.0,
    connect_timeout_seconds: float = 15.0,
    max_tokens: int = 96,
) -> str:
    client = _client(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
    )
    lang = reply_lang if reply_lang in ("hi", "en") else "hi"
    gender = agent_gender if agent_gender in ("male", "female") else "female"
    system = build_system_prompt(
        reply_lang=lang,
        agent_gender=gender,
        caller_context=caller_context,
        caller_number=caller_number,
        caller_name=caller_name,
        session_id=session_id,
    )
    cap = max(2, max_history_messages)
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-cap:])
    messages.append({"role": "user", "content": user_text})
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.72,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    if not choice.message or not choice.message.content:
        if lang == "hi":
            if gender == "male":
                return "माफ़ कीजिए, साफ़ नहीं सुनाई दिया। एक बार फिर बोलिए?"
            return "माफ़ कीजिए, आपकी बात साफ़ नहीं सुनाई दी। एक बार फिर बोलिए?"
        return "Sorry, I missed that. Could you say it once more?"
    return choice.message.content.strip()
