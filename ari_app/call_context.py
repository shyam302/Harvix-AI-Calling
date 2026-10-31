"""In-memory conversation context for one active phone call."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CallContext:
    """Chat history and metadata for a single call (lives until hangup)."""

    session_id: str
    channel_id: str
    caller_number: str | None = None
    caller_name: str | None = None
    dialed_number: str | None = None
    started_at: float = field(default_factory=time.time)
    messages: list[dict[str, str]] = field(default_factory=list)
    max_messages: int = 20
    preferred_lang: str = "hi"  # any language code — follows the caller, drives LLM + TTS
    caller_gender: str | None = None  # pitch and/or Wav2Vec2 gender model
    caller_gender_confident: bool = False
    caller_gender_model_done: bool = False  # Wav2Vec2 runs at most once per call
    caller_gender_from_model: bool = False  # True when Wav2Vec2 set caller_gender
    pitch_hz_history: list[float] = field(default_factory=list)
    gender_detect_caller: bool = True
    gender_adapt_when_confident: bool = True
    gender_mirror_caller: bool = False
    gender_mirror_grammar: bool = False
    gender_min_confident_samples: int = 4
    gender_confident_fraction: float = 0.75
    agent_gender: str = "neutral"  # LLM grammar: neutral | female | male
    agent_voice_gender: str = "female"  # TTS voice: female | male
    grammar_match_voice: bool = False
    hindi_grammar_use_agent_persona: bool = False  # else neutral until caller pitch known
    gender_grammar_until_confident: bool = True
    gender_grammar_min_turns: int = 2  # user turns before persona grammar may unlock
    # Last successful web-search snapshot — reused on cricket/news follow-ups
    # so a single fetch covers several conversational turns (e.g. "next match"
    # then "Gujarat Titans?" then "what time?").
    last_web_topic: str | None = None
    last_web_query: str | None = None
    last_web_context: str | None = None
    last_web_at: float = 0.0
    _logged_grammar_unlock: bool = field(default=False, repr=False)

    @classmethod
    def create(
        cls,
        *,
        channel_id: str,
        caller_number: str | None = None,
        caller_name: str | None = None,
        dialed_number: str | None = None,
        max_messages: int = 20,
        primary_lang: str = "hi",
        gender_detect_caller: bool = True,
        gender_adapt_when_confident: bool = True,
        gender_mirror_caller: bool = False,
        gender_mirror_grammar: bool = False,
        gender_min_confident_samples: int = 4,
        gender_confident_fraction: float = 0.75,
        agent_gender: str = "neutral",
        agent_voice_gender: str = "female",
        grammar_match_voice: bool = False,
        hindi_grammar_use_agent_persona: bool = False,
        gender_grammar_until_confident: bool = True,
        gender_grammar_min_turns: int = 2,
    ) -> CallContext:
        pl = (primary_lang or "hi").strip().lower() or "hi"
        ag = agent_gender if agent_gender in ("male", "female", "neutral") else "neutral"
        vg = agent_voice_gender if agent_voice_gender in ("male", "female") else "female"
        return cls(
            session_id=uuid.uuid4().hex[:12],
            channel_id=channel_id,
            caller_number=caller_number,
            caller_name=caller_name,
            dialed_number=dialed_number,
            max_messages=max(4, max_messages),
            preferred_lang=pl,
            gender_detect_caller=gender_detect_caller,
            gender_adapt_when_confident=gender_adapt_when_confident,
            gender_mirror_caller=gender_mirror_caller,
            gender_mirror_grammar=gender_mirror_grammar,
            gender_min_confident_samples=gender_min_confident_samples,
            gender_confident_fraction=gender_confident_fraction,
            agent_gender=ag,
            agent_voice_gender=vg,
            grammar_match_voice=grammar_match_voice,
            hindi_grammar_use_agent_persona=hindi_grammar_use_agent_persona,
            gender_grammar_until_confident=gender_grammar_until_confident,
            gender_grammar_min_turns=max(0, gender_grammar_min_turns),
        )

    def gender_grammar_ready(self) -> bool:
        """True when caller pitch is confident and enough user turns were heard."""
        if not self.gender_grammar_until_confident:
            return True
        if self.turn_count < self.gender_grammar_min_turns:
            return False
        return self.caller_gender_confident

    def set_caller_gender_from_model(
        self,
        gender: str,
        *,
        p_female: float,
        p_male: float,
    ) -> None:
        """One-shot gender from Common-Voice gender model (after greeting + long utterance)."""
        if gender not in ("male", "female"):
            return
        self.caller_gender = gender
        self.caller_gender_confident = True
        self.caller_gender_model_done = True
        self.caller_gender_from_model = True
        log.info(
            "Session %s caller gender from model -> %s (female=%.3f male=%.3f)",
            self.session_id,
            gender,
            p_female,
            p_male,
        )

    def set_lang_from_user(
        self,
        user_text: str,
        *,
        whisper_lang: str | None = None,
        whisper_prob: float | None = None,
        min_words: int = 2,
        min_prob: float = 0.6,
    ) -> str:
        """Follow whatever language the caller is speaking (sticky on weak evidence)."""
        from ari_app.lang import detect_reply_lang

        prev = self.preferred_lang
        self.preferred_lang = detect_reply_lang(
            user_text,
            whisper_lang=whisper_lang,
            whisper_prob=whisper_prob,
            current=prev,
            min_words=min_words,
            min_prob=min_prob,
        )
        if self.preferred_lang != prev:
            log.info(
                "Session %s language %s -> %s (whisper=%s p=%s): %r",
                self.session_id,
                prev,
                self.preferred_lang,
                whisper_lang or "-",
                f"{whisper_prob:.2f}" if whisper_prob is not None else "-",
                (user_text or "")[:80],
            )
        return self.preferred_lang

    def _adapt_caller_gender(self) -> bool:
        return (
            self.gender_adapt_when_confident
            or self.gender_mirror_grammar
            or self.gender_mirror_caller
        )

    def update_caller_gender_from_pitch(
        self,
        pitch_hz: float | None,
        *,
        female_min_hz: float,
        male_max_hz: float,
        min_samples: int = 2,
        min_confident_samples: int = 4,
        confident_fraction: float = 0.75,
    ) -> str | None:
        if not self.gender_detect_caller:
            return None
        from ari_app.gender import merge_gender_estimate

        prev = self.caller_gender
        prev_conf = self.caller_gender_confident
        (
            self.caller_gender,
            self.pitch_hz_history,
            self.caller_gender_confident,
        ) = merge_gender_estimate(
            self.caller_gender,
            pitch_hz,
            female_min_hz=female_min_hz,
            male_max_hz=male_max_hz,
            min_samples=min_samples,
            min_confident_samples=min_confident_samples,
            confident_fraction=confident_fraction,
            pitch_history=self.pitch_hz_history,
            lock_after_set=True,
        )
        if self.caller_gender_confident and (
            self.caller_gender != prev or not prev_conf
        ):
            log.info(
                "Session %s caller gender confident -> %s (pitch=%.0fHz history=%s)",
                self.session_id,
                self.caller_gender,
                pitch_hz or 0,
                [round(p) for p in self.pitch_hz_history],
            )
            if (
                self.gender_grammar_until_confident
                and not prev_conf
                and self.turn_count < self.gender_grammar_min_turns
            ):
                log.info(
                    "Session %s caller pitch confident -> %s; persona grammar still "
                    "neutral until user turn %s/%s",
                    self.session_id,
                    self.caller_gender,
                    self.turn_count,
                    self.gender_grammar_min_turns,
                )
        elif pitch_hz and not self.caller_gender_confident:
            log.debug(
                "Session %s pitch=%.0fHz history=%s (gender not confident yet)",
                self.session_id,
                pitch_hz,
                [round(p) for p in self.pitch_hz_history[-6:]],
            )
        return self.caller_gender

    def log_persona_grammar_unlock_if_needed(self) -> None:
        if self._logged_grammar_unlock or not self.gender_grammar_until_confident:
            return
        if not self.gender_grammar_ready():
            return
        self._logged_grammar_unlock = True
        log.info(
            "Session %s conversation mode ready (grammar=%s voice=%s)",
            self.session_id,
            self.resolved_grammar_gender(),
            self.resolved_voice_gender(),
        )

    def resolved_voice_gender(self) -> str:
        """TTS voice — fixed F3/M1 unless mirror + confident caller gender."""
        if (
            self.gender_mirror_caller
            and self.caller_gender_confident
            and self.caller_gender in ("male", "female")
        ):
            return self.caller_gender
        return self.agent_voice_gender

    def _bot_grammar_from_agent(self) -> str | None:
        """Bot persona (करती/करता) from AGENT_GENDER or TTS voice — not caller detection."""
        if self.hindi_grammar_use_agent_persona and self.agent_gender in (
            "male",
            "female",
        ):
            return self.agent_gender
        if self.grammar_match_voice and self.agent_voice_gender in ("male", "female"):
            return self.agent_voice_gender
        return None

    def resolved_grammar_gender(self) -> str:
        """
        Hindi bot self-reference (मैं करती/करता) for LLM + align_hindi_bot_grammar.

        Agent/voice persona wins over caller pitch/model when
        HINDI_GRAMMAR_USE_AGENT_PERSONA or GRAMMAR_MATCH_VOICE is set.
        Caller gender is passed separately via caller_grammar_hint() for आप forms.
        """
        bot = self._bot_grammar_from_agent()
        if bot:
            return bot

        if not self.gender_detect_caller:
            return "neutral"

        adapt = self.gender_adapt_when_confident or self.gender_mirror_grammar
        if adapt and self.caller_gender in ("male", "female"):
            if self.caller_gender_from_model and self.caller_gender_model_done:
                return self.caller_gender
            if self.caller_gender_confident:
                return self.caller_gender

        return "neutral"

    def grammar_is_locked_neutral(self) -> bool:
        return self.gender_grammar_until_confident and not self.gender_grammar_ready()

    def caller_grammar_hint(self) -> str | None:
        """
        Caller gender for respectful आप / addressing them — when bot persona differs.

        e.g. female-voice bot (करती) + male caller → still nudge male caller forms.
        """
        if not self.gender_detect_caller:
            return None
        if self.caller_gender not in ("male", "female"):
            return None
        if self.caller_gender_from_model and self.caller_gender_model_done:
            pass
        elif not self.caller_gender_confident:
            return None
        bot = self._bot_grammar_from_agent() or self.resolved_grammar_gender()
        if bot == self.caller_gender:
            return None
        return self.caller_gender

    def resolved_agent_gender(self) -> str:
        """Backward compat: voice gender for TTS helpers."""
        return self.resolved_voice_gender()

    def remember_web_context(
        self,
        *,
        topic: str | None,
        query: str | None,
        context: str | None,
    ) -> None:
        """Cache a successful web search so follow-up turns can reuse it."""
        if not context or not context.strip():
            return
        self.last_web_topic = (topic or "").strip().lower() or None
        self.last_web_query = (query or "").strip() or None
        self.last_web_context = context.strip()
        self.last_web_at = time.time()
        log.debug(
            "Session %s cached web context topic=%s query=%r chars=%d",
            self.session_id,
            self.last_web_topic or "-",
            (self.last_web_query or "")[:80],
            len(self.last_web_context),
        )

    def get_fresh_web_context(self, ttl_seconds: float = 180.0) -> str | None:
        """Return cached web context if still within TTL, else None."""
        if not self.last_web_context:
            return None
        if ttl_seconds <= 0:
            return self.last_web_context
        if time.time() - self.last_web_at > ttl_seconds:
            return None
        return self.last_web_context

    def clear_web_context(self) -> None:
        self.last_web_topic = None
        self.last_web_query = None
        self.last_web_context = None
        self.last_web_at = 0.0

    def add_assistant(self, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        self.messages.append({"role": "assistant", "content": t})
        self._trim()

    def add_user(self, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        self.messages.append({"role": "user", "content": t})
        self._trim()

    def _trim(self) -> None:
        if len(self.messages) > self.max_messages:
            drop = len(self.messages) - self.max_messages
            self.messages = self.messages[drop:]

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "user")

    def history_for_llm(self) -> list[dict[str, str]]:
        """Copy of prior turns (current user utterance is added by the LLM layer)."""
        return list(self.messages)

    def caller_context_line(self) -> str:
        parts: list[str] = []
        if self.caller_name and self.caller_name not in ("-", "unknown"):
            parts.append(f"name {self.caller_name}")
        if self.caller_number:
            parts.append(f"id {self.caller_number}")
        if self.dialed_number:
            parts.append(f"dialed {self.dialed_number}")
        return ", ".join(parts)

    def summary_for_log(self) -> str:
        return (
            f"session={self.session_id} channel={self.channel_id} "
            f"turns={self.turn_count} messages={len(self.messages)}"
        )

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "channel_id": self.channel_id,
            "caller_number": self.caller_number,
            "caller_name": self.caller_name,
            "dialed_number": self.dialed_number,
            "turn_count": self.turn_count,
            "message_count": len(self.messages),
            "messages": list(self.messages),
        }


class ActiveCallRegistry:
    """Tracks all in-progress call sessions (keyed by Asterisk channel id)."""

    def __init__(self) -> None:
        self._by_channel: dict[str, CallContext] = {}

    def register(self, ctx: CallContext) -> None:
        self._by_channel[ctx.channel_id] = ctx
        log.info(
            "Call session started %s caller=%s",
            ctx.summary_for_log(),
            ctx.caller_context_line() or "-",
        )

    def get(self, channel_id: str) -> CallContext | None:
        return self._by_channel.get(channel_id)

    def unregister(self, channel_id: str) -> CallContext | None:
        ctx = self._by_channel.pop(channel_id, None)
        if ctx:
            log.info(
                "Call session ended %s duration=%.0fs",
                ctx.summary_for_log(),
                time.time() - ctx.started_at,
            )
        return ctx

    def active_count(self) -> int:
        return len(self._by_channel)

    def list_active(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": c.session_id,
                "channel_id": c.channel_id,
                "caller_number": c.caller_number,
                "turn_count": c.turn_count,
                "message_count": len(c.messages),
            }
            for c in self._by_channel.values()
        ]
