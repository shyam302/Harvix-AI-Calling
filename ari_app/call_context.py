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
    preferred_lang: str = "hi"  # hi | en — sticky for LLM + Supertonic TTS
    consecutive_en_signals: int = 0  # hysteresis before hi→en switch
    caller_gender: str | None = None  # male | female — from pitch on caller audio
    pitch_hz_history: list[float] = field(default_factory=list)

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
    ) -> CallContext:
        pl = primary_lang if primary_lang in ("hi", "en") else "hi"
        return cls(
            session_id=uuid.uuid4().hex[:12],
            channel_id=channel_id,
            caller_number=caller_number,
            caller_name=caller_name,
            dialed_number=dialed_number,
            max_messages=max(4, max_messages),
            preferred_lang=pl,
        )

    def set_lang_from_user(
        self,
        user_text: str,
        *,
        whisper_lang: str | None = None,
        primary_lang: str = "hi",
        min_en_words: int = 4,
        consecutive_en_required: int = 2,
    ) -> str:
        from ari_app.lang import update_preferred_lang

        prev = self.preferred_lang
        self.preferred_lang, self.consecutive_en_signals = update_preferred_lang(
            self.preferred_lang,
            user_text,
            whisper_lang=whisper_lang,
            primary_lang=primary_lang,
            consecutive_en_signals=self.consecutive_en_signals,
            min_en_words=min_en_words,
            consecutive_en_required=consecutive_en_required,
        )
        if self.preferred_lang != prev:
            log.info(
                "Session %s language %s -> %s (en_signals=%s)",
                self.session_id,
                prev,
                self.preferred_lang,
                self.consecutive_en_signals,
            )
        return self.preferred_lang

    def update_caller_gender_from_pitch(
        self,
        pitch_hz: float | None,
        *,
        female_min_hz: float,
        male_max_hz: float,
        min_samples: int = 2,
    ) -> str | None:
        from ari_app.gender import merge_gender_estimate

        prev = self.caller_gender
        self.caller_gender, self.pitch_hz_history = merge_gender_estimate(
            self.caller_gender,
            pitch_hz,
            female_min_hz=female_min_hz,
            male_max_hz=male_max_hz,
            min_samples=min_samples,
            pitch_history=self.pitch_hz_history,
        )
        if self.caller_gender and self.caller_gender != prev:
            log.info(
                "Session %s caller gender -> %s (pitch=%.0fHz history=%s)",
                self.session_id,
                self.caller_gender,
                pitch_hz or 0,
                [round(p) for p in self.pitch_hz_history],
            )
        return self.caller_gender

    def agent_gender(self, *, default: str = "female") -> str:
        """Mirror caller gender for TTS/LLM; until detected use default bot gender."""
        return self.caller_gender or default

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
