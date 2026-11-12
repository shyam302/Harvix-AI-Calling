"""
Voice chat in the browser: the same conversation as a phone call, without Asterisk.

``WebChatSession`` is a ``CallSession`` whose "phone line" is a WebSocket:

    browser mic --(16 kHz PCM, binary frames)--> LiveVad (Silero)   = caller audio
    browser    <--(WAV, binary frames)---------- say_text / TTS     = bot audio
    on-screen keypad / typed text ----------------> DTMF / turns

Everything else is inherited unchanged from the phone bot: turn loop, STT, LLM,
knowledge base, web search, PIN check, registration, complaints, ticket status,
end-of-call, database logging. Only the methods that talked to Asterisk (answer,
play, live-VAD setup, recording, hang-up) are replaced here.

WebSocket protocol (/ws/chat?phone=<optional caller id>):
  browser -> server  binary: PCM16LE mono 16 kHz
                     text JSON: {"type":"text","text":...}   typed message
                                {"type":"dtmf","digit":"1"}  keypad
                                {"type":"played","id":N}     audio N finished
                                {"type":"hangup"}
  server -> browser  text JSON: {"type":"bot","text":...}  {"type":"you","text":...}
                                {"type":"audio","id":N} followed by one binary WAV
                                {"type":"stop"} (caller interrupted)  {"type":"end"}
                                {"type":"state","state":"listening|thinking|speaking"}
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import numpy as np

from ari_app import live_vad
from ari_app.call_session import CallSession
from ari_app.config import Settings

log = logging.getLogger(__name__)

Sender = Callable[[Any], Awaitable[None]]  # dict -> JSON text frame, bytes -> binary frame


class WebChatSession(CallSession):
    def __init__(
        self,
        *,
        settings: Settings,
        send: Sender,
        client: httpx.AsyncClient,
        caller_number: str | None = None,
    ) -> None:
        sounds = Path(tempfile.gettempdir()) / "webchat-sounds"
        super().__init__(
            settings=settings,
            channel_id=f"web-{uuid.uuid4().hex[:12]}",
            client=client,
            ari_base="http://127.0.0.1:1/ari",  # never used: all ARI calls are overridden
            auth=httpx.BasicAuth("-", "-"),
            sounds_en_dir=sounds,
            caller_number=(caller_number or "").strip() or None,
            dialed_number="web-chat",
        )
        self._send = send
        self._typed_queue: asyncio.Queue[str] = asyncio.Queue()
        self._typed_pending: str | None = None
        self._audio_id = 0
        self._played: dict[int, asyncio.Event] = {}
        self._last_turn_typed = False  # the browser already shows typed messages
        # Voice is OFF until the user switches the mic on: text in, text out (no TTS,
        # no silence timeouts). On: mic audio is listened to and replies are spoken.
        self._voice = False             # mic: listen to the browser's audio
        self._speak = False             # speaker: send spoken replies to the browser
        self._voice_changed = asyncio.Event()
        self._resumed = False

    # ---------------------------------------------------------------- browser input
    def feed_audio(self, data: bytes) -> None:
        if self._vad is not None and len(data) >= 2:
            self._vad.feed_pcm(np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2"))

    def resume(self, history: list) -> None:
        """Earlier messages of this browser tab's conversation, so the bot remembers them."""
        msgs = []
        for m in history[-20:]:
            if not isinstance(m, dict):
                continue
            role = {"you": "user", "bot": "assistant"}.get(m.get("role"))
            text = str(m.get("text") or "").strip()[:600]
            if role and text:
                msgs.append({"role": role, "content": text})
        if msgs:
            # straight into memory: these lines are already in the database from before
            self.ctx.messages.extend(msgs)
            self.ctx.messages = self.ctx.messages[-self.ctx.max_messages :]
            self._resumed = True
            log.info("[%s] web chat resumed with %s earlier messages", self.ctx.session_id, len(msgs))

    def set_voice(self, on: bool) -> None:
        if bool(on) != self._voice:
            self._voice = bool(on)
            log.info("[%s] web chat voice %s", self.ctx.session_id, "on" if on else "off")
            if not on and self._vad is not None:
                self._vad.disarm()
            self._voice_changed.set()

    def set_speaker(self, on: bool) -> None:
        self._speak = bool(on)
        log.info("[%s] web chat speaker %s", self.ctx.session_id, "on" if on else "off")

    def add_typed(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._typed_queue.put_nowait(text[:500])

    def audio_played(self, audio_id: int) -> None:
        ev = self._played.get(audio_id)
        if ev:
            ev.set()

    def _take_typed_text(self) -> str | None:
        text, self._typed_pending = self._typed_pending, None
        return text

    async def _emit(self, msg: Any) -> None:
        try:
            await self._send(msg)
        except Exception:  # browser gone
            self.mark_channel_dead()

    # ---------------------------------------------------------------- "phone line"
    async def _post(self, path: str, **kwargs: Any) -> httpx.Response:  # pragma: no cover
        raise RuntimeError(f"web chat has no Asterisk (POST {path})")

    async def _delete(self, path: str) -> None:
        return None

    async def answer(self) -> None:
        await self._emit({"type": "state", "state": "connected"})

    async def _channel_exists(self) -> bool:
        return not self._closed

    async def _setup_live_vad(self) -> None:
        s = self.settings
        self._vad = live_vad.LiveVad(
            end_silence_ms=s.vad_end_silence_ms,
            speech_threshold=s.vad_speech_threshold,
            min_speech_ms=s.vad_min_speech_ms,
            no_speech_timeout_s=s.vad_no_speech_timeout_s,
            barge_threshold=s.barge_in_threshold,
            barge_min_speech_ms=s.barge_in_min_speech_ms,
        )

    async def _teardown_live_vad(self) -> None:
        if self._vad is not None:
            self._vad.disarm()

    async def start_recording(self) -> str | None:
        return None  # no Asterisk recording fallback in the browser

    async def _start_call_recording(self) -> None:
        return None  # web chats keep a transcript (call_messages), no audio file

    async def _finish_call_recording(self) -> None:
        return None

    async def hangup(self) -> None:
        if self._hangup_done:
            return
        self._hangup_done = True
        self._closed = True
        await self._emit({"type": "end"})

    async def _identify_caller(self) -> str:
        opening = await super()._identify_caller()
        if self._resumed and self.ctx.user_id is None:
            # Continuing an earlier chat (no PIN step): no full greeting again.
            return "I'm still here. Go ahead." if self._reg_lang() != "hi" else "जी, बताइए।"
        return opening

    # ---------------------------------------------------------------- listening
    async def _capture_turn_vad(self, *, fresh: bool = False):
        """Wait for speech (live VAD) or a typed message, whichever comes first."""
        await self._emit({"type": "state", "state": "listening" if self._voice else "typing"})
        if not self._typed_queue.empty():
            return self._typed_turn(self._typed_queue.get_nowait())
        while not self._voice and not self._closed:
            # Text-only: wait for a typed message (no "no speech" timeouts), or for the
            # user to switch the mic on.
            self._voice_changed.clear()
            typed = asyncio.create_task(self._typed_queue.get())
            switched = asyncio.create_task(self._voice_changed.wait())
            try:
                done, _ = await asyncio.wait({typed, switched}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for t in (typed, switched):
                    if not t.done():
                        t.cancel()
            if typed in done and not typed.cancelled():
                return self._typed_turn(typed.result())
            await asyncio.gather(typed, return_exceptions=True)
        if self._closed:
            return None, 0.0
        await self._emit({"type": "state", "state": "listening"})
        voice = asyncio.create_task(super()._capture_turn_vad(fresh=fresh))
        typed = asyncio.create_task(self._typed_queue.get())
        try:
            done, _ = await asyncio.wait({voice, typed}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # Our caller gave up (e.g. the keypad answered a PIN first). Don't leave a
            # queue reader behind: it would swallow the next typed message.
            for t in (voice, typed):
                t.cancel()
            await asyncio.gather(voice, typed, return_exceptions=True)
            raise
        if typed in done:
            voice.cancel()
            await asyncio.gather(voice, return_exceptions=True)
            if self._vad is not None:
                self._vad.arm("barge" if self.settings.barge_in_enabled else "off")
            return self._typed_turn(typed.result())
        typed.cancel()
        self._last_turn_typed = False
        result = voice.result()
        if result is None:  # base class: "no audio at all" -> keep listening (mic may be off)
            if self._vad is not None:
                self._vad.disarm()
            return None, 0.0
        await self._emit({"type": "state", "state": "thinking"})
        return result

    def _typed_turn(self, text: str) -> tuple[Path, float]:
        """A typed message becomes a 1 s silent 'utterance' whose transcript is the text."""
        self._typed_pending = text
        self._last_turn_typed = True
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(live_vad.SAMPLE_RATE)
            wf.writeframes(np.zeros(live_vad.SAMPLE_RATE, dtype="<i2").tobytes())
        return path, 1.0

    async def _transcribe_turn(self, wav_path: Path):
        result = await super()._transcribe_turn(wav_path)
        if result[0] and not self._last_turn_typed:
            await self._emit({"type": "you", "text": result[0]})
        return result

    async def _hear_once(self, *, stt_prompt: str | None = None, answers=None):
        text = await super()._hear_once(stt_prompt=stt_prompt, answers=answers)
        if text and not self._last_turn_typed:
            shown = re.sub(r"\d", "•", text) if stt_prompt else text  # PIN / numbers
            await self._emit({"type": "you", "text": shown})
        return text

    # ---------------------------------------------------------------- speaking
    async def say_text(self, text: str, **kwargs: Any) -> str:
        # Phone wording -> chat wording ("end the call" -> "end the chat")
        for a, b in (("end the call", "end the chat"), ("for calling", "for chatting"),
                     ("on this call", "in this chat")):
            text = (text or "").replace(a, b)
        if text and text.strip() and not self._closed:
            await self._emit({"type": "bot", "text": text.strip()})
            if self._speak:
                await self._emit({"type": "state", "state": "speaking"})
        return await super().say_text(text, **kwargs)

    async def _synthesize_phrase(self, phrase: str, out_wav: Path, **kwargs: Any) -> None:
        if self._speak:  # speaker off: no speech to make
            await super()._synthesize_phrase(phrase, out_wav, **kwargs)

    async def _play_wav(self, out_wav: Path) -> bool:
        """Send one TTS chunk to the browser; True if the caller interrupted it."""
        if self._closed:
            return True
        if not self._speak or not out_wav.is_file():
            return False
        hd = out_wav.with_suffix(".wav16")  # 16 kHz twin sounds better than 8 kHz phone audio
        path = hd if hd.is_file() else out_wav
        try:
            data = path.read_bytes()
            with wave.open(str(path), "rb") as wf:
                seconds = wf.getnframes() / float(wf.getframerate() or 8000)
        except (OSError, wave.Error):
            return False
        self._audio_id += 1
        aid = self._audio_id
        done = asyncio.Event()
        self._played[aid] = done
        await self._emit({"type": "audio", "id": aid})
        await self._emit(data)
        vad = self._barge_vad()
        waiters = {asyncio.ensure_future(done.wait())}
        barge = None
        if vad is not None:
            barge = asyncio.ensure_future(vad.speech_started.wait())
            waiters.add(barge)
        try:
            finished, _ = await asyncio.wait(
                waiters, timeout=seconds + 8.0, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for w in waiters:
                w.cancel()
            self._played.pop(aid, None)
        if barge is not None and barge in finished and not done.is_set():
            log.info("[%s] barge-in: stopping browser playback", self.ctx.session_id)
            await self._emit({"type": "stop"})
            return True
        return self._closed
