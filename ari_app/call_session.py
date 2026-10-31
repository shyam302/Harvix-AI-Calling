"""Per-call session: record -> STT -> LLM -> TTS -> play (loop until hangup)."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from openai import APIConnectionError, APITimeoutError

from ari_app import (
    conversation,
    gender,
    gender_model,
    lang,
    live_vad,
    llm,
    stt,
    tts,
    voice_select,
    web_search,
)
from ari_app.call_context import CallContext
from ari_app.config import Settings

log = logging.getLogger(__name__)

# Whisper on near-silent RTP often emits a word right after the bot speaks.
_ALWAYS_DROP_SILENCE_JUNK = frozenset(
    {
        "by",
        "buy",
        "bai",
        "uh",
        "um",
        "hmm",
        "hm",
        "mm",
        "mhm",
        "mmm",
        "oh",
        "ah",
        "ha",
        "huh",
        "eh",
        "er",
    }
)
# Same hallucinations on silence — but real callers say these mid/end of call.
_EARLY_ONLY_SILENCE_JUNK = frozenset(
    {
        "bye",
        "thank you",
        "thanks",
        "okay",
        "ok",
        "yeah",
        "yep",
        "hey",
    }
)


def _log_turn_latency_breakdown(
    session_id: str,
    turn: int,
    *,
    pause_before_response_s: float,
    stt_s: float,
    web_s: float,
    llm_s: float,
    tts_synth_s: float,
    tts_play_s: float,
    reply_chars: int,
    record_duration_s: float,
    record_silence_s: float,
) -> None:
    """One-line diagnosis of where time went after the user stopped talking."""
    after_stop = (
        pause_before_response_s + stt_s + web_s + llm_s + tts_synth_s + tts_play_s
    )
    if after_stop <= 0:
        return

    def pct(x: float) -> float:
        return 100.0 * x / after_stop

    parts = [
        ("play", tts_play_s),
        ("synth", tts_synth_s),
        ("stt", stt_s),
        ("pause", pause_before_response_s),
        ("web", web_s),
        ("llm", llm_s),
    ]
    parts.sort(key=lambda p: p[1], reverse=True)
    top = parts[0][0]

    log.info(
        "[%s] turn=%s latency breakdown (after user stops talking): "
        "silence_end_wait~%.1fs + pause=%.2fs stt=%.2fs web=%.2fs "
        "llm=%.2fs tts_synth=%.2fs tts_play=%.2fs → total=%.2fs | reply=%s chars | "
        "recorded_audio=%.1fs",
        session_id,
        turn,
        record_silence_s,
        pause_before_response_s,
        stt_s,
        web_s,
        llm_s,
        tts_synth_s,
        tts_play_s,
        after_stop,
        reply_chars,
        record_duration_s,
    )
    log.info(
        "[%s] turn=%s share: play=%.0f%% synth=%.0f%% stt=%.0f%% web=%.0f%% llm=%.0f%% "
        "pause=%.0f%% | main bottleneck=%s",
        session_id,
        turn,
        pct(tts_play_s),
        pct(tts_synth_s),
        pct(stt_s),
        pct(web_s),
        pct(llm_s),
        pct(pause_before_response_s),
        top,
    )
    if tts_play_s >= 5.0 and reply_chars > 80:
        log.info(
            "[%s] hint: tts_play is how long Asterisk plays the reply (~%.0f%%). "
            "Use REPLY_MAX_SENTENCES=1 REPLY_MAX_CHARS=100–120 for faster turns.",
            session_id,
            pct(tts_play_s),
        )
    if tts_synth_s >= 2.5:
        log.info(
            "[%s] hint: tts_synth is Supertonic+ffmpeg; concurrent calls share GPU slots "
            "(WHISPER_MAX_CONCURRENT / TTS_MAX_CONCURRENT).",
            session_id,
        )


def _drop_likely_silence_hallucination(
    text: str,
    *,
    user_turn_count: int = 0,
    record_duration_s: float | None = None,
) -> str:
    """
    Drop Whisper junk on silence.

    Short fillers are always dropped. ``thank you`` / ``bye`` are only dropped on
    the first user turn (or very short clips) so end-of-call thanks still get a reply.
    """
    stripped = (text or "").strip()
    if not stripped:
        return ""
    key = stripped.lower().strip(".!?…").replace("'", "")
    if len(key) > 16:
        return stripped

    if key in _ALWAYS_DROP_SILENCE_JUNK:
        log.info("Dropping likely silence/noise transcript %r", stripped)
        return ""

    if key in _EARLY_ONLY_SILENCE_JUNK:
        long_enough = record_duration_s is not None and record_duration_s >= 2.0
        if user_turn_count >= 1 or long_enough:
            return stripped
        log.info(
            "Dropping likely silence/noise transcript %r (first turn, %.1fs)",
            stripped,
            record_duration_s or 0.0,
        )
        return ""

    return stripped


def channel_id_from_target_uri(target_uri: str) -> str | None:
    if target_uri.startswith("channel:"):
        return target_uri.split(":", 1)[1]
    return None


def stasis_channel_peer_info(ch: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """From StasisStart `channel` object: caller number, caller name, dialed extension."""
    caller = ch.get("caller") or {}
    raw_num = (caller.get("number") or "").strip()
    raw_name = (caller.get("name") or "").strip()
    caller_number = raw_num or None
    caller_name = raw_name or None
    dp = ch.get("dialplan") or {}
    exten = (dp.get("exten") or "").strip()
    dialed = exten or None
    return caller_number, caller_name, dialed


class CallSession:
    def __init__(
        self,
        *,
        settings: Settings,
        channel_id: str,
        client: httpx.AsyncClient,
        ari_base: str,
        auth: httpx.Auth,
        sounds_en_dir: Path,
        caller_number: str | None = None,
        caller_name: str | None = None,
        dialed_number: str | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        self.settings = settings
        self.channel_id = channel_id
        self.caller_number = caller_number
        self.caller_name = caller_name
        self.dialed_number = dialed_number
        self.ctx = ctx or CallContext.create(
            channel_id=channel_id,
            caller_number=caller_number,
            caller_name=caller_name,
            dialed_number=dialed_number,
            max_messages=settings.session_max_messages,
            primary_lang=settings.call_primary_lang,
            gender_detect_caller=settings.gender_detect_caller,
            gender_adapt_when_confident=settings.gender_adapt_when_confident
            or settings.gender_mirror_grammar,
            gender_mirror_caller=settings.gender_mirror_caller,
            gender_mirror_grammar=settings.gender_mirror_grammar,
            gender_min_confident_samples=settings.gender_min_confident_samples,
            gender_confident_fraction=settings.gender_confident_fraction,
            agent_gender=settings.agent_gender,
            agent_voice_gender=settings.agent_voice_gender,
            grammar_match_voice=settings.grammar_match_voice,
            hindi_grammar_use_agent_persona=settings.hindi_grammar_use_agent_persona,
            gender_grammar_until_confident=settings.gender_grammar_until_confident,
            gender_grammar_min_turns=settings.gender_grammar_min_turns,
        )
        self.client = client
        self.ari_base = ari_base.rstrip("/")
        self.auth = auth
        self.sounds_en_dir = sounds_en_dir
        self._closed = False
        self._hangup_done = False
        self._record_fut: asyncio.Future[dict] | None = None
        self._pending_record_name: str | None = None
        self._play_futs: dict[str, asyncio.Future[None]] = {}
        self._last_tts_synth_s: float = 0.0
        self._last_tts_play_s: float = 0.0
        self._consecutive_empty_listens: int = 0
        self._record_format: str = settings.record_format or "wav"
        self._gender_task: asyncio.Task[None] | None = None
        self._failed_queries: dict[str, float] = {}
        self._vad: live_vad.LiveVad | None = None
        self._vad_channels: list[str] = []
        self._vad_bridge: str | None = None

    def _is_likely_line_noise_clip(
        self,
        *,
        ari_duration_s: float,
        wav_duration_s: float | None,
    ) -> bool:
        """
        Asterisk often reports ~1s while the WAV is ~0.2s (echo/line noise right
        after bot speech) — not the caller talking.
        """
        if wav_duration_s is None:
            return False
        if wav_duration_s >= self.settings.record_min_duration_seconds:
            return False
        return ari_duration_s >= 0.8 and wav_duration_s < 0.5

    def _channel_path(self) -> str:
        # Channel id/name may contain "/" (e.g. Local/...); must be one URL path segment.
        seg = quote(self.channel_id, safe=":@.+-_~")
        return f"/channels/{seg}"

    async def _post(self, path: str, **kwargs: Any) -> httpx.Response:
        r = await self.client.post(f"{self.ari_base}{path}", auth=self.auth, **kwargs)
        if r.status_code >= 400:
            log.error("ARI POST %s -> %s %s", path, r.status_code, r.text)
        r.raise_for_status()
        return r

    async def _delete(self, path: str) -> None:
        try:
            r = await self.client.delete(f"{self.ari_base}{path}", auth=self.auth)
            if r.status_code not in (404, 200, 204):
                log.debug("ARI DELETE %s -> %s", path, r.status_code)
        except httpx.HTTPError:
            log.debug("ARI DELETE failed for %s", path)

    def mark_channel_dead(self) -> None:
        """Remote party hung up; cancel in-flight ARI waits."""
        self._closed = True
        if self._vad is not None:
            self._vad.ended.set()
            self._vad.speech_started.set()
        if self._record_fut and not self._record_fut.done():
            self._record_fut.cancel()
        for fut in list(self._play_futs.values()):
            if not fut.done():
                fut.cancel()
        self._play_futs.clear()

    async def hangup(self) -> None:
        if self._hangup_done:
            return
        self._hangup_done = True
        self._closed = True
        try:
            await self._delete(self._channel_path())
        except RuntimeError:
            pass  # httpx client already closed on server shutdown

    def arm_recording_wait(self, name: str) -> None:
        loop = asyncio.get_running_loop()
        self._pending_record_name = name
        self._record_fut = loop.create_future()

    def on_recording_finished(self, recording: dict) -> None:
        if not self._record_fut or self._record_fut.done():
            return
        if recording.get("name") != self._pending_record_name:
            return
        if not self._record_fut.done():
            self._record_fut.set_result(recording)

    def on_recording_failed(self, recording: dict) -> None:
        if not self._record_fut or self._record_fut.done():
            return
        if recording.get("name") != self._pending_record_name:
            return
        if not self._record_fut.done():
            self._record_fut.set_result(recording)

    def on_playback_finished(self, playback: dict) -> None:
        pb_id = playback.get("id")
        if not pb_id:
            return
        fut = self._play_futs.pop(pb_id, None)
        if fut and not fut.done():
            fut.set_result(None)

    async def _channel_exists(self) -> bool:
        """True if ARI still has this channel (GET /channels/{id})."""
        path = f"{self.ari_base}{self._channel_path()}"
        try:
            r = await self.client.get(path, auth=self.auth)
        except httpx.HTTPError:
            return False
        return r.status_code == 200

    async def answer(self) -> None:
        await self._post(f"{self._channel_path()}/answer")

    def _agent_voice_pair(self) -> tuple[str, str]:
        return voice_select.supertonic_voice_for_agent(
            self.settings, self.ctx.resolved_voice_gender()
        )

    def _continuous_mode(self) -> bool:
        return self.settings.continuous_conversation

    def _use_llm_streaming(self) -> bool:
        return self._continuous_mode() and self.settings.llm_streaming

    def _effective_record_silence_seconds(self) -> float:
        if self._continuous_mode():
            return self.settings.continuous_record_max_silence_seconds
        return self.settings.record_max_silence_seconds

    def _effective_tail_pad_seconds(self) -> float:
        if self._continuous_mode():
            return self.settings.continuous_tts_tail_pad_seconds
        return self.settings.tts_tail_pad_seconds

    def _clear_recording_wait(self) -> None:
        self._record_fut = None
        self._pending_record_name = None

    async def _prepare_to_listen(self) -> None:
        """Gap after bot speech + drain playback before opening the mic."""
        if self._closed:
            return
        if self._caller_barged_in():
            self._play_futs.clear()
            return
        drain_timeout = 0.35 if self._continuous_mode() else 2.0
        for fut in list(self._play_futs.values()):
            if not fut.done():
                try:
                    await asyncio.wait_for(asyncio.shield(fut), timeout=drain_timeout)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
        self._play_futs.clear()
        if self._continuous_mode():
            ms = (
                self.settings.continuous_pause_after_tts_ms
                + self.settings.continuous_pause_before_listen_ms
            )
        else:
            ms = self.settings.pause_after_tts_ms + self.settings.pause_before_listen_ms
        # Long TTS playback can still ring on the line — but keep this small.
        # Line-noise clips are now filtered by _is_likely_line_noise_clip without
        # counting toward the empty-streak, so we no longer need huge settles.
        if self._last_tts_play_s >= 6.0:
            ms += 250
        elif self._last_tts_play_s >= 3.0:
            ms += 120
        if ms > 0:
            log.debug(
                "[%s] pause %.0fms before listen (last_play=%.2fs)",
                self.ctx.session_id,
                ms,
                self._last_tts_play_s,
            )
            await asyncio.sleep(ms / 1000.0)

    async def _pause_before_response(self) -> None:
        """Brief gap after caller stops speaking, before STT/LLM (feels less abrupt)."""
        if self._continuous_mode():
            ms = self.settings.continuous_pause_before_response_ms
        else:
            ms = self.settings.pause_before_response_ms
        if ms > 0:
            await asyncio.sleep(ms / 1000.0)

    async def _setup_live_vad(self) -> None:
        """
        Snoop the caller's inbound audio into an externalMedia channel that streams
        RTP to a local UDP port, where Silero VAD detects the end of each utterance.
        Any failure leaves VAD off; Asterisk's maxSilenceSeconds still ends turns.
        """
        if not self.settings.vad_enabled:
            return
        tag = uuid.uuid4().hex[:12]
        snoop_id, media_id, bridge_id = f"cbsnoop-{tag}", f"cbmedia-{tag}", f"cbvad-{tag}"
        app = self.settings.stasis_app
        vad: live_vad.LiveVad | None = None
        try:
            vad = await live_vad.open_listener(
                end_silence_ms=self.settings.vad_end_silence_ms,
                speech_threshold=self.settings.vad_speech_threshold,
                no_speech_timeout_s=self.settings.vad_no_speech_timeout_s,
                barge_threshold=self.settings.barge_in_threshold,
                barge_min_speech_ms=self.settings.barge_in_min_speech_ms,
            )
            await self._post(
                f"{self._channel_path()}/snoop",
                params={"spy": "in", "app": app, "snoopId": snoop_id},
            )
            self._vad_channels.append(snoop_id)
            await self._post(
                "/channels/externalMedia",
                params={
                    "channelId": media_id,
                    "app": app,
                    "external_host": f"127.0.0.1:{vad.port}",
                    "format": "slin16",
                    "encapsulation": "rtp",
                    "transport": "udp",
                    "direction": "both",
                },
            )
            self._vad_channels.append(media_id)
            await self._post("/bridges", params={"type": "mixing", "bridgeId": bridge_id})
            self._vad_bridge = bridge_id
            await self._post(
                f"/bridges/{bridge_id}/addChannel",
                params={"channel": f"{snoop_id},{media_id}"},
            )
            self._vad = vad
            log.info(
                "[%s] live VAD on: caller audio -> udp 127.0.0.1:%s (end after %sms silence)",
                self.ctx.session_id,
                vad.port,
                self.settings.vad_end_silence_ms,
            )
        except Exception as exc:
            log.warning(
                "[%s] live VAD setup failed (%s); using Asterisk silence detection",
                self.ctx.session_id,
                exc,
            )
            if vad is not None:
                vad.close()
            await self._teardown_live_vad()

    async def _teardown_live_vad(self) -> None:
        if self._vad is not None:
            self._vad.close()
            self._vad = None
        for cid in self._vad_channels:
            await self._delete(f"/channels/{cid}")
        self._vad_channels.clear()
        if self._vad_bridge:
            await self._delete(f"/bridges/{self._vad_bridge}")
            self._vad_bridge = None

    def _caller_barged_in(self) -> bool:
        """Caller started talking while the bot was speaking or thinking."""
        vad = self._vad
        return vad is not None and vad.mode == "barge" and vad.speech_started.is_set()

    def _barge_vad(self) -> live_vad.LiveVad | None:
        """VAD armed to watch for barge-in during bot audio (None when disabled)."""
        vad = self._vad
        if vad is None or not self.settings.barge_in_enabled:
            return None
        if vad.mode == "off":
            vad.arm("barge")
        return vad if vad.mode == "barge" else None

    async def _capture_turn_vad(self) -> tuple[Path | None, float] | None:
        """
        Capture one caller utterance from the live stream.

        Returns (wav_path, seconds); wav_path is None when no speech was heard.
        Returns None if the stream carries no audio (VAD torn down; use ARI record).
        """
        vad = self._vad
        assert vad is not None
        if self._caller_barged_in():
            log.info("[%s] barge-in: caller talking over the bot, listening", self.ctx.session_id)
        vad.to_listen()
        try:
            await asyncio.wait_for(
                vad.ended.wait(), timeout=float(self.settings.record_max_duration_seconds)
            )
        except asyncio.TimeoutError:
            log.info("[%s] VAD: max turn length reached", self.ctx.session_id)
        if self._closed:
            vad.disarm()
            return None, 0.0

        heard = vad.heard_speech
        packets = vad.packets_armed
        pcm = vad.utterance_pcm() if heard else None
        # Keep watching while we think and speak, so the caller can cut in.
        if self.settings.barge_in_enabled:
            vad.arm("barge")
        else:
            vad.disarm()

        if not heard:
            if packets == 0:
                log.warning(
                    "[%s] no caller audio on the VAD stream; falling back to "
                    "Asterisk recording",
                    self.ctx.session_id,
                )
                await self._teardown_live_vad()
                return None
            return None, 0.0

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(live_vad.SAMPLE_RATE)
            wf.writeframes(pcm.astype("<i2").tobytes())
        seconds = len(pcm) / live_vad.SAMPLE_RATE
        log.info("[%s] VAD turn captured: %.2fs of caller audio", self.ctx.session_id, seconds)
        return path, seconds

    async def start_recording(self) -> str | None:
        name = f"cb-{uuid.uuid4().hex}"
        self.arm_recording_wait(name)
        silence = int(round(self._effective_record_silence_seconds()))
        silence = max(1, min(silence, 60))
        body = {
            "name": name,
            "format": self._record_format,
            "maxDurationSeconds": self.settings.record_max_duration_seconds,
            "maxSilenceSeconds": silence,
            "ifExists": "overwrite",
            "beep": False,
            "terminateOn": "none",
        }
        path = f"{self._channel_path()}/record"
        url = f"{self.ari_base}{path}"

        async def post_record() -> httpx.Response:
            return await self.client.post(url, auth=self.auth, json=body)

        try:
            r = await post_record()
            if r.status_code == 404 and await self._channel_exists():
                log.warning(
                    "ARI POST %s returned 404 but channel %s still exists; retry once after settle (%s)",
                    path,
                    self.channel_id,
                    (r.text or "").strip(),
                )
                await asyncio.sleep(0.4)
                r = await post_record()

            if r.status_code in (400, 422) and body["format"] != "wav":
                log.warning(
                    "ARI rejected record format %r (%s); falling back to wav",
                    body["format"],
                    (r.text or "").strip(),
                )
                self._record_format = "wav"
                body["format"] = "wav"
                r = await post_record()

            if r.status_code == 404:
                if self._record_fut and not self._record_fut.done():
                    self._record_fut.cancel()
                exists = await self._channel_exists()
                log.info(
                    "ARI record 404 for %s — channel_missing=%s; ending session.%s",
                    self.channel_id,
                    not exists,
                    " Typical: caller hung up, RTP timeout, or channel left Stasis."
                    if not exists
                    else " Recording failed while channel existed — see Asterisk full log.",
                )
                return None
            if r.status_code >= 400:
                log.error("ARI POST %s -> %s %s", path, r.status_code, r.text)
            r.raise_for_status()
        except Exception:
            if self._record_fut and not self._record_fut.done():
                self._record_fut.cancel()
            raise
        return name

    async def wait_recording(self) -> dict:
        if not self._record_fut:
            raise RuntimeError("Recording not armed")
        return await asyncio.wait_for(self._record_fut, timeout=120.0)

    async def fetch_recording_wav(self, recording_name: str) -> Path | None:
        url = f"{self.ari_base}/recordings/stored/{recording_name}/file"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            p = Path(tmp.name)
        ok = await stt.download_recording_file(
            url=url, dest=p, auth=self.auth, client=self.client
        )
        if not ok:
            p.unlink(missing_ok=True)
            return None
        return p

    async def delete_stored_recording(self, recording_name: str) -> None:
        await self._delete(f"/recordings/stored/{recording_name}")

    async def play_sound(self, media: str) -> bool:
        """Play media on the call. Returns True if the caller barged in and it was cut."""
        vad = self._barge_vad()
        if vad is not None and vad.speech_started.is_set():
            return True
        r = await self._post(f"{self._channel_path()}/play", json={"media": media})
        data = r.json()
        pb_id = data.get("id")
        if not pb_id:
            return False
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._play_futs[pb_id] = fut
        if vad is None:
            await asyncio.wait_for(fut, timeout=120.0)
            return False

        barge = asyncio.ensure_future(vad.speech_started.wait())
        try:
            done, _ = await asyncio.wait(
                {fut, barge}, timeout=120.0, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            barge.cancel()
        if fut in done:
            fut.result()  # raises CancelledError if the channel died
            return False
        if not done:
            raise asyncio.TimeoutError
        log.info("[%s] barge-in: stopping bot playback", self.ctx.session_id)
        self._play_futs.pop(pb_id, None)
        await self._delete(f"/playbacks/{pb_id}")
        return True

    def _tts_out_dir(self) -> Path:
        sub = self.settings.tts_sound_subdir.strip("/ ")
        out_dir = self.sounds_en_dir / sub
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            log.error(
                "Permission denied creating TTS directory %s. "
                "Run once (lab): sudo mkdir -p %s && sudo chmod 1777 %s",
                out_dir,
                out_dir,
                out_dir,
            )
            raise
        return out_dir

    async def _synthesize_phrase(
        self,
        phrase: str,
        out_wav: Path,
        *,
        session_lang: str,
        voice_en: str,
        voice_hi: str,
        speed_en: float,
        speed_hi: float,
        tail_pad_s: float = 0.0,
    ) -> None:
        await tts.synthesize_to_wav(
            phrase,
            out_wav,
            voice_en=voice_en,
            voice_hi=voice_hi,
            speed=self.settings.supertonic_speed,
            speed_en=speed_en,
            speed_hi=speed_hi,
            silence_duration=self.settings.supertonic_silence_duration,
            tail_pad_s=tail_pad_s,
            session_lang=session_lang,
            voice_name=voice_hi if session_lang == "hi" else voice_en,
            edge_voice_en=self.settings.edge_tts_voice_en,
            edge_voice_hi=self.settings.edge_tts_voice_hi,
            edge_rate=self.settings.edge_tts_rate,
        )

    async def _transcribe_turn(
        self, wav_path: Path
    ) -> tuple[str, str | None, float | None, float | None]:
        """
        STT and optional caller pitch in parallel (saves ~100–300ms per turn).

        Returns (text, whisper_lang, whisper_lang_probability, pitch_hz).
        """
        whisper_hint = lang.whisper_language_hint(
            self.ctx.preferred_lang,
            configured=self.settings.whisper_language,
        )
        stt_coro = stt.transcribe_wav(
            wav_path,
            self.settings.whisper_model,
            language=whisper_hint,
        )
        if not self.settings.gender_detect_caller:
            user_text, whisper_lang, whisper_prob = await stt_coro
            return user_text, whisper_lang, whisper_prob, None

        from ari_app.inference_pool import get_executor

        loop = asyncio.get_running_loop()
        pitch_coro = loop.run_in_executor(
            get_executor(), lambda: gender.estimate_pitch_hz(wav_path)
        )
        (user_text, whisper_lang, whisper_prob), pitch_hz = await asyncio.gather(
            stt_coro, pitch_coro
        )
        return user_text, whisper_lang, whisper_prob, pitch_hz

    async def _skip_turn_no_usable_speech(
        self,
        *,
        ari_duration_s: float,
        wav_duration_s: float | None = None,
        reason: str,
        count_streak: bool = True,
    ) -> bool:
        """
        Empty / too-short audio: optionally prompt, then listen again.

        Returns True when the session should end (too many consecutive empty turns).
        """
        max_empty = self.settings.record_max_consecutive_empty
        wav_part = (
            f" wav={wav_duration_s:.2f}s" if wav_duration_s is not None else ""
        )
        if count_streak:
            self._consecutive_empty_listens += 1
        action = (
            "listening again"
            if not self.settings.prompt_on_silence
            else "prompting caller"
        )
        if not count_streak:
            action = f"{action} (line noise, not counted)"
        log.info(
            "[%s] %s (ARI=%.2fs%s, empty_streak=%s/%s) — %s",
            self.ctx.session_id,
            reason,
            ari_duration_s,
            wav_part,
            self._consecutive_empty_listens,
            max_empty,
            action,
        )
        if not count_streak:
            return False
        if self._consecutive_empty_listens >= max_empty:
            log.info(
                "[%s] ending session after %s consecutive empty/silent listens",
                self.ctx.session_id,
                self._consecutive_empty_listens,
            )
            return True
        if self.settings.prompt_on_silence:
            await self.say_text(conversation.pick_retry_phrase(self.ctx.preferred_lang))
        return False

    def _start_gender_model_in_background(
        self,
        wav_path: Path,
        user_text: str,
        *,
        record_duration_s: float,
    ) -> None:
        """Classify caller gender off the reply path (it only refines later turns)."""
        if self.ctx.caller_gender_model_done or not self.settings.gender_model_enabled:
            return
        if self._gender_task is not None and not self._gender_task.done():
            return
        # The turn's WAV is deleted when the turn ends; classify a private copy.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            copy = Path(tmp.name)
        shutil.copyfile(wav_path, copy)

        async def run() -> None:
            try:
                await self._maybe_detect_caller_gender_model(
                    copy, user_text, record_duration_s=record_duration_s
                )
            except Exception:
                log.exception("[%s] gender model failed", self.ctx.session_id)
            finally:
                copy.unlink(missing_ok=True)

        self._gender_task = asyncio.create_task(run())

    async def _maybe_detect_caller_gender_model(
        self,
        wav_path: Path,
        user_text: str,
        *,
        record_duration_s: float,
    ) -> None:
        """
        Run Wav2Vec2 gender classifier once after greeting, on first long utterance.
        """
        if self.ctx.caller_gender_model_done:
            return
        if not self.settings.gender_model_enabled:
            return
        plain = (user_text or "").strip()
        if len(plain) < self.settings.gender_model_min_chars:
            log.debug(
                "[%s] gender model waiting: transcript %s chars < %s",
                self.ctx.session_id,
                len(plain),
                self.settings.gender_model_min_chars,
            )
            return
        if record_duration_s < self.settings.gender_model_min_duration_s:
            log.debug(
                "[%s] gender model waiting: recording %.2fs < %.2fs",
                self.ctx.session_id,
                record_duration_s,
                self.settings.gender_model_min_duration_s,
            )
            return

        from ari_app.inference_pool import get_executor

        if not gender_model.model_available():
            log.warning(
                "[%s] GENDER_MODEL_ENABLED but deps missing (transformers torch librosa)",
                self.ctx.session_id,
            )
            self.ctx.caller_gender_model_done = True
            return

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        result = await loop.run_in_executor(
            get_executor(),
            lambda: gender_model.classify_caller_gender(
                wav_path,
                min_confidence=self.settings.gender_model_min_confidence,
            ),
        )
        log.info(
            "[%s] gender model inference %.2fs",
            self.ctx.session_id,
            time.perf_counter() - t0,
        )
        self.ctx.caller_gender_model_done = True
        if result is None:
            return
        gender_label, p_female, p_male = result
        self.ctx.set_caller_gender_from_model(
            gender_label, p_female=p_female, p_male=p_male
        )

    async def _play_wav(self, out_wav: Path) -> bool:
        """Returns True if the caller barged in during playback."""
        sub = self.settings.tts_sound_subdir.strip("/ ")
        base = out_wav.name.replace(".wav", "")
        if await self.play_sound(f"sound:{sub}/{base}"):
            return True
        settle_ms = self.settings.tts_post_play_settle_ms
        if settle_ms > 0 and not self._closed:
            await asyncio.sleep(settle_ms / 1000.0)
        return False

    async def _speak_phrase_immediate(
        self,
        phrase: str,
        *,
        is_opening: bool = False,
        tail_pad_s: float | None = None,
    ) -> None:
        """Synthesize and play one short phrase (streaming / continuous mode)."""
        phrase = (phrase or "").strip()
        if not phrase or self._caller_barged_in():
            return
        pad = (
            self._effective_tail_pad_seconds()
            if tail_pad_s is None
            else tail_pad_s
        )
        voice_en, voice_hi = self._agent_voice_pair()
        if is_opening:
            speed_en = self.settings.supertonic_speed_greeting
            speed_hi = self.settings.supertonic_speed_greeting
        else:
            speed_en = self.settings.supertonic_speed_en
            speed_hi = self.settings.supertonic_speed_hi
        out_dir = self._tts_out_dir()
        wav_path = out_dir / f"tts-{uuid.uuid4().hex}.wav"
        await self._synthesize_phrase(
            phrase,
            wav_path,
            session_lang=self.ctx.preferred_lang,
            voice_en=voice_en,
            voice_hi=voice_hi,
            speed_en=speed_en,
            speed_hi=speed_hi,
            tail_pad_s=pad,
        )
        await self._play_wav(wav_path)
        tts.remove_tts_outputs(wav_path)

    # Words too common to act as a "city" follow-up after a weather question.
    _WEATHER_NOT_A_CITY = frozenset(
        {
            "yes",
            "no",
            "okay",
            "ok",
            "sure",
            "thanks",
            "thank you",
            "hello",
            "hi",
            "what",
            "where",
            "when",
            "why",
            "how",
            "now",
            "today",
            "tomorrow",
            "tonight",
        }
    )

    def _looks_like_city_name(self, text: str) -> str | None:
        """Caller probably just named a city / locality (for a weather follow-up)."""
        t = (text or "").strip().strip(".!?,")
        if not t:
            return None
        # Keep STT spelling but strip filler.
        t = re.sub(r"^(in|at|for|the)\s+", "", t, flags=re.IGNORECASE).strip()
        if not t:
            return None
        low = t.lower()
        if low in self._WEATHER_NOT_A_CITY:
            return None
        if len(t.split()) > 5:
            return None
        if not re.search(r"[A-Za-z\u0900-\u097F]", t):
            return None
        # Must look like a proper noun / place (skip "tell me", "the time", etc).
        if re.search(r"\b(time|date|score|news|tell|about|what|when|how|who)\b", low):
            return None
        return t

    def _is_followup_to_cached_web(
        self, user_text: str
    ) -> tuple[bool, str | None]:
        """
        Heuristic: caller is still on the previously-fetched topic.

        Returns ``(is_followup, refined_query)``.  The refined query is used to
        rerun a focused web search when the caller supplies missing context
        (e.g. names a city after the bot asked "which city?").
        """
        cached_topic = (self.ctx.last_web_topic or "").lower()
        if not cached_topic or not self.ctx.get_fresh_web_context(
            ttl_seconds=180.0
        ):
            return False, None
        # If the same topic detector matches the new text, that's a clear yes.
        # "general" is the detector's catch-all, so it never marks a follow-up.
        if (
            cached_topic != "general"
            and web_search.detect_search_topic(user_text or "") == cached_topic
        ):
            return True, None

        t = (user_text or "").lower()

        if cached_topic == "cricket":
            if len(t.split()) <= 6:
                followup_kw = (
                    r"\b(time|when|date|venue|where|score|result|next|tell|about|"
                    r"who|whom|whose|how)\b|"
                    r"(कब|कहाँ|कौन|कैसे|समय|तारीख|स्कोर|बताइए|बताओ|उसका|इसका)"
                )
                if re.search(followup_kw, t):
                    return True, None
                if web_search._CRICKET.search(t):
                    return True, None

        if cached_topic == "weather":
            # Caller is naming a city after "Which city are you asking about?".
            city = self._looks_like_city_name(user_text)
            if city:
                refined = f"weather today {city}"
                log.info(
                    "[%s] Weather follow-up city=%r -> refined query %r",
                    self.ctx.session_id,
                    city,
                    refined,
                )
                return True, refined
            # Generic "what's the temperature?", "अभी का मौसम?".
            if re.search(
                r"\b(weather|temperature|forecast|rain|humid)\b|"
                r"(मौसम|बारिश|तापमान|गर्मी|ठंड)",
                t,
            ):
                return True, None

        return False, None

    def _reply_kwargs(
        self, *, user_text: str, reply_lang: str, grammar: str
    ) -> dict[str, Any]:
        return dict(
            base_url=self.settings.vllm_base_url,
            api_key=self.settings.vllm_api_key,
            model=self.settings.vllm_model,
            user_text=user_text,
            history=self.ctx.history_for_llm(),
            reply_lang=reply_lang,
            agent_gender=grammar,
            caller_grammar_gender=self.ctx.caller_grammar_hint(),
            temperature=self.settings.llm_temperature,
            caller_number=self.ctx.caller_number,
            caller_name=self.ctx.caller_name,
            caller_context=self.ctx.caller_context_line() or None,
            max_history_messages=self.settings.session_max_messages,
            timeout_seconds=self.settings.vllm_timeout_seconds,
            connect_timeout_seconds=self.settings.vllm_connect_timeout_seconds,
            max_tokens=self.settings.llm_max_tokens,
        )

    async def _model_requested_lookup(self, user_text: str) -> str | None:
        """Search query when the model judges this turn needs facts it may not know."""
        if not self.settings.web_search_enabled:
            return None
        try:
            query = await llm.web_lookup_query(
                base_url=self.settings.vllm_base_url,
                api_key=self.settings.vllm_api_key,
                model=self.settings.vllm_model,
                user_text=user_text,
                history=self.ctx.history_for_llm(),
            )
        except Exception as exc:
            log.warning("[%s] web lookup check failed: %s", self.ctx.session_id, exc)
            return None
        if query:
            log.info("[%s] info not known -> web lookup %r", self.ctx.session_id, query)
        return query

    async def _search_for_model(self, query: str) -> tuple[str, float]:
        """Run the lookup; on failure tell the model not to guess or promise to check."""
        key = re.sub(r"\s+", " ", query.lower()).strip()
        failed_at = self._failed_queries.get(key)
        if failed_at is not None and time.monotonic() - failed_at < 120.0:
            log.info("[%s] skipping lookup %r (failed moments ago)", self.ctx.session_id, query)
            ctx_text, web_s = None, 0.0
        else:
            ctx_text, web_s, _ = await self._web_search(
                query, use_cache_on_failure=False, budget_s=3.5
            )
            if not ctx_text:
                self._failed_queries[key] = time.monotonic()
        if not ctx_text:
            ctx_text = (
                "\nIMPORTANT: you tried to look this up on the web just now and it did not "
                "work, so you do NOT have this information. In one short sentence, honestly "
                "tell the caller you couldn't find that right now, then offer something else "
                "you can help with. Do not say you will check, look into it, or get back to "
                "them, and do not guess numbers, names or dates."
            )
        return ctx_text, web_s

    async def _reply_with_lookup(
        self,
        *,
        user_text: str,
        reply_lang: str,
        grammar: str,
        web_context: str | None,
    ) -> tuple[str, str | None, float]:
        """
        Reply to the caller, searching the web first when the model says it needs to.

        The lookup check runs in parallel with a draft reply, so turns that need no
        search cost no extra time. Returns ``(reply, web_context, web_seconds)``.
        """
        kw = self._reply_kwargs(user_text=user_text, reply_lang=reply_lang, grammar=grammar)
        if web_context is not None or not self.settings.web_search_enabled:
            return await llm.reply_text(**kw, web_context=web_context), web_context, 0.0

        lookup = asyncio.create_task(self._model_requested_lookup(user_text))
        draft = asyncio.create_task(llm.reply_text(**kw, web_context=None))
        try:
            query = await lookup
        except BaseException:
            draft.cancel()
            raise
        if not query:
            return await draft, None, 0.0

        draft.cancel()
        await asyncio.gather(draft, return_exceptions=True)
        web_context, web_s = await self._search_for_model(query)
        return await llm.reply_text(**kw, web_context=web_context), web_context, web_s

    async def _web_search(
        self,
        search_text: str,
        *,
        topic_hint: str | None = None,
        use_cache_on_failure: bool = True,
        budget_s: float = 6.5,
    ) -> tuple[str | None, float, bool]:
        """Run one bounded web search. Returns ``(context, web_seconds, from_cache)``."""
        if not web_search.search_available():
            log.info("[%s] web search paused after recent failures", self.ctx.session_id)
            cached = (
                self.ctx.get_fresh_web_context(ttl_seconds=180.0)
                if use_cache_on_failure
                else None
            )
            return cached, 0.0, cached is not None
        t0 = time.perf_counter()
        # Keep phone latency bounded: web fetch should not block for too long.
        web_timeout = max(2.0, min(self.settings.web_search_timeout_seconds, budget_s))
        try:
            hits, query = await asyncio.wait_for(
                web_search.search_web(
                    search_text,
                    api_key=self.settings.google_cse_api_key,
                    cx=self.settings.google_cse_cx,
                    serpapi_api_key=self.settings.serpapi_api_key,
                    provider=self.settings.web_search_provider,
                    num_results=self.settings.web_search_num_results,
                    timeout_seconds=web_timeout,
                    crawl_max_pages=self.settings.crawl_max_pages,
                ),
                timeout=web_timeout,
            )
        except asyncio.TimeoutError:
            web_search.record_search_result(False)
            web_s = time.perf_counter() - t0
            log.warning(
                "[%s] Web search timed out in %.2fs; continuing without web context",
                self.ctx.session_id,
                web_s,
            )
            cached = (
                self.ctx.get_fresh_web_context(ttl_seconds=180.0)
                if use_cache_on_failure
                else None
            )
            if cached:
                return cached, web_s, True
            return None, web_s, False
        except Exception as exc:
            web_search.record_search_result(False)
            web_s = time.perf_counter() - t0
            log.warning(
                "[%s] Web search failed in %.2fs: %s; continuing without web context",
                self.ctx.session_id,
                web_s,
                exc,
            )
            cached = (
                self.ctx.get_fresh_web_context(ttl_seconds=180.0)
                if use_cache_on_failure
                else None
            )
            if cached:
                return cached, web_s, True
            return None, web_s, False

        web_s = time.perf_counter() - t0
        web_search.record_search_result(bool(hits))
        if not hits:
            log.info("[%s] Web search found nothing in %.2fs", self.ctx.session_id, web_s)
            return None, web_s, False
        topic = web_search.detect_search_topic(search_text) or topic_hint or "general"
        ctx_text = web_search.format_context_for_llm(hits, query=query, topic=topic)
        log.info(
            "[%s] Web search done: topic=%s hits=%s ctx=%schars in %.2fs",
            self.ctx.session_id,
            topic,
            len(hits),
            len(ctx_text),
            web_s,
        )
        self.ctx.remember_web_context(topic=topic, query=query, context=ctx_text)
        return ctx_text, web_s, False

    async def _maybe_fetch_web_context(
        self, user_text: str, reply_lang: str
    ) -> tuple[str | None, float, bool]:
        """
        Returns ``(context, web_seconds, from_cache)``.

        Decision order:
          1. If the previous turn cached the same topic recently (<30s) and the
             caller is still on it → reuse cache, skip re-fetch entirely.
          2. If the caller named a city after a weather question → refined fetch.
          3. If the turn triggers realtime search → fetch fresh and cache.
          4. If it's a follow-up to cached data → reuse cache.
          5. Otherwise → no web context.
        """
        if not self.settings.web_search_enabled:
            return None, 0.0, False

        cached_topic = (self.ctx.last_web_topic or "").lower()
        new_topic = web_search.detect_search_topic(user_text or "")
        needs_fresh = web_search.needs_realtime_search(user_text)
        is_followup, refined_query = self._is_followup_to_cached_web(user_text)

        # 1. Short-circuit: if we just fetched this topic and the cache is hot,
        # don't burn another 1–5s on identical data.
        same_topic_hot = (
            cached_topic
            and cached_topic != "general"
            and cached_topic == new_topic
            and refined_query is None  # don't short-circuit when caller added details
        )
        if same_topic_hot:
            cached_hot = self.ctx.get_fresh_web_context(ttl_seconds=30.0)
            if cached_hot:
                log.info(
                    "[%s] Web context reused (hot cache, topic=%s, age<30s)",
                    self.ctx.session_id,
                    cached_topic,
                )
                return cached_hot, 0.0, True

        # 2. Cached follow-up + refined query (e.g. "Which city?" → "Noida").
        # We still fetch, but with a focused query the caller just supplied.
        search_text = user_text
        if refined_query:
            search_text = refined_query
            needs_fresh = True
        elif is_followup and not needs_fresh:
            cached = self.ctx.get_fresh_web_context(ttl_seconds=180.0)
            if cached:
                log.info(
                    "[%s] Web context reused (topic=%s, age<180s)",
                    self.ctx.session_id,
                    cached_topic,
                )
                return cached, 0.0, True

        if not needs_fresh:
            return None, 0.0, False

        # Search silently: announcing every lookup gets repetitive on a call.
        return await self._web_search(search_text, topic_hint=new_topic)

    async def _reply_and_speak_streaming(
        self,
        *,
        user_text: str,
        reply_lang: str,
        grammar: str,
        web_context: str | None = None,
    ) -> tuple[str, str, float, float]:
        """Stream LLM sentences to TTS as they arrive.

        Returns ``(full_reply, spoken, llm_s, tts_s)``.
        """
        t_llm = time.perf_counter()
        t_tts = time.perf_counter()
        parts: list[str] = []
        max_sent = self.settings.reply_max_sentences
        llm_kw = dict(
            base_url=self.settings.vllm_base_url,
            api_key=self.settings.vllm_api_key,
            model=self.settings.vllm_model,
            user_text=user_text,
            history=self.ctx.history_for_llm(),
            reply_lang=reply_lang,
            agent_gender=grammar,
            caller_grammar_gender=self.ctx.caller_grammar_hint(),
            temperature=self.settings.llm_temperature,
            caller_number=self.ctx.caller_number,
            caller_name=self.ctx.caller_name,
            caller_context=self.ctx.caller_context_line() or None,
            web_context=web_context,
            max_history_messages=self.settings.session_max_messages,
            timeout_seconds=self.settings.vllm_timeout_seconds,
            connect_timeout_seconds=self.settings.vllm_connect_timeout_seconds,
            max_tokens=self.settings.llm_max_tokens,
            max_sentences=max_sent if max_sent > 0 else 0,
        )

        first_audio = True
        async for sent in llm.reply_text_stream(**llm_kw):
            if self._caller_barged_in():
                break
            sent = conversation.strip_robotic_phrases(
                conversation.strip_leading_filler(sent)
            )
            if not sent:
                continue
            if reply_lang == "hi":
                sent = conversation.align_hindi_bot_grammar(
                    sent, grammar_gender=grammar
                )
            parts.append(sent)
            if first_audio:
                log.info(
                    "[%s] streaming: first sentence to TTS (%.2fs after LLM start)",
                    self.ctx.session_id,
                    time.perf_counter() - t_llm,
                )
                first_audio = False
            await self._speak_phrase_immediate(sent, tail_pad_s=0.0)

        llm_s = time.perf_counter() - t_llm
        full_reply = " ".join(parts).strip()
        spoken = conversation.trim_reply_for_phone(
            full_reply,
            max_sentences=self.settings.reply_max_sentences,
            max_chars=self.settings.reply_max_chars,
        )
        tts_s = time.perf_counter() - t_tts
        log.info(
            "[%s] streaming reply sentences=%s llm+tts=%.2fs",
            self.ctx.session_id,
            len(parts),
            tts_s,
        )
        return full_reply, spoken, llm_s, tts_s

    async def say_text(
        self,
        text: str,
        *,
        record_in_session: bool = True,
        is_opening: bool = False,
    ) -> str:
        full_text = (text or "").strip()
        spoken = conversation.trim_reply_for_phone(
            full_text,
            max_sentences=self.settings.reply_max_sentences,
            max_chars=self.settings.reply_max_chars,
        )
        if spoken != full_text:
            log.info(
                "[%s] Reply trimmed for TTS: %r -> %r",
                self.ctx.session_id,
                full_text[:80],
                spoken[:80],
            )

        out_dir = self._tts_out_dir()
        chunks = conversation.split_tts_phrases(
            spoken,
            max_chars=self.settings.tts_chunk_max_chars,
        )
        if not chunks or self._caller_barged_in():
            return ""

        session_lang = self.ctx.preferred_lang
        voice_en, voice_hi = self._agent_voice_pair()
        tts_lang = tts.pick_supertonic_lang(spoken, session_lang=session_lang)
        voice = voice_hi if tts_lang == "hi" else voice_en
        if is_opening:
            speed_en = self.settings.supertonic_speed_greeting
            speed_hi = self.settings.supertonic_speed_greeting
        else:
            speed_en = self.settings.supertonic_speed_en
            speed_hi = self.settings.supertonic_speed_hi
        log.info(
            "[%s] TTS lang=%s voice=%s speed_en=%.2f speed_hi=%.2f %s chunks=%s chars=%s",
            self.ctx.session_id,
            tts_lang,
            voice,
            speed_en,
            speed_hi,
            "opening" if is_opening else "reply",
            len(chunks),
            len(spoken),
        )

        part_paths = [out_dir / f"tts-{uuid.uuid4().hex}.wav" for _ in chunks]
        tail_pad = self._effective_tail_pad_seconds()
        synth_kw = dict(
            session_lang=session_lang,
            voice_en=voice_en,
            voice_hi=voice_hi,
            speed_en=speed_en,
            speed_hi=speed_hi,
        )

        t_wall = time.perf_counter()
        synth_s = 0.0
        play_s = 0.0
        next_synth: asyncio.Task[None] | None = None
        n = len(chunks)
        played: list[str] = []

        for i in range(n):
            is_last = i == n - 1
            pad = tail_pad if is_last else 0.0
            if next_synth is not None:
                t0 = time.perf_counter()
                await next_synth
                synth_s += time.perf_counter() - t0
                next_synth = None
            else:
                t0 = time.perf_counter()
                await self._synthesize_phrase(
                    chunks[i], part_paths[i], tail_pad_s=pad, **synth_kw
                )
                synth_s += time.perf_counter() - t0

            if not is_last:
                next_synth = asyncio.create_task(
                    self._synthesize_phrase(
                        chunks[i + 1],
                        part_paths[i + 1],
                        tail_pad_s=0.0,
                        **synth_kw,
                    )
                )

            t1 = time.perf_counter()
            interrupted = await self._play_wav(part_paths[i])
            play_s += time.perf_counter() - t1
            tts.remove_tts_outputs(part_paths[i])
            played.append(chunks[i])
            if interrupted:
                break

        if next_synth is not None:
            # Caller cut in: drop the chunk synthesized ahead.
            await asyncio.gather(next_synth, return_exceptions=True)
        for path in part_paths[len(played) :]:
            tts.remove_tts_outputs(path)

        # History holds only what the caller actually heard ("…" if they cut in).
        spoken = " ".join(played)
        if len(played) < n or self._caller_barged_in():
            spoken += " …"
        if record_in_session and spoken:
            self.ctx.add_assistant(spoken)

        self._last_tts_synth_s = synth_s
        self._last_tts_play_s = play_s
        log.info(
            "[%s] TTS pipeline chunks=%s synth=%.2fs play=%.2fs wall=%.2fs",
            self.ctx.session_id,
            n,
            synth_s,
            play_s,
            time.perf_counter() - t_wall,
        )

        return spoken

    async def run_loop(self) -> None:
        try:
            log.info(
                "Call start %s channel=%s caller_number=%s caller_name=%s dialed=%s",
                self.ctx.summary_for_log(),
                self.channel_id,
                self.caller_number or "-",
                self.caller_name or "-",
                self.dialed_number or "-",
            )
            await self.answer()
            await self._setup_live_vad()
            opening = voice_select.opening_greeting_for_call(
                self.settings,
                self.ctx.preferred_lang,
                neutral_until_confident=self.ctx.gender_grammar_until_confident,
                caller_name=self.caller_name,
            )
            if lang.is_latin_english_content(opening, min_words=2):
                self.ctx.preferred_lang = "en"
            log.info(
                "[%s] conversation bot: neutral grammar, voice=%s; extra caution until "
                "%s user turns + confident pitch; continuous=%s stream=%s",
                self.ctx.session_id,
                self.ctx.resolved_voice_gender(),
                self.ctx.gender_grammar_min_turns,
                self._continuous_mode(),
                self._use_llm_streaming(),
            )
            # No language question: the bot answers in whatever language the caller uses.
            await self.say_text(opening, is_opening=True)

            while not self._closed:
                wav_path: Path | None = None
                try:
                    await self._prepare_to_listen()
                    if self._closed:
                        break
                    vad_turn = None
                    if self._vad is not None:
                        vad_turn = await self._capture_turn_vad()
                        if self._closed:
                            break
                    if vad_turn is None:
                        rec_name = await self.start_recording()
                        if rec_name is None:
                            break
                        rec_info = await self.wait_recording()
                        self._clear_recording_wait()
                except asyncio.CancelledError:
                    break
                try:
                    if vad_turn is not None:
                        wav_path, duration_sec = vad_turn
                        if wav_path is None:
                            if await self._skip_turn_no_usable_speech(
                                ari_duration_s=0.0,
                                reason="no speech (VAD)",
                            ):
                                break
                            continue
                    else:
                        state = rec_info.get("state")
                        raw_dur = rec_info.get("duration")
                        try:
                            duration_sec = (
                                float(raw_dur) if raw_dur is not None and raw_dur != "" else 0.0
                            )
                        except (TypeError, ValueError):
                            duration_sec = 0.0
                        log.info(
                            "RecordingFinished name=%s state=%s duration=%s cause=%s",
                            rec_info.get("name"),
                            state,
                            raw_dur,
                            rec_info.get("cause"),
                        )
                        if state == "failed":
                            log.warning(
                                "Recording failed cause=%r — fix ARI/channel (see Asterisk logs).",
                                rec_info.get("cause"),
                            )
                            await self.say_text(
                                conversation.pick_retry_phrase(self.ctx.preferred_lang)
                            )
                            continue
                        if state == "done" and duration_sec <= 0.0:
                            log.info(
                                "Recording reports duration 0s; still fetching file "
                                "(often silent / no inbound RTP — check RTP + PJSIP external_*)."
                            )

                        min_dur = self.settings.record_min_duration_seconds
                        if (
                            state == "done"
                            and 0 < duration_sec < min_dur
                        ):
                            log.info(
                                "[%s] Short utterance %.2fs < %.2fs — listening again",
                                self.ctx.session_id,
                                duration_sec,
                                min_dur,
                            )
                            continue

                        name = rec_info.get("name") or ""
                        if not name:
                            break
                        wav_path = await self.fetch_recording_wav(name)
                        await self.delete_stored_recording(name)
                        if wav_path is None:
                            silent = duration_sec <= 0.0 or (
                                0 < duration_sec < self.settings.record_min_duration_seconds
                            )
                            if silent:
                                if await self._skip_turn_no_usable_speech(
                                    ari_duration_s=duration_sec,
                                    reason="no recording file (silence)",
                                ):
                                    break
                                continue
                            log.warning(
                                "No recording file in ARI store (404). duration=%ss — "
                                "check RTP / mic if this repeats.",
                                duration_sec,
                            )
                            if await self._skip_turn_no_usable_speech(
                                ari_duration_s=duration_sec,
                                reason="recording file missing",
                            ):
                                break
                            continue

                    wav_dur = gender_model.wav_duration_seconds(wav_path)
                    min_dur = self.settings.record_min_duration_seconds
                    if wav_dur < min_dur:
                        noise = self._is_likely_line_noise_clip(
                            ari_duration_s=duration_sec,
                            wav_duration_s=wav_dur,
                        )
                        if await self._skip_turn_no_usable_speech(
                            ari_duration_s=duration_sec,
                            wav_duration_s=wav_dur,
                            reason=(
                                "WAV shorter than minimum — skip STT"
                                + (" (post-play noise)" if noise else "")
                            ),
                            count_streak=not noise,
                        ):
                            break
                        continue

                    t_pause = time.perf_counter()
                    await self._pause_before_response()
                    pause_before_response_s = time.perf_counter() - t_pause

                    t_turn = time.perf_counter()
                    (
                        user_text,
                        whisper_lang,
                        whisper_prob,
                        pitch_hz,
                    ) = await self._transcribe_turn(wav_path)
                    if pitch_hz is not None:
                        waiting_for_model = (
                            self.settings.gender_model_enabled
                            and not self.ctx.caller_gender_model_done
                        )
                        if not waiting_for_model:
                            self.ctx.update_caller_gender_from_pitch(
                                pitch_hz,
                                female_min_hz=self.settings.gender_female_min_hz,
                                male_max_hz=self.settings.gender_male_max_hz,
                                min_samples=self.settings.gender_min_pitch_samples,
                                min_confident_samples=self.settings.gender_min_confident_samples,
                                confident_fraction=self.settings.gender_confident_fraction,
                            )
                    user_text = _drop_likely_silence_hallucination(
                        user_text,
                        user_turn_count=self.ctx.turn_count,
                        record_duration_s=max(float(duration_sec or 0), wav_dur),
                    )
                    if not user_text:
                        if await self._skip_turn_no_usable_speech(
                            ari_duration_s=duration_sec,
                            wav_duration_s=wav_dur,
                            reason="empty STT transcript",
                        ):
                            break
                        continue

                    self._consecutive_empty_listens = 0

                    self._start_gender_model_in_background(
                        wav_path,
                        user_text,
                        record_duration_s=max(float(duration_sec or 0), wav_dur),
                    )
                    if pitch_hz is not None and not self.ctx.caller_gender_confident:
                        self.ctx.update_caller_gender_from_pitch(
                            pitch_hz,
                            female_min_hz=self.settings.gender_female_min_hz,
                            male_max_hz=self.settings.gender_male_max_hz,
                            min_samples=self.settings.gender_min_pitch_samples,
                            min_confident_samples=self.settings.gender_min_confident_samples,
                            confident_fraction=self.settings.gender_confident_fraction,
                        )

                    reply_lang = self.ctx.set_lang_from_user(
                        user_text,
                        whisper_lang=whisper_lang,
                        whisper_prob=whisper_prob,
                        min_words=self.settings.lang_switch_min_words,
                        min_prob=self.settings.lang_switch_min_prob,
                    )
                    stt_s = time.perf_counter() - t_turn
                    if (
                        self.ctx.caller_gender_from_model
                        and self.ctx.caller_gender_confident
                    ):
                        grammar_tag = " (caller gender-model)"
                    elif self.ctx.caller_gender in ("male", "female") and (
                        self.settings.gender_adapt_when_confident
                        or self.settings.gender_mirror_grammar
                    ):
                        grammar_tag = " (caller-adapted)"
                    elif (
                        self.ctx.resolved_grammar_gender() == "neutral"
                        and self.ctx.caller_gender not in ("male", "female")
                    ):
                        grammar_tag = " (neutral until caller known)"
                    else:
                        grammar_tag = ""
                    if self.ctx.caller_gender_confident:
                        caller_mark = "!"
                    elif not self.ctx.caller_gender_model_done:
                        caller_mark = "?"
                    else:
                        caller_mark = "~"
                    log.info(
                        "[%s] turn=%s lang=%s bot_grammar=%s%s caller=%s%s whisper=%s user: %s (stt %.2fs)",
                        self.ctx.session_id,
                        self.ctx.turn_count + 1,
                        reply_lang,
                        self.ctx.resolved_grammar_gender(),
                        grammar_tag,
                        self.ctx.caller_gender or "-",
                        caller_mark,
                        whisper_lang or "-",
                        user_text,
                        stt_s,
                    )
                    grammar = self.ctx.resolved_grammar_gender()

                    web_context, web_s, web_from_cache = await self._maybe_fetch_web_context(
                        user_text, reply_lang
                    )
                    if web_from_cache:
                        log.info(
                            "[%s] turn=%s using cached web context (topic=%s)",
                            self.ctx.session_id,
                            self.ctx.turn_count + 1,
                            self.ctx.last_web_topic or "-",
                        )

                    if self._use_llm_streaming():
                        if web_context is None:
                            query = await self._model_requested_lookup(user_text)
                            if query:
                                web_context, extra_s = await self._search_for_model(query)
                                web_s += extra_s
                        reply, _spoken, llm_s, tts_s = await self._reply_and_speak_streaming(
                            user_text=user_text,
                            reply_lang=reply_lang,
                            grammar=grammar,
                            web_context=web_context,
                        )
                    else:
                        t_llm = time.perf_counter()
                        reply, web_context, extra_s = await self._reply_with_lookup(
                            user_text=user_text,
                            reply_lang=reply_lang,
                            grammar=grammar,
                            web_context=web_context,
                        )
                        web_s += extra_s
                        llm_s = time.perf_counter() - t_llm - extra_s
                        reply = llm.strip_search_lines(reply)
                        reply = conversation.strip_robotic_phrases(
                            conversation.strip_leading_filler(reply)
                        )
                        if reply_lang == "hi":
                            fixed = conversation.align_hindi_bot_grammar(
                                reply, grammar_gender=grammar
                            )
                            if fixed != reply:
                                log.info(
                                    "[%s] Hindi grammar aligned (%s): %r -> %r",
                                    self.ctx.session_id,
                                    grammar,
                                    reply[:60],
                                    fixed[:60],
                                )
                                reply = fixed
                        if not lang.reply_should_be_devanagari(reply, reply_lang):
                            log.warning(
                                "[%s] LLM replied in Latin/English while lang=hi; "
                                "keeping Hindi TTS — consider stronger prompt",
                                self.ctx.session_id,
                            )
                        t_tts = time.perf_counter()
                        spoken = await self.say_text(reply, record_in_session=False)
                        tts_s = time.perf_counter() - t_tts
                        # Remember what the caller actually heard (trimmed, or cut short by
                        # barge-in); otherwise the model "remembers" things never said aloud.
                        reply = spoken

                    # Keep history correctly ordered: user spoke first, then bot.
                    self.ctx.add_user(user_text)
                    if reply and reply.strip():
                        self.ctx.add_assistant(reply.strip())
                    self.ctx.log_persona_grammar_unlock_if_needed()
                    reply_chars = len((reply or "").strip())
                    _log_turn_latency_breakdown(
                        self.ctx.session_id,
                        self.ctx.turn_count,
                        pause_before_response_s=pause_before_response_s,
                        stt_s=stt_s,
                        web_s=web_s,
                        llm_s=llm_s,
                        tts_synth_s=self._last_tts_synth_s,
                        tts_play_s=self._last_tts_play_s,
                        reply_chars=reply_chars,
                        record_duration_s=duration_sec,
                        record_silence_s=(
                            self.settings.vad_end_silence_ms / 1000.0
                            if self._vad is not None
                            else self._effective_record_silence_seconds()
                        ),
                    )
                except asyncio.CancelledError:
                    break
                except httpx.HTTPError as exc:
                    log.info("Channel HTTP error (hangup?): %s", exc)
                    break
                except (APITimeoutError, APIConnectionError) as exc:
                    log.warning(
                        "vLLM unreachable at %s: %s",
                        self.settings.vllm_base_url,
                        exc,
                    )
                    try:
                        await self.say_text(
                            conversation.pick_llm_unavailable_phrase(
                                self.ctx.preferred_lang
                            )
                        )
                    except Exception:
                        pass
                    continue
                except Exception as exc:
                    log.exception("Session error: %s", exc)
                    if "supertonic" in str(exc).lower() or "tts" in str(exc).lower():
                        log.error(
                            "Outbound TTS failed (Supertonic). "
                            "Check pip install supertonic and SUPERTONIC_VOICE_* in .env."
                        )
                    else:
                        try:
                            await self.say_text("Sorry, something went wrong. Goodbye.")
                        except Exception:
                            pass
                    break
                finally:
                    if wav_path:
                        wav_path.unlink(missing_ok=True)
        finally:
            await self._teardown_live_vad()
            await self.hangup()
