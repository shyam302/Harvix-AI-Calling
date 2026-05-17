"""Per-call session: record -> STT -> LLM -> TTS -> play (loop until hangup)."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from openai import APIConnectionError, APITimeoutError

from ari_app import conversation, gender, lang, llm, stt, tts, voice_select
from ari_app.call_context import CallContext
from ari_app.config import Settings

log = logging.getLogger(__name__)

# Whisper on near-silent RTP often emits one syllable; our LLM is told to answer
# goodbye warmly — that sounds like "bye" immediately after the opening greeting.
_SILENCE_TRANSCRIPT_JUNK = frozenset(
    {
        "bye",
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
        "yeah",
        "yep",
        "oh",
        "ah",
        "hey",
    }
)


def _drop_likely_silence_hallucination(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    key = stripped.lower().strip(".!?…").replace("'", "")
    if len(key) > 16:
        return stripped
    if key in _SILENCE_TRANSCRIPT_JUNK:
        log.info("Dropping likely silence/noise transcript %r", stripped)
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
        await self._delete(self._channel_path())

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
        agent = self.ctx.agent_gender(default="female")
        return voice_select.supertonic_voice_for_agent(self.settings, agent)

    def _clear_recording_wait(self) -> None:
        self._record_fut = None
        self._pending_record_name = None

    async def _prepare_to_listen(self) -> None:
        """Gap after bot speech + drain playback before opening the mic."""
        if self._closed:
            return
        for fut in list(self._play_futs.values()):
            if not fut.done():
                try:
                    await asyncio.wait_for(asyncio.shield(fut), timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
        self._play_futs.clear()
        ms = self.settings.pause_after_tts_ms + self.settings.pause_before_listen_ms
        if ms > 0:
            log.debug("[%s] pause %.0fms before listen", self.ctx.session_id, ms)
            await asyncio.sleep(ms / 1000.0)

    async def _pause_before_response(self) -> None:
        """Brief gap after caller stops speaking, before STT/LLM (feels less abrupt)."""
        ms = self.settings.pause_before_response_ms
        if ms > 0:
            await asyncio.sleep(ms / 1000.0)

    async def start_recording(self) -> str | None:
        name = f"cb-{uuid.uuid4().hex}"
        self.arm_recording_wait(name)
        silence = int(round(self.settings.record_max_silence_seconds))
        silence = max(1, min(silence, 60))
        body = {
            "name": name,
            "format": "wav",
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
        ok = await stt.download_recording_file(url=url, dest=p, auth=self.auth)
        if not ok:
            p.unlink(missing_ok=True)
            return None
        return p

    async def delete_stored_recording(self, recording_name: str) -> None:
        await self._delete(f"/recordings/stored/{recording_name}")

    async def play_sound(self, media: str) -> None:
        r = await self._post(f"{self._channel_path()}/play", json={"media": media})
        data = r.json()
        pb_id = data.get("id")
        if not pb_id:
            return
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._play_futs[pb_id] = fut
        await asyncio.wait_for(fut, timeout=120.0)

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
    ) -> None:
        await tts.synthesize_to_wav(
            phrase,
            out_wav,
            voice_en=voice_en,
            voice_hi=voice_hi,
            speed=self.settings.supertonic_speed,
            speed_en=self.settings.supertonic_speed_en,
            speed_hi=self.settings.supertonic_speed_hi,
            silence_duration=self.settings.supertonic_silence_duration,
            session_lang=session_lang,
            voice_name=voice_hi if session_lang == "hi" else voice_en,
        )

    async def _play_wav(self, out_wav: Path) -> None:
        sub = self.settings.tts_sound_subdir.strip("/ ")
        base = out_wav.name.replace(".wav", "")
        await self.play_sound(f"sound:{sub}/{base}")

    async def say_text(self, text: str, *, record_in_session: bool = True) -> str:
        spoken = conversation.trim_reply_for_phone(
            text,
            max_sentences=self.settings.reply_max_sentences,
            max_chars=self.settings.reply_max_chars,
        )
        if spoken != (text or "").strip():
            log.info(
                "[%s] Reply trimmed for phone: %r -> %r",
                self.ctx.session_id,
                text[:80],
                spoken[:80],
            )

        out_dir = self._tts_out_dir()
        chunks = conversation.split_tts_phrases(
            spoken,
            max_chars=self.settings.tts_chunk_max_chars,
        )
        if not chunks:
            return ""

        if record_in_session:
            self.ctx.add_assistant(spoken)

        session_lang = self.ctx.preferred_lang
        voice_en, voice_hi = self._agent_voice_pair()
        lang = tts.pick_supertonic_lang(spoken, session_lang=session_lang)
        voice = voice_hi if lang == "hi" else voice_en
        log.info(
            "[%s] TTS lang=%s voice=%s chunks=%s chars=%s",
            self.ctx.session_id,
            lang,
            voice,
            len(chunks),
            len(spoken),
        )

        wav_paths = [out_dir / f"tts-{uuid.uuid4().hex}.wav" for _ in chunks]
        synth_task: asyncio.Task[None] | None = asyncio.create_task(
            self._synthesize_phrase(
                chunks[0],
                wav_paths[0],
                session_lang=session_lang,
                voice_en=voice_en,
                voice_hi=voice_hi,
            )
        )

        for i, phrase in enumerate(chunks):
            assert synth_task is not None
            t0 = time.perf_counter()
            await synth_task
            synth_s = time.perf_counter() - t0

            if i + 1 < len(chunks):
                synth_task = asyncio.create_task(
                    self._synthesize_phrase(
                        chunks[i + 1],
                        wav_paths[i + 1],
                        session_lang=session_lang,
                        voice_en=voice_en,
                        voice_hi=voice_hi,
                    )
                )
            else:
                synth_task = None

            t1 = time.perf_counter()
            await self._play_wav(wav_paths[i])
            play_s = time.perf_counter() - t1
            log.info(
                "[%s] TTS chunk %s/%s synth=%.2fs play=%.2fs",
                self.ctx.session_id,
                i + 1,
                len(chunks),
                synth_s,
                play_s,
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
            opening = voice_select.opening_greeting_for_agent(
                self.settings,
                self.ctx.agent_gender(default="female"),
                self.settings.call_primary_lang,
            )
            await self.say_text(opening)

            while not self._closed:
                wav_path: Path | None = None
                try:
                    await self._prepare_to_listen()
                    if self._closed:
                        break
                    rec_name = await self.start_recording()
                    if rec_name is None:
                        break
                    rec_info = await self.wait_recording()
                    self._clear_recording_wait()
                except asyncio.CancelledError:
                    break
                try:
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
                        log.warning(
                            "No recording file in ARI store (404). Common when duration=0: "
                            "no audio was captured — fix inbound RTP / mic, then try again."
                        )
                        await self.say_text(
                            conversation.pick_retry_phrase(self.ctx.preferred_lang)
                        )
                        continue

                    await self._pause_before_response()

                    loop = asyncio.get_running_loop()
                    pitch_hz = await loop.run_in_executor(
                        None,
                        lambda: gender.estimate_pitch_hz(wav_path),
                    )
                    self.ctx.update_caller_gender_from_pitch(
                        pitch_hz,
                        female_min_hz=self.settings.gender_female_min_hz,
                        male_max_hz=self.settings.gender_male_max_hz,
                        min_samples=self.settings.gender_min_pitch_samples,
                    )

                    t_turn = time.perf_counter()
                    whisper_hint = lang.whisper_language_hint(
                        self.ctx.preferred_lang,
                        configured=self.settings.whisper_language,
                    )
                    user_text, whisper_lang = await stt.transcribe_wav(
                        wav_path,
                        self.settings.whisper_model,
                        language=whisper_hint,
                    )
                    user_text = _drop_likely_silence_hallucination(user_text)
                    if not user_text:
                        log.warning(
                            "Whisper returned empty text (ARI duration=%ss). "
                            "Audio likely silent — check mic, inbound UDP RTP 10000-20000, "
                            "PJSIP external_media_address on this host.",
                            raw_dur,
                        )
                        await self.say_text(
                            conversation.pick_retry_phrase(self.ctx.preferred_lang)
                        )
                        continue

                    prev_lang = self.ctx.preferred_lang
                    reply_lang = self.ctx.set_lang_from_user(
                        user_text,
                        whisper_lang=whisper_lang,
                        primary_lang=self.settings.call_primary_lang,
                        min_en_words=self.settings.lang_switch_min_en_words,
                        consecutive_en_required=self.settings.lang_switch_consecutive_en,
                    )
                    if reply_lang == "en" and prev_lang == "hi":
                        if lang.is_explicit_english_intent(user_text):
                            log.info(
                                "[%s] English switch (explicit phrase): %r",
                                self.ctx.session_id,
                                user_text[:80],
                            )
                        else:
                            log.info(
                                "[%s] English switch (detected English speech)",
                                self.ctx.session_id,
                            )
                    stt_s = time.perf_counter() - t_turn
                    log.info(
                        "[%s] turn=%s lang=%s whisper=%s user: %s (stt %.2fs)",
                        self.ctx.session_id,
                        self.ctx.turn_count + 1,
                        reply_lang,
                        whisper_lang or "-",
                        user_text,
                        stt_s,
                    )
                    t_llm = time.perf_counter()
                    reply = await llm.reply_text(
                        base_url=self.settings.vllm_base_url,
                        api_key=self.settings.vllm_api_key,
                        model=self.settings.vllm_model,
                        user_text=user_text,
                        history=self.ctx.history_for_llm(),
                        reply_lang=reply_lang,
                        agent_gender=self.ctx.agent_gender(default="female"),
                        caller_number=self.ctx.caller_number,
                        caller_name=self.ctx.caller_name,
                        caller_context=self.ctx.caller_context_line() or None,
                        session_id=self.ctx.session_id,
                        max_history_messages=self.settings.session_max_messages,
                        timeout_seconds=self.settings.vllm_timeout_seconds,
                        connect_timeout_seconds=self.settings.vllm_connect_timeout_seconds,
                        max_tokens=self.settings.llm_max_tokens,
                    )
                    llm_s = time.perf_counter() - t_llm
                    if not lang.reply_should_be_devanagari(reply, reply_lang):
                        log.warning(
                            "[%s] LLM replied in Latin/English while lang=hi; "
                            "keeping Hindi TTS — consider stronger prompt",
                            self.ctx.session_id,
                        )
                    self.ctx.add_user(user_text)
                    t_tts = time.perf_counter()
                    await self.say_text(reply)
                    tts_s = time.perf_counter() - t_tts
                    log.info(
                        "[%s] turn=%s latency: stt=%.2fs llm=%.2fs tts+play=%.2fs total=%.2fs ctx_msgs=%s",
                        self.ctx.session_id,
                        self.ctx.turn_count,
                        stt_s,
                        llm_s,
                        tts_s,
                        time.perf_counter() - t_turn,
                        len(self.ctx.messages),
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
            await self.hangup()
