"""Long-running Asterisk ARI WebSocket loop (used by CLI and uvicorn)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any
from urllib.parse import quote

import httpx
import websockets
from websockets.exceptions import InvalidStatus

from ari_app.call_context import ActiveCallRegistry, CallContext
from ari_app.call_session import (
    CallSession,
    channel_id_from_target_uri,
    stasis_channel_peer_info,
)
from ari_app.config import load_settings
from ari_app.stt import whisper_runtime_info

log = logging.getLogger("ari_callbot")
_conn_refused_hint_logged = False
_active_calls = ActiveCallRegistry()


def get_active_call_registry() -> ActiveCallRegistry:
    return _active_calls


def _sounds_en_dir():
    raw = os.getenv("ASTERISK_SOUNDS_EN", "/var/lib/asterisk/sounds/en")
    from pathlib import Path

    return Path(raw)


async def run_ari_forever() -> None:
    global _conn_refused_hint_logged
    settings = load_settings()
    if not settings.ari_password:
        log.error("Set ARI_PASSWORD in the environment or .env file.")
        sys.exit(1)

    ari_base = f"http://{settings.ari_host}:{settings.ari_port}/ari"
    auth = httpx.BasicAuth(settings.ari_username, settings.ari_password)
    api_key = quote(f"{settings.ari_username}:{settings.ari_password}", safe="")
    ws_url = (
        f"ws://{settings.ari_host}:{settings.ari_port}/ari/events"
        f"?app={quote(settings.stasis_app)}&api_key={api_key}"
    )

    sounds_en = _sounds_en_dir()
    sessions: dict[str, CallSession] = {}
    registry = _active_calls

    async def dispatch(ev: dict[str, Any], http_client: httpx.AsyncClient) -> None:
        et = ev.get("type")
        if et == "StasisStart":
            ch = ev.get("channel") or {}
            cid = ch.get("id")
            if not cid:
                return
            caller_number, caller_name, dialed = stasis_channel_peer_info(ch)
            ctx = CallContext.create(
                channel_id=cid,
                caller_number=caller_number,
                caller_name=caller_name,
                dialed_number=dialed,
                max_messages=settings.session_max_messages,
                primary_lang=settings.call_primary_lang,
            )
            registry.register(ctx)
            session = CallSession(
                settings=settings,
                channel_id=cid,
                client=http_client,
                ari_base=ari_base,
                auth=auth,
                sounds_en_dir=sounds_en,
                caller_number=caller_number,
                caller_name=caller_name,
                dialed_number=dialed,
                ctx=ctx,
            )
            sessions[cid] = session

            async def _task() -> None:
                try:
                    await session.run_loop()
                except Exception:
                    log.exception(
                        "Call session failed channel=%s session=%s",
                        cid,
                        ctx.session_id,
                    )
                finally:
                    sessions.pop(cid, None)
                    registry.unregister(cid)

            asyncio.create_task(_task())
            return

        if et == "RecordingFinished":
            rec = ev.get("recording") or {}
            cid = channel_id_from_target_uri(rec.get("target_uri") or "")
            if cid and cid in sessions:
                sessions[cid].on_recording_finished(rec)
            return

        if et == "RecordingFailed":
            rec = ev.get("recording") or {}
            cid = channel_id_from_target_uri(rec.get("target_uri") or "")
            if cid and cid in sessions:
                sessions[cid].on_recording_failed(rec)
            return

        if et == "PlaybackFinished":
            pb = ev.get("playback") or {}
            cid = channel_id_from_target_uri(pb.get("target_uri") or "")
            if cid and cid in sessions:
                sessions[cid].on_playback_finished(pb)
            return

        if et == "ChannelDestroyed":
            ch = ev.get("channel") or {}
            cid = ch.get("id")
            if cid and cid in sessions:
                sess = sessions.pop(cid, None)
                if sess:
                    sess.mark_channel_dead()
                registry.unregister(cid)
            return

    log.info(
        "ARI WebSocket app=%s sounds_en=%s vLLM=%s whisper=%s lang=%s model=%s tts=supertonic voice_en=%s voice_hi=%s",
        settings.stasis_app,
        sounds_en,
        settings.vllm_base_url,
        whisper_runtime_info(),
        settings.whisper_language or "auto",
        settings.whisper_model,
        settings.tts_voice,
        settings.tts_voice_hi or "(same as en)",
    )

    async with httpx.AsyncClient(timeout=180.0) as http_client:
        while True:
            try:
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    async for raw in ws:
                        try:
                            ev = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        await dispatch(ev, http_client)
            except InvalidStatus as exc:
                code = getattr(exc.response, "status_code", None) or getattr(
                    exc.response, "status", 0
                )
                if code == 401:
                    log.error(
                        "ARI HTTP 401: add a user stanza to /etc/asterisk/ari.conf whose "
                        "section name matches ARI_USERNAME and password matches ARI_PASSWORD "
                        "(see asterisk_snippets/ari_user_callbot.conf), then: "
                        'sudo asterisk -rx "module reload res_ari.so" or core reload.'
                    )
                    sys.exit(1)
                log.warning("ARI WebSocket rejected with HTTP %s; reconnecting in 2s...", code)
                await asyncio.sleep(2)
            except (websockets.ConnectionClosed, OSError) as exc:
                log.warning("ARI WebSocket error %s; reconnecting in 2s...", exc)
                err = getattr(exc, "errno", None)
                if err == 111 and not _conn_refused_hint_logged:
                    _conn_refused_hint_logged = True
                    log.warning(
                        "Connection refused on port %s: enable Asterisk HTTP in "
                        "/etc/asterisk/http.conf (enabled=yes, bindaddr, bindport). "
                        "See asterisk_snippets/http_enable_ari.txt",
                        settings.ari_port,
                    )
                await asyncio.sleep(2)
