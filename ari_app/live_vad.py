"""
Live voice activity on the caller's audio: end-of-turn detection, turn capture
and barge-in.

Asterisk's ``maxSilenceSeconds`` uses a fixed energy threshold; phone-line noise
often stays above it, so recordings ran 10-40 s after the caller stopped talking.
Instead, the call session snoops the caller's inbound audio into an ARI
externalMedia channel that streams RTP (slin16) to a local UDP port here, and
Silero VAD (bundled with faster-whisper) decides when the caller speaks and stops.

Modes:
  - listen: capture one caller utterance; ``ended`` fires ~END_SILENCE after speech.
  - barge:  while the bot is speaking or thinking; ``speech_started`` fires when the
            caller talks over it (stricter threshold so line echo does not trigger).
            The session then stops playback and switches to listen mode, keeping the
            audio already captured so the start of the caller's sentence is not lost.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Helper channels the session creates in its own Stasis app; ari_loop must not
# treat their StasisStart as a new call.
INTERNAL_CHANNEL_PREFIXES = ("cbsnoop-", "cbmedia-")

SAMPLE_RATE = 16000
_WINDOW = 512  # Silero window at 16 kHz (32 ms)
_CONTEXT = 64
_WINDOW_MS = 1000.0 * _WINDOW / SAMPLE_RATE
_PREROLL_S = 0.5  # audio kept before the detected speech onset
_RUN_BREAK_WINDOWS = 4  # ~130 ms of non-speech ends a speech run
_MAX_BUFFER_S = 120.0  # cap on audio held while armed

_model_lock = threading.Lock()
_model: Any = None


def _vad_model() -> Any:
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper.vad import get_vad_model

            _model = get_vad_model()
        return _model


def warmup_vad() -> None:
    _vad_model()


class LiveVad(asyncio.DatagramProtocol):
    """Receives caller RTP, runs Silero VAD per 32 ms window, buffers the turn."""

    def __init__(
        self,
        *,
        end_silence_ms: int = 700,
        speech_threshold: float = 0.5,
        min_speech_ms: int = 200,
        no_speech_timeout_s: float = 8.0,
        barge_threshold: float = 0.7,
        barge_min_speech_ms: int = 400,
    ) -> None:
        self.end_silence_s = end_silence_ms / 1000.0
        self.end_windows = max(3, int(end_silence_ms / _WINDOW_MS))
        self.listen_threshold = speech_threshold
        self.listen_min_windows = max(1, int(min_speech_ms / _WINDOW_MS))
        self.no_speech_timeout_s = no_speech_timeout_s
        self.barge_threshold = barge_threshold
        self.barge_min_windows = max(1, int(barge_min_speech_ms / _WINDOW_MS))

        self.transport: asyncio.DatagramTransport | None = None
        self.packets = 0
        self.speech_started = asyncio.Event()
        self.ended = asyncio.Event()
        self.mode = "off"  # off | listen | barge
        self._threshold = speech_threshold
        self._min_windows = self.listen_min_windows
        self._timeout_s = no_speech_timeout_s
        self._timer: asyncio.TimerHandle | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        self.heard_speech = False
        self.packets_armed = 0
        self._chunks: list[np.ndarray] = []
        self._total = 0  # int16 samples buffered
        self._pending = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)
        self._win_index = 0  # windows analysed since arm
        self._run = 0  # speech windows in the current run
        self._run_start_win = 0
        self._gap = 0  # non-speech windows since the run's last speech window
        self._silent_run = 0
        self._speech_start_sample: int | None = None
        self._armed_at = self._last_packet = time.monotonic()

    # --- asyncio.DatagramProtocol -------------------------------------------------
    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self.packets += 1
        self._last_packet = time.monotonic()
        if self.mode == "off" or self.ended.is_set() or len(data) <= 12:
            return
        self.packets_armed += 1
        # RTP header: 12 bytes + 4 per CSRC (no header extensions from Asterisk here).
        csrc = data[0] & 0x0F
        payload = data[12 + 4 * csrc :]
        if len(payload) < 2:
            return
        # slin16 in RTP is 16-bit big-endian (network order).
        pcm = np.frombuffer(payload[: len(payload) // 2 * 2], dtype=">i2").astype(np.int16)
        if self._total < _MAX_BUFFER_S * SAMPLE_RATE:
            self._chunks.append(pcm)
            self._total += len(pcm)
        self._feed(pcm.astype(np.float32) / 32768.0)

    # --- turn control --------------------------------------------------------------
    @property
    def port(self) -> int | None:
        if self.transport is None:
            return None
        return self.transport.get_extra_info("sockname")[1]

    def arm(self, mode: str) -> None:
        """Start a fresh capture in ``listen`` or ``barge`` mode."""
        self._cancel_timer()
        self._reset_state()
        self.speech_started = asyncio.Event()
        self.ended = asyncio.Event()
        self._set_mode(mode)
        self._schedule_tick()

    def to_listen(self) -> None:
        """
        Switch to listen mode, keeping audio and VAD state (speech that began during
        barge mode continues as this turn). Arms fresh if currently off.
        """
        if self.mode == "off":
            self.arm("listen")
            return
        self._set_mode("listen")
        self._armed_at = time.monotonic()

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode == "barge":
            self._threshold = self.barge_threshold
            self._min_windows = self.barge_min_windows
            self._timeout_s = math.inf
        else:
            self._threshold = self.listen_threshold
            self._min_windows = self.listen_min_windows
            self._timeout_s = self.no_speech_timeout_s

    def disarm(self) -> None:
        self.mode = "off"
        self._cancel_timer()

    def close(self) -> None:
        self.disarm()
        self.ended.set()
        if self.transport is not None:
            self.transport.close()
            self.transport = None

    def utterance_pcm(self) -> np.ndarray:
        """Captured int16 audio from just before speech onset to now."""
        if not self._chunks:
            return np.zeros(0, dtype=np.int16)
        audio = np.concatenate(self._chunks)
        start = self._speech_start_sample or 0
        return audio[max(0, start) :]

    # --- timers --------------------------------------------------------------------
    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule_tick(self) -> None:
        self._cancel_timer()
        self._timer = asyncio.get_running_loop().call_later(0.1, self._tick)

    def _tick(self) -> None:
        """Wall-clock checks: Asterisk / phones send no RTP while nobody talks."""
        self._timer = None
        if self.mode == "off" or self.ended.is_set():
            return
        now = time.monotonic()
        if (
            self.mode == "listen"
            and self.heard_speech
            and now - self._last_packet >= self.end_silence_s
        ):
            self.ended.set()
            return
        if not self.heard_speech and now - self._armed_at >= self._timeout_s:
            self.ended.set()
            return
        self._schedule_tick()

    # --- detection -----------------------------------------------------------------
    def _feed(self, samples: np.ndarray) -> None:
        buf = np.concatenate([self._pending, samples])
        n = len(buf) // _WINDOW
        if n == 0:
            self._pending = buf
            return
        windows = buf[: n * _WINDOW].reshape(n, _WINDOW)
        self._pending = buf[n * _WINDOW :]

        # Each window gets the previous 64 samples as context (same as faster-whisper).
        prev = np.vstack([self._context[None, :], windows[:-1, -_CONTEXT:]])
        self._context = windows[-1, -_CONTEXT:].copy()
        batch = np.concatenate([prev, windows], axis=1).astype(np.float32)
        h = np.zeros((1, 1, 128), dtype=np.float32)
        c = np.zeros((1, 1, 128), dtype=np.float32)
        out, _, _ = _vad_model().session.run(None, {"input": batch, "h": h, "c": c})
        for p in np.asarray(out).reshape(-1):
            self._on_window(float(p))
            if self.ended.is_set():
                return

    def _on_window(self, prob: float) -> None:
        win = self._win_index
        self._win_index += 1
        silence_threshold = max(0.05, self.listen_threshold - 0.15)

        if prob >= self._threshold:
            if self._run == 0:
                self._run_start_win = win
            self._run += 1
            self._gap = 0
            self._silent_run = 0
        else:
            self._gap += 1
            if self._gap >= _RUN_BREAK_WINDOWS:
                self._run = 0
            if prob < silence_threshold:
                self._silent_run += 1

        if not self.heard_speech and self._run >= self._min_windows:
            self.heard_speech = True
            onset = self._run_start_win * _WINDOW - int(_PREROLL_S * SAMPLE_RATE)
            self._speech_start_sample = max(0, onset)
            self.speech_started.set()

        if self.mode == "listen" and self.heard_speech and self._silent_run >= self.end_windows:
            self.ended.set()


async def open_listener(**kwargs: Any) -> LiveVad:
    """Bind a LiveVad to an ephemeral UDP port on 127.0.0.1."""
    loop = asyncio.get_running_loop()
    _, protocol = await loop.create_datagram_endpoint(
        lambda: LiveVad(**kwargs), local_addr=("127.0.0.1", 0)
    )
    return protocol
