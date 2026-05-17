"""Estimate caller pitch from phone audio to pick male/female TTS + LLM persona."""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def estimate_pitch_hz(wav_path: Path, *, min_duration_s: float = 0.35) -> float | None:
    """
    Median fundamental frequency (Hz) from 8 kHz mono WAV.
    Returns None if audio too short or too quiet.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            nframes = wf.getnframes()
            if sr < 4000 or nframes < int(sr * min_duration_s):
                return None
            raw = wf.readframes(min(nframes, int(sr * 4)))
    except wave.Error as exc:
        log.debug("pitch: cannot read wav %s: %s", wav_path, exc)
        return None

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return None
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)

    peak = float(np.max(np.abs(samples)))
    if peak < 400:
        return None

    frame = int(sr * 0.04)
    hop = max(frame // 2, 1)
    pitches: list[float] = []
    min_lag = max(int(sr / 320), 1)
    max_lag = min(int(sr / 70), len(samples) - 1)

    for start in range(0, len(samples) - frame, hop):
        chunk = samples[start : start + frame]
        if float(np.max(np.abs(chunk))) < peak * 0.15:
            continue
        corr = np.correlate(chunk, chunk, mode="full")
        corr = corr[len(corr) // 2 :]
        if max_lag >= len(corr):
            continue
        segment = corr[min_lag:max_lag]
        if segment.size == 0:
            continue
        lag = min_lag + int(np.argmax(segment))
        if lag > 0:
            pitches.append(sr / lag)

    if len(pitches) < 3:
        return None
    return float(np.median(pitches))


def pitch_to_gender(
    pitch_hz: float | None,
    *,
    female_min_hz: float = 165.0,
    male_max_hz: float = 155.0,
) -> str | None:
    if pitch_hz is None:
        return None
    if pitch_hz >= female_min_hz:
        return "female"
    if pitch_hz <= male_max_hz:
        return "male"
    return None


def merge_gender_estimate(
    current: str | None,
    pitch_hz: float | None,
    *,
    female_min_hz: float,
    male_max_hz: float,
    min_samples: int = 2,
    pitch_history: list[float] | None = None,
) -> tuple[str | None, list[float]]:
    """Sticky gender from median pitch over recent utterances."""
    history = list(pitch_history or [])
    if pitch_hz is not None and pitch_hz > 0:
        history.append(pitch_hz)
        history = history[-5:]

    if len(history) < min_samples:
        return current, history

    median = float(np.median(history))
    guess = pitch_to_gender(
        median, female_min_hz=female_min_hz, male_max_hz=male_max_hz
    )
    if guess is None:
        return current, history
    return guess, history
