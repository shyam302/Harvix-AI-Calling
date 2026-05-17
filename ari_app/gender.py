"""Estimate caller pitch from phone audio (male/female hint when confident)."""

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


def classify_pitch_history(
    history: list[float],
    *,
    female_min_hz: float,
    male_max_hz: float,
    min_confident_samples: int = 4,
    confident_fraction: float = 0.75,
) -> tuple[str | None, bool, float]:
    """
    Returns (gender guess, is_confident, confidence_score).

    Confident only when enough samples and most agree in clear male or female band.
    """
    if not history:
        return None, False, 0.0
    n = len(history)
    if n < min_confident_samples:
        return None, False, 0.0

    male_n = sum(1 for p in history if p <= male_max_hz)
    female_n = sum(1 for p in history if p >= female_min_hz)
    male_frac = male_n / n
    female_frac = female_n / n

    if male_frac >= confident_fraction and male_n >= min_confident_samples:
        return "male", True, male_frac
    if female_frac >= confident_fraction and female_n >= min_confident_samples:
        return "female", True, female_frac

    median = float(np.median(history))
    tentative = pitch_to_gender(
        median, female_min_hz=female_min_hz, male_max_hz=male_max_hz
    )
    return tentative, False, max(male_frac, female_frac)


def merge_gender_estimate(
    current: str | None,
    pitch_hz: float | None,
    *,
    female_min_hz: float,
    male_max_hz: float,
    min_samples: int = 2,
    min_confident_samples: int = 4,
    confident_fraction: float = 0.75,
    pitch_history: list[float] | None = None,
    lock_after_set: bool = True,
) -> tuple[str | None, list[float], bool]:
    """
    Update pitch history; set caller gender only when confident.

    Returns (gender, history, caller_gender_confident).
    Once confident, gender is locked for the call (lock_after_set).
    """
    history = list(pitch_history or [])
    if pitch_hz is not None and pitch_hz > 0:
        history.append(pitch_hz)
        history = history[-10:]

    if (
        lock_after_set
        and current in ("male", "female")
        and len(history) >= min_confident_samples
    ):
        return current, history, True

    if len(history) < min_samples:
        return current, history, False

    guess, confident, score = classify_pitch_history(
        history,
        female_min_hz=female_min_hz,
        male_max_hz=male_max_hz,
        min_confident_samples=min_confident_samples,
        confident_fraction=confident_fraction,
    )
    if confident and guess:
        log.debug(
            "pitch gender confident: %s (median=%.0fHz n=%s score=%.2f)",
            guess,
            float(np.median(history)),
            len(history),
            score,
        )
        return guess, history, True

    return current, history, False
