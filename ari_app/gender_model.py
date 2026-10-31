"""Caller gender from speech (Wav2Vec2), once per call after a long utterance."""

from __future__ import annotations

import logging
import threading
import wave
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_MODEL_NAME = "prithivMLmods/Common-Voice-Geneder-Detection"
_ID2LABEL = {"0": "female", "1": "male"}

_model: Any = None
_processor: Any = None
_load_lock = threading.Lock()
_load_failed = False


def model_available() -> bool:
    """True if transformers/torch/librosa are installed (model loads lazily)."""
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        import librosa  # noqa: F401
    except ImportError:
        return False
    return True


def _ensure_model() -> bool:
    global _model, _processor, _load_failed
    if _load_failed:
        return False
    if _model is not None and _processor is not None:
        return True
    with _load_lock:
        if _load_failed:
            return False
        if _model is not None:
            return True
        if not model_available():
            _load_failed = True
            log.warning(
                "Gender model disabled: pip install transformers torch librosa"
            )
            return False
        try:
            from transformers import (
                Wav2Vec2FeatureExtractor,
                Wav2Vec2ForSequenceClassification,
            )

            log.info("Loading caller gender model %s ...", _MODEL_NAME)
            _model = Wav2Vec2ForSequenceClassification.from_pretrained(_MODEL_NAME)
            _processor = Wav2Vec2FeatureExtractor.from_pretrained(_MODEL_NAME)
            _model.eval()
            log.info("Caller gender model ready")
            return True
        except Exception as exc:
            _load_failed = True
            log.error("Failed to load caller gender model: %s", exc)
            return False


def warmup_gender_model(*, enabled: bool = True) -> bool:
    """
    Preload Wav2Vec2 weights at process startup (avoids delay on first call).

    Returns True if the model is loaded and ready.
    """
    if not enabled:
        log.info("GENDER_MODEL_ENABLED=false — skipping gender model warmup")
        return False
    if not model_available():
        log.warning(
            "Gender model warmup skipped: pip install transformers torch librosa"
        )
        return False
    return _ensure_model()


def wav_duration_seconds(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            if rate <= 0:
                return 0.0
            return wf.getnframes() / float(rate)
    except wave.Error:
        return 0.0


def classify_caller_gender(
    wav_path: Path,
    *,
    min_confidence: float = 0.55,
) -> tuple[str, float, float] | None:
    """
    Classify speaker gender from a phone WAV (mono 8 kHz is fine; resampled to 16 kHz).

    Returns (gender, p_female, p_male) or None if unsure / error.
    """
    if not _ensure_model():
        return None

    dur = wav_duration_seconds(wav_path)
    if dur < 0.8:
        log.debug("Gender model skip: audio too short (%.2fs)", dur)
        return None

    try:
        import librosa
        import torch

        speech, sample_rate = librosa.load(str(wav_path), sr=16000, mono=True)
        if speech is None or len(speech) < int(16000 * 0.8):
            return None

        inputs = _processor(
            speech,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = _model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        if isinstance(probs, float):
            probs = [1.0 - probs, probs]

        p_female = float(probs[0])
        p_male = float(probs[1])
        if p_female >= p_male:
            gender = "female"
            conf = p_female
        else:
            gender = "male"
            conf = p_male

        if conf < min_confidence:
            log.info(
                "Gender model unsure on %s: female=%.3f male=%.3f (need %.2f)",
                wav_path.name,
                p_female,
                p_male,
                min_confidence,
            )
            return None

        log.info(
            "Gender model %s: %s (female=%.3f male=%.3f, audio=%.2fs)",
            wav_path.name,
            gender,
            p_female,
            p_male,
            dur,
        )
        return gender, p_female, p_male
    except Exception as exc:
        log.warning("Gender model failed for %s: %s", wav_path, exc)
        return None
