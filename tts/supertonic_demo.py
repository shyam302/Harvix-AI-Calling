"""Supertonic demo — GPU via ONNX Runtime (configure before TTS load)."""

from __future__ import annotations

import sys
import time

import numpy as np

# Project root on path when run as: python tts/supertonic_demo.py
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from ari_app.supertonic_onnx import (
    configure_supertonic_onnx,
    diagnose_cuda_onnx,
    ensure_cuda_library_path,
    log_session_providers,
    session_uses_cuda,
)


def _synth_and_save(
    tts,
    style,
    text: str,
    lang: str,
    out_path: str,
) -> None:
    t0 = time.perf_counter()
    wav, duration = tts.synthesize(text, voice_style=style, lang=lang)
    synth_s = time.perf_counter() - t0
    tts.save_audio(wav, out_path)
    save_s = time.perf_counter() - t0 - synth_s
    audio_s = float(np.asarray(duration).squeeze())
    print(
        f"[{lang}] {out_path} — audio {audio_s:.2f}s | "
        f"synth {synth_s:.2f}s | save {save_s:.2f}s | total {synth_s + save_s:.2f}s"
    )


def main() -> None:
    wall = time.perf_counter()
    # Before any onnxruntime import (pip CUDA libs are not on default LD path).
    ensure_cuda_library_path()

    providers = configure_supertonic_onnx(use_gpu=True)
    print("Configured ONNX providers:", providers)
    print("ONNX diagnosis:")
    print(diagnose_cuda_onnx())
    print()

    from supertonic import TTS

    t_load = time.perf_counter()
    tts = TTS(auto_download=True)
    style = tts.get_voice_style(voice_name="F3")
    load_s = time.perf_counter() - t_load
    print(f"Model load: {load_s:.2f}s")
    print("Active session providers (must show CUDAExecutionProvider):")
    log_session_providers(tts)

    _synth_and_save(
        tts,
        style,
        "आपका स्वागत है शिक्षा के दौरान आपको एक बार फिर से बात करने का मौका देंगे।",
        "hi",
        "output.wav",
    )
    _synth_and_save(
        tts,
        style,
        "A gentle breeze moved through the open window while everyone listened to the story.",
        "en",
        "output_en.wav",
    )

    print(f"Wall clock total: {time.perf_counter() - wall:.2f}s")
    if not session_uses_cuda(tts):
        print(
            "\nStill on CPU (libcurand.so.10 not found at runtime).\n"
            "  pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 "
            "nvidia-cudnn-cu12 nvidia-cufft-cu12\n"
            "  Close this terminal, open a new one, conda activate py311, run demo again."
        )


if __name__ == "__main__":
    main()
