"""Configure Supertonic ONNX Runtime to use CUDA when available."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_loader_patched = False
_cuda_lib_path_done = False
_cuda_lib_path_prepended: list[str] = []


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _parse_provider_env() -> list[str] | None:
    raw = (os.getenv("SUPERTONIC_ONNX_PROVIDERS") or "").strip()
    if not raw:
        return None
    mapping = {
        "cuda": "CUDAExecutionProvider",
        "gpu": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    out: list[str] = []
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        out.append(mapping.get(key, part.strip()))
    return out or None


def _site_packages_dirs() -> list[Path]:
    import site

    dirs: list[Path] = []
    for p in site.getsitepackages() + [site.getusersitepackages()]:
        if p:
            dirs.append(Path(p))
    return dirs


def _nvidia_pip_lib_dirs() -> list[Path]:
    """Lib folders from pip packages nvidia-*-cu12 (curand, cudnn, cublas, …)."""
    found: list[Path] = []
    seen: set[str] = set()
    for sp in _site_packages_dirs():
        nvidia_root = sp / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for lib_dir in nvidia_root.glob("*/lib"):
            if not lib_dir.is_dir():
                continue
            key = str(lib_dir.resolve())
            if key in seen:
                continue
            if any(lib_dir.glob("*.so*")):
                seen.add(key)
                found.append(lib_dir)
    return found


def ensure_cuda_library_path() -> list[str]:
    """
    Pip-installed CUDA libs are not on the default linker path.

    Call before ``import onnxruntime`` so libcurand.so.10 / libcudnn can load.
    Safe to call multiple times.
    """
    global _cuda_lib_path_done, _cuda_lib_path_prepended
    if _cuda_lib_path_done:
        return list(_cuda_lib_path_prepended)

    lib_dirs = _nvidia_pip_lib_dirs()
    if not lib_dirs:
        _cuda_lib_path_done = True
        return []

    prepend = [str(d) for d in lib_dirs]
    old = os.environ.get("LD_LIBRARY_PATH", "")
    merged: list[str] = []
    for p in prepend:
        if p and p not in merged and (not old or p not in old.split(":")):
            merged.append(p)
    if merged:
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged + ([old] if old else []))

    # Preload so ONNX CUDA provider finds deps even if LD_LIBRARY_PATH is late.
    try:
        import ctypes

        preload_names = (
            "libcudart.so.12",
            "libcublas.so.12",
            "libcublasLt.so.12",
            "libcurand.so.10",
            "libcufft.so.11",
            "libcudnn.so.9",
        )
        for lib_dir in lib_dirs:
            for name in preload_names:
                path = lib_dir / name
                if path.is_file():
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        log.debug("CUDA lib preload skipped: %s", exc)

    _cuda_lib_path_prepended = merged
    _cuda_lib_path_done = True
    log.info("CUDA library path prepended: %s", merged)
    return merged


def build_onnx_providers(*, use_gpu: bool | None = None) -> list[str]:
    """
    Plain provider name strings for ONNX Runtime.

    Supertonic's loader only accepts names that appear in
    ``ort.get_available_providers()`` (no provider option tuples).
    """
    ensure_cuda_library_path()
    import onnxruntime as ort

    if use_gpu is None:
        use_gpu = _bool_env("SUPERTONIC_USE_GPU", True)

    explicit = _parse_provider_env()
    available = ort.get_available_providers()

    if explicit:
        names = [p for p in explicit if p in available]
    elif use_gpu and "CUDAExecutionProvider" in available:
        names = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        names = ["CPUExecutionProvider"]

    return names or ["CPUExecutionProvider"]


def _cuda_load_error_hint(exc: BaseException) -> str:
    msg = str(exc)
    if any(
        x in msg
        for x in (
            "libcurand",
            "libcudnn",
            "libcublas",
            "libcudart",
            "libcufft",
        )
    ):
        return (
            "Missing CUDA 12 / cuDNN libraries for onnxruntime-gpu (e.g. libcurand.so.10). "
            "In py311: pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 "
            "nvidia-cudnn-cu12 nvidia-cufft-cu12 then re-run. "
            "site-packages/nvidia/*/lib is added to LD_LIBRARY_PATH automatically."
        )
    if "cuDNN" in msg or "CUDA 12" in msg:
        return (
            "onnxruntime-gpu needs CUDA 12.x and cuDNN 9.x on LD_LIBRARY_PATH. "
            "See the nvidia-*-cu12 pip packages above or system CUDA install."
        )
    return ""


def diagnose_cuda_onnx() -> str:
    """Human-readable ONNX GPU diagnosis for demo / logs."""
    lib_dirs = ensure_cuda_library_path()
    import onnxruntime as ort

    lines = [
        f"onnxruntime: {ort.__file__}",
        f"LD_LIBRARY_PATH (nvidia pip): {lib_dirs or '(none found)'}",
        f"available providers: {ort.get_available_providers()}",
    ]
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        lines.append(
            "CUDA provider not in build — install: "
            "pip uninstall onnxruntime onnxruntime-gpu -y && pip install onnxruntime-gpu"
        )
        return "\n".join(lines)

    lines.append(
        "Note: 'available providers' lists CUDA even when CUDA libs are broken; "
        "test load below is what matters."
    )

    try:
        from supertonic.config import DEFAULT_MODEL, DP_ONNX_REL_PATH
        from supertonic.loader import get_cache_dir

        model_dir = get_cache_dir(DEFAULT_MODEL)
        onnx_path = model_dir / DP_ONNX_REL_PATH
        if not onnx_path.is_file():
            lines.append(f"test model not found: {onnx_path} (run TTS once to download)")
            return "\n".join(lines)

        sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CUDAExecutionProvider"],
        )
        active = sess.get_providers()
        if active and active[0] == "CUDAExecutionProvider":
            lines.append(f"CUDA test load OK: {onnx_path.name} -> {active}")
        else:
            lines.append(
                f"CUDA test load FAILED: {onnx_path.name} bound to {active} "
                f"(libcurand.so.10 still not visible — restart shell after pip install)"
            )
    except Exception as exc:
        lines.append(f"CUDA test load FAILED: {exc}")
        hint = _cuda_load_error_hint(exc)
        if hint:
            lines.append(hint)

    return "\n".join(lines)


def _patch_supertonic_loader() -> None:
    """
    Replace load_onnx_modules so sessions try CUDA first and log failures.

    Also fixes stale import: ``from .config import DEFAULT_ONNX_PROVIDERS``
    in supertonic.loader keeps the old CPU list unless we patch loader too.
    """
    global _loader_patched
    if _loader_patched:
        return

    import onnxruntime as ort
    import supertonic.loader as st_loader
    from supertonic.config import (
        DEFAULT_INTER_OP_NUM_THREADS,
        DEFAULT_INTRA_OP_NUM_THREADS,
        DP_ONNX_REL_PATH,
        TEXT_ENC_ONNX_REL_PATH,
        VECTOR_EST_ONNX_REL_PATH,
        VOCODER_ONNX_REL_PATH,
    )

    def load_onnx_modules_patched(
        model_dir: Path | str,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
    ) -> tuple[
        ort.InferenceSession,
        ort.InferenceSession,
        ort.InferenceSession,
        ort.InferenceSession,
    ]:
        model_dir = Path(model_dir)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        intra_threads = (
            intra_op_num_threads
            if intra_op_num_threads is not None
            else DEFAULT_INTRA_OP_NUM_THREADS
        )
        inter_threads = (
            inter_op_num_threads
            if inter_op_num_threads is not None
            else DEFAULT_INTER_OP_NUM_THREADS
        )
        if intra_threads is not None:
            opts.intra_op_num_threads = intra_threads
        if inter_threads is not None:
            opts.inter_op_num_threads = inter_threads

        want_cuda = "CUDAExecutionProvider" in build_onnx_providers()
        available = ort.get_available_providers()

        def _open_session(onnx_path: Path) -> ort.InferenceSession:
            if not onnx_path.is_file():
                raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

            if want_cuda and "CUDAExecutionProvider" in available:
                try:
                    sess = ort.InferenceSession(
                        str(onnx_path),
                        sess_options=opts,
                        providers=["CUDAExecutionProvider"],
                    )
                    active = sess.get_providers()
                    if active and active[0] == "CUDAExecutionProvider":
                        log.info("Supertonic ONNX GPU: %s", onnx_path.name)
                        return sess
                    log.warning(
                        "Supertonic CUDA requested but %s active=%s",
                        onnx_path.name,
                        active,
                    )
                except Exception as exc:
                    hint = _cuda_load_error_hint(exc)
                    log.error(
                        "Supertonic CUDA failed for %s: %s%s",
                        onnx_path.name,
                        exc,
                        f" — {hint}" if hint else "",
                    )

            log.info("Supertonic ONNX CPU: %s", onnx_path.name)
            return ort.InferenceSession(
                str(onnx_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

        paths = (
            model_dir / DP_ONNX_REL_PATH,
            model_dir / TEXT_ENC_ONNX_REL_PATH,
            model_dir / VECTOR_EST_ONNX_REL_PATH,
            model_dir / VOCODER_ONNX_REL_PATH,
        )
        return tuple(_open_session(p) for p in paths)

    st_loader.load_onnx_modules = load_onnx_modules_patched
    _loader_patched = True
    log.info("Patched supertonic.loader.load_onnx_modules for GPU-first ONNX load")


def configure_supertonic_onnx(*, use_gpu: bool | None = None) -> list[str]:
    """Call before ``TTS()`` — patches config, loader binding, and loader function."""
    ensure_cuda_library_path()
    providers = build_onnx_providers(use_gpu=use_gpu)
    import onnxruntime as ort

    import supertonic.config as st_config

    st_config.DEFAULT_ONNX_PROVIDERS = providers

    # Import loader only after config patch; sync loader's imported name too.
    import supertonic.loader as st_loader

    st_loader.DEFAULT_ONNX_PROVIDERS = providers
    _patch_supertonic_loader()

    log.info(
        "Supertonic ONNX configured=%s available=%s package=%s",
        providers,
        ort.get_available_providers(),
        ort.__file__,
    )
    if use_gpu is not False and "CUDAExecutionProvider" not in ort.get_available_providers():
        log.warning(
            "SUPERTONIC_USE_GPU set but CUDAExecutionProvider missing. "
            "Use: pip uninstall onnxruntime -y && pip install onnxruntime-gpu"
        )
    return providers


def session_uses_cuda(tts_engine: Any) -> bool:
    model = getattr(tts_engine, "model", None)
    if model is None:
        return False
    for attr in ("dp_ort", "text_enc_ort", "vector_est_ort", "vocoder_ort"):
        sess = getattr(model, attr, None)
        if sess and sess.get_providers()[0] == "CUDAExecutionProvider":
            return True
    return False


def log_supertonic_session_providers(tts_engine: Any) -> None:
    """Log which ONNX provider each Supertonic session uses."""
    model = getattr(tts_engine, "model", None)
    if model is None:
        log.warning("Supertonic engine has no model attribute")
        return
    for attr in ("dp_ort", "text_enc_ort", "vector_est_ort", "vocoder_ort"):
        sess = getattr(model, attr, None)
        if sess is None:
            continue
        log.info("Supertonic %s: %s", attr, sess.get_providers())


def log_session_providers(tts_engine: Any) -> None:
    """Print which provider each ONNX session uses (demo CLI)."""
    model = getattr(tts_engine, "model", None)
    if model is None:
        print("No model on TTS engine")
        return
    for attr in ("dp_ort", "text_enc_ort", "vector_est_ort", "vocoder_ort"):
        sess = getattr(model, attr, None)
        if sess is None:
            continue
        print(f"  {attr}: {sess.get_providers()}")
    if not session_uses_cuda(tts_engine):
        print("  >>> Still on CPU — see CUDA diagnosis above.")
