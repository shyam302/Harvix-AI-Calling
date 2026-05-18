"""Shared thread pool + semaphores so multiple calls can use STT/TTS in parallel."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

_executor: ThreadPoolExecutor | None = None
_whisper_sem: threading.Semaphore | None = None
_tts_sem: threading.Semaphore | None = None
_whisper_slots: int = 1
_tts_slots: int = 2
_thread_workers: int = 4


def configure(
    *,
    whisper_max_concurrent: int,
    tts_max_concurrent: int,
    thread_workers: int | None = None,
) -> None:
    """Call once at process startup (before handling calls)."""
    global _executor, _whisper_sem, _tts_sem, _whisper_slots, _tts_slots, _thread_workers
    _whisper_slots = max(1, whisper_max_concurrent)
    _tts_slots = max(1, tts_max_concurrent)
    workers = thread_workers
    if workers is None:
        workers = max(4, _whisper_slots + _tts_slots + 2)
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
    _thread_workers = workers
    _executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="callbot-infer",
    )
    _whisper_sem = threading.Semaphore(_whisper_slots)
    _tts_sem = threading.Semaphore(_tts_slots)
    log.info(
        "Inference pool: whisper_slots=%s tts_slots=%s thread_workers=%s",
        _whisper_slots,
        _tts_slots,
        workers,
    )


def shutdown() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


def get_executor() -> ThreadPoolExecutor:
    if _executor is None:
        configure(whisper_max_concurrent=1, tts_max_concurrent=2)
    assert _executor is not None
    return _executor


def _run_with_sem(
    sem: threading.Semaphore,
    label: str,
    fn: Callable[[], T],
) -> T:
    t_wait = time.perf_counter()
    with sem:
        waited = time.perf_counter() - t_wait
        if waited >= 0.15:
            log.info("%s waited %.2fs for a free slot (other calls active)", label, waited)
        return fn()


def run_whisper(fn: Callable[[], T]) -> T:
    if _whisper_sem is None:
        return fn()
    return _run_with_sem(_whisper_sem, "Whisper STT", fn)


def run_tts(fn: Callable[[], T]) -> T:
    if _tts_sem is None:
        return fn()
    return _run_with_sem(_tts_sem, "Supertonic TTS", fn)


def pool_status() -> dict[str, int]:
    return {
        "whisper_max_concurrent": _whisper_slots,
        "tts_max_concurrent": _tts_slots,
        "thread_workers": _thread_workers,
    }
