"""
Whole-call recordings: both voices, compressed to Opus, kept on disk.

During the call Asterisk records a snoop channel (spy=both: caller + bot mixed).
When the call ends the WAV is fetched over ARI, compressed with ffmpeg to Opus
(mono, 16 kbps by default: ~60 KB per minute; the 8 kHz WAV is ~16x bigger) and
saved as

    RECORDINGS_DIR/YYYY/MM/DD/<HHMMSS>_<caller>_<session>.ogg

The path (relative to RECORDINGS_DIR) and size go into the calls table, so the
recordings page (/recordings on the bot's web server) can list and play them.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def recordings_root(configured: str) -> Path:
    path = Path(configured or "recordings")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def target_path(root: Path, *, started: datetime, caller: str | None, session_id: str) -> Path:
    safe_caller = re.sub(r"[^A-Za-z0-9_+-]", "", caller or "") or "unknown"
    day = root / f"{started:%Y}" / f"{started:%m}" / f"{started:%d}"
    return day / f"{started:%H%M%S}_{safe_caller}_{session_id}.ogg"


def _compress_sync(src: Path, dst: Path, kbps: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.ogg")
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-i", str(src),
            "-ac", "1",
            "-c:a", "libopus",
            "-b:a", f"{kbps}k",
            "-application", "voip",  # tuned for speech
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(dst)


async def compress_to_opus(src: Path, dst: Path, *, kbps: int = 16) -> int:
    """Compress a WAV to Opus; returns the new file size in bytes."""
    await asyncio.to_thread(_compress_sync, src, dst, kbps)
    return dst.stat().st_size


def temp_wav() -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        return Path(tmp.name)


def resolve_inside(root: Path, rel: str) -> Path | None:
    """Absolute path of a stored recording, refusing anything outside the root."""
    try:
        p = (root / rel).resolve()
        p.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    return p if p.is_file() else None
