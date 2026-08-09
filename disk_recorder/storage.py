"""Folder scanning, WAV writing and original deletion.

Source files are discovered recursively. Files whose stem contains the
re-record marker (``_R_``) are *outputs* and are never treated as sources,
which keeps re-runs idempotent (originals are deleted only once **every**
configured input device has saved its capture, so a second run finds nothing
left to do).
"""

from __future__ import annotations

import logging
import os
import wave
from pathlib import Path

import numpy as np

from .config import Settings

logger = logging.getLogger("disk_recorder.storage")


def find_sources(root: str, settings: Settings) -> list[Path]:
    """Recursively list audio files to re-record, excluding ``*_R_*`` outputs."""
    root_path = Path(root)
    exts = {ext.lower() for ext in settings.audio_exts}
    sources: list[Path] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        if settings.rerecord_marker in path.stem:
            continue
        sources.append(path)
    return sources


def rerecord_path(source: Path, prefix: str, settings: Settings) -> Path:
    """Return the ``<original_stem>_R_<prefix>.wav`` destination for a capture.

    ``prefix`` is the slot's manually entered device name; it identifies which
    input device produced the file.
    """
    return source.with_name(f"{source.stem}{settings.rerecord_marker}{prefix}.wav")


def save_wav(path: Path, pcm: np.ndarray, settings: Settings) -> None:
    """Write mono PCM16 to ``path`` (overwriting any prior output)."""
    pcm16 = np.asarray(pcm, dtype="<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(settings.channels)
        out.setsampwidth(settings.sample_width)
        out.setframerate(settings.sample_rate)
        out.writeframes(pcm16.tobytes())
    logger.info("saved %s (%u samples)", path.name, pcm16.size)


def delete_original(source: Path) -> None:
    """Remove the source file after its capture has been saved."""
    try:
        os.remove(source)
        logger.info("deleted original %s", source.name)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        logger.error("could not delete %s: %s", source, exc)
