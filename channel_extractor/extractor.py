"""Copy a re-recorded dataset and keep a single ``_R_<prefix>`` channel.

The extraction is three steps over a *copy* of the source dataset (the original
is never touched):

1. **filter** — delete every audio file whose stem does not end with
   ``_R_<prefix>`` (other channels and any leftover originals);
2. **rename** — strip the ``_R_<prefix>`` marker from the survivors so their
   names match the original clips (``116-288045-0024_R_fifine.wav`` →
   ``116-288045-0024.wav``);
3. **prune** — remove directories left empty by the filtering.

Non-audio files (``*.trans.txt`` transcripts, metadata) are kept verbatim.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("channel_extractor.extractor")

#: Re-record marker written by :mod:`disk_recorder.storage` (``<name>_R_<prefix>``).
MARKER = "_R_"

#: Audio extensions considered captures (matches ``disk_recorder.config.Settings``).
AUDIO_EXTS: tuple[str, ...] = (".wav", ".flac", ".ogg", ".mp3", ".aiff")


def extracted_root(source: Path, prefix: str, dest_parent: Path) -> Path:
    """Return the destination dataset path: ``<dest_parent>/<source_name>_R_<prefix>``."""
    return dest_parent / f"{source.name}{MARKER}{prefix}"


def _is_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTS


def _stripped_name(path: Path, prefix: str) -> str | None:
    """Return ``path``'s name with ``_R_<prefix>`` removed, or ``None`` if it does
    not belong to this channel."""
    suffix = f"{MARKER}{prefix}"
    if not path.stem.endswith(suffix):
        return None
    return path.stem[: -len(suffix)] + path.suffix


def extract(source: Path, prefix: str, dest_parent: Path, *, overwrite: bool = False) -> Path:
    """Extract the ``prefix`` channel of ``source`` into ``dest_parent``.

    :param source:      existing re-recorded dataset folder.
    :param prefix:      device/source prefix to keep (the part after ``_R_``).
    :param dest_parent: directory the extracted dataset is created inside.
    :param overwrite:   replace an existing extracted dataset instead of failing.
    :returns:           path of the created ``<source_name>_R_<prefix>`` dataset.
    """
    source = source.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"source dataset not found: {source}")

    target = extracted_root(source, prefix, dest_parent.resolve())
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"destination already exists: {target} (pass --overwrite)")
        logger.info("removing existing %s", target)
        shutil.rmtree(target)

    logger.info("copying %s -> %s", source, target)
    shutil.copytree(source, target)

    kept = 0
    removed = 0
    for path in sorted(target.rglob("*")):
        if not _is_audio(path):
            continue
        new_name = _stripped_name(path, prefix)
        if new_name is None:
            path.unlink()
            removed += 1
            logger.debug("removed other channel %s", path.name)
            continue
        renamed = path.with_name(new_name)
        path.rename(renamed)
        kept += 1
        logger.debug("kept %s -> %s", path.name, renamed.name)

    _prune_empty_dirs(target)

    logger.info("extracted '%s': kept %u, removed %u -> %s", prefix, kept, removed, target)
    if kept == 0:
        logger.warning("no '%s%s' audio found in %s — is the prefix correct?", MARKER, prefix, source)
    return target


def _prune_empty_dirs(root: Path) -> None:
    """Remove directories left empty after filtering (deepest first, keep ``root``)."""
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            logger.debug("pruned empty dir %s", path)
