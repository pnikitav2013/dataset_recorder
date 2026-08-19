"""Continuous *noise* capture: record the inputs with no playback at all.

The re-record :class:`~disk_recorder.pipeline.Pipeline` needs a reference file
to play and align to. Noise recording is the opposite mode: nothing is played,
nothing is aligned and no source folder is scanned — every enabled input slot
simply records the room continuously, and the stream is cut into fixed-length
chunks (5 minutes by default) written into a **separate output folder**:

    <noise folder>/<prefix>/<prefix>_YYYYmmdd_HHMMSS.wav

Each physical device is captured **once** per chunk (slots sharing one mic
array split out their own channel afterwards), exactly as in the re-record
pipeline, so a raw ``hw:`` device is never opened twice.

Chunks are back-to-back: the recorder captures a whole chunk, then writes it
while the device is idle, so there is a sub-second seam between consecutive
files. That is deliberate — a chunk is a self-contained WAV rather than a slice
of one endless stream, which is what makes a multi-day noise session survivable
(a crash costs at most one chunk).

The optional working-hours :class:`~disk_recorder.appconfig.Schedule` applies
here too: outside the window the recorder pauses instead of filling the disk
overnight.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from pathlib import Path

from . import diag, mel, storage
from .appconfig import Schedule
from .config import Settings
from .pipeline import Channel
from .sources import AudioSource, Window
from .state import SessionState
from .syncstate import RecState, SyncState

logger = logging.getLogger("disk_recorder.noise")

#: Wall-clock slack on top of a chunk's duration before a capture thread is
#: considered wedged (see ``pipeline._CAPTURE_GRACE_S``).
_CAPTURE_GRACE_S = 15.0
_ABORT_POLL_S = 1.5
_ABORT_JOIN_S = 5.0

#: How long board channels are given to lock before a chunk starts anyway.
_SYNC_TIMEOUT_S = 20.0

#: Only the tail of a chunk is turned into a spectrogram — a 5-minute log-mel
#: is both unreadable and expensive to compute.
_MEL_TAIL_S = 10.0

#: Period of the progress/resource heartbeat written to the log.
_HEARTBEAT_S = 60.0

#: Default chunk length in minutes.
DEFAULT_CHUNK_MINUTES = 5.0


class NoiseRecorder:
    """Record every enabled input continuously into fixed-length chunks."""

    def __init__(self, settings: Settings, state: SessionState) -> None:
        self._settings = settings
        self._state = state
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._schedule = Schedule()
        self._stop_reason: str | None = None

    # ----- control (same surface as Pipeline, so the GUI can treat them alike) -----

    def start(self, folder: str, channels: list[Channel], chunk_minutes: float,
              schedule: Schedule | None = None) -> None:
        if self.is_running():
            return
        self._schedule = schedule or Schedule()
        self._stop.clear()
        self._finished.clear()
        self._stop_reason = None
        self._thread = threading.Thread(
            target=self._run, args=(folder, channels, chunk_minutes),
            name="noise", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask for the run to end; the chunk in flight is kept.

        Both sources poll ``stop_event`` between reads, so a 5-minute capture
        unwinds in milliseconds and returns what it has — that partial chunk is
        still written, so pressing Stop never throws away the audio recorded
        since the last file.
        """
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ----- worker -----

    def _run(self, folder: str, channels: list[Channel],
             chunk_minutes: float) -> None:
        sources = _unique_sources(channels)
        opened: list[AudioSource] = []
        try:
            for source in sources:
                source.open()
                opened.append(source)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.error("device open failed: %s", exc)
            for source in opened:
                _safe_close(source)
            self._state.reset(0)
            self._state.set_status(f"device error: {exc}")
            self._state.set_running(False)
            return

        chunk_samples = max(1, round(chunk_minutes * 60.0 * self._settings.sample_rate))
        root = Path(folder)
        self._state.reset(0)
        self._state.init_channels([c.label for c in channels])
        self._state.set_status(f"noise recording — {chunk_minutes:g} min per file")
        logger.info("noise recording %d channel(s) into %s, %g min per file",
                    len(channels), root, chunk_minutes)

        heartbeat = threading.Thread(target=self._heartbeat, name="noise-heartbeat",
                                     daemon=True)
        heartbeat.start()
        try:
            while not self._stop.is_set():
                self._wait_active()
                if self._stop.is_set():
                    break
                try:
                    self._record_chunk(root, channels, chunk_samples)
                except Exception as exc:  # one bad chunk must not kill the run
                    logger.exception("unexpected error during a chunk: %s", exc)
                    self._state.add_errors(1)
                    self._recover(channels)
        finally:
            self._finished.set()
            for source in sources:
                _safe_close(source)
            self._state.set_rec_state(RecState.STOPPED.value)
            self._state.set_running(False)
            self._state.set_status(self._stop_reason or "stopped")
            logger.info("noise run finished (%s) — %s",
                        self._stop_reason or "stopped",
                        diag.format_stats(diag.process_stats()))

    def _heartbeat(self) -> None:
        while not self._finished.wait(_HEARTBEAT_S):
            snap = self._state.snapshot()
            logger.info("heartbeat: %s %d chunk(s) written, %d error(s) | %s",
                        snap.rec_state, snap.done, snap.errors,
                        diag.format_stats(diag.process_stats()))

    def _record_chunk(self, root: Path, channels: list[Channel],
                      chunk_samples: int) -> None:
        """Capture one chunk on every channel and write one WAV per channel."""
        # Board channels get a chance to lock first; unlike the re-record
        # pipeline a missed lock does not invalidate anything, so after the
        # timeout the chunk is recorded anyway and the errors are counted.
        self._state.set_rec_state(RecState.WAIT_SYNC.value)
        self._wait_all_synced(channels, _SYNC_TIMEOUT_S)
        if self._stop.is_set():
            return

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._state.set_current_file(f"{stamp} (in progress)")
        self._state.set_rec_state(RecState.RECORDING.value)
        started = time.monotonic()
        windows = self._capture_all(channels, chunk_samples)
        elapsed = time.monotonic() - started
        for i, channel in enumerate(channels):
            self._state.set_channel(i, sync_state=channel.source.sync_state().value)
        self._state.set_sync_state(_worst_sync(channels).value)

        errors = sum(w.errors for w in windows)
        self._state.add_errors(errors)

        saved: list[str] = []
        for channel, window in zip(channels, windows):
            if window.pcm.size == 0:
                logger.error("[%s]: empty capture — nothing to write for %s",
                             channel.label, stamp)
                self._state.record_failure(f"{channel.prefix} {stamp}")
                continue
            destination = storage.noise_path(root, channel.prefix, stamp)
            storage.save_wav(destination, window.pcm, self._settings)
            saved.append(destination.name)
            if window.underrun:
                # A short chunk is kept (it is still valid audio) but is worth
                # a line in the log: it means the device did not keep up.
                logger.warning("[%s]: chunk %s is short (%.1fs of %.1fs)",
                               channel.label, stamp,
                               window.pcm.size / self._settings.sample_rate,
                               chunk_samples / self._settings.sample_rate)
        if not saved:
            return

        panels = mel.compute_panels(
            [(c.label, _tail(w.pcm, self._settings.sample_rate))
             for c, w in zip(channels, windows) if w.pcm.size],
            self._settings.sample_rate)
        self._state.set_current_file(stamp)
        self._state.record_success(elapsed, ", ".join(saved), panels)
        logger.info("chunk %s -> %s (%.1fs)", stamp, ", ".join(saved), elapsed)

    def _capture_all(self, channels: list[Channel],
                     chunk_samples: int) -> list[Window]:
        """Capture one chunk per physical source, sliced into per-channel windows."""
        results: list[Window | None] = [None] * len(channels)
        groups: dict[int, list[int]] = {}
        for idx, channel in enumerate(channels):
            groups.setdefault(id(channel.source), []).append(idx)

        def worker(idxs: list[int]) -> None:
            source = channels[idxs[0]].source

            def progress(fraction: float) -> None:
                for i in idxs:
                    self._state.set_channel(i, progress=fraction)
                self._state.set_capture_progress(fraction)

            def sync(state) -> None:
                for i in idxs:
                    self._state.set_channel(i, sync_state=state.value)

            try:
                raw = source.record_raw(chunk_samples, 0, stop_event=self._stop,
                                        progress_cb=progress, sync_cb=sync)
                for i in idxs:
                    results[i] = source.extract_window(
                        raw, channels[i].channel, chunk_samples)
            except Exception as exc:  # device error → skip this chunk, keep running
                logger.error("capture failed on [%s]: %s", channels[idxs[0]].label, exc)
                for i in idxs:
                    results[i] = Window(pcm=_empty_pcm(), errors=1, underrun=True)

        jobs = [(idxs, threading.Thread(
            target=worker, args=(idxs,),
            name=f"noise-{channels[idxs[0]].prefix}", daemon=True))
            for idxs in groups.values()]
        for _idxs, thread in jobs:
            thread.start()

        budget = chunk_samples / self._settings.sample_rate + _CAPTURE_GRACE_S
        deadline = time.monotonic() + budget
        for _idxs, thread in jobs:
            thread.join(timeout=max(0.1, deadline - time.monotonic()))

        stuck = [(idxs, thread) for idxs, thread in jobs if thread.is_alive()]
        if stuck:
            for idxs, _thread in stuck:
                channel = channels[idxs[0]]
                logger.error("capture on [%s] exceeded %.1fs — asking it to stop",
                             channel.label, budget)
                _safe_abort(channel.source, channel.label, force=False)
            for _idxs, thread in stuck:
                thread.join(timeout=_ABORT_POLL_S)
            unresponsive = [(idxs, t) for idxs, t in stuck if t.is_alive()]
            for idxs, _thread in unresponsive:
                channel = channels[idxs[0]]
                _safe_abort(channel.source, channel.label, force=True)
            for _idxs, thread in unresponsive:
                thread.join(timeout=_ABORT_JOIN_S)
            self._handle_stuck(stuck, channels)

        self._state.set_capture_progress(1.0)
        return [w if w is not None else Window(pcm=_empty_pcm(), errors=1, underrun=True)
                for w in results]

    def _handle_stuck(self, stuck, channels: list[Channel]) -> None:
        """Re-open a device whose capture overran, or stop if it is wedged."""
        for idxs, thread in stuck:
            channel = channels[idxs[0]]
            if not thread.is_alive():
                _safe_reopen(channel.source, channel.label)
                continue
            logger.critical("capture thread for [%s] did not respond to abort — "
                            "the device is wedged in its driver; stopping the run",
                            channel.label)
            self._stop_reason = (
                f"stopped: input device [{channel.label}] stopped responding")
            self._state.set_status(self._stop_reason)
            self._stop.set()

    def _recover(self, channels: list[Channel]) -> None:
        """Re-open every source after an unexpected error in the chunk loop."""
        for channel in _unique_channels(channels):
            _safe_reopen(channel.source, channel.label)

    def _wait_all_synced(self, channels: list[Channel], timeout_s: float) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if self._stop.is_set():
                return False
            synced = True
            for i, channel in enumerate(channels):
                state = channel.source.sync_state()
                self._state.set_channel(i, sync_state=state.value)
                if state != SyncState.SYNCED:
                    synced = False
            self._state.set_sync_state(_worst_sync(channels).value)
            if synced:
                return True
            self._stop.wait(0.1)
        logger.warning("not all channels locked within %.0fs — recording anyway",
                       timeout_s)
        return False

    def _wait_active(self) -> None:
        """Pause while the configured working-hours window is closed."""
        if self._schedule.is_active():
            return
        self._state.set_rec_state(RecState.PAUSED.value)
        logger.info("outside working hours (%s–%s) — pausing until %s",
                    self._schedule.start, self._schedule.end, self._schedule.start)
        while not self._stop.is_set() and not self._schedule.is_active():
            self._state.set_status(
                f"paused — outside working hours ({self._schedule.start}–"
                f"{self._schedule.end}); resuming at {self._schedule.start}")
            self._stop.wait(15.0)
        if not self._stop.is_set():
            logger.info("inside working hours again — resuming")
            self._state.set_status("noise recording")


# ----- helpers (shared shape with pipeline, kept local to avoid a cycle) -----


def _tail(pcm, sample_rate: int):
    """Last :data:`_MEL_TAIL_S` seconds of a chunk, for the spectrogram."""
    samples = int(_MEL_TAIL_S * sample_rate)
    return pcm[-samples:] if pcm.size > samples else pcm


def _unique_channels(channels: list[Channel]) -> list[Channel]:
    unique: list[Channel] = []
    seen: set[int] = set()
    for channel in channels:
        if id(channel.source) not in seen:
            seen.add(id(channel.source))
            unique.append(channel)
    return unique


def _unique_sources(channels: list[Channel]) -> list[AudioSource]:
    return [c.source for c in _unique_channels(channels)]


def _worst_sync(channels: list[Channel]) -> SyncState:
    order = [SyncState.DISCONNECTED, SyncState.LOST, SyncState.SEARCHING, SyncState.SYNCED]
    states = [c.source.sync_state() for c in channels]
    if not states:
        return SyncState.DISCONNECTED
    return min(states, key=lambda s: order.index(s) if s in order else 0)


def _safe_close(source: AudioSource) -> None:
    try:
        source.close()
    except Exception:  # pragma: no cover
        pass


def _safe_abort(source: AudioSource, label: str, force: bool) -> None:
    try:
        source.abort(force=force)
    except Exception as exc:  # pragma: no cover - hardware dependent
        logger.error("abort failed on [%s]: %s", label, exc)


def _safe_reopen(source: AudioSource, label: str) -> None:
    try:
        source.reopen()
        logger.info("re-opened [%s]", label)
    except Exception as exc:  # pragma: no cover - hardware dependent
        logger.error("re-open failed on [%s]: %s — will retry later", label, exc)


def _empty_pcm():
    import numpy as np

    return np.zeros(0, dtype=np.int16)
