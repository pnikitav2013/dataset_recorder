"""Orchestration as a sync-aware, multi-device state machine.

One output device plays each source file **once** while every enabled input
device (an STM32 board over UART or a PC microphone, configured per slot)
captures it **simultaneously**. For every source file the pipeline:

1. **WAIT_SYNC** — holds playback until *all* board channels are locked
   (:class:`SyncState.SYNCED`); PC-mic channels are always ready.
2. **RECORDING** — plays the reference and records one window per channel in
   parallel worker threads (no per-file flush, so the leading burst is kept).
3. If *any* channel loses lock or underruns, the whole attempt is discarded and
   retried after re-sync (**RETRY**) — outputs are written only when **every**
   channel produced a clean capture.
4. Each clean capture is aligned to the reference (cross-correlation), trimmed
   and written as ``<original>_R_<prefix>.wav`` (``<prefix>`` identifies the
   device). The original is deleted once all channels have saved.

Per-channel sync/progress and a stacked per-device spectrogram are published
through :class:`SessionState`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import mel, playback, storage, sync
from .appconfig import Schedule
from .config import Settings
from .playback import Player
from .sources import AudioSource, Window
from .state import SessionState
from .syncstate import RecState, SyncState

logger = logging.getLogger("disk_recorder.pipeline")

# Captured ahead of the target to cover the stale leading burst (~0.85 s) plus
# speaker/acoustic/UART latency, giving cross-correlation room to lock on.
_GUARD_SECONDS = 1.5
_SYNC_TIMEOUT_S = 20.0


@dataclass
class Channel:
    """One configured output: a capture source, a channel slice and a prefix.

    Several channels may share one :attr:`source` (e.g. two slots reading
    different channels of the same mic array); the source is then captured once
    per window and each channel slices its own :attr:`channel` index.
    """

    source: AudioSource
    prefix: str          # manually entered slot name, used in <orig>_R_<prefix>
    label: str           # human-readable label for the GUI / spectrogram
    channel: int = -1    # which captured channel to keep (-1 = downmix all)


class Pipeline:
    def __init__(self, settings: Settings, state: SessionState) -> None:
        self._settings = settings
        self._state = state
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._schedule = Schedule()

    def start(self, root: str, channels: list[Channel], player: Player,
              schedule: Schedule | None = None) -> None:
        if self.is_running():
            return
        self._schedule = schedule or Schedule()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(root, channels, player), name="pipeline", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ----- worker -----

    def _run(self, root: str, channels: list[Channel], player: Player) -> None:
        # Distinct physical sources (several channels may share one mic stream).
        sources = self._unique_sources(channels)
        opened: list[AudioSource] = []
        try:
            for source in sources:
                source.open()
                opened.append(source)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.error("source open failed: %s", exc)
            for source in opened:
                self._safe_close(source)
            self._state.reset(0)
            self._state.set_status(f"source error: {exc}")
            self._state.set_running(False)
            return

        files = storage.find_sources(root, self._settings)
        self._state.reset(len(files))
        self._state.init_channels([c.label for c in channels])
        logger.info("found %d source file(s) under %s for %d input device(s)",
                    len(files), root, len(channels))
        try:
            for source_file in files:
                if self._stop.is_set():
                    break
                self._wait_active(player)
                if self._stop.is_set():
                    break
                self._state.set_current_file(source_file.name)
                try:
                    self._process_file(source_file, channels, player)
                except Exception as exc:  # one bad file must not kill the run
                    logger.exception("unexpected error on %s: %s", source_file.name, exc)
                    self._state.add_errors(1)
                    self._state.record_failure(str(source_file))
        finally:
            self._safe_stop(player)
            for source in sources:
                self._safe_close(source)
            self._state.set_rec_state(RecState.STOPPED.value if self._stop.is_set()
                                      else RecState.DONE.value)
            self._state.set_running(False)
            self._state.set_status("stopped" if self._stop.is_set() else "done")

    def _process_file(self, source_file: Path, channels: list[Channel],
                      player: Player) -> None:
        settings = self._settings
        try:
            reference = playback.load(str(source_file))
        except Exception as exc:
            logger.error("cannot load %s: %s", source_file, exc)
            self._state.record_failure(str(source_file))
            return

        reference_16k = playback.reference_16k(reference, settings.sample_rate)
        target_samples = round(reference.duration_s * settings.sample_rate)
        guard_samples = round(_GUARD_SECONDS * settings.sample_rate)

        for attempt in range(1, settings.max_retries + 1):
            if self._stop.is_set():
                return

            # 1) WAIT_SYNC: do not play until every channel is locked.
            self._state.set_rec_state(RecState.WAIT_SYNC.value)
            if not self._wait_all_synced(channels, _SYNC_TIMEOUT_S):
                logger.warning("%s attempt %d/%d: not all channels synced within %.0fs",
                               source_file.name, attempt, settings.max_retries, _SYNC_TIMEOUT_S)
                continue

            # 2) RECORDING: play once, capture every channel in parallel.
            self._state.set_rec_state(RecState.RECORDING.value)
            started = time.monotonic()
            player.play(reference)
            windows = self._capture_all(channels, target_samples, guard_samples)
            self._safe_stop(player)
            for i, channel in enumerate(channels):
                self._state.set_channel(i, sync_state=channel.source.sync_state().value)
            self._state.set_sync_state(self._worst_sync(channels).value)

            # 3) Any channel lost lock / underran → discard the whole attempt.
            errors = sum(w.errors for w in windows)
            lost = [c.label for c, w in zip(channels, windows) if w.sync_lost]
            short = [c.label for c, w in zip(channels, windows) if w.underrun]
            if lost or short:
                self._state.set_rec_state(RecState.RETRY.value)
                self._state.add_errors(max(errors, 1 if lost else 0))
                logger.warning("%s attempt %d/%d: discarding (lost=%s underrun=%s) — retrying all",
                               source_file.name, attempt, settings.max_retries, lost, short)
                continue
            self._state.add_errors(errors)

            # 4) Align every channel; a too-short aligned clip retries all.
            aligned = [sync.align(w.pcm, reference_16k, settings.sample_rate,
                                  target_samples, settings.extra_samples)
                       for w in windows]
            if any(a.underrun for a in aligned):
                logger.warning("%s attempt %d/%d: an aligned clip too short — retrying all",
                               source_file.name, attempt, settings.max_retries)
                continue
            for channel, a in zip(channels, aligned):
                if a.correlation < settings.min_correlation:
                    logger.warning("%s [%s]: weak alignment corr=%.3f (saving anyway)",
                                   source_file.name, channel.prefix, a.correlation)

            # 5) All channels clean → save every output, then delete the original.
            saved: list[str] = []
            for channel, a in zip(channels, aligned):
                destination = storage.rerecord_path(source_file, channel.prefix, settings)
                storage.save_wav(destination, a.pcm, settings)
                saved.append(destination.name)
            storage.delete_original(source_file)

            elapsed = time.monotonic() - started
            figure = mel.make_stacked_figure(
                [(c.label, a.pcm) for c, a in zip(channels, aligned)], settings.sample_rate)
            self._state.record_success(elapsed, ", ".join(saved), figure)
            logger.info("%s -> %s (%.2fs)", source_file.name, ", ".join(saved), elapsed)
            return

        logger.error("%s failed after %d attempts — keeping original",
                     source_file.name, settings.max_retries)
        self._state.record_failure(str(source_file))

    def _wait_active(self, player: Player) -> None:
        """Block while the configured working-hours window is closed.

        Stops any playback, marks the run PAUSED and polls every 15 s until the
        clock re-enters the active window (or the user stops). A no-op when the
        schedule is disabled.
        """
        if self._schedule.is_active():
            return
        self._safe_stop(player)
        self._state.set_rec_state(RecState.PAUSED.value)
        logger.info("outside working hours (%s–%s) — pausing until %s",
                    self._schedule.start, self._schedule.end, self._schedule.start)
        while not self._stop.is_set() and not self._schedule.is_active():
            self._state.set_status(
                f"paused — outside working hours ({self._schedule.start}–{self._schedule.end}); "
                f"resuming at {self._schedule.start}")
            self._stop.wait(15.0)
        if not self._stop.is_set():
            logger.info("inside working hours again — resuming")
            self._state.set_status("running")

    # ----- helpers -----

    def _capture_all(self, channels: list[Channel], target_samples: int,
                     guard_samples: int) -> list[Window]:
        """Capture each physical source once and slice a window per channel.

        Channels sharing a source (same mic, different channel) are captured by
        a single thread/stream and then sliced, so a raw ``hw:`` device is never
        opened twice.
        """
        results: list[Window | None] = [None] * len(channels)
        groups: dict[int, list[int]] = {}
        for idx, channel in enumerate(channels):
            groups.setdefault(id(channel.source), []).append(idx)

        def worker(idxs: list[int]) -> None:
            source = channels[idxs[0]].source

            def progress(f: float) -> None:
                for i in idxs:
                    self._state.set_channel(i, progress=f)

            def sync(st) -> None:
                for i in idxs:
                    self._state.set_channel(i, sync_state=st.value)

            try:
                raw = source.record_raw(target_samples, guard_samples,
                                        stop_event=self._stop,
                                        progress_cb=progress, sync_cb=sync)
                for i in idxs:
                    results[i] = source.extract_window(
                        raw, channels[i].channel, target_samples)
            except Exception as exc:  # device error → clean retry, not a dead thread
                logger.error("capture failed on [%s]: %s",
                             channels[idxs[0]].prefix, exc)
                for i in idxs:
                    results[i] = Window(pcm=_empty_pcm(), errors=1, underrun=True)

        threads = [threading.Thread(target=worker, args=(idxs,),
                                    name=f"capture-{channels[idxs[0]].prefix}", daemon=True)
                   for idxs in groups.values()]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self._state.set_capture_progress(1.0)
        return [w if w is not None else Window(pcm=_empty_pcm(), errors=1, underrun=True)
                for w in results]

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
            self._state.set_sync_state(self._worst_sync(channels).value)
            if synced:
                return True
            time.sleep(0.1)
        return all(c.source.sync_state() == SyncState.SYNCED for c in channels)

    @staticmethod
    def _worst_sync(channels: list[Channel]) -> SyncState:
        """Return the least-locked channel state (drives the global indicator)."""
        order = [SyncState.DISCONNECTED, SyncState.LOST, SyncState.SEARCHING, SyncState.SYNCED]
        states = [c.source.sync_state() for c in channels]
        if not states:
            return SyncState.DISCONNECTED
        return min(states, key=lambda s: order.index(s) if s in order else 0)

    @staticmethod
    def _unique_sources(channels: list[Channel]) -> list[AudioSource]:
        sources: list[AudioSource] = []
        seen: set[int] = set()
        for channel in channels:
            if id(channel.source) not in seen:
                seen.add(id(channel.source))
                sources.append(channel.source)
        return sources

    @staticmethod
    def _safe_close(source: AudioSource) -> None:
        try:
            source.close()
        except Exception:  # pragma: no cover
            pass

    @staticmethod
    def _safe_stop(player: Player) -> None:
        try:
            player.stop()
        except Exception:  # pragma: no cover
            pass


def _empty_pcm():
    import numpy as np

    return np.zeros(0, dtype=np.int16)
