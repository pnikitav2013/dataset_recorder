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

from . import diag, mel, playback, storage, sync
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

# Wall-clock slack on top of the theoretical capture duration before a capture
# thread is considered wedged. A blocking device read has no timeout of its
# own, so without this a suspended USB mic deadlocks the entire run.
_CAPTURE_GRACE_S = 8.0
#: How long a capture thread gets to notice a cooperative abort request. The
#: read loop polls between ~20 ms blocks, so this is very generous.
_ABORT_POLL_S = 1.5
#: How long it then gets to unwind after PortAudio has been forced to abort.
_ABORT_JOIN_S = 5.0

#: Consecutive failed files after which every unsynced source is re-opened,
#: instead of burning ``max_retries × _SYNC_TIMEOUT_S`` per file for days.
_RECOVER_AFTER_FAILURES = 2

#: Spectrograms are only for the operator, and the GUI shows one at a time, so
#: computing them for every file is wasted work on a fast corpus.
_MEL_MIN_INTERVAL_S = 2.5

#: Period of the progress/resource heartbeat written to the log.
_HEARTBEAT_S = 60.0


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
        self._finished = threading.Event()
        self._schedule = Schedule()
        self._last_mel_at = 0.0
        self._consecutive_failures = 0
        self._stop_reason: str | None = None

    def start(self, root: str, channels: list[Channel], player: Player,
              schedule: Schedule | None = None) -> None:
        if self.is_running():
            return
        self._schedule = schedule or Schedule()
        self._stop.clear()
        self._finished.clear()
        self._last_mel_at = 0.0
        self._consecutive_failures = 0
        self._stop_reason = None
        self._thread = threading.Thread(
            target=self._run, args=(root, channels, player), name="pipeline", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker to finish shutting the devices down."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ----- worker -----

    def _run(self, root: str, channels: list[Channel], player: Player) -> None:
        # Distinct physical sources (several channels may share one mic stream).
        sources = self._unique_sources(channels)
        opened: list[AudioSource] = []
        try:
            # The output device is opened once and stays open for the whole run
            # (see Player) — per-file open/close is what degrades system audio.
            player.open()
            if player.fallback_device is not None:
                raise RuntimeError(
                    "the configured output device could not be opened; playback "
                    "would go to another speaker, so captures would be invalid")
            for source in sources:
                source.open()
                opened.append(source)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.error("device open failed: %s", exc)
            for source in opened:
                self._safe_close(source)
            self._safe_close_player(player)
            self._state.reset(0)
            self._state.set_status(f"device error: {exc}")
            self._state.set_running(False)
            return

        files = storage.find_sources(root, self._settings)
        self._state.reset(len(files))
        self._state.init_channels([c.label for c in channels])
        logger.info("found %d source file(s) under %s for %d input device(s)",
                    len(files), root, len(channels))
        heartbeat = threading.Thread(target=self._heartbeat, name="heartbeat", daemon=True)
        heartbeat.start()
        try:
            for source_file in files:
                if self._stop.is_set():
                    break
                self._wait_active(player)
                if self._stop.is_set():
                    break
                self._state.set_current_file(source_file.name)
                try:
                    succeeded = self._process_file(source_file, channels, player)
                except Exception as exc:  # one bad file must not kill the run
                    logger.exception("unexpected error on %s: %s", source_file.name, exc)
                    self._state.add_errors(1)
                    self._state.record_failure(str(source_file))
                    succeeded = False
                self._note_outcome(succeeded, channels)
        finally:
            self._finished.set()
            self._safe_stop(player)
            self._safe_close_player(player)
            for source in sources:
                self._safe_close(source)
            self._state.set_rec_state(RecState.STOPPED.value if self._stop.is_set()
                                      else RecState.DONE.value)
            self._state.set_running(False)
            # Keep the reason a watchdog abort recorded — overwriting it with a
            # bare "stopped" is what makes an overnight failure unexplainable.
            if self._stop_reason:
                self._state.set_status(self._stop_reason)
            else:
                self._state.set_status("stopped" if self._stop.is_set() else "done")
            logger.info("run finished (%s) — %s",
                        self._stop_reason or ("stopped" if self._stop.is_set() else "done"),
                        diag.format_stats(diag.process_stats()))

    def _heartbeat(self) -> None:
        """Log progress and process resource usage until the run ends.

        On a multi-day session this is the only forensic trail there is: a
        monotonically climbing handle or RSS count shows up here hours before it
        turns into an outage.
        """
        while not self._finished.wait(_HEARTBEAT_S):
            snap = self._state.snapshot()
            logger.info("heartbeat: %s %d/%d done, %d error(s), %d failed, "
                        "avg %.2fs | %s", snap.rec_state, snap.done, snap.total,
                        snap.errors, snap.failed, snap.avg_record_s,
                        diag.format_stats(diag.process_stats()))

    def _note_outcome(self, succeeded: bool, channels: list[Channel]) -> None:
        """Recover unsynced sources after a run of consecutive failures.

        Without this, a board that dropped off the USB bus costs
        ``max_retries × _SYNC_TIMEOUT_S`` per file — 100 s of pure waiting each,
        for as many days as the run lasts.
        """
        if succeeded:
            self._consecutive_failures = 0
            return
        self._consecutive_failures += 1
        if self._consecutive_failures < _RECOVER_AFTER_FAILURES:
            return
        self._consecutive_failures = 0
        for channel in self._unique_channels(channels):
            if channel.source.sync_state() in (SyncState.SYNCED,):
                continue
            logger.warning("re-opening [%s] after %d failed file(s) (state=%s)",
                           channel.label, _RECOVER_AFTER_FAILURES,
                           channel.source.sync_state().value)
            self._safe_reopen(channel.source, channel.label)

    def _process_file(self, source_file: Path, channels: list[Channel],
                      player: Player) -> bool:
        """Re-record one file; ``True`` when every channel was saved."""
        settings = self._settings
        try:
            reference = playback.load(str(source_file))
        except Exception as exc:
            logger.error("cannot load %s: %s", source_file, exc)
            self._state.record_failure(str(source_file))
            return False

        reference_16k = playback.reference_16k(reference, settings.sample_rate)
        target_samples = round(reference.duration_s * settings.sample_rate)
        guard_samples = round(_GUARD_SECONDS * settings.sample_rate)

        for attempt in range(1, settings.max_retries + 1):
            if self._stop.is_set():
                return False

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
            panels = None
            now = time.monotonic()
            if now - self._last_mel_at >= _MEL_MIN_INTERVAL_S:
                self._last_mel_at = now
                panels = mel.compute_panels(
                    [(c.label, a.pcm) for c, a in zip(channels, aligned)],
                    settings.sample_rate)
            self._state.record_success(elapsed, ", ".join(saved), panels)
            logger.info("%s -> %s (%.2fs)", source_file.name, ", ".join(saved), elapsed)
            return True

        logger.error("%s failed after %d attempts — keeping original",
                     source_file.name, settings.max_retries)
        self._state.record_failure(str(source_file))
        return False

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

        jobs = [(idxs, threading.Thread(
            target=worker, args=(idxs,),
            name=f"capture-{channels[idxs[0]].prefix}", daemon=True))
            for idxs in groups.values()]
        for _idxs, thread in jobs:
            thread.start()

        # A device read has no timeout of its own: if a USB mic is suspended or
        # its driver stalls, the worker never returns and an unbounded join()
        # would freeze the run forever while the GUI still shows "recording".
        budget = ((target_samples + guard_samples) / self._settings.sample_rate
                  + _CAPTURE_GRACE_S)
        deadline = time.monotonic() + budget
        for _idxs, thread in jobs:
            thread.join(timeout=max(0.1, deadline - time.monotonic()))

        stuck = [(idxs, thread) for idxs, thread in jobs if thread.is_alive()]
        if stuck:
            # Escalate in two steps. The cooperative request is what normally
            # works and leaves the device in a reusable state; forcing PortAudio
            # to abort is reserved for a reader blocked inside the driver, since
            # it destroys the stream and can wedge a pending read for good.
            for idxs, _thread in stuck:
                channel = channels[idxs[0]]
                logger.error("capture on [%s] exceeded %.1fs — asking it to stop",
                             channel.label, budget)
                self._safe_abort(channel.source, channel.label, force=False)
            for _idxs, thread in stuck:
                thread.join(timeout=_ABORT_POLL_S)

            unresponsive = [(idxs, t) for idxs, t in stuck if t.is_alive()]
            for idxs, _thread in unresponsive:
                channel = channels[idxs[0]]
                self._safe_abort(channel.source, channel.label, force=True)
            for _idxs, thread in unresponsive:
                thread.join(timeout=_ABORT_JOIN_S)
            self._handle_stuck(stuck, channels)

        self._state.set_capture_progress(1.0)
        return [w if w is not None else Window(pcm=_empty_pcm(), errors=1, underrun=True)
                for w in results]

    def _handle_stuck(self, stuck: list[tuple[list[int], threading.Thread]],
                      channels: list[Channel]) -> None:
        """Recover (or give up on) sources whose capture thread overran.

        A thread that unwound after the abort leaves a closed device we can
        simply re-open. One that is *still* alive is holding the device inside
        the driver: re-opening would fail anyway, so the run is stopped loudly
        rather than left to spin producing empty captures for days.
        """
        for idxs, thread in stuck:
            channel = channels[idxs[0]]
            if not thread.is_alive():
                self._safe_reopen(channel.source, channel.label)
                continue
            logger.critical("capture thread for [%s] did not respond to abort — "
                            "the device is wedged in its driver; stopping the run",
                            channel.label)
            self._stop_reason = (
                f"stopped: input device [{channel.label}] stopped responding")
            self._state.set_status(self._stop_reason)
            self._stop.set()

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
        return [c.source for c in Pipeline._unique_channels(channels)]

    @staticmethod
    def _unique_channels(channels: list[Channel]) -> list[Channel]:
        """One channel per distinct physical source (shared mics collapse)."""
        unique: list[Channel] = []
        seen: set[int] = set()
        for channel in channels:
            if id(channel.source) not in seen:
                seen.add(id(channel.source))
                unique.append(channel)
        return unique

    @staticmethod
    def _safe_close(source: AudioSource) -> None:
        try:
            source.close()
        except Exception:  # pragma: no cover
            pass

    @staticmethod
    def _safe_abort(source: AudioSource, label: str, force: bool) -> None:
        try:
            source.abort(force=force)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.error("abort failed on [%s]: %s", label, exc)

    @staticmethod
    def _safe_reopen(source: AudioSource, label: str) -> None:
        try:
            source.reopen()
            logger.info("re-opened [%s]", label)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.error("re-open failed on [%s]: %s — will retry later", label, exc)

    @staticmethod
    def _safe_stop(player: Player) -> None:
        try:
            player.stop()
        except Exception:  # pragma: no cover
            pass

    @staticmethod
    def _safe_close_player(player: Player) -> None:
        try:
            player.close()
        except Exception:  # pragma: no cover
            pass


def _empty_pcm():
    import numpy as np

    return np.zeros(0, dtype=np.int16)
