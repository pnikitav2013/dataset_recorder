"""Thread-safe session state shared between the pipeline thread and the GUI.

The pipeline runs in a worker thread and only *writes* through the setter
methods here; the Tkinter window only *reads* through :meth:`snapshot`. All
access is guarded by a single lock so neither side ever sees a half-updated
record. Matplotlib figures are produced with the object API (never pyplot) so
they can be created off the main thread and drawn by the GUI thread.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field, replace

#: How many recent per-file durations the rolling average is computed over.
#: Bounded so a multi-day run does not accumulate an ever-growing list.
_AVG_WINDOW = 200


@dataclass(frozen=True)
class ChannelSnap:
    """Per input-device live status (one entry per enabled slot)."""

    label: str = ""
    sync_state: str = "disconnected"  # SyncState value
    progress: float = 0.0             # 0..1 progress of the current capture


@dataclass
class Snapshot:
    """Immutable view of the session for the GUI to render."""

    running: bool = False
    total: int = 0
    done: int = 0
    current_file: str = ""
    errors: int = 0
    failed: int = 0
    avg_record_s: float = 0.0
    capture_progress: float = 0.0     # 0..1 aggregate progress of the capture
    status: str = "idle"
    sync_state: str = "disconnected"  # worst SyncState across channels
    rec_state: str = "idle"           # RecState value
    figure_token: int = 0
    last_saved: str = ""
    failed_files: tuple[str, ...] = field(default_factory=tuple)
    channels: tuple[ChannelSnap, ...] = field(default_factory=tuple)


class SessionState:
    """Mutable, lock-protected session state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snap = Snapshot()
        self._panels = None                 # list[mel.Panel] | None
        self._figure_token = 0
        self._record_times: deque[float] = deque(maxlen=_AVG_WINDOW)
        self._failed_files: list[str] = []
        self._channels: list[ChannelSnap] = []

    # ----- writes (pipeline thread) -----

    def reset(self, total: int) -> None:
        with self._lock:
            self._snap = Snapshot(running=True, total=total, status="running")
            self._panels = None
            self._figure_token = 0
            self._record_times.clear()
            self._failed_files = []
            self._channels = []

    def init_channels(self, labels: list[str]) -> None:
        """Declare the active input-device channels (after sources open)."""
        with self._lock:
            self._channels = [ChannelSnap(label=label) for label in labels]
            self._snap.channels = tuple(self._channels)

    def set_channel(self, index: int, *, sync_state: str | None = None,
                    progress: float | None = None) -> None:
        with self._lock:
            if not 0 <= index < len(self._channels):
                return
            current = self._channels[index]
            self._channels[index] = replace(
                current,
                sync_state=current.sync_state if sync_state is None else sync_state,
                progress=current.progress if progress is None else max(0.0, min(1.0, progress)),
            )
            self._snap.channels = tuple(self._channels)

    def set_status(self, status: str) -> None:
        with self._lock:
            self._snap.status = status

    def set_sync_state(self, sync_state: str) -> None:
        with self._lock:
            self._snap.sync_state = sync_state

    def set_rec_state(self, rec_state: str) -> None:
        with self._lock:
            self._snap.rec_state = rec_state

    def set_running(self, running: bool) -> None:
        with self._lock:
            self._snap.running = running
            if not running and self._snap.status == "running":
                self._snap.status = "stopped"

    def set_current_file(self, name: str) -> None:
        with self._lock:
            self._snap.current_file = name
            self._snap.capture_progress = 0.0
            self._channels = [replace(c, progress=0.0) for c in self._channels]
            self._snap.channels = tuple(self._channels)

    def set_capture_progress(self, fraction: float) -> None:
        with self._lock:
            self._snap.capture_progress = max(0.0, min(1.0, fraction))

    def add_errors(self, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            self._snap.errors += count

    def record_success(self, record_seconds: float, saved_path: str,
                       panels=None) -> None:
        """Register a finished file; ``panels`` is optional.

        Spectrogram panels are throttled by the pipeline (the GUI only ever
        shows the most recent one), so ``None`` means "keep displaying the
        previous spectrogram" rather than "clear it".
        """
        with self._lock:
            self._record_times.append(record_seconds)
            self._snap.done += 1
            self._snap.avg_record_s = sum(self._record_times) / len(self._record_times)
            self._snap.last_saved = saved_path
            self._snap.capture_progress = 1.0
            if panels is not None:
                self._panels = panels
                self._figure_token += 1
                self._snap.figure_token = self._figure_token

    def record_failure(self, failed_path: str) -> None:
        with self._lock:
            self._snap.failed += 1
            self._failed_files.append(failed_path)
            self._snap.failed_files = tuple(self._failed_files)

    # ----- reads (GUI thread) -----

    def snapshot(self) -> Snapshot:
        with self._lock:
            # dataclasses are mutable; copy so the GUI never mutates live state.
            return Snapshot(**vars(self._snap))

    def panels_for(self, token: int):
        """Return ``(token, panels)`` if spectrograms newer than ``token`` exist."""
        with self._lock:
            if self._panels is not None and self._figure_token != token:
                return self._figure_token, self._panels
            return token, None
