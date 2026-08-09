"""Synchronisation / recording state machine.

The framed transport (``reliable_transport``) already resynchronises on the
``0xA5`` header and rejects frames with a bad CRC. :class:`SyncMonitor` turns
that low-level behaviour into an explicit, observable *lock state* over the
stream:

* :class:`SyncState` — are we locked onto the board's frame stream?
* :class:`RecState`  — what is the recording pipeline doing?

The board ships PCM in ~0.85 s bursts, so a short silence between bursts is
normal; only a CRC mismatch, a sequence gap or a longer-than-burst timeout
counts as a loss of lock.
"""

from __future__ import annotations

import threading
import time
from enum import Enum


class SyncState(str, Enum):
    DISCONNECTED = "disconnected"   # port closed / read error
    SEARCHING = "searching"         # open, hunting for valid contiguous frames
    SYNCED = "synced"               # locked: valid, contiguous frames flowing
    LOST = "lost"                   # was locked, then corruption/gap/timeout


class RecState(str, Enum):
    IDLE = "idle"
    WAIT_SYNC = "wait_sync"         # holding playback until the stream locks
    RECORDING = "recording"         # playing + capturing
    RETRY = "retry"                 # lost lock mid-capture; playback stopped
    PAUSED = "paused"               # outside the configured working hours
    DONE = "done"
    STOPPED = "stopped"


# Colour hints for the GUI indicators (background, foreground).
SYNC_COLORS = {
    SyncState.DISCONNECTED: ("#9e9e9e", "white"),
    SyncState.SEARCHING: ("#f9a825", "black"),
    SyncState.SYNCED: ("#2e7d32", "white"),
    SyncState.LOST: ("#c62828", "white"),
}

REC_COLORS = {
    RecState.IDLE: ("#9e9e9e", "white"),
    RecState.WAIT_SYNC: ("#f9a825", "black"),
    RecState.RECORDING: ("#1565c0", "white"),
    RecState.RETRY: ("#c62828", "white"),
    RecState.PAUSED: ("#6a1b9a", "white"),
    RecState.DONE: ("#2e7d32", "white"),
    RecState.STOPPED: ("#616161", "white"),
}


class SyncMonitor:
    """Track lock onto the board frame stream from transport callbacks.

    Fed from the serial reader thread (``note_message`` / ``note_warning``) and
    queried from the pipeline / GUI threads (``current``). All state is guarded
    by a lock.
    """

    def __init__(self, lock_frames: int = 3, burst_timeout_s: float = 1.6) -> None:
        self._lock = threading.Lock()
        self._state = SyncState.DISCONNECTED
        self._clean = 0
        self._lock_frames = lock_frames
        self._burst_timeout_s = burst_timeout_s
        self._last_message = 0.0

    def reset(self) -> None:
        with self._lock:
            self._state = SyncState.SEARCHING
            self._clean = 0
            self._last_message = time.monotonic()

    def note_message(self) -> None:
        """A valid frame was received (CRC ok, sequence in order)."""
        with self._lock:
            self._last_message = time.monotonic()
            if self._state != SyncState.SYNCED:
                self._clean += 1
                if self._clean >= self._lock_frames:
                    self._state = SyncState.SYNCED

    def note_warning(self, fatal: bool) -> None:
        """A transport warning occurred.

        ``fatal`` (CRC mismatch / sequence gap) breaks an established lock;
        non-fatal noise (resync search) only keeps us in SEARCHING.
        """
        with self._lock:
            self._clean = 0
            if fatal and self._state == SyncState.SYNCED:
                self._state = SyncState.LOST
            elif self._state == SyncState.SYNCED:
                # benign during lock; ignore
                pass
            elif self._state != SyncState.DISCONNECTED:
                self._state = SyncState.SEARCHING

    def note_disconnected(self) -> None:
        with self._lock:
            self._state = SyncState.DISCONNECTED
            self._clean = 0

    def current(self) -> SyncState:
        """Return the current state, applying the inter-burst timeout."""
        with self._lock:
            if (self._state == SyncState.SYNCED
                    and time.monotonic() - self._last_message > self._burst_timeout_s):
                self._state = SyncState.LOST
            return self._state
