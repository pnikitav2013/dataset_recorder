"""Audio capture sources.

A *source* exposes the synchronisation lock of the incoming stream and records
a window of mono PCM16 at the configured sample rate.

* :class:`BoardSource` — the STM32N6 microphone streamed over UART using the
  shared ``reliable_transport`` protocol. The transport resynchronises on the
  ``0xA5`` header and validates CRC; :class:`~disk_recorder.syncstate.SyncMonitor`
  turns that into an explicit lock state. Lock is maintained continuously (the
  reader thread always drains the port), so capture never loses the leading
  burst, and a CRC mismatch / sequence gap drops the lock mid-capture.
* :class:`MicSource` — a PC input device (via ``sounddevice``), offered as an
  alternative microphone. It has no framed transport, so it is always "locked".
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Reuse the firmware-shared protocol from py_recorder/ (parent directory).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliable_transport import ReliableTransport, TransportWarning  # noqa: E402

from .serial_link import SerialLink
from .syncstate import SyncMonitor, SyncState

logger = logging.getLogger("disk_recorder.sources")

ProgressCallback = Callable[[float], None]
SyncCallback = Callable[["SyncState"], None]

# Warnings that indicate real corruption (break an established lock).
_FATAL_WARNINGS = {TransportWarning.CRC, TransportWarning.SEQUENCE_GAP,
                   TransportWarning.INVALID_LENGTH}


@dataclass
class Window:
    """Outcome of one capture window."""

    pcm: np.ndarray            # int16, mono, at the source sample rate
    errors: int                # transport warnings observed during the window
    sync_lost: bool = False    # lock dropped mid-capture
    underrun: bool = False     # fewer samples than the target were captured
    warnings: list[str] = field(default_factory=list)


@dataclass
class RawCapture:
    """Raw, not-yet-sliced capture of one physical device for a window.

    A device is captured **once** per window (so two output channels can share a
    single mic stream); each consumer then derives its mono :class:`Window` from
    this via :meth:`AudioSource.extract_window`.
    """

    data: np.ndarray           # board: (n,) int16 mono; mic: (n, channels) int16
    rate: int                  # sample rate of ``data`` (mic devices may differ)
    errors: int = 0
    sync_lost: bool = False
    warnings: list[str] = field(default_factory=list)


class AudioSource:
    """Common interface for capture sources."""

    sample_rate: int

    def open(self) -> None:  # pragma: no cover - trivial
        pass

    def sync_state(self) -> SyncState:
        return SyncState.SYNCED

    def wait_until_synced(self, stop_event: threading.Event, timeout_s: float) -> bool:
        return True

    def record_raw(self, target_samples: int, guard_samples: int,
                   stop_event: Optional[threading.Event] = None,
                   progress_cb: Optional[ProgressCallback] = None,
                   sync_cb: Optional[SyncCallback] = None) -> RawCapture:
        """Capture one window of the physical device (all channels)."""
        raise NotImplementedError

    def extract_window(self, raw: RawCapture, channel: int,
                       target_samples: int) -> Window:
        """Derive a mono :class:`Window` for ``channel`` from a raw capture."""
        raise NotImplementedError

    def abort(self, force: bool = False) -> None:
        """Make an in-flight capture give up, from another thread.

        Called by the pipeline when a capture thread overruns its wall-clock
        budget — a hung device must not deadlock the whole run. ``force`` is a
        last resort for a reader that ignored the cooperative request.
        """

    def reopen(self) -> None:
        """Close and re-open the device after an aborted or failed capture."""
        self.close()
        self.open()

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class BoardSource(AudioSource):
    """Capture the STM32N6 PCM16 stream over ``reliable_transport``."""

    def __init__(self, link: SerialLink, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._link = link
        self._monitor = SyncMonitor()
        self._lock = threading.Lock()
        self._collecting = False
        self._buffer: list[bytes] = []
        self._collected = 0
        self._errors = 0
        self._warnings: list[str] = []
        # A single long-lived transport keeps lock across files; it self-resyncs.
        self._transport = ReliableTransport(
            write=lambda data: len(data),       # RX only; nothing is transmitted
            on_message=self._on_message,
            on_warning=self._on_warning,
        )

    def open(self) -> None:
        self._link.set_consumer(self._feed)
        self._link.set_disconnect_callback(self._monitor.note_disconnected)
        self._monitor.reset()
        self._link.open()

    def _feed(self, data: bytes) -> None:
        # Called from the serial reader thread; transport dispatches messages.
        self._transport.process(data)

    def _on_message(self, sequence: int, payload: bytes) -> None:
        self._monitor.note_message()
        with self._lock:
            if not self._collecting:
                return
            if len(payload) % 2 != 0:
                self._errors += 1
                self._warnings.append(f"odd payload seq={sequence}")
                return
            self._buffer.append(payload)
            self._collected += len(payload) // 2

    def _on_warning(self, warning: TransportWarning, sequence: int, detail: str) -> None:
        self._monitor.note_warning(fatal=warning in _FATAL_WARNINGS)
        with self._lock:
            if self._collecting:
                self._errors += 1
                self._warnings.append(f"{warning.value} seq={sequence} {detail}".strip())

    def sync_state(self) -> SyncState:
        return self._monitor.current()

    def wait_until_synced(self, stop_event: threading.Event, timeout_s: float) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if stop_event is not None and stop_event.is_set():
                return False
            if self._monitor.current() == SyncState.SYNCED:
                return True
            time.sleep(0.05)
        return self._monitor.current() == SyncState.SYNCED

    def record_raw(self, target_samples: int, guard_samples: int,
                   stop_event: Optional[threading.Event] = None,
                   progress_cb: Optional[ProgressCallback] = None,
                   sync_cb: Optional[SyncCallback] = None) -> RawCapture:
        need = target_samples + guard_samples
        with self._lock:
            self._buffer = []
            self._collected = 0
            self._errors = 0
            self._warnings = []
            self._collecting = True

        # Generous wall-clock cap so a stalled stream cannot block forever.
        cap_seconds = need / self.sample_rate + 3.0
        start = time.monotonic()
        sync_lost = False
        try:
            while True:
                with self._lock:
                    collected = self._collected
                state = self._monitor.current()
                if sync_cb is not None:
                    sync_cb(state)
                if collected >= need:
                    break
                if stop_event is not None and stop_event.is_set():
                    break
                if state == SyncState.LOST:
                    sync_lost = True
                    break
                if time.monotonic() - start > cap_seconds:
                    break
                if progress_cb is not None and need > 0:
                    progress_cb(min(1.0, collected / need))
                time.sleep(0.02)
        finally:
            with self._lock:
                self._collecting = False
                data = b"".join(self._buffer)
                errors = self._errors
                warnings = list(self._warnings)

        pcm = np.frombuffer(data, dtype=np.int16)
        if progress_cb is not None:
            progress_cb(1.0)
        return RawCapture(data=pcm, rate=self.sample_rate, errors=errors,
                          sync_lost=sync_lost, warnings=warnings)

    def extract_window(self, raw: RawCapture, channel: int,
                       target_samples: int) -> Window:
        # The board stream is mono; the channel index is irrelevant.
        pcm = raw.data
        return Window(pcm=pcm, errors=raw.errors, sync_lost=raw.sync_lost,
                      underrun=pcm.size < target_samples, warnings=raw.warnings)

    def close(self) -> None:
        self._link.close()


class MicSource(AudioSource):
    """Capture from a PC input device via ``sounddevice`` (always 'locked').

    Uses a *dedicated* :class:`sounddevice.InputStream` (not the module-global
    ``sd.rec``), so several PC microphones can record concurrently without
    clobbering each other's global stream context.

    Raw ALSA ``hw:`` inputs (USB mics, mic arrays) only accept their native
    rates/channel counts and reject a 16 kHz mono request, so capture is
    performed at a device-supported ``(rate, channels)`` and then downmixed to
    mono and resampled down to :attr:`sample_rate` (mirroring
    :class:`playback.Player`, which resamples *up* for playback). The accepted
    combination is remembered so the probe runs only once.
    """

    #: Wall-clock slack beyond the theoretical capture duration before a read is
    #: considered wedged (suspended USB device, stalled driver).
    _READ_GRACE_S = 5.0

    def __init__(self, device_index: Optional[int], sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._device = device_index
        self._capture_rate: Optional[int] = None
        self._capture_channels: Optional[int] = None
        self._stream = None   # kept open across all recordings in a session
        # Serialises abort/close against each other so PortAudio never gets an
        # abort on a stream that another thread is in the middle of closing.
        self._io_lock = threading.Lock()
        # Set by abort() and polled by the read loop. ``Pa_AbortStream`` alone
        # is not enough: the loop reads in ~20 ms blocks, so an aborted stream
        # just keeps returning and the capture would run to its full length.
        self._abort_requested = threading.Event()

    def _candidate_rates(self) -> list[int]:
        import sounddevice as sd

        rates: list[int] = []
        try:
            info = sd.query_devices(self._device, "input")
            default_rate = int(round(info.get("default_samplerate", 0)))
            if default_rate:
                rates.append(default_rate)
        except Exception:  # pragma: no cover - hardware dependent
            pass
        for rate in (self.sample_rate, 48000, 44100, 32000, 16000):
            if rate not in rates:
                rates.append(rate)
        return rates

    def _candidate_channels(self) -> list[int]:
        import sounddevice as sd

        max_in = 1
        try:
            info = sd.query_devices(self._device, "input")
            max_in = int(info.get("max_input_channels", 1))
        except Exception:  # pragma: no cover - hardware dependent
            pass
        # Capture the device's *full* channel count so any channel can later be
        # sliced out (one open serves several output slots on the same mic).
        # Always fall back to mono so the open never fails outright on a device
        # that only accepts a 1-channel stream.
        return [max_in, 1] if max_in > 1 else [1]

    def _open_stream(self):
        """Open and start a dedicated InputStream, probing accepted settings.

        Raw ``hw:`` mic arrays reject both odd rates and a mono request, so we
        try the configured rate then native rates, and 1 then the device's full
        channel count. Returns ``(stream, rate, channels)`` with the stream
        already started.
        """
        import sounddevice as sd

        if self._capture_rate is not None and self._capture_channels is not None:
            combos = [(self._capture_rate, self._capture_channels)]
        else:
            combos = [(r, c) for r in self._candidate_rates()
                      for c in self._candidate_channels()]
        last_exc: Exception | None = None
        for rate, channels in combos:
            try:
                stream = sd.InputStream(samplerate=rate, channels=channels,
                                        dtype="int16", device=self._device)
                stream.start()
                self._capture_rate, self._capture_channels = rate, channels
                if rate != self.sample_rate or channels != 1:
                    logger.info("input device %s capturing at %u Hz x%d ch "
                                "(→ mono %u Hz, per-slot channel slice)",
                                self._device, rate, channels, self.sample_rate)
                return stream, rate, channels
            except sd.PortAudioError as exc:
                last_exc = exc
                self._capture_rate = self._capture_channels = None
        raise RuntimeError(
            f"input device {self._device} accepts none of the tried "
            f"rate/channel combinations") from last_exc

    def _resample_to_target(self, pcm: np.ndarray, src_rate: int) -> np.ndarray:
        if src_rate == self.sample_rate or pcm.size == 0:
            return pcm
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(src_rate), int(self.sample_rate))
        up = self.sample_rate // divisor
        down = src_rate // divisor
        out = resample_poly(pcm.astype(np.float32), up, down)
        return np.clip(np.round(out), -32768, 32767).astype(np.int16)

    def open(self) -> None:
        """Open the capture stream once; kept alive across all recordings to
        avoid rapid ALSA PCM cycling that causes PortAudio to segfault during
        the Configure phase on quick re-open."""
        stream, _rate, _ch = self._open_stream()
        self._stream = stream

    def record_raw(self, target_samples: int, guard_samples: int,
                   stop_event: Optional[threading.Event] = None,
                   progress_cb: Optional[ProgressCallback] = None,
                   sync_cb: Optional[SyncCallback] = None) -> RawCapture:
        if sync_cb is not None:
            sync_cb(SyncState.SYNCED)
        out_frames = target_samples + guard_samples
        self._abort_requested.clear()

        # Lazily open on first call if open() was never called.
        if self._stream is None:
            self.open()

        stream = self._stream
        capture_rate = self._capture_rate
        channels = self._capture_channels

        need = int(round(out_frames * capture_rate / self.sample_rate))
        block = max(1, capture_rate // 50)   # ~20 ms read granularity

        # Flush samples that accumulated while the pipeline was between
        # recordings; we only want fresh audio for this window.
        try:
            avail = stream.read_available
            if avail > 0:
                stream.read(avail)
        except Exception:  # pragma: no cover - hardware dependent
            pass

        chunks: list[np.ndarray] = []
        got = 0
        overflows = 0
        read_exc: Optional[Exception] = None
        # Generous wall-clock cap so a suspended USB device or a stalled driver
        # cannot block this thread — and, through the pipeline's join, the whole
        # run — indefinitely. Mirrors the cap BoardSource already applies.
        deadline = time.monotonic() + need / capture_rate + self._READ_GRACE_S
        timed_out = False
        aborted = False
        try:
            while got < need:
                if stop_event is not None and stop_event.is_set():
                    break
                if self._abort_requested.is_set():
                    aborted = True
                    break
                if time.monotonic() > deadline:
                    timed_out = True
                    logger.error("input device %s: capture timed out with %d/%d "
                                 "frames — device stalled", self._device, got, need)
                    break
                to_read = min(block, need - got)
                data, overflowed = stream.read(to_read)
                # An input overflow means PortAudio dropped samples → a
                # discontinuity that shows up as a broadband "click" (vertical
                # line) in the spectrogram. Count them so the source of such
                # artefacts (xrun vs. on-device DSP) is observable.
                if overflowed:
                    overflows += 1
                chunks.append(np.asarray(data, dtype=np.int16).copy())
                got += data.shape[0]
                if progress_cb is not None:
                    progress_cb(min(1.0, got / need))
        except Exception as exc:  # pragma: no cover - hardware dependent
            # Stream died mid-read (or was aborted by the pipeline watchdog).
            # Mark it dead so the next call reopens.
            read_exc = exc
            self.close()
        finally:
            if progress_cb is not None:
                progress_cb(1.0)

        if read_exc is not None:
            raise read_exc
        if aborted:
            # The pipeline watchdog gave up on this capture; drop the stream so
            # the next attempt starts from a freshly opened device.
            self.close()
            raise RuntimeError(f"capture on input device {self._device} aborted")
        if timed_out:
            # The stream is wedged, not merely slow: drop it so the next attempt
            # starts from a freshly opened device.
            self.close()
            raise TimeoutError(
                f"input device {self._device} delivered {got}/{need} frames "
                f"before the capture deadline")

        warnings: list[str] = []
        if overflows:
            warnings.append(f"input overflow x{overflows} (dropped samples → clicks)")
            logger.warning("input device %s: %d overflow(s) during capture — "
                           "expect broadband clicks in the spectrogram",
                           self._device, overflows)

        captured = (np.concatenate(chunks) if chunks
                    else np.zeros((0, channels), dtype=np.int16))
        return RawCapture(data=captured, rate=capture_rate,
                          errors=overflows, warnings=warnings)

    def extract_window(self, raw: RawCapture, channel: int,
                       target_samples: int) -> Window:
        """Slice ``channel`` from a raw multichannel capture, → mono @ 16 kHz.

        ``channel`` < 0 (or out of range) downmixes all channels; otherwise the
        single requested channel is taken (e.g. one XVF3800 beam per slot).
        """
        captured = raw.data
        if captured.ndim == 2 and captured.shape[1] > 1:
            if 0 <= channel < captured.shape[1]:
                captured = captured[:, channel]
            else:
                # Downmix to mono in int32 to avoid int16 overflow on the sum.
                captured = captured.astype(np.int32).mean(axis=1).astype(np.int16)
        mono = np.ascontiguousarray(captured).reshape(-1)
        pcm = self._resample_to_target(mono, raw.rate)
        return Window(pcm=pcm, errors=raw.errors, sync_lost=raw.sync_lost,
                      underrun=pcm.size < target_samples, warnings=raw.warnings)

    def abort(self, force: bool = False) -> None:
        """Make an in-flight capture give up, from another thread.

        The cooperative flag is the mechanism that actually works: the read
        loop polls it between ~20 ms blocks and unwinds in milliseconds.

        ``Pa_AbortStream`` is deliberately **not** used for that, because it
        makes things worse — it stops the stream while a ``read()`` is pending,
        so the reader then waits forever for frames that will never arrive
        (measured: cooperative flag unblocks in 9 ms; adding the PortAudio
        abort wedged the reader indefinitely). It is only issued via ``force``,
        for a reader that is genuinely blocked inside the driver and therefore
        never reaches the flag check — at that point the stream is unusable
        anyway and there is nothing left to lose.
        """
        self._abort_requested.set()
        if not force:
            logger.warning("asking capture on input device %s to stop", self._device)
            return
        with self._io_lock:
            stream = self._stream
            if stream is None:
                return
            logger.error("capture on input device %s ignored the stop request — "
                         "forcing PortAudio to abort the stream", self._device)
            try:
                stream.abort()
            except Exception:  # pragma: no cover - hardware dependent
                pass

    def reopen(self) -> None:
        self._abort_requested.clear()
        super().reopen()

    def close(self) -> None:
        with self._io_lock:
            stream, self._stream = self._stream, None
            self._capture_rate = self._capture_channels = None
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:  # pragma: no cover - hardware dependent
                    pass
