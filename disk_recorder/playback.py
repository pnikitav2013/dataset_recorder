"""Reference audio preparation and non-blocking playback.

Loads an audio file, plays it on the chosen output device while the board
records, and produces a mono 16 kHz reference used by :mod:`disk_recorder.sync`
to align the capture.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("disk_recorder.playback")


@dataclass
class Reference:
    """A loaded audio file ready for playback and alignment."""

    samples: np.ndarray     # float32, shape (n,) mono or (n, channels)
    sample_rate: int
    duration_s: float
    path: str


def load(path: str) -> Reference:
    """Load an audio file as float32 samples."""
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    data = np.asarray(data, dtype=np.float32)
    frames = data.shape[0]
    duration = frames / float(sample_rate)
    return Reference(samples=data, sample_rate=sample_rate, duration_s=duration, path=path)


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Collapse a (n, channels) array to mono float32."""
    if samples.ndim == 2:
        return samples.mean(axis=1).astype(np.float32)
    return samples.astype(np.float32)


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample float32 samples along time (axis 0); keeps channel layout."""
    if src_rate == dst_rate:
        return samples.astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(int(src_rate), int(dst_rate))
    up = dst_rate // divisor
    down = src_rate // divisor
    return resample_poly(samples, up, down, axis=0).astype(np.float32)


def reference_16k(reference: Reference, target_rate: int = 16000) -> np.ndarray:
    """Return the reference as mono float32 resampled to ``target_rate``."""
    mono = to_mono(reference.samples)
    return _resample(mono, reference.sample_rate, target_rate)


class Player:
    """Playback on a **single, persistently open** output stream.

    The device is opened once per run (:meth:`open`) and stays open until
    :meth:`close`; each file is resampled to the stream's fixed rate and handed
    to the PortAudio callback, and :meth:`stop` merely silences the stream
    instead of tearing it down.

    This matters far more than it looks. The previous implementation used
    ``sd.play()`` / ``sd.stop()``, which open and close the device — and
    renegotiate its sample-rate — for **every single file**. Over a multi-day
    run that is tens of thousands of open/close cycles through the Windows
    audio engine (``audiodg.exe``), which leaks handles and eventually wedges
    system-wide audio until the machine is rebooted. One stream for the whole
    session removes the churn entirely, and as a bonus removes the per-file
    device-open latency that used to eat into the alignment guard window.
    """

    #: Rates tried when the device advertises nothing usable, best first.
    _FALLBACK_RATES = (48000, 44100, 32000, 16000)

    def __init__(self, output_device: int | None, routing: str = "both") -> None:
        self._device = output_device
        self._routing = routing
        self._stream = None
        self._rate: int | None = None
        self._channels: int | None = None
        self._open_lock = threading.Lock()
        self._done = threading.Event()
        self._done.set()
        # Published to the callback; assignment order is (None → pos → data) so
        # the callback never sees a new buffer with a stale read position.
        self._data: np.ndarray | None = None
        self._pos = 0
        #: Set when the configured device could not be opened and playback fell
        #: back to the system default — the run is then recording through an
        #: unknown speaker and the pipeline surfaces this as a hard error.
        self.fallback_device: int | None = None

    # ----- device probing -----

    def _device_info(self, device: int | None) -> dict:
        import sounddevice as sd

        try:
            return dict(sd.query_devices(device, "output"))
        except Exception:  # pragma: no cover - hardware dependent
            return {}

    def _device_rates(self, info: dict) -> list[int]:
        """Device-native rate first, then common rates as fallbacks."""
        rates: list[int] = []
        default_rate = int(round(info.get("default_samplerate", 0)))
        if default_rate:
            rates.append(default_rate)
        for rate in self._FALLBACK_RATES:
            if rate not in rates:
                rates.append(rate)
        return rates

    def _default_output(self) -> int | None:
        """System default output device index (shared mode, coexists with others)."""
        import sounddevice as sd

        try:
            default = sd.default.device
            out = default[1] if isinstance(default, (list, tuple)) else default
            return int(out) if out is not None and out >= 0 else None
        except Exception:  # pragma: no cover - hardware dependent
            return None

    # ----- stream lifecycle -----

    def _callback(self, outdata, frames, time_info, status) -> None:
        """Feed the queued file into the device; silence when nothing is queued.

        Runs on the PortAudio thread, so it only ever touches ``_data`` /
        ``_pos`` and must not allocate or block.
        """
        data = self._data
        if data is None:
            outdata[:] = 0
            return
        start = self._pos
        end = min(start + frames, len(data))
        count = end - start
        outdata[:count] = data[start:end]
        if count < frames:
            outdata[count:] = 0
            self._data = None
            self._pos = 0
            self._done.set()
        else:
            self._pos = end

    def _open_on(self, device: int | None) -> bool:
        """Open and start the persistent stream on ``device``; True on success."""
        import sounddevice as sd

        info = self._device_info(device)
        channels = 2 if int(info.get("max_output_channels", 2) or 2) >= 2 else 1
        for rate in self._device_rates(info):
            try:
                stream = sd.OutputStream(
                    samplerate=rate, channels=channels, dtype="float32",
                    device=device, latency="high", callback=self._callback)
                stream.start()
            except sd.PortAudioError as exc:
                logger.debug("device %s rejected %u Hz/%uch (%s)",
                             device, rate, channels, exc)
                continue
            self._stream, self._rate, self._channels = stream, rate, channels
            logger.info("output stream open on device %s (%s) at %u Hz/%uch — "
                        "kept open for the whole session",
                        device, info.get("name", "?"), rate, channels)
            if channels < 2 and self._routing != "both":
                logger.warning("device %s is mono — output routing '%s' ignored",
                               device, self._routing)
            return True
        return False

    def open(self) -> None:
        """Open the output device once for the whole run.

        Falls back to the system default output if the configured device cannot
        be opened at all (an exclusive host API held by another process), and
        records that in :attr:`fallback_device` so the caller can treat the run
        as compromised rather than silently recording through another speaker.
        """
        with self._open_lock:
            if self._stream is not None:
                return
            if self._open_on(self._device):
                return
            fallback = self._default_output()
            if fallback is not None and fallback != self._device:
                logger.error("output device %s cannot be opened — falling back "
                             "to the system default output %s",
                             self._device, fallback)
                if self._open_on(fallback):
                    self.fallback_device = fallback
                    return
            info = self._device_info(self._device)
            raise RuntimeError(
                f"output device {self._device} ({info.get('name', '?')}) accepts "
                f"no sample rate; the system default output also failed — pick a "
                f"non-exclusive host API (WASAPI/DirectSound/MME) for this speaker")

    def is_alive(self) -> bool:
        """True while the persistent stream is open and running."""
        stream = self._stream
        if stream is None:
            return False
        try:
            return not stream.closed and stream.active
        except Exception:  # pragma: no cover - hardware dependent
            return False

    # ----- playback -----

    def play(self, reference: Reference) -> None:
        """Queue ``reference`` on the open stream; returns immediately.

        The file is downmixed/routed and resampled to the stream's fixed rate,
        so playback never touches the device configuration.
        """
        if self._stream is None:
            self.open()
        if not self.is_alive():
            # The device died under us (unplugged, driver reset). Rebuild the
            # stream once rather than silently playing into a dead handle.
            logger.error("output stream is no longer active — reopening")
            self._teardown()
            self.open()

        payload = self._route(reference.samples) if self._channels == 2 \
            else to_mono(reference.samples).reshape(-1, 1)
        payload = _resample(payload, reference.sample_rate, self._rate)
        np.clip(payload, -1.0, 1.0, out=payload)

        self._data = None                     # stop the callback reading …
        self._pos = 0                         # … while the position is reset
        self._done.clear()
        self._data = np.ascontiguousarray(payload, dtype=np.float32)

    def _route(self, samples: np.ndarray) -> np.ndarray:
        """Downmix to mono, then map onto a stereo frame per the routing mode.

        ``both`` feeds the mono signal to both speakers; ``left``/``right`` feed
        it to a single speaker (the other channel stays silent).
        """
        mono = to_mono(samples)
        silence = np.zeros_like(mono)
        if self._routing == "left":
            return np.column_stack([mono, silence])
        if self._routing == "right":
            return np.column_stack([silence, mono])
        return np.column_stack([mono, mono])

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the queued file has played out (or ``timeout`` elapses)."""
        return self._done.wait(timeout)

    def stop(self) -> None:
        """Silence playback **without** closing the device."""
        self._data = None
        self._pos = 0
        self._done.set()

    def _teardown(self) -> None:
        stream, self._stream = self._stream, None
        self._rate = self._channels = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # pragma: no cover - hardware dependent
                pass

    def close(self) -> None:
        """Close the output device — called once, when the run ends."""
        with self._open_lock:
            self.stop()
            self._teardown()
        logger.info("output stream closed")
