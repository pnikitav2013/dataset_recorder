"""Reference audio preparation and non-blocking playback.

Loads an audio file, plays it on the chosen output device while the board
records, and produces a mono 16 kHz reference used by :mod:`disk_recorder.sync`
to align the capture.
"""

from __future__ import annotations

import logging
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
    """Non-blocking playback on a selected output device (``sounddevice``).

    Raw ALSA ``hw:`` outputs (e.g. HDMI) only accept their native rates, so if
    the file's rate is rejected the audio is resampled to a rate the device
    supports and playback is retried. Alignment is unaffected — it always uses
    the 16 kHz reference, independent of the playback rate.
    """

    def __init__(self, output_device: int | None, routing: str = "both") -> None:
        self._device = output_device
        self._routing = routing

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

    def _device_info(self, device: int | None) -> dict:
        import sounddevice as sd

        try:
            return dict(sd.query_devices(device, "output"))
        except Exception:  # pragma: no cover - hardware dependent
            return {}

    def _device_rates(self, info: dict) -> list[int]:
        rates: list[int] = []
        default_rate = int(round(info.get("default_samplerate", 0)))
        if default_rate:
            rates.append(default_rate)
        for rate in (48000, 44100, 32000, 16000):
            if rate not in rates:
                rates.append(rate)
        return rates

    def _default_output(self) -> int | None:
        """System default output device index (shared-mode, coexists with others)."""
        import sounddevice as sd

        try:
            default = sd.default.device
            out = default[1] if isinstance(default, (list, tuple)) else default
            return int(out) if out is not None and out >= 0 else None
        except Exception:  # pragma: no cover - hardware dependent
            return None

    def _candidate_devices(self) -> list[int | None]:
        """The chosen device first, then the system default as a fallback.

        The configured device may use an exclusive host API (e.g. Windows
        WDM-KS) that stops opening once another process grabs the speaker; the
        shared-mode default output keeps unattended runs alive.
        """
        candidates: list[int | None] = [self._device]
        fallback = self._default_output()
        if fallback is not None and fallback != self._device:
            candidates.append(fallback)
        return candidates

    def _play_on(self, device: int | None, reference: Reference) -> bool:
        """Try every ``(rate, channels)`` combo on one device; True if playing."""
        import sounddevice as sd

        info = self._device_info(device)
        max_out = int(info.get("max_output_channels", 2) or 2)

        # Preferred layout is the routed stereo frame; fall back to mono for
        # devices that only expose a single output channel.
        layouts: list[tuple[int, np.ndarray]] = []
        if max_out >= 2:
            layouts.append((2, self._route(reference.samples)))
        layouts.append((1, to_mono(reference.samples)))

        # Try the file's own rate first, then the device's supported rates.
        src_rate = reference.sample_rate
        rates = [src_rate] + [r for r in self._device_rates(info) if r != src_rate]

        for rate in rates:
            for channels, data in layouts:
                try:
                    sd.check_output_settings(
                        device=device, samplerate=rate, channels=channels)
                except Exception:
                    continue
                try:
                    payload = _resample(data, src_rate, rate)
                    sd.play(payload, rate, device=device)
                except sd.PortAudioError as exc:
                    # -9999 host errors (exclusive WDM-KS busy) land here too.
                    logger.warning("device %s rejected %u Hz/%uch (%s)",
                                   device, rate, channels, exc)
                    continue
                if device != self._device or rate != src_rate or channels < 2:
                    logger.info("playing %s on device %s at %u Hz/%uch (source %u Hz)",
                                reference.path, device, rate, channels, src_rate)
                return True
        return False

    def play(self, reference: Reference) -> None:
        """Begin playback; returns immediately.

        Probes ``(channels, rate)`` combinations each candidate device actually
        supports (via ``check_output_settings`` + a real open) and resamples /
        downmixes to the first that works. If the configured device cannot open
        — e.g. an exclusive Windows WDM-KS speaker seized by another process —
        playback falls back to the system default output device.
        """
        for device in self._candidate_devices():
            if self._play_on(device, reference):
                return

        info = self._device_info(self._device)
        name = info.get("name", self._device)
        raise RuntimeError(
            f"output device {self._device} ({name}) accepts no sample rate; "
            f"the system default output also failed — pick a non-exclusive "
            f"host API (MME/WASAPI/DirectSound) for this speaker")

    def wait(self) -> None:  # pragma: no cover - hardware dependent
        import sounddevice as sd

        sd.wait()

    def stop(self) -> None:
        import sounddevice as sd

        sd.stop()
