"""Log-mel spectrogram computation and figure building.

The spectrogram parameters and maths are taken from
``py_recorder/microphone_recorder.py`` so the GUI shows the same representation
the rest of the tooling uses. Figures are built with the matplotlib object API
(``Figure``), never ``pyplot``, so they can be created off the main thread and
drawn by the Tkinter thread.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("disk_recorder.mel")

# Mel spectrogram parameters (mirrors microphone_recorder._MEL_CFG).
_MEL_CFG = {
    "window_size_ms": 20.0,
    "hop_length_ms": 10.0,
    "n_mels": 64,
    "fft_size": 512,
    "lower_edge_hz": 80.0,
    "upper_edge_hz": None,   # → Nyquist
    "power": 2.0,
    "log_zero_guard_value": 5.9604644775390625e-08,
    "log_floor": 1e-6,
    "preemph": 0.97,
    "dither": 1e-05,
    "mel_htk": False,
    "mel_norm": "slaney",
    "normalization": "per_feature",
}


def _triangular_mel_fb(sr: int, n_fft: int, n_mels: int, fmin: float, fmax) -> np.ndarray:
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    if fmax is None:
        fmax = sr / 2.0
    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, center, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(lo, center):
            fb[m - 1, k] = (k - lo) / max(center - lo, 1)
        for k in range(center, hi):
            fb[m - 1, k] = (hi - k) / max(hi - center, 1)
    return fb


def compute_mel(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return a (n_mels, T) log-mel spectrogram for int16 PCM samples."""
    cfg = _MEL_CFG
    win_samples = int(sample_rate * cfg["window_size_ms"] / 1000)
    hop_samples = int(sample_rate * cfg["hop_length_ms"] / 1000)
    n_fft = cfg["fft_size"]

    audio = pcm.astype(np.float32) / 32768.0
    if audio.size < win_samples:
        audio = np.pad(audio, (0, win_samples - audio.size))

    if cfg["dither"] > 0:
        audio = audio + cfg["dither"] * np.random.randn(len(audio)).astype(np.float32)
    if cfg["preemph"] > 0:
        audio = np.concatenate([[audio[0]], audio[1:] - cfg["preemph"] * audio[:-1]])

    window = np.hanning(win_samples + 1)[:-1].astype(np.float32)
    n_frames = max(1, (len(audio) - win_samples) // hop_samples + 1)
    frames = np.stack(
        [audio[i * hop_samples: i * hop_samples + win_samples] for i in range(n_frames)
         if i * hop_samples + win_samples <= len(audio)]
    )
    frames = frames * window
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power = np.abs(spectrum) ** cfg["power"]

    try:
        import librosa

        mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=cfg["n_mels"],
            fmin=cfg["lower_edge_hz"], fmax=cfg["upper_edge_hz"],
            htk=cfg["mel_htk"], norm=cfg["mel_norm"],
        )
    except ImportError:
        mel_fb = _triangular_mel_fb(sample_rate, n_fft, cfg["n_mels"],
                                    cfg["lower_edge_hz"], cfg["upper_edge_hz"])

    mel = mel_fb @ power.T
    mel = np.log(mel + cfg["log_zero_guard_value"])
    mel = np.maximum(mel, np.log(cfg["log_floor"]))
    if cfg["normalization"] == "per_feature":
        mean = mel.mean(axis=1, keepdims=True)
        std = mel.std(axis=1, keepdims=True) + 1e-8
        mel = (mel - mean) / std
    return mel


#: One capturing device's precomputed spectrogram: ``(label, (n_mels, T) array)``.
Panel = tuple[str, np.ndarray]


def compute_panels(items: list[tuple[str, np.ndarray]],
                   sample_rate: int) -> list[Panel]:
    """Compute one log-mel array per device from ``(label, pcm)`` pairs.

    Only the *data* crosses the thread boundary — the pipeline never builds
    matplotlib objects. Over a multi-day run that is tens of thousands of
    Figures (each a web of reference cycles that only the generational GC can
    collect) that are never created in the first place; the GUI redraws a
    single long-lived Figure instead.
    """
    return [(label, compute_mel(pcm, sample_rate)) for label, pcm in items]


def _draw_panel(ax, mel_data: np.ndarray, fig, title: Optional[str],
                show_xlabel: bool) -> None:
    """Render one precomputed log-mel spectrogram onto an existing axis."""
    hop_s = _MEL_CFG["hop_length_ms"] / 1000.0
    duration = mel_data.shape[1] * hop_s
    img = ax.imshow(mel_data, aspect="auto", origin="lower",
                    extent=[0, duration, 0, _MEL_CFG["n_mels"]], interpolation="nearest")
    fig.colorbar(img, ax=ax, label="Log-mel (norm)")
    if show_xlabel:
        ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    if title:
        ax.set_title(title)


def render_panels(fig, panels: list[Panel]) -> None:
    """Draw ``panels`` as stacked subplots into an **existing**, reused Figure.

    The figure is cleared first, so colorbars and axes from the previous file
    are discarded rather than accumulating.
    """
    fig.clear()
    n = max(1, len(panels))
    fig.set_size_inches(7, 2.6 * n + 0.4, forward=False)
    for i, (label, mel_data) in enumerate(panels):
        ax = fig.add_subplot(n, 1, i + 1)
        _draw_panel(ax, mel_data, fig, label, show_xlabel=(i == n - 1))
    fig.tight_layout()


def make_figure(pcm: np.ndarray, sample_rate: int, title: Optional[str] = None):
    """Build a matplotlib Figure of the log-mel spectrogram (no pyplot)."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7, 3.2), dpi=100)
    ax = fig.add_subplot(111)
    _draw_panel(ax, compute_mel(pcm, sample_rate), fig,
                os.path.basename(title) if title else None, show_xlabel=True)
    fig.tight_layout()
    return fig


