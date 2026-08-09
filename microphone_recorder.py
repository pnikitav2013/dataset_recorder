#!/usr/bin/env python3
"""Record framed STM32 microphone PCM16 data into a mono WAV file and plot mel spectrogram."""

from __future__ import annotations

import argparse
import logging
import sys
import wave

from reliable_transport import ReliableTransport, TransportWarning

# Mel spectrogram parameters (fixed)
_MEL_CFG = {
    "window_size_ms": 20.0,
    "hop_length_ms": 10.0,
    "n_mels": 64,
    "fft_size": 512,
    "lower_edge_hz": 80.0,
    "upper_edge_hz": None,   # → Nyquist
    "power": 2.0,
    "log_zero_guard_value": 5.9604644775390625e-08,  # 2^-24 ≈ float16 eps
    "log_floor": 1e-6,
    "preemph": 0.97,
    "dither": 1e-05,
    "mel_htk": False,
    "mel_norm": "slaney",
    "normalization": "per_feature",  # mean/std per mel bin across time
}


def _compute_mel_spectrogram(pcm_bytes: bytes, sample_rate: int) -> "np.ndarray":
    """Return a (n_mels, T) log-mel spectrogram array."""
    import numpy as np

    cfg = _MEL_CFG
    win_samples = int(sample_rate * cfg["window_size_ms"] / 1000)
    hop_samples = int(sample_rate * cfg["hop_length_ms"] / 1000)
    n_fft = cfg["fft_size"]

    # int16 → float32 in [-1, 1]
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # dither
    if cfg["dither"] > 0:
        audio += cfg["dither"] * np.random.randn(len(audio)).astype(np.float32)

    # pre-emphasis
    if cfg["preemph"] > 0:
        audio = np.concatenate([[audio[0]], audio[1:] - cfg["preemph"] * audio[:-1]])

    # STFT with periodic Hann window
    window = np.hanning(win_samples + 1)[:-1].astype(np.float32)
    n_frames = max(1, (len(audio) - win_samples) // hop_samples + 1)
    frames = np.stack(
        [audio[i * hop_samples : i * hop_samples + win_samples] for i in range(n_frames)
         if i * hop_samples + win_samples <= len(audio)]
    )  # (T, win_samples)
    frames = frames * window  # apply window
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)  # (T, n_fft//2+1)
    power = np.abs(spectrum) ** cfg["power"]  # (T, n_fft//2+1)

    # mel filterbank via librosa
    try:
        import librosa
        mel_fb = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=cfg["n_mels"],
            fmin=cfg["lower_edge_hz"],
            fmax=cfg["upper_edge_hz"],
            htk=cfg["mel_htk"],
            norm=cfg["mel_norm"],
        )  # (n_mels, n_fft//2+1)
    except ImportError:
        # fallback: triangular mel filterbank without norm
        mel_fb = _triangular_mel_fb(sample_rate, n_fft, cfg["n_mels"],
                                     cfg["lower_edge_hz"], cfg["upper_edge_hz"])

    mel = (mel_fb @ power.T)  # (n_mels, T)

    # log with zero guard (add type)
    mel = np.log(mel + cfg["log_zero_guard_value"])
    mel = np.maximum(mel, np.log(cfg["log_floor"]))

    # per-feature (per mel bin) mean/std normalization
    if cfg["normalization"] == "per_feature":
        mean = mel.mean(axis=1, keepdims=True)
        std = mel.std(axis=1, keepdims=True) + 1e-8
        mel = (mel - mean) / std

    return mel


def _triangular_mel_fb(sr: int, n_fft: int, n_mels: int, fmin: float, fmax) -> "np.ndarray":
    """Minimal triangular mel filterbank without librosa (slaney-style, no norm)."""
    import numpy as np

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


def _plot_mel(mel: "np.ndarray", output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping mel spectrogram plot", file=sys.stderr)
        return

    import os

    cfg = _MEL_CFG
    hop_s = cfg["hop_length_ms"] / 1000.0
    duration = mel.shape[1] * hop_s

    fig, ax = plt.subplots(figsize=(12, 4))
    img = ax.imshow(
        mel,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, cfg["n_mels"]],
        interpolation="nearest",
    )
    fig.colorbar(img, ax=ax, label="Log-mel (normalized)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title(f"Log-mel spectrogram — {os.path.basename(output_path)}")
    plt.tight_layout()

    png_path = os.path.splitext(output_path)[0] + "_mel.png"
    fig.savefig(png_path, dpi=150)
    print(f"mel spectrogram saved to {png_path}", file=sys.stderr)

    plt.show()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", help="serial port, for example /dev/ttyACM0 or COM5")
    parser.add_argument("--output", required=True, help="destination WAV path")
    parser.add_argument("--baud", type=int, default=921600, help="UART bitrate (default: 921600)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="PCM sample rate in Hz (default: 16000)")
    parser.add_argument("--no-mel", action="store_true", help="skip mel spectrogram plot after recording")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        import serial
    except ImportError:
        print("pyserial is missing; run ./create_venv.sh first", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("py_recorder.microphone")
    pcm_buffer: list[bytes] = []

    def on_warning(warning: TransportWarning, sequence: int, detail: str) -> None:
        suffix = f" ({detail})" if detail else ""
        logger.warning("transport %s, sequence=%u%s", warning.value, sequence, suffix)

    try:
        with serial.Serial(arguments.port, arguments.baud, timeout=0.05) as port, wave.open(
            arguments.output, "wb"
        ) as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(arguments.sample_rate)

            def on_message(sequence: int, payload: bytes) -> None:
                if len(payload) % 2 != 0:
                    logger.warning("discarding odd-sized PCM payload, sequence=%u", sequence)
                    return
                output.writeframesraw(payload)
                pcm_buffer.append(payload)

            transport = ReliableTransport(port.write, on_message=on_message, on_warning=on_warning)
            logger.info(
                "recording %s at %u baud to %s; press Ctrl-C to stop",
                arguments.port,
                arguments.baud,
                arguments.output,
            )
            while True:
                transport.process(port.read(port.in_waiting or 1))
    except KeyboardInterrupt:
        total_samples = sum(len(b) for b in pcm_buffer) // 2
        logger.info("saved %u PCM16 samples", total_samples)
        if not arguments.no_mel and pcm_buffer:
            logger.info("computing mel spectrogram…")
            try:
                mel = _compute_mel_spectrogram(b"".join(pcm_buffer), arguments.sample_rate)
                _plot_mel(mel, arguments.output)
            except Exception as exc:  # noqa: BLE001
                logger.warning("mel spectrogram failed: %s", exc)
        return 0
    except (OSError, ValueError, wave.Error) as error:
        logger.error("recording failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
