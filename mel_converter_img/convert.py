#!/usr/bin/env python3
"""Batch mel-spectrogram image generator.

Recursively scans a directory for WAV files and writes a PNG mel spectrogram
image next to each one, using the same parameters as disk_recorder/mel.py.

Usage:
    python convert.py <folder> [--out-dir <dir>] [--ext png|jpg]
    python convert.py .                  # process current directory
    python convert.py /path/to/wavs --out-dir /path/to/images
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Allow importing disk_recorder.mel from the sibling package.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import numpy as np
import soundfile as sf
from disk_recorder.mel import compute_mel, _MEL_CFG


def wav_to_image(wav_path: Path, out_path: Path, ext: str = "png") -> None:
    pcm_float, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    # Convert to int16 range expected by compute_mel.
    pcm_int16 = (pcm_float * 32768.0).astype(np.float32)

    mel = compute_mel(pcm_int16, sr)

    from matplotlib.figure import Figure

    hop_s = _MEL_CFG["hop_length_ms"] / 1000.0
    duration = mel.shape[1] * hop_s

    fig = Figure(figsize=(10, 4), dpi=100)
    ax = fig.add_subplot(111)
    img = ax.imshow(
        mel,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, _MEL_CFG["n_mels"]],
        interpolation="nearest",
    )
    fig.colorbar(img, ax=ax, label="Log-mel (norm)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title(wav_path.name)
    fig.tight_layout()

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.print_figure(str(out_path), format=ext.lstrip("."))


def find_wavs(root: Path):
    for path in sorted(root.rglob("*.wav")):
        yield path
    for path in sorted(root.rglob("*.WAV")):
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mel spectrogram PNGs for WAV files.")
    parser.add_argument("folder", type=Path, help="Root folder to scan recursively.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (mirrors source tree). Default: image next to each WAV.",
    )
    parser.add_argument("--ext", default="png", choices=["png", "jpg", "pdf"], help="Image format.")
    args = parser.parse_args()

    root = args.folder.resolve()
    if not root.is_dir():
        sys.exit(f"Error: {root} is not a directory")

    wavs = list(find_wavs(root))
    if not wavs:
        print(f"No WAV files found under {root}")
        return

    print(f"Found {len(wavs)} WAV file(s) under {root}")

    for wav in wavs:
        if args.out_dir:
            rel = wav.relative_to(root)
            out = args.out_dir / rel.with_suffix(f".{args.ext}")
        else:
            out = wav.with_suffix(f".{args.ext}")

        print(f"  {wav.relative_to(root)}  →  {out}")
        try:
            wav_to_image(wav, out, args.ext)
        except Exception as exc:
            print(f"    ERROR: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
