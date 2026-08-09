#!/usr/bin/env python3
"""Loudness-match a folder of recordings to a reference folder (RMS / dBFS).

The target level is derived from the **reference** dataset (median per-file RMS),
then every audio file under the **input** folder is scaled so its RMS matches that
target. A true-peak guard prevents clipping: if the required gain would push the
peak past ``--peak-ceiling`` dBFS, the gain is reduced and the file is reported as
peak-limited (it lands a bit quieter than target instead of clipping).

Originals are never modified — output goes to a mirrored ``--out`` tree
(default: ``<input>_norm``) unless ``--in-place`` is given.

Pure standard library (``wave`` + ``array`` + ``math``); handles PCM WAV,
16/24/32-bit integer and 32-bit float, mono or multi-channel. Other containers
(flac/ogg/mp3) are skipped with a warning — decode them to WAV first.

Examples
--------
    # match the stm32n6 recordings to the X3800 reference level
    python3 normalize_loudness.py --ref ref_x3800/ --input rec_stm32n6/

    # write in place, override the target explicitly
    python3 normalize_loudness.py --ref ref/ --input data/ --in-place --target-dbfs -23
"""
from __future__ import annotations

import argparse
import array
import math
import statistics
import sys
import wave
from pathlib import Path

WAV_EXT = ".wav"
OTHER_AUDIO = (".flac", ".ogg", ".mp3", ".aiff", ".m4a")


# --------------------------------------------------------------------------- IO
def _read_wav(path: Path):
    """Return ``(samples_float, sample_rate, n_channels, sampwidth, is_float)``.

    ``samples_float`` is a flat ``array('d')`` of all interleaved samples scaled
    to roughly [-1, 1).
    """
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())

    out = array.array("d")
    if sw == 2:  # int16
        a = array.array("h")
        a.frombytes(raw)
        scale = 1.0 / 32768.0
        out.extend(x * scale for x in a)
        is_float = False
    elif sw == 1:  # uint8 (offset binary)
        a = array.array("B")
        a.frombytes(raw)
        scale = 1.0 / 128.0
        out.extend((x - 128) * scale for x in a)
        is_float = False
    elif sw == 3:  # int24, little-endian packed
        scale = 1.0 / 8388608.0
        for i in range(0, len(raw), 3):
            v = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16)
            if v & 0x800000:
                v -= 0x1000000
            out.append(v * scale)
        is_float = False
    elif sw == 4:  # int32 or float32 — wave can't tell; assume int32 PCM
        a = array.array("i")
        a.frombytes(raw)
        scale = 1.0 / 2147483648.0
        out.extend(x * scale for x in a)
        is_float = False
    else:
        raise ValueError(f"unsupported sample width {sw} bytes")
    return out, sr, ch, sw, is_float


def _write_wav(path: Path, samples, sr: int, ch: int, sw: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sw == 2:
        a = array.array("h")
        for x in samples:  # round then clamp to valid int16
            v = int(round(x * 32768))
            a.append(-32768 if v < -32768 else (32767 if v > 32767 else v))
        body = a.tobytes()
    elif sw == 1:
        a = array.array("B")
        a.extend(_clamp8(x) for x in samples)
        body = a.tobytes()
    elif sw == 3:
        body = bytearray()
        for x in samples:
            v = int(round(x * 8388608))
            v = -8388608 if v < -8388608 else (8388607 if v > 8388607 else v)
            v &= 0xFFFFFF
            body += bytes((v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF))
        body = bytes(body)
    elif sw == 4:
        a = array.array("i")
        for x in samples:
            v = int(round(x * 2147483648))
            v = -2147483648 if v < -2147483648 else (2147483647 if v > 2147483647 else v)
            a.append(v)
        body = a.tobytes()
    else:
        raise ValueError(f"unsupported sample width {sw}")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(body)


def _clamp8(x):
    v = int(round(x * 128)) + 128
    return 0 if v < 0 else (255 if v > 255 else v)


# ----------------------------------------------------------------- measurement
def _stats(samples):
    """Return ``(rms, peak)`` of DC-removed samples, both linear amplitude."""
    n = len(samples)
    if n == 0:
        return 0.0, 0.0
    dc = math.fsum(samples) / n
    sq = math.fsum((x - dc) * (x - dc) for x in samples)
    rms = math.sqrt(sq / n)
    peak = max((abs(x - dc) for x in samples), default=0.0)
    return rms, peak


def _dbfs(x):
    return 20.0 * math.log10(x) if x > 1e-12 else -120.0


def _iter_wavs(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == WAV_EXT:
            yield p
        elif ext in OTHER_AUDIO:
            print(f"  ! skip (not PCM WAV, decode first): {p}", file=sys.stderr)


# ------------------------------------------------------------------------ main
def measure_reference(ref_dir: Path):
    """Return ``(target_rms_linear, count)`` from the reference set's median RMS."""
    rms_list = []
    for p in _iter_wavs(ref_dir):
        try:
            samples, *_ = _read_wav(p)
        except Exception as e:  # noqa: BLE001
            print(f"  ! ref read failed {p}: {e}", file=sys.stderr)
            continue
        rms, _ = _stats(samples)
        if rms > 0:
            rms_list.append(rms)
    if not rms_list:
        raise SystemExit(f"no usable reference WAVs in {ref_dir}")
    target = statistics.median(rms_list)
    mean = statistics.fmean(rms_list)
    print(f"reference: {len(rms_list)} files | "
          f"median RMS {_dbfs(target):.1f} dBFS, mean {_dbfs(mean):.1f} dBFS")
    return target, len(rms_list)


def normalize_folder(input_dir: Path, out_dir: Path, target_rms: float,
                     peak_ceiling_db: float, in_place: bool) -> None:
    ceiling = 10.0 ** (peak_ceiling_db / 20.0)
    n = limited = 0
    for src in _iter_wavs(input_dir):
        try:
            samples, sr, ch, sw, _ = _read_wav(src)
        except Exception as e:  # noqa: BLE001
            print(f"  ! read failed {src}: {e}", file=sys.stderr)
            continue
        rms, peak = _stats(samples)
        if rms <= 0:
            print(f"  - skip silent {src.name}")
            continue

        gain = target_rms / rms
        note = ""
        if peak * gain > ceiling:  # would clip -> back off
            gain = ceiling / peak
            limited += 1
            note = f" [peak-limited, {_dbfs(rms * gain):.1f} dBFS]"

        scaled = array.array("d", (x * gain for x in samples))

        dst = src if in_place else out_dir / src.relative_to(input_dir)
        _write_wav(dst, scaled, sr, ch, sw)
        n += 1
        print(f"  {src.name}: {_dbfs(rms):.1f} -> {_dbfs(rms * gain):.1f} dBFS "
              f"(gain {20 * math.log10(gain):+.1f} dB){note}")

    where = "in place" if in_place else str(out_dir)
    print(f"\nnormalised {n} file(s) -> {where}"
          + (f"; {limited} peak-limited (acoustically quieter than target)" if limited else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, type=Path,
                    help="folder of reference recordings defining the target level")
    ap.add_argument("--input", required=True, type=Path,
                    help="folder of recordings to normalise (searched recursively)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: <input>_norm). Ignored with --in-place")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the input files instead of writing a copy")
    ap.add_argument("--target-dbfs", type=float, default=None,
                    help="override the reference-derived target RMS (dBFS)")
    ap.add_argument("--peak-ceiling", type=float, default=-1.0,
                    help="max output peak in dBFS, clip guard (default -1.0)")
    args = ap.parse_args(argv)

    if not args.input.is_dir():
        raise SystemExit(f"input not a folder: {args.input}")

    if args.target_dbfs is not None:
        target = 10.0 ** (args.target_dbfs / 20.0)
        print(f"target: {args.target_dbfs:.1f} dBFS (override)")
    else:
        if not args.ref.is_dir():
            raise SystemExit(f"ref not a folder: {args.ref}")
        target, _ = measure_reference(args.ref)

    out_dir = args.out or args.input.with_name(args.input.name + "_norm")
    if not args.in_place and out_dir.resolve() == args.input.resolve():
        raise SystemExit("out dir equals input; use --in-place to overwrite")

    normalize_folder(args.input, out_dir, target, args.peak_ceiling, args.in_place)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
