"""Probe host audio output devices to find one that actually opens.

Run on the target machine (in the recorder's venv):

    python probe_audio.py

It lists every output device with its host API, then *actually opens and
starts* a short silent stream on each at a couple of rates. Only devices that
open are usable for playback; the report tells you which device index to select
in the recorder GUI. If NONE open, the audio problem is at the OS/driver level
(no working speaker endpoint), not in the recorder.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd


def _refresh() -> None:
    """Rebuild PortAudio's device list (indices go stale if devices changed)."""
    try:
        sd._terminate()
        sd._initialize()
    except Exception as exc:  # pragma: no cover
        print(f"  (could not refresh PortAudio: {exc})")


def _try_open(index: int, rate: int, channels: int) -> str:
    """Return 'ok' if a stream opens+starts, else the error text."""
    frames = int(rate * 0.1)
    buf = np.zeros((frames, channels), dtype=np.float32)
    try:
        sd.check_output_settings(device=index, samplerate=rate, channels=channels)
    except Exception as exc:
        return f"check failed: {exc}"
    try:
        with sd.OutputStream(device=index, samplerate=rate, channels=channels):
            pass
    except Exception as exc:
        return f"open failed: {exc}"
    # A real (silent) write catches devices that open but can't start.
    try:
        sd.play(buf, rate, device=index, blocking=True)
    except Exception as exc:
        return f"play failed: {exc}"
    return "ok"


def main() -> int:
    _refresh()
    host_apis = sd.query_hostapis()
    devices = sd.query_devices()
    default = sd.default.device
    default_out = default[1] if isinstance(default, (list, tuple)) else default
    print(f"PortAudio version: {sd.get_portaudio_version()}")
    print(f"default output index: {default_out}\n")

    usable: list[int] = []
    for index, dev in enumerate(devices):
        if dev["max_output_channels"] <= 0:
            continue
        api = host_apis[dev["hostapi"]]["name"] if dev["hostapi"] < len(host_apis) else "?"
        default_rate = int(round(dev.get("default_samplerate", 0)))
        mark = " (system default)" if index == default_out else ""
        print(f"[{index}] {dev['name']} ({api}){mark}")
        ok_here = False
        for rate in dict.fromkeys([default_rate, 48000, 44100]):
            if not rate:
                continue
            result = _try_open(index, rate, min(2, dev["max_output_channels"]))
            print(f"      {rate} Hz: {result}")
            if result == "ok":
                ok_here = True
        if ok_here:
            usable.append(index)
        print()

    if usable:
        print(f"USABLE output device indices: {usable}")
        print("Select one of these in the recorder GUI.")
    else:
        print("NO output device could be opened.")
        print("This is an OS/driver problem, not the recorder:")
        print("  - plug in / enable the speaker or headphones (jack detection)")
        print("  - Windows Sound settings: is a working output present & enabled?")
        print("  - try playing any sound in Windows first")
        print("  - restart the 'Windows Audio' service or update the Realtek driver")
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
