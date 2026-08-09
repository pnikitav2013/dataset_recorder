"""Enumeration of serial ports and host audio devices.

This is the *connection discovery* layer used to populate the GUI selectors:
which COM port the STM32 (ST-Link Virtual COM Port) is on, and which output /
input audio devices are available.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("disk_recorder.devices")

# Legacy 16550 placeholders Linux always exposes (/dev/ttyS0..N) with no USB
# identity; they drown out the real ST-Link VCP, so they are hidden by default.
_LEGACY_TTY = re.compile(r"^/dev/ttyS\d+$")


@dataclass(frozen=True)
class SerialPortInfo:
    device: str          # e.g. "COM5" or "/dev/ttyACM0"
    description: str
    hwid: str
    is_usb: bool = False

    @property
    def label(self) -> str:
        return f"{self.device} — {self.description}"


@dataclass(frozen=True)
class AudioDeviceInfo:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.name} ({self.host_api})"


def list_serial_ports(include_legacy: bool = False) -> list[SerialPortInfo]:
    """Return available serial ports (ST-Link VCP shows up here).

    USB ports (those reporting a VID:PID, e.g. the ST-Link VCP) are listed first.
    Legacy ``/dev/ttyS*`` placeholders are hidden unless ``include_legacy`` is set.
    """
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        logger.error("pyserial unavailable (%s) — run with the venv python "
                     "(sys.executable=%s)", exc, _executable())
        return []
    ports: list[SerialPortInfo] = []
    for port in list_ports.comports():
        is_usb = port.vid is not None
        if not is_usb and not include_legacy and _LEGACY_TTY.match(port.device or ""):
            continue
        ports.append(
            SerialPortInfo(
                device=port.device,
                description=port.description or "n/a",
                hwid=port.hwid or "",
                is_usb=is_usb,
            )
        )
    # USB devices first, then alphabetical so the ST-Link VCP is easy to spot.
    ports.sort(key=lambda p: (not p.is_usb, p.device))
    return ports


def _query_audio() -> tuple[list[dict], list[dict]]:
    import sounddevice as sd

    devices = sd.query_devices()
    host_apis = sd.query_hostapis()
    return list(devices), list(host_apis)


def _executable() -> str:
    import sys

    return sys.executable


def _audio_devices(want_output: bool) -> list[AudioDeviceInfo]:
    try:
        devices, host_apis = _query_audio()
    except Exception as exc:  # pragma: no cover - depends on host audio stack
        logger.error("sounddevice/PortAudio unavailable (%s) — run with the venv "
                     "python (sys.executable=%s)", exc, _executable())
        return []
    result: list[AudioDeviceInfo] = []
    for index, dev in enumerate(devices):
        channels = dev["max_output_channels"] if want_output else dev["max_input_channels"]
        if channels <= 0:
            continue
        api_index = dev.get("hostapi", 0)
        api_name = host_apis[api_index]["name"] if 0 <= api_index < len(host_apis) else "?"
        result.append(
            AudioDeviceInfo(
                index=index,
                name=dev["name"],
                host_api=api_name,
                max_input_channels=dev["max_input_channels"],
                max_output_channels=dev["max_output_channels"],
                default_samplerate=dev.get("default_samplerate", 0.0),
            )
        )
    return result


def list_output_devices() -> list[AudioDeviceInfo]:
    """Return audio devices that can play sound."""
    return _audio_devices(want_output=True)


def list_input_devices() -> list[AudioDeviceInfo]:
    """Return audio devices that can capture sound (for the PC-mic source)."""
    return _audio_devices(want_output=False)


def default_output_index() -> Optional[int]:
    try:
        import sounddevice as sd

        default = sd.default.device
        out = default[1] if isinstance(default, (list, tuple)) else default
        return int(out) if out is not None and out >= 0 else None
    except Exception:  # pragma: no cover
        return None
