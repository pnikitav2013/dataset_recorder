"""Enumeration of serial ports and host audio devices.

This is the *connection discovery* layer used to populate the GUI selectors:
which COM port the STM32 (ST-Link Virtual COM Port) is on, and which output /
input audio devices are available.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("disk_recorder.devices")

# Legacy 16550 placeholders Linux always exposes (/dev/ttyS0..N) with no USB
# identity; they drown out the real ST-Link VCP, so they are hidden by default.
_LEGACY_TTY = re.compile(r"^/dev/ttyS\d+$")

# Windows host APIs, best first. WASAPI in shared mode is the modern path and
# survives long unattended runs; DirectSound and MME both go through the legacy
# ``winmm`` layer. ``Windows WDM-KS`` is deliberately last: it takes the device
# exclusively and is by far the most fragile under repeated use, which is
# exactly what wedges system audio on a multi-day session.
_WINDOWS_API_PRIORITY = ("windows wasapi", "windows directsound", "mme")
_DISCOURAGED_APIS = ("windows wdm-ks",)

# On Linux the sound server (which resamples freely) beats a raw ALSA ``hw:``
# device, which only accepts its native rate/channel count.
_POSIX_NAME_PRIORITY = ("pipewire", "pulse", "default")


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


def log_enumeration(outputs: list[AudioDeviceInfo],
                    inputs: list[AudioDeviceInfo]) -> None:
    """Log every device PortAudio reports, exactly as enumerated.

    Nothing in this module ever *filters* the device lists — host-API ranking
    only decides which entry is preselected. This dump is what answers "is my
    device missing, or just not selected?" without guessing.
    """
    logger.info("audio devices: %d with outputs, %d with inputs",
                len(outputs), len(inputs))
    for kind, listing in (("output", outputs), ("input", inputs)):
        for device in listing:
            logger.info("  %-6s [%d] %s | %s | in=%d out=%d @ %.0f Hz%s",
                        kind, device.index, device.name, device.host_api,
                        device.max_input_channels, device.max_output_channels,
                        device.default_samplerate,
                        "  <- discouraged host API" if is_discouraged(device) else "")


def host_api_rank(device: AudioDeviceInfo) -> int:
    """Rank a device by how well its host API survives a multi-day run.

    Lower is better. Non-Windows hosts have a single sensible API, so they all
    rank equally and the name-based preference decides instead.
    """
    api = device.host_api.strip().lower()
    if api in _DISCOURAGED_APIS:
        return len(_WINDOWS_API_PRIORITY) + 1
    if api in _WINDOWS_API_PRIORITY:
        return _WINDOWS_API_PRIORITY.index(api)
    return len(_WINDOWS_API_PRIORITY)


def is_discouraged(device: AudioDeviceInfo) -> bool:
    """True for host APIs known to destabilise long unattended sessions."""
    return device.host_api.strip().lower() in _DISCOURAGED_APIS


def preferred_index(devices: list[AudioDeviceInfo], want_output: bool = True) -> int:
    """Index (into ``devices``) of the device to select by default.

    On Windows this picks the best host API (WASAPI ≫ DirectSound ≫ MME, never
    WDM-KS) for the system default device; elsewhere it prefers the sound
    server over raw ALSA devices. Returns 0 when nothing better is known.
    """
    if not devices:
        return 0

    if sys.platform != "win32":
        for keyword in _POSIX_NAME_PRIORITY:
            for i, device in enumerate(devices):
                if keyword in device.name.lower():
                    return i

    default = default_output_index() if want_output else default_input_index()
    default_name = next((d.name for d in devices if d.index == default), None)

    # Same physical device, best host API: PortAudio lists one entry per API.
    candidates = [i for i, d in enumerate(devices)
                  if default_name is not None and d.name == default_name]
    if not candidates:
        candidates = list(range(len(devices)))
    return min(candidates, key=lambda i: (host_api_rank(devices[i]), i))


def _default_device_index(slot: int) -> Optional[int]:
    try:
        import sounddevice as sd

        default = sd.default.device
        value = default[slot] if isinstance(default, (list, tuple)) else default
        return int(value) if value is not None and value >= 0 else None
    except Exception:  # pragma: no cover
        return None


def default_input_index() -> Optional[int]:
    return _default_device_index(0)


def default_output_index() -> Optional[int]:
    return _default_device_index(1)
