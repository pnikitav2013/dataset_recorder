"""Persistent GUI configuration (input-device slots) stored as JSON.

The window configures **five fixed input-device slots**. Each slot is one of
three variants — ``off`` (unused), ``board`` (STM32 over UART) or ``mic`` (a PC
input device) — and carries the manually editable *prefix* that names its
re-recorded outputs (``<original>_R_<prefix>.wav``).

So the operator does not re-enter everything on every launch, the whole setup
(folder, output device and the three slots) is serialised to a JSON file in the
**current working directory** — i.e. wherever ``python -m disk_recorder`` was
started — and reloaded on the next start.

Device handles are persisted by a *stable* identity (serial port device string,
audio device name) rather than the volatile PortAudio/enumeration index, so a
saved choice still resolves after devices are re-enumerated.
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path

logger = logging.getLogger("disk_recorder.appconfig")

#: Filename written/read in the launch (current working) directory.
CONFIG_FILENAME = "disk_recorder_config.json"

#: Number of fixed input-device slots shown in the GUI.
SLOT_COUNT = 5

# Output routing: the played reference is always downmixed to mono, then sent to
# one or both speakers of a stereo output device.
ROUTE_BOTH = "both"
ROUTE_LEFT = "left"
ROUTE_RIGHT = "right"
OUTPUT_ROUTES = (ROUTE_BOTH, ROUTE_LEFT, ROUTE_RIGHT)

# Slot variants.
TYPE_OFF = "off"
TYPE_BOARD = "board"
TYPE_MIC = "mic"
SLOT_TYPES = (TYPE_OFF, TYPE_BOARD, TYPE_MIC)

_DEFAULT_BAUD = 921600


@dataclass
class Schedule:
    """Optional working-hours window that auto-pauses the run overnight.

    When :attr:`enabled`, the pipeline only records while the local clock is
    inside ``[start, end)`` (``HH:MM``, 24-hour). Outside that window it pauses
    and resumes automatically at :attr:`start`. A window where ``start > end``
    wraps past midnight (e.g. ``22:00``–``06:00`` is active overnight).
    """

    enabled: bool = False
    start: str = "08:00"   # resume time (morning)
    end: str = "22:00"     # pause time (night)

    @staticmethod
    def _to_minutes(hhmm: str) -> int | None:
        try:
            hour_str, minute_str = str(hhmm).strip().split(":")
            hour, minute = int(hour_str), int(minute_str)
        except (ValueError, AttributeError):
            return None
        if 0 <= hour < 24 and 0 <= minute < 60:
            return hour * 60 + minute
        return None

    def valid(self) -> bool:
        s, e = self._to_minutes(self.start), self._to_minutes(self.end)
        return s is not None and e is not None and s != e

    def is_active(self, now: datetime.datetime | None = None) -> bool:
        """True if recording is allowed right now (always True when disabled)."""
        if not self.enabled or not self.valid():
            return True
        s = self._to_minutes(self.start)
        e = self._to_minutes(self.end)
        now = now or datetime.datetime.now()
        cur = now.hour * 60 + now.minute
        if s < e:
            return s <= cur < e
        return cur >= s or cur < e   # wraps past midnight


@dataclass
class SlotConfig:
    """Configuration of one input-device slot."""

    type: str = TYPE_OFF
    prefix: str = ""
    serial_port: str = ""      # board: serial device string, e.g. "/dev/ttyACM0"
    baud: int = _DEFAULT_BAUD  # board: UART baud rate
    input_device: str = ""     # mic: audio input device name
    # mic: host API of the chosen device ("Windows WASAPI", "MME", …). The same
    # microphone is listed once per host API and they behave differently, so the
    # name alone does not identify what the operator picked.
    input_host_api: str = ""
    channel: int = -1          # mic: -1 = downmix all; >=0 = take that channel

    def enabled(self) -> bool:
        return self.type in (TYPE_BOARD, TYPE_MIC)

    @classmethod
    def from_raw(cls, raw: dict) -> "SlotConfig":
        """Build from JSON, ignoring keys this version does not know about."""
        fields = {f.name for f in dataclass_fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in fields})


@dataclass
class AppConfig:
    """Whole-window configuration persisted between runs."""

    folder: str = ""
    output_device: str = ""    # audio output device name
    output_host_api: str = ""  # host API of that device (see SlotConfig)
    output_routing: str = ROUTE_BOTH   # mono playback routed to left/right/both
    window_width: int = 1680   # last main-window width in pixels
    window_height: int = 1440  # last main-window height in pixels
    schedule: Schedule = field(default_factory=Schedule)   # working-hours window
    slots: list[SlotConfig] = field(
        default_factory=lambda: [
            SlotConfig(prefix=f"mic{i + 1}") for i in range(SLOT_COUNT)
        ]
    )

    def __post_init__(self) -> None:
        # Coerce a plain dict (from JSON) into a Schedule.
        if isinstance(self.schedule, dict):
            self.schedule = Schedule(**self.schedule)
        # Always keep exactly SLOT_COUNT slots, tolerating malformed JSON.
        slots = [s if isinstance(s, SlotConfig) else SlotConfig.from_raw(s)
                 for s in self.slots]
        while len(slots) < SLOT_COUNT:
            slots.append(SlotConfig(prefix=f"mic{len(slots) + 1}"))
        self.slots = slots[:SLOT_COUNT]

    # ----- persistence -----

    @staticmethod
    def default_path() -> Path:
        return Path.cwd() / CONFIG_FILENAME

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        path = path or cls.default_path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.info("no saved config at %s — using defaults", path)
            return cls()
        except (OSError, ValueError) as exc:
            logger.warning("could not read config %s (%s) — using defaults", path, exc)
            return cls()
        slots = [SlotConfig.from_raw(s) for s in raw.get("slots", [])]
        routing = raw.get("output_routing", ROUTE_BOTH)
        if routing not in OUTPUT_ROUTES:
            routing = ROUTE_BOTH
        def _dim(key: str, default: int) -> int:
            try:
                return max(1, int(raw.get(key, default)))
            except (TypeError, ValueError):
                return default

        sched_raw = raw.get("schedule", {})
        schedule = Schedule(**sched_raw) if isinstance(sched_raw, dict) else Schedule()
        return cls(
            folder=raw.get("folder", ""),
            output_device=raw.get("output_device", ""),
            output_host_api=raw.get("output_host_api", ""),
            output_routing=routing,
            window_width=_dim("window_width", 1680),
            window_height=_dim("window_height", 1440),
            schedule=schedule,
            slots=slots,
        )

    def save(self, path: Path | None = None) -> None:
        path = path or self.default_path()
        try:
            path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
            logger.info("saved config to %s", path)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            logger.warning("could not save config %s: %s", path, exc)
