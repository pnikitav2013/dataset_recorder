"""Tkinter window: controls, progress, per-device status and spectrograms.

The window is the only UI (no web API). It enumerates devices and exposes a
**five-slot input panel**: each slot is one of three variants — *Off*, *STM32
(UART)* or *PC microphone* — with its own manually entered **prefix** that names
the slot's re-recorded outputs (``<original>_R_<prefix>.wav``). The whole setup
(folder, output device, the three slots) is persisted to a JSON file in the
launch directory so it need not be re-entered after a restart.

On *Start* the enabled slots become capture channels; the pipeline plays each
file once and records every channel at the same time, saving only when all
succeed. The window polls :class:`SessionState` to refresh progress, the
per-channel sync indicators and the stacked per-device spectrogram. Tk widgets
are only ever touched from the main thread.
"""

from __future__ import annotations

import logging
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from . import devices
from .appconfig import (OUTPUT_ROUTES, SLOT_COUNT, SLOT_TYPES, TYPE_BOARD,
                        TYPE_MIC, TYPE_OFF, AppConfig, Schedule, SlotConfig)
from .config import Settings
from .pipeline import Channel, Pipeline
from .playback import Player
from .serial_link import SerialLink
from .sources import BoardSource, MicSource
from .state import SessionState
from .syncstate import REC_COLORS, SYNC_COLORS, RecState, SyncState

logger = logging.getLogger("disk_recorder.gui")

# Slot variant <-> display label mapping for the per-slot type selector.
TYPE_DISPLAY = {
    TYPE_OFF: "Off",
    TYPE_BOARD: "STM32 (UART)",
    TYPE_MIC: "PC microphone",
}
DISPLAY_TYPE = {v: k for k, v in TYPE_DISPLAY.items()}

_NO_INPUT = "(no input devices)"
_NO_PORT = "(no serial ports — is pyserial installed?)"
_CHANNEL_AUTO = "auto (mix)"

# Output-routing variant <-> display label mapping.
ROUTE_DISPLAY = {
    "both": "Both speakers",
    "left": "Left speaker",
    "right": "Right speaker",
}
DISPLAY_ROUTE = {v: k for k, v in ROUTE_DISPLAY.items()}


@dataclass
class _SlotSpec:
    """Validated capture request for one enabled slot (no source built yet)."""

    kind: str               # TYPE_BOARD or TYPE_MIC
    prefix: str
    channel: int = -1
    port: str = ""          # board
    baud: int = 0           # board
    device_index: Optional[int] = None   # mic
    device_name: str = ""   # mic


class SlotPanel:
    """Widgets and logic for one input-device slot."""

    def __init__(self, parent: tk.Widget, index: int, settings: Settings,
                 on_type_change) -> None:
        self._settings = settings
        self._index = index
        self._on_type_change = on_type_change
        self._serial_ports: list[devices.SerialPortInfo] = []
        self._input_devices: list[devices.AudioDeviceInfo] = []
        self._want_serial = ""      # desired (saved) serial device to reselect
        self._want_input = ""       # desired (saved) input device name to reselect

        frame = ttk.LabelFrame(parent, text=f"Input device {index + 1}")
        frame.pack(fill="x", padx=4, pady=4)
        self._frame = frame

        # Row 0: type + prefix.
        ttk.Label(frame, text="Type:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self._type_var = tk.StringVar(value=TYPE_DISPLAY[TYPE_OFF])
        self._type_combo = ttk.Combobox(
            frame, state="readonly", width=16, textvariable=self._type_var,
            values=[TYPE_DISPLAY[t] for t in SLOT_TYPES])
        self._type_combo.grid(row=0, column=1, sticky="w", padx=4)
        self._type_combo.bind("<<ComboboxSelected>>", lambda _e: self._type_changed())

        ttk.Label(frame, text="Prefix:").grid(row=0, column=2, sticky="e", padx=4)
        self._prefix_var = tk.StringVar(value=f"mic{index + 1}")
        self._prefix_entry = ttk.Entry(frame, textvariable=self._prefix_var, width=14)
        self._prefix_entry.grid(row=0, column=3, sticky="w", padx=4)

        # Row 1: STM32 COM port + baud.
        ttk.Label(frame, text="COM port:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self._port_combo = ttk.Combobox(frame, state="readonly", width=40)
        self._port_combo.grid(row=1, column=1, columnspan=2, sticky="we", padx=4)
        ttk.Label(frame, text="Baud:").grid(row=1, column=3, sticky="e", padx=4)
        self._baud_var = tk.StringVar(value=str(settings.baud))
        self._baud_entry = ttk.Entry(frame, textvariable=self._baud_var, width=10)
        self._baud_entry.grid(row=1, column=4, sticky="w", padx=4)

        # Row 2: PC mic input device.
        ttk.Label(frame, text="Input device:").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        self._input_combo = ttk.Combobox(frame, state="readonly", width=52)
        self._input_combo.grid(row=2, column=1, columnspan=4, sticky="we", padx=4)

        # Row 3: which channel of a multi-channel device to keep (e.g. the
        # reSpeaker XVF3800 ASR beam), or "auto (mix)" to downmix all channels.
        ttk.Label(frame, text="Channel:").grid(row=3, column=0, sticky="w", padx=4, pady=3)
        self._channel_var = tk.StringVar(value=_CHANNEL_AUTO)
        self._channel_combo = ttk.Combobox(
            frame, state="readonly", width=12, textvariable=self._channel_var,
            values=[_CHANNEL_AUTO] + [str(i) for i in range(8)])
        self._channel_combo.grid(row=3, column=1, sticky="w", padx=4)

        frame.columnconfigure(1, weight=1)
        self._update_enabled()

    # ----- type / state -----

    def _type(self) -> str:
        return DISPLAY_TYPE.get(self._type_var.get(), TYPE_OFF)

    def _type_changed(self) -> None:
        self._update_enabled()
        if self._on_type_change is not None:
            self._on_type_change()

    def _update_enabled(self) -> None:
        kind = self._type()
        is_board = kind == TYPE_BOARD
        is_mic = kind == TYPE_MIC
        enabled = is_board or is_mic
        self._prefix_entry.configure(state="normal" if enabled else "disabled")
        self._port_combo.configure(state="readonly" if is_board else "disabled")
        self._baud_entry.configure(state="normal" if is_board else "disabled")
        self._input_combo.configure(state="readonly" if is_mic else "disabled")
        self._channel_combo.configure(state="readonly" if is_mic else "disabled")

    # ----- devices -----

    def set_devices(self, serial_ports: list[devices.SerialPortInfo],
                    input_devices: list[devices.AudioDeviceInfo]) -> None:
        self._serial_ports = serial_ports
        self._input_devices = input_devices
        self._port_combo["values"] = [p.label for p in serial_ports] or [_NO_PORT]
        self._input_combo["values"] = (
            [f"{d.label} — {d.max_input_channels}ch" for d in input_devices] or [_NO_INPUT])
        self._select_serial(self._want_serial)
        self._select_input(self._want_input)

    def _select_serial(self, device_str: str) -> None:
        idx = next((i for i, p in enumerate(self._serial_ports)
                    if p.device == device_str), 0 if self._serial_ports else -1)
        if idx >= 0:
            self._port_combo.current(idx)

    def _select_input(self, name: str) -> None:
        idx = next((i for i, d in enumerate(self._input_devices)
                    if d.name == name), 0 if self._input_devices else -1)
        if idx >= 0:
            self._input_combo.current(idx)

    def _channel_value(self) -> int:
        value = self._channel_var.get()
        if value == _CHANNEL_AUTO:
            return -1
        try:
            return int(value)
        except ValueError:
            return -1

    # ----- config <-> widgets -----

    def apply_config(self, cfg: SlotConfig) -> None:
        self._type_var.set(TYPE_DISPLAY.get(cfg.type, TYPE_DISPLAY[TYPE_OFF]))
        if cfg.prefix:
            self._prefix_var.set(cfg.prefix)
        self._baud_var.set(str(cfg.baud))
        self._channel_var.set(_CHANNEL_AUTO if cfg.channel < 0 else str(cfg.channel))
        self._want_serial = cfg.serial_port
        self._want_input = cfg.input_device
        self._select_serial(cfg.serial_port)
        self._select_input(cfg.input_device)
        self._update_enabled()

    def read_config(self) -> SlotConfig:
        port = ""
        if 0 <= self._port_combo.current() < len(self._serial_ports):
            port = self._serial_ports[self._port_combo.current()].device
        input_name = ""
        if 0 <= self._input_combo.current() < len(self._input_devices):
            input_name = self._input_devices[self._input_combo.current()].name
        try:
            baud = int(self._baud_var.get())
        except ValueError:
            baud = self._settings.baud
        return SlotConfig(type=self._type(), prefix=self._prefix_var.get().strip(),
                          serial_port=port, baud=baud, input_device=input_name,
                          channel=self._channel_value())

    # ----- spec building -----

    def build_spec(self) -> Optional[_SlotSpec]:
        """Validate the slot and return a :class:`_SlotSpec`, or ``None`` if Off.

        Raises ``ValueError`` with a user-facing message on misconfiguration.
        Source objects are built later by the App so slots on the same mic can
        share a single stream.
        """
        kind = self._type()
        if kind == TYPE_OFF:
            return None
        prefix = self._prefix_var.get().strip()
        if not prefix:
            raise ValueError(f"Input device {self._index + 1}: enter a prefix name.")

        if kind == TYPE_BOARD:
            if not self._serial_ports or self._port_combo.current() < 0:
                raise ValueError(f"Input device {self._index + 1}: choose a COM port.")
            try:
                baud = int(self._baud_var.get())
            except ValueError:
                raise ValueError(f"Input device {self._index + 1}: baud must be a number.")
            port = self._serial_ports[self._port_combo.current()]
            return _SlotSpec(kind=TYPE_BOARD, prefix=prefix, port=port.device, baud=baud)

        # TYPE_MIC
        if not self._input_devices or self._input_combo.current() < 0:
            raise ValueError(f"Input device {self._index + 1}: choose an input device.")
        dev = self._input_devices[self._input_combo.current()]
        return _SlotSpec(kind=TYPE_MIC, prefix=prefix, channel=self._channel_value(),
                         device_index=dev.index, device_name=dev.name)


class App:
    def __init__(self, root: tk.Tk, settings: Settings) -> None:
        self._root = root
        self._settings = settings
        self._state = SessionState()
        self._pipeline = Pipeline(settings, self._state)
        self._config = AppConfig.load()

        self._output_devices: list[devices.AudioDeviceInfo] = []
        self._input_devices: list[devices.AudioDeviceInfo] = []
        self._serial_ports: list[devices.SerialPortInfo] = []
        self._slots: list[SlotPanel] = []
        self._channel_indicators: list[tk.Label] = []
        self._channel_labels: tuple[str, ...] = ()
        self._fig_token = 0
        self._canvas = None

        root.title("disk_recorder — STM32 multi-mic re-recorder")
        root.geometry(f"{self._config.window_width}x{self._config.window_height}")

        # Two-column layout: all the configuration slots live in a left column;
        # progress and spectrograms fill the right column.
        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)
        self._left = ttk.Frame(main)
        self._left.pack(side="left", fill="y", padx=(8, 4), pady=6)
        self._right = ttk.Frame(main)
        self._right.pack(side="left", fill="both", expand=True)

        self._build_controls()
        self._build_status()
        self._build_canvas()
        self.refresh_devices()
        self._apply_config()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        # Persist the window size whenever the user resizes it (debounced so the
        # config file is not rewritten on every intermediate <Configure> event).
        self._resize_job: Optional[str] = None
        root.bind("<Configure>", self._on_configure)
        self._tick()

    # ----- layout -----

    def _build_controls(self) -> None:
        frame = ttk.LabelFrame(self._left, text="Setup")
        frame.pack(fill="both", expand=True)

        # folder
        ttk.Label(frame, text="Folder:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self._folder_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self._folder_var, width=70).grid(
            row=0, column=1, sticky="we", padx=4)
        ttk.Button(frame, text="Browse…", command=self._browse).grid(row=0, column=2, padx=4)

        # output device
        ttk.Label(frame, text="Output device:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self._output_combo = ttk.Combobox(frame, state="readonly", width=68)
        self._output_combo.grid(row=1, column=1, sticky="we", padx=4)
        frame.columnconfigure(1, weight=1)

        # output routing: mono playback to left / right / both speakers
        ttk.Label(frame, text="Output routing:").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        self._routing_var = tk.StringVar(value=ROUTE_DISPLAY["both"])
        self._routing_combo = ttk.Combobox(
            frame, state="readonly", width=20, textvariable=self._routing_var,
            values=[ROUTE_DISPLAY[r] for r in OUTPUT_ROUTES])
        self._routing_combo.grid(row=2, column=1, sticky="w", padx=4)

        # working-hours schedule: auto-pause overnight, resume in the morning
        sched = ttk.Frame(frame)
        sched.grid(row=3, column=0, columnspan=3, sticky="w", padx=4, pady=3)
        self._sched_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(sched, text="Working hours — pause at",
                        variable=self._sched_enabled).pack(side="left")
        self._sched_end = tk.StringVar(value="22:00")
        ttk.Entry(sched, textvariable=self._sched_end, width=7).pack(side="left", padx=4)
        ttk.Label(sched, text="resume at").pack(side="left")
        self._sched_start = tk.StringVar(value="08:00")
        ttk.Entry(sched, textvariable=self._sched_start, width=7).pack(side="left", padx=4)
        ttk.Label(sched, text="(HH:MM)").pack(side="left")

        # input-device slots
        slots_frame = ttk.LabelFrame(frame, text="Input devices (recorded simultaneously)")
        slots_frame.grid(row=4, column=0, columnspan=3, sticky="we", padx=4, pady=6)
        for i in range(SLOT_COUNT):
            self._slots.append(SlotPanel(slots_frame, i, self._settings, self._on_slot_type_change))

        # buttons
        button_row = ttk.Frame(frame)
        button_row.grid(row=5, column=0, columnspan=3, sticky="w", padx=4, pady=6)
        ttk.Button(button_row, text="Refresh devices", command=self.refresh_devices).pack(side="left")
        self._start_btn = ttk.Button(button_row, text="Start", command=self._start)
        self._start_btn.pack(side="left", padx=6)
        self._stop_btn = ttk.Button(button_row, text="Stop", command=self._stop, state="disabled")
        self._stop_btn.pack(side="left")

    def _build_status(self) -> None:
        frame = ttk.LabelFrame(self._right, text="Progress")
        frame.pack(fill="x", padx=8, pady=6)

        # Recording state + a dynamic per-channel sync indicator row.
        indicators = ttk.Frame(frame)
        indicators.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(indicators, text="Recording:").pack(side="left")
        self._rec_indicator = tk.Label(indicators, text="—", width=14, relief="raised",
                                       bg="#9e9e9e", fg="white", padx=6, pady=3)
        self._rec_indicator.pack(side="left", padx=2)

        self._channels_frame = ttk.Frame(frame)
        self._channels_frame.pack(fill="x", padx=6, pady=(4, 0))

        self._progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self._progress.pack(fill="x", padx=6, pady=6)

        self._status_var = tk.StringVar(value="idle")
        ttk.Label(frame, textvariable=self._status_var, font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", padx=6)

        grid = ttk.Frame(frame)
        grid.pack(fill="x", padx=6, pady=4)
        self._labels: dict[str, tk.StringVar] = {}
        fields = [
            ("Current file", "current_file"),
            ("Done / total", "progress"),
            ("Registered errors", "errors"),
            ("Failed files", "failed"),
            ("Avg record time", "avg"),
        ]
        for col, (title, key) in enumerate(fields):
            ttk.Label(grid, text=title + ":").grid(row=0, column=col, sticky="w", padx=6)
            var = tk.StringVar(value="-")
            self._labels[key] = var
            ttk.Label(grid, textvariable=var).grid(row=1, column=col, sticky="w", padx=6)

    def _build_canvas(self) -> None:
        self._canvas_frame = ttk.LabelFrame(
            self._right, text="Last recording — per-device log-mel spectrograms")
        self._canvas_frame.pack(fill="both", expand=True, padx=8, pady=6)

    # ----- config persistence -----

    def _apply_config(self) -> None:
        self._folder_var.set(self._config.folder)
        self._select_output(self._config.output_device)
        self._routing_var.set(ROUTE_DISPLAY.get(self._config.output_routing,
                                                ROUTE_DISPLAY["both"]))
        self._sched_enabled.set(self._config.schedule.enabled)
        self._sched_start.set(self._config.schedule.start)
        self._sched_end.set(self._config.schedule.end)
        for slot, cfg in zip(self._slots, self._config.slots):
            slot.apply_config(cfg)
            slot.set_devices(self._serial_ports, self._input_devices)

    def _collect_config(self) -> AppConfig:
        output_name = ""
        if 0 <= self._output_combo.current() < len(self._output_devices):
            output_name = self._output_devices[self._output_combo.current()].name
        return AppConfig(
            folder=self._folder_var.get().strip(),
            output_device=output_name,
            output_routing=DISPLAY_ROUTE.get(self._routing_var.get(), "both"),
            window_width=self._root.winfo_width(),
            window_height=self._root.winfo_height(),
            schedule=Schedule(
                enabled=self._sched_enabled.get(),
                start=self._sched_start.get().strip(),
                end=self._sched_end.get().strip(),
            ),
            slots=[slot.read_config() for slot in self._slots],
        )

    def _save_config(self) -> None:
        self._collect_config().save()

    def _on_configure(self, event: tk.Event) -> None:
        # Only react to top-level resize events, and debounce: save 600 ms after
        # the user stops dragging the window edge.
        if event.widget is not self._root:
            return
        if self._resize_job is not None:
            self._root.after_cancel(self._resize_job)
        self._resize_job = self._root.after(600, self._save_config)

    def _on_close(self) -> None:
        self._save_config()
        self._root.destroy()

    def _on_slot_type_change(self) -> None:  # placeholder hook for live updates
        pass

    # ----- device handling -----

    def refresh_devices(self) -> None:
        self._output_devices = devices.list_output_devices()
        self._input_devices = devices.list_input_devices()
        self._serial_ports = devices.list_serial_ports()

        logger.info("devices: %d output, %d input, %d serial port(s)",
                    len(self._output_devices), len(self._input_devices),
                    len(self._serial_ports))

        self._output_combo["values"] = (
            [d.label for d in self._output_devices]
            or ["(no output devices — is sounddevice installed?)"])
        if self._output_devices and self._output_combo.current() < 0:
            self._output_combo.current(self._preferred_output_index())
        elif not self._output_devices:
            self._output_combo.current(0)

        for slot in self._slots:
            slot.set_devices(self._serial_ports, self._input_devices)

    def _select_output(self, name: str) -> None:
        idx = next((i for i, d in enumerate(self._output_devices) if d.name == name), None)
        if idx is not None:
            self._output_combo.current(idx)
        elif self._output_devices:
            self._output_combo.current(self._preferred_output_index())

    def _preferred_output_index(self) -> int:
        """Prefer the system mixer (pulse/pipewire/default) which resamples
        freely, over raw ALSA ``hw:`` / HDMI outputs that reject odd rates."""
        for keyword in ("pipewire", "pulse", "default"):
            for i, dev in enumerate(self._output_devices):
                if keyword in dev.name.lower():
                    return i
        default_out = devices.default_output_index()
        return next((i for i, d in enumerate(self._output_devices)
                     if d.index == default_out), 0)

    def _browse(self) -> None:
        folder = filedialog.askdirectory(title="Choose audio folder")
        if folder:
            self._folder_var.set(folder)

    # ----- run control -----

    def _start(self) -> None:
        folder = self._folder_var.get().strip()
        if not folder:
            messagebox.showwarning("disk_recorder", "Choose an audio folder first.")
            return
        if not self._output_devices or self._output_combo.current() < 0:
            messagebox.showwarning("disk_recorder", "Choose an output device.")
            return

        try:
            specs = [s for slot in self._slots if (s := slot.build_spec()) is not None]
        except ValueError as exc:
            messagebox.showwarning("disk_recorder", str(exc))
            return
        if not specs:
            messagebox.showwarning("disk_recorder", "Enable at least one input device slot.")
            return
        prefixes = [s.prefix for s in specs]
        if len(set(prefixes)) != len(prefixes):
            messagebox.showwarning("disk_recorder", "Each enabled slot needs a unique prefix.")
            return
        schedule = Schedule(
            enabled=self._sched_enabled.get(),
            start=self._sched_start.get().strip(),
            end=self._sched_end.get().strip(),
        )
        if schedule.enabled and not schedule.valid():
            messagebox.showwarning(
                "disk_recorder",
                "Working hours need two distinct HH:MM times (e.g. 08:00 and 22:00).")
            return
        channels = self._assemble_channels(specs)

        self._save_config()
        output_index = self._output_devices[self._output_combo.current()].index
        routing = DISPLAY_ROUTE.get(self._routing_var.get(), "both")
        player = Player(output_index, routing)
        self._pipeline.start(folder, channels, player, schedule)
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")

    def _assemble_channels(self, specs: list[_SlotSpec]) -> list[Channel]:
        """Build capture channels, sharing one MicSource per physical device so
        two slots can read different channels of the same mic without opening
        the (exclusive) ``hw:`` device twice."""
        sr = self._settings.sample_rate
        mic_cache: dict[int, MicSource] = {}
        channels: list[Channel] = []
        for spec in specs:
            if spec.kind == TYPE_BOARD:
                source = BoardSource(SerialLink(spec.port, spec.baud), sr)
                label = f"{spec.prefix} · {spec.port}"
                channels.append(Channel(source=source, prefix=spec.prefix, label=label))
            else:
                source = mic_cache.get(spec.device_index)
                if source is None:
                    source = MicSource(spec.device_index, sr)
                    mic_cache[spec.device_index] = source
                ch_label = "mix" if spec.channel < 0 else f"ch{spec.channel}"
                label = f"{spec.prefix} · {spec.device_name} [{ch_label}]"
                channels.append(Channel(source=source, prefix=spec.prefix,
                                        label=label, channel=spec.channel))
        return channels

    def _stop(self) -> None:
        self._pipeline.stop()
        self._status_var.set("stopping…")

    # ----- polling -----

    def _tick(self) -> None:
        snap = self._state.snapshot()
        if snap.total > 0:
            # In-flight fraction = the slowest channel (a file is done only once
            # every channel has captured), falling back to the aggregate.
            capture = (min((c.progress for c in snap.channels), default=snap.capture_progress)
                       if snap.channels else snap.capture_progress)
            overall = (snap.done + capture) / snap.total * 100.0
            self._progress["value"] = min(100.0, overall)
        else:
            self._progress["value"] = 0
        self._status_var.set(f"{snap.status}")
        self._update_indicator(self._rec_indicator, RecState, REC_COLORS, snap.rec_state)
        self._refresh_channel_indicators(snap.channels)
        self._labels["current_file"].set(snap.current_file or "-")
        self._labels["progress"].set(f"{snap.done} / {snap.total}")
        self._labels["errors"].set(str(snap.errors))
        self._labels["failed"].set(str(snap.failed))
        self._labels["avg"].set(f"{snap.avg_record_s:.2f} s" if snap.avg_record_s else "-")

        running = self._pipeline.is_running()
        self._start_btn.configure(state="disabled" if running else "normal")
        self._stop_btn.configure(state="normal" if running else "disabled")

        self._refresh_figure()
        self._root.after(500, self._tick)

    def _refresh_channel_indicators(self, channels) -> None:
        labels = tuple(c.label for c in channels)
        if labels != self._channel_labels:
            for widget in self._channels_frame.winfo_children():
                widget.destroy()
            self._channel_indicators = []
            for ch in channels:
                ttk.Label(self._channels_frame, text=f"{ch.label}:").pack(side="left", padx=(6, 2))
                indicator = tk.Label(self._channels_frame, text="—", width=12, relief="raised",
                                     bg="#9e9e9e", fg="white", padx=4, pady=2)
                indicator.pack(side="left", padx=(0, 8))
                self._channel_indicators.append(indicator)
            self._channel_labels = labels
        for indicator, ch in zip(self._channel_indicators, channels):
            self._update_indicator(indicator, SyncState, SYNC_COLORS, ch.sync_state)

    @staticmethod
    def _update_indicator(label: tk.Label, enum_cls, colors: dict, value: str) -> None:
        try:
            member = enum_cls(value)
            bg, fg = colors[member]
            text = member.name
        except (ValueError, KeyError):
            bg, fg, text = "#9e9e9e", "white", value
        label.configure(text=text, bg=bg, fg=fg)

    def _refresh_figure(self) -> None:
        token, figure = self._state.figure_for(self._fig_token)
        if figure is None:
            return
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
        self._canvas = FigureCanvasTkAgg(figure, master=self._canvas_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._fig_token = token


def main() -> int:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("python: %s", sys.executable)
    root = tk.Tk()
    App(root, Settings())
    root.mainloop()
    return 0
