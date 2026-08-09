"""disk_recorder — re-record a folder of audio through multiple input devices.

The package plays every audio file in a folder once through a chosen output
device while simultaneously capturing up to three enabled input devices (STM32N6
mic streams over the ``reliable_transport`` UART protocol and/or PC microphones),
aligns each capture to the played reference, and — only once every device
succeeds — stores each next to the original as ``<name>_R_<prefix>.wav``.

Each module owns one concern:

* :mod:`disk_recorder.config`      — tunable settings
* :mod:`disk_recorder.appconfig`   — JSON-persisted GUI configuration (input slots)
* :mod:`disk_recorder.devices`     — serial port / audio device enumeration
* :mod:`disk_recorder.serial_link` — serial port + background reader thread
* :mod:`disk_recorder.sources`     — capture from the board (or a PC microphone)
* :mod:`disk_recorder.playback`    — load and play reference audio
* :mod:`disk_recorder.sync`        — cross-correlation alignment
* :mod:`disk_recorder.mel`         — log-mel spectrogram figure
* :mod:`disk_recorder.storage`     — folder scan / WAV writing / deletion
* :mod:`disk_recorder.pipeline`    — orchestration with retries
* :mod:`disk_recorder.state`       — thread-safe session state
* :mod:`disk_recorder.gui`         — Tkinter window
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
