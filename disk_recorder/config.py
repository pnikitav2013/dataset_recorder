"""Tunable settings for the disk_recorder pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Settings:
    """Static configuration shared across the pipeline.

    The defaults match the firmware: the STM32N6 mic_stream emits mono PCM16 at
    16 kHz (see ``src/middleware/mic_stream/mic_stream.h``) over UART at the rate
    used by ``py_recorder`` (921600 baud).
    """

    # --- transport / capture format ---
    baud: int = 921600
    sample_rate: int = 16000          # board stream rate; capture/save rate
    channels: int = 1
    sample_width: int = 2             # bytes per PCM16 sample

    # --- timing / synchronisation ---
    # The board ships data in ~0.85 s bursts (13600-sample half buffer), so we
    # capture a generous window and align with cross-correlation afterwards.
    record_headroom_s: float = 1.2    # extra capture time beyond the file length
    max_extra_ms: int = 100           # allowed length increase of the saved clip

    # --- retry / quality gates ---
    max_retries: int = 5              # replays per file before marking it failed
    min_correlation: float = 0.10     # below this, alignment is treated as a miss

    # --- file naming ---
    # A re-recorded output is named ``<original_stem>_R_<prefix>.wav`` where
    # ``<prefix>`` identifies the capturing input device (slot prefix). The
    # marker also flags outputs so a re-scan never treats them as sources.
    rerecord_marker: str = "_R_"
    audio_exts: tuple[str, ...] = field(
        default_factory=lambda: (".wav", ".flac", ".ogg", ".mp3", ".aiff")
    )

    @property
    def extra_samples(self) -> int:
        """Number of samples corresponding to :attr:`max_extra_ms`."""
        return round(self.max_extra_ms / 1000.0 * self.sample_rate)
