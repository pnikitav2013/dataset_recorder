"""Framed one-way streaming transport shared with the STM32 firmware.

Frame: ``A5 sequence length_le payload crc32_le``.  Frames are never retried:
CRC and sequence warnings identify damaged or missing data without delaying the
current stream.
"""

from __future__ import annotations

import binascii
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Deque, Optional

DATA_START = 0xA5
MAX_PAYLOAD = 1024


class TransportWarning(str, Enum):
    SYNC_SEARCH = "sync_search"
    INVALID_LENGTH = "invalid_length"
    CRC = "crc_mismatch"
    SEQUENCE_GAP = "sequence_gap"
    WRITE_SHORT = "write_short"


@dataclass(frozen=True)
class Packet:
    sequence: int
    payload: bytes


@dataclass
class _WireFrame:
    frame: bytes
    offset: int = 0


WarningCallback = Callable[[TransportWarning, int, str], None]
MessageCallback = Callable[[int, bytes], None]
WriteCallback = Callable[[bytes], int]


def crc32(data: bytes) -> int:
    """Return CRC-32/ISO-HDLC (IEEE 802.3)."""
    return binascii.crc32(data) & 0xFFFFFFFF


def make_frame(sequence: int, payload: bytes = b"") -> bytes:
    """Build one data frame."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload exceeds {MAX_PAYLOAD} bytes")
    header = bytes((DATA_START, sequence & 0xFF)) + len(payload).to_bytes(2, "little")
    return header + payload + crc32(header + payload).to_bytes(4, "little")


class StreamParser:
    """Incremental parser that resynchronizes after damaged stream data."""

    def __init__(self, on_warning: Optional[WarningCallback] = None) -> None:
        self._buffer = bytearray()
        self._on_warning = on_warning

    def _warn(self, warning: TransportWarning, sequence: int = 0, detail: str = "") -> None:
        if self._on_warning is not None:
            self._on_warning(warning, sequence, detail)

    def feed(self, data: bytes) -> list[Packet]:
        self._buffer.extend(data)
        packets: list[Packet] = []
        while self._buffer:
            try:
                start = self._buffer.index(DATA_START)
            except ValueError:
                discarded = len(self._buffer)
                self._buffer.clear()
                self._warn(TransportWarning.SYNC_SEARCH, detail=f"discarded {discarded} noise byte(s)")
                break
            if start:
                del self._buffer[:start]
                self._warn(TransportWarning.SYNC_SEARCH, detail=f"discarded {start} noise byte(s)")
            if len(self._buffer) < 4:
                break
            sequence = self._buffer[1]
            payload_length = int.from_bytes(self._buffer[2:4], "little")
            if payload_length > MAX_PAYLOAD:
                self._warn(TransportWarning.INVALID_LENGTH, sequence, f"length={payload_length}")
                del self._buffer[0]
                continue
            frame_length = payload_length + 8
            if len(self._buffer) < frame_length:
                break
            frame = bytes(self._buffer[:frame_length])
            received_crc = int.from_bytes(frame[-4:], "little")
            calculated_crc = crc32(frame[:-4])
            if received_crc != calculated_crc:
                self._warn(TransportWarning.CRC, sequence,
                           f"received=0x{received_crc:04X}, calculated=0x{calculated_crc:04X}")
                del self._buffer[0]
                continue
            packets.append(Packet(sequence, frame[4:-4]))
            del self._buffer[:frame_length]
        return packets


class ReliableTransport:
    """Buffered framed stream with no ACK, waiting, or retransmission."""

    def __init__(self, write: WriteCallback, on_message: Optional[MessageCallback] = None,
                 on_warning: Optional[WarningCallback] = None, max_queue: int = 8) -> None:
        self._write = write
        self._on_message = on_message
        self._on_warning = on_warning
        self._max_queue = max_queue
        self._parser = StreamParser(self._warn)
        self._queue: Deque[tuple[int, bytes]] = deque()
        self._wire: Optional[_WireFrame] = None
        self._next_tx_sequence = 0
        self._rx_expected: Optional[int] = None

    def _warn(self, warning: TransportWarning, sequence: int = 0, detail: str = "") -> None:
        if self._on_warning is not None:
            self._on_warning(warning, sequence, detail)

    def send(self, payload: bytes) -> int:
        """Buffer a payload for one-time transmission and return its sequence."""
        payload = bytes(payload)
        if len(payload) > MAX_PAYLOAD:
            raise ValueError(f"payload exceeds {MAX_PAYLOAD} bytes")
        if len(self._queue) >= self._max_queue:
            raise BufferError("stream transport queue is full")
        sequence = self._next_tx_sequence
        self._next_tx_sequence = (self._next_tx_sequence + 1) & 0xFF
        self._queue.append((sequence, payload))
        return sequence

    def feed(self, data: bytes) -> None:
        """Accept received bytes; valid later frames continue after a loss."""
        for packet in self._parser.feed(data):
            if self._rx_expected is not None and packet.sequence != self._rx_expected:
                self._warn(TransportWarning.SEQUENCE_GAP, packet.sequence,
                           f"expected={self._rx_expected}")
            self._rx_expected = (packet.sequence + 1) & 0xFF
            if self._on_message is not None:
                self._on_message(packet.sequence, packet.payload)

    def _start_frame(self) -> None:
        if self._wire is None and self._queue:
            sequence, payload = self._queue.popleft()
            self._wire = _WireFrame(make_frame(sequence, payload))

    def _flush(self) -> bool:
        self._start_frame()
        if self._wire is None:
            return False
        remaining = self._wire.frame[self._wire.offset:]
        written = self._write(remaining)
        if written is None:
            written = len(remaining)
        written = min(max(int(written), 0), len(remaining))
        if written == 0:
            self._warn(TransportWarning.WRITE_SHORT)
            return False
        self._wire.offset += written
        if self._wire.offset == len(self._wire.frame):
            self._wire = None
            return True
        return False

    def process(self, data: bytes = b"", now: object = None) -> None:
        """Parse input and flush up to eight queued output frames."""
        del now  # Compatibility with prior API; streaming mode has no timer.
        if data:
            self.feed(data)
        for _ in range(8):
            if not self._flush():
                break
