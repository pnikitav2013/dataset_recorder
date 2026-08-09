"""Serial port wrapper with a resilient background reader thread.

Owns the raw byte pump only: it opens the port, continuously reads incoming
bytes in a daemon thread and hands them to a consumer callback (the board
source feeds them into ``ReliableTransport``). On a read error (the common
Linux "device reports readiness to read but returned no data" hiccup, or a USB
re-enumeration) it closes, notifies a disconnect callback and keeps trying to
reopen, so a transient glitch drops the lock instead of killing the run.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("disk_recorder.serial_link")

Consumer = Callable[[bytes], None]
DisconnectCallback = Callable[[], None]


class SerialLink:
    """Open a serial port and pump received bytes to a consumer."""

    def __init__(self, port: str, baud: int, read_timeout: float = 0.05) -> None:
        self._port_name = port
        self._baud = baud
        self._read_timeout = read_timeout
        self._serial = None
        self._consumer: Optional[Consumer] = None
        self._on_disconnect: Optional[DisconnectCallback] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def set_consumer(self, consumer: Consumer) -> None:
        self._consumer = consumer

    def set_disconnect_callback(self, callback: DisconnectCallback) -> None:
        self._on_disconnect = callback

    def open(self) -> None:
        self._open_port()
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, name="serial-reader", daemon=True)
        self._thread.start()

    def _open_port(self) -> None:
        import serial

        self._serial = serial.Serial(self._port_name, self._baud, timeout=self._read_timeout)
        logger.info("opened %s @ %u baud", self._port_name, self._baud)

    def _reader(self) -> None:
        while not self._stop.is_set():
            port = self._serial
            if port is None:
                if not self._try_reopen():
                    time.sleep(0.5)
                    continue
                port = self._serial
            try:
                data = port.read(port.in_waiting or 1)
            except Exception as exc:
                logger.error("serial read failed: %s — will reconnect", exc)
                if self._on_disconnect is not None:
                    self._on_disconnect()
                self._safe_close()
                time.sleep(0.3)
                continue
            if data and self._consumer is not None:
                self._consumer(data)

    def _try_reopen(self) -> bool:
        try:
            self._open_port()
            return True
        except Exception as exc:
            logger.debug("reopen %s failed: %s", self._port_name, exc)
            return False

    def _safe_close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:  # pragma: no cover
                pass
        self._serial = None

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._safe_close()
        self._thread = None
        logger.info("closed %s", self._port_name)
