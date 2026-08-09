#!/usr/bin/env python3
"""Record STM32 reliable UART logs and optionally send framed messages back."""

from __future__ import annotations

import argparse
import logging
from queue import Empty, Queue
import sys
from threading import Thread

from reliable_transport import ReliableTransport, TransportWarning


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", help="serial port, for example /dev/ttyACM0 or COM5")
    parser.add_argument("--baud", type=int, default=921600, help="UART bitrate (default: 921600)")
    parser.add_argument("--send", action="append", default=[], metavar="TEXT", help="send TEXT after connecting")
    parser.add_argument("--stdin", action="store_true", help="send each line entered on stdin")
    return parser.parse_args()


def stdin_reader(lines: Queue[str]) -> None:
    for line in sys.stdin:
        lines.put(line.rstrip("\n"))


def main() -> int:
    arguments = parse_arguments()
    try:
        import serial
    except ImportError:
        print("pyserial is missing; run ./create_venv.sh first", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("py_recorder")

    def on_warning(warning: TransportWarning, sequence: int, detail: str) -> None:
        suffix = f" ({detail})" if detail else ""
        logger.warning("transport %s, sequence=%u%s", warning.value, sequence, suffix)

    def on_message(sequence: int, payload: bytes) -> None:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            logger.info("RX sequence=%u: %s", sequence, payload.hex(" "))
            return
        # Firmware logger sends a line in several payloads; preserve its format.
        sys.stdout.write(text)
        sys.stdout.flush()

    try:
        with serial.Serial(arguments.port, arguments.baud, timeout=0.05) as port:
            transport = ReliableTransport(
                port.write,
                on_message=on_message,
                on_warning=on_warning,
            )
            for text in arguments.send:
                sequence = transport.send(text.encode("utf-8"))
                logger.info("queued TX sequence=%u: %s", sequence, text)

            lines: Queue[str] = Queue()
            if arguments.stdin:
                Thread(target=stdin_reader, args=(lines,), daemon=True).start()
                logger.info("stdin mode enabled; each entered line is sent as one packet")

            logger.info("connected to %s at %u baud; press Ctrl-C to stop", arguments.port, arguments.baud)
            while True:
                try:
                    while True:
                        text = lines.get_nowait()
                        sequence = transport.send(text.encode("utf-8"))
                        logger.info("queued TX sequence=%u: %s", sequence, text)
                except Empty:
                    pass

                received = port.read(port.in_waiting or 1)
                transport.process(received)
    except KeyboardInterrupt:
        logger.info("stopped")
        return 0
    except (OSError, ValueError) as error:
        logger.error("serial connection failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
