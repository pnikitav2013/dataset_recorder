import unittest

from reliable_transport import DATA_START, ReliableTransport, TransportWarning, make_frame


class ReliableTransportTest(unittest.TestCase):
    def test_transmits_frames_without_acknowledgements(self) -> None:
        wire: list[bytes] = []
        transport = ReliableTransport(wire.append)

        transport.send(b"first")
        transport.send(b"second")
        transport.process()

        self.assertEqual(wire, [make_frame(0, b"first"), make_frame(1, b"second")])

    def test_crc_and_sequence_warnings_continue_the_stream(self) -> None:
        warnings: list[TransportWarning] = []
        received: list[bytes] = []
        transport = ReliableTransport(
            lambda data: len(data),
            lambda _sequence, payload: received.append(payload),
            lambda kind, *_: warnings.append(kind),
        )
        bad = bytearray(make_frame(2, b"bad"))
        bad[-1] ^= 1
        transport.process(bytes(bad))
        transport.process(make_frame(5, b"first"))
        transport.process(make_frame(7, b"after-gap"))

        self.assertIn(TransportWarning.CRC, warnings)
        self.assertIn(TransportWarning.SEQUENCE_GAP, warnings)
        self.assertEqual(received, [b"first", b"after-gap"])

    def test_frame_header(self) -> None:
        frame = make_frame(12)
        self.assertEqual(frame[:4], bytes((DATA_START, 12, 0, 0)))
        self.assertEqual(len(frame), 8)


if __name__ == "__main__":
    unittest.main()
