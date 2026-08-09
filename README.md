# py_recorder

Host implementation of the STM32 framed UART stream. It validates CRC and
re-synchronizes after line noise. Frames are not acknowledged or retried, so a
live audio stream continues with the newest valid data after a loss.

Create the local environment once:

```bash
cd py_recorder
./create_venv.sh
```

Record the firmware UART (the default UART speed is 921600):

```bash
.venv/bin/python recorder.py /dev/ttyACM0
```

Record microphone PCM16 data into a WAV file:

```bash
.venv/bin/python microphone_recorder.py /dev/ttyACM0 --output microphone.wav
```

The microphone stream is mono PCM16 at 16000 Hz by default. Use
`--sample-rate` only if the firmware capture rate has been changed.

To send an initial message and then send each terminal line to the MCU:

```bash
.venv/bin/python recorder.py /dev/ttyACM0 --send ping --stdin
```

Wire protocol: `A5 seq lenLE payload crc32LE`. CRC is CRC-32/ISO-HDLC over
all preceding bytes. A CRC error or non-consecutive sequence is logged as a
damaged/missing region.
