# disk_recorder

Re-record a folder of audio files through **up to three input devices at once**
(STM32N6 microphones and/or PC microphones).

For every audio file under a chosen folder (recursively) the app:

1. plays the file **once** on a selected **output device** (a speaker next to the
   boards/mics),
2. simultaneously **captures every enabled input device** — an STM32N6 mic stream
   (mono PCM16 @ 16 kHz over the `reliable_transport` UART protocol, the same one
   used by `microphone_recorder.py`) or a PC input device,
3. **aligns** each capture to the played reference by cross-correlation (absorbs
   speaker/acoustic/mic/UART latency → sub-100 ms start alignment),
4. saves each aligned clip next to the original as `<name>_R_<prefix>.wav`, where
   `<prefix>` is the slot's manually entered device name.

A file is only committed when **every** enabled device produces a clean capture:
if any channel reports a transport error / loses lock / underruns, the whole
attempt is discarded and the file is **replayed and re-recorded for all devices**
(up to `max_retries`). Once all outputs are written, the original is **deleted**.
Outputs are recognised by the `_R_` marker in their name and are never re-scanned
as sources.

### Input-device panel (three slots)

The window has **three fixed input-device slots**. Each slot picks one of three
variants and shows only the fields that apply:

| Variant         | Fields |
|-----------------|--------|
| `Off`           | (slot unused) |
| `STM32 (UART)`  | prefix, COM port, baud |
| `PC microphone` | prefix, input device |

The **prefix** is typed by hand and identifies the device in the output filename
(`…_R_<prefix>.wav`). The whole setup (folder, output device, the three slots) is
saved to `disk_recorder_config.json` in the **launch directory** and reloaded on
the next start, so nothing has to be re-entered.

A Tkinter window shows live progress, a **per-device sync indicator**, a stacked
**log-mel spectrogram per device** (one above another), the number of registered
errors and the average record time per file.

## Run

```bash
cd py_recorder
./create_venv.sh                 # or: pip install -r requirements.txt
python -m disk_recorder          # opens the window (run from py_recorder/)
```

On Windows 10: from an activated venv, `python -m disk_recorder` in `py_recorder\`.
`tkinter` ships with the python.org installer; `sounddevice` bundles PortAudio.

## Modules

| Module        | Responsibility |
|---------------|----------------|
| `config`      | tunable `Settings` (baud, sample rate, headroom, retries, margins, `_R_` marker) |
| `appconfig`   | JSON-persisted GUI config (folder, output device, three input-device slots) |
| `devices`     | enumerate serial ports + audio output/input devices |
| `serial_link` | open the port + background reader thread |
| `sources`     | `BoardSource` (STM32 UART) / `MicSource` (PC mic) capture + error counting |
| `playback`    | load file, non-blocking play, 16 kHz mono reference |
| `sync`        | cross-correlation alignment + trim |
| `mel`         | log-mel spectrogram `Figure`s, incl. stacked per-device (object API, thread-safe) |
| `storage`     | recursive scan (excludes `*_R_*`), WAV write, original deletion |
| `pipeline`    | multi-device orchestration: play once → capture all → align → save all, all-or-retry |
| `state`       | thread-safe session state (per-channel status) shared with the GUI |
| `gui`         | the Tkinter window (three-slot input panel, persistence) |

The framing protocol is **not** duplicated — it is imported from
`py_recorder/reliable_transport.py`.
