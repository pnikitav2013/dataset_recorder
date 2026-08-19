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

### Noise mode (no playback)

The **Mode** selector at the top of the window switches the whole run between
two jobs:

| Mode | What it does |
|------|--------------|
| `Re-record folder (play + capture)` | the default described above |
| `Noise only (no playback, chunked files)` | records the enabled inputs continuously — nothing is played, no folder is scanned, nothing is aligned or deleted |

In noise mode you pick a **separate destination folder** and the **length of one
file** (5 minutes by default). Every enabled slot records the room at the same
time and the stream is cut into fixed-length chunks:

```
<noise folder>/<prefix>/<prefix>_YYYYmmdd_HHMMSS.wav
```

Chunks are self-contained WAVs written back-to-back (a sub-second seam between
files), so a crash or a power cut costs at most one chunk instead of a
multi-day recording. *Stop* keeps the partial chunk recorded so far. The
working-hours schedule applies here too, so an overnight pause does not fill the
disk. The source folder, output device and routing are greyed out in this mode —
nothing is played back.

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

## Multi-day runs

The app is built to be left running for days, which puts the *host audio stack*
— not the pipeline — on the critical path. Three rules follow from that.

**The output device is opened once.** `Player` holds a single `OutputStream` for
the whole session and resamples every file to its fixed rate. Opening and
closing the device per file (what `sd.play()` does) means tens of thousands of
open/close/format-renegotiation cycles through the Windows audio engine, which
leaks handles until system audio breaks and only a reboot fixes it.

**Nothing waits forever.** Device reads have no timeout of their own, so every
capture has a wall-clock budget; overrunning it makes the pipeline ask the
capture to stop, escalate to `Pa_AbortStream` if it does not, re-open the
device, and — if the device is wedged inside its driver — stop the run with the
reason in the status line instead of hanging with the GUI still showing
"recording".

**Everything is logged.** Progress plus process RSS / handle count / thread
count are written every 60 s to `disk_recorder.log` (rotating, in the launch
directory). A climbing handle count shows up there hours before it becomes an
outage.

### Windows 10 host settings

Worth doing once on the machine that runs the sessions:

* pick the **WASAPI** entry for the speaker and the microphones (the app
  prefers it automatically and warns if a WDM-KS device is selected);
* Sound → device properties → *Advanced*: uncheck **"Allow applications to take
  exclusive control"**, and set the same default format (e.g. 48000 Hz, 16 bit)
  for the input and the output;
* device properties → *Enhancements*: **disable all sound effects**;
* Power options: disable **USB selective suspend**, and untick *"Allow the
  computer to turn off this device to save power"* on the USB hubs and audio
  devices. The app already inhibits sleep and display timeout while running.

## Modules

| Module        | Responsibility |
|---------------|----------------|
| `config`      | tunable `Settings` (baud, sample rate, headroom, retries, margins, `_R_` marker) |
| `appconfig`   | JSON-persisted GUI config (mode, folders, chunk length, output device, input-device slots) |
| `devices`     | enumerate serial ports + audio devices, rank host APIs (WASAPI ≫ MME ≫ WDM-KS) |
| `serial_link` | open the port + background reader thread |
| `sources`     | `BoardSource` (STM32 UART) / `MicSource` (PC mic) capture, capture deadlines + cooperative abort |
| `playback`    | load file, **one persistent output stream**, 16 kHz mono reference |
| `diag`        | process RSS/handle sampling for the heartbeat, Windows sleep inhibition |
| `sync`        | cross-correlation alignment + trim |
| `mel`         | log-mel spectrogram `Figure`s, incl. stacked per-device (object API, thread-safe) |
| `storage`     | recursive scan (excludes `*_R_*`), WAV write, original deletion |
| `noise`       | continuous no-playback capture, cut into fixed-length chunks per device |
| `pipeline`    | multi-device orchestration: play once → capture all → align → save all, all-or-retry |
| `state`       | thread-safe session state (per-channel status) shared with the GUI |
| `gui`         | the Tkinter window (three-slot input panel, persistence) |

The framing protocol is **not** duplicated — it is imported from
`py_recorder/reliable_transport.py`.
