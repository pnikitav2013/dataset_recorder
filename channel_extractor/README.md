# channel_extractor

Split a single re-recorded channel out of a [`disk_recorder`](../disk_recorder)
dataset into its own dataset.

A `disk_recorder` run leaves every captured input device next to the original
clip as `<name>_R_<prefix>.wav`, so one folder mixes several sources:

```
288045_ref.5/
├── 116-288045-0024_R_fifine.wav
├── 116-288045-0024_R_X3800.wav
├── 116-288045-0024_R_mic1_stm32.wav
├── 116-288045-0025_R_fifine.wav
├── ...
└── 116-288045.trans.txt
```

This tool **copies** the dataset (the source is never modified), then keeps only
the requested channel:

1. deletes every audio file that is not `_R_<prefix>` (other channels, leftover
   originals);
2. strips the `_R_<prefix>` marker from the survivors so their names match the
   original clips again (`116-288045-0024_R_fifine.wav` → `116-288045-0024.wav`);
3. prunes any directory left empty.

Non-audio files (`*.trans.txt` transcripts, metadata) are copied verbatim. The
extracted dataset root is named `<source_name>_R_<prefix>`.

## Usage

```bash
python -m channel_extractor <source_dataset> <prefix> <dest_parent>
```

- `source_dataset` — the re-recorded folder to extract from.
- `prefix` — the channel to keep (the part after `_R_`, e.g. `fifine`,
  `mic1_stm32`).
- `dest_parent` — directory the new dataset is created inside; the resulting
  folder is `<source_name>_R_<prefix>`.

Example — extract the `fifine` channel:

```bash
cd py_recorder
python -m channel_extractor disk_recorder/tmp/288045_ref.5 fifine ./out
# -> ./out/288045_ref.5_R_fifine/116-288045-0024.wav, ...
python -m channel_extractor /media/nikita/Новый том/save_K/self_pj_26/data_set_buff/500_wav_r stm32n6 /media/nikita/Новый том/save_K/self_pj_26/data_set_buff


```

Flags: `--overwrite` replaces an existing extracted dataset; `-v/--verbose`
logs each kept/removed file.

## Environment

Pure standard library — no third-party dependencies. Use the shared
`py_recorder` virtualenv created by [`../create_venv.sh`](../create_venv.sh):

```bash
cd py_recorder
./create_venv.sh                 # if .venv does not exist yet
.venv/bin/python -m channel_extractor <source> <prefix> <dest>
```

(Any Python 3.10+ interpreter works too.)
