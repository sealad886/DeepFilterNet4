# Datasets

This repository does **not** distribute audio. Use the manifests and scripts below to build
local training datasets for DeepFilterNet4 while keeping licensing permissive.

## Manifests

- Prototype: `datasets/prototype/manifest.yaml`
- Production: `datasets/production/manifest.yaml`

Both manifests explicitly **exclude CC-BY-NC and CC Sampling+** sources.

## Recommended local layout

```
<data_root>/
  lists/
    vctk_clean.txt
    librispeech_clean.txt
    musan_noise.txt
    musan_music.txt
    fsd50k_filtered.txt
    air_rir.txt
    openair_rir.txt
    acousticrooms_rir.txt
  hdf5/
    speech_clean.hdf5
    noise_music.hdf5
    rir.hdf5
  dataset.cfg
```

## Step 1: Download datasets (manual)

Use the dataset sources listed in `ATTRIBUTION.md` and download them to a local
location (one folder per dataset). This repo intentionally does not redistribute audio.

### Optional: automated downloads (opt-in)

`scripts/datasets/download_datasets.sh` now supports direct downloads for:
VCTK, LibriSpeech, MUSAN, FSD50K, and AcousticRooms, plus AIR/OpenAIR via `audb`.

```
PROFILE=prototype \
DOWNLOAD=1 \
AGREE_LICENSES=1 \
INSTALL_AUDB=1 \
DATA_DIR=/path/to/data \
  bash scripts/datasets/download_datasets.sh
```

Notes:
- Set `PROFILE=production` to include LibriSpeech + AcousticRooms by default.
- `AGREE_LICENSES=1` is required to proceed.
- If you do not want `audb` auto-install, set `INSTALL_AUDB=0` and install it yourself.
- FSD50K downloads are split zip files; this requires the `zip` and `unzip` tools.
- For faster downloads, install `aria2` and keep `USE_ARIA2=1` (default). You can tweak
  `ARIA2_CONN`, `ARIA2_SPLIT`, and `ARIA2_MIN_SPLIT` for bandwidth.
- `scripts/datasets/check_audb.py`, `scripts/datasets/audb_download.py`, and
  `scripts/datasets/fsd50k_filter.py` are helper modules invoked by the shell script; run them
  directly only if you are debugging.
- Some hosts (e.g., `datashare.ed.ac.uk`, `zenodo.org`) do not handle multi-range downloads well;
  the script automatically forces single-connection downloads for those URLs.
- The VCTK host does not reliably support resume; the script will re-download it from scratch
  if a partial file is detected.

## Step 2: Generate file lists

Use the helper script (skeleton) to create list files:

```
DATA_DIR=/path/to/data \
VCTK_DIR=/path/to/VCTK-Corpus-0.92 \
LIBRISPEECH_DIR=/path/to/LibriSpeech \
MUSAN_DIR=/path/to/musan \
FSD50K_DIR=/path/to/FSD50K \
AIR_RIR_DIR=/path/to/AIR \
OPENAIR_DIR=/path/to/OpenAIR \
ACOUSTICROOMS_DIR=/path/to/AcousticRooms \
  bash scripts/datasets/download_datasets.sh
```

Notes:
- For the **prototype**, you can omit LibriSpeech and AcousticRooms.
- FSD50K filtering is **required**: keep only CC0/CC-BY clips.
  The script expects a metadata CSV and filters by the license column.

## Step 3: Combine lists for each dataset type

Prototype example:

```
cat "$DATA_DIR/lists/vctk_clean.txt" > "$DATA_DIR/lists/clean_all.txt"
cat "$DATA_DIR/lists/musan_noise.txt" \
    "$DATA_DIR/lists/musan_music.txt" \
    "$DATA_DIR/lists/fsd50k_filtered.txt" > "$DATA_DIR/lists/noise_music.txt"
cat "$DATA_DIR/lists/air_rir.txt" \
    "$DATA_DIR/lists/openair_rir.txt" > "$DATA_DIR/lists/rir_all.txt"
```

Production example (adds LibriSpeech + AcousticRooms):

```
cat "$DATA_DIR/lists/vctk_clean.txt" \
    "$DATA_DIR/lists/librispeech_clean.txt" > "$DATA_DIR/lists/clean_all.txt"
cat "$DATA_DIR/lists/musan_noise.txt" \
    "$DATA_DIR/lists/musan_music.txt" \
    "$DATA_DIR/lists/fsd50k_filtered.txt" > "$DATA_DIR/lists/noise_music.txt"
cat "$DATA_DIR/lists/air_rir.txt" \
    "$DATA_DIR/lists/openair_rir.txt" \
    "$DATA_DIR/lists/acousticrooms_rir.txt" > "$DATA_DIR/lists/rir_all.txt"
```

## Step 4: Build HDF5 files (48 kHz)

From the `DeepFilterNet/` directory:

```
python -m df.scripts.prepare_data speech \
  /path/to/data/lists/clean_all.txt /path/to/data/hdf5/speech_clean.hdf5 \
  --sr 48000 --dtype int16

python -m df.scripts.prepare_data noise \
  /path/to/data/lists/noise_music.txt /path/to/data/hdf5/noise_music.hdf5 \
  --sr 48000 --dtype int16

python -m df.scripts.prepare_data rir \
  /path/to/data/lists/rir_all.txt /path/to/data/hdf5/rir.hdf5 \
  --sr 48000 --dtype int16
```

Or run the helper:

```
DATA_DIR=/path/to/data PROFILE=prototype bash scripts/datasets/build_hdf5.sh
```

Apple Silicon profile (lower default workers, memory-friendly defaults):

```
DATA_DIR=/path/to/data PROFILE=apple bash scripts/datasets/build_hdf5.sh
```

Notes:
- `dtype=int16` keeps storage smaller. Use `float32` if you need maximum fidelity.
- The `prepare_data` script tags HDF5 files by group name (`speech`, `noise`, `rir`).
  The dataset loader uses these groups to mix clean speech and noise.

## Step 5: Choose the dataset.cfg

Copy one of the templates to your data directory:

```
cp datasets/prototype/dataset.cfg /path/to/data/dataset.cfg
# or
cp datasets/production/dataset.cfg /path/to/data/dataset.cfg
```

Then train:

```
python -m df.train /path/to/data/dataset.cfg /path/to/data/hdf5 /path/to/output_dir
```

## Apple Silicon tips

- Start with `PROFILE=apple` and keep list sizes small for the prototype.
- Use `dtype=int16` during HDF5 prep to reduce disk and memory pressure.
- If prep is slow, lower workers: `NUM_WORKERS=1` or `2`.
- Keep the sample rate at 48 kHz to match DFNet4 defaults.

## Licensing notes

- Do **not** include CC-BY-NC or CC Sampling+ content in your training lists.
- Keep your dataset sources and licenses documented in `ATTRIBUTION.md`.
- Do not redistribute audio without explicit permission from the original dataset license.
