# Datasets

This repository does **not** distribute audio. Use the manifests and scripts below to build
local training datasets for DeepFilterNet4 while keeping licensing permissive.

## Manifests

- Prototype: `datasets/prototype/manifest.yaml`
- Production: `datasets/production/manifest.yaml`

Both manifests explicitly **exclude CC-BY-NC and CC Sampling+** sources. The
default automated background-music path therefore uses FMA as its canonical
open-music source. MTG-Jamendo is supported only as an **optional local
supplement** because its official dataset terms are more restrictive than the
default permissive path; enable it only when that matches your local research
use.

## Recommended local layout

```
<data_root>/
  lists/
    vctk_clean.txt
    librispeech_clean.txt
    musan_noise.txt
    fsd50k_filtered.txt
    background_music.txt
    background_music_expanded.txt
    background_music_fma.txt
    background_music_mtg_jamendo.txt
    background_music_catalog.tsv
    noise_all.txt
    noise_music.txt
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
VCTK, LibriSpeech, MUSAN noise, FMA, FSD50K, and AcousticRooms, plus
AIR/OpenAIR via `audb`. MTG-Jamendo is supported as an explicit opt-in
download/source when you want additional chart-adjacent music coverage and the
dataset terms are acceptable for your local workflow.

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
- FMA is enabled by default because it is the canonical automated
  background-music base. MTG-Jamendo stays opt-in (`DOWNLOAD_MTG_JAMENDO=1`)
  because its official release is aimed at non-commercial research/academic
  use.
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
- If you already have archives in `downloads/`, the script will extract them to `raw/` automatically
  (e.g., VCTK zip or LibriSpeech tarballs).
- Verification results are cached in `downloads/.verify_cache.tsv` so re-runs don't re-scan large
  archives. Set `VERIFY_CACHE=0` to disable.

## Step 2: Generate file lists

Use the helper script (skeleton) to create list files:

```
DATA_DIR=/path/to/data \
VCTK_DIR=/path/to/VCTK-Corpus-0.92 \
LIBRISPEECH_DIR=/path/to/LibriSpeech \
MUSAN_DIR=/path/to/musan \
FMA_DIR=/path/to/FMA \
MTG_JAMENDO_DIR=/path/to/mtg-jamendo \
FSD50K_DIR=/path/to/FSD50K \
AIR_RIR_DIR=/path/to/AIR \
OPENAIR_DIR=/path/to/OpenAIR \
ACOUSTICROOMS_DIR=/path/to/AcousticRooms \
  bash scripts/datasets/download_datasets.sh
```

Notes:
- For the **prototype**, you can omit LibriSpeech and AcousticRooms.
- FSD50K filtering is still **required** for the generic-noise pool: keep only
  CC0/CC-BY clips.
- `download_datasets.sh` now curates dedicated background music from FMA plus
  optional MTG-Jamendo metadata/audio. Use `BACKGROUND_MUSIC_TARGET_COUNT`
  (default `2000`) and `BACKGROUND_MUSIC_MIN_COUNT` (default `500`) to control
  the curated chart-style pool size.

## Step 3: Combine lists for each dataset type

For current `df_mlx` training, keep **generic noise** and **dedicated background
music** as separate lists. `download_datasets.sh` now writes the combined files
for you; there is no longer a MUSAN-music concatenation step in the canonical
background-music path.

- `noise_all.txt` = environmental/other non-music noise
- `background_music.txt` = canonical curated chart-style music list used when
  `--music-list` / `p_background_music` is enabled
- `background_music_expanded.txt` = full eligible chart-style pool from FMA +
  optional MTG-Jamendo
- `noise_music.txt` = backward-compatible combined list for legacy HDF5 / older workflows

The generated music outputs now follow this contract:

- `background_music.txt` aims at roughly 500–2000 songs matching the genre mix
  of mainstream compilation CDs such as “Now That’s What I Call Music”:
  pop, pop-rock / alternative rock, dance / EDM, country-pop / americana,
  and adjacent R&B / hip-hop crossover material.
- `background_music_expanded.txt` is the uncapped eligible pool behind the
  curated set.
- `background_music_fma.txt` and `background_music_mtg_jamendo.txt` expose the
  source-specific subsets.
- `background_music_catalog.tsv` records source, bucket, score, and matched
  tokens for audit/debugging.

When you run `download_datasets.sh`, the combined artifacts are produced as:

```
bash scripts/datasets/download_datasets.sh --profile production
```

## Step 4: Build the MLX datastore (recommended for `df_mlx` training)

The MLX datastore caches **source** speech/noise/RIR audio into sharded NPZ
files for faster loading, but training still does dynamic mixing (new SNR/noise/RIR
combinations each epoch). That means you **do not** need to rebuild the datastore
just to try a different loss profile or to get fresh random mixtures.

You **do** need to rebuild the datastore when any of these change:

- the clean/noise/RIR file lists,
- the clean-speech preprocessing choice,
- `--sample-rate` or `--segment-length`,
- short-speech handling (`--min-duration` / `--merge-short`).

Recommended build for Apple Silicon / vadlite-style training:

```
DATA_DIR=/path/to/data \
OUTPUT_DIR=/path/to/cache \
  bash scripts/datasets/build_mlx_datastore.sh \
    --profile apple \
    --merge-short
```

If you want to inline the current DeepFilterNet3 clean-speech preprocessing step
while rebuilding the datastore:

```
DATA_DIR=/path/to/data \
OUTPUT_DIR=/path/to/cache \
  bash scripts/datasets/build_mlx_datastore.sh \
    --profile apple \
    --merge-short \
    --preprocess-clean-speech
```

To append the CHAINS speaking-style corpus to the clean-speech list during the
same build, enable it explicitly:

```
DATA_DIR=/path/to/data \
OUTPUT_DIR=/path/to/cache \
  bash scripts/datasets/build_mlx_datastore.sh \
    --profile apple \
    --merge-short \
    --include-chains \
    --chains-dir /Volumes/TrainingData/CHAINS \
    --preprocess-clean-speech
```

Notes:
- `--merge-short` is recommended when you want to preserve more short utterances
  instead of skipping speech clips shorter than `--min-duration`.
- `build_mlx_datastore.sh` prefers `lists/background_music.txt` as the
  dedicated `--music-list`, falling back to
  `lists/background_music_expanded.txt` only when the curated list is absent.
- If you pass `--prepare-background-music`, the builder synthesizes additional
  dirty speaker/room/live-ish variants from that music list before sharding and
  merges them into `background_music.prepared_merged.txt`.
- `--music-prepare-style speaker_room` is the default preset and is the
  recommended starting point for consumer-speaker-in-room exposure.
- If you already ran standalone clean-speech preprocessing, you can point
  `--clean-list` at the resulting list instead of enabling
  `--preprocess-clean-speech` again.
- `--include-chains` keeps the released mono CHAINS styles (`solo`, `sync`,
  `retell`, `whsp`, `fast`) as-is and extracts only the RSI speaker channel;
  the RSI target channel is excluded so the datastore does not duplicate the
  repeated prompt speaker recordings.
- If you are rebuilding into a new location to avoid duplicate disk usage,
  delete the old cache directory first.

Validate the datastore before a long run:

```
python -m df_mlx.validate_audio_cache /path/to/cache
```

Launch the full vadlite-style run with an explicit cache-dir override so the
training command uses the datastore you just built:

```
python -m df_mlx.train_dynamic \
  --run-config df_mlx/configs/run_profiles/baseline_dfn3_gan_vad_speech_full_vadlite.toml \
  --cache-dir /path/to/cache
```

To emphasize loud background-music suppression while keeping speech intact, use
the dedicated music path and music-specific gain controls:

```bash
python -m df_mlx.train_dynamic \
  --cache-dir /path/to/cache \
  --p-background-music 0.5 \
  --background-music-gain-range 0 12
```

## Step 5: Build HDF5 files (48 kHz)

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

## Step 6: Choose the dataset.cfg

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
