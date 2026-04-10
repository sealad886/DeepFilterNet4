#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DEFAULT_DATA_DIR="/Volumes/TrainingData/datasets"
DEFAULT_CHAINS_DIR="/Volumes/TrainingData/CHAINS"
if [[ ! -d "${DEFAULT_DATA_DIR}" ]]; then
  DEFAULT_DATA_DIR="${ROOT_DIR}/data"
fi

detect_apple_silicon_tier() {
  if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "non-apple"
    return
  fi

  local brand_string
  brand_string="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  case "${brand_string}" in
    *Ultra*)
      echo "ultra"
      ;;
    *Max*)
      echo "max"
      ;;
    *Pro*)
      echo "pro"
      ;;
    Apple*)
      echo "entry"
      ;;
    *)
      echo "entry"
      ;;
  esac
}

APPLE_SILICON_TIER="$(detect_apple_silicon_tier)"

detect_active_virtualenv_python() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    if [[ -x "${VIRTUAL_ENV}/bin/python3" ]]; then
      echo "${VIRTUAL_ENV}/bin/python3"
      return 0
    fi
    if [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
      echo "${VIRTUAL_ENV}/bin/python"
      return 0
    fi
  fi
  return 1
}

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if ACTIVE_VENV_PYTHON="$(detect_active_virtualenv_python)"; then
    PYTHON_BIN="${ACTIVE_VENV_PYTHON}"
  elif [[ -x "${ROOT_DIR}/.venv/bin/python3" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python3"
  elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Error: could not find python3 or python on PATH" >&2
    exit 1
  fi
fi

DEFAULT_MLX_PREPROCESS_MODEL="${ROOT_DIR}/models/mlx/DeepFilterNet3-MLX"
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" && -f "${DEFAULT_MLX_PREPROCESS_MODEL}/config.ini" ]]; then
  DEFAULT_PREPROCESS_MODEL="${DEFAULT_MLX_PREPROCESS_MODEL}"
else
  DEFAULT_PREPROCESS_MODEL="DeepFilterNet3"
fi

write_atomic_stream() {
  local out_file="$1"
  local tmp_file="${out_file}.tmp.$$"
  cat > "${tmp_file}"
  mv "${tmp_file}" "${out_file}"
}

merge_unique_file_lists() {
  local out_file="$1"
  shift
  {
    for list_file in "$@"; do
      if [[ -f "${list_file}" ]]; then
        cat "${list_file}"
      fi
    done
  } | awk 'NF && $0 !~ /^#/ && !seen[$0]++' | write_atomic_stream "${out_file}"
  echo "[ok] wrote $(wc -l < "${out_file}") entries -> ${out_file}"
}

compute_common_base_dir() {
  local list_file="$1"
  "${PYTHON_BIN}" - "${list_file}" <<'PY'
from pathlib import Path
import os
import sys

list_path = Path(sys.argv[1]).expanduser().resolve()
parent_dirs: list[str] = []
with list_path.open() as handle:
    for raw_line in handle:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parent_dirs.append(str(Path(line).expanduser().resolve().parent))

if not parent_dirs:
    raise SystemExit(f"No speech entries found in {list_path}")

print(os.path.commonpath(parent_dirs))
PY
}

temp_cleanup_paths=()
cleanup_temp_files() {
  local path
  for path in "${temp_cleanup_paths[@]:-}"; do
    if [[ -n "${path}" && -f "${path}" ]]; then
      rm -f "${path}"
    fi
  done
}
trap cleanup_temp_files EXIT

file_list_entry_stats() {
  local list_file="$1"
  "${PYTHON_BIN}" - "${list_file}" <<'PY'
from pathlib import Path
import os
import sys

path = Path(sys.argv[1])
total = 0
existing = 0
if path.is_file():
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            entry = raw_line.strip()
            if not entry or entry.startswith("#"):
                continue
            total += 1
            if os.path.exists(entry):
                existing += 1
print(f"{total} {existing}")
PY
}

sanitize_existing_file_list() {
  local input_list="$1"
  local output_list="$2"
  "${PYTHON_BIN}" - "${input_list}" "${output_list}" <<'PY'
from pathlib import Path
import os
import sys

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
output_path.parent.mkdir(parents=True, exist_ok=True)
with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
    for raw_line in src:
        entry = raw_line.strip()
        if not entry or entry.startswith("#"):
            continue
        if os.path.exists(entry):
            dst.write(f"{entry}\n")
PY
}

phase_elapsed() {
  local s=$SECONDS
  printf '%dm%02ds' $((s / 60)) $((s % 60))
}

# ---------------------------------------------------------------------------
# Phase-skip verification helpers
# ---------------------------------------------------------------------------
# These check actual output files on disk (isfile + size > 0) to decide
# whether a phase can be skipped.  Lists and indices may be stale after a
# crash — only the files themselves are trustworthy.

verify_file_list() {
  local list_file="$1"
  "${PYTHON_BIN}" - "${list_file}" <<'PY'
import os, sys, time
t0 = time.monotonic()
list_path = sys.argv[1]
if not os.path.isfile(list_path):
    sys.exit(2)
total = 0
valid = 0
with open(list_path, encoding="utf-8") as fh:
    for raw in fh:
        p = raw.strip()
        if not p or p.startswith("#"):
            continue
        total += 1
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            valid += 1
if total == 0:
    sys.exit(2)
elapsed = time.monotonic() - t0
print(f"{valid} {total} {elapsed:.1f}")
sys.exit(0 if valid == total else 1)
PY
}

check_preprocess_complete() {
  local input_list="$1"
  local output_root="$2"
  local base_dir="$3"
  "${PYTHON_BIN}" - "${input_list}" "${output_root}" "${base_dir}" <<'PY'
import os, sys, time
from pathlib import Path

t0 = time.monotonic()
input_list = sys.argv[1]
output_root = Path(sys.argv[2])
base_dir = Path(sys.argv[3])

if not os.path.isfile(input_list):
    sys.exit(2)

total = 0
covered = 0
with open(input_list, encoding="utf-8") as fh:
    for raw in fh:
        p = raw.strip()
        if not p or p.startswith("#"):
            continue
        total += 1
        inp = Path(p).expanduser().resolve()
        try:
            rel = inp.relative_to(base_dir.expanduser().resolve())
        except ValueError:
            rel = Path("_external") / inp.name
        expected = output_root / rel.with_suffix(".wav")
        if expected.is_file() and expected.stat().st_size > 0:
            covered += 1

if total == 0:
    sys.exit(2)
elapsed = time.monotonic() - t0
print(f"{covered} {total} {elapsed:.1f}")
sys.exit(0 if covered == total else 1)
PY
}

check_music_prep_complete() {
  local input_list="$1"
  local output_root="$2"
  local base_dir="$3"
  local style="$4"
  "${PYTHON_BIN}" - "${input_list}" "${output_root}" "${base_dir}" "${style}" <<'PY'
import os, sys, time
from pathlib import Path

t0 = time.monotonic()
input_list = sys.argv[1]
output_root = Path(sys.argv[2])
base_dir = Path(sys.argv[3])
style = sys.argv[4]

if not os.path.isfile(input_list):
    sys.exit(2)

total = 0
covered = 0
with open(input_list, encoding="utf-8") as fh:
    for raw in fh:
        p = raw.strip()
        if not p or p.startswith("#"):
            continue
        total += 1
        inp = Path(p).expanduser().resolve()
        try:
            rel = inp.relative_to(base_dir.expanduser().resolve())
        except ValueError:
            rel = Path("_external") / inp.name
        # Check variant_0 exists — if it does, all variants were rendered
        # (the script processes all variants per source before moving on)
        # Path mirrors build_output_path(): rel_parent/{stem}__{ext}/{stem}.{style}_v00.wav
        suffix_label = inp.suffix.lower().lstrip(".") or "audio"
        variant_dir = output_root / rel.parent / f"{inp.stem}__{suffix_label}"
        expected = variant_dir / f"{inp.stem}.{style}_v00.wav"
        if expected.is_file() and expected.stat().st_size > 0:
            covered += 1

if total == 0:
    sys.exit(2)
elapsed = time.monotonic() - t0
print(f"{covered} {total} {elapsed:.1f}")
sys.exit(0 if covered == total else 1)
PY
}

usage_helptext() {
  cat <<EOF
Usage:
  ./build_mlx_datastore.sh [options]

Build the MLX sharded audio cache used by df_mlx dynamic training.

Core options:
  --data-dir PATH             Base dataset directory
  --output-dir PATH           Output cache directory
  --list-dir PATH             Directory containing clean/noise/RIR file lists
  --profile NAME              prototype | production | apple
                              (apple auto-tunes worker defaults by chip class)
  --clean-list PATH           Clean speech file list
  --noise-list PATH           Noise/music file list
  --music-list PATH           Optional dedicated background music file list
                              (default: LIST_DIR/background_music.txt, the
                              curated chart-style FMA/optional-MTG set
                              generated by download_datasets.sh; falls back to
                              LIST_DIR/background_music_expanded.txt)
  --rir-list PATH             Optional RIR file list
  --include-chains            Append CHAINS speaking-style recordings to the clean list
                              (mono styles + extracted RSI speaker channel)
  --chains-dir PATH           CHAINS corpus root (default: ${DEFAULT_CHAINS_DIR})

Audio/cache options:
  --sample-rate HZ            Target sample rate (default: 48000)
  --segment-length SEC        Target segment length in seconds (default: 5.0)
  --snr-min DB                Minimum SNR (default: -5)
  --snr-max DB                Maximum SNR (default: 40)
  --rir-prob P                Probability of RIR augmentation (default: 0.5)
  --num-workers N             Parallel workers for cache building
  --shard-size N              Files per shard
  --min-duration SEC          Minimum clean-speech duration before skip/merge
  --merge-short               Merge short speech files instead of skipping them
  --no-merge-short            Force skipping short speech files
  --max-pending-gb N          Max in-flight async shard writer budget in GB

Optional clean-speech preprocessing:
  --preprocess-clean-speech   Enhance clean speech with DeepFilterNet3 before caching
                              (speech list only; obvious noise/RIR inputs are rejected)
  --preprocess-output-root P  Directory for preprocessed speech mirror tree
  --preprocess-base-dir P     Base dir used to preserve relative paths
  --preprocess-output-list P  File list written for preprocessed outputs
  --preprocess-model NAME     Model name or model dir (default: repo-local
                              models/mlx/DeepFilterNet3-MLX on Apple Silicon
                              when available, otherwise DeepFilterNet3)
  --preprocess-device DEV     cpu | cuda | mps | auto
  --preprocess-workers N      Input-loading workers for preprocessing
                              (default: chip-aware under apple profile)
  --preprocess-probe-workers N
                              Parallel ffprobe workers used to estimate pending
                              clean-speech duration before enhancement
  --preprocess-probe-cache P  Optional JSON cache for ffprobe duration results
                              (default: auto path derived from the preprocess output list)
  --preprocess-enhance-batch-size N
                              Batch size for MLX enhancement (default: auto;
                              currently 4 for MLX, 1 for torch)
  --preprocess-overwrite      Rebuild preprocessed files even if they already exist; otherwise resume is automatic

Optional background-music preparation:
  --prepare-background-music  Render synthetic room/speaker/live-ish variants
                              from the music list before cache building
  --music-prepare-output-root P
                              Directory for prepared music mirror tree
  --music-prepare-base-dir P  Base dir used to preserve relative paths
  --music-prepare-output-list P
                              File list written for prepared music outputs
  --music-prepare-rir-list P  Optional RIR list used while dirtying music
                              (default: reuse --rir-list when present)
  --music-prepare-style STYLE
                              Background-music playback preset passed to
                              prepare_background_music.py
                              (default: speaker_room)
  --music-prepare-variants N  Variants to render per source music file
                              (default: 2)
  --music-prepare-seed N      Base seed for deterministic rendering
                              (default: 1337)
  --music-prepare-overwrite   Rebuild prepared music files even if they already exist; otherwise resume is automatic

General:
  --force                     Run all phases regardless of existing outputs
                              (skips phase-level completion checks; per-file
                              resume still applies unless --*-overwrite is set)
  -h, --help                  Show this help message and exit

Environment variables remain supported and are used as fallbacks when the
equivalent CLI option is not provided.

Detected default DATA_DIR on this machine: ${DEFAULT_DATA_DIR}

Examples:
  # Build a cache from existing lists and keep short speech coverage by merging
  # sub-segment utterances instead of skipping them.
  ./build_mlx_datastore.sh --profile apple --merge-short

  # Rebuild the cache and inline the DFN3 clean-speech preprocessing step.
  ./build_mlx_datastore.sh \
    --profile apple \
    --merge-short \
    --preprocess-clean-speech

  # Expand the music set with synthetic speaker-in-room/live-ish variants.
  ./build_mlx_datastore.sh \
    --profile apple \
    --merge-short \
    --prepare-background-music

  # Swap the preparation preset when you want a different playback texture.
  ./build_mlx_datastore.sh \
    --profile apple \
    --merge-short \
    --prepare-background-music \
    --music-prepare-style club_live

  # Append CHAINS speaking-style speech and preprocess it in the same pass.
  ./build_mlx_datastore.sh \
    --profile apple \
    --merge-short \
    --include-chains \
    --chains-dir /Volumes/TrainingData/CHAINS \
    --preprocess-clean-speech
EOF
}

CLI_DATA_DIR=""
CLI_OUTPUT_DIR=""
CLI_LIST_DIR=""
CLI_PROFILE=""
CLI_CLEAN_LIST=""
CLI_NOISE_LIST=""
CLI_MUSIC_LIST=""
CLI_RIR_LIST=""
CLI_CHAINS_DIR=""
CLI_SR=""
CLI_SEGMENT_LENGTH=""
CLI_SNR_MIN=""
CLI_SNR_MAX=""
CLI_RIR_PROB=""
CLI_NUM_WORKERS=""
CLI_SHARD_SIZE=""
CLI_MIN_DURATION=""
CLI_MAX_PENDING_BYTES=""
CLI_PREPROCESS_OUTPUT_ROOT=""
CLI_PREPROCESS_BASE_DIR=""
CLI_PREPROCESS_OUTPUT_LIST=""
CLI_PREPROCESS_MODEL=""
CLI_PREPROCESS_DEVICE=""
CLI_PREPROCESS_WORKERS=""
CLI_PREPROCESS_PROBE_WORKERS=""
CLI_PREPROCESS_PROBE_CACHE=""
CLI_PREPROCESS_ENHANCE_BATCH_SIZE=""
CLI_MUSIC_PREPARE_OUTPUT_ROOT=""
CLI_MUSIC_PREPARE_BASE_DIR=""
CLI_MUSIC_PREPARE_OUTPUT_LIST=""
CLI_MUSIC_PREPARE_RIR_LIST=""
CLI_MUSIC_PREPARE_STYLE=""
CLI_MUSIC_PREPARE_VARIANTS=""
CLI_MUSIC_PREPARE_SEED=""
CLI_MERGE_SHORT=""
PREPROCESS_CLEAN_SPEECH=0
PREPROCESS_OVERWRITE=0
PREPARE_BACKGROUND_MUSIC=0
MUSIC_PREPARE_OVERWRITE=0
INCLUDE_CHAINS=0
FORCE_ALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      CLI_DATA_DIR="$2"
      shift 2
      ;;
    --output-dir)
      CLI_OUTPUT_DIR="$2"
      shift 2
      ;;
    --list-dir)
      CLI_LIST_DIR="$2"
      shift 2
      ;;
    --profile)
      CLI_PROFILE="$2"
      shift 2
      ;;
    --clean-list)
      CLI_CLEAN_LIST="$2"
      shift 2
      ;;
    --noise-list)
      CLI_NOISE_LIST="$2"
      shift 2
      ;;
    --music-list)
      CLI_MUSIC_LIST="$2"
      shift 2
      ;;
    --rir-list)
      CLI_RIR_LIST="$2"
      shift 2
      ;;
    --include-chains)
      INCLUDE_CHAINS=1
      shift
      ;;
    --chains-dir)
      CLI_CHAINS_DIR="$2"
      shift 2
      ;;
    --sample-rate)
      CLI_SR="$2"
      shift 2
      ;;
    --segment-length)
      CLI_SEGMENT_LENGTH="$2"
      shift 2
      ;;
    --snr-min)
      CLI_SNR_MIN="$2"
      shift 2
      ;;
    --snr-max)
      CLI_SNR_MAX="$2"
      shift 2
      ;;
    --rir-prob)
      CLI_RIR_PROB="$2"
      shift 2
      ;;
    --num-workers)
      CLI_NUM_WORKERS="$2"
      shift 2
      ;;
    --shard-size)
      CLI_SHARD_SIZE="$2"
      shift 2
      ;;
    --min-duration)
      CLI_MIN_DURATION="$2"
      shift 2
      ;;
    --merge-short)
      CLI_MERGE_SHORT="true"
      shift
      ;;
    --no-merge-short)
      CLI_MERGE_SHORT="false"
      shift
      ;;
    --max-pending-gb)
      CLI_MAX_PENDING_BYTES="$2"
      shift 2
      ;;
    --preprocess-clean-speech)
      PREPROCESS_CLEAN_SPEECH=1
      shift
      ;;
    --preprocess-output-root)
      CLI_PREPROCESS_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --preprocess-base-dir)
      CLI_PREPROCESS_BASE_DIR="$2"
      shift 2
      ;;
    --preprocess-output-list)
      CLI_PREPROCESS_OUTPUT_LIST="$2"
      shift 2
      ;;
    --preprocess-model)
      CLI_PREPROCESS_MODEL="$2"
      shift 2
      ;;
    --preprocess-device)
      CLI_PREPROCESS_DEVICE="$2"
      shift 2
      ;;
    --preprocess-workers)
      CLI_PREPROCESS_WORKERS="$2"
      shift 2
      ;;
    --preprocess-probe-workers)
      CLI_PREPROCESS_PROBE_WORKERS="$2"
      shift 2
      ;;
    --preprocess-probe-cache)
      CLI_PREPROCESS_PROBE_CACHE="$2"
      shift 2
      ;;
    --preprocess-enhance-batch-size)
      CLI_PREPROCESS_ENHANCE_BATCH_SIZE="$2"
      shift 2
      ;;
    --preprocess-overwrite)
      PREPROCESS_OVERWRITE=1
      shift
      ;;
    --prepare-background-music)
      PREPARE_BACKGROUND_MUSIC=1
      shift
      ;;
    --music-prepare-output-root)
      CLI_MUSIC_PREPARE_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --music-prepare-base-dir)
      CLI_MUSIC_PREPARE_BASE_DIR="$2"
      shift 2
      ;;
    --music-prepare-output-list)
      CLI_MUSIC_PREPARE_OUTPUT_LIST="$2"
      shift 2
      ;;
    --music-prepare-rir-list)
      CLI_MUSIC_PREPARE_RIR_LIST="$2"
      shift 2
      ;;
    --music-prepare-style)
      CLI_MUSIC_PREPARE_STYLE="$2"
      shift 2
      ;;
    --music-prepare-variants)
      CLI_MUSIC_PREPARE_VARIANTS="$2"
      shift 2
      ;;
    --music-prepare-seed)
      CLI_MUSIC_PREPARE_SEED="$2"
      shift 2
      ;;
    --music-prepare-overwrite)
      MUSIC_PREPARE_OVERWRITE=1
      shift
      ;;
    --force)
      FORCE_ALL=1
      shift
      ;;
    -h|--help)
      usage_helptext
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage_helptext >&2
      exit 1
      ;;
  esac
done

PREPROCESS_BASE_DIR_WAS_SET=0
if [[ -n "${CLI_PREPROCESS_BASE_DIR}" || -n "${PREPROCESS_BASE_DIR:-}" ]]; then
  PREPROCESS_BASE_DIR_WAS_SET=1
fi

DATA_DIR="${CLI_DATA_DIR:-${DATA_DIR:-${DEFAULT_DATA_DIR}}}"
OUTPUT_DIR="${CLI_OUTPUT_DIR:-${OUTPUT_DIR:-${DATA_DIR}/mlx_audio_cache}}"
LIST_DIR="${CLI_LIST_DIR:-${LIST_DIR:-${DATA_DIR}/lists}}"
PROFILE="${CLI_PROFILE:-${PROFILE:-apple}}"

SR="${CLI_SR:-${SR:-48000}}"
SEGMENT_LENGTH="${CLI_SEGMENT_LENGTH:-${SEGMENT_LENGTH:-5.0}}"
SNR_MIN="${CLI_SNR_MIN:-${SNR_MIN:--5}}"
SNR_MAX="${CLI_SNR_MAX:-${SNR_MAX:-40}}"
RIR_PROB="${CLI_RIR_PROB:-${RIR_PROB:-0.5}}"

CLEAN_LIST="${CLI_CLEAN_LIST:-${CLEAN_LIST:-${LIST_DIR}/clean_all.txt}}"
DEFAULT_NOISE_LIST="${LIST_DIR}/noise_music.txt"
if [[ -f "${LIST_DIR}/noise_all.txt" ]]; then
  DEFAULT_NOISE_LIST="${LIST_DIR}/noise_all.txt"
fi
NOISE_LIST="${CLI_NOISE_LIST:-${NOISE_LIST:-${DEFAULT_NOISE_LIST}}}"
DEFAULT_MUSIC_LIST="${LIST_DIR}/background_music.txt"
if [[ ! -s "${DEFAULT_MUSIC_LIST}" && -s "${LIST_DIR}/background_music_expanded.txt" ]]; then
  DEFAULT_MUSIC_LIST="${LIST_DIR}/background_music_expanded.txt"
fi
MUSIC_LIST="${CLI_MUSIC_LIST:-${MUSIC_LIST:-${DEFAULT_MUSIC_LIST}}}"
RIR_LIST="${CLI_RIR_LIST:-${RIR_LIST:-${LIST_DIR}/rir_all.txt}}"
CHAINS_DIR="${CLI_CHAINS_DIR:-${CHAINS_DIR:-${DEFAULT_CHAINS_DIR}}}"
CHAINS_PREPARED_ROOT="${CHAINS_PREPARED_ROOT:-${DATA_DIR}/prepared/chains_speech}"
CHAINS_LIST="${CHAINS_LIST:-${LIST_DIR}/chains_clean.txt}"
COMBINED_CLEAN_LIST="${COMBINED_CLEAN_LIST:-${LIST_DIR}/clean_all.with_chains.txt}"

PREPROCESS_OUTPUT_ROOT="${CLI_PREPROCESS_OUTPUT_ROOT:-${PREPROCESS_OUTPUT_ROOT:-${DATA_DIR}/preprocessed/dfn3_speech_clean}}"
PREPROCESS_BASE_DIR="${CLI_PREPROCESS_BASE_DIR:-${PREPROCESS_BASE_DIR:-${DATA_DIR}/raw}}"
PREPROCESS_OUTPUT_LIST="${CLI_PREPROCESS_OUTPUT_LIST:-${PREPROCESS_OUTPUT_LIST:-${LIST_DIR}/clean_all.preprocessed.txt}}"
PREPROCESS_MODEL="${CLI_PREPROCESS_MODEL:-${PREPROCESS_MODEL:-${DEFAULT_PREPROCESS_MODEL}}}"
PREPROCESS_DEVICE="${CLI_PREPROCESS_DEVICE:-${PREPROCESS_DEVICE:-}}"
PREPROCESS_PROBE_WORKERS="${CLI_PREPROCESS_PROBE_WORKERS:-${PREPROCESS_PROBE_WORKERS:-}}"
PREPROCESS_PROBE_CACHE="${CLI_PREPROCESS_PROBE_CACHE:-${PREPROCESS_PROBE_CACHE:-}}"
PREPROCESS_ENHANCE_BATCH_SIZE="${CLI_PREPROCESS_ENHANCE_BATCH_SIZE:-${PREPROCESS_ENHANCE_BATCH_SIZE:-}}"
MUSIC_PREPARE_OUTPUT_ROOT="${CLI_MUSIC_PREPARE_OUTPUT_ROOT:-${MUSIC_PREPARE_OUTPUT_ROOT:-${DATA_DIR}/prepared/background_music_roomy}}"
MUSIC_PREPARE_BASE_DIR="${CLI_MUSIC_PREPARE_BASE_DIR:-${MUSIC_PREPARE_BASE_DIR:-${DATA_DIR}/raw}}"
MUSIC_PREPARE_OUTPUT_LIST="${CLI_MUSIC_PREPARE_OUTPUT_LIST:-${MUSIC_PREPARE_OUTPUT_LIST:-${LIST_DIR}/background_music.prepared.txt}}"
MUSIC_PREPARE_RIR_LIST="${CLI_MUSIC_PREPARE_RIR_LIST:-${MUSIC_PREPARE_RIR_LIST:-${RIR_LIST}}}"
MUSIC_PREPARE_STYLE="${CLI_MUSIC_PREPARE_STYLE:-${MUSIC_PREPARE_STYLE:-speaker_room}}"
MUSIC_PREPARE_VARIANTS="${CLI_MUSIC_PREPARE_VARIANTS:-${MUSIC_PREPARE_VARIANTS:-2}}"
MUSIC_PREPARE_SEED="${CLI_MUSIC_PREPARE_SEED:-${MUSIC_PREPARE_SEED:-1337}}"
MUSIC_PREPARE_MERGED_LIST="${MUSIC_PREPARE_MERGED_LIST:-${LIST_DIR}/background_music.prepared_merged.txt}"

case "${PROFILE}" in
  prototype)
    NUM_WORKERS_DEFAULT=1
    SHARD_SIZE_DEFAULT=100
    ;;
  production)
    NUM_WORKERS_DEFAULT=8
    SHARD_SIZE_DEFAULT=500
    ;;
  apple)
    case "${APPLE_SILICON_TIER}" in
      ultra)
        NUM_WORKERS_DEFAULT=8
        PREPROCESS_WORKERS_DEFAULT=4
        ;;
      max)
        NUM_WORKERS_DEFAULT=6
        PREPROCESS_WORKERS_DEFAULT=4
        ;;
      pro)
        NUM_WORKERS_DEFAULT=4
        PREPROCESS_WORKERS_DEFAULT=2
        ;;
      *)
        NUM_WORKERS_DEFAULT=2
        PREPROCESS_WORKERS_DEFAULT=1
        ;;
    esac
    SHARD_SIZE_DEFAULT=500
    ;;
  *)
    echo "Error: unsupported PROFILE '${PROFILE}' (expected prototype, production, or apple)" >&2
    exit 1
    ;;
esac

if [[ "${PROFILE}" != "apple" ]]; then
  PREPROCESS_WORKERS_DEFAULT=2
fi

NUM_WORKERS="${CLI_NUM_WORKERS:-${NUM_WORKERS:-${NUM_WORKERS_DEFAULT}}}"
SHARD_SIZE="${CLI_SHARD_SIZE:-${SHARD_SIZE:-${SHARD_SIZE_DEFAULT}}}"
MAX_PENDING_BYTES="${CLI_MAX_PENDING_BYTES:-${MAX_PENDING_BYTES:-8}}"
MIN_DURATION="${CLI_MIN_DURATION:-${MIN_DURATION:-${SEGMENT_LENGTH}}}"
MERGE_SHORT="${CLI_MERGE_SHORT:-${MERGE_SHORT:-false}}"
PREPROCESS_WORKERS="${CLI_PREPROCESS_WORKERS:-${PREPROCESS_WORKERS:-${PREPROCESS_WORKERS_DEFAULT}}}"

MUSIC_LIST_SELECTION_NOTE=""
MUSIC_LIST_TOTAL_ENTRIES=0
MUSIC_LIST_EXISTING_ENTRIES=0
if [[ -f "${MUSIC_LIST}" ]]; then
  read -r MUSIC_LIST_TOTAL_ENTRIES MUSIC_LIST_EXISTING_ENTRIES < <(file_list_entry_stats "${MUSIC_LIST}")
fi
if [[ "${MUSIC_LIST}" == "${LIST_DIR}/background_music.txt" && "${MUSIC_LIST_EXISTING_ENTRIES}" -eq 0 && -f "${LIST_DIR}/background_music_expanded.txt" ]]; then
  expanded_total=0
  expanded_existing=0
  read -r expanded_total expanded_existing < <(file_list_entry_stats "${LIST_DIR}/background_music_expanded.txt")
  if [[ "${expanded_existing}" -gt 0 ]]; then
    MUSIC_LIST_SELECTION_NOTE="background_music.txt has 0 existing entries; using background_music_expanded.txt instead"
    MUSIC_LIST="${LIST_DIR}/background_music_expanded.txt"
    MUSIC_LIST_TOTAL_ENTRIES="${expanded_total}"
    MUSIC_LIST_EXISTING_ENTRIES="${expanded_existing}"
  fi
fi

echo "=============================================="
echo "DeepFilterNet MLX Audio Cache Builder"
echo "=============================================="
echo "Profile:            ${PROFILE}"
if [[ "${PROFILE}" == "apple" ]]; then
  echo "Apple tier:         ${APPLE_SILICON_TIER}"
fi
echo "Root dir:           ${ROOT_DIR}"
echo "Python:             ${PYTHON_BIN}"
echo "Data dir:           ${DATA_DIR}"
echo "Output dir:         ${OUTPUT_DIR}"
echo "List dir:           ${LIST_DIR}"
echo "CHAINS speech:      $([[ ${INCLUDE_CHAINS} -eq 1 ]] && echo "enabled" || echo "disabled")"
if [[ ${INCLUDE_CHAINS} -eq 1 ]]; then
  echo "CHAINS dir:         ${CHAINS_DIR}"
  echo "CHAINS prepared:    ${CHAINS_PREPARED_ROOT}"
  echo "CHAINS list:        ${CHAINS_LIST}"
fi
echo "Clean list:         ${CLEAN_LIST}"
echo "Noise list:         ${NOISE_LIST}"
if [[ -f "${MUSIC_LIST}" ]]; then
  echo "Music list:         ${MUSIC_LIST}"
  echo "Music entries:      ${MUSIC_LIST_EXISTING_ENTRIES}/${MUSIC_LIST_TOTAL_ENTRIES} existing"
  if [[ "${MUSIC_LIST}" == "${LIST_DIR}/background_music.txt" ]]; then
    echo "Music flavor:       curated chart-style set (FMA + optional MTG-Jamendo)"
  elif [[ "${MUSIC_LIST}" == "${LIST_DIR}/background_music_expanded.txt" ]]; then
    echo "Music flavor:       expanded chart-style eligible pool (FMA + optional MTG-Jamendo)"
  fi
  if [[ -n "${MUSIC_LIST_SELECTION_NOTE}" ]]; then
    echo "Music selection:    ${MUSIC_LIST_SELECTION_NOTE}"
  fi
else
  echo "Music list:         (none - dedicated background music disabled)"
fi
echo "Prepare music:      $([[ ${PREPARE_BACKGROUND_MUSIC} -eq 1 ]] && echo "enabled" || echo "disabled")"
if [[ ${PREPARE_BACKGROUND_MUSIC} -eq 1 ]]; then
  echo "Music prep root:    ${MUSIC_PREPARE_OUTPUT_ROOT}"
  echo "Music prep base:    ${MUSIC_PREPARE_BASE_DIR}"
  echo "Music prep list:    ${MUSIC_PREPARE_OUTPUT_LIST}"
  echo "Music prep style:   ${MUSIC_PREPARE_STYLE}"
  echo "Music prep variants: ${MUSIC_PREPARE_VARIANTS}"
  echo "Music prep seed:    ${MUSIC_PREPARE_SEED}"
  if [[ -n "${MUSIC_PREPARE_RIR_LIST}" ]]; then
    echo "Music prep RIRs:    ${MUSIC_PREPARE_RIR_LIST}"
  else
    echo "Music prep RIRs:    (none)"
  fi
  echo "Music prep mode:    $([[ ${MUSIC_PREPARE_OVERWRITE} -eq 1 ]] && echo "overwrite" || echo "resume")"
fi
if [[ -f "${RIR_LIST}" ]]; then
  echo "RIR list:           ${RIR_LIST}"
else
  echo "RIR list:           (none - RIR augmentation disabled)"
fi
echo "Sample rate:        ${SR} Hz"
echo "Segment length:     ${SEGMENT_LENGTH}s"
echo "Min duration:       ${MIN_DURATION}s"
echo "Short speech mode:  ${MERGE_SHORT}"
echo "SNR range:          [${SNR_MIN}, ${SNR_MAX}] dB"
echo "RIR prob:           ${RIR_PROB}"
echo "Workers:            ${NUM_WORKERS}"
echo "Shard size:         ${SHARD_SIZE}"
echo "Max pending budget: ${MAX_PENDING_BYTES} GB"
if [[ ${PREPROCESS_CLEAN_SPEECH} -eq 1 ]]; then
  echo "Preprocess speech:  enabled"
  echo "Preprocess model:   ${PREPROCESS_MODEL}"
  echo "Preprocess backend: auto (Apple Silicon uses df_mlx for MLX bundles; torch otherwise)"
  echo "Preprocess root:    ${PREPROCESS_OUTPUT_ROOT}"
  echo "Preprocess base:    ${PREPROCESS_BASE_DIR}"
  echo "Preprocess list:    ${PREPROCESS_OUTPUT_LIST}"
  echo "Preprocess workers: ${PREPROCESS_WORKERS}"
  if [[ -n "${PREPROCESS_PROBE_WORKERS}" ]]; then
    echo "Preprocess probe workers: ${PREPROCESS_PROBE_WORKERS}"
  else
    echo "Preprocess probe workers: auto"
  fi
  if [[ -n "${PREPROCESS_PROBE_CACHE}" ]]; then
    echo "Preprocess probe cache: ${PREPROCESS_PROBE_CACHE}"
  else
    echo "Preprocess probe cache: auto"
  fi
  if [[ -n "${PREPROCESS_DEVICE}" ]]; then
    echo "Preprocess device:  ${PREPROCESS_DEVICE}"
  else
    echo "Preprocess device:  auto"
  fi
  echo "Preprocess mode:    $([[ ${PREPROCESS_OVERWRITE} -eq 1 ]] && echo "overwrite" || echo "resume")"
  if [[ -n "${PREPROCESS_ENHANCE_BATCH_SIZE}" ]]; then
    echo "Preprocess batch:   ${PREPROCESS_ENHANCE_BATCH_SIZE}"
  else
    echo "Preprocess batch:   auto"
  fi
else
  echo "Preprocess speech:  disabled"
fi
if [[ ${FORCE_ALL} -eq 1 ]]; then
  echo "Force mode:         enabled (all phase completion checks disabled)"
fi
echo "=============================================="

if [[ "${MERGE_SHORT}" != "true" && "${MIN_DURATION}" != "0" && "${MIN_DURATION}" != "0.0" ]]; then
  echo "Warning: speech clips shorter than ${MIN_DURATION}s will be skipped."
  echo "         For more speech diversity, consider --merge-short or --min-duration 0."
  echo ""
fi

if [[ ! -f "${CLEAN_LIST}" ]]; then
  echo "Error: clean speech list not found: ${CLEAN_LIST}" >&2
  exit 1
fi
if [[ ! -f "${NOISE_LIST}" ]]; then
  echo "Error: noise list not found: ${NOISE_LIST}" >&2
  exit 1
fi

CLEAN_LIST_TO_USE="${CLEAN_LIST}"
MUSIC_LIST_INPUT="${MUSIC_LIST}"
MUSIC_LIST_TO_USE="${MUSIC_LIST}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LIST_DIR}"

cd "${ROOT_DIR}/DeepFilterNet"

SECONDS=0
if [[ ${INCLUDE_CHAINS} -eq 1 ]]; then
  if [[ ! -d "${CHAINS_DIR}" ]]; then
    echo "Error: CHAINS corpus root not found: ${CHAINS_DIR}" >&2
    exit 1
  fi

  CHAINS_SKIPPED=0
  if [[ ${FORCE_ALL} -eq 0 ]]; then
    if verify_result="$(verify_file_list "${CHAINS_LIST}" 2>/dev/null)"; then
      read -r vfl_valid vfl_total vfl_time <<< "${verify_result}"
      printf '[skip] CHAINS preparation: %s files verified on disk (%ss)\n' \
        "$(printf '%d' "${vfl_valid}")" "${vfl_time}"
      CHAINS_SKIPPED=1
    fi
  elif [[ ${FORCE_ALL} -eq 1 ]]; then
    echo "[force] CHAINS preparation: --force set, running phase"
  fi

  if [[ ${CHAINS_SKIPPED} -eq 0 ]]; then
    echo ""
    echo "Preparing CHAINS clean-speech additions..."
    chains_cmd=(
      "${PYTHON_BIN}"
      "${ROOT_DIR}/scripts/datasets/prepare_chains_speech.py"
      --chains-dir "${CHAINS_DIR}"
      --prepared-root "${CHAINS_PREPARED_ROOT}"
      --output-list "${CHAINS_LIST}"
    )
    "${chains_cmd[@]}"
    if [[ ! -f "${CHAINS_LIST}" ]]; then
      echo "Error: CHAINS preparation did not produce list ${CHAINS_LIST}" >&2
      exit 1
    fi
  fi
  # Always regenerate the merged clean list (even when the producer was skipped)
  merge_unique_file_lists "${COMBINED_CLEAN_LIST}" "${CLEAN_LIST}" "${CHAINS_LIST}"
  CLEAN_LIST_TO_USE="${COMBINED_CLEAN_LIST}"
fi
echo "[timing] CHAINS preparation: $(phase_elapsed)"

SECONDS=0
if [[ ${PREPROCESS_CLEAN_SPEECH} -eq 1 ]]; then
  PREPROCESS_BASE_DIR_TO_USE="${PREPROCESS_BASE_DIR}"
  if [[ ${INCLUDE_CHAINS} -eq 1 && ${PREPROCESS_BASE_DIR_WAS_SET} -eq 0 ]]; then
    PREPROCESS_BASE_DIR_TO_USE="$(compute_common_base_dir "${CLEAN_LIST_TO_USE}")"
  fi

  PREPROCESS_SKIPPED=0
  if [[ ${FORCE_ALL} -eq 0 && ${PREPROCESS_OVERWRITE} -eq 0 ]]; then
    # Two-part check: (1) every input has a valid output file on disk,
    # AND (2) the output list itself is valid (all referenced files exist).
    # Both must pass — a crash can leave outputs on disk without a valid list.
    if pp_result="$(check_preprocess_complete "${CLEAN_LIST_TO_USE}" "${PREPROCESS_OUTPUT_ROOT}" "${PREPROCESS_BASE_DIR_TO_USE}" 2>/dev/null)"; then
      if verify_file_list "${PREPROCESS_OUTPUT_LIST}" >/dev/null 2>&1; then
        read -r pp_covered pp_total pp_time <<< "${pp_result}"
        printf '[skip] Clean-speech preprocessing: %s/%s inputs have valid outputs (%ss)\n' \
          "$(printf '%d' "${pp_covered}")" "$(printf '%d' "${pp_total}")" "${pp_time}"
        PREPROCESS_SKIPPED=1
      else
        read -r pp_covered pp_total pp_time <<< "${pp_result}"
        printf '[check] Clean-speech preprocessing: outputs exist but list is stale → running phase to regenerate list\n'
      fi
    else
      if [[ -n "${pp_result:-}" ]]; then
        read -r pp_covered pp_total pp_time <<< "${pp_result}"
        printf '[check] Clean-speech preprocessing: %s/%s inputs covered, %d pending → running phase\n' \
          "$(printf '%d' "${pp_covered}")" "$(printf '%d' "${pp_total}")" "$((pp_total - pp_covered))"
      fi
    fi
  elif [[ ${PREPROCESS_OVERWRITE} -eq 1 ]]; then
    echo "[force] Clean-speech preprocessing: --preprocess-overwrite set, running phase"
  elif [[ ${FORCE_ALL} -eq 1 ]]; then
    echo "[force] Clean-speech preprocessing: --force set, running phase"
  fi

  if [[ ${PREPROCESS_SKIPPED} -eq 0 ]]; then
    echo ""
    echo "Running clean-speech preprocessing before cache build..."
    echo "Only the clean/speech list is eligible; noise and RIR lists are left untouched."
    if [[ "${PREPROCESS_BASE_DIR_TO_USE}" != "${PREPROCESS_BASE_DIR}" ]]; then
      echo "Preprocess base auto-adjusted for combined speech roots: ${PREPROCESS_BASE_DIR_TO_USE}"
    fi
    preprocess_cmd=(
      "${PYTHON_BIN}"
      "${ROOT_DIR}/scripts/datasets/preprocess_clean_speech.py"
      --file-list "${CLEAN_LIST_TO_USE}"
      --output-root "${PREPROCESS_OUTPUT_ROOT}"
      --base-dir "${PREPROCESS_BASE_DIR_TO_USE}"
      --output-list "${PREPROCESS_OUTPUT_LIST}"
      --model-base-dir "${PREPROCESS_MODEL}"
      --num-workers "${PREPROCESS_WORKERS}"
    )
    if [[ -n "${PREPROCESS_PROBE_WORKERS}" ]]; then
      preprocess_cmd+=(--probe-workers "${PREPROCESS_PROBE_WORKERS}")
    fi
    if [[ -n "${PREPROCESS_PROBE_CACHE}" ]]; then
      preprocess_cmd+=(--probe-cache "${PREPROCESS_PROBE_CACHE}")
    fi
    if [[ -n "${PREPROCESS_DEVICE}" ]]; then
      preprocess_cmd+=(--device "${PREPROCESS_DEVICE}")
    fi
    if [[ -n "${PREPROCESS_ENHANCE_BATCH_SIZE}" ]]; then
      preprocess_cmd+=(--enhance-batch-size "${PREPROCESS_ENHANCE_BATCH_SIZE}")
    fi
    if [[ ${PREPROCESS_OVERWRITE} -eq 1 ]]; then
      preprocess_cmd+=(--overwrite)
    fi
    "${preprocess_cmd[@]}"
  fi
  CLEAN_LIST_TO_USE="${PREPROCESS_OUTPUT_LIST}"
  if [[ ! -f "${CLEAN_LIST_TO_USE}" ]]; then
    echo "Error: preprocessing did not produce clean list ${CLEAN_LIST_TO_USE}" >&2
    exit 1
  fi
fi
echo "[timing] Clean-speech preprocessing: $(phase_elapsed)"

if [[ -f "${MUSIC_LIST}" ]]; then
  if [[ "${MUSIC_LIST_EXISTING_ENTRIES}" -le 0 ]]; then
    echo "[warn] music list has no existing files: ${MUSIC_LIST}" >&2
    MUSIC_LIST_INPUT=""
    MUSIC_LIST_TO_USE=""
  elif [[ "${MUSIC_LIST_EXISTING_ENTRIES}" -lt "${MUSIC_LIST_TOTAL_ENTRIES}" ]]; then
    MUSIC_LIST_INPUT="$(mktemp "${TMPDIR:-/tmp}/dfn-music-list.XXXXXX.txt")"
    temp_cleanup_paths+=("${MUSIC_LIST_INPUT}")
    sanitize_existing_file_list "${MUSIC_LIST}" "${MUSIC_LIST_INPUT}"
    echo "[warn] filtered $((MUSIC_LIST_TOTAL_ENTRIES - MUSIC_LIST_EXISTING_ENTRIES)) missing music paths from ${MUSIC_LIST}" >&2
    MUSIC_LIST_TO_USE="${MUSIC_LIST_INPUT}"
  fi
fi

SECONDS=0
if [[ ${PREPARE_BACKGROUND_MUSIC} -eq 1 ]]; then
  if [[ -z "${MUSIC_LIST_INPUT}" || ! -f "${MUSIC_LIST_INPUT}" ]]; then
    echo ""
    echo "Background-music preparation requested, but no usable music list is available."
    echo "Continuing without prepared music variants."
  else
    MUSIC_PREP_SKIPPED=0
    if [[ ${FORCE_ALL} -eq 0 && ${MUSIC_PREPARE_OVERWRITE} -eq 0 ]]; then
      # Two-part check: (1) every input source has variant_0 on disk,
      # AND (2) the output list itself is valid.
      if mp_result="$(check_music_prep_complete "${MUSIC_LIST_INPUT}" "${MUSIC_PREPARE_OUTPUT_ROOT}" "${MUSIC_PREPARE_BASE_DIR}" "${MUSIC_PREPARE_STYLE}" 2>/dev/null)"; then
        if verify_file_list "${MUSIC_PREPARE_OUTPUT_LIST}" >/dev/null 2>&1; then
          read -r mp_covered mp_total mp_time <<< "${mp_result}"
          printf '[skip] Background-music preparation: %s sources fully prepared (%ss)\n' \
            "$(printf '%d' "${mp_covered}")" "${mp_time}"
          MUSIC_PREP_SKIPPED=1
        else
          printf '[check] Background-music preparation: outputs exist but list is stale → running phase to regenerate list\n'
        fi
      else
        if [[ -n "${mp_result:-}" ]]; then
          read -r mp_covered mp_total mp_time <<< "${mp_result}"
          printf '[check] Background-music preparation: %s/%s sources covered, %d pending → running phase\n' \
            "$(printf '%d' "${mp_covered}")" "$(printf '%d' "${mp_total}")" "$((mp_total - mp_covered))"
        fi
      fi
    elif [[ ${MUSIC_PREPARE_OVERWRITE} -eq 1 ]]; then
      echo "[force] Background-music preparation: --music-prepare-overwrite set, running phase"
    elif [[ ${FORCE_ALL} -eq 1 ]]; then
      echo "[force] Background-music preparation: --force set, running phase"
    fi

    if [[ ${MUSIC_PREP_SKIPPED} -eq 0 ]]; then
      echo ""
      echo "Preparing degraded background-music variants before cache build..."
      music_prepare_cmd=(
        "${PYTHON_BIN}"
        "${ROOT_DIR}/scripts/datasets/prepare_background_music.py"
        --file-list "${MUSIC_LIST_INPUT}"
        --output-root "${MUSIC_PREPARE_OUTPUT_ROOT}"
        --base-dir "${MUSIC_PREPARE_BASE_DIR}"
        --output-list "${MUSIC_PREPARE_OUTPUT_LIST}"
        --sample-rate "${SR}"
        --style "${MUSIC_PREPARE_STYLE}"
        --variants-per-source "${MUSIC_PREPARE_VARIANTS}"
        --seed "${MUSIC_PREPARE_SEED}"
      )
      if [[ -n "${MUSIC_PREPARE_RIR_LIST}" && -f "${MUSIC_PREPARE_RIR_LIST}" ]]; then
        music_prepare_cmd+=(--rir-list "${MUSIC_PREPARE_RIR_LIST}")
      fi
      if [[ ${MUSIC_PREPARE_OVERWRITE} -eq 1 ]]; then
        music_prepare_cmd+=(--overwrite)
      fi
      "${music_prepare_cmd[@]}"
      if [[ ! -f "${MUSIC_PREPARE_OUTPUT_LIST}" ]]; then
        echo "Error: music preparation did not produce list ${MUSIC_PREPARE_OUTPUT_LIST}" >&2
        exit 1
      fi
    fi
    # Always regenerate the merged list (even when the producer was skipped)
    merge_unique_file_lists "${MUSIC_PREPARE_MERGED_LIST}" "${MUSIC_LIST_INPUT}" "${MUSIC_PREPARE_OUTPUT_LIST}"
    MUSIC_LIST_TO_USE="${MUSIC_PREPARE_MERGED_LIST}"
  fi
fi
echo "[timing] Background-music preparation: $(phase_elapsed)"

SECONDS=0
echo ""
echo "Starting audio cache build..."
echo "Speech list used: ${CLEAN_LIST_TO_USE}"
echo "Resume mode is enabled - previously cached files will be skipped."
echo ""

build_cmd=(
  "${PYTHON_BIN}"
  -m
  df_mlx.build_audio_cache
  --speech-list "${CLEAN_LIST_TO_USE}"
  --noise-list "${NOISE_LIST}"
  --output-dir "${OUTPUT_DIR}"
  --sample-rate "${SR}"
  --segment-length "${SEGMENT_LENGTH}"
  --min-duration "${MIN_DURATION}"
  --shard-size "${SHARD_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --snr-min "${SNR_MIN}"
  --snr-max "${SNR_MAX}"
  --p-reverb "${RIR_PROB}"
  --resume
  --max-pending-bytes "${MAX_PENDING_BYTES}"
)

if [[ -f "${MUSIC_LIST_TO_USE}" ]]; then
  build_cmd+=(--music-list "${MUSIC_LIST_TO_USE}")
fi

if [[ -f "${RIR_LIST}" ]]; then
  build_cmd+=(--rir-list "${RIR_LIST}")
fi
if [[ "${MERGE_SHORT}" == "true" ]]; then
  build_cmd+=(--merge-short)
fi

"${build_cmd[@]}"
echo "[timing] Audio cache build: $(phase_elapsed)"

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo "Audio cache:       ${OUTPUT_DIR}"
echo "Config:            ${OUTPUT_DIR}/config.json"
echo "Speech list used:  ${CLEAN_LIST_TO_USE}"
if [[ -f "${MUSIC_LIST_TO_USE}" ]]; then
  echo "Music list used:   ${MUSIC_LIST_TO_USE}"
fi
echo ""
echo "Validate cache:"
echo "  ${PYTHON_BIN} -m df_mlx.validate_audio_cache \"${OUTPUT_DIR}\""
echo ""
echo "Recommended full run (vadlite-style):"
echo "  ${PYTHON_BIN} -m df_mlx.train_dynamic \\"
echo "    --run-config df_mlx/configs/run_profiles/baseline_dfn3_gan_vad_speech_full_vadlite.toml \\"
echo "    --cache-dir \"${OUTPUT_DIR}\""
echo "=============================================="
