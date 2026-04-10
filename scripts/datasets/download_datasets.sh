#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DEFAULT_DATA_DIR="/Volumes/TrainingData/datasets"
if [[ ! -d "${DEFAULT_DATA_DIR}" ]]; then
  DEFAULT_DATA_DIR="${ROOT_DIR}/data"
fi

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

INTERRUPTED=0
on_interrupt() {
  INTERRUPTED=1
  echo "[info] download interrupted by user" >&2
  exit 130
}
trap on_interrupt INT

usage_helptext() {
  cat <<EOF
Usage:
  ./download_datasets.sh [options]

Download/extract the speech, noise, background-music, and RIR corpora used for
DeepFilterNet training, then generate the file lists consumed by the datastore
and HDF5 builders.

Core options:
  --data-dir PATH                 Base dataset directory (default: ${DEFAULT_DATA_DIR})
  --download-dir PATH             Archive download directory (default: DATA_DIR/downloads)
  --extract-dir PATH              Extraction/raw-data directory (default: DATA_DIR/raw)
  --list-dir PATH                 Output directory for generated file lists (default: DATA_DIR/lists)
  --profile NAME                  prototype | production | apple (default: prototype)
  --download                      Force download mode on (default: enabled)
  --no-download                   Skip downloads and only regenerate file lists

Download behavior:
  --agree-licenses                Confirm dataset license acceptance (default: enabled)
  --no-agree-licenses             Require explicit approval via env/flag before downloading
  --keep-archives                 Retain downloaded archives after extraction (default: enabled)
  --no-keep-archives              Delete archives after extraction
  --resume                        Resume partial downloads when possible (default: enabled)
  --no-resume                     Redownload from scratch when a partial file exists

Transfer tuning:
  --use-aria2                     Use aria2c when available (default: enabled)
  --no-aria2                      Disable aria2c and fall back to curl/wget
  --aria2-parallel                Allow multi-URL aria2 queue mode (default: enabled)
  --no-aria2-parallel             Disable aria2 queue mode
  --aria2-conn N                  Connections per file (default: 16)
  --aria2-split N                 aria2 split count per file (default: 16)
  --aria2-min-split SIZE          Minimum aria2 split size (default: 1M)
  --aria2-max-concurrent N        Max concurrent aria2 downloads (default: 8)
  --aria2-file-alloc MODE         aria2 file allocation mode (default: none)
  --aria2-user-agent STRING       User agent for aria2/curl/wget (default: Mozilla/5.0)
  --zenodo-referer URL            Referer header for Zenodo downloads (default: https://zenodo.org/records/4060432/)
  --verify-cache                  Cache archive verification results (default: enabled)
  --no-verify-cache               Disable archive verification caching
  --verify-cache-file PATH        Verification cache file (default: DOWNLOAD_DIR/.verify_cache.tsv)
  --use-gh-auth                   Use gh auth token for GitHub URLs when available (default: enabled)
  --no-gh-auth                    Never request a gh auth token

AIR/OpenAIR via audb:
  --use-audb                      Use audb for AIR/OpenAIR downloads (default: enabled)
  --no-audb                       Disable audb-backed AIR/OpenAIR downloads
  --install-audb                  Install audb automatically when missing
  --no-install-audb               Do not auto-install audb (default)
  --audb-dir PATH                 audb root directory (default: EXTRACT_DIR/audb)
  --air-version VERSION           AIR audb version (default: 1.4.2)
  --openair-version VERSION       OpenAIR audb version (default: 1.0.0)

Dataset selection:
  --download-vctk / --no-download-vctk                       Override VCTK download toggle (default: enabled)
  --download-librispeech / --no-download-librispeech         Override LibriSpeech toggle (default: production=1, apple/prototype=0)
  --download-musan / --no-download-musan                     Override MUSAN toggle (default: enabled)
  --download-fma / --no-download-fma                         Override FMA download toggle (default: enabled)
  --download-mtg-jamendo / --no-download-mtg-jamendo         Override MTG-Jamendo download toggle (default: disabled)
  --download-fsd50k / --no-download-fsd50k                   Override FSD50K toggle (default: enabled)
  --download-air / --no-download-air                         Override AIR toggle (default: enabled)
  --download-openair / --no-download-openair                 Override OpenAIR toggle (default: enabled)
  --download-acousticrooms / --no-download-acousticrooms     Override AcousticRooms toggle (default: production=1, apple/prototype=0)

Dataset path overrides:
  --vctk-dir PATH                 Existing VCTK root (default: EXTRACT_DIR/VCTK-Corpus-0.92)
  --librispeech-dir PATH          Existing LibriSpeech root (default: EXTRACT_DIR/LibriSpeech)
  --musan-dir PATH                Existing MUSAN root (default: EXTRACT_DIR/musan)
  --fma-dir PATH                  Existing FMA root (default: EXTRACT_DIR/FMA)
  --mtg-jamendo-dir PATH          Existing MTG-Jamendo root (default: EXTRACT_DIR/mtg-jamendo)
  --fsd50k-dir PATH               Existing FSD50K root (default: EXTRACT_DIR/FSD50K)
  --air-rir-dir PATH              Existing AIR RIR root (default: AUDB_DIR/data)
  --openair-dir PATH              Existing OpenAIR root (default: AUDB_DIR/wav)
  --acousticrooms-dir PATH        Existing AcousticRooms root (default: EXTRACT_DIR/AcousticRooms)

Source overrides:
  --vctk-url URL                  VCTK archive URL (default: Zenodo mirror of VCTK 0.92 zip; matches official MD5)
  --librispeech-parts STRING      Space-separated LibriSpeech parts (default: profile-specific)
  --fma-subset NAME               FMA audio subset: medium | large | full (default: profile-specific)
  --background-music-target-count N
                                  Curated chart-style song target count (default: 2000)
  --background-music-min-count N  Minimum acceptable curated song count (default: 500)
  --fsd50k-base-url URL           FSD50K base URL (default: https://zenodo.org/records/4060432/files)

General:
  -h, --help                      Show this help message and exit

Environment variables remain supported and are used as fallbacks when the
equivalent CLI option is not provided; CLI flags take precedence.

Examples:
  # Rebuild file lists from datasets you already downloaded.
  ./download_datasets.sh --profile apple --no-download

  # Download the production corpus mix, then generate file lists.
  ./download_datasets.sh --profile production --download
EOF
}

CLI_DATA_DIR=""
CLI_DOWNLOAD_DIR=""
CLI_EXTRACT_DIR=""
CLI_LIST_DIR=""
CLI_PROFILE=""
CLI_DOWNLOAD=""
CLI_AGREE_LICENSES=""
CLI_KEEP_ARCHIVES=""
CLI_RESUME=""
CLI_USE_ARIA2=""
CLI_ARIA2_PARALLEL=""
CLI_ARIA2_CONN=""
CLI_ARIA2_SPLIT=""
CLI_ARIA2_MIN_SPLIT=""
CLI_ARIA2_MAX_CONCURRENT=""
CLI_ARIA2_FILE_ALLOC=""
CLI_ARIA2_USER_AGENT=""
CLI_ZENODO_REFERER=""
CLI_VERIFY_CACHE=""
CLI_VERIFY_CACHE_FILE=""
CLI_USE_GH_AUTH=""
CLI_USE_AUDB=""
CLI_INSTALL_AUDB=""
CLI_AUDB_DIR=""
CLI_AIR_VERSION=""
CLI_OPENAIR_VERSION=""
CLI_DOWNLOAD_VCTK=""
CLI_DOWNLOAD_LIBRISPEECH=""
CLI_DOWNLOAD_MUSAN=""
CLI_DOWNLOAD_FMA=""
CLI_DOWNLOAD_MTG_JAMENDO=""
CLI_DOWNLOAD_FSD50K=""
CLI_DOWNLOAD_AIR=""
CLI_DOWNLOAD_OPENAIR=""
CLI_DOWNLOAD_ACOUSTICROOMS=""
CLI_VCTK_DIR=""
CLI_LIBRISPEECH_DIR=""
CLI_MUSAN_DIR=""
CLI_FMA_DIR=""
CLI_MTG_JAMENDO_DIR=""
CLI_FSD50K_DIR=""
CLI_AIR_RIR_DIR=""
CLI_OPENAIR_DIR=""
CLI_ACOUSTICROOMS_DIR=""
CLI_VCTK_URL=""
CLI_LIBRISPEECH_PARTS=""
CLI_FMA_SUBSET=""
CLI_BACKGROUND_MUSIC_TARGET_COUNT=""
CLI_BACKGROUND_MUSIC_MIN_COUNT=""
CLI_FSD50K_BASE_URL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      CLI_DATA_DIR="$2"
      shift 2
      ;;
    --download-dir)
      CLI_DOWNLOAD_DIR="$2"
      shift 2
      ;;
    --extract-dir)
      CLI_EXTRACT_DIR="$2"
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
    --download)
      CLI_DOWNLOAD="1"
      shift
      ;;
    --no-download)
      CLI_DOWNLOAD="0"
      shift
      ;;
    --agree-licenses)
      CLI_AGREE_LICENSES="1"
      shift
      ;;
    --no-agree-licenses)
      CLI_AGREE_LICENSES="0"
      shift
      ;;
    --keep-archives)
      CLI_KEEP_ARCHIVES="1"
      shift
      ;;
    --no-keep-archives)
      CLI_KEEP_ARCHIVES="0"
      shift
      ;;
    --resume)
      CLI_RESUME="1"
      shift
      ;;
    --no-resume)
      CLI_RESUME="0"
      shift
      ;;
    --use-aria2)
      CLI_USE_ARIA2="1"
      shift
      ;;
    --no-aria2)
      CLI_USE_ARIA2="0"
      shift
      ;;
    --aria2-parallel)
      CLI_ARIA2_PARALLEL="1"
      shift
      ;;
    --no-aria2-parallel)
      CLI_ARIA2_PARALLEL="0"
      shift
      ;;
    --aria2-conn)
      CLI_ARIA2_CONN="$2"
      shift 2
      ;;
    --aria2-split)
      CLI_ARIA2_SPLIT="$2"
      shift 2
      ;;
    --aria2-min-split)
      CLI_ARIA2_MIN_SPLIT="$2"
      shift 2
      ;;
    --aria2-max-concurrent)
      CLI_ARIA2_MAX_CONCURRENT="$2"
      shift 2
      ;;
    --aria2-file-alloc)
      CLI_ARIA2_FILE_ALLOC="$2"
      shift 2
      ;;
    --aria2-user-agent)
      CLI_ARIA2_USER_AGENT="$2"
      shift 2
      ;;
    --zenodo-referer)
      CLI_ZENODO_REFERER="$2"
      shift 2
      ;;
    --verify-cache)
      CLI_VERIFY_CACHE="1"
      shift
      ;;
    --no-verify-cache)
      CLI_VERIFY_CACHE="0"
      shift
      ;;
    --verify-cache-file)
      CLI_VERIFY_CACHE_FILE="$2"
      shift 2
      ;;
    --use-gh-auth)
      CLI_USE_GH_AUTH="1"
      shift
      ;;
    --no-gh-auth)
      CLI_USE_GH_AUTH="0"
      shift
      ;;
    --use-audb)
      CLI_USE_AUDB="1"
      shift
      ;;
    --no-audb)
      CLI_USE_AUDB="0"
      shift
      ;;
    --install-audb)
      CLI_INSTALL_AUDB="1"
      shift
      ;;
    --no-install-audb)
      CLI_INSTALL_AUDB="0"
      shift
      ;;
    --audb-dir)
      CLI_AUDB_DIR="$2"
      shift 2
      ;;
    --air-version)
      CLI_AIR_VERSION="$2"
      shift 2
      ;;
    --openair-version)
      CLI_OPENAIR_VERSION="$2"
      shift 2
      ;;
    --download-vctk)
      CLI_DOWNLOAD_VCTK="1"
      shift
      ;;
    --no-download-vctk)
      CLI_DOWNLOAD_VCTK="0"
      shift
      ;;
    --download-librispeech)
      CLI_DOWNLOAD_LIBRISPEECH="1"
      shift
      ;;
    --no-download-librispeech)
      CLI_DOWNLOAD_LIBRISPEECH="0"
      shift
      ;;
    --download-musan)
      CLI_DOWNLOAD_MUSAN="1"
      shift
      ;;
    --no-download-musan)
      CLI_DOWNLOAD_MUSAN="0"
      shift
      ;;
    --download-fma)
      CLI_DOWNLOAD_FMA="1"
      shift
      ;;
    --no-download-fma)
      CLI_DOWNLOAD_FMA="0"
      shift
      ;;
    --download-mtg-jamendo)
      CLI_DOWNLOAD_MTG_JAMENDO="1"
      shift
      ;;
    --no-download-mtg-jamendo)
      CLI_DOWNLOAD_MTG_JAMENDO="0"
      shift
      ;;
    --download-fsd50k)
      CLI_DOWNLOAD_FSD50K="1"
      shift
      ;;
    --no-download-fsd50k)
      CLI_DOWNLOAD_FSD50K="0"
      shift
      ;;
    --download-air)
      CLI_DOWNLOAD_AIR="1"
      shift
      ;;
    --no-download-air)
      CLI_DOWNLOAD_AIR="0"
      shift
      ;;
    --download-openair)
      CLI_DOWNLOAD_OPENAIR="1"
      shift
      ;;
    --no-download-openair)
      CLI_DOWNLOAD_OPENAIR="0"
      shift
      ;;
    --download-acousticrooms)
      CLI_DOWNLOAD_ACOUSTICROOMS="1"
      shift
      ;;
    --no-download-acousticrooms)
      CLI_DOWNLOAD_ACOUSTICROOMS="0"
      shift
      ;;
    --vctk-dir)
      CLI_VCTK_DIR="$2"
      shift 2
      ;;
    --librispeech-dir)
      CLI_LIBRISPEECH_DIR="$2"
      shift 2
      ;;
    --musan-dir)
      CLI_MUSAN_DIR="$2"
      shift 2
      ;;
    --fma-dir)
      CLI_FMA_DIR="$2"
      shift 2
      ;;
    --mtg-jamendo-dir)
      CLI_MTG_JAMENDO_DIR="$2"
      shift 2
      ;;
    --fsd50k-dir)
      CLI_FSD50K_DIR="$2"
      shift 2
      ;;
    --air-rir-dir)
      CLI_AIR_RIR_DIR="$2"
      shift 2
      ;;
    --openair-dir)
      CLI_OPENAIR_DIR="$2"
      shift 2
      ;;
    --acousticrooms-dir)
      CLI_ACOUSTICROOMS_DIR="$2"
      shift 2
      ;;
    --vctk-url)
      CLI_VCTK_URL="$2"
      shift 2
      ;;
    --librispeech-parts)
      CLI_LIBRISPEECH_PARTS="$2"
      shift 2
      ;;
    --fma-subset)
      CLI_FMA_SUBSET="$2"
      shift 2
      ;;
    --background-music-target-count)
      CLI_BACKGROUND_MUSIC_TARGET_COUNT="$2"
      shift 2
      ;;
    --background-music-min-count)
      CLI_BACKGROUND_MUSIC_MIN_COUNT="$2"
      shift 2
      ;;
    --fsd50k-base-url)
      CLI_FSD50K_BASE_URL="$2"
      shift 2
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

DATA_DIR="${CLI_DATA_DIR:-${DATA_DIR:-${DEFAULT_DATA_DIR}}}"
LIST_DIR="${CLI_LIST_DIR:-${LIST_DIR:-${DATA_DIR}/lists}}"

# Download controls (opt-in)
PROFILE="${CLI_PROFILE:-${PROFILE:-prototype}}"  # prototype | production | apple
DOWNLOAD="${CLI_DOWNLOAD:-${DOWNLOAD:-1}}"       # set to 1 to enable downloads
AGREE_LICENSES="${CLI_AGREE_LICENSES:-${AGREE_LICENSES:-1}}"  # set to 1 to confirm license acceptance
DOWNLOAD_DIR="${CLI_DOWNLOAD_DIR:-${DOWNLOAD_DIR:-${DATA_DIR}/downloads}}"
EXTRACT_DIR="${CLI_EXTRACT_DIR:-${EXTRACT_DIR:-${DATA_DIR}/raw}}"
KEEP_ARCHIVES="${CLI_KEEP_ARCHIVES:-${KEEP_ARCHIVES:-1}}"
RESUME="${CLI_RESUME:-${RESUME:-1}}"              # try to resume partial downloads when possible
USE_ARIA2="${CLI_USE_ARIA2:-${USE_ARIA2:-1}}"        # use aria2c if available
ARIA2_PARALLEL="${CLI_ARIA2_PARALLEL:-${ARIA2_PARALLEL:-1}}"  # download multiple URLs concurrently
ARIA2_CONN="${CLI_ARIA2_CONN:-${ARIA2_CONN:-16}}"      # connections per file
ARIA2_SPLIT="${CLI_ARIA2_SPLIT:-${ARIA2_SPLIT:-16}}"    # number of splits
ARIA2_MIN_SPLIT="${CLI_ARIA2_MIN_SPLIT:-${ARIA2_MIN_SPLIT:-1M}}"
ARIA2_MAX_CONCURRENT="${CLI_ARIA2_MAX_CONCURRENT:-${ARIA2_MAX_CONCURRENT:-8}}"
ARIA2_FILE_ALLOC="${CLI_ARIA2_FILE_ALLOC:-${ARIA2_FILE_ALLOC:-none}}"
ARIA2_USER_AGENT="${CLI_ARIA2_USER_AGENT:-${ARIA2_USER_AGENT:-Mozilla/5.0}}"
ZENODO_REFERER="${CLI_ZENODO_REFERER:-${ZENODO_REFERER:-https://zenodo.org/records/4060432/}}"
VERIFY_CACHE="${CLI_VERIFY_CACHE:-${VERIFY_CACHE:-1}}"
VERIFY_CACHE_FILE="${CLI_VERIFY_CACHE_FILE:-${VERIFY_CACHE_FILE:-${DOWNLOAD_DIR}/.verify_cache.tsv}}"

# GitHub authentication (uses gh CLI if available)
USE_GH_AUTH="${CLI_USE_GH_AUTH:-${USE_GH_AUTH:-1}}"  # use gh auth token for GitHub URLs
GH_TOKEN=""

# Optional: use audb to download AIR/OpenAIR (install if missing)
USE_AUDB="${CLI_USE_AUDB:-${USE_AUDB:-1}}"
INSTALL_AUDB="${CLI_INSTALL_AUDB:-${INSTALL_AUDB:-0}}"
AUDB_DIR="${CLI_AUDB_DIR:-${AUDB_DIR:-${EXTRACT_DIR}/audb}}"
AIR_VERSION="${CLI_AIR_VERSION:-${AIR_VERSION:-1.4.2}}"
OPENAIR_VERSION="${CLI_OPENAIR_VERSION:-${OPENAIR_VERSION:-1.0.0}}"

# Dataset download toggles (set explicitly to 0/1, or leave empty to follow PROFILE)
DOWNLOAD_VCTK="${CLI_DOWNLOAD_VCTK:-${DOWNLOAD_VCTK:-}}"
DOWNLOAD_LIBRISPEECH="${CLI_DOWNLOAD_LIBRISPEECH:-${DOWNLOAD_LIBRISPEECH:-}}"
DOWNLOAD_MUSAN="${CLI_DOWNLOAD_MUSAN:-${DOWNLOAD_MUSAN:-}}"
DOWNLOAD_FMA="${CLI_DOWNLOAD_FMA:-${DOWNLOAD_FMA:-}}"
DOWNLOAD_MTG_JAMENDO="${CLI_DOWNLOAD_MTG_JAMENDO:-${DOWNLOAD_MTG_JAMENDO:-}}"
DOWNLOAD_FSD50K="${CLI_DOWNLOAD_FSD50K:-${DOWNLOAD_FSD50K:-}}"
DOWNLOAD_AIR="${CLI_DOWNLOAD_AIR:-${DOWNLOAD_AIR:-}}"
DOWNLOAD_OPENAIR="${CLI_DOWNLOAD_OPENAIR:-${DOWNLOAD_OPENAIR:-}}"
DOWNLOAD_ACOUSTICROOMS="${CLI_DOWNLOAD_ACOUSTICROOMS:-${DOWNLOAD_ACOUSTICROOMS:-}}"

BACKGROUND_MUSIC_TARGET_COUNT="${CLI_BACKGROUND_MUSIC_TARGET_COUNT:-${BACKGROUND_MUSIC_TARGET_COUNT:-2000}}"
BACKGROUND_MUSIC_MIN_COUNT="${CLI_BACKGROUND_MUSIC_MIN_COUNT:-${BACKGROUND_MUSIC_MIN_COUNT:-500}}"

mkdir -p "${LIST_DIR}"

echo "[config] profile=${PROFILE} download=${DOWNLOAD} data_dir=${DATA_DIR}"
echo "[config] download_dir=${DOWNLOAD_DIR} extract_dir=${EXTRACT_DIR} list_dir=${LIST_DIR}"

# Dataset root paths (override if you already downloaded elsewhere)
VCTK_DIR="${CLI_VCTK_DIR:-${VCTK_DIR:-${EXTRACT_DIR}/VCTK-Corpus-0.92}}"
LIBRISPEECH_DIR="${CLI_LIBRISPEECH_DIR:-${LIBRISPEECH_DIR:-${EXTRACT_DIR}/LibriSpeech}}"
MUSAN_DIR="${CLI_MUSAN_DIR:-${MUSAN_DIR:-${EXTRACT_DIR}/musan}}"
FMA_DIR="${CLI_FMA_DIR:-${FMA_DIR:-${EXTRACT_DIR}/FMA}}"
MTG_JAMENDO_DIR="${CLI_MTG_JAMENDO_DIR:-${MTG_JAMENDO_DIR:-${EXTRACT_DIR}/mtg-jamendo}}"
FSD50K_DIR="${CLI_FSD50K_DIR:-${FSD50K_DIR:-${EXTRACT_DIR}/FSD50K}}"
# AIR/OpenAIR via audb: AIR goes to data/, OpenAIR goes to wav/
AIR_RIR_DIR="${CLI_AIR_RIR_DIR:-${AIR_RIR_DIR:-${AUDB_DIR}/data}}"
OPENAIR_DIR="${CLI_OPENAIR_DIR:-${OPENAIR_DIR:-${AUDB_DIR}/wav}}"
ACOUSTICROOMS_DIR="${CLI_ACOUSTICROOMS_DIR:-${ACOUSTICROOMS_DIR:-${EXTRACT_DIR}/AcousticRooms}}"

# Source overrides
VCTK_URL="${CLI_VCTK_URL:-${VCTK_URL:-https://zenodo.org/records/10691876/files/VCTK-Corpus-0.92.zip?download=1}}"
LIBRISPEECH_PARTS="${CLI_LIBRISPEECH_PARTS:-${LIBRISPEECH_PARTS:-}}"
FMA_SUBSET="${CLI_FMA_SUBSET:-${FMA_SUBSET:-}}"
FSD50K_BASE_URL="${CLI_FSD50K_BASE_URL:-${FSD50K_BASE_URL:-https://zenodo.org/records/4060432/files}}"

MTG_JAMENDO_DATASET="${MTG_JAMENDO_DATASET:-raw_30s}"
MTG_JAMENDO_TYPE="${MTG_JAMENDO_TYPE:-audio-low}"

ARIA2_PARALLEL_ACTIVE=0
ARIA2_INPUT_FILE="${DOWNLOAD_DIR}/aria2-input.txt"
EXTRACT_QUEUE_FILE="${DOWNLOAD_DIR}/extract-queue.txt"
FSD50K_MERGE_QUEUE_FILE="${DOWNLOAD_DIR}/fsd50k-merge.txt"

require_dir() {
  local name="$1"
  local path="$2"
  if [[ -z "${path}" || ! -d "${path}" ]]; then
    echo "[skip] ${name} not found: ${path}" >&2
    return 1
  fi
  return 0
}

detect_fma_audio_dir() {
  local root="$1"
  local candidate
  for candidate in \
    "${root}/fma_full" \
    "${root}/fma_large" \
    "${root}/fma_medium" \
    "${root}/fma_small"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

fma_dataset_ready() {
  local root="$1"
  local audio_dir
  [[ -n "${root}" && -d "${root}" ]] || return 1
  [[ -f "${root}/fma_metadata/tracks.csv" && -f "${root}/fma_metadata/genres.csv" ]] || return 1
  audio_dir="$(detect_fma_audio_dir "${root}" 2>/dev/null || true)"
  [[ -n "${audio_dir}" ]] || return 1
  find "${audio_dir}" -type f -name '*.mp3' -print -quit | grep -q .
}

detect_mtg_jamendo_audio_dir() {
  local root="$1"
  local candidate
  for candidate in \
    "${root}/raw_30s/audio-low" \
    "${root}/raw_30s/audio" \
    "${root}/audio-low" \
    "${root}/audio" \
    "${root}"; do
    if [[ -d "${candidate}" ]] && \
      find "${candidate}" -maxdepth 3 -type f \( -name '*.mp3' -o -name '*.low.mp3' \) -print -quit | grep -q .; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

mtg_jamendo_dataset_ready() {
  local root="$1"
  [[ -n "${root}" && -d "${root}" ]] || return 1
  [[ -f "${root}/data/raw.meta.tsv" ]] || return 1
  [[ -f "${root}/data/autotagging.tsv" || -f "${root}/data/raw_30s_cleantags_50artists.tsv" ]] || return 1
  detect_mtg_jamendo_audio_dir "${root}" >/dev/null 2>&1
}

write_list() {
  local src_dir="$1"
  local out_file="$2"
  local pattern="$3"
  local tmp_file="${out_file}.tmp.$$"
  find "${src_dir}" -type f \( -iname "${pattern}" \) | sort > "${tmp_file}"
  mv "${tmp_file}" "${out_file}"
  echo "[ok] wrote $(wc -l < "${out_file}") entries -> ${out_file}"
}

write_atomic_file() {
  local out_file="$1"
  local tmp_file="${out_file}.tmp.$$"
  cat > "${tmp_file}"
  mv "${tmp_file}" "${out_file}"
}

processing_marker_path() {
  local dest="$1"
  local artifact_name="$2"
  echo "${dest}/.${artifact_name}.complete"
}

write_processing_marker() {
  local marker_path="$1"
  local tmp_path="${marker_path}.tmp.$$"
  printf "complete\n" > "${tmp_path}"
  mv "${tmp_path}" "${marker_path}"
}

archive_processing_marker() {
  local archive="$1"
  local dest="$2"
  processing_marker_path "${dest}" "$(basename "${archive}")"
}

list_archive_roots() {
  local archive="$1"
  case "${archive}" in
    *.zip)
      unzip -Z1 "${archive}" | awk -F'/' 'NF {print $1}' | sed '/^$/d' | sort -u
      ;;
    *.tar.gz|*.tgz)
      tar -tzf "${archive}" | awk -F'/' 'NF {print $1}' | sed '/^$/d' | sort -u
      ;;
    *)
      return 1
      ;;
  esac
}

archive_outputs_exist() {
  local archive="$1"
  local dest="$2"
  local found=1
  while IFS= read -r root; do
    if [[ -z "${root}" ]]; then
      continue
    fi
    found=0
    if [[ ! -e "${dest}/${root}" ]]; then
      return 1
    fi
  done < <(list_archive_roots "${archive}")
  [[ ${found} -eq 0 ]]
}

sync_tree_into_dest() {
  local src_dir="$1"
  local dest_dir="$2"
  mkdir -p "${dest_dir}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${src_dir}/" "${dest_dir}/"
  elif command -v ditto >/dev/null 2>&1; then
    ditto "${src_dir}" "${dest_dir}"
  else
    cp -R "${src_dir}/." "${dest_dir}/"
  fi
}

stat_size() {
  local path="$1"
  if stat -f %z "${path}" >/dev/null 2>&1; then
    stat -f %z "${path}"
  else
    stat -c %s "${path}"
  fi
}

remove_zero_byte_download_artifacts() {
  local path="$1"
  if [[ -f "${path}" ]] && [[ "$(stat_size "${path}")" -eq 0 ]]; then
    rm -f "${path}"
  fi
  if [[ -f "${path}.aria2" ]]; then
    rm -f "${path}.aria2"
  fi
}

stat_mtime() {
  local path="$1"
  if stat -f %m "${path}" >/dev/null 2>&1; then
    stat -f %m "${path}"
  else
    stat -c %Y "${path}"
  fi
}

checksum_file() {
  local path="$1"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${path}" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  else
    python3 "${ROOT_DIR}/scripts/datasets/sha256sum.py" "${path}"
  fi
}

supports_range_requests() {
  local url="$1"
  local headers
  headers=$(curl -sI -L --max-time 10 "${url}" 2>/dev/null | tr -d '\r' | grep -Ei '^Accept-Ranges:' || true)
  if [[ "${headers}" =~ [Bb]ytes ]]; then
    return 0
  fi
  # Zenodo's HEAD responses do not always advertise range support even though
  # ranged GET requests succeed. Probe a single byte so the downloader can use
  # multipart range downloads when the mirror actually supports them.
  if [[ "${url}" == *"zenodo.org"* ]]; then
    local probe
    probe=$(curl -L -r 0-0 -o /dev/null -D - -s --max-time 15 "${url}" 2>/dev/null | tr -d '\r')
    if printf '%s\n' "${probe}" | grep -Eq '^HTTP/[0-9.]+ 206'; then
      return 0
    fi
    if printf '%s\n' "${probe}" | grep -Eiq '^Content-Range:[[:space:]]*bytes[[:space:]]+0-0/'; then
      return 0
    fi
    return 1
  fi
  return 1
}

content_length_for_url() {
  local url="$1"
  curl -sI -L --max-time 15 "${url}" 2>/dev/null | tr -d '\r' | awk '
    tolower($1) == "content-length:" {
      print $2
      exit
    }
  '
}

should_use_parallel_curl_download() {
  local url="$1"
  local out="$2"
  [[ "$(basename "${out}")" == "VCTK-Corpus-0.92.zip" && "${url}" == *"zenodo.org"* ]] || return 1
  command -v curl >/dev/null 2>&1 || return 1
  supports_range_requests "${url}"
}

download_file_parallel_curl_ranges() {
  local url="$1"
  local out="$2"
  local total_size
  total_size="$(content_length_for_url "${url}")"
  if [[ ! "${total_size}" =~ ^[0-9]+$ ]] || [[ "${total_size}" -le 0 ]]; then
    echo "[warn] could not determine content length for parallel curl download, falling back: ${url}" >&2
    curl -L --fail -H "Referer: ${ZENODO_REFERER}" -o "${out}" "${url}"
    return
  fi

  local parts="${ARIA2_CONN}"
  if [[ "${ARIA2_SPLIT}" -lt "${parts}" ]]; then
    parts="${ARIA2_SPLIT}"
  fi
  if [[ "${parts}" -lt 2 ]]; then
    curl -L --fail -H "Referer: ${ZENODO_REFERER}" -o "${out}" "${url}"
    return
  fi

  local part_dir="${out}.parts"
  local tmp_out="${out}.tmp"
  local chunk_size=$(( (total_size + parts - 1) / parts ))
  local -a pids=()
  local part_idx start end expected_size part_file

  if [[ "${RESUME}" != "1" ]]; then
    rm -rf "${part_dir}"
    rm -f "${tmp_out}" "${out}"
  fi
  mkdir -p "${part_dir}"

  for ((part_idx = 0; part_idx < parts; part_idx++)); do
    start=$(( part_idx * chunk_size ))
    if [[ "${start}" -ge "${total_size}" ]]; then
      break
    fi
    end=$(( start + chunk_size - 1 ))
    if [[ "${end}" -ge "${total_size}" ]]; then
      end=$(( total_size - 1 ))
    fi
    expected_size=$(( end - start + 1 ))
    part_file="${part_dir}/part.$(printf '%03d' "${part_idx}")"

    if [[ "${RESUME}" == "1" && -f "${part_file}" ]] && [[ "$(stat_size "${part_file}")" -eq "${expected_size}" ]]; then
      continue
    fi

    rm -f "${part_file}"
    curl -L --fail --silent --show-error \
      --retry 5 --retry-delay 2 \
      -H "Referer: ${ZENODO_REFERER}" \
      --range "${start}-${end}" \
      -o "${part_file}" \
      "${url}" &
    pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[error] parallel curl range download failed: ${url}" >&2
    return 1
  fi

  rm -f "${tmp_out}"
  for ((part_idx = 0; part_idx < parts; part_idx++)); do
    part_file="${part_dir}/part.$(printf '%03d' "${part_idx}")"
    if [[ ! -f "${part_file}" ]]; then
      continue
    fi
    cat "${part_file}" >> "${tmp_out}"
  done

  if [[ "$(stat_size "${tmp_out}")" -ne "${total_size}" ]]; then
    echo "[error] merged parallel download has wrong size: ${tmp_out}" >&2
    return 1
  fi

  mv "${tmp_out}" "${out}"
  rm -rf "${part_dir}"
}

get_gh_token() {
  if [[ -n "${GH_TOKEN}" ]]; then
    echo "${GH_TOKEN}"
    return 0
  fi
  if [[ "${USE_GH_AUTH}" != "1" ]]; then
    return 1
  fi
  if command -v gh >/dev/null 2>&1; then
    local token
    token=$(gh auth token 2>/dev/null || true)
    if [[ -n "${token}" ]]; then
      GH_TOKEN="${token}"
      echo "${token}"
      return 0
    fi
  fi
  return 1
}

is_github_url() {
  local url="$1"
  [[ "${url}" == *"github.com"* || "${url}" == *"githubusercontent.com"* || "${url}" == *"raw.githubusercontent.com"* ]]
}

is_fsd50k_url() {
  local url="$1"
  [[ "${url}" == *"FSD50K."* ]]
}

cache_lookup() {
  local path="$1"
  local size
  size="$(stat_size "${path}")"
  if [[ ! -f "${VERIFY_CACHE_FILE}" ]]; then
    return 1
  fi
  # Match on path and size only; mtime is stored but not used for lookup
  # so that touching a file doesn't invalidate the cache
  awk -F'\t' -v p="${path}" -v s="${size}" \
    '$1==p && $2==s {found=1} END {exit(found?0:1)}' \
    "${VERIFY_CACHE_FILE}" >/dev/null 2>&1
}

cache_store() {
  local path="$1"
  local size mtime checksum tmp
  size="$(stat_size "${path}")"
  mtime="$(stat_mtime "${path}")"
  checksum="$(checksum_file "${path}")"
  mkdir -p "$(dirname "${VERIFY_CACHE_FILE}")"
  tmp="${VERIFY_CACHE_FILE}.tmp"
  if [[ -f "${VERIFY_CACHE_FILE}" ]]; then
    awk -F'\t' -v p="${path}" '$1!=p' "${VERIFY_CACHE_FILE}" > "${tmp}" || true
  else
    : > "${tmp}"
  fi
  printf "%s\t%s\t%s\t%s\n" "${path}" "${size}" "${mtime}" "${checksum}" >> "${tmp}"
  mv "${tmp}" "${VERIFY_CACHE_FILE}"
}

should_download() {
  local flag="$1"
  if [[ "${flag}" == "1" ]]; then
    return 0
  fi
  if [[ "${flag}" == "0" ]]; then
    return 1
  fi
  # Default by profile
  case "${PROFILE}" in
    production)
      return 0
      ;;
    apple|prototype|*)
      return 0
      ;;
  esac
}

init_parallel_downloads() {
  ARIA2_PARALLEL_ACTIVE=0
  if [[ "${DOWNLOAD}" != "1" ]]; then
    return 0
  fi
  if [[ "${USE_ARIA2}" != "1" || "${ARIA2_PARALLEL}" != "1" ]]; then
    return 0
  fi
  if ! command -v aria2c >/dev/null 2>&1; then
    return 0
  fi
  mkdir -p "${DOWNLOAD_DIR}"
  : > "${ARIA2_INPUT_FILE}"
  : > "${EXTRACT_QUEUE_FILE}"
  : > "${FSD50K_MERGE_QUEUE_FILE}"
  ARIA2_PARALLEL_ACTIVE=1
}

need_cmd() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "Missing required command: ${name}" >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local out="$2"
  local force_curl="${3:-0}"
  local aria2_conn="${ARIA2_CONN}"
  local aria2_split="${ARIA2_SPLIT}"
  local aria2_file_alloc="${ARIA2_FILE_ALLOC:-prealloc}"
  local aria2_continue="1"
  if is_fsd50k_url "${url}"; then
    force_curl=1
  fi
  if [[ -f "${out}" ]] && [[ "$(stat_size "${out}")" -eq 0 ]]; then
    echo "[warn] removing zero-byte placeholder before retry: ${out}" >&2
    remove_zero_byte_download_artifacts "${out}"
  fi
  if [[ -f "${out}" ]]; then
    if [[ "${RESUME}" != "1" ]]; then
      echo "[skip] exists: ${out}"
      return 0
    fi
    if [[ -f "${out}.aria2" ]]; then
      echo "[resume] found aria2 metadata: ${out}.aria2"
    else
      if verify_archive "${out}"; then
        echo "[skip] exists and verified: ${out}"
        return 0
      fi
      echo "[warn] existing file failed verification, attempting resume: ${out}"
    fi
  fi
  if should_use_parallel_curl_download "${url}" "${out}"; then
    download_file_parallel_curl_ranges "${url}" "${out}"
    return $?
  fi
  if [[ "${force_curl}" != "1" && "${USE_ARIA2}" == "1" ]] && command -v aria2c >/dev/null 2>&1; then
    # Some hosts do not handle multi-range requests well (hardcoded fallbacks).
    if [[ "${url}" == *"datashare.ed.ac.uk"* || "${url}" == *"datashare.is.ed.ac.uk"* ]]; then
      aria2_conn=1
      aria2_split=1
      aria2_file_alloc="none"
      aria2_continue="0"
      rm -f "${out}.aria2"
      if [[ -f "${out}" ]]; then
        rm -f "${out}"
      fi
    elif ! supports_range_requests "${url}"; then
      # Server doesn't advertise range support; use single connection
      echo "[info] server does not support range requests, using single connection: ${url}" >&2
      aria2_conn=1
      aria2_split=1
    fi
    # Build aria2c auth header for GitHub URLs
    # NOTE: Token is passed via command line, which is visible via `ps` on multi-user systems.
    # For shared environments, consider setting GH_TOKEN via a secrets manager or using
    # `gh release download` directly for GitHub releases.
    local aria2_auth_header=""
    local gh_token
    if is_github_url "${url}" && gh_token=$(get_gh_token); then
      aria2_auth_header="--header=Authorization: token ${gh_token}"
      echo "[info] using GitHub authentication for: ${url}" >&2
    fi
    local status=0
    if [[ "${aria2_continue}" == "1" ]]; then
      # shellcheck disable=SC2086  # Intentionally unquoted to allow empty expansion
      aria2c -x "${aria2_conn}" -s "${aria2_split}" -k "${ARIA2_MIN_SPLIT}" -c \
        --check-integrity=true \
        --file-allocation="${aria2_file_alloc}" \
        --user-agent="${ARIA2_USER_AGENT}" \
        ${aria2_auth_header} \
        -d "$(dirname "${out}")" -o "$(basename "${out}")" "${url}" </dev/null || status=$?
    else
      # shellcheck disable=SC2086  # Intentionally unquoted to allow empty expansion
      aria2c -x "${aria2_conn}" -s "${aria2_split}" -k "${ARIA2_MIN_SPLIT}" \
        --check-integrity=true \
        --file-allocation="${aria2_file_alloc}" \
        --user-agent="${ARIA2_USER_AGENT}" \
        ${aria2_auth_header} \
        -d "$(dirname "${out}")" -o "$(basename "${out}")" "${url}" </dev/null || status=$?
    fi
    if [[ "${status}" -ne 0 ]]; then
      remove_zero_byte_download_artifacts "${out}"
      if [[ "${INTERRUPTED}" == "1" || "${status}" -eq 130 ]]; then
        echo "[info] download interrupted by user" >&2
        exit 130
      fi
      echo "[warn] aria2c failed (exit ${status}), falling back to curl/wget: ${url}" >&2
      rm -f "${out}.aria2"
      USE_ARIA2=0
    fi
  fi
  if [[ "${force_curl}" == "1" || "${USE_ARIA2}" != "1" ]]; then
    # NOTE: Token is passed via command line, which is visible via `ps` on multi-user systems.
    # For shared environments, consider using `gh release download` directly for GitHub releases.
    local auth_header=""
    local gh_token
    if is_github_url "${url}" && gh_token=$(get_gh_token); then
      auth_header="-H 'Authorization: token ${gh_token}'"
      echo "[info] using GitHub authentication for: ${url}" >&2
    fi
    if command -v curl >/dev/null 2>&1; then
      if [[ "${RESUME}" == "1" ]]; then
        if [[ -n "${auth_header}" ]]; then
          if ! curl -L -C - --fail -H "Authorization: token ${gh_token}" -o "${out}" "${url}"; then
            remove_zero_byte_download_artifacts "${out}"
            echo "[warn] curl resume failed, retrying full download: ${url}" >&2
            rm -f "${out}"
            if ! curl -L --fail -H "Authorization: token ${gh_token}" -o "${out}" "${url}"; then
              remove_zero_byte_download_artifacts "${out}"
              return 1
            fi
          fi
        else
          if ! curl -L -C - --fail -o "${out}" "${url}"; then
            remove_zero_byte_download_artifacts "${out}"
            echo "[warn] curl resume failed, retrying full download: ${url}" >&2
            rm -f "${out}"
            if ! curl -L --fail -o "${out}" "${url}"; then
              remove_zero_byte_download_artifacts "${out}"
              return 1
            fi
          fi
        fi
      else
        if [[ -n "${auth_header}" ]]; then
          if ! curl -L --fail -H "Authorization: token ${gh_token}" -o "${out}" "${url}"; then
            remove_zero_byte_download_artifacts "${out}"
            return 1
          fi
        else
          if ! curl -L --fail -o "${out}" "${url}"; then
            remove_zero_byte_download_artifacts "${out}"
            return 1
          fi
        fi
      fi
    elif [[ "${force_curl}" == "1" ]]; then
      echo "Need curl to download FSD50K files: ${url}" >&2
      exit 1
    elif command -v wget >/dev/null 2>&1; then
      if [[ "${RESUME}" == "1" ]]; then
        if [[ -n "${auth_header}" ]]; then
          if ! wget -c --header="Authorization: token ${gh_token}" -O "${out}" "${url}"; then
            remove_zero_byte_download_artifacts "${out}"
            echo "[warn] wget resume failed, retrying full download: ${url}" >&2
            rm -f "${out}"
            if ! wget --header="Authorization: token ${gh_token}" -O "${out}" "${url}"; then
              remove_zero_byte_download_artifacts "${out}"
              return 1
            fi
          fi
        else
          if ! wget -c -O "${out}" "${url}"; then
            remove_zero_byte_download_artifacts "${out}"
            echo "[warn] wget resume failed, retrying full download: ${url}" >&2
            rm -f "${out}"
            if ! wget -O "${out}" "${url}"; then
              remove_zero_byte_download_artifacts "${out}"
              return 1
            fi
          fi
        fi
      else
        if [[ -n "${auth_header}" ]]; then
          if ! wget --header="Authorization: token ${gh_token}" -O "${out}" "${url}"; then
            remove_zero_byte_download_artifacts "${out}"
            return 1
          fi
        else
          if ! wget -O "${out}" "${url}"; then
            remove_zero_byte_download_artifacts "${out}"
            return 1
          fi
        fi
      fi
    else
      echo "Need curl or wget to download files." >&2
      exit 1
    fi
  fi
}

extract_archive() {
  local archive="$1"
  local dest="$2"
  local marker_path
  local stage_dir
  marker_path="$(archive_processing_marker "${archive}" "${dest}")"
  if [[ -f "${marker_path}" ]]; then
    echo "[skip] already processed archive: ${archive}" >&2
    return 0
  fi
  if archive_outputs_exist "${archive}" "${dest}"; then
    echo "[skip] processed outputs already exist for $(basename "${archive}")" >&2
    write_processing_marker "${marker_path}"
    return 0
  fi

  stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/$(basename "${dest}").$(basename "${archive}").XXXXXX")"
  mkdir -p "${dest}"
  case "${archive}" in
    *.tar.gz|*.tgz)
      tar -xzf "${archive}" -C "${stage_dir}"
      ;;
    *.zip)
      # Use Python's zipfile directly — macOS's ancient unzip (2009) chokes
      # on bzip2-compressed zips (exit 81) and has other limitations.
      python3 -c "
import zipfile, sys, os
z = zipfile.ZipFile(sys.argv[1])
dest = sys.argv[2]
for info in z.infolist():
    out = os.path.join(dest, info.filename)
    if not os.path.exists(out):
        z.extract(info, dest)
" "${archive}" "${stage_dir}"
      ;;
    *)
      echo "Unknown archive format: ${archive}" >&2
      rm -rf "${stage_dir}"
      exit 1
      ;;
  esac
  sync_tree_into_dest "${stage_dir}" "${dest}"
  rm -rf "${stage_dir}"
  write_processing_marker "${marker_path}"
}

verify_archive() {
  local archive="$1"
  local status=0
  if [[ "${VERIFY_CACHE}" == "1" ]]; then
    if cache_lookup "${archive}"; then
      return 0
    fi
  fi
  case "${archive}" in
    *.zip)
      command -v python3 >/dev/null 2>&1 || return 0
      # Check if this is a split zip (has .z01 sibling) - skip verification
      # since the parts are verified individually and the main .zip alone
      # cannot be verified without the parts present
      local base="${archive%.zip}"
      if [[ -f "${base}.z01" ]]; then
        # Split zip: trust that parts are valid, cache based on size only
        if [[ "${VERIFY_CACHE}" == "1" ]]; then
          cache_store "${archive}"
        fi
        return 0
      fi
      # Use Python's zipfile directly — macOS's ancient unzip (2009) chokes
      # on bzip2-compressed zips (exit 81) and has other limitations.
      python3 -c "
import zipfile, sys
z = zipfile.ZipFile(sys.argv[1])
bad = z.testzip()
sys.exit(1 if bad else 0)
" "${archive}" >/dev/null 2>&1
      status=$?
      if [[ ${status} -eq 0 && "${VERIFY_CACHE}" == "1" ]]; then
        cache_store "${archive}"
      fi
      return ${status}
      ;;
    *.tar.gz|*.tgz)
      command -v tar >/dev/null 2>&1 || return 0
      tar -tzf "${archive}" >/dev/null 2>&1
      status=$?
      if [[ ${status} -eq 0 && "${VERIFY_CACHE}" == "1" ]]; then
        cache_store "${archive}"
      fi
      return ${status}
      ;;
    *)
      return 0
      ;;
  esac
}

extract_if_present() {
  local name="$1"
  local archive="$2"
  local dest="$3"
  if [[ -f "${archive}" ]]; then
    echo "[info] extracting ${name} from ${archive}"
    extract_archive "${archive}" "${dest}"
    return 0
  fi
  return 1
}

queue_download() {
  local url="$1"
  local out="$2"
  local aria2_conn="${ARIA2_CONN}"
  local aria2_split="${ARIA2_SPLIT}"
  local aria2_file_alloc="${ARIA2_FILE_ALLOC}"
  local out_dir
  out_dir="$(dirname "${out}")"
  local aria2_continue="true"
  local aria2_allow_overwrite="false"
  local aria2_retry_wait=""
  local aria2_max_tries=""
  local aria2_user_agent="${ARIA2_USER_AGENT}"

  if [[ -f "${out}" ]] && [[ "$(stat_size "${out}")" -eq 0 ]]; then
    echo "[warn] removing zero-byte placeholder before aria2 retry: ${out}" >&2
    remove_zero_byte_download_artifacts "${out}"
  fi

  if [[ -f "${out}" ]]; then
    if [[ "${RESUME}" != "1" ]]; then
      echo "[skip] exists: ${out}"
      return 0
    fi
    if [[ -f "${out}.aria2" ]]; then
      echo "[resume] found aria2 metadata: ${out}.aria2"
    else
      if verify_archive "${out}"; then
        echo "[skip] exists and verified: ${out}"
        return 0
      fi
      echo "[warn] existing file failed verification, attempting resume: ${out}"
    fi
  fi

  if [[ "${url}" == *"datashare.ed.ac.uk"* || "${url}" == *"datashare.is.ed.ac.uk"* ]]; then
    aria2_conn=1
    aria2_split=1
    aria2_file_alloc="none"
    aria2_continue="false"
    aria2_allow_overwrite="true"
    if [[ -f "${out}.aria2" ]]; then
      rm -f "${out}.aria2"
    fi
    if [[ -f "${out}" && ! -f "${out}.aria2" ]]; then
      rm -f "${out}"
    fi
  elif ! supports_range_requests "${url}"; then
    # Server doesn't advertise range support; use single connection
    echo "[info] server does not support range requests, using single connection: ${url}" >&2
    aria2_conn=1
    aria2_split=1
  elif [[ "${url}" == *"zenodo.org"* ]]; then
    aria2_retry_wait="10"
    aria2_max_tries="10"
  fi

  mkdir -p "${out_dir}"
  # Get GitHub auth token if applicable
  # NOTE: Token is written to aria2 input file. Ensure this file is not world-readable
  # on shared systems. The file is created in DOWNLOAD_DIR with default permissions.
  local gh_token
  local use_gh_auth=""
  if is_github_url "${url}" && gh_token=$(get_gh_token); then
    use_gh_auth="1"
    echo "[info] queuing with GitHub authentication: ${url}" >&2
  fi
  {
    echo "${url}"
    echo "  dir=${out_dir}"
    echo "  out=$(basename "${out}")"
    echo "  split=${aria2_split}"
    echo "  max-connection-per-server=${aria2_conn}"
    echo "  min-split-size=${ARIA2_MIN_SPLIT}"
    echo "  file-allocation=${aria2_file_alloc}"
    echo "  user-agent=${aria2_user_agent}"
    if [[ -n "${use_gh_auth}" ]]; then
      echo "  header=Authorization: token ${gh_token}"
    fi
    if [[ "${url}" == *"zenodo.org"* ]]; then
      echo "  header=Referer: ${ZENODO_REFERER}"
    fi
    if [[ "${aria2_continue}" == "true" && "${RESUME}" == "1" ]]; then
      echo "  continue=true"
    else
      echo "  continue=false"
    fi
    if [[ "${aria2_allow_overwrite}" == "true" ]]; then
      echo "  allow-overwrite=true"
    fi
    if [[ -n "${aria2_retry_wait}" ]]; then
      echo "  retry-wait=${aria2_retry_wait}"
    fi
    if [[ -n "${aria2_max_tries}" ]]; then
      echo "  max-tries=${aria2_max_tries}"
    fi
  } >> "${ARIA2_INPUT_FILE}"
}

queue_extract() {
  local archive="$1"
  local dest="$2"
  local url="$3"
  echo "${archive}|${dest}|${url}" >> "${EXTRACT_QUEUE_FILE}"
}

run_parallel_downloads() {
  if [[ "${ARIA2_PARALLEL_ACTIVE}" != "1" ]]; then
    return 0
  fi
  if [[ ! -s "${ARIA2_INPUT_FILE}" ]]; then
    return 0
  fi
  local status=0
  aria2c -i "${ARIA2_INPUT_FILE}" -c --check-integrity=true \
    -j "${ARIA2_MAX_CONCURRENT}" -x "${ARIA2_CONN}" -s "${ARIA2_SPLIT}" \
    -k "${ARIA2_MIN_SPLIT}" --user-agent="${ARIA2_USER_AGENT}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    if [[ -f "${ARIA2_INPUT_FILE}" ]]; then
      awk '
        /^  dir=/ {dir=substr($0, 7)}
        /^  out=/ {
          out=substr($0, 7)
          if (dir != "" && out != "") {
            print dir "/" out
          }
        }
      ' "${ARIA2_INPUT_FILE}" | while IFS= read -r path; do
        if [[ -n "${path}" ]]; then
          remove_zero_byte_download_artifacts "${path}"
        fi
      done
    fi
    if [[ "${INTERRUPTED}" == "1" || "${status}" -eq 130 ]]; then
      echo "[info] download interrupted by user" >&2
      exit 130
    fi
    echo "[error] aria2c failed (exit ${status})" >&2
    exit "${status}"
  fi
}

process_extract_queue() {
  if [[ ! -s "${EXTRACT_QUEUE_FILE}" ]]; then
    return 0
  fi
  # Read from fd 3 so child commands (aria2c, unzip, tar) cannot consume
  # the queue via inherited stdin — a classic bash while-read pitfall.
  while IFS='|' read -r archive dest url <&3; do
    if [[ -z "${archive}" ]]; then
      continue
    fi
    if ! verify_archive "${archive}"; then
      echo "[warn] archive failed verification, retrying: ${archive}" >&2
      rm -f "${archive}"
      download_file "${url}" "${archive}"
      verify_archive "${archive}"
    fi
    extract_archive "${archive}" "${dest}"
    if [[ "${KEEP_ARCHIVES}" == "0" ]]; then
      rm -f "${archive}"
    fi
  done 3< "${EXTRACT_QUEUE_FILE}"
}

process_fsd50k_merge_queue() {
  if [[ ! -s "${FSD50K_MERGE_QUEUE_FILE}" ]]; then
    return 0
  fi
  # Read from fd 3 to prevent child commands from consuming the queue via stdin.
  while IFS='|' read -r prefix out_dir <&3; do
    if [[ -z "${prefix}" ]]; then
      continue
    fi
    fsd50k_merge_and_unzip "${prefix}" "${out_dir}"
  done 3< "${FSD50K_MERGE_QUEUE_FILE}"
}

download_and_extract() {
  local url="$1"
  local dest="$2"
  local force_curl=0
  local filename
  filename="$(basename "${url}")"
  filename="${filename%%\?*}"
  if is_fsd50k_url "${url}"; then
    force_curl=1
  fi
  mkdir -p "${DOWNLOAD_DIR}"
  if [[ "${ARIA2_PARALLEL_ACTIVE}" == "1" && "${force_curl}" != "1" ]]; then
    if should_use_parallel_curl_download "${url}" "${DOWNLOAD_DIR}/${filename}"; then
      download_file "${url}" "${DOWNLOAD_DIR}/${filename}" "${force_curl}"
      if ! verify_archive "${DOWNLOAD_DIR}/${filename}"; then
        echo "[warn] archive failed verification, retrying: ${filename}" >&2
        rm -f "${DOWNLOAD_DIR:?}/${filename}"
        download_file "${url}" "${DOWNLOAD_DIR}/${filename}" "${force_curl}"
        verify_archive "${DOWNLOAD_DIR}/${filename}"
      fi
      extract_archive "${DOWNLOAD_DIR}/${filename}" "${dest}"
      if [[ "${KEEP_ARCHIVES}" == "0" ]]; then
        rm -f "${DOWNLOAD_DIR:?}/${filename}"
      fi
      return 0
    fi
    local archive_path="${DOWNLOAD_DIR}/${filename}"
    queue_download "${url}" "${archive_path}"
    queue_extract "${archive_path}" "${dest}" "${url}"
    return 0
  fi
  download_file "${url}" "${DOWNLOAD_DIR}/${filename}" "${force_curl}"
  if ! verify_archive "${DOWNLOAD_DIR}/${filename}"; then
    echo "[warn] archive failed verification, retrying: ${filename}" >&2
    rm -f "${DOWNLOAD_DIR:?}/${filename}"
    download_file "${url}" "${DOWNLOAD_DIR}/${filename}" "${force_curl}"
    verify_archive "${DOWNLOAD_DIR}/${filename}"
  fi
  extract_archive "${DOWNLOAD_DIR}/${filename}" "${dest}"
  if [[ "${KEEP_ARCHIVES}" == "0" ]]; then
    rm -f "${DOWNLOAD_DIR:?}/${filename}"
  fi
}

fsd50k_merge_and_unzip() {
  local zip_base="$1"
  local out_dir="$2"
  local merged_zip="${DOWNLOAD_DIR}/${zip_base}.merged.zip"
  local unpack_marker
  unpack_marker="$(archive_processing_marker "${merged_zip}" "${out_dir}")"
  if [[ -f "${unpack_marker}" ]]; then
    echo "[skip] already processed merged FSD50K archive: ${zip_base}" >&2
    return 0
  fi
  python3 "${ROOT_DIR}/scripts/datasets/zip_merge_progress.py" \
    --download-dir "${DOWNLOAD_DIR}" \
    --zip-base "${zip_base}"
  extract_archive "${merged_zip}" "${out_dir}"
  if [[ "${KEEP_ARCHIVES}" == "0" ]]; then
    rm -f "${DOWNLOAD_DIR:?}/${zip_base}.merged.zip"
  fi
}

download_fsd50k_split() {
  local prefix="$1"  # FSD50K.dev_audio or FSD50K.eval_audio
  local out_dir="$2"
  local base_url="${FSD50K_BASE_URL}"
  local parts=()
  if [[ "${prefix}" == "FSD50K.dev_audio" ]]; then
    parts=(z01 z02 z03 z04 z05 zip)
  else
    parts=(z01 zip)
  fi
  mkdir -p "${DOWNLOAD_DIR}"
  for part in "${parts[@]}"; do
    local url="${base_url}/${prefix}.${part}"
    if [[ "${url}" == *"zenodo.org"* && "${url}" != *"download="* ]]; then
      url="${url}?download=1"
    fi
    download_file "${url}" "${DOWNLOAD_DIR}/${prefix}.${part}" "1"
  done
  fsd50k_merge_and_unzip "${prefix}.zip" "${out_dir}"
}

maybe_install_audb() {
  if python3 "${ROOT_DIR}/scripts/datasets/check_audb.py" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "${INSTALL_AUDB}" == "1" ]]; then
    if command -v uv >/dev/null 2>&1; then
      uv pip install audb
    else
      python3 -m pip install audb
    fi
  else
    return 1
  fi
}

download_with_audb() {
  python3 "${ROOT_DIR}/scripts/datasets/audb_download.py"
}

# Optional downloads (opt-in)
if [[ "${DOWNLOAD}" == "1" ]]; then
  if [[ "${AGREE_LICENSES}" != "1" ]]; then
    echo "Set AGREE_LICENSES=1 to confirm you accept dataset licenses." >&2
    exit 1
  fi
  init_parallel_downloads
  need_cmd tar
  need_cmd unzip

  mkdir -p "${DOWNLOAD_DIR}" "${EXTRACT_DIR}"

  # Defaults by profile
  if [[ -z "${DOWNLOAD_VCTK}" ]]; then
    DOWNLOAD_VCTK=1
  fi
  if [[ -z "${DOWNLOAD_MUSAN}" ]]; then
    DOWNLOAD_MUSAN=1
  fi
  if [[ -z "${DOWNLOAD_FMA}" ]]; then
    DOWNLOAD_FMA=1
  fi
  if [[ -z "${DOWNLOAD_MTG_JAMENDO}" ]]; then
    DOWNLOAD_MTG_JAMENDO=0
  fi
  if [[ -z "${DOWNLOAD_FSD50K}" ]]; then
    DOWNLOAD_FSD50K=1
  fi
  if [[ -z "${DOWNLOAD_AIR}" ]]; then
    DOWNLOAD_AIR=1
  fi
  if [[ -z "${DOWNLOAD_OPENAIR}" ]]; then
    DOWNLOAD_OPENAIR=1
  fi
  if [[ -z "${DOWNLOAD_LIBRISPEECH}" ]]; then
    case "${PROFILE}" in
      production) DOWNLOAD_LIBRISPEECH=1 ;;
      apple|prototype|*) DOWNLOAD_LIBRISPEECH=0 ;;
    esac
  fi
  if [[ -z "${DOWNLOAD_ACOUSTICROOMS}" ]]; then
    case "${PROFILE}" in
      production) DOWNLOAD_ACOUSTICROOMS=1 ;;
      apple|prototype|*) DOWNLOAD_ACOUSTICROOMS=0 ;;
    esac
  fi
  if [[ -z "${FMA_SUBSET}" ]]; then
    case "${PROFILE}" in
      production) FMA_SUBSET="large" ;;
      apple|prototype|*) FMA_SUBSET="medium" ;;
    esac
  fi

  # VCTK (zip has no top-level dir, so extract into dedicated subdirectory)
  if should_download "${DOWNLOAD_VCTK}"; then
    mkdir -p "${EXTRACT_DIR}/VCTK-Corpus-0.92"
    download_and_extract "${VCTK_URL}" "${EXTRACT_DIR}/VCTK-Corpus-0.92"
  fi

  # LibriSpeech
  if should_download "${DOWNLOAD_LIBRISPEECH}"; then
    if [[ -z "${LIBRISPEECH_PARTS}" ]]; then
      if [[ "${PROFILE}" == "production" ]]; then
        LIBRISPEECH_PARTS="train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other"
      else
        LIBRISPEECH_PARTS="train-clean-100 dev-clean test-clean"
      fi
    fi
    for part in ${LIBRISPEECH_PARTS}; do
      download_and_extract "http://www.openslr.org/resources/12/${part}.tar.gz" "${EXTRACT_DIR}"
    done
  fi

  # MUSAN
  if should_download "${DOWNLOAD_MUSAN}"; then
    download_and_extract "https://www.openslr.org/resources/17/musan.tar.gz" "${EXTRACT_DIR}"
  fi

  # FMA (Creative Commons music corpus; canonical automated background-music base)
  if should_download "${DOWNLOAD_FMA}"; then
    mkdir -p "${FMA_DIR}"
    download_and_extract "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip" "${FMA_DIR}"
    case "${FMA_SUBSET}" in
      medium|large|full)
        download_and_extract "https://os.unil.cloud.switch.ch/fma/fma_${FMA_SUBSET}.zip" "${FMA_DIR}"
        ;;
      *)
        echo "Error: unsupported FMA_SUBSET '${FMA_SUBSET}' (expected medium, large, or full)" >&2
        exit 1
        ;;
    esac
  fi

  # MTG-Jamendo (optional due size/license profile; downloads the official raw_30s audio-low tar set)
  if should_download "${DOWNLOAD_MTG_JAMENDO}"; then
    if [[ "${MTG_JAMENDO_DATASET}" != "raw_30s" || "${MTG_JAMENDO_TYPE}" != "audio-low" ]]; then
      echo "Error: only MTG-Jamendo raw_30s audio-low is currently supported" >&2
      exit 1
    fi
    mkdir -p "${MTG_JAMENDO_DIR}/data" "${DOWNLOAD_DIR}/mtg-jamendo"
    download_file \
      "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw.meta.tsv" \
      "${MTG_JAMENDO_DIR}/data/raw.meta.tsv"
    download_file \
      "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw_30s_cleantags_50artists.tsv" \
      "${MTG_JAMENDO_DIR}/data/raw_30s_cleantags_50artists.tsv"
    download_file \
      "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/download/raw_30s_audio-low_sha256_tars.txt" \
      "${DOWNLOAD_DIR}/mtg-jamendo/raw_30s_audio-low_sha256_tars.txt"

    # Read from fd 3 to prevent child commands (aria2c) from consuming the
    # tar list via inherited stdin.
    while read -r _sha256 tar_name <&3; do
      if [[ -z "${tar_name:-}" ]]; then
        continue
      fi
      download_and_extract \
        "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/${tar_name}" \
        "${MTG_JAMENDO_DIR}"
    done 3< "${DOWNLOAD_DIR}/mtg-jamendo/raw_30s_audio-low_sha256_tars.txt"
  fi

  # FSD50K
  if should_download "${DOWNLOAD_FSD50K}"; then
    need_cmd zip
    mkdir -p "${FSD50K_DIR}"
    # FSD50K downloads are forced through curl (never aria2/wget).
    # Metadata + ground truth + docs
    download_and_extract "${FSD50K_BASE_URL}/FSD50K.metadata.zip?download=1" "${FSD50K_DIR}"
    download_and_extract "${FSD50K_BASE_URL}/FSD50K.ground_truth.zip?download=1" "${FSD50K_DIR}"
    download_and_extract "${FSD50K_BASE_URL}/FSD50K.doc.zip?download=1" "${FSD50K_DIR}"
    # Split audio zips
    download_fsd50k_split "FSD50K.dev_audio" "${FSD50K_DIR}"
    download_fsd50k_split "FSD50K.eval_audio" "${FSD50K_DIR}"
  fi

  # AIR/OpenAIR via audb (optional)
  # audb downloads AIR to data/ and OpenAIR to wav/ within AUDB_DIR
  if [[ "${USE_AUDB}" == "1" ]]; then
    if maybe_install_audb; then
      mkdir -p "${AUDB_DIR}"
      if should_download "${DOWNLOAD_AIR}"; then
        AUDB_NAME="air" AUDB_VERSION="${AIR_VERSION}" AUDB_ROOT="${AUDB_DIR}" download_with_audb
      fi
      if should_download "${DOWNLOAD_OPENAIR}"; then
        AUDB_NAME="openair" AUDB_VERSION="${OPENAIR_VERSION}" AUDB_ROOT="${AUDB_DIR}" download_with_audb
      fi
    else
      echo "[skip] audb not installed; set INSTALL_AUDB=1 to auto-install." >&2
    fi
  fi

  # AcousticRooms
  if should_download "${DOWNLOAD_ACOUSTICROOMS}"; then
    mkdir -p "${ACOUSTICROOMS_DIR}"
    download_and_extract "https://github.com/facebookresearch/AcousticRooms/raw/clean-main/single_channel_ir.zip" "${ACOUSTICROOMS_DIR}"
    download_and_extract "https://github.com/facebookresearch/AcousticRooms/raw/clean-main/metadata.zip" "${ACOUSTICROOMS_DIR}"
    # AcousticRooms has nested zips inside single_channel_ir that need extraction
    if [[ -d "${ACOUSTICROOMS_DIR}/single_channel_ir_1" ]]; then
      nested_marker=""
      nested_marker="$(processing_marker_path "${ACOUSTICROOMS_DIR}/single_channel_ir_1" "nested_acousticrooms_zips")"
      if [[ -f "${nested_marker}" ]]; then
        echo "[skip] nested AcousticRooms zips already extracted" >&2
      else
        echo "[info] extracting nested AcousticRooms zips..."
        for nested_zip in "${ACOUSTICROOMS_DIR}/single_channel_ir_1"/*.zip; do
          if [[ -f "${nested_zip}" ]]; then
            extract_archive "${nested_zip}" "${ACOUSTICROOMS_DIR}/single_channel_ir_1/"
          fi
        done
        write_processing_marker "${nested_marker}"
      fi
    fi
  fi

  if [[ "${ARIA2_PARALLEL_ACTIVE}" == "1" ]]; then
    run_parallel_downloads
    process_fsd50k_merge_queue
    process_extract_queue
  fi
fi

# Ensure extracted datasets if archives already exist
if [[ ! -d "${VCTK_DIR}" ]]; then
  # Check if VCTK was extracted flat into EXTRACT_DIR (no subdirectory)
  if [[ -d "${EXTRACT_DIR}/wav48_silence_trimmed" && -f "${EXTRACT_DIR}/speaker-info.txt" ]]; then
    VCTK_DIR="${EXTRACT_DIR}"
  else
    mkdir -p "${EXTRACT_DIR}/VCTK-Corpus-0.92"
    extract_if_present "VCTK" "${DOWNLOAD_DIR}/VCTK-Corpus-0.92.zip" "${EXTRACT_DIR}/VCTK-Corpus-0.92"
    if [[ -d "${EXTRACT_DIR}/VCTK-Corpus-0.92/wav48_silence_trimmed" ]]; then
      VCTK_DIR="${EXTRACT_DIR}/VCTK-Corpus-0.92"
    elif [[ -d "${EXTRACT_DIR}/VCTK-Corpus" ]]; then
      VCTK_DIR="${EXTRACT_DIR}/VCTK-Corpus"
    fi
  fi
fi

if [[ ! -d "${LIBRISPEECH_DIR}" ]]; then
  shopt -s nullglob
  archives=(
    "${DOWNLOAD_DIR}"/train-*.tar.gz
    "${DOWNLOAD_DIR}"/dev-*.tar.gz
    "${DOWNLOAD_DIR}"/test-*.tar.gz
  )
  if (( ${#archives[@]} > 0 )); then
    echo "[info] extracting LibriSpeech archives from ${DOWNLOAD_DIR}"
    for tgz in "${archives[@]}"; do
      extract_archive "${tgz}" "${EXTRACT_DIR}"
    done
  fi
  shopt -u nullglob
fi

if ! fma_dataset_ready "${FMA_DIR}"; then
  extract_if_present "FMA metadata" "${DOWNLOAD_DIR}/fma_metadata.zip" "${FMA_DIR}"

  fma_archives=()
  for archive_name in fma_full.zip fma_large.zip fma_medium.zip fma_small.zip; do
    if [[ -f "${DOWNLOAD_DIR}/${archive_name}" ]]; then
      fma_archives+=("${DOWNLOAD_DIR}/${archive_name}")
    fi
  done
  if (( ${#fma_archives[@]} > 0 )); then
    echo "[info] extracting FMA archives from ${DOWNLOAD_DIR}"
    for zip in "${fma_archives[@]}"; do
      extract_archive "${zip}" "${FMA_DIR}"
    done
  fi
fi

if ! mtg_jamendo_dataset_ready "${MTG_JAMENDO_DIR}"; then
  shopt -s nullglob
  mtg_archives=("${DOWNLOAD_DIR}"/mtg-jamendo/*.tar)
  if (( ${#mtg_archives[@]} > 0 )); then
    echo "[info] extracting MTG-Jamendo archives from ${DOWNLOAD_DIR}/mtg-jamendo"
    for archive in "${mtg_archives[@]}"; do
      extract_archive "${archive}" "${MTG_JAMENDO_DIR}"
    done
  fi
  shopt -u nullglob
fi

if [[ -d "${FSD50K_DIR}" ]]; then
  # Check for JSON metadata files (the actual format FSD50K uses)
  if [[ ! -f "${FSD50K_DIR}/FSD50K.metadata/dev_clips_info_FSD50K.json" ]]; then
    extract_if_present "FSD50K metadata" "${DOWNLOAD_DIR}/FSD50K.metadata.zip" "${FSD50K_DIR}"
  fi
fi

# Clean speech lists
if require_dir "VCTK" "${VCTK_DIR}"; then
  # VCTK 0.92 uses FLAC format in wav48_silence_trimmed
  write_list "${VCTK_DIR}" "${LIST_DIR}/vctk_clean.txt" "*.flac"
fi
if require_dir "LibriSpeech" "${LIBRISPEECH_DIR}"; then
  write_list "${LIBRISPEECH_DIR}" "${LIST_DIR}/librispeech_clean.txt" "*.flac"
fi

# Noise source lists
if require_dir "MUSAN" "${MUSAN_DIR}"; then
  write_list "${MUSAN_DIR}/noise" "${LIST_DIR}/musan_noise.txt" "*.wav"
fi

# FSD50K filtered list (CC0/CC-BY only)
# Expected inputs:
# - ${FSD50K_DIR}/FSD50K.metadata/ with metadata CSVs
# - ${FSD50K_DIR}/FSD50K.dev_audio/ and/or FSD50K.eval_audio/
# Update CSV/column names as needed after download.
if require_dir "FSD50K" "${FSD50K_DIR}"; then
  export FSD50K_DIR
  export LIST_DIR
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/datasets/fsd50k_filter.py"
fi

FMA_PRESENT=0
if fma_dataset_ready "${FMA_DIR}"; then
  FMA_PRESENT=1
else
  echo "[skip] FMA dataset incomplete or not extracted: ${FMA_DIR}" >&2
fi

MTG_JAMENDO_PRESENT=0
if mtg_jamendo_dataset_ready "${MTG_JAMENDO_DIR}"; then
  MTG_JAMENDO_PRESENT=1
else
  echo "[skip] MTG-Jamendo dataset incomplete or not extracted: ${MTG_JAMENDO_DIR}" >&2
fi

# RIR lists
if require_dir "AIR" "${AIR_RIR_DIR}"; then
  write_list "${AIR_RIR_DIR}" "${LIST_DIR}/air_rir.txt" "*.wav"
fi
if require_dir "OpenAIR" "${OPENAIR_DIR}"; then
  write_list "${OPENAIR_DIR}" "${LIST_DIR}/openair_rir.txt" "*.wav"
fi
if require_dir "AcousticRooms" "${ACOUSTICROOMS_DIR}"; then
  write_list "${ACOUSTICROOMS_DIR}" "${LIST_DIR}/acousticrooms_rir.txt" "*.wav"
fi

# Combine lists for downstream datastore/HDF5 builders based on profile
echo "[info] combining lists for profile=${PROFILE}..."

# Clean speech: VCTK (always) + LibriSpeech (production/apple with DOWNLOAD_LIBRISPEECH=1)
{
  if [[ -f "${LIST_DIR}/vctk_clean.txt" ]]; then
    cat "${LIST_DIR}/vctk_clean.txt"
  fi
  if [[ -f "${LIST_DIR}/librispeech_clean.txt" ]]; then
    cat "${LIST_DIR}/librispeech_clean.txt"
  fi
} > "${LIST_DIR}/clean_all.txt"
{
  if [[ -f "${LIST_DIR}/vctk_clean.txt" ]]; then
    cat "${LIST_DIR}/vctk_clean.txt"
  fi
  if [[ -f "${LIST_DIR}/librispeech_clean.txt" ]]; then
    cat "${LIST_DIR}/librispeech_clean.txt"
  fi
} | write_atomic_file "${LIST_DIR}/clean_all.txt"
echo "[ok] wrote $(wc -l < "${LIST_DIR}/clean_all.txt") entries -> ${LIST_DIR}/clean_all.txt"

# Generic noise only: MUSAN noise + FSD50K filtered
{
  if [[ -f "${LIST_DIR}/musan_noise.txt" ]]; then
    cat "${LIST_DIR}/musan_noise.txt"
  fi
  if [[ -f "${LIST_DIR}/fsd50k_filtered.txt" ]]; then
    cat "${LIST_DIR}/fsd50k_filtered.txt"
  fi
} | write_atomic_file "${LIST_DIR}/noise_all.txt"
echo "[ok] wrote $(wc -l < "${LIST_DIR}/noise_all.txt") entries -> ${LIST_DIR}/noise_all.txt"

# Dedicated background music: chart-style open music curated from FMA plus
# optional MTG-Jamendo inputs. `background_music.txt` is the canonical capped
# curated list; `background_music_expanded.txt` contains the full eligible pool.
if [[ "${FMA_PRESENT}" == "1" || "${MTG_JAMENDO_PRESENT}" == "1" ]]; then
  music_curator_args=(
    "${ROOT_DIR}/scripts/datasets/curate_background_music.py"
    --list-dir "${LIST_DIR}"
    --target-count "${BACKGROUND_MUSIC_TARGET_COUNT}"
    --min-count "${BACKGROUND_MUSIC_MIN_COUNT}"
  )
  if [[ "${FMA_PRESENT}" == "1" ]]; then
    music_curator_args+=(--fma-dir "${FMA_DIR}")
  fi
  if [[ "${MTG_JAMENDO_PRESENT}" == "1" ]]; then
    music_curator_args+=(--mtg-jamendo-dir "${MTG_JAMENDO_DIR}")
  fi
  "${PYTHON_BIN}" "${music_curator_args[@]}"
else
  echo "[warn] no FMA or MTG-Jamendo music corpus present; dedicated background-music lists will be empty" >&2
  : > "${LIST_DIR}/background_music.txt"
  : > "${LIST_DIR}/background_music_expanded.txt"
fi

# Backward-compatible combined noise+music list (deprecated; raw-list and cache
# builders now prefer noise_all.txt + dedicated background_music.txt).
{
  if [[ -f "${LIST_DIR}/noise_all.txt" ]]; then
    cat "${LIST_DIR}/noise_all.txt"
  fi
  if [[ -f "${LIST_DIR}/background_music.txt" ]]; then
    cat "${LIST_DIR}/background_music.txt"
  elif [[ -f "${LIST_DIR}/background_music_expanded.txt" ]]; then
    cat "${LIST_DIR}/background_music_expanded.txt"
  fi
} | write_atomic_file "${LIST_DIR}/noise_music.txt"
echo "[ok] wrote $(wc -l < "${LIST_DIR}/noise_music.txt") entries -> ${LIST_DIR}/noise_music.txt"

# RIR: AIR + OpenAIR + AcousticRooms (production/apple with DOWNLOAD_ACOUSTICROOMS=1)
{
  if [[ -f "${LIST_DIR}/air_rir.txt" ]]; then
    cat "${LIST_DIR}/air_rir.txt"
  fi
  if [[ -f "${LIST_DIR}/openair_rir.txt" ]]; then
    cat "${LIST_DIR}/openair_rir.txt"
  fi
  if [[ -f "${LIST_DIR}/acousticrooms_rir.txt" ]]; then
    cat "${LIST_DIR}/acousticrooms_rir.txt"
  fi
} | write_atomic_file "${LIST_DIR}/rir_all.txt"
echo "[ok] wrote $(wc -l < "${LIST_DIR}/rir_all.txt") entries -> ${LIST_DIR}/rir_all.txt"

# Sanity checks for build_hdf5.sh
errors=0
if [[ ! -s "${LIST_DIR}/clean_all.txt" ]]; then
  echo "[error] clean_all.txt is empty - need at least VCTK" >&2
  errors=1
fi
if [[ ! -s "${LIST_DIR}/noise_music.txt" ]]; then
  echo "[error] noise_music.txt is empty - need generic noise and/or curated background music" >&2
  errors=1
fi
if [[ ! -s "${LIST_DIR}/background_music.txt" ]]; then
  echo "[error] background_music.txt is empty - need FMA and/or MTG-Jamendo chart-style music sources" >&2
  errors=1
fi
if [[ ! -s "${LIST_DIR}/rir_all.txt" ]]; then
  echo "[error] rir_all.txt is empty - need AIR, OpenAIR, or AcousticRooms" >&2
  errors=1
fi
if [[ "${errors}" -eq 1 ]]; then
  echo "[error] one or more combined lists are empty; build_hdf5.sh will fail" >&2
  exit 1
fi

cat <<'MSG'
Done.
Combined lists ready for downstream builders:
  - clean_all.txt (speech)
  - noise_all.txt (generic noise, excludes dedicated music)
  - background_music.txt (canonical curated chart-style background music; target-count capped)
  - background_music_expanded.txt (full eligible chart-style pool from FMA + optional MTG-Jamendo)
  - background_music_fma.txt / background_music_mtg_jamendo.txt (source-specific music subsets)
  - background_music_catalog.tsv (curation audit table with bucket/source scores)
  - noise_music.txt (deprecated compatibility mix of generic noise + curated music)
  - rir_all.txt (room impulse responses)

Next steps:
  MLX datastore (reuse existing raw datasets, preserve more short-speech coverage):
    DATA_DIR=/path/to/data OUTPUT_DIR=/path/to/cache \
      bash scripts/datasets/build_mlx_datastore.sh --profile <profile> --merge-short

  MLX datastore with inline DFN3 clean-speech preprocessing:
    DATA_DIR=${DATA_DIR:-/path/to/data} OUTPUT_DIR=${OUTPUT_DIR:-/path/to/cache} \
      bash scripts/datasets/build_mlx_datastore.sh \
        --profile <profile> \
        --merge-short \
        --preprocess-clean-speech

  HDF5 datastore:
    DATA_DIR=${DATA_DIR:-/path/to/data} PROFILE=${PROFILE:-<profile>} bash scripts/datasets/build_hdf5.sh
MSG
