#!/usr/bin/env bash
set -euo pipefail

INTERRUPTED=0
on_interrupt() {
  INTERRUPTED=1
  echo "[info] download interrupted by user" >&2
  exit 130
}
trap on_interrupt INT

# Dataset prep helper
# - Optional download mode (opt-in) for datasets with direct download URLs.
# - Generates file lists for prepare_data.py.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
LIST_DIR="${LIST_DIR:-${DATA_DIR}/lists}"

# Download controls (opt-in)
PROFILE="${PROFILE:-prototype}"  # prototype | production | apple (apple downloads prototype set)
DOWNLOAD="${DOWNLOAD:-1}"         # set to 1 to enable downloads
AGREE_LICENSES="${AGREE_LICENSES:-1}"  # set to 1 to confirm license acceptance
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${DATA_DIR}/downloads}"
EXTRACT_DIR="${EXTRACT_DIR:-${DATA_DIR}/raw}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-1}"
RESUME="${RESUME:-1}"              # try to resume partial downloads when possible
USE_ARIA2="${USE_ARIA2:-1}"        # use aria2c if available
ARIA2_PARALLEL="${ARIA2_PARALLEL:-1}"  # download multiple URLs concurrently
ARIA2_CONN="${ARIA2_CONN:-8}"      # connections per file
ARIA2_SPLIT="${ARIA2_SPLIT:-8}"    # number of splits
ARIA2_MIN_SPLIT="${ARIA2_MIN_SPLIT:-1M}"
ARIA2_MAX_CONCURRENT="${ARIA2_MAX_CONCURRENT:-6}"
ARIA2_FILE_ALLOC="${ARIA2_FILE_ALLOC:-prealloc}"
ARIA2_USER_AGENT="${ARIA2_USER_AGENT:-Mozilla/5.0}"
ZENODO_REFERER="${ZENODO_REFERER:-https://zenodo.org/records/4060432/}"

# Optional: use audb to download AIR/OpenAIR (install if missing)
USE_AUDB="${USE_AUDB:-1}"
INSTALL_AUDB="${INSTALL_AUDB:-0}"
AUDB_DIR="${AUDB_DIR:-${EXTRACT_DIR}/audb}"

# Dataset download toggles (set explicitly to 0/1, or leave empty to follow PROFILE)
DOWNLOAD_VCTK="${DOWNLOAD_VCTK:-}"
DOWNLOAD_LIBRISPEECH="${DOWNLOAD_LIBRISPEECH:-}"
DOWNLOAD_MUSAN="${DOWNLOAD_MUSAN:-}"
DOWNLOAD_FSD50K="${DOWNLOAD_FSD50K:-}"
DOWNLOAD_AIR="${DOWNLOAD_AIR:-}"
DOWNLOAD_OPENAIR="${DOWNLOAD_OPENAIR:-}"
DOWNLOAD_ACOUSTICROOMS="${DOWNLOAD_ACOUSTICROOMS:-}"

mkdir -p "${LIST_DIR}"

# Dataset root paths (override if you already downloaded elsewhere)
VCTK_DIR="${VCTK_DIR:-${EXTRACT_DIR}/VCTK-Corpus}"
LIBRISPEECH_DIR="${LIBRISPEECH_DIR:-${EXTRACT_DIR}/LibriSpeech}"
MUSAN_DIR="${MUSAN_DIR:-${EXTRACT_DIR}/musan}"
FSD50K_DIR="${FSD50K_DIR:-${EXTRACT_DIR}/FSD50K}"
AIR_RIR_DIR="${AIR_RIR_DIR:-${EXTRACT_DIR}/AIR}"
OPENAIR_DIR="${OPENAIR_DIR:-${EXTRACT_DIR}/OpenAIR}"
ACOUSTICROOMS_DIR="${ACOUSTICROOMS_DIR:-${EXTRACT_DIR}/AcousticRooms}"

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

write_list() {
  local src_dir="$1"
  local out_file="$2"
  local pattern="$3"
  find "${src_dir}" -type f \( -iname "${pattern}" \) | sort > "${out_file}"
  echo "[ok] wrote $(wc -l < "${out_file}") entries -> ${out_file}"
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
  local aria2_conn="${ARIA2_CONN}"
  local aria2_split="${ARIA2_SPLIT}"
  local aria2_file_alloc="${ARIA2_FILE_ALLOC:-prealloc}"
  local aria2_continue="1"
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
  if [[ "${USE_ARIA2}" == "1" ]] && command -v aria2c >/dev/null 2>&1; then
    # Some hosts do not handle multi-range requests well.
    if [[ "${url}" == *"datashare.ed.ac.uk"* || "${url}" == *"datashare.is.ed.ac.uk"* ]]; then
      aria2_conn=1
      aria2_split=1
      aria2_file_alloc="none"
      aria2_continue="0"
      rm -f "${out}.aria2"
      if [[ -f "${out}" ]]; then
        rm -f "${out}"
      fi
    elif [[ "${url}" == *"zenodo.org"* ]]; then
      aria2_conn=1
      aria2_split=1
      aria2_file_alloc="none"
    fi
    if [[ "${aria2_continue}" == "1" ]]; then
      aria2c -x "${aria2_conn}" -s "${aria2_split}" -k "${ARIA2_MIN_SPLIT}" -c \
        --check-integrity=true \
        --file-allocation="${aria2_file_alloc}" \
        --user-agent="${ARIA2_USER_AGENT}" \
        -d "$(dirname "${out}")" -o "$(basename "${out}")" "${url}"
    else
      aria2c -x "${aria2_conn}" -s "${aria2_split}" -k "${ARIA2_MIN_SPLIT}" \
      --check-integrity=true \
      --file-allocation="${aria2_file_alloc}" \
      --user-agent="${ARIA2_USER_AGENT}" \
      -d "$(dirname "${out}")" -o "$(basename "${out}")" "${url}"
    fi
    status=$?
    if [[ "${status}" -ne 0 ]]; then
      if [[ "${INTERRUPTED}" == "1" || "${status}" -eq 130 ]]; then
        echo "[info] download interrupted by user" >&2
        exit 130
      fi
      echo "[warn] aria2c failed (exit ${status}), falling back to curl/wget: ${url}" >&2
      rm -f "${out}.aria2"
      USE_ARIA2=0
    fi
  fi
  if [[ "${USE_ARIA2}" != "1" ]]; then
    if command -v curl >/dev/null 2>&1; then
      if [[ "${RESUME}" == "1" ]]; then
        if ! curl -L -C - --fail -o "${out}" "${url}"; then
          echo "[warn] curl resume failed, retrying full download: ${url}" >&2
          rm -f "${out}"
          curl -L --fail -o "${out}" "${url}"
        fi
      else
        curl -L --fail -o "${out}" "${url}"
      fi
    elif command -v wget >/dev/null 2>&1; then
      if [[ "${RESUME}" == "1" ]]; then
        if ! wget -c -O "${out}" "${url}"; then
          echo "[warn] wget resume failed, retrying full download: ${url}" >&2
          rm -f "${out}"
          wget -O "${out}" "${url}"
        fi
      else
        wget -O "${out}" "${url}"
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
  mkdir -p "${dest}"
  case "${archive}" in
    *.tar.gz|*.tgz)
      tar -xzf "${archive}" -C "${dest}"
      ;;
    *.zip)
      unzip -q "${archive}" -d "${dest}"
      ;;
    *)
      echo "Unknown archive format: ${archive}" >&2
      exit 1
      ;;
  esac
}

verify_archive() {
  local archive="$1"
  case "${archive}" in
    *.zip)
      command -v unzip >/dev/null 2>&1 || return 0
      unzip -tqq "${archive}" >/dev/null 2>&1
      return $?
      ;;
    *.tar.gz|*.tgz)
      command -v tar >/dev/null 2>&1 || return 0
      tar -tzf "${archive}" >/dev/null 2>&1
      return $?
      ;;
    *)
      return 0
      ;;
  esac
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
  elif [[ "${url}" == *"zenodo.org"* ]]; then
    aria2_conn=1
    aria2_split=1
    aria2_file_alloc="none"
    aria2_retry_wait="10"
    aria2_max_tries="10"
  fi

  mkdir -p "${out_dir}"
  {
    echo "${url}"
    echo "  dir=${out_dir}"
    echo "  out=$(basename "${out}")"
    echo "  split=${aria2_split}"
    echo "  max-connection-per-server=${aria2_conn}"
    echo "  min-split-size=${ARIA2_MIN_SPLIT}"
    echo "  file-allocation=${aria2_file_alloc}"
    echo "  user-agent=${aria2_user_agent}"
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
  aria2c -i "${ARIA2_INPUT_FILE}" -c --check-integrity=true \
    -j "${ARIA2_MAX_CONCURRENT}" -x "${ARIA2_CONN}" -s "${ARIA2_SPLIT}" \
    -k "${ARIA2_MIN_SPLIT}" --user-agent="${ARIA2_USER_AGENT}"
  status=$?
  if [[ "${status}" -ne 0 ]]; then
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
  while IFS='|' read -r archive dest url; do
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
  done < "${EXTRACT_QUEUE_FILE}"
}

process_fsd50k_merge_queue() {
  if [[ ! -s "${FSD50K_MERGE_QUEUE_FILE}" ]]; then
    return 0
  fi
  while IFS='|' read -r prefix out_dir; do
    if [[ -z "${prefix}" ]]; then
      continue
    fi
    fsd50k_merge_and_unzip "${prefix}" "${out_dir}"
  done < "${FSD50K_MERGE_QUEUE_FILE}"
}

download_and_extract() {
  local url="$1"
  local dest="$2"
  local filename
  filename="$(basename "${url}")"
  filename="${filename%%\?*}"
  mkdir -p "${DOWNLOAD_DIR}"
  if [[ "${ARIA2_PARALLEL_ACTIVE}" == "1" ]]; then
    local archive_path="${DOWNLOAD_DIR}/${filename}"
    queue_download "${url}" "${archive_path}"
    queue_extract "${archive_path}" "${dest}" "${url}"
    return 0
  fi
  download_file "${url}" "${DOWNLOAD_DIR}/${filename}"
  if ! verify_archive "${DOWNLOAD_DIR}/${filename}"; then
    echo "[warn] archive failed verification, retrying: ${filename}" >&2
    rm -f "${DOWNLOAD_DIR:?}/${filename}"
    download_file "${url}" "${DOWNLOAD_DIR}/${filename}"
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
  (cd "${DOWNLOAD_DIR}" && zip -s 0 "${zip_base}" --out "${zip_base}.merged.zip")
  unzip -q "${DOWNLOAD_DIR}/${zip_base}.merged.zip" -d "${out_dir}"
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
  if [[ "${ARIA2_PARALLEL_ACTIVE}" == "1" ]]; then
    for part in "${parts[@]}"; do
      local url="${base_url}/${prefix}.${part}"
      if [[ "${url}" == *"zenodo.org"* && "${url}" != *"download="* ]]; then
        url="${url}?download=1"
      fi
      queue_download "${url}" "${DOWNLOAD_DIR}/${prefix}.${part}"
    done
    echo "${prefix}.zip|${out_dir}" >> "${FSD50K_MERGE_QUEUE_FILE}"
  else
    for part in "${parts[@]}"; do
      local url="${base_url}/${prefix}.${part}"
      if [[ "${url}" == *"zenodo.org"* && "${url}" != *"download="* ]]; then
        url="${url}?download=1"
      fi
      download_file "${url}" "${DOWNLOAD_DIR}/${prefix}.${part}"
    done
    fsd50k_merge_and_unzip "${prefix}.zip" "${out_dir}"
  fi
}

maybe_install_audb() {
  if python3 "${ROOT_DIR}/scripts/datasets/check_audb.py" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "${INSTALL_AUDB}" == "1" ]]; then
    python3 -m pip install --user audb
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

  # VCTK
  if should_download "${DOWNLOAD_VCTK}"; then
    VCTK_URL="${VCTK_URL:-https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip}"
    download_and_extract "${VCTK_URL}" "${EXTRACT_DIR}"
  fi

  # LibriSpeech
  if should_download "${DOWNLOAD_LIBRISPEECH}"; then
    if [[ -z "${LIBRISPEECH_PARTS:-}" ]]; then
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

  # FSD50K
  if should_download "${DOWNLOAD_FSD50K}"; then
    need_cmd zip
    mkdir -p "${FSD50K_DIR}"
    FSD50K_BASE_URL="${FSD50K_BASE_URL:-https://zenodo.org/records/4060432/files}"
    # Metadata + ground truth + docs
    download_and_extract "${FSD50K_BASE_URL}/FSD50K.metadata.zip?download=1" "${FSD50K_DIR}"
    download_and_extract "${FSD50K_BASE_URL}/FSD50K.ground_truth.zip?download=1" "${FSD50K_DIR}"
    download_and_extract "${FSD50K_BASE_URL}/FSD50K.doc.zip?download=1" "${FSD50K_DIR}"
    # Split audio zips
    download_fsd50k_split "FSD50K.dev_audio" "${FSD50K_DIR}"
    download_fsd50k_split "FSD50K.eval_audio" "${FSD50K_DIR}"
  fi

  # AIR/OpenAIR via audb (optional)
  if [[ "${USE_AUDB}" == "1" ]]; then
    if maybe_install_audb; then
      mkdir -p "${AUDB_DIR}"
      if should_download "${DOWNLOAD_AIR}"; then
        AUDB_NAME="air" AUDB_VERSION="${AIR_VERSION:-1.4.2}" AUDB_ROOT="${AUDB_DIR}" download_with_audb
        AIR_RIR_DIR="${AUDB_DIR}/air"
      fi
      if should_download "${DOWNLOAD_OPENAIR}"; then
        AUDB_NAME="openair" AUDB_VERSION="${OPENAIR_VERSION:-1.0.0}" AUDB_ROOT="${AUDB_DIR}" download_with_audb
        OPENAIR_DIR="${AUDB_DIR}/openair"
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
  fi

  if [[ "${ARIA2_PARALLEL_ACTIVE}" == "1" ]]; then
    run_parallel_downloads
    process_fsd50k_merge_queue
    process_extract_queue
  fi
fi

# Clean speech lists
if require_dir "VCTK" "${VCTK_DIR}"; then
  write_list "${VCTK_DIR}" "${LIST_DIR}/vctk_clean.txt" "*.wav"
fi
if require_dir "LibriSpeech" "${LIBRISPEECH_DIR}"; then
  write_list "${LIBRISPEECH_DIR}" "${LIST_DIR}/librispeech_clean.txt" "*.flac"
fi

# Noise + music lists
if require_dir "MUSAN" "${MUSAN_DIR}"; then
  write_list "${MUSAN_DIR}/noise" "${LIST_DIR}/musan_noise.txt" "*.wav"
  write_list "${MUSAN_DIR}/music" "${LIST_DIR}/musan_music.txt" "*.wav"
fi

# FSD50K filtered list (CC0/CC-BY only)
# Expected inputs:
# - ${FSD50K_DIR}/FSD50K.metadata/ with metadata CSVs
# - ${FSD50K_DIR}/FSD50K.dev_audio/ and/or FSD50K.eval_audio/
# Update CSV/column names as needed after download.
if require_dir "FSD50K" "${FSD50K_DIR}"; then
  export FSD50K_DIR
  export LIST_DIR
  python3 "${ROOT_DIR}/scripts/datasets/fsd50k_filter.py"
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

cat <<'MSG'
Done.
Next steps:
- Combine lists for your target config (prototype vs production).
- Run df.scripts.prepare_data to build speech_clean.hdf5, noise_music.hdf5, rir.hdf5.
MSG
