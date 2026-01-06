#!/usr/bin/env bash
set -euo pipefail

# Build HDF5 datasets for DeepFilterNet training.
# Requires: a Python environment with DeepFilterNet deps installed.
# pip install DeepFilterNet4[train,eval,...]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
LIST_DIR="${LIST_DIR:-${DATA_DIR}/lists}"
HDF5_DIR="${HDF5_DIR:-${DATA_DIR}/hdf5}"
PROFILE="${PROFILE:-prototype}"  # prototype | production | apple
CFG_OUT="${CFG_OUT:-${DATA_DIR}/dataset.cfg}"
SR="${SR:-48000}"
DTYPE="${DTYPE:-int16}"
FORCE_COPY_CFG="${FORCE_COPY_CFG:-0}"

CLEAN_LIST="${CLEAN_LIST:-${LIST_DIR}/clean_all.txt}"
NOISE_LIST="${NOISE_LIST:-${LIST_DIR}/noise_music.txt}"
RIR_LIST="${RIR_LIST:-${LIST_DIR}/rir_all.txt}"

CFG_TEMPLATE="${ROOT_DIR}/datasets/${PROFILE}/dataset.cfg"

mkdir -p "${HDF5_DIR}"

if [[ ! -f "${CLEAN_LIST}" ]]; then
  echo "Missing clean list: ${CLEAN_LIST}" >&2
  exit 1
fi
if [[ ! -f "${NOISE_LIST}" ]]; then
  echo "Missing noise list: ${NOISE_LIST}" >&2
  exit 1
fi
if [[ ! -f "${RIR_LIST}" ]]; then
  echo "Missing RIR list: ${RIR_LIST}" >&2
  exit 1
fi

if [[ ! -f "${CFG_OUT}" || "${FORCE_COPY_CFG}" == "1" ]]; then
  if [[ ! -f "${CFG_TEMPLATE}" ]]; then
    echo "Missing dataset.cfg template: ${CFG_TEMPLATE}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${CFG_OUT}")"
  cp "${CFG_TEMPLATE}" "${CFG_OUT}"
  echo "[ok] wrote dataset.cfg -> ${CFG_OUT}"
else
  echo "[skip] dataset.cfg exists -> ${CFG_OUT} (set FORCE_COPY_CFG=1 to overwrite)"
fi

pushd "${ROOT_DIR}/DeepFilterNet" >/dev/null

# Apple Silicon defaults: fewer workers, lower memory pressure
if [[ "${PROFILE}" == "apple" ]]; then
  NUM_WORKERS="${NUM_WORKERS:-2}"
  MAX_FREQ="${MAX_FREQ:-24000}"
else
  NUM_WORKERS="${NUM_WORKERS:-4}"
  MAX_FREQ="${MAX_FREQ:--1}"
fi

python -m df.scripts.prepare_data speech \
  "${CLEAN_LIST}" "${HDF5_DIR}/speech_clean.hdf5" \
  --sr "${SR}" --dtype "${DTYPE}" --num_workers "${NUM_WORKERS}" --max_freq "${MAX_FREQ}"

python -m df.scripts.prepare_data noise \
  "${NOISE_LIST}" "${HDF5_DIR}/noise_music.hdf5" \
  --sr "${SR}" --dtype "${DTYPE}" --num_workers "${NUM_WORKERS}" --max_freq "${MAX_FREQ}"

python -m df.scripts.prepare_data rir \
  "${RIR_LIST}" "${HDF5_DIR}/rir.hdf5" \
  --sr "${SR}" --dtype "${DTYPE}" --num_workers "${NUM_WORKERS}" --max_freq "${MAX_FREQ}"

popd >/dev/null

echo "Done."
echo "HDF5 output: ${HDF5_DIR}"
echo "Dataset config: ${CFG_OUT}"
