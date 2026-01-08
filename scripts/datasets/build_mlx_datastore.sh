#!/usr/bin/env bash
set -euo pipefail

# Build MLX datastore or generate file lists for DeepFilterNet training.
#
# Two modes available:
#   1. DYNAMIC (default): Generate file lists for on-the-fly mixing
#      - Matches original Rust DataLoader behavior
#      - Full dataset diversity each epoch
#      - No pre-computation required
#
#   2. PRECOMPUTE: Pre-compute spectral features in sharded format
#      - Faster training startup
#      - Fixed noise/SNR combinations
#      - Limited diversity
#
# Requirements:
#   - Python environment with: numpy, scipy, soundfile, tqdm
#   - Audio file lists (clean speech, noise, optional RIR)
#
# Usage:
#   ./build_mlx_datastore.sh                    # Dynamic mode (default)
#   MODE=precompute ./build_mlx_datastore.sh   # Pre-computed mode
#
# Environment variables:
#   MODE          - Build mode: dynamic | precompute (default: dynamic)
#   DATA_DIR      - Base data directory (default: /Volumes/TrainingData/datasets)
#   OUTPUT_DIR    - Output directory for MLX datastore/file lists
#   LIST_DIR      - Directory containing file lists
#   PROFILE       - Build profile: prototype | production | apple (default: apple)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ============================================================================
# Configuration
# ============================================================================

# Build mode
MODE="${MODE:-dynamic}"

# Data paths
DATA_DIR="${DATA_DIR:-/Volumes/TrainingData/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/mlx_datastore}"
LIST_DIR="${LIST_DIR:-${DATA_DIR}/lists}"

# Build profile
PROFILE="${PROFILE:-apple}"

# Audio parameters
SR="${SR:-48000}"
FFT_SIZE="${FFT_SIZE:-960}"
HOP_SIZE="${HOP_SIZE:-480}"
NB_ERB="${NB_ERB:-32}"
NB_DF="${NB_DF:-96}"

# Training data parameters
SNR_MIN="${SNR_MIN:--5}"
SNR_MAX="${SNR_MAX:-25}"
RIR_PROB="${RIR_PROB:-0.5}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-5.0}"

# Dataset splits
TRAIN_SPLIT="${TRAIN_SPLIT:-0.9}"
VALID_SPLIT="${VALID_SPLIT:-0.05}"

# File lists
CLEAN_LIST="${CLEAN_LIST:-${LIST_DIR}/clean_all.txt}"
NOISE_LIST="${NOISE_LIST:-${LIST_DIR}/noise_music.txt}"
RIR_LIST="${RIR_LIST:-${LIST_DIR}/rir_all.txt}"

# Random seed for reproducibility
SEED="${SEED:-42}"

# ============================================================================
# Profile-specific settings
# ============================================================================

case "${PROFILE}" in
  prototype)
    # Quick test build
    MAX_SAMPLES="${MAX_SAMPLES:-1000}"
    SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-100}
    NUM_WORKERS=${NUM_WORKERS:-1}
    ;;
  production)
    # Full dataset build
    NUM_WORKERS=${NUM_WORKERS:-4}
    SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-1000}
    ;;
  apple)
    # Apple Silicon optimized (memory-friendly)
    NUM_WORKERS=${NUM_WORKERS:-2}
    SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-500}
    ;;
esac

# Performance tuning
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-500}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # Empty = process all

# ============================================================================
# Validation
# ============================================================================

echo "=============================================="
echo "DeepFilterNet MLX Datastore Builder"
echo "=============================================="
echo "Mode:           ${MODE}"
echo "Profile:        ${PROFILE}"
echo "Data dir:       ${DATA_DIR}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "List dir:       ${LIST_DIR}"
echo "Sample rate:    ${SR} Hz"
echo "FFT/Hop:        ${FFT_SIZE}/${HOP_SIZE}"
echo "ERB/DF bands:   ${NB_ERB}/${NB_DF}"
echo "SNR range:      [${SNR_MIN}, ${SNR_MAX}] dB"
echo "Segment length: ${SEGMENT_LENGTH}s"
echo "Train/Valid:    ${TRAIN_SPLIT}/${VALID_SPLIT}"
echo "Workers:        ${NUM_WORKERS}"
if [[ -n "${MAX_SAMPLES}" ]]; then
  echo "Max samples:    ${MAX_SAMPLES}"
fi
echo "=============================================="

# Check file lists exist
if [[ ! -f "${CLEAN_LIST}" ]]; then
  echo "Error: Clean speech list not found: ${CLEAN_LIST}" >&2
  echo ""
  echo "Please create a file list with one audio file path per line."
  echo "Example:"
  echo "  find /path/to/clean/speech -name '*.wav' > ${CLEAN_LIST}"
  exit 1
fi

if [[ ! -f "${NOISE_LIST}" ]]; then
  echo "Error: Noise list not found: ${NOISE_LIST}" >&2
  echo ""
  echo "Please create a file list with one audio file path per line."
  echo "Example:"
  echo "  find /path/to/noise -name '*.wav' > ${NOISE_LIST}"
  exit 1
fi

# RIR list is optional
RIR_ARG=""
if [[ -f "${RIR_LIST}" ]]; then
  RIR_ARG="--rir-list ${RIR_LIST}"
  echo "RIR list:       ${RIR_LIST}"
else
  echo "RIR list:       (none - will skip RIR augmentation)"
fi

# ============================================================================
# Build based on mode
# ============================================================================

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "Starting build..."
echo ""

cd "${ROOT_DIR}/DeepFilterNet"

if [[ "${MODE}" == "dynamic" ]]; then
  # ========================================================================
  # DYNAMIC MODE: Generate file lists for on-the-fly mixing
  # ========================================================================
  echo "Mode: DYNAMIC (on-the-fly mixing)"
  echo "  - Matches original Rust DataLoader behavior"
  echo "  - Full dataset diversity each epoch"
  echo "  - No pre-computation required"
  echo ""

  # Generate file lists and config
  python -m df_mlx.generate_file_lists \
    --speech-list "${CLEAN_LIST}" \
    --noise-list "${NOISE_LIST}" \
    ${RIR_ARG:+--rir-list "${RIR_LIST}"} \
    --output-dir "${OUTPUT_DIR}" \
    --sample-rate "${SR}" \
    --segment-length "${SEGMENT_LENGTH}" \
    --p-reverb "${RIR_PROB}" \
    --generate-config

  echo ""
  echo "=============================================="
  echo "Build complete!"
  echo "=============================================="
  echo "File lists:     ${OUTPUT_DIR}"
  echo "Config:         ${OUTPUT_DIR}/config.json"
  echo ""
  echo "To start training with DYNAMIC on-the-fly mixing:"
  echo "  python -m df_mlx.train_dynamic \\"
  echo "    --config ${OUTPUT_DIR}/config.json \\"
  echo "    --epochs 100 \\"
  echo "    --batch-size 8 \\"
  echo "    --p-reverb ${RIR_PROB}"
  echo ""
  echo "Or with file lists directly:"
  echo "  python -m df_mlx.train_dynamic \\"
  echo "    --speech-list ${OUTPUT_DIR}/speech_files.txt \\"
  echo "    --noise-list ${OUTPUT_DIR}/noise_files.txt \\"
  if [[ -f "${RIR_LIST}" ]]; then
    echo "    --rir-list ${OUTPUT_DIR}/rir_files.txt \\"
  fi
  echo "    --epochs 100 \\"
  echo "    --batch-size 8"
  echo "=============================================="

else
  # ========================================================================
  # PRECOMPUTE MODE: Pre-compute spectral features
  # ========================================================================
  echo "Mode: PRECOMPUTE (pre-computed features)"
  echo "  - Faster training startup"
  echo "  - Fixed noise/SNR combinations"
  echo "  - Limited diversity"
  echo ""

  # Construct max samples argument
  MAX_SAMPLES_ARG=""
  if [[ -n "${MAX_SAMPLES}" ]]; then
    MAX_SAMPLES_ARG="--max-samples ${MAX_SAMPLES}"
  fi

  python -m df_mlx.prepare_data \
    --speech-list "${CLEAN_LIST}" \
    --noise-list "${NOISE_LIST}" \
    ${RIR_ARG} \
    --output-dir "${OUTPUT_DIR}" \
    --sample-rate "${SR}" \
    --fft-size "${FFT_SIZE}" \
    --hop-size "${HOP_SIZE}" \
    --nb-erb "${NB_ERB}" \
    --nb-df "${NB_DF}" \
    --snr-min "${SNR_MIN}" \
    --snr-max "${SNR_MAX}" \
    --rir-prob "${RIR_PROB}" \
    --train-split "${TRAIN_SPLIT}" \
    --valid-split "${VALID_SPLIT}" \
    --samples-per-shard "${SAMPLES_PER_SHARD}" \
    --segment-length "${SEGMENT_LENGTH}" \
    --seed "${SEED}" \
    --num-workers "${NUM_WORKERS}" \
    ${MAX_SAMPLES_ARG}

  echo ""
  echo "=============================================="
  echo "Build complete!"
  echo "=============================================="
  echo "Datastore:  ${OUTPUT_DIR}"
  echo "Index:      ${OUTPUT_DIR}/index.json"
  echo ""
  echo "To start training with PRE-COMPUTED datastore:"
  echo "  python -m df_mlx.train_with_data \\"
  echo "    --datastore ${OUTPUT_DIR} \\"
  echo "    --epochs 100 \\"
  echo "    --batch-size 8"
  echo ""
  echo "NOTE: For better diversity, consider using dynamic mode instead:"
  echo "  MODE=dynamic ./build_mlx_datastore.sh"
  echo "=============================================="
fi
