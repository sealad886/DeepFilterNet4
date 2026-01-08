#!/usr/bin/env bash
set -euo pipefail

# Build MLX audio cache for DeepFilterNet training.
#
# This creates a pre-processed audio cache that enables:
# - FAST loading: Pre-resampled numpy arrays in sharded NPZ files
# - DYNAMIC mixing: Speech + noise + RIR combined at training time
# - FULL DIVERSITY: Different combinations each epoch (like original Rust)
#
# The cache stores processed audio arrays, NOT pre-computed features.
# Features (STFT, ERB, DF) are computed dynamically during training.
#
# Requirements:
#   - Python environment with: numpy, scipy, soundfile, tqdm
#   - Audio file lists (clean speech, noise, optional RIR)
#
# Usage:
#   ./build_mlx_datastore.sh
#
# Environment variables:
#   DATA_DIR      - Base data directory (default: /Volumes/TrainingData/datasets)
#   OUTPUT_DIR    - Output directory for audio cache
#   LIST_DIR      - Directory containing file lists
#   PROFILE       - Build profile: prototype | production | apple (default: apple)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ============================================================================
# Configuration
# ============================================================================

# Data paths
DATA_DIR="${DATA_DIR:-/Volumes/TrainingData/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/mlx_audio_cache}"
LIST_DIR="${LIST_DIR:-${DATA_DIR}/lists}"

# Build profile
PROFILE="${PROFILE:-apple}"

# Audio parameters
SR="${SR:-48000}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-5.0}"

# Mixing parameters (stored in config.json for training)
SNR_MIN="${SNR_MIN:--5}"
SNR_MAX="${SNR_MAX:-40}"
RIR_PROB="${RIR_PROB:-0.5}"

# File lists
CLEAN_LIST="${CLEAN_LIST:-${LIST_DIR}/clean_all.txt}"
NOISE_LIST="${NOISE_LIST:-${LIST_DIR}/noise_music.txt}"
RIR_LIST="${RIR_LIST:-${LIST_DIR}/rir_all.txt}"

# ============================================================================
# Profile-specific settings
# ============================================================================

case "${PROFILE}" in
  prototype)
    # Quick test build
    NUM_WORKERS=${NUM_WORKERS:-1}
    SHARD_SIZE=${SHARD_SIZE:-100}
    ;;
  production)
    # Full dataset build
    NUM_WORKERS=${NUM_WORKERS:-8}
    SHARD_SIZE=${SHARD_SIZE:-500}
    ;;
  apple)
    # Apple Silicon optimized (memory-friendly)
    NUM_WORKERS=${NUM_WORKERS:-4}
    SHARD_SIZE=${SHARD_SIZE:-500}
    ;;
esac

# Performance tuning
SHARD_SIZE="${SHARD_SIZE:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# ============================================================================
# Validation
# ============================================================================

echo "=============================================="
echo "DeepFilterNet MLX Audio Cache Builder"
echo "=============================================="
echo "Profile:        ${PROFILE}"
echo "Data dir:       ${DATA_DIR}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "List dir:       ${LIST_DIR}"
echo "Sample rate:    ${SR} Hz"
echo "Segment length: ${SEGMENT_LENGTH}s"
echo "SNR range:      [${SNR_MIN}, ${SNR_MAX}] dB"
echo "RIR prob:       ${RIR_PROB}"
echo "Workers:        ${NUM_WORKERS}"
echo "Shard size:     ${SHARD_SIZE}"
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
# Build audio cache
# ============================================================================

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "Starting audio cache build..."
echo ""
echo "This pre-processes all audio files (resample, normalize) and saves"
echo "them in sharded NPZ format for efficient loading during training."
echo ""
echo "The actual mixing (speech + noise + RIR @ random SNR) happens"
echo "dynamically during training - giving full diversity each epoch."
echo ""
echo "Resume mode is enabled by default - will skip already-cached files."
echo ""

cd "${ROOT_DIR}/DeepFilterNet"

python -m df_mlx.build_audio_cache \
  --speech-list "${CLEAN_LIST}" \
  --noise-list "${NOISE_LIST}" \
  ${RIR_ARG} \
  --output-dir "${OUTPUT_DIR}" \
  --sample-rate "${SR}" \
  --segment-length "${SEGMENT_LENGTH}" \
  --shard-size "${SHARD_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --snr-min "${SNR_MIN}" \
  --snr-max "${SNR_MAX}" \
  --p-reverb "${RIR_PROB}" \
  --resume

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo "Audio cache:  ${OUTPUT_DIR}"
echo "Config:       ${OUTPUT_DIR}/config.json"
echo ""
echo "To start training with DYNAMIC mixing:"
echo "  python -m df_mlx.train_dynamic \\"
echo "    --cache-dir ${OUTPUT_DIR} \\"
echo "    --epochs 100 \\"
echo "    --batch-size 8"
echo ""
echo "Key advantages over pre-computed datastores:"
echo "  - Full dataset diversity (all files available each epoch)"
echo "  - Different noise/SNR/RIR combinations each epoch"
echo "  - Matches original Rust DataLoader behavior"
echo "=============================================="
