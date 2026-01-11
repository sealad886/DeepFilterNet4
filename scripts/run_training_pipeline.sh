#!/usr/bin/env bash
#
# run_training_pipeline.sh
#
# Automated training pipeline for DeepFilterNet4 models
# Runs wall training (100 epochs) followed by dynamic training (100 epochs)
# with robust logging, checkpointing, and signal handling.
#
# Usage:
#   ./scripts/run_training_pipeline.sh \
#     --wall-dataset /path/to/wall_processed \
#     --wall-checkpoint /path/to/checkpoints/dfnetmf_wall \
#     --dynamic-cache /path/to/audio_cache \
#     --dynamic-checkpoint /path/to/checkpoints/dfnet4_dynamic \
#     --epochs 100 \
#     --wall-batch-size 32 \
#     --dynamic-batch-size 8

set -euo pipefail

# ============================================================================
# Configuration and Default Values
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"

# Default values
WALL_DATASET="${WALL_DATASET:-/Users/andrew/DataDump/datasets/wall_processed}"
WALL_CHECKPOINT="${WALL_CHECKPOINT:-/Users/andrew/DataDump/checkpoints/dfnetmf_wall}"
DYNAMIC_CACHE="${DYNAMIC_CACHE:-/Users/andrew/DataDump/audio_cache}"
DYNAMIC_CHECKPOINT="${DYNAMIC_CHECKPOINT:-/Users/andrew/DataDump/checkpoints/dfnet4_dynamic}"
EPOCHS="${EPOCHS:-100}"
WALL_BATCH_SIZE="${WALL_BATCH_SIZE:-32}"
DYNAMIC_BATCH_SIZE="${DYNAMIC_BATCH_SIZE:-8}"
WALL_METHOD="${WALL_METHOD:-WF}"

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wall-dataset)
      WALL_DATASET="$2"
      shift 2
      ;;
    --wall-checkpoint)
      WALL_CHECKPOINT="$2"
      shift 2
      ;;
    --dynamic-cache)
      DYNAMIC_CACHE="$2"
      shift 2
      ;;
    --dynamic-checkpoint)
      DYNAMIC_CHECKPOINT="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --wall-batch-size)
      WALL_BATCH_SIZE="$2"
      shift 2
      ;;
    --dynamic-batch-size)
      DYNAMIC_BATCH_SIZE="$2"
      shift 2
      ;;
    --wall-method)
      WALL_METHOD="$2"
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 [OPTIONS]

Automated training pipeline for DeepFilterNet4 models.

Options:
  --wall-dataset PATH          Path to wall camera dataset (default: $WALL_DATASET)
  --wall-checkpoint PATH       Checkpoint directory for wall training (default: $WALL_CHECKPOINT)
  --dynamic-cache PATH         Path to audio cache for dynamic training (default: $DYNAMIC_CACHE)
  --dynamic-checkpoint PATH    Checkpoint directory for dynamic training (default: $DYNAMIC_CHECKPOINT)
  --epochs N                   Number of epochs for each training run (default: $EPOCHS)
  --wall-batch-size N          Batch size for wall training (default: $WALL_BATCH_SIZE)
  --dynamic-batch-size N       Batch size for dynamic training (default: $DYNAMIC_BATCH_SIZE)
  --wall-method METHOD         Wall training method: WF or MVDR (default: $WALL_METHOD)
  -h, --help                   Show this help message

Environment Variables:
  You can also set configuration via environment variables with the same names as the options.

Example:
  $0 --epochs 100 --wall-batch-size 32 --dynamic-batch-size 8
EOF
      exit 0
      ;;
    *)
      echo "❌ Unknown option: $1"
      echo "Use -h or --help for usage information"
      exit 1
      ;;
  esac
done

# ============================================================================
# Setup and Validation
# ============================================================================

# Ensure logs directory exists
mkdir -p "$LOGS_DIR"

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WALL_LOG="$LOGS_DIR/wall_training_${TIMESTAMP}.log"
DYNAMIC_LOG="$LOGS_DIR/dynamic_training_${TIMESTAMP}.log"
PIPELINE_LOG="$LOGS_DIR/pipeline_${TIMESTAMP}.log"

# Redirect pipeline output to log file
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "=" ================================================================
echo "DeepFilterNet4 Training Pipeline"
echo "=" ================================================================
echo "Started: $(date)"
echo
echo "Configuration:"
echo "  Wall Dataset:        $WALL_DATASET"
echo "  Wall Checkpoint:     $WALL_CHECKPOINT"
echo "  Dynamic Cache:       $DYNAMIC_CACHE"
echo "  Dynamic Checkpoint:  $DYNAMIC_CHECKPOINT"
echo "  Epochs:              $EPOCHS"
echo "  Wall Batch Size:     $WALL_BATCH_SIZE"
echo "  Dynamic Batch Size:  $DYNAMIC_BATCH_SIZE"
echo "  Wall Method:         $WALL_METHOD"
echo
echo "Logs:"
echo "  Pipeline:            $PIPELINE_LOG"
echo "  Wall Training:       $WALL_LOG"
echo "  Dynamic Training:    $DYNAMIC_LOG"
echo

# Validate dataset and cache directories exist
if [[ ! -d "$WALL_DATASET" ]]; then
  echo "❌ Wall dataset directory not found: $WALL_DATASET"
  exit 1
fi

if [[ ! -d "$DYNAMIC_CACHE" ]]; then
  echo "⚠️  Dynamic cache directory not found: $DYNAMIC_CACHE"
  echo "   Continuing anyway - dynamic training will fail if cache is required"
fi

# ============================================================================
# Signal Handling
# ============================================================================

# Track PIDs of background processes
WALL_PID=""
DYNAMIC_PID=""
INTERRUPTED=false

cleanup() {
  if [[ "$INTERRUPTED" == "true" ]]; then
    echo
    echo "=" ===============================================================
    echo "⚠️  Pipeline interrupted - cleanup already performed"
    echo "=" ===============================================================
    return
  fi

  INTERRUPTED=true
  echo
  echo "=" ================================================================
  echo "⚠️  Training Pipeline Interrupted (SIGINT/SIGTERM)"
  echo "=" ================================================================

  if [[ -n "$WALL_PID" ]] && kill -0 "$WALL_PID" 2>/dev/null; then
    echo "Stopping wall training (PID: $WALL_PID)..."
    kill -INT "$WALL_PID" 2>/dev/null || true
    wait "$WALL_PID" 2>/dev/null || true
    echo "✅ Wall training stopped"
  fi

  if [[ -n "$DYNAMIC_PID" ]] && kill -0 "$DYNAMIC_PID" 2>/dev/null; then
    echo "Stopping dynamic training (PID: $DYNAMIC_PID)..."
    kill -INT "$DYNAMIC_PID" 2>/dev/null || true
    wait "$DYNAMIC_PID" 2>/dev/null || true
    echo "✅ Dynamic training stopped"
  fi

  echo
  echo "Pipeline stopped at: $(date)"
  echo "Logs saved to: $LOGS_DIR"
  exit 130
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# Phase 1: Wall Training
# ============================================================================

echo "=" ================================================================
echo "Phase 1: Wall Training"
echo "=" ================================================================
echo "Starting wall training (method: $WALL_METHOD, epochs: $EPOCHS, batch: $WALL_BATCH_SIZE)"
echo "Log file: $WALL_LOG"
echo

WALL_START=$(date +%s)

# Run wall training with resume support
cd "$REPO_ROOT"
python scripts/train_dfnetmf_wall.py \
  --dataset "$WALL_DATASET" \
  --checkpoint-dir "$WALL_CHECKPOINT" \
  --method "$WALL_METHOD" \
  --batch-size "$WALL_BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --resume \
  > "$WALL_LOG" 2>&1 &

WALL_PID=$!
echo "Wall training started (PID: $WALL_PID)"

# Wait for wall training to complete
if wait "$WALL_PID"; then
  WALL_EXIT=0
  WALL_END=$(date +%s)
  WALL_DURATION=$((WALL_END - WALL_START))

  echo "✅ Wall training completed successfully"
  echo "   Duration: $(($WALL_DURATION / 60)) minutes $(($WALL_DURATION % 60)) seconds"
  echo "   Checkpoints: $WALL_CHECKPOINT"
else
  WALL_EXIT=$?
  WALL_END=$(date +%s)
  WALL_DURATION=$((WALL_END - WALL_START))

  echo "❌ Wall training failed (exit code: $WALL_EXIT)"
  echo "   Duration: $(($WALL_DURATION / 60)) minutes $(($WALL_DURATION % 60)) seconds"
  echo "   Check logs: $WALL_LOG"
  exit "$WALL_EXIT"
fi

WALL_PID=""  # Clear PID after completion
echo

# ============================================================================
# Phase 2: Dynamic Training
# ============================================================================

echo "=" ================================================================
echo "Phase 2: Dynamic Training"
echo "=" ================================================================
echo "Starting dynamic training (epochs: $EPOCHS, batch: $DYNAMIC_BATCH_SIZE)"
echo "Log file: $DYNAMIC_LOG"
echo

DYNAMIC_START=$(date +%s)

# Run dynamic training with resume support
cd "$REPO_ROOT"
python -m DeepFilterNet.df_mlx.train_dynamic \
  --cache-dir "$DYNAMIC_CACHE" \
  --epochs "$EPOCHS" \
  --batch-size "$DYNAMIC_BATCH_SIZE" \
  --checkpoint-dir "$DYNAMIC_CHECKPOINT" \
  > "$DYNAMIC_LOG" 2>&1 &

DYNAMIC_PID=$!
echo "Dynamic training started (PID: $DYNAMIC_PID)"

# Wait for dynamic training to complete
if wait "$DYNAMIC_PID"; then
  DYNAMIC_EXIT=0
  DYNAMIC_END=$(date +%s)
  DYNAMIC_DURATION=$((DYNAMIC_END - DYNAMIC_START))

  echo "✅ Dynamic training completed successfully"
  echo "   Duration: $(($DYNAMIC_DURATION / 60)) minutes $(($DYNAMIC_DURATION % 60)) seconds"
  echo "   Checkpoints: $DYNAMIC_CHECKPOINT"
else
  DYNAMIC_EXIT=$?
  DYNAMIC_END=$(date +%s)
  DYNAMIC_DURATION=$((DYNAMIC_END - DYNAMIC_START))

  echo "❌ Dynamic training failed (exit code: $DYNAMIC_EXIT)"
  echo "   Duration: $(($DYNAMIC_DURATION / 60)) minutes $(($DYNAMIC_DURATION % 60)) seconds"
  echo "   Check logs: $DYNAMIC_LOG"
  exit "$DYNAMIC_EXIT"
fi

DYNAMIC_PID=""  # Clear PID after completion
echo

# ============================================================================
# Pipeline Summary
# ============================================================================

PIPELINE_END=$(date +%s)
TOTAL_DURATION=$((PIPELINE_END - WALL_START))

echo "=" ================================================================
echo "Training Pipeline Complete"
echo "=" ================================================================
echo "Total Duration: $(($TOTAL_DURATION / 3600))h $(($TOTAL_DURATION % 3600 / 60))m $(($TOTAL_DURATION % 60))s"
echo
echo "Phase Results:"
echo "  Wall Training:     ✅ Success ($(($WALL_DURATION / 60))m $(($WALL_DURATION % 60))s)"
echo "  Dynamic Training:  ✅ Success ($(($DYNAMIC_DURATION / 60))m $(($DYNAMIC_DURATION % 60))s)"
echo
echo "Checkpoints:"
echo "  Wall:     $WALL_CHECKPOINT"
echo "  Dynamic:  $DYNAMIC_CHECKPOINT"
echo
echo "Logs:"
echo "  Pipeline:  $PIPELINE_LOG"
echo "  Wall:      $WALL_LOG"
echo "  Dynamic:   $DYNAMIC_LOG"
echo
echo "Completed: $(date)"
echo "=" ================================================================

exit 0
