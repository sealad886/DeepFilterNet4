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

# Preserve the original stdout/stderr so we can keep interactive output (tqdm)
# even after we redirect pipeline logs.
exec 3>&1 4>&2

# ============================================================================
# Configuration and Default Values
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"

# Default values
WALL_DATASET="${WALL_DATASET:-/Users/andrew/DataDump/datasets/wall_processed}"
WALL_CHECKPOINT="${WALL_CHECKPOINT:-/Users/andrew/DataDump/checkpoints/dfnetmf_wall}"
DYNAMIC_CACHE="${DYNAMIC_CACHE:-/Users/andrew/DataDump/datasets/mlx_datastore}"
DYNAMIC_CHECKPOINT="${DYNAMIC_CHECKPOINT:-/Users/andrew/DataDump/checkpoints}"
EPOCHS="${EPOCHS:-100}"
WALL_BATCH_SIZE="${WALL_BATCH_SIZE:-32}"
DYNAMIC_BATCH_SIZE="${DYNAMIC_BATCH_SIZE:-8}"
WALL_METHOD="${WALL_METHOD:-WF}"
DYNAMIC_SNR_MIN="${DYNAMIC_SNR_MIN:--5}"
DYNAMIC_SNR_MAX="${DYNAMIC_SNR_MAX:-40}"
DYNAMIC_SNR_EXTREME_MIN="${DYNAMIC_SNR_EXTREME_MIN:--20}"
DYNAMIC_SNR_EXTREME_MAX="${DYNAMIC_SNR_EXTREME_MAX:--5}"
DYNAMIC_P_EXTREME_SNR="${DYNAMIC_P_EXTREME_SNR:-0.1}"
DYNAMIC_SPEECH_GAIN_MIN="${DYNAMIC_SPEECH_GAIN_MIN:--12}"
DYNAMIC_SPEECH_GAIN_MAX="${DYNAMIC_SPEECH_GAIN_MAX:-12}"
DYNAMIC_NOISE_GAIN_MIN="${DYNAMIC_NOISE_GAIN_MIN:--12}"
DYNAMIC_NOISE_GAIN_MAX="${DYNAMIC_NOISE_GAIN_MAX:-12}"
DYNAMIC_VAD_LOSS_WEIGHT="${DYNAMIC_VAD_LOSS_WEIGHT:-0.05}"
DYNAMIC_VAD_THRESHOLD="${DYNAMIC_VAD_THRESHOLD:-0.6}"
DYNAMIC_VAD_MARGIN="${DYNAMIC_VAD_MARGIN:-0.05}"
DYNAMIC_VAD_SPEECH_LOSS_WEIGHT="${DYNAMIC_VAD_SPEECH_LOSS_WEIGHT:-0.0}"
DYNAMIC_VAD_WARMUP_EPOCHS="${DYNAMIC_VAD_WARMUP_EPOCHS:-5}"
DYNAMIC_VAD_SNR_GATE="${DYNAMIC_VAD_SNR_GATE:--10}"
DYNAMIC_VAD_SNR_GATE_WIDTH="${DYNAMIC_VAD_SNR_GATE_WIDTH:-6}"
DYNAMIC_VAD_BAND_LOW="${DYNAMIC_VAD_BAND_LOW:-300}"
DYNAMIC_VAD_BAND_HIGH="${DYNAMIC_VAD_BAND_HIGH:-3400}"
DYNAMIC_VAD_Z_THRESHOLD="${DYNAMIC_VAD_Z_THRESHOLD:-0.0}"
DYNAMIC_VAD_Z_SLOPE="${DYNAMIC_VAD_Z_SLOPE:-1.0}"
DYNAMIC_EVAL_SISDR="${DYNAMIC_EVAL_SISDR:-0}"
DYNAMIC_CHECK_CHKPTS="${DYNAMIC_CHECK_CHKPTS:-0}"

# ============================================================================
# Ensure one pipeline run at a time (lock file)
# ============================================================================
#
# Goals:
# - Atomic lock acquisition (no racy "check then create")
# - Detect and clear stale locks
# - ALWAYS remove lock on any normal/abnormal exit we can catch (EXIT trap)
# - Handle SIGINT/SIGTERM cleanly; SIGKILL cannot be trapped in Unix.
#
LOCKFILE=""  # set after LOGS_DIR exists

checkLock() {
  # Requires: LOGS_DIR exists
  LOCKFILE="${LOGS_DIR}/training_pipeline.lock"

  # Try atomic creation with noclobber. If it fails, lock already exists.
  if ( set -o noclobber; printf '%s\n' "$$" >"$LOCKFILE" ) 2>/dev/null; then
    return 0
  fi

  # Lock exists: decide whether it's active or stale.
  local existing_pid=""
  existing_pid="$(head -n 1 "$LOCKFILE" 2>/dev/null || true)"

  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "❌ ERROR: Another training pipeline appears to be running (PID: $existing_pid)." >&4
    echo "   Lock file: $LOCKFILE" >&4
    echo "   If that process is definitely gone, remove the lock: rm -f \"$LOCKFILE\"" >&4
    return 1
  fi

  echo "⚠️  Found stale lock file; removing it: $LOCKFILE" >&4
  rm -f "$LOCKFILE" 2>/dev/null || true

  # Retry once after removing stale lock.
  if ( set -o noclobber; printf '%s\n' "$$" >"$LOCKFILE" ) 2>/dev/null; then
    return 0
  fi

  echo "❌ ERROR: Failed to acquire lock after clearing stale lock: $LOCKFILE" >&4
  return 1
}

releaseLock() {
  # Remove the lock if we own it (best-effort). Never fail the script because of cleanup.
  [[ -n "${LOCKFILE:-}" ]] || return 0
  [[ -f "$LOCKFILE" ]] || return 0

  local lock_pid=""
  lock_pid="$(head -n 1 "$LOCKFILE" 2>/dev/null || true)"

  # Only remove if we created it (or if unreadable/empty, remove anyway).
  if [[ -z "$lock_pid" || "$lock_pid" == "$$" ]]; then
    rm -f "$LOCKFILE" 2>/dev/null || true
  fi
}

# Ensure we always remove the lock on any exit path we can catch.
trap 'releaseLock' EXIT

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
    --dynamic-snr-range)
      DYNAMIC_SNR_MIN="$2"
      DYNAMIC_SNR_MAX="$3"
      shift 3
      ;;
    --dynamic-snr-range-extreme)
      DYNAMIC_SNR_EXTREME_MIN="$2"
      DYNAMIC_SNR_EXTREME_MAX="$3"
      shift 3
      ;;
    --dynamic-p-extreme-snr)
      DYNAMIC_P_EXTREME_SNR="$2"
      shift 2
      ;;
    --dynamic-speech-gain-range)
      DYNAMIC_SPEECH_GAIN_MIN="$2"
      DYNAMIC_SPEECH_GAIN_MAX="$3"
      shift 3
      ;;
    --dynamic-noise-gain-range)
      DYNAMIC_NOISE_GAIN_MIN="$2"
      DYNAMIC_NOISE_GAIN_MAX="$3"
      shift 3
      ;;
    --dynamic-vad-loss-weight)
      DYNAMIC_VAD_LOSS_WEIGHT="$2"
      shift 2
      ;;
    --dynamic-vad-threshold)
      DYNAMIC_VAD_THRESHOLD="$2"
      shift 2
      ;;
    --dynamic-vad-margin)
      DYNAMIC_VAD_MARGIN="$2"
      shift 2
      ;;
    --dynamic-vad-speech-loss-weight)
      DYNAMIC_VAD_SPEECH_LOSS_WEIGHT="$2"
      shift 2
      ;;
    --dynamic-vad-warmup-epochs)
      DYNAMIC_VAD_WARMUP_EPOCHS="$2"
      shift 2
      ;;
    --dynamic-vad-snr-gate)
      DYNAMIC_VAD_SNR_GATE="$2"
      shift 2
      ;;
    --dynamic-vad-snr-gate-width)
      DYNAMIC_VAD_SNR_GATE_WIDTH="$2"
      shift 2
      ;;
    --dynamic-vad-band-low)
      DYNAMIC_VAD_BAND_LOW="$2"
      shift 2
      ;;
    --dynamic-vad-band-high)
      DYNAMIC_VAD_BAND_HIGH="$2"
      shift 2
      ;;
    --dynamic-vad-z-threshold)
      DYNAMIC_VAD_Z_THRESHOLD="$2"
      shift 2
      ;;
    --dynamic-vad-z-slope)
      DYNAMIC_VAD_Z_SLOPE="$2"
      shift 2
      ;;
    --dynamic-eval-sisdr)
      DYNAMIC_EVAL_SISDR=1
      shift 1
      ;;
    --check-chkpts)
      DYNAMIC_CHECK_CHKPTS=1
      shift 1
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
  --dynamic-snr-range MIN MAX  Base SNR range for dynamic mixing (default: $DYNAMIC_SNR_MIN $DYNAMIC_SNR_MAX)
  --dynamic-snr-range-extreme MIN MAX  Extreme SNR range for near-obscured speech (default: $DYNAMIC_SNR_EXTREME_MIN $DYNAMIC_SNR_EXTREME_MAX)
  --dynamic-p-extreme-snr P    Probability of extreme SNR sampling (default: $DYNAMIC_P_EXTREME_SNR)
  --dynamic-speech-gain-range MIN MAX  Speech gain range in dB (default: $DYNAMIC_SPEECH_GAIN_MIN $DYNAMIC_SPEECH_GAIN_MAX)
  --dynamic-noise-gain-range MIN MAX   Noise gain range in dB (default: $DYNAMIC_NOISE_GAIN_MIN $DYNAMIC_NOISE_GAIN_MAX)
  --dynamic-vad-loss-weight W  VAD loss weight (default: $DYNAMIC_VAD_LOSS_WEIGHT)
  --dynamic-vad-threshold T    VAD threshold for gating (default: $DYNAMIC_VAD_THRESHOLD)
  --dynamic-vad-margin M       VAD margin (default: $DYNAMIC_VAD_MARGIN)
  --dynamic-vad-speech-loss-weight W  VAD speech-structure loss weight (default: $DYNAMIC_VAD_SPEECH_LOSS_WEIGHT)
  --dynamic-vad-warmup-epochs N  VAD warmup epochs (default: $DYNAMIC_VAD_WARMUP_EPOCHS)
  --dynamic-vad-snr-gate DB    VAD SNR gate threshold dB (default: $DYNAMIC_VAD_SNR_GATE)
  --dynamic-vad-snr-gate-width DB  VAD SNR gate width dB (default: $DYNAMIC_VAD_SNR_GATE_WIDTH)
  --dynamic-vad-band-low HZ    VAD band low cutoff Hz (default: $DYNAMIC_VAD_BAND_LOW)
  --dynamic-vad-band-high HZ   VAD band high cutoff Hz (default: $DYNAMIC_VAD_BAND_HIGH)
  --dynamic-vad-z-threshold Z  VAD z-threshold (default: $DYNAMIC_VAD_Z_THRESHOLD)
  --dynamic-vad-z-slope S      VAD z-slope (default: $DYNAMIC_VAD_Z_SLOPE)
  --dynamic-eval-sisdr         Enable SI-SDR validation metric (default: $DYNAMIC_EVAL_SISDR)
  --check-chkpts               Validate dynamic checkpoints before resume/start (default: $DYNAMIC_CHECK_CHKPTS)
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

# Acquire the lock *after* logs dir exists but *before* we start long-running work.
# If we can't acquire it, quit cleanly (the EXIT trap will be a no-op because
# LOCKFILE hasn't been created by us).
if ! checkLock; then
  exit 1
fi

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WALL_LOG="$LOGS_DIR/wall_training_${TIMESTAMP}.log"
DYNAMIC_LOG="$LOGS_DIR/dynamic_training_${TIMESTAMP}.log"
PIPELINE_LOG="$LOGS_DIR/pipeline_${TIMESTAMP}.log"

 # Log pipeline stdout/stderr to file while still printing to the user's terminal.
 # Note: this makes the pipeline's own fds non-TTY; training processes that need
 # a TTY (tqdm) should explicitly inherit the original stderr via fd 4.
exec > >(tee -a "$PIPELINE_LOG" >&3) 2> >(tee -a "$PIPELINE_LOG" >&4)

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
echo "  Dynamic SNR Range:   $DYNAMIC_SNR_MIN to $DYNAMIC_SNR_MAX dB"
echo "  Extreme SNR Range:   $DYNAMIC_SNR_EXTREME_MIN to $DYNAMIC_SNR_EXTREME_MAX dB (p=$DYNAMIC_P_EXTREME_SNR)"
echo "  Speech Gain Range:   $DYNAMIC_SPEECH_GAIN_MIN to $DYNAMIC_SPEECH_GAIN_MAX dB"
echo "  Noise Gain Range:    $DYNAMIC_NOISE_GAIN_MIN to $DYNAMIC_NOISE_GAIN_MAX dB"
echo
echo "Logs:"
echo "  Pipeline:            $PIPELINE_LOG"
echo "  Wall Training:       $WALL_LOG"
echo "  Dynamic Training:    $DYNAMIC_LOG"
echo
echo "Lock:"
echo "  Lock file:           $LOCKFILE"
echo "  PID:                 $$"
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

# ==========================================================================
# Python executable selection
# ==========================================================================
# The pipeline is often run in environments where `python` may not be on PATH
# unless a venv is active. Prefer the current PATH, then fall back to repo .venv.
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    echo "❌ Could not find a usable Python interpreter. Activate the venv (e.g. 've .venv') or set PYTHON_BIN." >&4
    exit 127
  fi
fi

# Print for visibility
echo "  Python:             $PYTHON_BIN"

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
WALL_CMD=(
  "$PYTHON_BIN" -u scripts/train_dfnetmf_wall.py
  --dataset "$WALL_DATASET"
  --checkpoint-dir "$WALL_CHECKPOINT"
  --method "$WALL_METHOD"
  --batch-size "$WALL_BATCH_SIZE"
  --epochs "$EPOCHS"
  --resume
)

# Keep tqdm interactive on the terminal (stderr), while logging stdout to the wall log.
"${WALL_CMD[@]}" >"$WALL_LOG" 2>&4 &

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
DYNAMIC_CMD=(
  "$PYTHON_BIN" -u -m DeepFilterNet.df_mlx.train_dynamic
  --cache-dir "$DYNAMIC_CACHE"
  --epochs "$EPOCHS"
  --batch-size "$DYNAMIC_BATCH_SIZE"
  --checkpoint-dir "$DYNAMIC_CHECKPOINT"
  --save-strategy steps
  --save-steps 1000
  --save-total-limit 10
  --checkpoint-batches 1000
  --prefetch-size 64
  --snr-range "$DYNAMIC_SNR_MIN" "$DYNAMIC_SNR_MAX"
  --snr-range-extreme "$DYNAMIC_SNR_EXTREME_MIN" "$DYNAMIC_SNR_EXTREME_MAX"
  --p-extreme-snr "$DYNAMIC_P_EXTREME_SNR"
  --speech-gain-range "$DYNAMIC_SPEECH_GAIN_MIN" "$DYNAMIC_SPEECH_GAIN_MAX"
  --noise-gain-range "$DYNAMIC_NOISE_GAIN_MIN" "$DYNAMIC_NOISE_GAIN_MAX"
  --vad-loss-weight "$DYNAMIC_VAD_LOSS_WEIGHT"
  --vad-threshold "$DYNAMIC_VAD_THRESHOLD"
  --vad-margin "$DYNAMIC_VAD_MARGIN"
  --vad-speech-loss-weight "$DYNAMIC_VAD_SPEECH_LOSS_WEIGHT"
  --vad-warmup-epochs "$DYNAMIC_VAD_WARMUP_EPOCHS"
  --vad-snr-gate "$DYNAMIC_VAD_SNR_GATE"
  --vad-snr-gate-width "$DYNAMIC_VAD_SNR_GATE_WIDTH"
  --vad-band-low "$DYNAMIC_VAD_BAND_LOW"
  --vad-band-high "$DYNAMIC_VAD_BAND_HIGH"
  --vad-z-threshold "$DYNAMIC_VAD_Z_THRESHOLD"
  --vad-z-slope "$DYNAMIC_VAD_Z_SLOPE"
  --fp16 -v --backbone-type attention
  --resume --resume-data
)

if [[ "$DYNAMIC_EVAL_SISDR" == "1" ]]; then
  DYNAMIC_CMD+=(--eval-sisdr)
fi
if [[ "$DYNAMIC_CHECK_CHKPTS" == "1" ]]; then
  DYNAMIC_CMD+=(--check-chkpts)
fi

# Keep tqdm interactive on the terminal (stderr), while logging stdout to the dynamic log.
"${DYNAMIC_CMD[@]}" >"$DYNAMIC_LOG" 2>&4 &

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
