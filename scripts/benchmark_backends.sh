#!/usr/bin/env bash
#
# benchmark_backends.sh
#
# Run short, comparable quality benchmarks across backbones (attention/mamba/gru).
# Uses MLXDataStream by default when available and logs validation metrics.
#
# Usage:
#   ./scripts/benchmark_backends.sh --cache-dir /path/to/cache
#
# Notes:
# - Results are stochastic (dynamic mixing); keep runs short and comparable.
# - Validation always uses PrefetchDataLoader with 2 workers (per train_dynamic.py).
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"

# Defaults (override via env or CLI)
DEFAULT_CACHE_DIR=""
if [[ -n "${DFNET_CACHE_DIR:-}" ]]; then
  DEFAULT_CACHE_DIR="$DFNET_CACHE_DIR"
elif [[ -d "/Users/andrew/DataDump/datasets/mlx_datastore" ]]; then
  DEFAULT_CACHE_DIR="/Users/andrew/DataDump/datasets/mlx_datastore"
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

CACHE_DIR="${CACHE_DIR:-$DEFAULT_CACHE_DIR}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints/backend_bench}"
RUN_ID="${RUN_ID:-latest}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZES="${BATCH_SIZES:-128}"
BACKBONES="${BACKBONES:-attention,mamba,gru}"
EVAL_FREQUENCIES="${EVAL_FREQUENCIES:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_SIZE="${PREFETCH_SIZE:-8}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
USE_FP16="${USE_FP16:-1}"
USE_MLX_DATA="${USE_MLX_DATA:-1}"
EVAL_SISDR="${EVAL_SISDR:-1}"
VAD_LOSS_WEIGHT="${VAD_LOSS_WEIGHT:-0.05}"
VAD_SPEECH_LOSS_WEIGHT="${VAD_SPEECH_LOSS_WEIGHT:-0.03}"
MEM_SAMPLE_INTERVAL="${MEM_SAMPLE_INTERVAL:-1}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-64}"
MAX_VALID_BATCHES="${MAX_VALID_BATCHES:-8}"
VERBOSE="${VERBOSE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ANALYSIS_ONLY="${ANALYSIS_ONLY:-0}"

SUMMARY_JSONL="${SUMMARY_JSONL:-$LOGS_DIR/backend_benchmark_${RUN_ID}.jsonl}"
SUMMARY_CSV="${SUMMARY_CSV:-$LOGS_DIR/backend_benchmark_${RUN_ID}.csv}"

EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $0 --cache-dir PATH [options]

Options:
  --cache-dir PATH            Cache directory (default: $CACHE_DIR)
  --checkpoint-root PATH      Root checkpoint dir (default: $CHECKPOINT_ROOT)
  --run-id ID                 Tag for this run (default: $RUN_ID)
  --epochs N                  Epochs per run (default: $EPOCHS)
  --batch-sizes LIST          Comma-separated (default: $BATCH_SIZES)
  --backbones LIST            Comma-separated (default: $BACKBONES)
  --eval-frequencies LIST     Comma-separated (default: $EVAL_FREQUENCIES)
  --num-workers N             MLXDataStream workers (default: $NUM_WORKERS)
  --prefetch-size N           MLXDataStream prefetch (default: $PREFETCH_SIZE)
  --validate-every N          Validate every N epochs (default: $VALIDATE_EVERY)
  --no-fp16                   Disable FP16
  --no-mlx-data               Force PrefetchDataLoader
  --no-eval-sisdr             Disable SI-SDR during validation
  --vad-loss-weight VAL       VAD loss weight (default: $VAD_LOSS_WEIGHT)
  --vad-speech-loss-weight VAL  Speech-structure loss weight (default: $VAD_SPEECH_LOSS_WEIGHT)
  --mem-sample-interval SEC   Memory sampling interval (default: $MEM_SAMPLE_INTERVAL)
  --max-train-batches N       Cap training batches per epoch (default: $MAX_TRAIN_BATCHES)
  --max-valid-batches N       Cap validation batches (default: $MAX_VALID_BATCHES)
  --verbose                   Enable verbose training logs (-v)
  --no-skip-existing          Re-run even if results already exist
  --analysis-only             Do not launch new runs; only parse existing logs to refresh summaries
  --python PATH               Python interpreter (default: $PYTHON_BIN)
  --extra-arg ARG             Extra arg passed to train_dynamic.py (repeatable)
  -h, --help                  Show this help

Examples:
  $0 --cache-dir /data/cache
  $0 --cache-dir /data/cache --batch-sizes 64,128 --eval-frequencies 10,50
  $0 --cache-dir /data/cache --extra-arg --vad-loss-weight --extra-arg 0.05
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --checkpoint-root)
      CHECKPOINT_ROOT="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-sizes)
      BATCH_SIZES="$2"
      shift 2
      ;;
    --backbones)
      BACKBONES="$2"
      shift 2
      ;;
    --eval-frequencies)
      EVAL_FREQUENCIES="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --prefetch-size)
      PREFETCH_SIZE="$2"
      shift 2
      ;;
    --validate-every)
      VALIDATE_EVERY="$2"
      shift 2
      ;;
    --no-fp16)
      USE_FP16=0
      shift 1
      ;;
    --no-mlx-data)
      USE_MLX_DATA=0
      shift 1
      ;;
    --no-eval-sisdr)
      EVAL_SISDR=0
      shift 1
      ;;
    --vad-loss-weight)
      VAD_LOSS_WEIGHT="$2"
      shift 2
      ;;
    --vad-speech-loss-weight)
      VAD_SPEECH_LOSS_WEIGHT="$2"
      shift 2
      ;;
    --mem-sample-interval)
      MEM_SAMPLE_INTERVAL="$2"
      shift 2
      ;;
    --max-train-batches)
      MAX_TRAIN_BATCHES="$2"
      shift 2
      ;;
    --max-valid-batches)
      MAX_VALID_BATCHES="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift 1
      ;;
    --no-skip-existing)
      SKIP_EXISTING=0
      shift 1
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --analysis-only)
      ANALYSIS_ONLY=1
      shift 1
      ;;
    --extra-arg)
      EXTRA_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$CACHE_DIR" ]]; then
  echo "ERROR: --cache-dir is required (or set DFNET_CACHE_DIR)"
  usage
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter not found or not executable: $PYTHON_BIN"
  exit 1
fi

mkdir -p "$LOGS_DIR" "$CHECKPOINT_ROOT"

export DFNET_TQDM=1
export PYTHONUNBUFFERED=1

IFS=',' read -r -a BACKBONE_LIST <<< "${BACKBONES// /}"
IFS=',' read -r -a BATCH_LIST <<< "${BATCH_SIZES// /}"
IFS=',' read -r -a EVAL_LIST <<< "${EVAL_FREQUENCIES// /}"

COMMON_ARGS=(
  --cache-dir "$CACHE_DIR"
  --epochs "$EPOCHS"
  --validate-every "$VALIDATE_EVERY"
  --prefetch-size "$PREFETCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --vad-loss-weight "$VAD_LOSS_WEIGHT"
  --vad-speech-loss-weight "$VAD_SPEECH_LOSS_WEIGHT"
  --max-train-batches "$MAX_TRAIN_BATCHES"
  --max-valid-batches "$MAX_VALID_BATCHES"
)

if [[ "$USE_FP16" == "1" ]]; then
  COMMON_ARGS+=(--fp16)
else
  COMMON_ARGS+=(--no-fp16)
fi

if [[ "$USE_MLX_DATA" == "0" ]]; then
  COMMON_ARGS+=(--no-mlx-data)
fi

if [[ "$EVAL_SISDR" == "1" ]]; then
  COMMON_ARGS+=(--eval-sisdr)
fi

if [[ "$VERBOSE" == "1" ]]; then
  COMMON_ARGS+=(-v)
fi

echo "=== Backend quality benchmark ==="
echo "Run ID:        $RUN_ID"
echo "Cache dir:     $CACHE_DIR"
echo "Checkpoints:   $CHECKPOINT_ROOT"
echo "Backbones:     ${BACKBONE_LIST[*]}"
echo "Batch sizes:   ${BATCH_LIST[*]}"
echo "Eval freqs:    ${EVAL_LIST[*]}"
echo "Workers:       $NUM_WORKERS"
echo "Prefetch:      $PREFETCH_SIZE"
echo "Epochs:        $EPOCHS"
echo "Validate:      every $VALIDATE_EVERY epoch(s)"
echo "Eval SI-SDR:   $([[ \"$EVAL_SISDR\" == \"1\" ]] && echo on || echo off)"
echo "VAD loss:      $VAD_LOSS_WEIGHT"
echo "VAD speech:    $VAD_SPEECH_LOSS_WEIGHT"
echo "Train batches: ${MAX_TRAIN_BATCHES:-all}"
echo "Valid batches: ${MAX_VALID_BATCHES:-all}"
echo "Mem sample:    ${MEM_SAMPLE_INTERVAL}s"
echo "Analysis only: $ANALYSIS_ONLY"
echo

for backbone in "${BACKBONE_LIST[@]}"; do
  for batch_size in "${BATCH_LIST[@]}"; do
    for eval_freq in "${EVAL_LIST[@]}"; do
      run_name="${backbone}_bs${batch_size}_e${eval_freq}"
      ckpt_dir="$CHECKPOINT_ROOT/$RUN_ID/$run_name"
      log_file="$LOGS_DIR/backend_${RUN_ID}_${run_name}.log"
      marker_file="$ckpt_dir/bench_complete.json"

      if [[ "$SKIP_EXISTING" == "1" && -f "$marker_file" && "$ANALYSIS_ONLY" == "0" ]]; then
        echo "Skipping $run_name (already complete)"
        continue
      fi

      mkdir -p "$ckpt_dir"

      cmd=(
        "$PYTHON_BIN"
        -m DeepFilterNet.df_mlx.train_dynamic
        "${COMMON_ARGS[@]}"
        --eval-frequency "$eval_freq"
        --batch-size "$batch_size"
        --checkpoint-dir "$ckpt_dir"
        --save-strategy epoch
        --save-total-limit 3
        --checkpoint-batches 1000
        --backbone-type "$backbone"
        --resume
        --resume-data
      )

      if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
      fi

      if [[ "$ANALYSIS_ONLY" == "0" ]]; then
        echo "=== Running $run_name ==="
        echo "Command: ${cmd[*]}"
        echo "Log: $log_file"

        # Stream stderr (tqdm) to terminal and log; prefix stdout with run name.
        # Run in background so we can sample memory.
        start_ts="$(date +%s)"
        {
          "${cmd[@]}" \
            > >(sed -u "s/^/[$run_name] /" | tee -a "$log_file") \
            2> >(tee -a "$log_file" >&2)
        } &
        run_pid=$!

        max_rss_kb=0
        sum_rss_kb=0
        rss_samples=0

        while kill -0 "$run_pid" 2>/dev/null; do
          rss_kb="$(ps -o rss= -p "$run_pid" | tr -d ' ')"
          if [[ -n "$rss_kb" ]]; then
            sum_rss_kb=$((sum_rss_kb + rss_kb))
            rss_samples=$((rss_samples + 1))
            if [[ "$rss_kb" -gt "$max_rss_kb" ]]; then
              max_rss_kb="$rss_kb"
            fi
          fi
          sleep "$MEM_SAMPLE_INTERVAL"
        done

        wait "$run_pid" || true
        end_ts="$(date +%s)"
        elapsed_sec=$((end_ts - start_ts))

        avg_rss_mb="0"
        max_rss_mb="0"
        if [[ "$rss_samples" -gt 0 ]]; then
          avg_rss_mb="$(awk -v sum_kb="$sum_rss_kb" -v samples="$rss_samples" 'BEGIN {printf "%.1f", (sum_kb / samples) / 1024.0}')"
          max_rss_mb="$(awk -v max_kb="$max_rss_kb" 'BEGIN {printf "%.1f", max_kb / 1024.0}')"
        fi
      else
        echo "=== Analysis-only: parsing existing artifacts for $run_name ==="
        start_ts=0
        end_ts=0
        elapsed_sec=0
        avg_rss_mb=0
        max_rss_mb=0
      fi

      if [[ "$ANALYSIS_ONLY" == "1" ]]; then
        if [[ ! -f "$log_file" && ! -f "$marker_file" ]]; then
          echo "=== Analysis-only: missing artifacts for $run_name; skipping ==="
          continue
        fi
      fi

      # Parse results and write marker/summary.
      "$PYTHON_BIN" - "$log_file" "$marker_file" "$SUMMARY_JSONL" "$SUMMARY_CSV" \
        "$RUN_ID" "$run_name" "$backbone" "$batch_size" "$eval_freq" \
        "$avg_rss_mb" "$max_rss_mb" "$elapsed_sec" <<'PY'
import csv
import json
import re
import sys
from datetime import datetime

(
    log_path,
    marker_path,
    jsonl_path,
    csv_path,
    run_id,
    run_name,
    backbone,
    batch_size,
    eval_freq,
    avg_rss_mb,
    max_rss_mb,
    elapsed_sec,
) = sys.argv[1:]

metrics_line = None
epoch_line = None
throughput_vals = []
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "Validating metrics:" in line:
            metrics_line = line.strip()
        if "✓ Epoch" in line:
            epoch_line = line.strip()
            m = re.search(r"samples @\s+([0-9.]+)/s", line)
            if m:
                try:
                    throughput_vals.append(float(m.group(1)))
                except ValueError:
                    pass

metrics = {}
if metrics_line:
    parts = metrics_line.split(":", 1)[-1]
    for match in re.finditer(r"([a-zA-Z\-]+)=([-0-9.]+)(dB)?", parts):
        key = match.group(1)
        val = float(match.group(2))
        metrics[key] = val

epoch = {}
if epoch_line:
    m = re.search(r"Train: ([0-9.]+)", epoch_line)
    if m:
        epoch["train"] = float(m.group(1))
    m = re.search(r"Valid: ([0-9.]+)", epoch_line)
    if m:
        epoch["valid"] = float(m.group(1))
    m = re.search(r"Best: ([0-9.]+)", epoch_line)
    if m:
        epoch["best"] = float(m.group(1))

avg_throughput = sum(throughput_vals) // len(throughput_vals) if throughput_vals else None
max_throughput = max(throughput_vals) if throughput_vals else None

accuracy = None
accuracy_metric = None
if "si-sdr" in metrics:
    accuracy = metrics["si-sdr"]
    accuracy_metric = "si-sdr"
elif "valid" in epoch:
    accuracy = -epoch["valid"]
    accuracy_metric = "neg_valid_loss"

payload = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "run_id": run_id,
    "run_name": run_name,
    "backbone": backbone,
    "batch_size": int(batch_size),
    "eval_frequency": int(eval_freq),
    "metrics": metrics,
    "epoch": epoch,
    "avg_throughput": avg_throughput,
    "max_throughput": max_throughput,
    "avg_rss_mb": float(avg_rss_mb),
    "max_rss_mb": float(max_rss_mb),
    "elapsed_sec": int(elapsed_sec),
    "accuracy": accuracy,
    "accuracy_metric": accuracy_metric,
    "log": log_path,
}

with open(marker_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

with open(jsonl_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(payload) + "\n")

fieldnames = [
    "timestamp", "run_id", "run_name", "backbone", "batch_size", "eval_frequency",
    "train", "valid", "best", "spec", "vad", "speech", "resid", "p_ref", "p_out", "gate", "si-sdr",
    "avg_throughput", "max_throughput", "avg_rss_mb", "max_rss_mb", "elapsed_sec",
    "accuracy", "accuracy_metric",
]

row = {
    "timestamp": payload["timestamp"],
    "run_id": run_id,
    "run_name": run_name,
    "backbone": backbone,
    "batch_size": int(batch_size),
    "eval_frequency": int(eval_freq),
    "train": epoch.get("train"),
    "valid": epoch.get("valid"),
    "best": epoch.get("best"),
    "spec": metrics.get("spec"),
    "vad": metrics.get("vad"),
    "speech": metrics.get("speech"),
    "resid": metrics.get("resid"),
    "p_ref": metrics.get("p_ref"),
    "p_out": metrics.get("p_out"),
    "gate": metrics.get("gate"),
    "si-sdr": metrics.get("si-sdr"),
    "avg_throughput": avg_throughput,
    "max_throughput": max_throughput,
    "avg_rss_mb": float(avg_rss_mb),
    "max_rss_mb": float(max_rss_mb),
    "elapsed_sec": int(elapsed_sec),
    "accuracy": accuracy,
    "accuracy_metric": accuracy_metric,
}

write_header = False
try:
    with open(csv_path, "r", encoding="utf-8") as f:
        if f.readline().strip() != ",".join(fieldnames):
            write_header = True
except FileNotFoundError:
    write_header = True

with open(csv_path, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY

      echo "Saved summary: $marker_file"
      echo
    done
  done
done

echo "Benchmark complete."
echo "JSONL summary: $SUMMARY_JSONL"
echo "CSV summary:   $SUMMARY_CSV"
