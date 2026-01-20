#!/usr/bin/env bash
#
# run_curves.sh
#
# Run short dynamic-training ablations (10 epochs) and save summary metrics to JSON.
# Safe to re-run: each run auto-resumes and parsing is idempotent.
#
# Usage:
#   ./scripts/run_curves.sh --cache-dir /path/to/cache --checkpoint-root /path/to/ckpts/curves
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"
PYTHON_BIN="${PYTHON_BIN:-python}"
export DFNET_TQDM=1
export PYTHONUNBUFFERED=1

# Defaults (override via env or CLI)
CACHE_DIR="${CACHE_DIR:-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints/curves}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BACKBONE_TYPE="${BACKBONE_TYPE:-attention}"
PREFETCH_SIZE="${PREFETCH_SIZE:-64}"
USE_FP16="${USE_FP16:-1}"
VERBOSE="${VERBOSE:-1}"

SNR_MIN="${SNR_MIN:--5}"
SNR_MAX="${SNR_MAX:-40}"
SNR_EXTREME_MIN="${SNR_EXTREME_MIN:--20}"
SNR_EXTREME_MAX="${SNR_EXTREME_MAX:--5}"
P_EXTREME_SNR="${P_EXTREME_SNR:-0.1}"
SPEECH_GAIN_MIN="${SPEECH_GAIN_MIN:--12}"
SPEECH_GAIN_MAX="${SPEECH_GAIN_MAX:-12}"
NOISE_GAIN_MIN="${NOISE_GAIN_MIN:--12}"
NOISE_GAIN_MAX="${NOISE_GAIN_MAX:-12}"

VAD_THRESHOLD="${VAD_THRESHOLD:-0.6}"
VAD_MARGIN="${VAD_MARGIN:-0.05}"
VAD_WARMUP_EPOCHS="${VAD_WARMUP_EPOCHS:-5}"
VAD_SNR_GATE="${VAD_SNR_GATE:--10}"
VAD_SNR_GATE_WIDTH="${VAD_SNR_GATE_WIDTH:-6}"
VAD_BAND_LOW="${VAD_BAND_LOW:-300}"
VAD_BAND_HIGH="${VAD_BAND_HIGH:-3400}"
VAD_Z_THRESHOLD="${VAD_Z_THRESHOLD:-0.0}"
VAD_Z_SLOPE="${VAD_Z_SLOPE:-1.0}"

SUMMARY_JSON="${SUMMARY_JSON:-$LOGS_DIR/curves_summary.json}"

usage() {
  cat <<EOF
Usage: $0 --cache-dir PATH [--checkpoint-root PATH] [--epochs N] [--batch-size N]

Options:
  --cache-dir PATH         Required: dynamic audio cache dir
  --checkpoint-root PATH   Root directory for ablation checkpoints (default: $CHECKPOINT_ROOT)
  --epochs N               Epochs per run (default: $EPOCHS)
  --batch-size N           Batch size (default: $BATCH_SIZE)
  --snr-range MIN MAX      Base SNR range (default: $SNR_MIN $SNR_MAX)
  --snr-range-extreme MIN MAX  Extreme SNR range (default: $SNR_EXTREME_MIN $SNR_EXTREME_MAX)
  --p-extreme-snr P        Extreme SNR sampling probability (default: $P_EXTREME_SNR)
  --speech-gain-range MIN MAX  Speech gain range (default: $SPEECH_GAIN_MIN $SPEECH_GAIN_MAX)
  --noise-gain-range MIN MAX   Noise gain range (default: $NOISE_GAIN_MIN $NOISE_GAIN_MAX)
  --backbone-type NAME     Backbone type (default: $BACKBONE_TYPE)
  --no-fp16                Disable FP16
  --no-verbose             Disable verbose logging
  -h, --help               Show this help
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
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --snr-range)
      SNR_MIN="$2"
      SNR_MAX="$3"
      shift 3
      ;;
    --snr-range-extreme)
      SNR_EXTREME_MIN="$2"
      SNR_EXTREME_MAX="$3"
      shift 3
      ;;
    --p-extreme-snr)
      P_EXTREME_SNR="$2"
      shift 2
      ;;
    --speech-gain-range)
      SPEECH_GAIN_MIN="$2"
      SPEECH_GAIN_MAX="$3"
      shift 3
      ;;
    --noise-gain-range)
      NOISE_GAIN_MIN="$2"
      NOISE_GAIN_MAX="$3"
      shift 3
      ;;
    --backbone-type)
      BACKBONE_TYPE="$2"
      shift 2
      ;;
    --no-fp16)
      USE_FP16=0
      shift 1
      ;;
    --no-verbose)
      VERBOSE=0
      shift 1
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
  echo "ERROR: --cache-dir is required"
  usage
  exit 1
fi

mkdir -p "$LOGS_DIR" "$CHECKPOINT_ROOT"

COMMON_ARGS=(
  --cache-dir "$CACHE_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --checkpoint-batches 1000
  --save-strategy epoch
  --save-total-limit 5
  --prefetch-size "$PREFETCH_SIZE"
  --snr-range "$SNR_MIN" "$SNR_MAX"
  --snr-range-extreme "$SNR_EXTREME_MIN" "$SNR_EXTREME_MAX"
  --p-extreme-snr "$P_EXTREME_SNR"
  --speech-gain-range "$SPEECH_GAIN_MIN" "$SPEECH_GAIN_MAX"
  --noise-gain-range "$NOISE_GAIN_MIN" "$NOISE_GAIN_MAX"
  --vad-threshold "$VAD_THRESHOLD"
  --vad-margin "$VAD_MARGIN"
  --vad-warmup-epochs "$VAD_WARMUP_EPOCHS"
  --vad-snr-gate "$VAD_SNR_GATE"
  --vad-snr-gate-width "$VAD_SNR_GATE_WIDTH"
  --vad-band-low "$VAD_BAND_LOW"
  --vad-band-high "$VAD_BAND_HIGH"
  --vad-z-threshold "$VAD_Z_THRESHOLD"
  --vad-z-slope "$VAD_Z_SLOPE"
  --backbone-type "$BACKBONE_TYPE"
  --eval-sisdr
  --resume
  --resume-data
)

if [[ "$USE_FP16" == "1" ]]; then
  COMMON_ARGS+=(--fp16)
else
  COMMON_ARGS+=(--no-fp16)
fi

if [[ "$VERBOSE" == "1" ]]; then
  COMMON_ARGS+=(-v)
fi

RUN_NAMES=("baseline" "vad_low" "vad_mid" "vad_speech")
RUN_VAD_W=("0" "0.02" "0.05" "0.05")
RUN_SPEECH_W=("0" "0" "0" "0.03")
RUN_TOTAL=${#RUN_NAMES[@]}

for idx in "${!RUN_NAMES[@]}"; do
  name="${RUN_NAMES[$idx]}"
  vad_w="${RUN_VAD_W[$idx]}"
  speech_w="${RUN_SPEECH_W[$idx]}"
  ckpt_dir="$CHECKPOINT_ROOT/$name"
  log_file="$LOGS_DIR/curves_${name}.log"
  run_no=$((idx + 1))
  ts="$(date +"%Y-%m-%d %H:%M:%S")"

  mkdir -p "$ckpt_dir"
  echo "============================================================"
  echo "[$ts] Run $run_no/$RUN_TOTAL: $name | vad_w=$vad_w | speech_w=$speech_w"
  echo "Checkpoint: $ckpt_dir"
  echo "Log: $log_file"

  if [[ -f "$ckpt_dir/final.safetensors" ]]; then
    echo "  Final checkpoint exists; skipping training."
  else
    cmd=(
      "$PYTHON_BIN" -u -m DeepFilterNet.df_mlx.train_dynamic
      --checkpoint-dir "$ckpt_dir"
      --vad-loss-weight "$vad_w"
      --vad-speech-loss-weight "$speech_w"
      "${COMMON_ARGS[@]}"
    )

    set +e
    "${cmd[@]}" \
      2> >(tee -a "$log_file" >&2) \
      | awk -v prefix="[$name] " '{print prefix $0; fflush();}' \
      | tee -a "$log_file"
    status=$?
    set -e

    if [[ $status -ne 0 ]]; then
      echo "Run exited non-zero (status=$status). Re-run to resume."
      # Still parse any new metrics collected so far.
    fi
  fi

  "$PYTHON_BIN" - <<PY
import json
import os
import re
from datetime import datetime

log_path = r"$log_file"
summary_path = r"$SUMMARY_JSON"
run_name = r"$name"
ckpt_dir = r"$ckpt_dir"
vad_w = float(r"$vad_w")
speech_w = float(r"$speech_w")

epoch_re = re.compile(
    r"Epoch (\\d+)/(\\d+) complete \\| Train: ([0-9.]+)"
    r"(?: \\| Spec: ([0-9.]+) \\| VAD: ([0-9.]+) \\| Speech: ([0-9.]+))?"
    r" \\| Valid: ([0-9.]+)"
)

metrics_re = re.compile(r"metrics: (.*)$")

epochs = []
val_metrics = []

if os.path.exists(log_path):
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            m = epoch_re.search(line)
            if m:
                epochs.append(
                    {
                        "epoch": int(m.group(1)),
                        "total_epochs": int(m.group(2)),
                        "train_loss": float(m.group(3)),
                        "train_spec": float(m.group(4)) if m.group(4) else None,
                        "train_vad": float(m.group(5)) if m.group(5) else None,
                        "train_speech": float(m.group(6)) if m.group(6) else None,
                        "valid_loss": float(m.group(7)),
                    }
                )
                continue
            m = metrics_re.search(line)
            if m:
                parts = [p.strip() for p in m.group(1).split("|")]
                metrics = {}
                for part in parts:
                    if "=" not in part:
                        continue
                    key, val = part.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if val.endswith("dB"):
                        val = val[:-2]
                    if val.endswith("%"):
                        val = val[:-1]
                    try:
                        metrics[key] = float(val)
                    except ValueError:
                        pass
                val_metrics.append(metrics)

# attach validation metrics by order
for i, ep in enumerate(epochs):
    if i < len(val_metrics):
        for k, v in val_metrics[i].items():
            ep[f"val_{k}"] = v

summary = {}
if os.path.exists(summary_path):
    with open(summary_path, "r") as f:
        summary = json.load(f)

summary.setdefault("runs", {})
summary["runs"][run_name] = {
    "updated_at": datetime.utcnow().isoformat() + "Z",
    "config": {
        "vad_loss_weight": vad_w,
        "vad_speech_loss_weight": speech_w,
    },
    "completed": os.path.exists(os.path.join(ckpt_dir, "final.safetensors")),
    "epochs": epochs,
}

os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Updated summary: {summary_path}")
PY
done

echo "============================================================"
echo "Curve summary JSON: $SUMMARY_JSON"
