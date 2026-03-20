#!/usr/bin/env bash
# launch_remaining.sh — Run E1, E3, E0 sequentially, then evaluate all.
# Run this after E2 has completed.
#
# Usage:
#   nohup bash launch_remaining.sh > /tmp/dfn_remaining.log 2>&1 &
#   # or in tmux/screen:
#   bash launch_remaining.sh 2>&1 | tee /tmp/dfn_remaining.log
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
DFN="${REPO_ROOT}/DeepFilterNet"
CONFIGS="${DFN}/df_mlx/configs/run_profiles"
LOG_DIR="/tmp/dfn_experiment_logs"

mkdir -p "$LOG_DIR"

run_exp() {
    local name="$1" config="$2" log="${LOG_DIR}/${name}.log"
    echo ""
    echo "═══ ${name} — started $(date) ═══"
    PYTHONUNBUFFERED=1 "$PYTHON" -m df_mlx.train_dynamic \
        --run-config "${config}" > "${log}" 2>&1
    local rc=$?
    echo "═══ ${name} — finished $(date) (exit ${rc}) ═══"
    return $rc
}

cd "$DFN"

# Check if E2 is done
e2_final="/Users/andrew/DataDump/checkpoints/e2_mrstft_enhanced_no_gan/final.safetensors"
if [ ! -f "$e2_final" ]; then
    e2_pid=$(pgrep -f "mrstft_enhanced_no_gan" || true)
    if [ -n "$e2_pid" ]; then
        echo "E2 still running (PID ${e2_pid}). Waiting..."
        while kill -0 "$e2_pid" 2>/dev/null; do sleep 60; done
        echo "E2 process finished."
    else
        echo "Warning: E2 may not have finished — no final.safetensors and no process."
    fi
fi

FAILED=()
run_exp "E1-MPD-Only" "${CONFIGS}/mpd_only_reduced.toml" || FAILED+=("E1")
run_exp "E3-Frozen-Disc" "${CONFIGS}/frozen_disc_fm.toml" || FAILED+=("E3")
run_exp "E0-Baseline" "${CONFIGS}/run_pipeline_awesome_gan_silero_single_oom_safe.toml" || FAILED+=("E0")

echo ""
echo "All training done. Failed: ${FAILED[*]:-none}"
echo ""

echo "Running quality evaluation..."
PYTHONUNBUFFERED=1 "$PYTHON" evaluate_experiments.py \
    --n-samples 50 \
    --output "${LOG_DIR}/quality_results.json" \
    2>&1 | tee "${LOG_DIR}/quality_eval.log"

echo ""
echo "Complete at $(date). Results: ${LOG_DIR}/quality_results.json"
