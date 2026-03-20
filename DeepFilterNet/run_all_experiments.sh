#!/usr/bin/env bash
# run_all_experiments.sh — Sequential training + quality evaluation for E0–E3.
#
# Usage:
#   nohup bash run_all_experiments.sh > /tmp/dfn_experiments.log 2>&1 &
#
# Or in tmux/screen:
#   bash run_all_experiments.sh 2>&1 | tee /tmp/dfn_experiments.log
#
# Monitor progress:
#   bash monitor_experiments.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
DEEPFILTERNET="${REPO_ROOT}/DeepFilterNet"
CONFIGS="${DEEPFILTERNET}/df_mlx/configs/run_profiles"
LOG_DIR="/tmp/dfn_experiment_logs"

mkdir -p "$LOG_DIR"

echo "========================================"
echo " DeepFilterNet4 Experiment Runner"
echo " Started: $(date)"
echo " Repo: ${REPO_ROOT}"
echo "========================================"

# Verify Python and imports
"$PYTHON" -c "
import sys; sys.path.insert(0, '${DEEPFILTERNET}')
from df_mlx.train_dynamic import train
print('Training import OK')
" || { echo "FATAL: training import failed"; exit 1; }

run_experiment() {
    local name="$1"
    local config="$2"
    local log="${LOG_DIR}/${name}.log"

    echo ""
    echo "════════════════════════════════════════"
    echo " Starting: ${name}"
    echo " Config:   ${config}"
    echo " Log:      ${log}"
    echo " Time:     $(date)"
    echo "════════════════════════════════════════"

    PYTHONUNBUFFERED=1 "$PYTHON" -m df_mlx.train_dynamic \
        --run-config "${config}" \
        > "${log}" 2>&1

    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "  ✅ ${name} completed successfully at $(date)"
    else
        echo "  ❌ ${name} failed (exit code ${rc}) at $(date)"
        echo "  Check log: ${log}"
    fi
    return $rc
}

# ── Run experiments in order: fastest first ──────────────────────────
# E2 = No GAN (fastest, ~50-80 hrs)
# E1 = MPD-only (fast, ~100 hrs)
# E3 = Frozen disc (medium, GAN active until epoch 23)
# E0 = Full GAN baseline (slowest)

FAILED=()

run_experiment "E2-NoGAN-MRSTFT" "${CONFIGS}/mrstft_enhanced_no_gan.toml" || FAILED+=("E2")
run_experiment "E1-MPD-Only" "${CONFIGS}/mpd_only_reduced.toml" || FAILED+=("E1")
run_experiment "E3-Frozen-Disc" "${CONFIGS}/frozen_disc_fm.toml" || FAILED+=("E3")
run_experiment "E0-Baseline" "${CONFIGS}/run_pipeline_awesome_gan_silero_single_oom_safe.toml" || FAILED+=("E0")

echo ""
echo "════════════════════════════════════════"
echo " All training runs finished at $(date)"
echo "════════════════════════════════════════"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  ⚠️  Failed experiments: ${FAILED[*]}"
fi

# ── Run quality evaluation ──────────────────────────────────────────
echo ""
echo "Running quality evaluation..."
cd "${DEEPFILTERNET}"
PYTHONUNBUFFERED=1 "$PYTHON" evaluate_experiments.py \
    --n-samples 50 \
    --output "${LOG_DIR}/quality_results.json" \
    2>&1 | tee "${LOG_DIR}/quality_eval.log"

echo ""
echo "════════════════════════════════════════"
echo " Experiment suite complete at $(date)"
echo " Results: ${LOG_DIR}/quality_results.json"
echo " Logs:    ${LOG_DIR}/"
echo "════════════════════════════════════════"
