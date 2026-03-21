#!/usr/bin/env bash
# run_all_experiments.sh — Sequential training + quality evaluation for E0–E3.
#
# Automatically resumes partially-completed experiments from their latest
# checkpoint. Already-finished experiments (with final.safetensors) are skipped.
#
# Usage (standalone terminal, VS Code not needed):
#   cd ~/zRepos/DeepFilterNet/DeepFilterNet
#   nohup bash run_all_experiments.sh > /tmp/dfn_experiments.log 2>&1 &
#
# Monitor progress:
#   bash monitor_experiments.sh
#   # or: cat /Users/andrew/DataDump/checkpoints/<exp>/data_checkpoint.json
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
DEEPFILTERNET="${REPO_ROOT}/DeepFilterNet"
CONFIGS="${DEEPFILTERNET}/df_mlx/configs/run_profiles"
LOG_DIR="/tmp/dfn_experiment_logs"
CKPT_BASE="/Users/andrew/DataDump/checkpoints"

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

is_finished() {
    local ckpt_dir="$1"
    [ -f "${ckpt_dir}/final.safetensors" ]
}

has_checkpoint() {
    local ckpt_dir="$1"
    ls "${ckpt_dir}"/epoch_*.safetensors "${ckpt_dir}"/interrupted_epoch_*.safetensors "${ckpt_dir}"/best.safetensors 2>/dev/null | grep -q .
}

run_experiment() {
    local name="$1"
    local config="$2"
    local ckpt_dir="$3"
    local log="${LOG_DIR}/${name}.log"

    # Skip if already finished
    if is_finished "$ckpt_dir"; then
        echo ""
        echo "  ⏭  ${name}: already finished (final.safetensors exists), skipping"
        return 0
    fi

    # Build resume flags if partial checkpoint exists
    local resume_flags=""
    if has_checkpoint "$ckpt_dir"; then
        resume_flags="--resume --resume-data"
        echo ""
        echo "════════════════════════════════════════"
        echo " RESUMING: ${name}"
        echo " Config:   ${config}"
        echo " Ckpt dir: ${ckpt_dir}"
        echo " Log:      ${log}"
        echo " Time:     $(date)"
        echo "════════════════════════════════════════"
    else
        echo ""
        echo "════════════════════════════════════════"
        echo " Starting: ${name} (fresh)"
        echo " Config:   ${config}"
        echo " Log:      ${log}"
        echo " Time:     $(date)"
        echo "════════════════════════════════════════"
    fi

    # shellcheck disable=SC2086
    PYTHONUNBUFFERED=1 "$PYTHON" -m df_mlx.train_dynamic \
        --run-config "${config}" \
        ${resume_flags} \
        >> "${log}" 2>&1

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
# E2 = No GAN (fastest)
# E1 = MPD-only
# E3 = Frozen disc (GAN active until epoch 23)
# E0 = Full GAN baseline (slowest)

FAILED=()

run_experiment "E2-NoGAN-MRSTFT" \
    "${CONFIGS}/mrstft_enhanced_no_gan.toml" \
    "${CKPT_BASE}/e2_mrstft_enhanced_no_gan" || FAILED+=("E2")

run_experiment "E1-MPD-Only" \
    "${CONFIGS}/mpd_only_reduced.toml" \
    "${CKPT_BASE}/e1_mpd_only_reduced" || FAILED+=("E1")

run_experiment "E3-Frozen-Disc" \
    "${CONFIGS}/frozen_disc_fm.toml" \
    "${CKPT_BASE}/e3_frozen_disc_fm" || FAILED+=("E3")

run_experiment "E0-Baseline" \
    "${CONFIGS}/run_pipeline_awesome_gan_silero_single_oom_safe.toml" \
    "${CKPT_BASE}/gan_from_40" || FAILED+=("E0")

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
