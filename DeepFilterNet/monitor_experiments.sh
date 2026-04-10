#!/usr/bin/env bash
# monitor_experiments.sh — Check progress of running experiment training.
#
# Usage: bash monitor_experiments.sh
#
set -euo pipefail

LOG_DIR="/tmp/dfn_experiment_logs"
CKPT_BASE="/Users/andrew/DataDump/checkpoints"

declare -A CKPT_DIRS=(
    ["E0"]="${CKPT_BASE}/gan_from_40"
    ["E1"]="${CKPT_BASE}/e1_mpd_only_reduced"
    ["E2"]="${CKPT_BASE}/e2_mrstft_enhanced_no_gan"
    ["E3"]="${CKPT_BASE}/e3_frozen_disc_fm"
)

declare -A LOG_FILES=(
    ["E0"]="${LOG_DIR}/E0-Baseline.log"
    ["E1"]="${LOG_DIR}/E1-MPD-Only.log"
    ["E2"]="${LOG_DIR}/E2-NoGAN-MRSTFT.log"
    ["E3"]="${LOG_DIR}/E3-Frozen-Disc.log"
)

echo "════════════════════════════════════════════════════════════"
echo " DeepFilterNet4 Experiment Progress — $(date)"
echo "════════════════════════════════════════════════════════════"

# Check if training is running
running=$(ps aux | grep "df_mlx.train_dynamic" | grep -v grep | wc -l | tr -d ' ')
echo "  Training processes running: ${running}"
if [ "$running" -gt 0 ]; then
    ps aux | grep "df_mlx.train_dynamic" | grep -v grep | awk '{printf "    PID %s CPU=%s%% MEM=%s%%\n", $2, $3, $4}'
fi

echo ""
echo "── Checkpoint Status ──────────────────────────────────────"
for exp in E2 E1 E3 E0; do
    dir="${CKPT_DIRS[$exp]}"
    if [ ! -d "$dir" ]; then
        echo "  ${exp}: (not started)"
        continue
    fi

    # Count epoch checkpoint files
    n_epochs=$(ls "${dir}"/epoch_*.safetensors 2>/dev/null | grep -v state | grep -v complete | wc -l | tr -d ' ')
    has_best=$([ -f "${dir}/best.safetensors" ] && echo "yes" || echo "no")
    has_final=$([ -f "${dir}/final.safetensors" ] && echo "yes" || echo "no")

    # Get data checkpoint info
    data_ckpt="${dir}/data_checkpoint.json"
    if [ -f "$data_ckpt" ]; then
        epoch_idx=$(python3 -c "import json; d=json.load(open('${data_ckpt}')); print(d.get('epoch',0))" 2>/dev/null || echo "?")
        batch_idx=$(python3 -c "import json; d=json.load(open('${data_ckpt}')); print(d.get('batch_idx',0))" 2>/dev/null || echo "?")
        echo "  ${exp}: epoch ${epoch_idx}, batch ${batch_idx} | epochs_saved=${n_epochs} best=${has_best} final=${has_final}"
    else
        echo "  ${exp}: ${n_epochs} epoch checkpoints | best=${has_best} final=${has_final}"
    fi
done

echo ""
echo "── Log Tail ─────────────────────────────────────────────"
for exp in E2 E1 E3 E0; do
    log="${LOG_FILES[$exp]}"
    if [ -f "$log" ]; then
        last_line=$(grep -E "Epoch|loss|step|Complete|Error|FATAL" "$log" 2>/dev/null | grep -v "^$" | tail -1)
        size=$(du -sh "$log" | cut -f1)
        echo "  ${exp} (${size}): ${last_line:-<no progress lines yet>}"
    fi
done

echo ""
echo "── Quality Results ──────────────────────────────────────"
results="${LOG_DIR}/quality_results.json"
if [ -f "$results" ]; then
    echo "  Results available: ${results}"
    python3 -c "
import json
with open('${results}') as f:
    data = json.load(f)
for k, v in sorted(data.items()):
    if 'error' in v:
        print(f'  {k}: ERROR: {v[\"error\"]}')
    else:
        print(f'  {k}: PESQ={v[\"pesq_mean\"]:.3f} SI-SDR={v[\"sisdr_mean\"]:.2f}dB STOI={v[\"stoi_mean\"]:.4f}')
" 2>/dev/null || echo "  (parse error)"
else
    echo "  Not yet available"
fi

echo "════════════════════════════════════════════════════════════"
