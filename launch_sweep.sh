#!/bin/bash
# Launch a grid sweep over model variants and learning rates.
# Each combination is submitted as an independent SLURM job.
#
# Usage:
#   bash launch_sweep.sh
#   bash launch_sweep.sh --dry-run    # print commands without submitting

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Sweep grid ─────────────────────────────────────────────────────────────
DATA_DIR="/home/zong/cv-dataset/test_data_scientist"
OUTPUT_DIR="/home/zong/cv-dataset/output"

MODEL_VALUES=(b1, b2)
BATCH_SIZE_VALUES=(32)
LR_VALUES=(1e-3 3e-4 1e-4)
LR_BACKBONE_VALUES=(1e-5, 0)     # 0 = frozen backbone
EPOCHS_VALUES=(30)

# ── Submit one job per combination ─────────────────────────────────────────
for MODEL in "${MODEL_VALUES[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZE_VALUES[@]}"; do
        for LR in "${LR_VALUES[@]}"; do
            for LR_BB in "${LR_BACKBONE_VALUES[@]}"; do
                for EPOCHS in "${EPOCHS_VALUES[@]}"; do
                    JOB_NAME="crack_${MODEL}_bs${BATCH_SIZE}_lr${LR}_lrbb${LR_BB}_ep${EPOCHS}"

                    CMD=(
                        sbatch
                        --job-name="$JOB_NAME"
                        --export="ALL,MODEL=$MODEL,BATCH_SIZE=$BATCH_SIZE,LR=$LR,LR_BACKBONE=$LR_BB,EPOCHS=$EPOCHS,DATA_DIR=$DATA_DIR,OUTPUT_DIR=$OUTPUT_DIR"
                        job.sh
                    )

                    if $DRY_RUN; then
                        echo "[DRY-RUN] ${CMD[*]}"
                    else
                        "${CMD[@]}"
                        echo "Submitted: $JOB_NAME"
                    fi
                done
            done
        done
    done
done

echo ""
echo "Total jobs: $(( ${#MODEL_VALUES[@]} * ${#BATCH_SIZE_VALUES[@]} * ${#LR_VALUES[@]} * ${#LR_BACKBONE_VALUES[@]} * ${#EPOCHS_VALUES[@]} ))"
