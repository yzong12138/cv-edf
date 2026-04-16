#!/bin/bash
#SBATCH --job-name=crack
#SBATCH --partition=electronic               # adjust to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zong/cv-edf/logs/%x_%j.out
#SBATCH --error=/home/zong/cv-edf/logs/%x_%j.err

# ── Environment ────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="/home/zong/cv-edf"
mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"

# ── Parameters (overridable via --export on sbatch command line) ───────────
MODEL="${MODEL:-b0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
EPOCHS="${EPOCHS:-30}"
DATA_DIR="${DATA_DIR:-/home/zong/cv-dataset/test_data_scientist}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zong/cv-dataset/output}"

# Rename the job now that all variables are resolved
JOB_NAME="crack_${MODEL}_bs${BATCH_SIZE}_lr${LR}_lrbb${LR_BACKBONE}_ep${EPOCHS}"
scontrol update JobId="$SLURM_JOB_ID" Name="$JOB_NAME"

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Model      : efficientnet-$MODEL"
echo "Batch size : $BATCH_SIZE"
echo "LR (head)  : $LR"
echo "LR (bb)    : $LR_BACKBONE"
echo "Epochs     : $EPOCHS"
echo "=========================================="

uv run python main.py \
    --model        "$MODEL"       \
    --batch-size   "$BATCH_SIZE"  \
    --lr           "$LR"          \
    --lr-backbone  "$LR_BACKBONE" \
    --epochs       "$EPOCHS"      \
    --data-dir     "$DATA_DIR"    \
    --output-dir   "$OUTPUT_DIR"
