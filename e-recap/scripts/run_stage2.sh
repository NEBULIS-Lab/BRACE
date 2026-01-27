#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs checkpoints

LR=${1:-1e-4}
EPOCHS=${2:-2}

LOG_FILE="logs/stage2_$(date +%Y%m%d_%H%M%S).log"

echo "[Stage2] Training pruning modules with LR=${LR}, EPOCHS=${EPOCHS}" | tee "$LOG_FILE"
python3 -u src/stage2_pruning.py \
  --data_path "data/raw/dolly15k" \
  --saliency_path "checkpoints/saliency.pt" \
  --output_path "checkpoints/pruning_module.pt" \
  --learning_rate "${LR}" \
  --epochs "${EPOCHS}" \
  2>&1 | tee -a "$LOG_FILE"
