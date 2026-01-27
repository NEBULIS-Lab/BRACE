#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs checkpoints data

NUM_SAMPLES=${1:-1000}

LOG_FILE="logs/stage1_$(date +%Y%m%d_%H%M%S).log"

echo "[Stage1] Running saliency collection with NUM_SAMPLES=${NUM_SAMPLES}" | tee "$LOG_FILE"
python -u src/stage1_saliency.py \
  --data_path "data/raw/dolly15k" \
  --output_path "checkpoints/saliency.pt" \
  --num_samples "${NUM_SAMPLES}" \
  2>&1 | tee -a "$LOG_FILE"
