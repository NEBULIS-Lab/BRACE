#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/e-recap/src:${PYTHONPATH:-}"

MODEL="checkpoints/qwen2-7b-instruct"
DATA="data/raw/dolly15k"
SALIENCY="checkpoints/saliency.pt"
OUT="checkpoints/pruning_module.pt"
LR="1e-4"
EPOCHS="2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --saliency) SALIENCY="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: scripts/run_e_recap_stage2.sh [--model <hf_dir>] [--data <dataset_dir>] [--saliency <saliency.pt>] [--out <pruner.pt>] [--lr <float>] [--epochs <int>]

Defaults:
  --model     checkpoints/qwen2-7b-instruct
  --data      data/raw/dolly15k
  --saliency  checkpoints/saliency.pt
  --out       checkpoints/pruning_module.pt
  --lr        1e-4
  --epochs    2
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

python -u e-recap/src/stage2_pruning.py \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --saliency_path "$SALIENCY" \
  --output_path "$OUT" \
  --learning_rate "$LR" \
  --epochs "$EPOCHS"

