#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/e-recap/src:${PYTHONPATH:-}"

MODEL="checkpoints/qwen2-7b-instruct"
DATA="data/raw/dolly15k"
OUT="checkpoints/saliency.pt"
NUM_SAMPLES="1000"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: scripts/run_e_recap_stage1.sh [--model <hf_dir>] [--data <dataset_dir>] [--out <saliency.pt>] [--num-samples <n>]

Defaults:
  --model        checkpoints/qwen2-7b-instruct
  --data         data/raw/dolly15k
  --out          checkpoints/saliency.pt
  --num-samples  1000
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

python -u e-recap/src/stage1_saliency.py \
  --model_path "$MODEL" \
  --data_path "$DATA" \
  --out_path "$OUT" \
  --num_samples "$NUM_SAMPLES"

