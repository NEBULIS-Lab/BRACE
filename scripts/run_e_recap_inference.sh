#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/e-recap/src:${PYTHONPATH:-}"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  cat <<EOF
Usage: scripts/run_e_recap_inference.sh [args...]

Wrapper for: python -u e-recap/src/inference_erecap.py

Example:
  scripts/run_e_recap_inference.sh --mode profile --config keep07 \\
    --model_path checkpoints/qwen2-7b-instruct \\
    --pruning_ckpt checkpoints/pruning_module.pt
EOF
  exit 0
fi

python -u e-recap/src/inference_erecap.py "$@"

