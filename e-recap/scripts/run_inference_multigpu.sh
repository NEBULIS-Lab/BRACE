#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Set PYTHONPATH to ensure imports work correctly
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

MODE=${1:-profile}
shift || true

if [ "$MODE" = "profile" ]; then
  echo "[Multi-GPU Inference] Profiling baseline vs E-RECAP (device_map=auto)"
  python3 -u src/inference_erecap_multigpu.py \
    --mode profile \
    --lengths 1024 2048 4096 8192 16384 32768 \
    "$@"

elif [ "$MODE" = "generate" ]; then
  PROMPT="$*"
  if [ -z "$PROMPT" ]; then
    PROMPT="Hello, E-RECAP (multi-GPU)!"
  fi
  echo "[Multi-GPU Inference] Generating baseline text"
  python3 -u src/inference_erecap_multigpu.py \
    --mode generate \
    --prompt "$PROMPT"

else
  echo "Unknown mode: $MODE"
  echo "Usage: $0 [profile|generate] [extra-args-or-prompt]"
  exit 1
fi
