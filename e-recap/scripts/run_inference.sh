#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Set PYTHONPATH to ensure imports work correctly
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

MODE=${1:-profile}
BENCHMARK_MODE=${2:-prefill}  # prefill or end2end

if [ "$MODE" = "profile" ]; then
  echo "=========================================="
  echo "[Inference] Running all configurations"
  echo "Benchmark Mode: $BENCHMARK_MODE"
  echo "=========================================="
  echo ""
  
  # Run keep09 configuration
  echo "[Config: keep09] Profiling baseline vs E-RECAP (keep_ratio=0.9)"
  echo "----------------------------------------"
  python3 -u src/inference_erecap.py \
    --mode profile \
    --config keep09 \
    --benchmark_mode "$BENCHMARK_MODE" \
    --lengths 1024 2048 4096 8192 16384 32768 \
    "${@:3}"
  echo ""
  
  # Run keep08 configuration
  echo "[Config: keep08] Profiling baseline vs E-RECAP (keep_ratio=0.8)"
  echo "----------------------------------------"
  python3 -u src/inference_erecap.py \
    --mode profile \
    --config keep08 \
    --benchmark_mode "$BENCHMARK_MODE" \
    --lengths 1024 2048 4096 8192 16384 32768 \
    "${@:3}"
  echo ""
  
  # Run keep07 configuration
  echo "[Config: keep07] Profiling baseline vs E-RECAP (keep_ratio=0.7)"
  echo "----------------------------------------"
  python3 -u src/inference_erecap.py \
    --mode profile \
    --config keep07 \
    --benchmark_mode "$BENCHMARK_MODE" \
    --lengths 1024 2048 4096 8192 16384 32768 \
    "${@:3}"
  echo ""
  
  echo "=========================================="
  echo "[OK] All configurations completed!"
  echo "=========================================="
  
elif [ "$MODE" = "generate" ]; then
  shift
  PROMPT="$*"
  if [ -z "$PROMPT" ]; then
    PROMPT="Hello, E-RECAP! Please introduce yourself."
  fi
  echo "[Inference] Generating text with baseline model"
  python3 -u src/inference_erecap.py \
    --mode generate \
    --prompt "$PROMPT"
else
  echo "Unknown mode: $MODE"
  echo "Usage: $0 [profile|generate] [benchmark_mode] [extra-args...]"
  echo ""
  echo "  profile (default): Run keep09, keep08, and keep07 configurations"
  echo "    benchmark_mode: 'prefill' (default) or 'end2end'"
  echo "  generate: Generate text with the model"
  echo ""
  echo "Examples:"
  echo "  $0 profile prefill              # Run prefill-only benchmarks"
  echo "  $0 profile end2end               # Run end-to-end benchmarks"
  echo "  $0 generate 'Your prompt here'"
  exit 1
fi
