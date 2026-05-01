#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="Lift"
ROBOTS=("Panda")
EPISODES="1"
STEPS="300"
SEED="0"
OUT_DIR="artifacts/robosuite_demos"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --robots)
      shift
      ROBOTS=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do ROBOTS+=("$1"); shift; done
      ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: scripts/run_robosuite_collect.sh [--env <name>] [--robots <r1> <r2> ...] [--episodes <n>] [--out-dir <dir>]
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

python experiments/robosuite/collect_random_demos_npz.py \
  --env "$ENV_NAME" \
  --robots "${ROBOTS[@]}" \
  --episodes "$EPISODES" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR"
