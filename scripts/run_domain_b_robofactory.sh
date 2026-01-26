#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/smoke/robofactory_lift_barrier.json"
RUNS_ROOT="runs"
RUN_NAME="robofactory_smoke"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: scripts/run_domain_b_robofactory.sh [--config <json>] [--run-name <name>] [--runs-root <dir>]

Requires: a working RoboFactory workspace + assets.
Env:
  BRACE_ROBOFACTORY_DATA_ROOT (recommended): where OpenMARL run_dir/checkpoints/caches live.
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

python experiments/robofactory/run_tasks.py \
  --config "$CONFIG" \
  --runs_root "$RUNS_ROOT" \
  --run_name "$RUN_NAME"
