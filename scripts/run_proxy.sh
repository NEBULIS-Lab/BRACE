#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/smoke/proxy_controller.json"
RUNS_ROOT="runs"
RUN_NAME="proxy_smoke"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: scripts/run_proxy.sh [--config <json>] [--run-name <name>] [--runs-root <dir>]

Runs the synthetic proxy runner (no external simulators required) and writes:
  - runs/<run_id>/*
  - artifacts/tables/<run_id>.md (small proxy summary table)

Tip: You can also run the generic postprocess pipeline:
  scripts/postprocess_run.sh runs/<run_id>
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

python experiments/proxy/brace_controller_proxy_runner.py \
  --config "$CONFIG" \
  --runs_root "$RUNS_ROOT" \
  --run_name "$RUN_NAME"

