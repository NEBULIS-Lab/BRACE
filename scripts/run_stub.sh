#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT="runs"
RUN_NAME="stub_smoke"
EPISODES="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: scripts/run_stub.sh [--run-name <name>] [--runs-root <dir>] [--episodes <n>]

Runs a tiny, dependency-free stub that only validates:
  - run directory creation (runs/<run_id>/)
  - events.jsonl / episode_metrics.jsonl schema
  - phase accounting (event_type="phase")

Tip: Postprocess into tables with:
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

python experiments/habitat/run_stub.py \
  --runs_root "$RUNS_ROOT" \
  --run_name "$RUN_NAME" \
  --episodes "$EPISODES"

