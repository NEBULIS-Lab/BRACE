#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/habitat_setup_smoke.json"
RUNS_ROOT="runs"
RUN_NAME="habitat_smoke"
HABITAT_SETUP_ROOT="${HABITAT_SETUP_ROOT:-habitat-setup}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --habitat-setup-root) HABITAT_SETUP_ROOT="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: scripts/run_domain_a_habitat.sh [--config <json>] [--run-name <name>] [--runs-root <dir>] [--habitat-setup-root <dir>]

Requires: a working Habitat environment + a local habitat-setup checkout.
Env:
  BRACE_HABITAT_PY (optional): python executable for Habitat env
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

python experiments/habitat/run_habitat_setup_real.py \
  --config "$CONFIG" \
  --runs_root "$RUNS_ROOT" \
  --run_name "$RUN_NAME" \
  --habitat_setup_root "$HABITAT_SETUP_ROOT"

