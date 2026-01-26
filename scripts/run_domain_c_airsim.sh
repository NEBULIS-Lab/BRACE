#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/airsim/domainc_multidrone_demo.json"
RUNS_ROOT="runs"
RUN_NAME="airsim_demo"
UE_ENV="airsimnh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --ue-env) UE_ENV="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: scripts/run_domain_c_airsim.sh [--config <json>] [--run-name <name>] [--runs-root <dir>] [--ue-env airsimnh|blocks|...]

Requires: AirSim UE binaries and BRACE_AIRSIM_ENVS_ROOT set.
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

python experiments/airsim/run_domainc.py \
  --config "$CONFIG" \
  --runs_root "$RUNS_ROOT" \
  --run_name "$RUN_NAME" \
  --ue_env "$UE_ENV"

