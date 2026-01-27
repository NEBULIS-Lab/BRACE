#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] RoboFactory smoke"
scripts/run_robofactory.sh --run-name rf_smoke

echo "[2/3] Habitat smoke (requires habitat-setup)"
scripts/run_habitat.sh --run-name habitat_smoke

echo "[3/3] AirSim smoke (requires BRACE_AIRSIM_ENVS_ROOT)"
scripts/run_airsim.sh --run-name airsim_smoke --ue-env blocks

echo "OK: smoke_all done"
