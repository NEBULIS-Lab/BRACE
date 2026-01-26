#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  cat <<EOF
Usage: scripts/postprocess_run.sh runs/<run_id>

Runs strict schema checks + aggregation + trigger/controller coverage.
Outputs append-only markdown tables under artifacts/tables/.
EOF
  exit 0
fi

RUN_DIR="$1"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: run dir not found: $RUN_DIR" >&2
  exit 2
fi

python analysis/schema_check.py "$RUN_DIR" --strict
python analysis/aggregate_runs.py "$RUN_DIR" --write_tables
python analysis/trigger_audit_table.py "$RUN_DIR" --write_tables
python analysis/trigger_field_coverage.py "$RUN_DIR" --write_tables
python analysis/controller_field_coverage.py "$RUN_DIR" --write_tables

echo "OK: postprocessed $RUN_DIR"

