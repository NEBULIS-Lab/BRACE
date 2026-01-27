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

# Best-effort extra tables (domain- or feature-specific).
DOMAIN="$(python - "$RUN_DIR" <<'PY' 2>/dev/null || true
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
obj = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
cfg = obj.get("config") or {}
print(cfg.get("domain") or "")
PY
)"

case "$DOMAIN" in
  robofactory*|robocasa*)
    python analysis/domainb_table.py "$RUN_DIR" --write_tables || echo "[WARN] domainb_table failed"
    ;;
  airsim*)
    python analysis/domainc_table.py "$RUN_DIR" --write_tables || echo "[WARN] domainc_table failed"
    ;;
  *)
    ;;
esac

python analysis/clarification_table.py "$RUN_DIR" --write_tables || echo "[WARN] clarification_table skipped"
python analysis/budget_match_table.py "$RUN_DIR" --write_tables || echo "[WARN] budget_match_table skipped"
python analysis/trigger_audit_table.py "$RUN_DIR" --write_tables
python analysis/trigger_field_coverage.py "$RUN_DIR" --write_tables
python analysis/controller_field_coverage.py "$RUN_DIR" --write_tables

echo "OK: postprocessed $RUN_DIR"
