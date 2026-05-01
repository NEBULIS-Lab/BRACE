#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_NAME="libero_10"
TASK_ID="0"
BDDL_FILE=""
DEMO_FILE=""
OUT="artifacts/libero/task.png"
LIBERO_ROOT="${BRACE_LIBERO_ROOT:-${LIBERO_ROOT:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark-name) BENCHMARK_NAME="$2"; shift 2 ;;
    --task-id) TASK_ID="$2"; shift 2 ;;
    --bddl-file) BDDL_FILE="$2"; shift 2 ;;
    --demo-file) DEMO_FILE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --libero-root) LIBERO_ROOT="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: scripts/run_libero_render.sh --benchmark-name <suite> --task-id <id> --bddl-file <file> --demo-file <file> [--out <png>]

Requires:
  - a working LIBERO installation in the active Python env, or
  - BRACE_LIBERO_ROOT / LIBERO_ROOT pointing to a LIBERO workspace
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BDDL_FILE" || -z "$DEMO_FILE" ]]; then
  echo "--bddl-file and --demo-file are required" >&2
  exit 2
fi

python experiments/libero/render_single_task.py \
  --benchmark-name "$BENCHMARK_NAME" \
  --task-id "$TASK_ID" \
  --bddl-file "$BDDL_FILE" \
  --demo-file "$DEMO_FILE" \
  --libero-root "$LIBERO_ROOT" \
  --out "$OUT"
