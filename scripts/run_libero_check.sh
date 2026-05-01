#!/usr/bin/env bash
set -euo pipefail

LIBERO_ROOT="${BRACE_LIBERO_ROOT:-${LIBERO_ROOT:-}}"
VERIFY="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --libero-root) LIBERO_ROOT="$2"; shift 2 ;;
    --verify-files) VERIFY="true"; shift 1 ;;
    -h|--help)
      cat <<EOF
Usage: scripts/run_libero_check.sh [--libero-root <root>] [--verify-files]
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

ARGS=()
if [[ "$VERIFY" == "true" ]]; then
  ARGS+=(--verify-files)
fi

python experiments/libero/check_task_suites.py \
  --libero-root "$LIBERO_ROOT" \
  "${ARGS[@]}"
