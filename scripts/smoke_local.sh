#!/usr/bin/env bash
set -euo pipefail

echo "[1/2] Stub run (no external deps)"
STUB_RUN_DIR=$(scripts/run_stub.sh --run-name local_stub --episodes 1 | tail -n 1)
echo "Stub run: $STUB_RUN_DIR"
scripts/postprocess_run.sh "$STUB_RUN_DIR"

echo "[2/2] Proxy run (synthetic)"
PROXY_RUN_DIR=$(scripts/run_proxy.sh --run-name local_proxy | head -n 1)
echo "Proxy run: $PROXY_RUN_DIR"
scripts/postprocess_run.sh "$PROXY_RUN_DIR"

echo "OK: smoke_local done"

