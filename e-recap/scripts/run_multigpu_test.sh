#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[Multi-GPU Test] Running multi-gpu memory experiment..."
python3 -u src/multigpu_test.py \
    --lengths 4096 8192 16384 32768 65536 131072
