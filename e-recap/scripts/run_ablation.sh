#!/bin/bash
# Ablation study script

echo "[Ablation] Running ablation study..."

python3 src/evaluation/ablation.py \
    --out "results/ablation_summary.json"

