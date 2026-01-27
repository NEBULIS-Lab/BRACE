#!/bin/bash
# Generate comprehensive latency curves from JSON data
# Generates one comprehensive figure for single-GPU and one for multi-GPU

OUT_DIR=${1:-"results/fig"}

echo "[Plot Latency] Generating comprehensive comparison plots..."
echo ""

# Generate comprehensive plots (both single-GPU and multi-GPU)
python3 src/evaluation/plot_latency.py --plot-comprehensive "$OUT_DIR"
echo ""

echo "[OK] All comprehensive plots generated!"

