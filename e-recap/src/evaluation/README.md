# Evaluation Tools

This directory contains evaluation and visualization tools for E-RECAP.

## Files

- `erecap_wrapper.py`: E-RECAP inference wrapper class
- `longbench_eval.py`: LongBench evaluation script
- `lmeval_runner.py`: lm-eval-harness runner
- `ablation.py`: Ablation study script
- `plot_latency.py`: Latency curve plotting script
- `parse_latency_log.py`: Log file parser for latency data

## Usage

### Plotting Latency Curves

1. **Prepare JSON data files:**
   - `results/latency_baseline.json`: Baseline latency data
   - `results/latency_erecap.json`: E-RECAP latency data
   
   Format:
   ```json
   {
     "4096": 0.7065,
     "8192": 1.2684,
     "16384": 2.3311,
     "32768": 4.1234
   }
   ```

2. **Generate plots:**
   ```bash
   bash scripts/run_plot_latency.sh
   ```
   
   Or directly:
   ```bash
   python3 src/evaluation/plot_latency.py \
       --baseline results/latency_baseline.json \
       --erecap results/latency_erecap.json \
       --out_dir results/fig
   ```

3. **Output files:**
   - `results/fig/latency_curve.png`: Prefill latency comparison
   - `results/fig/speedup_curve.png`: Speedup vs sequence length
   - `results/fig/flops_curve.png`: Estimated FLOPs reduction

### Parsing Log Files

If you have log files from inference runs, parse them first:

```bash
python3 src/evaluation/parse_latency_log.py \
    --log logs/inference.log \
    --baseline results/latency_baseline.json \
    --erecap results/latency_erecap.json
```

Expected log format:
```
[Length 4096] baseline=0.7065s  erecap=0.2527s  speedup=2.80x
[Length 8192] baseline=1.2684s  erecap=0.4920s  speedup=2.58x
```

