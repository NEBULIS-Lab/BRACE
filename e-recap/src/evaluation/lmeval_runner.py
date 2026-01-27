"""
lm-eval-harness runner for E-RECAP evaluation
Note: This requires custom integration with lm-eval-harness to support E-RECAP.
Currently provides a wrapper script structure.
"""
import subprocess
import argparse
import os
import json
import sys

LM_TASKS = ["copa", "piqa", "winogrande", "math_qa", "boolq", "cb", "wic", "wsc"]


def run_lmeval(model_type, output_path):
    """
    Use lm-eval-harness via subprocess to evaluate E-RECAP.
    
    Note: This requires lm-eval-harness to be installed and configured
    to support E-RECAP model wrapper. For now, this is a placeholder.
    
    Args:
        model_type: "baseline" or "erecap"
        output_path: Path to save results
    """
    model_path = "checkpoints/qwen2-7b-instruct"
    pruner_path = "checkpoints/pruning_module.pt" if model_type == "erecap" else None
    
    # Check if lm-eval is installed
    try:
        result = subprocess.run(
            ["lm_eval", "--version"],
            capture_output=True,
            text=True
        )
    except FileNotFoundError:
        print("[Error] lm-eval-harness not found. Install with: pip install lm-eval")
        print("[Info] For E-RECAP support, you need to create a custom model wrapper.")
        print("[Info] See Phase D for E-RECAP integration with lm-eval-harness.")
        return
    
    # Note: Standard lm-eval doesn't support E-RECAP pruning yet
    # This would need a custom model wrapper (see Phase D)
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", ",".join(LM_TASKS),
        "--num_fewshot", "5",
        "--output_path", output_path,
        "--batch_size", "1",
    ]
    
    print("[lm-eval] Running:", " ".join(cmd))
    print("[Warning] This will run baseline model only. E-RECAP support requires Phase D integration.")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] lm-eval results saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[Error] lm-eval failed: {e}")
        print("[Info] Make sure lm-eval-harness is properly installed and configured.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lm-eval-harness runner for E-RECAP")
    parser.add_argument("--type", type=str, default="baseline", 
                       choices=["baseline", "erecap"],
                       help="Model type: baseline or erecap")
    parser.add_argument("--out", type=str, default="results/lmeval.json",
                       help="Output JSON path")
    
    args = parser.parse_args()
    
    run_lmeval(args.type, args.out)

