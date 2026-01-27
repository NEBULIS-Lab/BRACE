"""
lm-eval-harness runner for E-RECAP (setup phase - no inference execution)

Main entry point for running lm-eval-harness evaluation with E-RECAP models.
This script sets up the evaluation pipeline without executing actual inference.

IMPORTANT: This is a setup/placeholder implementation.
- Does NOT execute real inference
- Does NOT load any actual model
- Does NOT require GPU
- Only validates setup and saves configuration

TODO: In actual evaluation phases, implement real model loading and inference.
"""
import argparse
import json
import os
import sys

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from src.evaluation.lmeval.longbench_task import LongBenchTask
from src.evaluation.lmeval.erecap_model import ERECAPModel


def main():
    """
    Main entry point for lm-eval-harness setup.
    
    This function:
    1. Parses command-line arguments
    2. Initializes E-RECAP model wrapper (no actual loading)
    3. Loads LongBench task (data loading only)
    4. Runs setup evaluation (no inference)
    5. Saves setup result to JSON
    """
    parser = argparse.ArgumentParser(
        description="lm-eval-harness runner for E-RECAP (setup phase - no inference)"
    )
    parser.add_argument(
        "--task_config",
        type=str,
        required=True,
        help="Path to LongBench task JSON file (e.g., data/LongBench/narrativeqa.json)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="checkpoints/qwen2-7b-instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default=None,
        help="Path to pruning module checkpoint (None for baseline, "
             "e.g., checkpoints/pruning_module.pt for E-RECAP)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path for setup result (default: auto-generated)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Auto-generate output path if not provided
    if args.output is None:
        task_name = os.path.basename(args.task_config).replace(".json", "")
        model_type = "erecap" if args.pruner else "baseline"
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(
            output_dir,
            f"lmeval_{task_name}_{model_type}_setup.json"
        )
    
    print("=" * 60)
    print("[LM-EVAL] E-RECAP Evaluation Setup (No Inference)")
    print("=" * 60)
    print(f"Task config: {args.task_config}")
    print(f"Model: {args.model_name}")
    print(f"Pruning module: {args.pruner or 'None (baseline)'}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Check if task file exists
    if not os.path.exists(args.task_config):
        print(f"[Error] Task file not found: {args.task_config}")
        print("[Info] LongBench task files should be JSON format.")
        print("[Info] Example: data/LongBench/narrativeqa.json")
        return 1
    
    # Initialize E-RECAP model wrapper (no actual loading)
    print("\n[Step 1] Initializing model wrapper...")
    try:
        model = ERECAPModel(
            model_name=args.model_name,
            pruning_module=args.pruner,
            device=args.device
        )
        print("[OK] Model wrapper initialized")
    except Exception as e:
        print(f"[Error] Failed to initialize model wrapper: {e}")
        return 1
    
    # Load LongBench task (data loading only)
    print("\n[Step 2] Loading LongBench task...")
    try:
        task = LongBenchTask(args.task_config)
        if len(task.dataset) == 0:
            print(f"[Warning] Task dataset is empty: {args.task_config}")
        else:
            print(f"[OK] Task loaded: {len(task.dataset)} samples")
    except Exception as e:
        print(f"[Error] Failed to load task: {e}")
        return 1
    
    # Run setup evaluation (no inference)
    print("\n[Step 3] Running setup evaluation (no inference)...")
    try:
        result = task.evaluate(model)
        print("[OK] Setup evaluation completed")
    except Exception as e:
        print(f"[Error] Setup evaluation failed: {e}")
        return 1
    
    # Add additional metadata
    result["setup_info"] = {
        "model_name": args.model_name,
        "pruning_module": args.pruner,
        "device": args.device,
        "task_config": args.task_config,
        "output_path": args.output,
    }
    
    # Save setup result
    print(f"\n[Step 4] Saving setup result...")
    try:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Setup result saved to: {args.output}")
    except Exception as e:
        print(f"[Error] Failed to save result: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("[LM-EVAL] Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Implement actual model loading in ERECAPModel")
    print("2. Implement inference methods (generate_until, loglikelihood)")
    print("3. Run actual evaluation with lm-eval-harness")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

