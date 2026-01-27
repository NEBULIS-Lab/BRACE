"""
Ablation study script for E-RECAP
Compares different pruning configurations (no training, inference only)
"""
import json
import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.evaluation.erecap_wrapper import ERECAPInference

# Ablation configurations
# Note: These checkpoint files need to be trained separately with different configs
ABLATION_CONFIGS = {
    "baseline": None,
    "erecap": "checkpoints/pruning_module.pt",
    # The following configs require retraining with different settings:
    # "no_rank_loss": "checkpoints/pruning_no_rank.pt",  # Train without ranking loss
    # "no_mse_loss": "checkpoints/pruning_no_mse.pt",    # Train without MSE loss
    # "ratio_0.8": "checkpoints/pruning_ratio_0.8.pt",    # Train with keep_ratio=0.8
    # "ratio_0.6": "checkpoints/pruning_ratio_0.6.pt",   # Train with keep_ratio=0.6
}


def run_ablation(output_path="results/ablation_summary.json", test_prompt=None):
    """
    Run ablation study comparing different configurations.
    
    Args:
        output_path: Path to save results JSON
        test_prompt: Test prompt (default: standard prompt)
    """
    if test_prompt is None:
        test_prompt = "Explain the importance of regularization in deep learning."
    
    results = {}
    
    for name, pruner_path in ABLATION_CONFIGS.items():
        print(f"[Ablation] Testing {name}")
        
        try:
            if pruner_path and not os.path.exists(pruner_path):
                print(f"[Warning] Checkpoint {pruner_path} not found, skipping {name}")
                results[name] = {
                    "error": f"Checkpoint not found: {pruner_path}",
                    "status": "skipped"
                }
                continue
            
            model = ERECAPInference(
                model_path="checkpoints/qwen2-7b-instruct",
                pruner_path=pruner_path,
                device="cuda"
            )
            
            import time
            start = time.time()
            output = model.generate(test_prompt, max_new_tokens=64)
            latency = time.time() - start
            
            output_length = len(model.tokenizer(output)["input_ids"])
            prompt_length = len(model.tokenizer(test_prompt)["input_ids"])
            
            results[name] = {
                "text": output,
                "output_length": output_length,
                "prompt_length": prompt_length,
                "latency": latency,
                "status": "success"
            }
            
            print(f"  Output length: {output_length}, Latency: {latency:.4f}s")
            
        except Exception as e:
            print(f"[Error] {name} failed: {e}")
            results[name] = {
                "error": str(e),
                "status": "failed"
            }
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Ablation results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for E-RECAP")
    parser.add_argument("--out", type=str, default="results/ablation_summary.json",
                       help="Output JSON path")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Test prompt (default: standard prompt)")
    
    args = parser.parse_args()
    
    run_ablation(args.out, args.prompt)

