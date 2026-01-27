"""
LongBench evaluation script for E-RECAP
Evaluates long-context QA tasks performance
"""
import json
import time
import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from datasets import load_dataset
from src.evaluation.erecap_wrapper import ERECAPInference


def evaluate_longbench(model_type, task, output_path, max_length=32768, num_samples=30):
    """
    LongBench evaluation wrapper.
    
    Only prefill performance + generation quality for long-context QA tasks.
    
    Args:
        model_type: "baseline" or "erecap"
        task: LongBench task name (e.g., "hotpotqa", "2wikimqa")
        output_path: Path to save results JSON
        max_length: Maximum context length
        num_samples: Number of samples to evaluate
    """
    print(f"[LongBench] Loading task: {task}")
    
    try:
        dataset = load_dataset("THUDM/LongBench", task)
    except Exception as e:
        print(f"[Error] Failed to load LongBench dataset: {e}")
        print("[Info] You may need to install: pip install datasets")
        return
    
    if model_type == "baseline":
        pruner = None
        print("[LongBench] Using baseline (no pruning).")
    else:
        pruner = "checkpoints/pruning_module.pt"
        print("[LongBench] Using E-RECAP pruning module.")
    
    # Initialize model
    try:
        sd_model = ERECAPInference(
            model_path="checkpoints/qwen2-7b-instruct",
            pruner_path=pruner,
            device="cuda",
            use_flash=False,
        )
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return
    
    results = []
    dataset_split = dataset.get("validation") or dataset.get("test") or dataset.get("train")
    
    if dataset_split is None:
        print("[Error] No suitable dataset split found")
        return
    
    for i, sample in enumerate(dataset_split):
        if i >= num_samples:
            break
            
        try:
            context = sample.get("context", "")
            question = sample.get("question", "")
            
            if not context or not question:
                continue
                
            prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
            
            # Truncate if too long
            tokenized = sd_model.tokenizer(prompt, return_tensors="pt")
            if len(tokenized["input_ids"][0]) > max_length:
                # Truncate context
                context_tokens = sd_model.tokenizer(context, return_tensors="pt")["input_ids"][0]
                question_tokens = sd_model.tokenizer(question, return_tensors="pt")["input_ids"][0]
                max_context_len = max_length - len(question_tokens) - 20  # Reserve for prompt template
                if max_context_len > 0:
                    truncated_context = sd_model.tokenizer.decode(context_tokens[:max_context_len])
                    prompt = f"Context: {truncated_context}\n\nQuestion: {question}\nAnswer:"
            
            start = time.time()
            generated = sd_model.generate(prompt, max_new_tokens=64)
            latency = time.time() - start
            
            context_length = len(sd_model.tokenizer(prompt)["input_ids"])
            
            results.append({
                "id": i,
                "latency": latency,
                "output": generated,
                "context_length": context_length,
                "question": question[:100] if len(question) > 100 else question,  # Truncate for storage
            })
            
            print(f"[{i+1}/{num_samples}] Latency={latency:.4f}s | len={context_length}")
            
        except Exception as e:
            print(f"[Error] Sample {i} failed: {e}")
            continue
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] LongBench results saved to {output_path}")
    print(f"[Summary] Evaluated {len(results)} samples, avg latency: {sum(r['latency'] for r in results)/len(results):.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongBench evaluation for E-RECAP")
    parser.add_argument("--task", type=str, default="hotpotqa", 
                       help="LongBench task name")
    parser.add_argument("--type", type=str, default="baseline", 
                       choices=["baseline", "erecap"],
                       help="Model type: baseline or erecap")
    parser.add_argument("--out", type=str, default="results/longbench_hotpotqa.json",
                       help="Output JSON path")
    parser.add_argument("--num_samples", type=int, default=30,
                       help="Number of samples to evaluate")
    parser.add_argument("--max_length", type=int, default=32768,
                       help="Maximum context length")
    
    args = parser.parse_args()
    
    evaluate_longbench(
        model_type=args.type,
        task=args.task,
        output_path=args.out,
        max_length=args.max_length,
        num_samples=args.num_samples
    )

