import argparse
import json
import os
import sys
import io
from .model_wrapper import ModelWrapper
from .evaluator import LongBenchEvaluator, TeeOutput

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model", type=str, default="checkpoints/qwen2-7b-instruct")
    parser.add_argument("--pruning_module", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/longbench_result.json")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "erecap"])
    parser.add_argument("--do_inference", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--prediction_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--keep_ratio", type=float, default=1.0)

    args = parser.parse_args()

    # Capture all output from the beginning (both display and save)
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    stdout_tee = TeeOutput(sys.stdout, stdout_capture)
    stderr_tee = TeeOutput(sys.stderr, stderr_capture)
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        sys.stdout = stdout_tee
        sys.stderr = stderr_tee
        
        print("[LongBench] Starting evaluation...")
        print(f"  Task: {args.task}")
        print(f"  Model: {args.model}")

        model = ModelWrapper(
            model_name=args.model,
            pruning_module_path=args.pruning_module,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            keep_ratio=args.keep_ratio,
        )

        model.load_model(real_load=args.do_inference)

        evaluator = LongBenchEvaluator(
            task_path=args.task,
            model=model,
            max_samples=args.max_samples,
        )

        result = evaluator.evaluate(
            do_inference=args.do_inference,
            save_predictions=args.save_predictions,
            prediction_path=args.prediction_path,
        )
        
        # Add command-line arguments to result
        result["command_args"] = {
            "task": args.task,
            "model": args.model,
            "pruning_module": args.pruning_module,
            "mode": args.mode,
            "keep_ratio": args.keep_ratio,
            "max_new_tokens": args.max_new_tokens,
            "max_samples": args.max_samples,
        }
        
        # Get all captured logs (including initialization)
        all_stdout_logs = stdout_capture.getvalue()
        all_stderr_logs = stderr_capture.getvalue()
        
        # Merge logs (evaluator may have added its own logs)
        if "stdout_logs" in result and result.get("stdout_logs"):
            # Merge: prepend initialization logs, then inference logs
            init_logs = [line for line in all_stdout_logs.split("\n") if line.strip()]
            result["stdout_logs"] = init_logs + result["stdout_logs"]
        else:
            result["stdout_logs"] = all_stdout_logs.split("\n") if all_stdout_logs else []
            
        if "stderr_logs" in result and result.get("stderr_logs"):
            init_stderr = [line for line in all_stderr_logs.split("\n") if line.strip()]
            result["stderr_logs"] = init_stderr + result["stderr_logs"]
        else:
            result["stderr_logs"] = all_stderr_logs.split("\n") if all_stderr_logs else []
        
        # Restore original streams before file I/O
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[OK] Summary saved to {args.output}")
        
    except Exception as e:
        # Restore streams on error
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        raise

if __name__ == "__main__":
    main()
