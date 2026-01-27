import json
import os
import sys
import io
from typing import Dict, List, Optional
from .dataset import LongBenchDataset
from .model_wrapper import ModelWrapper


class TeeOutput:
    """A class that writes to both a file-like object and stdout/stderr"""
    def __init__(self, original_stream, capture_stream):
        self.original_stream = original_stream
        self.capture_stream = capture_stream
    
    def write(self, text):
        self.original_stream.write(text)
        self.capture_stream.write(text)
        self.original_stream.flush()
    
    def flush(self):
        self.original_stream.flush()
        self.capture_stream.flush()

def _simple_match(pred: str, answers: List[str]) -> float:
    pred = pred.lower()
    for ans in answers:
        if ans.lower() in pred:
            return 1.0
    return 0.0

class LongBenchEvaluator:
    def __init__(self, task_path: str, model: ModelWrapper, max_samples=None):
        self.task_path = task_path
        self.dataset = LongBenchDataset(task_path)
        self.model = model
        self.max_samples = max_samples

    def evaluate(
        self,
        do_inference: bool = False,
        save_predictions: bool = False,
        prediction_path: Optional[str] = None,
    ) -> Dict:
        total = len(self.dataset)
        if self.max_samples:
            total = min(total, self.max_samples)

        print(f"[Eval] Task loaded: {len(self.dataset)} total")
        print(f"[Eval] Evaluating: {total} samples")
        print(f"[Eval] Model: {self.model.model_name}")

        if not do_inference:
            print("[Eval] Setup only. No inference.")
            return {
                "task": self.task_path,
                "num_total": len(self.dataset),
                "num_eval": total,
                "model": self.model.model_name,
                "mode": self.model.mode,
                "status": "setup_completed",
            }

        print("[Eval] >>> Real inference START <<<")

        # Capture stdout and stderr while still displaying to console
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Create Tee objects to both display and capture
        stdout_tee = TeeOutput(sys.stdout, stdout_capture)
        stderr_tee = TeeOutput(sys.stderr, stderr_capture)
        
        hits = 0
        preds = []

        # Capture logs during inference while still showing output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            sys.stdout = stdout_tee
            sys.stderr = stderr_tee
            
            for i in range(total):
                item = self.dataset[i]
                pred = self.model.infer(item["input"])
                hit = _simple_match(pred, item["answers"])
                hits += hit

                preds.append({
                    "id": i,
                    "input": item["input"],
                    "prediction": pred,
                    "answers": item["answers"],
                    "hit": hit,
                })
        finally:
            # Restore original streams
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        hit_rate = hits / total

        # Get captured logs
        stdout_logs = stdout_capture.getvalue()
        stderr_logs = stderr_capture.getvalue()
        
        print("[Eval] DONE")
        print(f"[Eval] Hit Rate: {hit_rate:.4f}")

        if save_predictions:
            if prediction_path is None:
                base = os.path.basename(self.task_path).replace(".json", "")
                prediction_path = f"results/longbench_{base}_pred.json"

            os.makedirs("results", exist_ok=True)
            with open(prediction_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, ensure_ascii=False, indent=2)

            print(f"[Eval] Predictions saved to {prediction_path}")

        result = {
            "task": self.task_path,
            "num_total": len(self.dataset),
            "num_eval": total,
            "hit_rate": hit_rate,
            "model": self.model.model_name,
            "mode": self.model.mode,
            "keep_ratio": self.model.keep_ratio,
            "max_new_tokens": self.model.max_new_tokens,
            "status": "inference_completed",
        }
        
        # Add logs if captured
        if stdout_logs:
            result["stdout_logs"] = stdout_logs.split("\n")
        if stderr_logs:
            result["stderr_logs"] = stderr_logs.split("\n")
        
        return result
