"""
Custom LongBench task for lm-eval-harness

This module provides a custom Task class that integrates LongBench tasks
with the lm-eval-harness evaluation framework.

IMPORTANT: This is a setup/placeholder implementation.
- Does NOT execute real inference
- Does NOT load models
- Only defines task structure and data loading
"""
import json
import os
from typing import Iterator, Dict, Optional


class LongBenchTask:
    """
    Custom LongBench task for lm-eval-harness.
    
    Compatible with lm-eval-harness Task interface.
    This class handles LongBench dataset loading and formatting
    for evaluation with lm-eval-harness.
    
    NOTE: Does NOT run inference during setup stage.
    
    Attributes:
        data_path: Path to LongBench task JSON file
        dataset: Loaded dataset (list of dicts)
        task_name: Task name (derived from filename)
    """
    
    VERSION = 1
    
    def __init__(self, data_path: str, **kwargs):
        """
        Initialize LongBench task.
        
        Args:
            data_path: Path to LongBench task JSON file
            **kwargs: Additional arguments (compatible with lm-eval Task interface)
        """
        self.data_path = data_path
        self.dataset = self._load_dataset(data_path)
        self.task_name = os.path.basename(data_path).replace(".json", "")
        
        # Store kwargs for compatibility
        self.config = kwargs
        
        print(f"[LongBenchTask] Initialized: {self.task_name}")
        print(f"  Data path: {self.data_path}")
        print(f"  Dataset size: {len(self.dataset)} samples")
    
    def _load_dataset(self, path: str) -> list:
        """
        Load LongBench dataset from JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            List of task samples
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, dict):
                    # Check common keys
                    if "data" in data:
                        return data["data"]
                    elif "test" in data:
                        return data["test"]
                    elif "validation" in data:
                        return data["validation"]
                    elif "train" in data:
                        return data["train"]
                    else:
                        # Single item or convert dict to list
                        return [data] if data else []
                return data if isinstance(data, list) else []
        except FileNotFoundError:
            print(f"[Warning] File not found: {path}")
            return []
        except json.JSONDecodeError as e:
            print(f"[Error] Invalid JSON in {path}: {e}")
            return []
    
    def has_training_docs(self) -> bool:
        """
        Check if task has training documents.
        
        Returns:
            False (LongBench tasks typically don't use training docs)
        """
        return False
    
    def has_validation_docs(self) -> bool:
        """
        Check if task has validation documents.
        
        Returns:
            True (LongBench tasks use validation/test splits)
        """
        return len(self.dataset) > 0
    
    def validation_docs(self) -> Iterator[Dict]:
        """
        Iterate over validation documents.
        
        Yields:
            Dict with 'input' and 'answers' keys
        """
        for item in self.dataset:
            yield {
                "input": item.get("input", item.get("question", "")),
                "answers": item.get("answers", item.get("answer", [""])),
                "context": item.get("context", ""),  # Optional context
            }
    
    def doc_to_text(self, doc: Dict) -> str:
        """
        Convert document to input text for the model.
        
        Args:
            doc: Document dict with 'input' and optional 'context'
            
        Returns:
            Input text string
        """
        # LongBench format: usually just "input" field
        # Some tasks may have "context" + "question" structure
        if "input" in doc:
            return doc["input"]
        elif "question" in doc and "context" in doc:
            # Combine context and question if separate
            return f"{doc['context']}\n\n{doc['question']}"
        else:
            return str(doc)
    
    def doc_to_target(self, doc: Dict) -> str:
        """
        Convert document to target answer.
        
        Args:
            doc: Document dict with 'answers'
            
        Returns:
            Target answer string (first answer if multiple)
        """
        answers = doc.get("answers", doc.get("answer", [""]))
        if isinstance(answers, list):
            return answers[0] if answers else ""
        return str(answers)
    
    def evaluate(self, model, dataset=None, **kwargs) -> Dict:
        """
        Setup-only version: DOES NOT EXECUTE REAL INFERENCE.
        
        This method is called by lm-eval-harness during evaluation setup.
        In the actual evaluation phase, this will run real inference.
        
        Args:
            model: Model wrapper instance (ERECAPModel or baseline)
            dataset: Optional dataset override
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dict with setup information (no actual metrics)
        """
        print(f"[LM-EVAL] LongBench task loaded: {self.task_name}")
        print(f"[LM-EVAL] Data path: {self.data_path}")
        print(f"[LM-EVAL] Dataset size: {len(self.dataset)} samples")
        print(f"[LM-EVAL] Model wrapper: {model.__class__.__name__}")
        print(f"[LM-EVAL] Model config: {model.model_name if hasattr(model, 'model_name') else 'N/A'}")
        
        if hasattr(model, 'pruning_module') and model.pruning_module:
            print(f"[LM-EVAL] Mode: E-RECAP (with pruning)")
        else:
            print(f"[LM-EVAL] Mode: Baseline (no pruning)")
        
        print(f"[LM-EVAL] No inference executed in this setup phase.")
        
        return {
            "task": self.task_name,
            "task_path": self.data_path,
            "dataset_size": len(self.dataset),
            "model": str(model),
            "status": "setup_completed",
            "message": "No inference executed in setup phase"
        }

