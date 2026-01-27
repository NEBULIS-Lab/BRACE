"""
LongBench dataset loader (no model execution)
Utility class for loading LongBench tasks
"""
import json
from typing import List, Dict, Optional


class LongBenchDataset:
    """Utility class for loading LongBench tasks (no execution)."""

    def __init__(self, path: str):
        """
        Initialize LongBench dataset from JSON file.
        
        Args:
            path: Path to LongBench task JSON file
        """
        self.path = path
        self.data = self.load_json(path)

    def load_json(self, path: str) -> List[Dict]:
        """
        Load JSON file containing LongBench task data.
        
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
                    # If it's a dict with 'data' key or similar
                    if "data" in data:
                        return data["data"]
                    elif "test" in data:
                        return data["test"]
                    elif "validation" in data:
                        return data["validation"]
                    else:
                        # Convert dict to list
                        return [data]
                return data
        except FileNotFoundError:
            print(f"[Warning] File not found: {path}")
            return []
        except json.JSONDecodeError as e:
            print(f"[Error] Invalid JSON in {path}: {e}")
            return []

    def __len__(self):
        """Return number of samples in dataset"""
        return len(self.data)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Dict with 'input' and 'answers' keys
        """
        if index >= len(self.data):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.data)}")
        
        item = self.data[index]
        return {
            "input": item.get("input", ""),
            "answers": item.get("answers", [""])
        }
    
    def get_all_samples(self) -> List[Dict]:
        """Get all samples from the dataset"""
        return [self.__getitem__(i) for i in range(len(self.data))]

