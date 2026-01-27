"""
LeRobot dataset loader for Pi0/Pi0.5 training.

This module follows openpi's data format and conventions:
- Uses LeRobot dataset format (HuggingFace datasets)
- Returns data in openpi's expected format for Pi0 models
- Handles action chunking (action_horizon=50)
- Provides 3 camera views as required by Pi0 architecture
"""

import torch
import numpy as np
from typing import Dict, List
from pathlib import Path

# Import base class from core
from robofactory.policy.core import BaseVLADataset

try:
    # Try new lerobot structure (v0.4+)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    try:
        # Fallback to old structure (v0.2-0.3)
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("Warning: LeRobot not installed. Please install with: pip install lerobot>=0.2.0")
        LeRobotDataset = None


class RobotLeRobotDataset(BaseVLADataset):
    """
    PyTorch Dataset for loading LeRobot format data for Pi0/Pi0.5 training.
    
    Follows openpi conventions:
    - Input image keys: "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"
    - Input state key: "state"
    - Output key: "actions" (action chunked sequence)
    - Task key: "task" (language instruction)
    
    Returns data in format expected by openpi's Pi0 models.
    """
    
    def __init__(
        self,
        repo_id: str,
        action_horizon: int = 50,
        train: bool = True,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize dataset.
        
        Args:
            repo_id: LeRobot repository ID (e.g., "robofactory/LiftBarrier-rf_Agent0_150")
            action_horizon: Number of future actions to predict
            train: If True, load training split; otherwise validation split
            val_split: Fraction of data to use for validation
            seed: Random seed for train/val split
        """
        if LeRobotDataset is None:
            raise ImportError("LeRobot is required. Install with: pip install lerobot>=0.2.0")
        
        self.repo_id = repo_id
        self.action_horizon = action_horizon
        self.train = train
        
        # Load LeRobot dataset with action chunking (following openpi pattern)
        print(f"Loading LeRobot dataset: {repo_id}")
        
        # Check if repo_id is a local path (contains '/')
        if '/' in repo_id:
            # Local dataset - convert to absolute path
            repo_path = Path(repo_id).resolve() if not Path(repo_id).is_absolute() else Path(repo_id)
            
            # If path doesn't exist as given, try with /workspace/OpenMARL/ prefix
            if not repo_path.exists():
                repo_path = Path("/workspace/OpenMARL") / repo_id
            
            if not repo_path.exists():
                raise FileNotFoundError(f"LeRobot dataset not found at {repo_path}")
            
            print(f"  Loading from local path: {repo_path}")
            # For local datasets, root should be the dataset directory itself
            # repo_id is just used as a name/identifier
            # Pass revision="main" to prevent HuggingFace Hub checks
            # Don't specify episodes to load all available episodes
            self.dataset = LeRobotDataset(
                repo_id=repo_path.name,  # Dataset name for identification
                root=str(repo_path),  # Full path to the dataset directory
                revision="main",  # Prevents Hub version checking
                episodes=None,  # Load all episodes (avoids needing meta/episodes dir)
                delta_timestamps={
                    "actions": [i / 10.0 for i in range(action_horizon)],  # 10 FPS
                },
            )
        else:
            # HuggingFace repo - use repo_id directly
            print(f"  Loading from HuggingFace Hub: {repo_id}")
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                delta_timestamps={
                    "actions": [i / 10.0 for i in range(action_horizon)],  # 10 FPS
                },
            )
        
        # Split into train/val
        total_samples = len(self.dataset)
        num_val = int(total_samples * val_split)
        num_train = total_samples - num_val
        
        indices = np.arange(total_samples)
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        
        if train:
            self.indices = indices[:num_train]
        else:
            self.indices = indices[num_train:]
        
        print(f"Loaded {len(self.indices)} samples for {'train' if train else 'val'}")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Returns dict with openpi-compatible format for Pi0:
        {
            "image": dict of 3 camera images,
            "image_mask": dict of 3 image validity masks,
            "state": proprio/robot state,
            "actions": action sequence [action_horizon, action_dim],
            "prompt": language instruction,
        }
        """
        real_idx = self.indices[idx]
        sample = self.dataset[real_idx]
        
        # Extract data following openpi's conventions
        # Pi0 expects images in dict format with specific keys
        images = {}
        image_masks = {}
        
        # Pi0 requires EXACTLY these 3 keys (in order)
        # IMPORTANT: openpi expects images in HWC format (height, width, channels), NOT CHW!
        
        # base_0_rgb: Base/exterior camera (third-person view)
        if "base_0_rgb" in sample:
            images["base_0_rgb"] = torch.from_numpy(
                np.array(sample["base_0_rgb"])
            ).float()
            # openpi expects HWC format - convert CHW to HWC if needed
            if images["base_0_rgb"].ndim == 3 and images["base_0_rgb"].shape[0] == 3:
                # CHW -> HWC
                images["base_0_rgb"] = images["base_0_rgb"].permute(1, 2, 0)
            # Normalize to [0, 1]
            if images["base_0_rgb"].max() > 1.0:
                images["base_0_rgb"] = images["base_0_rgb"] / 255.0
            image_masks["base_0_rgb"] = torch.tensor(True)
        else:
            raise ValueError(f"Missing required camera: base_0_rgb in sample")
        
        # left_wrist_0_rgb: Second camera (global/overhead view in RoboFactory)
        if "left_wrist_0_rgb" in sample:
            images["left_wrist_0_rgb"] = torch.from_numpy(
                np.array(sample["left_wrist_0_rgb"])
            ).float()
            # openpi expects HWC format
            if images["left_wrist_0_rgb"].ndim == 3 and images["left_wrist_0_rgb"].shape[0] == 3:
                images["left_wrist_0_rgb"] = images["left_wrist_0_rgb"].permute(1, 2, 0)
            if images["left_wrist_0_rgb"].max() > 1.0:
                images["left_wrist_0_rgb"] = images["left_wrist_0_rgb"] / 255.0
            image_masks["left_wrist_0_rgb"] = torch.tensor(True)
        else:
            # Fallback: create dummy image
            images["left_wrist_0_rgb"] = torch.zeros_like(images["base_0_rgb"])
            image_masks["left_wrist_0_rgb"] = torch.tensor(False)
        
        # right_wrist_0_rgb: Third camera (wrist/gripper view)
        if "right_wrist_0_rgb" in sample:
            images["right_wrist_0_rgb"] = torch.from_numpy(
                np.array(sample["right_wrist_0_rgb"])
            ).float()
            # openpi expects HWC format
            if images["right_wrist_0_rgb"].ndim == 3 and images["right_wrist_0_rgb"].shape[0] == 3:
                images["right_wrist_0_rgb"] = images["right_wrist_0_rgb"].permute(1, 2, 0)
            if images["right_wrist_0_rgb"].max() > 1.0:
                images["right_wrist_0_rgb"] = images["right_wrist_0_rgb"] / 255.0
            image_masks["right_wrist_0_rgb"] = torch.tensor(True)
        else:
            # Fallback: create dummy image
            images["right_wrist_0_rgb"] = torch.zeros_like(images["base_0_rgb"])
            image_masks["right_wrist_0_rgb"] = torch.tensor(False)
        
        # State/proprio (openpi convention: "state")
        state = torch.from_numpy(np.array(sample["state"])).float()
        
        # Actions (already chunked by LeRobot with delta_timestamps)
        # Shape: [action_horizon, action_dim]
        actions = torch.from_numpy(np.array(sample["actions"])).float()
        
        # Language instruction
        prompt = sample.get("task", "")
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        
        return {
            "image": images,
            "image_mask": image_masks,
            "state": state,
            "actions": actions,
            "prompt": prompt,
        }
    
    def get_statistics(self) -> Dict:
        """
        Compute dataset statistics for normalization.
        
        Following openpi's normalization pattern with quantile normalization.
        """
        # Collect all states and actions
        all_states = []
        all_actions = []
        
        # Sample subset for efficiency (up to 1000 samples or full dataset)
        sample_size = min(len(self), 1000)
        sample_indices = np.random.choice(len(self), sample_size, replace=False)
        
        print(f"Computing statistics from {sample_size} samples...")
        
        for idx in sample_indices:
            sample = self[idx]
            all_states.append(sample["state"].numpy())
            # Flatten action sequence for statistics
            all_actions.append(sample["actions"].numpy().reshape(-1, sample["actions"].shape[-1]))
        
        all_states = np.array(all_states)
        all_actions = np.concatenate(all_actions, axis=0)
        
        # Compute statistics (following openpi pattern with quantiles)
        stats = {
            "state": {
                "mean": torch.from_numpy(all_states.mean(axis=0)),
                "std": torch.from_numpy(all_states.std(axis=0)),
                "q01": torch.from_numpy(np.percentile(all_states, 1, axis=0)),
                "q99": torch.from_numpy(np.percentile(all_states, 99, axis=0)),
            },
            "action": {
                "mean": torch.from_numpy(all_actions.mean(axis=0)),
                "std": torch.from_numpy(all_actions.std(axis=0)),
                "q01": torch.from_numpy(np.percentile(all_actions, 1, axis=0)),
                "q99": torch.from_numpy(np.percentile(all_actions, 99, axis=0)),
            },
        }
        
        return stats


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched dictionary with all tensors properly stacked
    """
    # Stack tensors in batch
    collated = {
        "image": {},
        "image_mask": {},
        "state": torch.stack([item["state"] for item in batch]),
        "actions": torch.stack([item["actions"] for item in batch]),
        "prompt": [item["prompt"] for item in batch],
    }
    
    # Stack images (all 3 camera views)
    for key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
        if key in batch[0]["image"]:
            collated["image"][key] = torch.stack([item["image"][key] for item in batch])
            collated["image_mask"][key] = torch.stack([item["image_mask"][key] for item in batch])
    
    return collated

