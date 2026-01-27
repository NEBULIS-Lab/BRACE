"""
Template workspace class for new VLA implementations.

This provides a minimal example of how to implement a training workspace
that inherits from BaseVLAWorkspace.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from robofactory.policy.core import BaseVLAWorkspace, BaseVLADataset
from .policy import SimpleModel


class SimpleDataset(BaseVLADataset):
    """
    Simple demonstration dataset.
    
    Replace this with your actual dataset implementation.
    """
    
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        image_size: int = 224,
        action_dim: int = 8,
        state_dim: int = 8,
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to data directory
            train: Whether this is training data
            image_size: Target image size
            action_dim: Action dimension
            state_dim: State dimension
        """
        self.data_dir = data_dir
        self.train = train
        self.image_size = image_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # In real implementation, load actual data here
        # This is just dummy data for demonstration
        self.num_samples = 1000 if train else 100
        
        # Generate dummy statistics
        self._statistics = {
            'action': {
                'mean': np.zeros(action_dim, dtype=np.float32),
                'std': np.ones(action_dim, dtype=np.float32),
                'q01': np.full(action_dim, -2.0, dtype=np.float32),
                'q99': np.full(action_dim, 2.0, dtype=np.float32),
            },
            'state': {
                'mean': np.zeros(state_dim, dtype=np.float32),
                'std': np.ones(state_dim, dtype=np.float32),
                'q01': np.full(state_dim, -2.0, dtype=np.float32),
                'q99': np.full(state_dim, 2.0, dtype=np.float32),
            }
        }
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        In real implementation, load actual data from disk.
        This returns dummy data for demonstration.
        """
        # Dummy random data
        return {
            'image': torch.randn(3, self.image_size, self.image_size),
            'state': torch.randn(self.state_dim),
            'action': torch.randn(self.action_dim),
            'instruction': "Complete the task",
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        return self._statistics


class NewPolicyWorkspace(BaseVLAWorkspace):
    """
    Training workspace for the new policy.
    
    Inherits from BaseVLAWorkspace which provides:
    - DDP distributed training
    - Optimizer and scheduler setup
    - Training loop with validation
    - WandB logging
    - Checkpoint save/load
    
    You only need to implement:
    - _init_model(): Initialize your model
    - _init_dataset(): Create train/val datasets
    - _compute_loss(): Compute training loss
    """
    
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        """
        Initialize workspace.
        
        Args:
            cfg: Hydra configuration
            output_dir: Output directory
        """
        super().__init__(cfg, output_dir)
        
        # Store model-specific config
        self.model_cfg = cfg.model
        self.data_cfg = cfg.task
    
    def _init_model(self) -> nn.Module:
        """
        Initialize the model.
        
        Returns:
            Initialized model (not yet wrapped with DDP)
        """
        model = SimpleModel(
            image_size=self.model_cfg.get('image_size', 224),
            state_dim=self.model_cfg.get('state_dim', 8),
            action_dim=self.model_cfg.get('action_dim', 8),
            hidden_dim=self.model_cfg.get('hidden_dim', 256),
        )
        return model.to(self.device)
    
    def _init_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        Create training and validation datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        train_dataset = SimpleDataset(
            data_dir=self.data_cfg.data_dir,
            train=True,
            image_size=self.model_cfg.get('image_size', 224),
            action_dim=self.model_cfg.get('action_dim', 8),
            state_dim=self.model_cfg.get('state_dim', 8),
        )
        
        val_dataset = SimpleDataset(
            data_dir=self.data_cfg.data_dir,
            train=False,
            image_size=self.model_cfg.get('image_size', 224),
            action_dim=self.model_cfg.get('action_dim', 8),
            state_dim=self.model_cfg.get('state_dim', 8),
        )
        
        return train_dataset, val_dataset
    
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute training loss from a batch.
        
        Args:
            batch: Batch dictionary with 'image', 'state', 'action' keys
            
        Returns:
            Loss tensor (scalar)
        """
        # Get inputs
        image = batch['image']
        state = batch['state']
        action_target = batch['action']
        
        # Forward pass
        # Handle DDP wrapped model
        model = self.model.module if hasattr(self.model, 'module') else self.model
        action_pred = model(image, state)
        
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(action_pred, action_target)
        
        return loss
    
    def _get_checkpoint_dir(self):
        """Get checkpoint directory."""
        policy_type = self.cfg.get('exp_name', 'new_policy')
        task_name = self.cfg.task.name if hasattr(self.cfg.task, 'name') else 'unknown'
        agent_id = self.cfg.get('agent_id', 0)
        from pathlib import Path
        return Path('robofactory/checkpoints') / policy_type / f"{task_name}_Agent{agent_id}"

