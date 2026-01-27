"""
Abstract base dataset interface for VLA policies.

This module provides the BaseVLADataset abstract class that defines the standard
interface for all VLA dataset implementations. This ensures consistent data handling
across different dataset formats (RLDS, LeRobot, Zarr, etc.).

Patterns extracted from:
- BaseVLADataset (OpenVLA): get_statistics interface
- BaseImageDataset (Diffusion-Policy): get_validation_dataset, normalizer interface
- RobotLeRobotDataset (Pi0): statistics computation with quantiles
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from torch.utils.data import Dataset


class BaseVLADataset(Dataset, ABC):
    """
    Abstract base class for VLA datasets.
    
    All VLA dataset implementations should inherit from this class and
    implement the abstract methods. This provides a consistent interface
    for loading data, computing statistics, and creating train/val splits.
    
    Expected return format from __getitem__:
        {
            'image': torch.Tensor or Dict[str, torch.Tensor]  # Camera images
            'state': torch.Tensor  # Robot proprioceptive state
            'action': torch.Tensor  # Action to predict
            'instruction': str  # Language instruction
        }
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - 'image': Image tensor(s) - (C, H, W) or dict of {view_name: (C, H, W)}
                - 'state': Proprioceptive state tensor - (state_dim,)
                - 'action': Action tensor - (action_dim,) or (horizon, action_dim)
                - 'instruction': Language instruction string
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics for normalization.
        
        Statistics are used for action/state normalization during training
        and denormalization during inference.
        
        Returns:
            Dictionary containing statistics:
                {
                    'action': {
                        'mean': np.ndarray,  # Mean for each dimension
                        'std': np.ndarray,   # Std for each dimension
                        'min': np.ndarray,   # Min for each dimension
                        'max': np.ndarray,   # Max for each dimension
                        'q01': np.ndarray,   # 1st percentile (for quantile norm)
                        'q99': np.ndarray,   # 99th percentile (for quantile norm)
                    },
                    'state': {
                        'mean': np.ndarray,
                        'std': np.ndarray,
                        'min': np.ndarray,
                        'max': np.ndarray,
                        'q01': np.ndarray,
                        'q99': np.ndarray,
                    }
                }
        """
        pass
    
    def get_validation_dataset(self) -> 'BaseVLADataset':
        """
        Get validation split of this dataset.
        
        Default implementation returns self, assuming the dataset was
        initialized with train=False. Subclasses can override this.
        
        Returns:
            Validation dataset instance
        """
        return self
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get dataset metadata/info.
        
        Returns:
            Dictionary with dataset information (e.g., num_episodes, action_dim)
        """
        return {}


def compute_dataset_statistics(
    actions: np.ndarray,
    states: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute statistics from action and state arrays.
    
    Helper function for computing normalization statistics.
    
    Args:
        actions: Array of actions with shape (N, action_dim) or (N, horizon, action_dim)
        states: Optional array of states with shape (N, state_dim)
        
    Returns:
        Dictionary of statistics for 'action' and optionally 'state'
    """
    # Flatten actions if they have horizon dimension
    if actions.ndim == 3:
        # (N, horizon, action_dim) -> (N * horizon, action_dim)
        actions = actions.reshape(-1, actions.shape[-1])
    
    statistics = {
        'action': {
            'mean': actions.mean(axis=0).astype(np.float32),
            'std': actions.std(axis=0).astype(np.float32),
            'min': actions.min(axis=0).astype(np.float32),
            'max': actions.max(axis=0).astype(np.float32),
            'q01': np.percentile(actions, 1, axis=0).astype(np.float32),
            'q99': np.percentile(actions, 99, axis=0).astype(np.float32),
        }
    }
    
    if states is not None:
        statistics['state'] = {
            'mean': states.mean(axis=0).astype(np.float32),
            'std': states.std(axis=0).astype(np.float32),
            'min': states.min(axis=0).astype(np.float32),
            'max': states.max(axis=0).astype(np.float32),
            'q01': np.percentile(states, 1, axis=0).astype(np.float32),
            'q99': np.percentile(states, 99, axis=0).astype(np.float32),
        }
    
    return statistics


class DatasetSplitter:
    """
    Utility class for creating train/validation splits.
    
    Usage:
        splitter = DatasetSplitter(total_samples=1000, val_split=0.1, seed=42)
        train_indices = splitter.train_indices
        val_indices = splitter.val_indices
    """
    
    def __init__(
        self,
        total_samples: int,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize splitter.
        
        Args:
            total_samples: Total number of samples
            val_split: Fraction of data for validation (0.0 to 1.0)
            seed: Random seed for reproducible splits
        """
        self.total_samples = total_samples
        self.val_split = val_split
        self.seed = seed
        
        # Create indices
        indices = np.arange(total_samples)
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        
        num_val = int(total_samples * val_split)
        self.val_indices = indices[:num_val]
        self.train_indices = indices[num_val:]
    
    def get_split(self, train: bool = True) -> np.ndarray:
        """
        Get indices for train or validation split.
        
        Args:
            train: If True, return training indices; otherwise validation
            
        Returns:
            Array of indices
        """
        return self.train_indices if train else self.val_indices

