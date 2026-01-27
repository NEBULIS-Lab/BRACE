"""
OpenVLA policy interface for inference.

This module provides the OpenVLAPolicy class for inference, inheriting
from the core BaseVLAPolicy for interface consistency across all VLA policies.
"""

from typing import Dict, Optional, Any
from pathlib import Path
import json
import torch
import numpy as np

# Import core base class and shared utilities
from robofactory.policy.core import BaseVLAPolicy

from ..model.openvla_wrapper import OpenVLAModel


class OpenVLAPolicy(BaseVLAPolicy):
    """
    Policy interface for OpenVLA model inference.
    
    This provides a simple interface for loading trained models
    and predicting actions during evaluation. Inherits from BaseVLAPolicy
    for consistent interface across all VLA policy implementations.
    
    Attributes:
        model: OpenVLAModel instance
        device: Device for inference
        instruction: Current instruction (can be updated)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        action_statistics: Optional[Dict] = None,
        action_dim: int = 8,
    ):
        """
        Initialize policy from checkpoint.
        
        FIX: Now loads full action statistics (mean, std, min, max) from 
        checkpoint's action_stats.json for proper 8-DOF detokenization.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            action_statistics: Dictionary with action mean/std/min/max for denormalization
            action_dim: Dimension of action space
        """
        super().__init__(device=device, action_dim=action_dim)
        
        # Load model
        print(f"Loading OpenVLA policy from {checkpoint_path}")
        self.model = OpenVLAModel.from_pretrained(
            checkpoint_path,
            device=device,
        )
        self.model.eval()
        
        # FIX: Load action statistics from checkpoint directory if not provided
        if action_statistics is None:
            action_statistics = self._load_action_statistics(checkpoint_path)
        
        # FIX: Set action statistics with min/max for proper detokenization
        if action_statistics is not None:
            mean = np.array(action_statistics.get('mean', action_statistics.get('action_mean', [])))
            std = np.array(action_statistics.get('std', action_statistics.get('action_std', [])))
            action_min = action_statistics.get('min', action_statistics.get('action_min'))
            action_max = action_statistics.get('max', action_statistics.get('action_max'))
            
            if action_min is not None:
                action_min = np.array(action_min)
            if action_max is not None:
                action_max = np.array(action_max)
            
            self.model.set_action_statistics(
                mean=mean,
                std=std,
                action_min=action_min,
                action_max=action_max,
            )
            print(f"Loaded action statistics: dim={len(mean)}")
        else:
            print("WARNING: No action statistics found! Using default bridge_orig stats.")
        
        # Default instruction (can be updated)
        self._instruction = None
    
    def _load_action_statistics(self, checkpoint_path: str) -> Optional[Dict]:
        """
        Load action statistics from checkpoint directory.
        
        Searches for action_stats.json in the checkpoint directory.
        
        Args:
            checkpoint_path: Path to model checkpoint directory
            
        Returns:
            Dictionary with action statistics or None if not found
        """
        checkpoint_dir = Path(checkpoint_path)
        
        # Look for action_stats.json in checkpoint directory
        possible_paths = [
            checkpoint_dir / "action_stats.json",
            checkpoint_dir.parent / "action_stats.json",
            checkpoint_dir.parent.parent / "action_stats.json",
        ]
        
        for stats_file in possible_paths:
            if stats_file.exists():
                print(f"Found action statistics at {stats_file}")
                with open(stats_file, 'r') as f:
                    return json.load(f)
        
        return None
    
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict action from observation (BaseVLAPolicy interface).
        
        Args:
            observation: Dictionary containing:
                - 'image': Image observation (H, W, C) or (C, H, W)
                - 'proprio': Optional proprioceptive state
            instruction: Language instruction (uses default if None)
            
        Returns:
            Predicted action as numpy array
        """
        if instruction is None:
            instruction = self._instruction or ""
        return self.predict(observation, instruction)
    
    def predict(
        self,
        observation: Dict[str, np.ndarray],
        instruction: str,
    ) -> np.ndarray:
        """
        Predict action given observation and instruction.
        
        Args:
            observation: Dictionary containing:
                - 'image': Image observation (H, W, C) or (C, H, W)
                - 'proprio': Optional proprioceptive state
            instruction: Language instruction
            
        Returns:
            Predicted action as numpy array
        """
        # Get image
        image = observation.get('image')
        if image is None:
            # Try alternative keys
            for key in ['rgb', 'sensor_data', 'images']:
                if key in observation:
                    image = observation[key]
                    if isinstance(image, dict):
                        # Get first camera
                        image = list(image.values())[0]
                    break
        
        if image is None:
            raise ValueError("No image found in observation")
        
        # Convert to torch tensor if needed
        if isinstance(image, np.ndarray):
            # Check if HWC or CHW format
            if image.shape[-1] == 3:
                # HWC -> CHW
                image = np.transpose(image, (2, 0, 1))
            
            # Normalize to [0, 1] if needed
            if image.max() > 1.0:
                image = image / 255.0
            
            image = torch.from_numpy(image).float()
        
        # Ensure on correct device
        image = image.to(self.device)
        
        # Predict action
        with torch.no_grad():
            action = self.model.predict_action(
                image=image,
                instruction=instruction,
                do_sample=False,
            )
        
        return action
    
    def reset(self):
        """Reset policy state (if any)."""
        pass
    
    def set_instruction(self, instruction: str):
        """
        Set default instruction for predict_action.
        
        Args:
            instruction: Language instruction to use
        """
        self._instruction = instruction
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda:0",
        statistics_path: Optional[str] = None,
        **kwargs,
    ) -> 'OpenVLAPolicy':
        """
        Load policy from checkpoint with automatic statistics loading.
        
        Args:
            checkpoint_path: Path to model checkpoint directory
            device: Device to load model on
            statistics_path: Optional path to statistics JSON file
            **kwargs: Additional arguments passed to __init__
            
        Returns:
            Loaded OpenVLAPolicy instance
        """
        action_statistics = None
        
        # Try to load statistics
        if statistics_path is not None:
            stats_file = Path(statistics_path)
        else:
            # Look for statistics in common locations
            checkpoint_dir = Path(checkpoint_path)
            possible_paths = [
                checkpoint_dir / "statistics.json",
                checkpoint_dir.parent / "statistics.json",
                checkpoint_dir.parent.parent / "statistics.json",
            ]
            for stats_file in possible_paths:
                if stats_file.exists():
                    break
            else:
                stats_file = None
        
        if stats_file and stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                action_statistics = stats.get('action', stats)
        
        return cls(
            checkpoint_path=checkpoint_path,
            device=device,
            action_statistics=action_statistics,
            **kwargs,
        )

