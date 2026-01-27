"""
Base policy class for Diffusion-Policy.

This module provides the base image policy class for Diffusion-Policy,
with VLA interface compatibility for integration with the unified framework.
"""

from typing import Dict, Optional, Any
import numpy as np
import torch
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer

class BaseImagePolicy(ModuleAttrMixin):
    """
    Base class for image-based Diffusion Policies.
    
    Provides the interface for:
    - predict_action: Get action from observation
    - reset: Reset policy state
    - set_normalizer: Set action/observation normalizer
    
    This class maintains backward compatibility while also supporting
    the unified VLA policy interface from robofactory.policy.core.
    
    Attributes:
        normalizer: LinearNormalizer for action/observation normalization
    """
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict action from observation dictionary.
        
        This is the main inference method for Diffusion-Policy.
        
        Args:
            obs_dict: Dictionary mapping observation names to tensors
                      Each tensor has shape (B, T_o, *) where B is batch size,
                      T_o is observation horizon
        
        Returns:
            Dictionary containing 'action' key with predicted actions
            Shape: (B, T_a, D_a) where T_a is action horizon, D_a is action dim
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset policy state for stateful policies.
        
        Called at the beginning of each episode to clear any internal state
        such as hidden states for recurrent policies.
        """
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        Set normalizer for action/observation normalization.
        
        Args:
            normalizer: LinearNormalizer instance with fitted statistics
        """
        raise NotImplementedError()
    
    # ========== VLA interface compatibility ===========
    def predict_action_vla(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        VLA-compatible action prediction interface.
        
        Wraps predict_action for compatibility with BaseVLAPolicy interface.
        
        Args:
            observation: Dictionary with observation data
            instruction: Language instruction (unused in Diffusion-Policy)
            
        Returns:
            Single action as numpy array
        """
        # Convert observation to expected format
        obs_dict = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value).float()
            elif isinstance(value, torch.Tensor):
                tensor = value.float()
            else:
                continue
            
            # Add batch and time dimensions if needed
            if tensor.ndim == 3:  # (C, H, W) -> (1, 1, C, H, W)
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.ndim == 1:  # (D,) -> (1, 1, D)
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            
            obs_dict[key] = tensor
        
        # Predict action
        with torch.no_grad():
            result = self.predict_action(obs_dict)
        
        # Extract first action from sequence
        action = result['action'][0, 0].cpu().numpy()
        return action

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda:0",
        **kwargs,
    ) -> 'BaseImagePolicy':
        """
        Load policy from checkpoint.
        
        Factory method for VLA interface compatibility.
        Note: Actual implementation should be in subclasses.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            **kwargs: Additional arguments
            
        Returns:
            Loaded policy instance
        """
        raise NotImplementedError("from_checkpoint should be implemented in subclass")
