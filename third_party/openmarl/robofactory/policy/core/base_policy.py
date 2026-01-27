"""
Abstract base policy interface for VLA policies.

This module provides the BaseVLAPolicy abstract class that defines the standard
interface for all VLA policy implementations. This ensures consistent behavior
across different policy types (Diffusion-Policy, OpenVLA, Pi0, etc.) and enables
the unified evaluation framework.

Patterns extracted from:
- BaseImagePolicy (Diffusion-Policy): predict_action, reset, set_normalizer
- OpenVLAPolicy: predict, reset, from_checkpoint pattern
- Pi0Policy: predict_action, reset, callable interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import numpy as np
import torch


class BaseVLAPolicy(ABC):
    """
    Abstract base class for VLA (Vision-Language-Action) policies.
    
    All VLA policy implementations should inherit from this class and
    implement the abstract methods. This provides a consistent interface
    for training, evaluation, and deployment.
    
    Attributes:
        device: Device the policy is running on
        action_dim: Dimension of the action space
    """
    
    def __init__(self, device: str = "cuda:0", action_dim: int = 8):
        """
        Initialize base policy.
        
        Args:
            device: Device to run inference on
            action_dim: Dimension of action space
        """
        self.device = device
        self.action_dim = action_dim
    
    @abstractmethod
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict action from observation.
        
        This is the main inference method that all policies must implement.
        
        Args:
            observation: Dictionary containing observation data, typically:
                - 'images': Camera images (single or multi-view)
                         Can be np.ndarray [num_cameras, H, W, 3] or dict of images
                - 'state': Robot proprioceptive state [state_dim]
            instruction: Optional language instruction for VLA models
            
        Returns:
            Predicted action as numpy array with shape [action_dim]
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        Reset policy state.
        
        Called at the beginning of each episode. Implementations should reset
        any internal state such as:
        - Action buffers (for action chunking)
        - Observation history (for temporal models)
        - Hidden states (for recurrent models)
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda:0",
        **kwargs,
    ) -> 'BaseVLAPolicy':
        """
        Load policy from checkpoint.
        
        Factory method to create a policy instance from a saved checkpoint.
        Implementations should handle their specific checkpoint format.
        
        Args:
            checkpoint_path: Path to checkpoint file or directory
            device: Device to load model on
            **kwargs: Additional arguments for specific implementations
            
        Returns:
            Loaded policy instance
        """
        pass
    
    def __call__(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Callable interface for compatibility.
        
        Allows using the policy as a callable: action = policy(obs)
        
        Args:
            observation: Observation dictionary
            instruction: Optional language instruction
            
        Returns:
            Predicted action
        """
        return self.predict_action(observation, instruction)
    
    def to(self, device: str) -> 'BaseVLAPolicy':
        """
        Move policy to device.
        
        Args:
            device: Target device (e.g., 'cuda:0', 'cpu')
            
        Returns:
            Self for chaining
        """
        self.device = device
        return self
    
    def eval(self) -> 'BaseVLAPolicy':
        """
        Set policy to evaluation mode.
        
        Returns:
            Self for chaining
        """
        return self
    
    def train(self, mode: bool = True) -> 'BaseVLAPolicy':
        """
        Set policy to training mode.
        
        Args:
            mode: If True, set to training mode; otherwise evaluation mode
            
        Returns:
            Self for chaining
        """
        return self


class BaseActionChunkingPolicy(BaseVLAPolicy):
    """
    Base class for policies that use action chunking.
    
    Action chunking predicts a sequence of future actions and executes them
    one at a time. This is common in VLA models like Pi0 and Diffusion-Policy.
    
    Attributes:
        action_horizon: Number of actions to predict at once
        action_buffer: Buffer storing predicted action sequence
        action_idx: Current index in the action buffer
    """
    
    def __init__(
        self,
        device: str = "cuda:0",
        action_dim: int = 8,
        action_horizon: int = 50,
    ):
        """
        Initialize action chunking policy.
        
        Args:
            device: Device to run inference on
            action_dim: Dimension of action space
            action_horizon: Number of future actions to predict
        """
        super().__init__(device=device, action_dim=action_dim)
        self.action_horizon = action_horizon
        self.action_buffer = []
        self.action_idx = 0
    
    def reset(self):
        """Reset action buffer and index."""
        self.action_buffer = []
        self.action_idx = 0
    
    @abstractmethod
    def _predict_action_sequence(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict action sequence from observation.
        
        Must be implemented by subclass.
        
        Args:
            observation: Observation dictionary
            instruction: Optional language instruction
            
        Returns:
            Action sequence with shape [action_horizon, action_dim]
        """
        pass
    
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get single action, managing action buffer internally.
        
        If buffer is empty or exhausted, predict new action sequence.
        Returns the next action from the buffer.
        
        Args:
            observation: Observation dictionary
            instruction: Optional language instruction
            
        Returns:
            Single action with shape [action_dim]
        """
        # Check if we need to predict new action sequence
        if len(self.action_buffer) == 0 or self.action_idx >= len(self.action_buffer):
            # Predict new action sequence
            action_sequence = self._predict_action_sequence(observation, instruction)
            
            # Store in buffer
            self.action_buffer = [action_sequence[i] for i in range(len(action_sequence))]
            self.action_idx = 0
        
        # Get current action from buffer
        action = self.action_buffer[self.action_idx]
        self.action_idx += 1
        
        return action


class BaseObservationHistoryPolicy(BaseVLAPolicy):
    """
    Base class for policies that use observation history.
    
    Some policies (like Diffusion-Policy) require a history of observations
    rather than just the current observation.
    
    Attributes:
        n_obs_steps: Number of observation steps to maintain
        obs_buffer: Buffer storing observation history
    """
    
    def __init__(
        self,
        device: str = "cuda:0",
        action_dim: int = 8,
        n_obs_steps: int = 3,
    ):
        """
        Initialize observation history policy.
        
        Args:
            device: Device to run inference on
            action_dim: Dimension of action space
            n_obs_steps: Number of observation history steps
        """
        super().__init__(device=device, action_dim=action_dim)
        self.n_obs_steps = n_obs_steps
        self.obs_buffer = []
    
    def reset(self):
        """Reset observation buffer."""
        self.obs_buffer = []
    
    def update_observation(self, observation: Dict[str, Any]):
        """
        Add observation to history buffer.
        
        Args:
            observation: Current observation dictionary
        """
        self.obs_buffer.append(observation)
        if len(self.obs_buffer) > self.n_obs_steps:
            self.obs_buffer.pop(0)
    
    def get_stacked_observations(self) -> Dict[str, Any]:
        """
        Get stacked observation history.
        
        Pads with the first observation if history is incomplete.
        
        Returns:
            Dictionary with stacked observations
        """
        if len(self.obs_buffer) == 0:
            raise RuntimeError("No observations in buffer. Call update_observation first.")
        
        # Pad with first observation if needed
        obs_list = list(self.obs_buffer)
        while len(obs_list) < self.n_obs_steps:
            obs_list.insert(0, obs_list[0])
        
        # Stack observations
        result = {}
        for key in obs_list[0].keys():
            values = [obs[key] for obs in obs_list]
            if isinstance(values[0], np.ndarray):
                result[key] = np.stack(values, axis=0)
            elif isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values, dim=0)
            else:
                result[key] = values
        
        return result

