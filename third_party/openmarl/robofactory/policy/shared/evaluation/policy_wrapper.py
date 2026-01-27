"""
Policy wrapper interface for evaluation.

This module provides abstract base classes for policy wrappers that adapt
different policy implementations to a common evaluation interface.

Consolidated patterns from:
- Pi0PolicyWrapper: Action chunking, observation buffer
- OpenVLAPolicyWrapper: Simple prediction interface
- DP class: DPRunner integration, observation history
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np


class BasePolicyWrapper(ABC):
    """
    Abstract base class for policy wrappers in evaluation.
    
    All policy implementations should have a wrapper that inherits from this
    class to ensure they work with the unified evaluation framework.
    
    The wrapper handles:
    - Observation buffering/history
    - Action prediction interface
    - Optional action chunking
    """
    
    def __init__(
        self,
        task_name: str,
        device: str = 'cuda:0',
    ):
        """
        Initialize policy wrapper.
        
        Args:
            task_name: Name of the task being evaluated
            device: Device for inference
        """
        self.task_name = task_name
        self.device = device
        self.obs_buffer: List[Dict] = []
    
    @abstractmethod
    def load_policy(self, checkpoint_path: str, **kwargs):
        """
        Load policy from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            **kwargs: Additional arguments for specific implementations
        """
        pass
    
    def update_obs(self, observation: Dict[str, Any]):
        """
        Update observation buffer with new observation.
        
        Args:
            observation: Current observation dictionary
        """
        self.obs_buffer.append(observation)
        # Keep limited history
        if len(self.obs_buffer) > 10:
            self.obs_buffer.pop(0)
    
    @abstractmethod
    def get_action(
        self,
        observation: Optional[Dict[str, Any]] = None,
    ) -> List[np.ndarray]:
        """
        Get action(s) from policy.
        
        Args:
            observation: Current observation (if None, uses buffered observation)
            
        Returns:
            List of actions for multi-step execution compatibility.
            For single-step policies, this is typically [action] * num_repeat.
        """
        pass
    
    def get_last_obs(self) -> Optional[Dict[str, Any]]:
        """
        Get the last observation from buffer.
        
        Returns:
            Last observation or None if buffer is empty
        """
        return self.obs_buffer[-1] if self.obs_buffer else None
    
    def reset(self):
        """Reset the policy wrapper state."""
        self.obs_buffer = []


class ActionChunkingWrapper(BasePolicyWrapper):
    """
    Policy wrapper with action chunking support.
    
    For policies that predict action sequences (like Pi0, Diffusion-Policy),
    this wrapper manages an action buffer and returns actions one at a time.
    """
    
    def __init__(
        self,
        task_name: str,
        device: str = 'cuda:0',
        action_horizon: int = 50,
        action_repeat: int = 6,
    ):
        """
        Initialize action chunking wrapper.
        
        Args:
            task_name: Name of the task
            device: Device for inference
            action_horizon: Number of actions predicted at once
            action_repeat: Number of times to repeat action for return value
        """
        super().__init__(task_name, device)
        self.action_horizon = action_horizon
        self.action_repeat = action_repeat
        self.action_buffer: List[np.ndarray] = []
        self.action_idx = 0
    
    @abstractmethod
    def _predict_action_sequence(
        self,
        observation: Dict[str, Any],
    ) -> np.ndarray:
        """
        Predict action sequence from policy.
        
        Must be implemented by subclass.
        
        Args:
            observation: Current observation
            
        Returns:
            Action sequence with shape (action_horizon, action_dim)
        """
        pass
    
    def get_action(
        self,
        observation: Optional[Dict[str, Any]] = None,
    ) -> List[np.ndarray]:
        """
        Get action from buffer, predicting new sequence if needed.
        
        Args:
            observation: Current observation (if None, uses buffered)
            
        Returns:
            List of repeated actions for multi-step execution
        """
        # Use buffered observation if not provided
        if observation is None and len(self.obs_buffer) > 0:
            observation = self.obs_buffer[-1]
        
        if observation is None:
            raise ValueError("No observation available")
        
        # Check if we need to predict new action sequence
        if len(self.action_buffer) == 0 or self.action_idx >= len(self.action_buffer):
            # Predict new action sequence
            action_sequence = self._predict_action_sequence(observation)
            
            # Store in buffer
            if isinstance(action_sequence, np.ndarray) and action_sequence.ndim == 2:
                self.action_buffer = [action_sequence[i] for i in range(len(action_sequence))]
            else:
                self.action_buffer = [action_sequence]
            self.action_idx = 0
        
        # Get current action from buffer
        action = self.action_buffer[self.action_idx]
        self.action_idx += 1
        
        # Return repeated action for multi-step execution
        return [action for _ in range(self.action_repeat)]
    
    def reset(self):
        """Reset wrapper including action buffer."""
        super().reset()
        self.action_buffer = []
        self.action_idx = 0


class ObservationHistoryWrapper(BasePolicyWrapper):
    """
    Policy wrapper with observation history support.
    
    For policies that require observation history (like some Diffusion-Policy
    configurations), this wrapper manages stacked observations.
    """
    
    def __init__(
        self,
        task_name: str,
        device: str = 'cuda:0',
        n_obs_steps: int = 3,
        action_repeat: int = 6,
    ):
        """
        Initialize observation history wrapper.
        
        Args:
            task_name: Name of the task
            device: Device for inference
            n_obs_steps: Number of observation history steps required
            action_repeat: Number of times to repeat action for return value
        """
        super().__init__(task_name, device)
        self.n_obs_steps = n_obs_steps
        self.action_repeat = action_repeat
    
    def get_stacked_observations(self) -> Dict[str, np.ndarray]:
        """
        Get stacked observation history.
        
        Pads with first observation if history is incomplete.
        
        Returns:
            Dictionary with stacked observations
        """
        if len(self.obs_buffer) == 0:
            raise ValueError("No observations in buffer")
        
        # Pad with first observation if needed
        obs_list = list(self.obs_buffer)
        while len(obs_list) < self.n_obs_steps:
            obs_list.insert(0, obs_list[0])
        
        # Take only the last n_obs_steps
        obs_list = obs_list[-self.n_obs_steps:]
        
        # Stack observations by key
        result = {}
        for key in obs_list[0].keys():
            values = [obs[key] for obs in obs_list]
            if isinstance(values[0], np.ndarray):
                result[key] = np.stack(values, axis=0)
            else:
                result[key] = values
        
        return result
    
    @abstractmethod
    def _predict_action(
        self,
        stacked_observation: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Predict action from stacked observations.
        
        Must be implemented by subclass.
        
        Args:
            stacked_observation: Dictionary of stacked observations
            
        Returns:
            Action array
        """
        pass
    
    def get_action(
        self,
        observation: Optional[Dict[str, Any]] = None,
    ) -> List[np.ndarray]:
        """
        Get action using observation history.
        
        Args:
            observation: Current observation (added to buffer if provided)
            
        Returns:
            List of repeated actions for multi-step execution
        """
        # Add observation to buffer if provided
        if observation is not None:
            self.update_obs(observation)
        
        # Get stacked observations
        stacked_obs = self.get_stacked_observations()
        
        # Predict action
        action = self._predict_action(stacked_obs)
        
        # Return repeated action
        return [action for _ in range(self.action_repeat)]

