"""
Template policy class for new VLA implementations.

This provides a minimal example of how to implement a new VLA policy
that inherits from BaseVLAPolicy.
"""

from typing import Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn

from robofactory.policy.core import BaseVLAPolicy
from robofactory.policy.shared import get_task_instruction


class SimpleModel(nn.Module):
    """Example simple model for demonstration."""
    
    def __init__(
        self,
        image_size: int = 224,
        state_dim: int = 8,
        action_dim: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.image_size = image_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Simple CNN for image encoding
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        # MLP for action prediction
        self.mlp = nn.Sequential(
            nn.Linear(64 + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image: Image tensor (B, C, H, W)
            state: State tensor (B, state_dim)
            
        Returns:
            Action tensor (B, action_dim)
        """
        # Encode image
        image_features = self.image_encoder(image)
        
        # Concatenate with state
        features = torch.cat([image_features, state], dim=-1)
        
        # Predict action
        action = self.mlp(features)
        return action


class NewPolicy(BaseVLAPolicy):
    """
    Template VLA policy implementation.
    
    Replace this with your actual policy implementation.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda:0",
        action_dim: int = 8,
        state_dim: int = 8,
        task_name: Optional[str] = None,
    ):
        """
        Initialize policy.
        
        Args:
            checkpoint_path: Path to checkpoint (optional)
            device: Device for inference
            action_dim: Dimension of action space
            state_dim: Dimension of state space
            task_name: Task name for instruction
        """
        super().__init__(device=device, action_dim=action_dim)
        
        self.state_dim = state_dim
        
        # Initialize model
        self.model = SimpleModel(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        # Set task instruction
        if task_name:
            self._instruction = get_task_instruction(task_name)
        else:
            self._instruction = "Complete the task"
        
        self.model.eval()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        self.model.load_state_dict(state_dict)
    
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict action from observation.
        
        Args:
            observation: Dictionary with 'image' and 'state' keys
            instruction: Language instruction (currently unused in simple model)
            
        Returns:
            Predicted action as numpy array
        """
        # Extract image
        image = observation.get('image')
        if image is None:
            raise ValueError("Observation must contain 'image' key")
        
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            # HWC -> CHW
            if image.shape[-1] == 3:
                image = np.transpose(image, (2, 0, 1))
            # Normalize
            if image.max() > 1.0:
                image = image / 255.0
            image = torch.from_numpy(image).float()
        
        # Add batch dimension
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Extract state
        state = observation.get('state', np.zeros(self.state_dim))
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        
        # Predict
        with torch.no_grad():
            action = self.model(image, state)
        
        return action[0].cpu().numpy()
    
    def reset(self):
        """Reset policy state."""
        pass
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda:0",
        task_name: Optional[str] = None,
        **kwargs,
    ) -> 'NewPolicy':
        """
        Load policy from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            task_name: Task name for instruction
            **kwargs: Additional arguments
            
        Returns:
            Loaded policy instance
        """
        return cls(
            checkpoint_path=checkpoint_path,
            device=device,
            task_name=task_name,
            **kwargs,
        )

