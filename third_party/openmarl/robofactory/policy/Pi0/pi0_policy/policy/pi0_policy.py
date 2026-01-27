"""
Pi0/Pi0.5 policy interface for RoboFactory evaluation.

This module provides the policy interface that connects to RoboFactory's
evaluation pipeline. Inherits from the core BaseVLAPolicy for interface
consistency across all VLA policy implementations.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from omegaconf import OmegaConf

# Disable torch.compile/dynamo for evaluation to avoid caching issues with patched functions
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from ..model.pi0_wrapper import Pi0Model

# Import core base class and shared utilities
from robofactory.policy.core import BaseVLAPolicy
from robofactory.policy.shared import get_task_instruction


class Pi0Policy(BaseVLAPolicy):
    """
    Pi0/Pi0.5 policy for RoboFactory evaluation.
    
    Provides a simple interface for action prediction during rollouts.
    Inherits from BaseVLAPolicy for consistent interface across VLA policies.
    
    Attributes:
        model: Pi0Model instance
        cfg: Configuration object
        language_instruction: Current language instruction
        action_horizon: Number of actions predicted at once
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        task_name: Optional[str] = None,
        device: str = "cuda:0",
        action_dim: int = 8,
    ):
        """
        Initialize policy from checkpoint.
        
        Args:
            checkpoint_path: Path to trained checkpoint directory
            config_path: Path to config YAML (optional, will try to infer)
            task_name: Task name for language instruction
            device: Device to run inference on
            action_dim: Dimension of action space
        """
        super().__init__(device=device, action_dim=action_dim)
        
        self.checkpoint_path = Path(checkpoint_path)
        self.task_name = task_name
        
        # Load config
        if config_path is None:
            # Try to find config in checkpoint directory or parent
            config_candidates = [
                self.checkpoint_path / "config.yaml",
                self.checkpoint_path.parent.parent / "config.yaml",
            ]
            config_path = next((c for c in config_candidates if c.exists()), None)
            
            if config_path is None:
                raise ValueError(f"Could not find config.yaml near {checkpoint_path}")
        
        self.cfg = OmegaConf.load(config_path)
        
        # Initialize model
        self.model = Pi0Model(
            model_variant=self.cfg.model.model_variant,
            paligemma_variant=self.cfg.model.paligemma_variant,
            action_expert_variant=self.cfg.model.action_expert_variant,
            pretrained_checkpoint=None,  # Will load from checkpoint
            action_dim=self.cfg.model.action_dim,
            action_horizon=self.cfg.model.action_horizon,
            max_token_len=self.cfg.model.max_token_len,
            torch_dtype=getattr(torch, self.cfg.model.dtype),
            pytorch_training_precision=self.cfg.model.pytorch_training_precision,
            device=device,
            use_gradient_checkpointing=False,  # Disable for inference
        )
        
        # Store action horizon for action chunking
        self.action_horizon = self.cfg.model.action_horizon
        
        # Load checkpoint weights
        self._load_checkpoint()
        
        # Set to eval mode
        self.model.eval()
        
        # Get language instruction using shared utilities
        if task_name:
            self.language_instruction = get_task_instruction(task_name, policy_type='pi0')
        else:
            self.language_instruction = "Complete the task"
        
        print(f"Initialized Pi0 policy from {checkpoint_path}")
        print(f"Language instruction: {self.language_instruction}")
        print(f"Action dim: {self.cfg.model.action_dim}, Horizon: {self.cfg.model.action_horizon}")
    
    def _load_checkpoint(self):
        """Load model weights from checkpoint."""
        model_file = self.checkpoint_path / "model.safetensors"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        import safetensors.torch
        state_dict = safetensors.torch.load_file(model_file, device='cpu')
        # Use strict=False because PaLI-Gemma has tied weights (embed_tokens shares with lm_head)
        self.model.model.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded model weights from {model_file}")
        
        # Load normalization stats if available
        metadata_file = self.checkpoint_path / "metadata.pt"
        if metadata_file.exists():
            metadata = torch.load(metadata_file, map_location='cpu')
            if "normalization_stats" in metadata:
                stats = metadata["normalization_stats"]
                self.model.set_normalization_statistics(
                    action_q01=stats["action_q01"],
                    action_q99=stats["action_q99"],
                    state_q01=stats["state_q01"],
                    state_q99=stats["state_q99"],
                )
                print(f"Loaded normalization statistics from checkpoint")
            else:
                print("WARNING: No normalization_stats in checkpoint metadata!")
                print("  This checkpoint was saved before the fix. Actions will be incorrectly scaled.")
                print("  Please retrain or manually add statistics to the checkpoint.")
    
    def predict_action(
        self,
        observation: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Predict action sequence from observation.
        
        Args:
            observation: Dict with keys:
                - "images": List/array of camera images [num_cameras, H, W, 3] or dict
                - "state": Robot state [state_dim]
                
        Returns:
            Action sequence array [action_horizon, action_dim] for action chunking.
            The ActionChunkingWrapper will buffer these and return one at a time.
        """
        # Prepare images
        if isinstance(observation["images"], dict):
            # Already in dict format with keys
            images = observation["images"]
        else:
            # Convert from array to dict (following RoboFactory -> Pi0 mapping)
            imgs = observation["images"]
            if isinstance(imgs, np.ndarray):
                # Assuming order: [head, global, wrist] as per data conversion
                images = {
                    "base_0_rgb": imgs[0],         # head/side
                    "left_wrist_0_rgb": imgs[1],   # global/overhead
                    "right_wrist_0_rgb": imgs[2],  # wrist/gripper
                }
            else:
                raise ValueError("Unexpected image format")
        
        # Convert to tensors and add batch dimension
        # OpenPI expects images in NHWC format [batch, height, width, channels]
        # with masks as 1D boolean tensors of shape [batch]
        image_tensors = {}
        image_masks = {}
        
        for key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            if key in images:
                img = images[key]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                
                # Ensure HWC format
                if img.ndim == 3 and img.shape[0] == 3:  # CHW -> HWC
                    img = img.permute(1, 2, 0)
                
                # Normalize to [0, 1] if needed
                if img.max() > 1.0:
                    img = img / 255.0
                
                # Add batch dimension -> [1, H, W, C] (NHWC)
                image_tensors[key] = img.unsqueeze(0).to(self.device)
                # Mask should be 1D tensor [batch] to match training collate format
                image_masks[key] = torch.tensor([True], dtype=torch.bool).to(self.device)
            else:
                # Create dummy if missing [1, H, W, C]
                image_tensors[key] = torch.zeros(1, 224, 224, 3).to(self.device)
                image_masks[key] = torch.tensor([False], dtype=torch.bool).to(self.device)
        
        # Prepare state [1, state_dim]
        state = torch.from_numpy(observation["state"]).float().unsqueeze(0).to(self.device)
        
        # Predict actions
        with torch.no_grad():
            action_sequence = self.model.predict(
                image=image_tensors,
                image_mask=image_masks,
                state=state,
                prompt=self.language_instruction,
            )
        
        # Return FULL action sequence for action chunking
        # model.predict() already removes batch dim, so action_sequence is [action_horizon, action_dim]
        # The ActionChunkingWrapper will buffer all actions and return them one at a time
        return action_sequence.cpu().numpy()
    
    def reset(self):
        """Reset policy state (if needed)."""
        pass
    
    def __call__(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Callable interface for compatibility."""
        return self.predict_action(observation)
    
    def set_instruction(self, instruction: str):
        """
        Set language instruction.
        
        Args:
            instruction: New language instruction
        """
        self.language_instruction = instruction
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda:0",
        task_name: Optional[str] = None,
        **kwargs,
    ) -> 'Pi0Policy':
        """
        Load policy from checkpoint.
        
        Factory method for VLA interface compatibility.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            device: Device to load model on
            task_name: Optional task name for instruction
            **kwargs: Additional arguments
            
        Returns:
            Loaded Pi0Policy instance
        """
        return cls(
            checkpoint_path=checkpoint_path,
            device=device,
            task_name=task_name,
            **kwargs,
        )

