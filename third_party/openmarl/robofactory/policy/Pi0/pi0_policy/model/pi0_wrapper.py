"""
Pi0/Pi0.5 model wrapper for RoboFactory.

This module provides a wrapper around the openpi Pi0 PyTorch models,
adapting them for RoboFactory's multi-agent manipulation tasks.

The wrapper uses openpi as an external dependency and follows
openpi's data format and training patterns.
"""
import os
import torch
import torch.nn as nn
from pathlib import Path
import logging
import torch.distributed as dist
from typing import Optional, Dict, Tuple, List

# Disable torch.compile/dynamo to avoid issues with patched functions
# This is particularly important for evaluation where we monkey-patch
# the image format conversion
import torch._dynamo
torch._dynamo.config.suppress_errors = True

try:
    # Import from openpi (treated as external dependency)
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import download
    from openpi.models import model as _model
    import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
except ImportError as e:
    print(f"Error importing openpi: {e}")
    print("Please ensure openpi is installed: pip install git+https://github.com/Physical-Intelligence/openpi.git@main")
    raise


class Pi0Model(nn.Module):
    """
    Wrapper for Pi0/Pi0.5 models from openpi.
    
    Handles:
    - Model loading from pretrained checkpoints
    - Action prediction with flow matching
    - Multi-GPU compatibility
    - Action normalization/denormalization (using openpi's quantile normalization)
    """
    
    def __init__(
        self,
        model_variant: str = "pi0",  # "pi0" or "pi05"
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        pretrained_checkpoint: Optional[str] = None,
        action_dim: int = 8,
        action_horizon: int = 50,
        max_token_len: Optional[int] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        pytorch_training_precision: str = "bfloat16",
        device: str = "cuda:0",
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        self.model_variant = model_variant
        self.action_dim = action_dim  # Actual action dim for output (e.g., 8)
        self.action_horizon = action_horizon
        self.device = device
        self.pytorch_training_precision = pytorch_training_precision
        
        # Pretrained Pi0 models use 32-dim actions internally
        # We must use 32 for model config to match pretrained weights
        pretrained_action_dim = 32
        
        # Create Pi0 config (following openpi's Pi0Config)
        # Use pretrained_action_dim=32 to match pretrained weights
        config = Pi0Config(
            dtype=str(torch_dtype).split('.')[-1],  # "bfloat16" or "float32"
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
            action_dim=pretrained_action_dim,  # Must be 32 to match pretrained weights
            action_horizon=action_horizon,
            max_token_len=max_token_len or (200 if model_variant == "pi05" else 48),
            pi05=(model_variant == "pi05"),
        )
        
        self.config = config
        
        # Initialize model (openpi's PI0Pytorch)
        print(f"Initializing {model_variant} model with config:")
        print(f"  - paligemma_variant: {paligemma_variant}")
        print(f"  - action_expert_variant: {action_expert_variant}")
        print(f"  - action_dim (internal): {pretrained_action_dim}")
        print(f"  - action_dim (output): {action_dim}")
        print(f"  - action_horizon: {action_horizon}")
        print(f"  - max_token_len: {config.max_token_len}")
        print(f"  - precision: {pytorch_training_precision}")
        
        self.model = PI0Pytorch(config)
        
        # Monkey-patch to fix NHWC -> NCHW conversion issue
        # openpi's preprocessing outputs NHWC but vision model expects NCHW
        self._patch_image_format_conversion()
        
        # Load pretrained weights if provided
        if pretrained_checkpoint:
            self.load_pretrained(pretrained_checkpoint)
        
        # Set precision for training
        if pytorch_training_precision == "bfloat16":
            self.model = self.model.to(torch.bfloat16)
        
        # Move to device
        self.model = self.model.to(device)
        
        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Action statistics (for normalization following openpi's quantile normalization)
        self.register_buffer('action_q01', torch.zeros(action_dim))
        self.register_buffer('action_q99', torch.ones(action_dim))
        self.register_buffer('state_q01', torch.zeros(action_dim))
        self.register_buffer('state_q99', torch.ones(action_dim))
        
        logging.info(f"Initialized {model_variant} model on {device}")
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def _patch_image_format_conversion(self):
        """
        Monkey-patch the model to convert images from NHWC to NCHW.
        
        openpi's preprocessing outputs NHWC format (batch, height, width, channels),
        but the SigLIP vision model expects NCHW format (batch, channels, height, width).
        This patch intercepts the PaliGemma forward call to convert format.
        """
        # Patch at the paligemma.model.get_image_features level
        # This is the actual transformers PaliGemma model that expects NCHW
        paligemma = self.model.paligemma_with_expert.paligemma
        original_get_image_features = paligemma.model.get_image_features
        
        def patched_get_image_features(pixel_values, **kwargs):
            # Handle various input formats and convert to NCHW
            if pixel_values.ndim == 3:
                # 3D tensor: either [H, W, C] or [C, H, W]
                if pixel_values.shape[-1] in [1, 3, 4]:  # [H, W, C]
                    pixel_values = pixel_values.permute(2, 0, 1).unsqueeze(0)
                elif pixel_values.shape[0] in [1, 3, 4]:  # [C, H, W]
                    pixel_values = pixel_values.unsqueeze(0)
                else:
                    pixel_values = pixel_values.unsqueeze(0)
            elif pixel_values.ndim == 4:
                # 4D tensor: either [B, H, W, C] or [B, C, H, W]
                if pixel_values.shape[-1] in [1, 3, 4] and pixel_values.shape[1] > 4:  # [B, H, W, C]
                    pixel_values = pixel_values.permute(0, 3, 1, 2)  # -> [B, C, H, W]
            
            return original_get_image_features(pixel_values, **kwargs)
        
        paligemma.model.get_image_features = patched_get_image_features
        print("✓ Patched get_image_features for NHWC -> NCHW conversion")
    
    def load_pretrained(self, checkpoint_path: str):
        """
        Load pretrained weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (can be GCS path or local path)
        """
        
        # Get distributed training info
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        is_distributed = world_size > 1
        
        if rank == 0:
            print(f"Loading pretrained weights from {checkpoint_path}...")
        
        # Download from GCS if needed (using openpi's download utility)
        # Only rank 0 downloads to avoid race condition on shared filesystem
        if rank == 0:
            local_path = download.maybe_download(checkpoint_path)
            print(f"Downloaded checkpoint to: {local_path}")
        
        # Wait for rank 0 to finish downloading
        if is_distributed and dist.is_initialized():
            dist.barrier()
        
        # Other ranks now get the local path (should be cached)
        if rank != 0:
            local_path = download.maybe_download(checkpoint_path)
        
        # OpenPI checkpoints are in Orbax format (JAX), need to load and convert to PyTorch
        # The checkpoint structure is: checkpoint_dir/params/ (Orbax format)
        params_path = Path(local_path) / "params"
        if not params_path.exists():
            raise FileNotFoundError(f"Checkpoint params directory not found: {params_path}")
        
        print(f"Loading Orbax checkpoint from {params_path}...")
        
        # Import required modules for loading Orbax checkpoints
        import numpy as np
        import orbax.checkpoint as ocp
        
        # Load params directly with Orbax PyTreeCheckpointer
        with ocp.PyTreeCheckpointer() as ckptr:
            restored = ckptr.restore(params_path)
        
        # Extract params from the restored structure
        if isinstance(restored, dict) and 'params' in restored:
            jax_params = restored['params']
        else:
            jax_params = restored
        
        print(f"Loaded checkpoint with {len(jax_params)} top-level keys")
        
        # Convert JAX params structure to PyTorch state dict
        print("Converting JAX parameters to PyTorch format...")
        pytorch_state_dict = self._convert_jax_to_pytorch(jax_params)
        
        # Load into model with strict=False to allow for architecture differences
        missing_keys, unexpected_keys = self.model.load_state_dict(pytorch_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Warning: Unexpected keys: {unexpected_keys}")
        
        print(f"✓ Loaded pretrained weights from {checkpoint_path}")
    
    def _convert_jax_to_pytorch(self, jax_params: dict) -> dict:
        """
        Convert JAX parameter structure to PyTorch state dict.
        
        JAX uses nested dicts with structure like:
        {'PaliGemmaWithExpert': {'img_model': {'encoder': {'layers': {'0': {'mlp': {'wi': array(...)}}}}}}
        
        PyTorch uses flat keys like:
        'paligemma_with_expert.img_model.encoder.layers.0.mlp.wi.weight'
        
        Args:
            jax_params: Nested dict of JAX parameters (numpy arrays)
            
        Returns:
            Flat dict suitable for PyTorch load_state_dict
        """
        import numpy as np
        import flax.traverse_util
        
        def _flatten_and_convert(params, parent_key='', sep='.'):
            """Recursively flatten nested dict and convert arrays to tensors."""
            items = []
            for k, v in params.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                
                # Convert PascalCase to snake_case for PyTorch convention
                new_key = self._pascal_to_snake(new_key)
                
                if isinstance(v, dict):
                    items.extend(_flatten_and_convert(v, new_key, sep=sep).items())
                else:
                    # Convert JAX array or numpy array to torch tensor
                    if hasattr(v, '__array__'):  # Works for both JAX arrays and numpy arrays
                        array = np.asarray(v)  # Convert JAX array to numpy first
                        tensor = torch.from_numpy(array)
                    else:
                        tensor = v
                    
                    # JAX uses (out_features, in_features) for Linear layers
                    # PyTorch also uses (out_features, in_features) BUT for nn.Linear.weight
                    # However, JAX's convention for matmul is transposed from PyTorch's
                    # So we need to transpose weight matrices
                    
                    # Handle different parameter types
                    is_weight = 'kernel' in new_key or 'wi' in new_key or 'wo' in new_key or 'weight' in new_key
                    is_bias = 'bias' in new_key
                    is_norm = 'scale' in new_key or 'gamma' in new_key or 'norm' in new_key
                    
                    if is_weight and not is_norm:
                        # This is a linear layer weight - need to transpose if it's 2D
                        if tensor.ndim == 2:
                            tensor = tensor.T  # Transpose from JAX convention to PyTorch
                        
                        # Update key naming
                        new_key = new_key.replace('kernel', 'weight').replace('wi', 'weight').replace('wo', 'weight')
                        if not new_key.endswith('.weight'):
                            new_key += '.weight'
                    elif is_bias:
                        if not new_key.endswith('.bias'):
                            new_key += '.bias'
                    elif is_norm:
                        # LayerNorm parameters
                        new_key = new_key.replace('scale', 'weight').replace('gamma', 'weight')
                        if not new_key.endswith('.weight') and not new_key.endswith('.bias'):
                            new_key += '.weight'
                    
                    items.append((new_key, tensor))
            
            return dict(items)
        
        pytorch_dict = _flatten_and_convert(jax_params)
        print(f"Converted {len(pytorch_dict)} parameters from JAX to PyTorch format")
        
        return pytorch_dict
    
    @staticmethod
    def _pascal_to_snake(name: str) -> str:
        """Convert PascalCase to snake_case."""
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        return name.lower()
    
    def set_normalization_statistics(
        self, 
        action_q01: torch.Tensor,
        action_q99: torch.Tensor,
        state_q01: torch.Tensor,
        state_q99: torch.Tensor,
    ):
        """
        Set normalization statistics (following openpi's quantile normalization).
        
        Args:
            action_q01: 1st percentile of actions
            action_q99: 99th percentile of actions
            state_q01: 1st percentile of states
            state_q99: 99th percentile of states
        """
<<<<<<< Current (Your changes)
        # Convert to float32 to avoid dtype promotion issues with numpy-derived float64 tensors
        self.action_q01 = action_q01.float().to(self.device)
        self.action_q99 = action_q99.float().to(self.device)
        self.state_q01 = state_q01.float().to(self.device)
        self.state_q99 = state_q99.float().to(self.device)
=======
        # Convert to float32 and move to device
        action_q01 = action_q01.float().to(self.device)
        action_q99 = action_q99.float().to(self.device)
        state_q01 = state_q01.float().to(self.device)
        state_q99 = state_q99.float().to(self.device)
        
        # =====================================================================
        # FIX: Handle dimension mismatch for checkpoints trained with wrong state dims
        # Training data may have used 9-dim qpos (7 joints + 2 gripper fingers)
        # But evaluation uses 8-dim state (7 joints + 1 gripper command)
        # =====================================================================
        if state_q01.shape[-1] != self.action_dim:
            print(f"\n⚠️  WARNING: State statistics dimension mismatch!")
            print(f"   Checkpoint has {state_q01.shape[-1]}-dim states, model expects {self.action_dim}-dim")
            if state_q01.shape[-1] == 9 and self.action_dim == 8:
                # Checkpoint trained with 9-dim qpos (7 joints + 2 gripper fingers)
                # Truncate to 8-dim: keep 7 joints + average the 2 gripper values
                print(f"   Applying fix: Using first 7 joints + average of last 2 gripper dims")
                state_q01_fixed = torch.cat([
                    state_q01[..., :7],  # 7 joints
                    state_q01[..., 7:9].mean(dim=-1, keepdim=True)  # average gripper
                ])
                state_q99_fixed = torch.cat([
                    state_q99[..., :7],  # 7 joints
                    state_q99[..., 7:9].mean(dim=-1, keepdim=True)  # average gripper
                ])
                state_q01 = state_q01_fixed
                state_q99 = state_q99_fixed
                print(f"   ✓ Fixed: State statistics now {self.action_dim}-dim\n")
            else:
                # Other mismatch - just truncate
                print(f"   Truncating to first {self.action_dim} dimensions\n")
                state_q01 = state_q01[..., :self.action_dim]
                state_q99 = state_q99[..., :self.action_dim]
        
        # Similar check for actions
        if action_q01.shape[-1] != self.action_dim:
            print(f"⚠️  WARNING: Action statistics dimension mismatch!")
            print(f"   Checkpoint has {action_q01.shape[-1]}-dim actions, model expects {self.action_dim}-dim")
            print(f"   Truncating to first {self.action_dim} dimensions\n")
            action_q01 = action_q01[..., :self.action_dim]
            action_q99 = action_q99[..., :self.action_dim]
        
        self.action_q01 = action_q01
        self.action_q99 = action_q99
        self.state_q01 = state_q01
        self.state_q99 = state_q99
>>>>>>> Incoming (Background Agent changes)
    
    def normalize_quantile(self, data: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
        """Normalize data using quantile normalization (openpi convention)."""
        return (data - q01) / (q99 - q01 + 1e-8)
    
    def denormalize_quantile(self, data: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to original scale."""
        return data * (q99 - q01) + q01
    
    def forward(
        self,
        image: Dict[str, torch.Tensor],
        image_mask: Dict[str, torch.Tensor],
        state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        prompt: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for training.
        
        Args:
            image: Dict of camera images {"base_0_rgb": ..., "left_wrist_0_rgb": ..., "right_wrist_0_rgb": ...}
            image_mask: Dict of image validity masks
            state: Robot proprioception [batch, state_dim]
            actions: Action sequences [batch, action_horizon, action_dim] (for training)
            prompt: List of language instructions
        
        Returns:
            loss (if training), predicted_actions (if inference)
        """
        batch_size = state.shape[0]
        
        # Prepare observation in openpi format
        # OpenPI expects an object with specific attributes (not a dict)
        from types import SimpleNamespace
        
        # Get batch size for creating dummy tokens
        batch_size = state.shape[0] if state.ndim > 1 else 1
        max_token_len = self.config.max_token_len
        
        # Pi0/Pi0.5 pretrained models have specific dimension requirements
        # Pi0: state goes through state_proj (32 dims required)
        # Pi0.5: state is tokenized as discrete input (NOT passed through state_proj)
        # Both: actions always need 32 dimensions
        pretrained_dim = 32
        is_pi05 = self.config.pi05
        
        # =====================================================================
        # FIX: Normalize state BEFORE padding (Pi0 only)
        # Pi0.5 uses discrete state tokenization so skip normalization
        # =====================================================================
        if not is_pi05:
            # Normalize state using quantile normalization
            state = self.normalize_quantile(state, self.state_q01, self.state_q99)
            
            # Pad state from 8 -> 32 dimensions
            current_state_dim = state.shape[-1]
            if current_state_dim < pretrained_dim:
                state_padding = torch.zeros(
                    (*state.shape[:-1], pretrained_dim - current_state_dim),
                    dtype=state.dtype,
                    device=state.device
                )
                state = torch.cat([state, state_padding], dim=-1)
        
        # =====================================================================
        # FIX: Normalize actions BEFORE padding (both Pi0 and Pi0.5)
        # =====================================================================
        if actions is not None:
            # Normalize actions using quantile normalization
            actions = self.normalize_quantile(actions, self.action_q01, self.action_q99)
            
            # Pad actions from 8 -> 32 dimensions
            current_action_dim = actions.shape[-1]
            if current_action_dim < pretrained_dim:
                action_padding = torch.zeros(
                    (*actions.shape[:-1], pretrained_dim - current_action_dim),
                    dtype=actions.dtype,
                    device=actions.device
                )
                actions = torch.cat([actions, action_padding], dim=-1)
        
        # Tokenize prompt if provided, otherwise create empty tokens
        if prompt is not None and prompt != "":
            # For now, create dummy token tensors since we don't have access to the gated PaliGemma tokenizer
            # In production, you would use: tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
            # Create dummy tokens of shape [batch_size, max_token_len]
            # Use token_id = 0 for empty/padding
            tokenized_prompt = torch.zeros((batch_size, max_token_len), dtype=torch.long, device=self.device)
            tokenized_prompt_mask = torch.zeros((batch_size, max_token_len), dtype=torch.bool, device=self.device)
            token_ar_mask = torch.zeros((batch_size, max_token_len), dtype=torch.bool, device=self.device)
            token_loss_mask = torch.zeros((batch_size, max_token_len), dtype=torch.bool, device=self.device)
        else:
            # No prompt - create empty token tensors
            tokenized_prompt = torch.zeros((batch_size, max_token_len), dtype=torch.long, device=self.device)
            tokenized_prompt_mask = torch.zeros((batch_size, max_token_len), dtype=torch.bool, device=self.device)
            token_ar_mask = torch.zeros((batch_size, max_token_len), dtype=torch.bool, device=self.device)
            token_loss_mask = torch.zeros((batch_size, max_token_len), dtype=torch.bool, device=self.device)
        
        observation = SimpleNamespace(
            images=image,  # Dict of images (stay in float32 for OpenPI preprocessing)
            image_masks=image_mask,  # Dict of image masks
            state=state,   # State tensor (now normalized for Pi0, unchanged for Pi0.5)
            tokenized_prompt=tokenized_prompt,  # Tokenized prompt tensor
            tokenized_prompt_mask=tokenized_prompt_mask,  # Attention mask for prompt
            token_ar_mask=token_ar_mask,  # Autoregressive mask for tokens
            token_loss_mask=token_loss_mask,  # Loss mask for tokens
        )
        
        # Also store raw prompt if needed
        if prompt is not None:
            observation.prompt = prompt
        
        if actions is not None:
            # Training mode: compute loss
            # Use autocast for proper mixed-precision handling
            # This ensures dtype consistency throughout OpenPI's internal computations
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = self.model.forward(observation, actions)
            # Reduce to scalar if loss has multiple elements (e.g., per-sample losses)
            if loss.numel() > 1:
                loss = loss.mean()
            return loss, None
        else:
            # Inference mode: sample actions
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred_actions = self.model.sample_actions(self.device, observation)
            # Truncate actions from padded 32 dims back to actual action_dim
            # pred_actions shape: [batch, action_horizon, 32] -> [batch, action_horizon, action_dim]
            pred_actions = pred_actions[..., :self.action_dim]
            
            # =====================================================================
            # FIX: Denormalize predicted actions back to original scale
            # =====================================================================
            pred_actions = self.denormalize_quantile(pred_actions, self.action_q01, self.action_q99)
            
            return None, pred_actions

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            batch: Batch dictionary from dataloader
            
        Returns:
            Loss scalar
        """
        loss, _ = self.forward(
            image=batch["image"],
            image_mask=batch["image_mask"],
            state=batch["state"],
            actions=batch["actions"],
            prompt=batch.get("prompt"),
        )
        return loss
    
    def predict(
        self,
        image: Dict[str, torch.Tensor],
        image_mask: Dict[str, torch.Tensor],
        state: torch.Tensor,
        prompt: str,
    ) -> torch.Tensor:
        """
        Predict actions for inference.
        
        Args:
            image: Dict of camera images
            image_mask: Dict of image validity masks
            state: Current robot state
            prompt: Language instruction
        
        Returns:
            Action sequence [action_horizon, action_dim]
        """
        with torch.no_grad():
            _, actions = self.forward(
                image=image,
                image_mask=image_mask,
                state=state,
                actions=None,
                prompt=[prompt],
            )
        
        return actions[0]  # Remove batch dimension

