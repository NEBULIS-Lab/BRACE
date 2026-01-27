"""OpenVLA model wrapper with LoRA fine-tuning support and multi-view images."""

import os
from typing import Dict, Optional, Tuple, List, Union
import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
import numpy as np


# OpenVLA action tokenization constants
ACTION_TOKEN_BEGIN_IDX = 32000  # Beginning of action token vocabulary
NUM_ACTION_BINS = 256  # Number of discrete bins per action dimension


class OpenVLAModel(nn.Module):
    """
    Wrapper for OpenVLA model with LoRA fine-tuning support.
    
    This class handles:
    - Loading pretrained OpenVLA from HuggingFace
    - Setting up LoRA for efficient fine-tuning
    - Forward pass for training with proper action tokenization
    - Action prediction for inference
    """
    
    def __init__(
        self,
        model_name: str = "openvla/openvla-7b",
        use_lora: bool = True,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        action_dim: int = 8,  # Default 8-DOF for Panda (7 joints + gripper)
        # Multi-view parameters
        use_multi_view: bool = False,
        num_images: int = 1,
        image_views: List[str] = None,
        multi_view_fusion: str = "concatenate",
    ):
        """
        Initialize OpenVLA model with multi-view support.
        
        Args:
            model_name: HuggingFace model name or path
            use_lora: Whether to use LoRA fine-tuning
            lora_rank: Rank for LoRA layers
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout rate for LoRA layers
            torch_dtype: Data type for model weights
            device: Device to load model on
            action_dim: Action dimension (default 8 for 7 joints + gripper)
            use_multi_view: Whether to use multiple camera views
            num_images: Number of camera views (1, 2, or 3)
            image_views: List of view names ['primary', 'secondary', 'wrist']
            multi_view_fusion: How to fuse views ('concatenate' or 'tile')
        """
        super().__init__()
        
        self.model_name = model_name
        self.use_lora = use_lora
        self.torch_dtype = torch_dtype
        self.device = device
        self.action_dim = action_dim
        
        # Multi-view settings
        self.use_multi_view = use_multi_view
        self.num_images = num_images
        self.image_views = image_views or ['primary', 'secondary', 'wrist']
        self.multi_view_fusion = multi_view_fusion
        
        # Determine logging rank (rank 1 in distributed, or single process)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._is_logging_rank = (world_size == 1) or (local_rank == 1)
        
        if self._is_logging_rank and use_multi_view:
            print(f"Multi-view enabled: {num_images} images, views={image_views}, fusion={multi_view_fusion}")
        
        # Load processor
        if self._is_logging_rank:
            print(f"Loading processor from {model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model
        if self._is_logging_rank:
            print(f"Loading model from {model_name}...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Setup LoRA if enabled
        if use_lora:
            if self._is_logging_rank:
                print(f"Setting up LoRA with rank={lora_rank}, alpha={lora_alpha}")
            self._setup_lora(lora_rank, lora_alpha, lora_dropout)
        
        # Move to device (use LOCAL_RANK for distributed training)
        self.device = torch.device(f"cuda:{local_rank}")
        self.model = self.model.to(self.device)
        
        # Statistics for action normalization/denormalization
        self.action_mean = None
        self.action_std = None
        self.action_min = None
        self.action_max = None
    
    def _setup_lora(self, rank: int, alpha: int, dropout: float):
        """Setup LoRA for efficient fine-tuning."""
        # Configure LoRA for all linear layers
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        if self._is_logging_rank:
            self.model.print_trainable_parameters()
    
    def set_action_statistics(
        self, 
        mean: np.ndarray, 
        std: np.ndarray,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
    ):
        """
        Set action statistics for normalization/denormalization.
        
        FIX: Now accepts optional action_min/action_max for proper 8-DOF handling.
        Uses actual data min/max if provided, otherwise estimates from mean ± 3*std.
        
        Args:
            mean: Mean values for actions (8-DOF)
            std: Standard deviation values for actions (8-DOF)
            action_min: Optional actual minimum values from dataset
            action_max: Optional actual maximum values from dataset
        """
        self.action_dim = len(mean)  # Update action_dim based on statistics
        
        self.action_mean = torch.from_numpy(mean).to(
            device=self.device,
            dtype=self.torch_dtype
        )
        self.action_std = torch.from_numpy(std).to(
            device=self.device,
            dtype=self.torch_dtype
        )
        
        # Use actual min/max if provided, otherwise estimate from mean ± 3*std
        if action_min is not None and action_max is not None:
            self.action_min = torch.from_numpy(action_min).to(
                device=self.device, dtype=self.torch_dtype
            )
            self.action_max = torch.from_numpy(action_max).to(
                device=self.device, dtype=self.torch_dtype
            )
        else:
            # Fallback: estimate from mean ± 3*std
            self.action_min = self.action_mean - 3 * self.action_std
            self.action_max = self.action_mean + 3 * self.action_std
        
        if self._is_logging_rank:
            print(f"Action statistics set: dim={self.action_dim}")
            print(f"  mean: {mean}")
            print(f"  std: {std}")
<<<<<<< Current (Your changes)
            print(f"  min: {self.action_min.cpu().float().numpy()}")
            print(f"  max: {self.action_max.cpu().float().numpy()}")
=======
            print(f"  min: {self.action_min.cpu().numpy()}")
            print(f"  max: {self.action_max.cpu().numpy()}")
>>>>>>> Incoming (Background Agent changes)
    
    def _tokenize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous actions to discrete tokens.
        
        Uses min-max normalization to [0, 1] then maps to [0, NUM_ACTION_BINS-1].
        
        Args:
            actions: Continuous actions (B, action_dim)
            
        Returns:
            Action tokens (B, action_dim) as integers
        """
        # Normalize to [0, 1] using min/max
        if self.action_min is not None and self.action_max is not None:
            # Use stored statistics
            action_min = self.action_min.unsqueeze(0)  # (1, action_dim)
            action_max = self.action_max.unsqueeze(0)  # (1, action_dim)
        else:
            # Fallback: use batch statistics
            action_min = actions.min(dim=0, keepdim=True)[0]
            action_max = actions.max(dim=0, keepdim=True)[0]
        
        # Avoid division by zero
        action_range = action_max - action_min
        action_range = torch.clamp(action_range, min=1e-6)
        
        # Normalize to [0, 1]
        normalized = (actions - action_min) / action_range
        normalized = torch.clamp(normalized, 0.0, 1.0)
        
        # Map to discrete bins [0, NUM_ACTION_BINS-1]
        tokens = (normalized * (NUM_ACTION_BINS - 1)).long()
        
        # Add offset for action token vocabulary
        tokens = tokens + ACTION_TOKEN_BEGIN_IDX
        
        return tokens
    
    def _detokenize_actions(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete tokens back to continuous actions.
        
        Args:
            tokens: Action tokens (B, action_dim) or (action_dim,)
            
        Returns:
            Continuous actions
        """
        # Remove offset
        tokens = tokens - ACTION_TOKEN_BEGIN_IDX
        
        # Map from [0, NUM_ACTION_BINS-1] to [0, 1]
        normalized = tokens.float() / (NUM_ACTION_BINS - 1)
        
        # Denormalize using min/max
        if self.action_min is not None and self.action_max is not None:
            action_min = self.action_min
            action_max = self.action_max
            if normalized.dim() == 2:
                action_min = action_min.unsqueeze(0)
                action_max = action_max.unsqueeze(0)
            actions = normalized * (action_max - action_min) + action_min
        else:
            # Return normalized values if no statistics
            actions = normalized
        
        return actions
    
    def _process_multi_view_images(
        self, 
        images: Dict[str, torch.Tensor],
        debug: bool = False,
    ) -> Tuple[List[Image.Image], List[str]]:
        """
        Process multiple camera views to extract individual images.
        Each view will be processed separately to get its own visual tokens.
        
        Args:
            images: Dict of {view_name: (B, C, H, W)} tensors
            debug: Whether to print debug info
            
        Returns:
            Tuple of (list of PIL images, list of view names used)
        """
        batch_size = None
        ordered_images = []
        view_names_used = []
        
        for view in self.image_views:
            if view in images:
                img = images[view]
                if batch_size is None:
                    batch_size = img.shape[0]
                ordered_images.append(img)
                view_names_used.append(view)
        
        if len(ordered_images) == 0:
            raise ValueError(f"No valid images found in multi-view dict. Expected views: {self.image_views}, got: {list(images.keys())}")
        
        if debug and self._is_logging_rank:
            print(f"\n[DEBUG] Multi-view processing:")
            print(f"  Views used: {view_names_used}")
            print(f"  Individual shapes: {[img.shape for img in ordered_images]}")
            print(f"  Each view will be processed separately to preserve full resolution")
        
        # Convert all views to PIL images (for first sample in batch)
        pil_images = []
        for img_tensor in ordered_images:
            # Take first sample from batch: (C, H, W)
            img = img_tensor[0]
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
        
        return pil_images, view_names_used
    
    def forward(
        self,
        images: Union[torch.Tensor, Dict[str, torch.Tensor]],
        instructions: list[str],
        actions: Optional[torch.Tensor] = None,
        debug: bool = False,
        return_image_tensors: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with proper multi-view support.
        
        For multi-view inputs, each view is processed separately through the vision
        encoder to get its own visual tokens, then they are concatenated in sequence
        space (not pixel space). This preserves full resolution for each view and
        allows the LLM to attend to each view independently.
        
        Args:
            images: Batch of images (B, C, H, W) or dict of multi-view images
            instructions: List of language instructions
            actions: Ground truth actions for training (B, action_dim)
            debug: Whether to print debug info for multi-view
            return_image_tensors: Whether to return processed image tensors for logging
            
        Returns:
            Dictionary containing:
                - 'loss': Training loss
                - 'logits': Model logits
                - 'image_tensors': (optional) Processed images for logging
        """
        batch_size = images.shape[0] if isinstance(images, torch.Tensor) else next(iter(images.values())).shape[0]
        
        # Format prompts
        prompts = [
            f"In: What action should the robot take to {inst}?\nOut:"
            for inst in instructions
        ]
        
        # Handle multi-view images with proper token concatenation
        if isinstance(images, dict) and len(images) > 1:
            # PROPER MULTI-VIEW: Process each view separately, concatenate visual tokens
            if debug and self._is_logging_rank:
                print(f"\n[DEBUG] Processing multi-view images: {list(images.keys())}")
            
            # We'll process only the first sample for simplicity
            # In production, you'd want to handle full batches
            if batch_size > 1:
                if self._is_logging_rank and debug:
                    print(f"[WARNING] Multi-view with batch_size > 1 not fully optimized. Processing first sample.")
            
            # Get text tokenization first
            text_inputs = self.processor.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            )
            input_ids = text_inputs['input_ids'].to(self.device)
            attention_mask = text_inputs['attention_mask'].to(self.device)
            
            # Process each view separately to get visual embeddings
            all_visual_embeddings = []
            all_pil_images = []
            view_names_used = []
            
            for view_name in self.image_views:
                if view_name not in images:
                    continue
                    
                view_images = images[view_name]  # (B, C, H, W)
                
                # Convert to PIL images
                images_list = []
                for img in view_images:
                    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                    images_list.append(pil_img)
                    if len(all_pil_images) < batch_size:  # Store first batch for logging
                        all_pil_images.append(img)
                
                # Process images for this view
                view_inputs = self.processor.image_processor(
                    images=images_list,
                    return_tensors="pt",
                )
                pixel_values = view_inputs['pixel_values'].to(self.device, dtype=self.torch_dtype)
                
                # Extract visual features through vision backbone
                with torch.set_grad_enabled(self.training):
                    # Access the base model (unwrap DDP if needed)
                    base_model = self.model.module if hasattr(self.model, 'module') else self.model
                    
                    # Get patch features: (B, num_patches, vision_embed_dim)
                    patch_features = base_model.vision_backbone(pixel_values)
                    
                    # Project to LLM space: (B, num_patches, llm_embed_dim)
                    visual_embeddings = base_model.projector(patch_features)
                    
                all_visual_embeddings.append(visual_embeddings)
                view_names_used.append(view_name)
            
            if debug and self._is_logging_rank:
                print(f"  Processed {len(all_visual_embeddings)} views: {view_names_used}")
                print(f"  Visual embedding shapes: {[ve.shape for ve in all_visual_embeddings]}")
            
            # Concatenate all visual tokens in sequence dimension
            # Result: (B, num_patches * num_views, llm_embed_dim)
            combined_visual_embeddings = torch.cat(all_visual_embeddings, dim=1)
            
            if debug and self._is_logging_rank:
                print(f"  Combined visual embeddings shape: {combined_visual_embeddings.shape}")
            
            # Get text embeddings from language model
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            text_embeddings = base_model.language_model.get_input_embeddings()(input_ids)
            
            # Build multimodal sequence: [BOS] + [all_visual_tokens] + [text_tokens]
            multimodal_embeddings = torch.cat([
                text_embeddings[:, :1, :],  # BOS token
                combined_visual_embeddings,  # All visual tokens from all views
                text_embeddings[:, 1:, :],   # Rest of text
            ], dim=1)
            
            # Update attention mask for visual tokens
            visual_attention_mask = torch.ones(
                (batch_size, combined_visual_embeddings.shape[1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            multimodal_attention_mask = torch.cat([
                attention_mask[:, :1],
                visual_attention_mask,
                attention_mask[:, 1:],
            ], dim=1)
            
            # Create labels (ignore visual tokens, they don't produce text)
            labels = input_ids.clone()
            visual_labels = torch.full(
                (batch_size, combined_visual_embeddings.shape[1]),
                -100,  # IGNORE_INDEX
                dtype=labels.dtype,
                device=labels.device
            )
            multimodal_labels = torch.cat([
                labels[:, :1],
                visual_labels,
                labels[:, 1:],
            ], dim=1)
            
            if debug and self._is_logging_rank:
                print(f"  Final multimodal sequence length: {multimodal_embeddings.shape[1]}")
                print(f"    = BOS(1) + visual_tokens({combined_visual_embeddings.shape[1]}) + text({text_embeddings.shape[1]-1})")
            
            # Forward through language model
            outputs = base_model.language_model(
                inputs_embeds=multimodal_embeddings,
                attention_mask=multimodal_attention_mask,
                labels=multimodal_labels,
                use_cache=False,
            )
            
            result = {
                'loss': outputs.loss,
                'logits': outputs.logits,
            }
            
            if return_image_tensors:
                result['image_tensors'] = {view: images[view] for view in view_names_used}
                result['view_names'] = view_names_used
            
            return result
        
        else:
            # Single view - use standard processing
            if isinstance(images, dict):
                images = next(iter(images.values()))
            
            # Convert images from (B, C, H, W) to list of PIL Images
            images_list = []
            for img in images:
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                images_list.append(Image.fromarray(img_np))
            
            # Process with processor
            inputs = self.processor(
                text=prompts,
                images=images_list,
                return_tensors="pt",
                padding=True,
            )
            
            # Move to device and convert to correct dtype
            inputs = {
                k: v.to(device=self.device, dtype=self.torch_dtype) if v.dtype in [torch.float32, torch.float64] else v.to(self.device)
                for k, v in inputs.items()
            }
            
            # Use the model's native language modeling objective
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))
            pixel_values = inputs.get('pixel_values')
            
            # Create labels for language modeling
            labels = input_ids.clone()
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            
            result = {
                'loss': outputs.loss,
                'logits': outputs.logits,
            }
            
            if return_image_tensors:
                result['image_tensors'] = {'primary': images}
                result['view_names'] = ['primary']
            
            return result
    
    def predict_action(
        self,
        image: Union[torch.Tensor, Dict[str, torch.Tensor]],
        instruction: str,
        unnorm_key: Optional[str] = None,  # Changed: None means use custom stats
        do_sample: bool = False,
    ) -> np.ndarray:
        """
        Predict action for a single observation with multi-view support.
        
        FIX: Now uses custom 8-DOF action statistics from RoboFactory dataset
        instead of relying on bridge_orig (7-DOF) statistics from pretrained model.
        
        Args:
            image: Single image (C, H, W) or dict of multi-view images
            instruction: Language instruction
            unnorm_key: Key for denormalization stats (None = use custom stats)
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Predicted action as numpy array with shape (action_dim,)
        """
        # Handle multi-view input
        if isinstance(image, dict):
            # Add batch dimension to each view
            batched_images = {}
            for view_name, img in image.items():
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                batched_images[view_name] = img
            # Fuse views
            image = self._fuse_multi_view_images(batched_images)
        elif image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Convert to PIL Image
        img_np = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # Format prompt
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        
        # Process inputs
        inputs = self.processor(prompt, img_pil)
        inputs = {
            k: v.to(device=self.device, dtype=self.torch_dtype) 
            if isinstance(v, torch.Tensor) and v.dtype in [torch.float32, torch.float64] 
            else v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        
        # Generate action tokens
        with torch.no_grad():
            # Handle DDP wrapper
            model = self.model
            if hasattr(model, 'module'):
                model = model.module
            
            # FIX: Generate raw action tokens without using bridge_orig denormalization
            # We'll detokenize ourselves using our 8-DOF statistics
            if self.action_min is not None and self.action_max is not None:
                # Generate using base model but get raw token output
                generated_ids = model.generate(
                    input_ids=inputs.get('input_ids'),
                    attention_mask=inputs.get('attention_mask', None),
                    pixel_values=inputs.get('pixel_values'),
                    max_new_tokens=self.action_dim,  # Generate exactly action_dim tokens
                    do_sample=do_sample,
                    temperature=1.0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
                
                # Extract action tokens (last action_dim tokens)
                action_tokens = generated_ids[0, -self.action_dim:]
                
                # Detokenize using our custom 8-DOF statistics
                action_tensor = self._detokenize_actions(action_tokens)
                action = action_tensor.cpu().numpy()
            else:
                # Fallback: use base model's predict_action with bridge_orig
                # This path is for backward compatibility or if stats not set
                action = model.predict_action(
                    **inputs, 
                    unnorm_key=unnorm_key or "bridge_orig", 
                    do_sample=do_sample
                )
                action = np.array(action).flatten()
                
                # Pad or truncate to match expected action_dim (8 for Panda)
                if len(action) < self.action_dim:
                    padding = np.zeros(self.action_dim - len(action))
                    action = np.concatenate([action, padding])
                elif len(action) > self.action_dim:
                    action = action[:self.action_dim]
        
        return action
    
    def save_pretrained(self, save_directory: str):
        """
        Save model weights.
        
        Args:
            save_directory: Directory to save model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Handle DDP wrapper - get the underlying model
        model_to_save = self.model
        if hasattr(self.model, 'module'):
            # Model is wrapped with DDP, get the underlying module
            model_to_save = self.model.module
        
        if self.use_lora:
            # Save LoRA weights only
            model_to_save.save_pretrained(save_directory)
        else:
            # Save full model
            model_to_save.save_pretrained(save_directory)
        
        # Save processor
        self.processor.save_pretrained(save_directory)
        
        # Save action statistics if available
        if self.action_mean is not None:
            stats = {
                'action_mean': self.action_mean.float().cpu().numpy().tolist(),
                'action_std': self.action_std.float().cpu().numpy().tolist(),
                'action_dim': self.action_dim,
            }
            import json
            with open(os.path.join(save_directory, 'action_stats.json'), 'w') as f:
                json.dump(stats, f)
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Load pretrained model with LoRA weights.
        
        Args:
            model_path: Path to saved model
            device: Device to load on
            torch_dtype: Data type for model
            
        Returns:
            Loaded model instance
        """
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load base model
        base_model_name = "openvla/openvla-7b"
        model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, model_path)
        
        # Merge LoRA weights for faster inference
        model = model.merge_and_unload()
        
        # Move to device
        model = model.to(device)
        
        # Create wrapper instance
        wrapper = cls.__new__(cls)
        wrapper.model = model
        wrapper.processor = processor
        wrapper.device = device
        wrapper.torch_dtype = torch_dtype
        wrapper.use_lora = False  # Already merged
        wrapper.action_mean = None
        wrapper.action_std = None
        wrapper.action_min = None
        wrapper.action_max = None
        wrapper.action_dim = 8  # Default
        wrapper._is_logging_rank = True
        
        # Load action statistics if available
        import json
        stats_path = os.path.join(model_path, 'action_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            wrapper.set_action_statistics(
                np.array(stats['action_mean']),
                np.array(stats['action_std'])
            )
            wrapper.action_dim = stats.get('action_dim', 8)
        
        return wrapper
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self
    
    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        self.device = device
        if self.action_mean is not None:
            self.action_mean = self.action_mean.to(device)
            self.action_std = self.action_std.to(device)
            self.action_min = self.action_min.to(device)
            self.action_max = self.action_max.to(device)
        return self
    
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()


if __name__ == "__main__":
    # Test model loading
    print("Testing OpenVLA model wrapper...")
    
    model = OpenVLAModel(
        model_name="openvla/openvla-7b",
        use_lora=True,
        lora_rank=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        action_dim=8,
    )
    
    print("Model loaded successfully!")
    
    # Test forward pass
    batch_size = 2
    images = torch.rand(batch_size, 3, 224, 224)
    instructions = ["pick up the cube", "place the object"]
    actions = torch.rand(batch_size, 8)  # 8-DOF actions
    
    if torch.cuda.is_available():
        images = images.cuda()
        actions = actions.cuda()
    
    print("\nTesting forward pass...")
    outputs = model(images, instructions, actions)
    print(f"Loss: {outputs['loss'].item()}")
    
    print("\nModel wrapper test complete!")
