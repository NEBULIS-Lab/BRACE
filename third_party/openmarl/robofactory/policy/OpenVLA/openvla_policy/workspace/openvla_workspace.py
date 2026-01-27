"""
OpenVLA training workspace with multi-GPU support.

This module implements the training loop for OpenVLA models using PyTorch DDP
for distributed training across multiple GPUs.

Now inherits from BaseVLAWorkspace for consistent behavior across all policies
while maintaining OpenVLA-specific features:
- LoRA fine-tuning with PEFT
- Multi-view image support
- Simulation evaluation
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image

import robofactory.tasks
from ..model.openvla_wrapper import OpenVLAModel
from ..dataset.robot_rlds_dataset import RobotRLDSDataset, collate_fn

# Import base workspace and shared utilities
from robofactory.policy.core import BaseVLAWorkspace


class OpenVLAWorkspace(BaseVLAWorkspace):
    """
    Training workspace for OpenVLA models.
    
    Inherits from BaseVLAWorkspace and implements OpenVLA-specific:
    - Model initialization with LoRA
    - Multi-view image support
    - Simulation evaluation
    - LoRA checkpoint saving
    """
    
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        """
        Initialize workspace.
        
        Args:
            cfg: Hydra configuration
            output_dir: Output directory for logs and checkpoints
        """
        # Call parent init
        super().__init__(cfg, output_dir)
        
        # OpenVLA-specific: Early stopping tracking
        self.epochs_without_improvement = 0
        
        # Step-level metrics for detailed logging
        self.step_losses = []
        
        # Simulation evaluation environment (lazy init)
        self.eval_env = None
    
    def _get_checkpoint_dir(self) -> Path:
        """Get OpenVLA-specific checkpoint directory."""
        cfg = self.cfg
        rlds_path = cfg.task.dataset.rlds_path
        return Path('robofactory/checkpoints/openvla') / Path(rlds_path).stem
    
    def _init_model(self) -> nn.Module:
        """Initialize OpenVLA model with LoRA and multi-view support."""
        cfg = self.cfg
        
        if self._is_main_process:
            print("Initializing OpenVLA model...")
        
        # Get model settings from config
        action_dim = getattr(cfg.model, 'action_dim', 8)
        use_multi_view = getattr(cfg.model, 'use_multi_view', False)
        num_images = getattr(cfg.model, 'num_images', 1)
        image_views = list(getattr(cfg.model, 'image_views', ['primary', 'secondary', 'wrist']))
        multi_view_fusion = getattr(cfg.model, 'multi_view_fusion', 'concatenate')
        
        model = OpenVLAModel(
            model_name=cfg.model.model_name,
            use_lora=cfg.model.use_lora,
            lora_rank=cfg.model.lora_rank,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            torch_dtype=getattr(torch, cfg.model.torch_dtype),
            device=cfg.training.device,
            action_dim=action_dim,
            use_multi_view=use_multi_view,
            num_images=num_images,
            image_views=image_views,
            multi_view_fusion=multi_view_fusion,
        )
        
        return model
    
    def _init_dataset(self) -> Tuple[Dataset, Dataset]:
        """Create training and validation datasets with multi-view support."""
        cfg = self.cfg
        
        # Get multi-view settings
        use_multi_view = getattr(cfg.model, 'use_multi_view', False)
        image_views = list(getattr(cfg.model, 'image_views', ['primary', 'secondary', 'wrist']))
        
        if self._is_main_process:
            print(f"Loading RLDS dataset: {cfg.task.dataset.rlds_path}")
        
        # Training dataset
        train_dataset = RobotRLDSDataset(
            data_dir=cfg.task.dataset.rlds_path,
            train=True,
            image_size=(cfg.model.image_size, cfg.model.image_size),
            augment=cfg.training.image_aug,
            augment_crop_ratio=cfg.training.augment_crop_ratio,
            val_split=cfg.training.val_split,
            seed=cfg.training.seed,
            use_multi_view=use_multi_view,
            image_views=image_views,
        )
        
        # Validation dataset
        val_dataset = RobotRLDSDataset(
            data_dir=cfg.task.dataset.rlds_path,
            train=False,
            image_size=(cfg.model.image_size, cfg.model.image_size),
            augment=False,
            augment_crop_ratio=cfg.training.augment_crop_ratio,
            val_split=cfg.training.val_split,
            seed=cfg.training.seed,
            use_multi_view=use_multi_view,
            image_views=image_views,
        )
        
        # Setup action statistics
        self._setup_action_statistics(train_dataset)
        
        if self._is_main_process:
            print(f"Created datasets:")
            print(f"  Train: {len(train_dataset)} samples")
            print(f"  Val: {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def _setup_action_statistics(self, dataset: RobotRLDSDataset):
        """Setup action statistics for denormalization.
        
        FIX: Now also stores and passes min/max from actual data for proper
        8-DOF tokenization/detokenization.
        """
        stats = dataset.get_statistics()
        if 'action' in stats:
            action_mean = stats['action']['mean']
            action_std = stats['action']['std']
            
            # FIX: Also get min/max from dataset if available, otherwise compute from data
            # First, try to get from statistics.json
            if 'min' in stats['action'] and 'max' in stats['action']:
                action_min = stats['action']['min']
                action_max = stats['action']['max']
            else:
                # Compute min/max by iterating through samples
                if self._is_main_process:
                    print("Computing action min/max from dataset...")
                all_actions = []
                for i in range(min(1000, len(dataset))):  # Sample up to 1000
                    sample = dataset[i]
                    all_actions.append(sample['action'].numpy())
                all_actions = np.stack(all_actions)
                action_min = all_actions.min(axis=0).astype(np.float32)
                action_max = all_actions.max(axis=0).astype(np.float32)
            
            # Store for checkpoint saving
            self._action_stats = {
                'mean': action_mean,
                'std': action_std,
                'min': action_min,
                'max': action_max,
            }
            
            # Update model's action_dim based on actual data
            data_action_dim = len(action_mean)
            if data_action_dim != self.model.action_dim:
                if self._is_main_process:
                    print(f"Updating action_dim from {self.model.action_dim} to {data_action_dim}")
                self.model.action_dim = data_action_dim
            
            # FIX: Pass min/max to model for proper tokenization
            self.model.set_action_statistics(
                mean=action_mean, 
                std=action_std,
                action_min=action_min,
                action_max=action_max,
            )
            
            if self._is_main_process:
                print(f"Loaded {len(dataset)} samples, action_dim={data_action_dim}")
                print(f"  Action min: {action_min}")
                print(f"  Action max: {action_max}")
    
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute training loss using OpenVLA model."""
        # Extract batch components
        images = batch['image']
        actions = batch['action']
        instructions = batch['instruction']
        
        # Forward pass
        outputs = self.model(images, instructions, actions)
        return outputs['loss']
    
    def _get_collate_fn(self):
        """Return OpenVLA's custom collate function."""
        return collate_fn
    
    def run(self):
        """
        Main training loop.
        
        Overrides parent to add OpenVLA-specific features:
        - Multi-view debug logging
        - Simulation evaluation
        - Early stopping
        - LoRA checkpoint saving
        """
        cfg = self.cfg
        
        # Print config at start
        if self._is_main_process:
            print("=" * 80)
            print("Training Configuration:")
            print("=" * 80)
            print(OmegaConf.to_yaml(cfg))
            print("=" * 80)
        
        # Initialize model
        if self._is_main_process:
            print("Initializing model...")
        self.model = self._init_model()
        
        # Initialize optimizer
        if self._is_main_process:
            print("Initializing optimizer...")
        self._init_optimizer()
        
        # Initialize datasets and dataloaders
        if self._is_main_process:
            print("Initializing dataloaders...")
        train_dataset, val_dataset = self._init_dataset()
        self._init_dataloaders(train_dataset, val_dataset)
        
        # Wrap with DDP if distributed
        if self.distributed:
            self.model.model = DDP(
                self.model.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False,
            )
            if self._is_main_process:
                print(f"Using DistributedDataParallel with {self.world_size} GPUs")
        
        # Resume from checkpoint
        resuming = self._load_checkpoint_if_exists()
        
        # Initialize wandb
        self._init_wandb(resuming)
        
        # Log model info
        if self._is_main_process and cfg.logging.mode == "online":
            wandb.config.update({
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }, allow_val_change=True)
        
        # Debug dataset info
        if self._is_main_process and cfg.training.debug:
            self._debug_dataset_info()
        
        # Training loop
        num_epochs = cfg.training.num_epochs
        early_stopped = False
        train_metrics = {'train_loss': self.best_loss}
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if self.distributed and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # Train epoch
            train_metrics = self._train_epoch()
            
            # Validation
            val_metrics = {}
            if (epoch + 1) % cfg.training.val_every == 0:
                val_metrics = self._validate()
            
            # Log metrics
            if self._is_main_process:
                metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
                if hasattr(cfg.logging, 'log_learning_rate') and cfg.logging.log_learning_rate:
                    metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
                
                if cfg.logging.mode == "online":
                    wandb.log(metrics, step=self.global_step)
            
            # Early stopping check
            current_loss = train_metrics.get('train_loss', float('inf'))
            if self._check_early_stopping(current_loss):
                early_stopped = True
            
            # Synchronize loss across ranks
            if self.distributed:
                loss_tensor = torch.tensor([current_loss], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                current_loss = loss_tensor.item()
            
            # Save best checkpoint
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                if self._is_main_process:
                    self._save_checkpoint(name='best.ckpt')
            
            if self.distributed:
                dist.barrier()
            
            # Simulation evaluation
            if hasattr(cfg.training, 'eval_in_sim') and cfg.training.eval_in_sim:
                eval_every = getattr(cfg.training, 'eval_sim_every_n_epochs', 10)
                if (epoch + 1) % eval_every == 0:
                    sim_metrics = self._evaluate_in_simulation()
                    if self._is_main_process and sim_metrics:
                        print(f"Simulation eval: {sim_metrics}")
                    if self.distributed:
                        dist.barrier()
            
            # Periodic checkpoint
            if (epoch + 1) % cfg.training.checkpoint_every == 0:
                if self._is_main_process:
                    self._save_checkpoint()
            
            if self.distributed:
                dist.barrier()
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Break if early stopped
            if early_stopped:
                break
        
        # Final checkpoint
        if self._is_main_process:
            self._save_checkpoint(name='final.ckpt')
        
        # Final simulation evaluation
        if self.distributed:
            dist.barrier()
        
        if self._is_main_process and hasattr(cfg.training, 'eval_in_sim') and cfg.training.eval_in_sim:
            print("Running final simulation evaluation...")
            sim_metrics = self._evaluate_in_simulation()
            if sim_metrics:
                print(f"Final simulation eval: {sim_metrics}")
                if cfg.logging.mode == "online":
                    for k, v in sim_metrics.items():
                        wandb.run.summary[f"final_{k}"] = v
        
        if self.distributed:
            dist.barrier()
        
        # Cleanup
        self._cleanup()
        
        # Finish wandb
        if self._is_main_process and cfg.logging.mode == "online":
            wandb.run.summary["final_train_loss"] = train_metrics.get('train_loss', 0)
            wandb.run.summary["best_loss"] = self.best_loss
            wandb.run.summary["total_epochs"] = self.epoch + 1
            wandb.run.summary["early_stopped"] = early_stopped
            wandb.finish()
            
            print("\n" + "=" * 80)
            print("Training complete!")
            print("=" * 80)
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with OpenVLA-specific logging."""
        cfg = self.cfg
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        self.step_losses = []
        
        log_every = getattr(cfg.logging, 'log_every_n_steps', 10)
        
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Train Epoch {self.epoch}",
            disable=not self._is_main_process,
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = self._move_to_device(batch)
            
            # Debug first batch
            debug_this_batch = self._is_main_process and cfg.training.debug and self.epoch == 0 and batch_idx == 0
            if debug_this_batch:
                self._debug_batch_info(batch)
            
            # Forward pass with optional image logging
            should_log_images = (
                self._is_main_process and 
                cfg.logging.mode == "online" and
                batch_idx == 0 and 
                self.epoch % getattr(cfg.logging, 'log_images_every_n_epochs', 1) == 0
            )
            
            outputs = self.model(
                batch['image'], 
                batch['instruction'], 
                batch['action'], 
                debug=debug_this_batch,
                return_image_tensors=should_log_images
            )
            loss = outputs['loss']
            
            # Log images to wandb
            if should_log_images and 'image_tensors' in outputs:
                self._log_images_to_wandb(
                    outputs['image_tensors'], 
                    outputs.get('view_names', []), 
                    batch['instruction'][:3]
                )
            
            # Backward pass
            loss = loss / cfg.training.gradient_accumulate_every
            loss.backward()
            
            step_loss = loss.item() * cfg.training.gradient_accumulate_every
            self.step_losses.append(step_loss)
            total_loss += step_loss
            num_batches += 1
            
            # Update weights
            if (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                # Compute gradient norm
                grad_norm = None
                if self._is_main_process and hasattr(cfg.logging, 'log_gradients') and cfg.logging.log_gradients:
                    grad_norm = self._compute_grad_norm()
                
                # Gradient clipping
                if cfg.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        cfg.training.max_grad_norm
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Step-level logging
                if self._is_main_process and cfg.logging.mode == "online" and self.global_step % log_every == 0:
                    step_metrics = {
                        'train/step_loss': step_loss,
                        'train/global_step': self.global_step,
                    }
                    if grad_norm is not None:
                        step_metrics['train/grad_norm'] = grad_norm
                    if hasattr(cfg.logging, 'log_learning_rate') and cfg.logging.log_learning_rate:
                        step_metrics['train/learning_rate'] = self.optimizer.param_groups[0]['lr']
                    wandb.log(step_metrics, step=self.global_step)
            
            pbar.set_postfix({'loss': step_loss})
            
            # Early exit for debugging
            if cfg.training.debug and cfg.training.max_train_steps is not None and batch_idx >= cfg.training.max_train_steps:
                break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            'train_loss_std': np.std(self.step_losses) if self.step_losses else 0.0,
            'train_loss_min': min(self.step_losses) if self.step_losses else 0.0,
            'train_loss_max': max(self.step_losses) if self.step_losses else 0.0,
        }
    
    def _validate(self) -> Dict[str, float]:
        """Validate for one epoch."""
        cfg = self.cfg
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.val_dataloader,
            desc=f"Val Epoch {self.epoch}",
            disable=not self._is_main_process,
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                batch = self._move_to_device(batch)
                
                outputs = self.model(batch['image'], batch['instruction'], batch['action'])
                loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
                
                if cfg.training.debug and cfg.training.max_val_steps is not None and batch_idx >= cfg.training.max_val_steps:
                    break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'val_loss': avg_loss}
    
    def _init_wandb(self, resuming: bool = False):
        """Initialize wandb logging."""
        if not self._is_main_process:
            return
        
        cfg = self.cfg
        if cfg.logging.mode == "disabled":
            return
        
        from robofactory.policy.core import init_wandb
        self.wandb_run_id = init_wandb(
            cfg=cfg,
            resuming=resuming,
            is_main_process=self._is_main_process,
            checkpoint_dir=self._get_checkpoint_dir(),
            run_id=self.wandb_run_id,
        )
    
    def _check_early_stopping(self, current_loss: float) -> bool:
        """Check if early stopping should be triggered."""
        cfg = self.cfg
        
        if not (hasattr(cfg.training, 'early_stopping') and cfg.training.early_stopping):
            return False
        
        patience = getattr(cfg.training, 'early_stopping_patience', 10)
        min_delta = getattr(cfg.training, 'early_stopping_min_delta', 1e-7)
        
        if current_loss < self.best_loss - min_delta:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self._is_main_process:
                print(f"No improvement for {self.epochs_without_improvement}/{patience} epochs. "
                      f"Best: {self.best_loss:.8f}, Current: {current_loss:.8f}")
            
            if self.epochs_without_improvement >= patience:
                if self._is_main_process:
                    print(f"Early stopping triggered!")
                return True
        
        return False
    
    def _compute_grad_norm(self) -> float:
        """Compute the total gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _debug_dataset_info(self):
        """Print debug info about the dataset."""
        cfg = self.cfg
        print("\n" + "=" * 60)
        print("DEBUG: Verifying dataset format and multi-view images")
        print("=" * 60)
        
        dataset = self.train_dataloader.dataset
        sample = dataset[0]
        
        use_multi_view = getattr(cfg.model, 'use_multi_view', False)
        print(f"Multi-view enabled: {use_multi_view}")
        
        if isinstance(sample['image'], dict):
            print(f"Multi-view images in dataset:")
            for view_name, view_tensor in sample['image'].items():
                print(f"  - {view_name}: shape={view_tensor.shape}")
        else:
            print(f"Single image: shape={sample['image'].shape}")
        
        print("=" * 60 + "\n")
    
    def _debug_batch_info(self, batch: Dict[str, Any]):
        """Print debug info about a batch."""
        print(f"\n" + "=" * 60)
        print(f"[DEBUG] First training batch (epoch {self.epoch})")
        print("=" * 60)
        
        images = batch['image']
        if isinstance(images, dict):
            print(f"Multi-view images: {len(images)} views")
            for view_name, view_tensor in images.items():
                print(f"  - {view_name}: shape={view_tensor.shape}")
        else:
            print(f"Single image: shape={images.shape}")
        
        instructions = batch['instruction']
        print(f"\nInstructions (first 3 of {len(instructions)}):")
        for i, inst in enumerate(instructions[:3]):
            print(f"  [{i}] '{inst}'")
        print("=" * 60 + "\n")
    
    def _log_images_to_wandb(self, image_tensors: Dict[str, torch.Tensor], 
                             view_names: List[str], instructions: List[str]):
        """Log input images to wandb."""
        try:
            wandb_images = []
            num_samples = min(3, next(iter(image_tensors.values())).shape[0])
            
            for sample_idx in range(num_samples):
                instruction = instructions[sample_idx] if sample_idx < len(instructions) else "N/A"
                
                view_images = []
                for view_name in view_names:
                    if view_name in image_tensors:
                        img_tensor = image_tensors[view_name][sample_idx]
                        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                        img_np = np.clip(img_np, 0, 1)
                        img_np = (img_np * 255).astype(np.uint8)
                        view_images.append(img_np)
                
                if len(view_images) > 1:
                    combined_img = np.concatenate(view_images, axis=1)
                else:
                    combined_img = view_images[0]
                
                caption = f"Views: {', '.join(view_names)} | Instruction: {instruction}"
                wandb_images.append(wandb.Image(combined_img, caption=caption))
            
            wandb.log({"train/input_images": wandb_images}, step=self.global_step)
            
        except Exception as e:
            print(f"[WARNING] Failed to log images to wandb: {e}")
    
    def _save_checkpoint(self, name: Optional[str] = None):
        """Save checkpoint with LoRA weights."""
        if not self._is_main_process:
            return
        
        checkpoint_dir = self._get_checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if name is None:
            name = f'epoch_{self.epoch + 1}.ckpt'
        
        checkpoint_path = checkpoint_dir / name
        
        # Save model (LoRA weights)
        model_save_path = str(checkpoint_path.with_suffix(''))
        self.model.save_pretrained(model_save_path)
        
        # Save training state
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'optimizer_state': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'wandb_run_id': self.wandb_run_id,
            'model_save_path': model_save_path,
        }
        if self.scheduler is not None:
            state['scheduler_state'] = self.scheduler.state_dict()
        
        torch.save(state, checkpoint_path)
        
        # Save as latest
        latest_path = checkpoint_dir / 'latest.ckpt'
        torch.save(state, latest_path)
        
        # Save metadata JSON
        metadata = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': float(self.best_loss),
            'wandb_run_id': self.wandb_run_id,
        }
        with open(checkpoint_dir / 'training_state.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # FIX: Save action statistics for proper inference
        # This ensures the model can properly detokenize actions during evaluation
        if hasattr(self, '_action_stats'):
            action_stats_path = checkpoint_dir / 'action_stats.json'
            action_stats = {
                'action_mean': self._action_stats['mean'].tolist(),
                'action_std': self._action_stats['std'].tolist(),
                'action_min': self._action_stats['min'].tolist(),
                'action_max': self._action_stats['max'].tolist(),
                'action_dim': len(self._action_stats['mean']),
            }
            with open(action_stats_path, 'w') as f:
                json.dump(action_stats, f, indent=2)
            print(f"Saved action statistics to {action_stats_path}")
            
    def _load_checkpoint_if_exists(self) -> bool:
        """Load the latest checkpoint if it exists."""
        cfg = self.cfg
        if not cfg.training.get('resume', True):
            return False
        
        checkpoint_dir = self._get_checkpoint_dir()
        checkpoint_path = checkpoint_dir / 'latest.ckpt'
        
        if not checkpoint_path.exists():
            if self._is_main_process:
                print(f"No checkpoint found in {checkpoint_dir}, starting fresh")
            return False
        
        if self._is_main_process:
            print(f"Resuming from {checkpoint_path}")
        
        state = torch.load(checkpoint_path, map_location='cpu')
        
        # Restore training state
        self.epoch = state.get('epoch', 0) + 1
        self.global_step = state.get('global_step', 0)
        self.best_loss = state.get('best_loss', float('inf'))
        self.epochs_without_improvement = state.get('epochs_without_improvement', 0)
        self.wandb_run_id = state.get('wandb_run_id', None)
        
        # Restore optimizer
        if 'optimizer_state' in state:
            self.optimizer.load_state_dict(state['optimizer_state'])
            for param_state in self.optimizer.state.values():
                for k, v in param_state.items():
                    if isinstance(v, torch.Tensor):
                        param_state[k] = v.to(self.device)
        
        # Restore scheduler
        if 'scheduler_state' in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler_state'])
        
        # Load LoRA weights
        model_save_path = state.get('model_save_path')
        if model_save_path and Path(model_save_path).exists():
            try:
                base_model = self.model.model.module if self.distributed else self.model.model
                if hasattr(base_model, 'load_adapter'):
                    base_model.load_adapter(model_save_path, adapter_name="default")
                    print(f"Loaded LoRA adapter from {model_save_path}")
            except Exception as e:
                print(f"Warning: Could not load LoRA adapter: {e}")
        
        if self._is_main_process:
            print(f"Resumed from epoch {self.epoch}, step {self.global_step}")
        
        return True
    
    def _cleanup(self):
        """Cleanup resources."""
        # Cleanup eval environment
        if self.eval_env is not None:
            try:
                self.eval_env.close()
                self.eval_env = None
            except:
                pass
        
        # Cleanup distributed
        from robofactory.policy.core import cleanup_distributed
        cleanup_distributed()
    
    # =========================================================================
    # Simulation Evaluation (kept as-is for full functionality)
    # =========================================================================
    
    def _init_eval_env(self):
        """Lazy initialization of evaluation environment."""
        if self.eval_env is not None:
            return True
        
        cfg = self.cfg
        
        # Setup headless rendering
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["DISPLAY"] = ""
        
        try:
            import gymnasium as gym
        except (ImportError, FileNotFoundError, OSError) as e:
            print(f"Warning: Could not import simulation dependencies: {e}")
            return False
        
        env_id = cfg.task.name if hasattr(cfg.task, 'name') else None
        if env_id and not env_id.endswith('-rf'):
            env_id += '-rf'
        
        if env_id is None:
            return False
        
        num_parallel = getattr(cfg.training, 'eval_sim_num_envs', 4)
        
        try:
            env_kwargs = dict(
                obs_mode='rgb',
                control_mode='pd_joint_pos',
                render_mode='rgb_array',
                num_envs=num_parallel,
                sim_backend='gpu',
            )
            
            self.eval_env = gym.make(env_id, **env_kwargs)
            self.eval_num_parallel = num_parallel
            print(f"Initialized eval env: {env_id} with {num_parallel} parallel envs (GPU)")
            return True
            
        except Exception as e:
            print(f"GPU backend failed ({e}), falling back to CPU...")
            try:
                env_kwargs['sim_backend'] = 'cpu'
                env_kwargs['num_envs'] = 1
                self.eval_env = gym.make(env_id, **env_kwargs)
                self.eval_num_parallel = 1
                return True
            except Exception as e2:
                print(f"Warning: Could not create eval environment: {e2}")
                return False
    
    @torch.no_grad()
    def _evaluate_in_simulation(self) -> Dict[str, Any]:
        """Run model evaluation in simulation."""
        if not self._is_main_process:
            return {}
        
        if not self._init_eval_env():
            return {}
        
        cfg = self.cfg
        env = self.eval_env
        num_parallel = self.eval_num_parallel
        
        num_episodes = getattr(cfg.training, 'eval_sim_episodes', 5)
        max_steps = getattr(cfg.training, 'eval_sim_max_steps', 200)
        num_batches = (num_episodes + num_parallel - 1) // num_parallel
        
        try:
            self.model.eval()
            instruction = getattr(cfg.task, 'instruction', 'complete the task')
            
            all_successes = []
            all_rewards = []
            
            for batch_idx in range(num_batches):
                obs, info = env.reset()
                episode_rewards = np.zeros(num_parallel)
                
                for step in range(max_steps):
                    images = self._extract_images_from_obs(obs, num_parallel)
                    if images is None:
                        break
                    
                    if not isinstance(images, list):
                        images = [images]
                    
                    actions = []
                    for i in range(num_parallel):
                        img = images[i] if i < len(images) else images[0]
                        action = self.model.predict_action(
                            img.to(self.device),
                            instruction,
                            do_sample=False
                        )
                        actions.append(action)
                    
                    action_batch = np.stack(actions, axis=0) if num_parallel > 1 else actions[0]
                    obs, reward, terminated, truncated, info = env.step(action_batch)
                    
                    reward_np = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else np.array(reward)
                    if np.isscalar(reward_np):
                        episode_rewards[0] += reward_np
                    else:
                        episode_rewards += reward_np.flatten()[:num_parallel]
                    
                    terminated_np = terminated.cpu().numpy() if isinstance(terminated, torch.Tensor) else np.array(terminated)
                    truncated_np = truncated.cpu().numpy() if isinstance(truncated, torch.Tensor) else np.array(truncated)
                    done = np.logical_or(terminated_np, truncated_np)
                    if np.all(done):
                        break
                
                if isinstance(info, dict):
                    success = info.get('success', False)
                    success_np = success.cpu().numpy() if isinstance(success, torch.Tensor) else np.array(success)
                    if np.isscalar(success_np):
                        all_successes.append(success_np)
                    else:
                        all_successes.extend(success_np.flatten()[:num_parallel].tolist())
                
                all_rewards.extend(episode_rewards[:num_parallel].tolist())
            
            successes = all_successes[:num_episodes]
            rewards = all_rewards[:num_episodes]
            
            success_rate = sum(successes) / len(successes) if successes else 0.0
            avg_reward = np.mean(rewards) if rewards else 0.0
            
            metrics = {
                'eval/sim_success_rate': float(success_rate),
                'eval/sim_avg_reward': float(avg_reward),
            }
            
            if cfg.logging.mode == "online":
                wandb.log(metrics, step=self.global_step)
            
            self.model.train()
            return metrics
            
        except Exception as e:
            print(f"Warning: Simulation evaluation failed: {e}")
            self.model.train()
            return {}
    
    def _extract_images_from_obs(self, obs, num_parallel):
        """Extract images from observation dict."""
        try:
            if isinstance(obs, dict):
                if 'rgb' in obs:
                    image = obs['rgb']
                elif 'image' in obs:
                    image = obs['image']
                elif 'sensor_data' in obs:
                    sensor_data = obs['sensor_data']
                    cam_key = list(sensor_data.keys())[0]
                    image = sensor_data[cam_key].get('rgb', None)
                else:
                    return None
            else:
                image = obs
            
            if image is None:
                return None
            
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    images = []
                    for i in range(min(num_parallel, image.shape[0])):
                        img = image[i]
                        if img.max() > 1.0:
                            img = img.float() / 255.0
                        if img.shape[-1] in [3, 4]:
                            img = img.permute(2, 0, 1)
                        if img.shape[0] == 4:
                            img = img[:3]
                        images.append(img)
                    return images
                elif image.dim() == 3:
                    if image.max() > 1.0:
                        image = image.float() / 255.0
                    if image.shape[-1] in [3, 4]:
                        image = image.permute(2, 0, 1)
                    if image.shape[0] == 4:
                        image = image[:3]
                    return [image]
            
            if isinstance(image, np.ndarray):
                if image.ndim == 4:
                    images = []
                    for i in range(min(num_parallel, image.shape[0])):
                        img = image[i]
                        if img.max() > 1.0:
                            img = img.astype(np.float32) / 255.0
                        if img.shape[-1] == 3:
                            img = np.transpose(img, (2, 0, 1))
                        images.append(torch.from_numpy(img).float())
                    return images
                else:
                    if image.max() > 1.0:
                        image = image.astype(np.float32) / 255.0
                    if image.ndim == 3 and image.shape[-1] == 3:
                        image = np.transpose(image, (2, 0, 1))
                    return [torch.from_numpy(image).float()]
            
            return None
        except Exception as e:
            print(f"Warning: Could not extract images: {e}")
            return None


if __name__ == "__main__":
    print("OpenVLA workspace module loaded successfully!")
