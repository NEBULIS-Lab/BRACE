"""
Pi0/Pi0.5 training workspace with multi-GPU support.

This module implements the training loop for Pi0 models using PyTorch DDP
for distributed training across multiple GPUs.

Now inherits from BaseVLAWorkspace for consistent behavior across all policies
while maintaining Pi0-specific features:
- Safetensors checkpoint format (following openpi)
- Quantile normalization for actions and states
- Custom collate function for LeRobot format

Follows openpi's training patterns from scripts/train_pytorch.py.
"""

import re
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

from ..model.pi0_wrapper import Pi0Model
from ..dataset.robot_lerobot_dataset import RobotLeRobotDataset, collate_fn

# Import base workspace and shared utilities
from robofactory.policy.core import BaseVLAWorkspace

# Suppress verbose OpenPI logs
logging.getLogger("openpi").setLevel(logging.WARNING)


class Pi0Workspace(BaseVLAWorkspace):
    """
    Training workspace for Pi0/Pi0.5 models.
    
    Inherits from BaseVLAWorkspace and implements Pi0-specific:
    - Model initialization (PI0Pytorch from openpi)
    - Dataset loading (LeRobot format with custom collate)
    - Loss computation (openpi's internal loss)
    - Checkpoint saving (safetensors format)
    """
    
    def __init__(self, cfg: OmegaConf = None, output_dir: Optional[str] = None, **kwargs):
        """
        Initialize workspace.
        
        Args:
            cfg: Hydra configuration
            output_dir: Output directory for logs and checkpoints
            **kwargs: Additional arguments from Hydra
        """
        # If cfg is not passed, reconstruct it from kwargs
        if cfg is None:
            cfg = OmegaConf.create(kwargs)
        
        # Call parent init
        super().__init__(cfg, output_dir)
        
        # Pi0-specific: For periodic logging (following openpi)
        self.step_infos = []
        self.start_time = time.time()
    
    def _get_checkpoint_dir(self) -> Path:
        """Get Pi0-specific checkpoint directory."""
        cfg = self.cfg
        
        # Get task name and agent info
        task_name = cfg.task.name if hasattr(cfg.task, 'name') else cfg.get('task_name', 'unknown')
        agent_id = cfg.get('agent_id', 0)
        
        # Model type (pi0 or pi05)
        model_type = cfg.model.get('model_variant', 'pi0')
        
        return Path('robofactory/checkpoints') / model_type / f"{task_name}_Agent{agent_id}"
    
    def _init_model(self) -> nn.Module:
        """Initialize Pi0 model."""
        cfg = self.cfg
        
        if self._is_main_process:
            print("Initializing Pi0 model...")
        
        model = Pi0Model(
            model_variant=cfg.model.model_variant,
            paligemma_variant=cfg.model.paligemma_variant,
            action_expert_variant=cfg.model.action_expert_variant,
            pretrained_checkpoint=cfg.model.pretrained_checkpoint,
            action_dim=cfg.model.action_dim,
            action_horizon=cfg.model.action_horizon,
            max_token_len=cfg.model.max_token_len,
            torch_dtype=getattr(torch, cfg.model.dtype),
            pytorch_training_precision=cfg.model.pytorch_training_precision,
            device=str(self.device),
            use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
        )
        
        return model
    
    def _init_dataset(self) -> Tuple[Dataset, Dataset]:
        """Create training and validation datasets (LeRobot format)."""
        cfg = self.cfg
        
        # Get lerobot_path from the correct location
        lerobot_path = cfg.task.lerobot_path if hasattr(cfg.task, 'lerobot_path') else cfg.task.dataset.lerobot_path
        
        if self._is_main_process:
            print(f"Loading LeRobot dataset: {lerobot_path}")
        
        # Training dataset
        train_dataset = RobotLeRobotDataset(
            repo_id=lerobot_path,
            action_horizon=cfg.model.action_horizon,
            train=True,
            val_split=cfg.training.val_split,
            seed=cfg.training.seed,
        )
        
        # Validation dataset
        val_dataset = RobotLeRobotDataset(
            repo_id=lerobot_path,
            action_horizon=cfg.model.action_horizon,
            train=False,
            val_split=cfg.training.val_split,
            seed=cfg.training.seed,
        )
        
        # Setup normalization statistics (following openpi's quantile normalization)
        self._setup_normalization(train_dataset)
        
        if self._is_main_process:
            print(f"Created datasets:")
            print(f"  Train: {len(train_dataset)} samples")
            print(f"  Val: {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def _setup_normalization(self, dataset: RobotLeRobotDataset):
        """Setup normalization statistics from dataset."""
        stats = dataset.get_statistics()
        if 'action' in stats and 'state' in stats:
            model_for_stats = self.model.module if isinstance(self.model, DDP) else self.model
            model_for_stats.set_normalization_statistics(
                action_q01=stats['action']['q01'],
                action_q99=stats['action']['q99'],
                state_q01=stats['state']['q01'],
                state_q99=stats['state']['q99'],
            )
            
            if self._is_main_process:
                print(f"Set normalization statistics:")
                print(f"  Action q01: {stats['action']['q01']}")
                print(f"  Action q99: {stats['action']['q99']}")
    
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute training loss using Pi0 model."""
        model_to_use = self.model.module if isinstance(self.model, DDP) else self.model
        loss = model_to_use.compute_loss(batch)
        return loss
    
    def _get_collate_fn(self):
        """Return Pi0's custom collate function for LeRobot format."""
        return collate_fn
    
    def run(self):
        """
        Main training loop.
        
        Overrides parent to add Pi0-specific features:
        - Sample observation logging
        - Safetensors checkpoint format
        """
        cfg = self.cfg
        
        # Print config at start
        if self._is_main_process:
            print("=" * 80)
            print("Initializing Pi0 Training")
            print("=" * 80)
            print(OmegaConf.to_yaml(cfg))
            print("=" * 80)
        
        # Initialize model
        self.model = self._init_model()
        if self._is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Initialize optimizer (from base class)
        self._init_optimizer()
        
        # Initialize datasets and dataloaders
        train_dataset, val_dataset = self._init_dataset()
        self._init_dataloaders(train_dataset, val_dataset)
        
        # Wrap with DDP if distributed
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
            if self._is_main_process:
                print(f"Using DistributedDataParallel with {self.world_size} GPUs")
        
        # Resume from checkpoint
        resuming = self._load_checkpoint_if_exists()
        
        # Initialize wandb
        self._init_wandb(resuming)
        
        # Log sample observations (only on new runs)
        if not resuming:
            self._log_sample_observations()
        
        # Training loop
        num_epochs = cfg.training.num_epochs
        train_loss = 0.0
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if self.distributed and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # Train epoch
            train_loss = self._train_epoch()
            
            # Validation
            val_loss = None
            if (epoch + 1) % cfg.training.val_every == 0:
                val_loss = self._validate()
                
                if self._is_main_process:
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
                    
                    if cfg.logging.wandb_enabled:
                        wandb.log({
                            'val_loss': val_loss,
                            'epoch': epoch + 1,
                        }, step=self.global_step)
            
            # Track best loss
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self._save_checkpoint(name="best")
            
            # Periodic checkpoint
            if (epoch + 1) % cfg.training.checkpoint_every == 0:
                self._save_checkpoint(name=f"epoch_{epoch + 1}")
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Final checkpoint
        self._save_checkpoint(name=f"epoch_{num_epochs}")
        
        # Cleanup
        from robofactory.policy.core import cleanup_distributed
        cleanup_distributed()
        
        # Finish wandb
        if self._is_main_process and cfg.logging.wandb_enabled:
            wandb.run.summary["final_train_loss"] = train_loss
            wandb.run.summary["best_loss"] = self.best_loss
            wandb.run.summary["total_epochs"] = num_epochs
            wandb.finish()
            
            print("\n" + "=" * 80)
            print("Training completed!")
            print("=" * 80)
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Follows openpi's training pattern with periodic logging.
        """
        cfg = self.cfg
        model_to_train = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_train.model.train()
        
        # Reset periodic logging
        self.step_infos = []
        self.start_time = time.time()
        total_loss = 0.0
        num_batches = 0
        
        # Get logging frequency
        log_every = cfg.logging.get('log_every_n_steps', 20)
        
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Train Epoch {self.epoch + 1}",
            disable=not self._is_main_process,
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Compute loss
            loss = self._compute_loss(batch)
            loss = loss / cfg.training.gradient_accumulate_every
            loss.backward()
            
            # Track step loss
            step_loss = loss.item() * cfg.training.gradient_accumulate_every
            self.step_infos.append({
                "loss": step_loss,
                "lr": self.optimizer.param_groups[0]['lr'],
            })
            total_loss += step_loss
            num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model_to_train.parameters(),
                    cfg.training.max_grad_norm
                )
                
                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Periodic logging (following openpi: every N steps)
                if self.global_step % log_every == 0 and len(self.step_infos) > 0:
                    elapsed = time.time() - self.start_time
                    
                    # Aggregate stats
                    avg_loss = np.mean([info["loss"] for info in self.step_infos])
                    avg_lr = np.mean([info["lr"] for info in self.step_infos])
                    
                    # Log to wandb
                    if self._is_main_process and cfg.logging.wandb_enabled:
                        log_payload = {
                            "loss": avg_loss,
                            "learning_rate": avg_lr,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "grad_norm": grad_norm.item(),
                            "time_per_step": elapsed / len(self.step_infos) if len(self.step_infos) > 0 else 0,
                        }
                        wandb.log(log_payload, step=self.global_step)
                    
                    # Reset stats collection
                    self.step_infos = []
                    self.start_time = time.time()
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{step_loss:.4e}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    "step": self.global_step
                })
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate(self) -> float:
        """Validate the model."""
        model_to_eval = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_eval.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.val_dataloader,
            desc="Validation",
            disable=not self._is_main_process,
        )
        
        with torch.no_grad():
            for batch in pbar:
                batch = self._move_to_device(batch)
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _init_wandb(self, resuming: bool = False):
        """Initialize wandb logging."""
        if not self._is_main_process:
            return
        
        cfg = self.cfg
        if not cfg.logging.wandb_enabled:
            return
        
        # Use core init_wandb for consistent behavior
        from robofactory.policy.core import init_wandb
        self.wandb_run_id = init_wandb(
            cfg=cfg,
            resuming=resuming,
            is_main_process=self._is_main_process,
            checkpoint_dir=self._get_checkpoint_dir(),
            run_id=self.wandb_run_id,
        )
    
    def _log_sample_observations(self):
        """Log sample observations to wandb (following openpi's pattern)."""
        if not self._is_main_process:
            return
        
        cfg = self.cfg
        if not cfg.logging.wandb_enabled or cfg.logging.mode == "disabled":
            return
        
        try:
            # Get a sample batch
            sample_batch = next(iter(self.train_dataloader))
            sample_batch = self._move_to_device(sample_batch)
            
            # Create sample images for wandb
            images_to_log = []
            image_dict = sample_batch["image"]
            batch_size = next(iter(image_dict.values())).shape[0]
            
            for i in range(min(5, batch_size)):
                img_list = []
                for cam_name in sorted(image_dict.keys()):
                    img = image_dict[cam_name][i].cpu().numpy()
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    img_list.append(img)
                
                img_concatenated = np.concatenate(img_list, axis=1)
                images_to_log.append(wandb.Image(img_concatenated, caption=f"Sample {i}"))
            
            wandb.log({"sample_observations": images_to_log}, step=0)
            
        except Exception as e:
            print(f"Warning: Could not log sample observations to wandb: {e}")
    
    def _save_checkpoint(self, name: str):
        """
        Save model checkpoint using safetensors (following openpi convention).
        
        Args:
            name: Checkpoint name (e.g., "epoch_1", "best", "final")
        """
        if not self._is_main_process:
            return
        
        checkpoint_dir = self._get_checkpoint_dir()
        checkpoint_path = checkpoint_dir / name
        tmp_checkpoint_path = checkpoint_dir / f"tmp_{name}"
        
        # Create temp directory
        import shutil
        if tmp_checkpoint_path.exists():
            shutil.rmtree(tmp_checkpoint_path)
        tmp_checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Get model to save
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        
        # Save model using safetensors (following openpi)
        import safetensors.torch
        safetensors.torch.save_model(model_to_save.model, tmp_checkpoint_path / "model.safetensors")
        
        # Save optimizer
        torch.save(self.optimizer.state_dict(), tmp_checkpoint_path / "optimizer.pt")
        
        # Save scheduler if exists
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), tmp_checkpoint_path / "scheduler.pt")
        
        # Save training metadata (including normalization stats for evaluation)
        metadata = {
            "global_step": self.global_step,
            "epoch": self.epoch + 1,  # Save completed epoch number
            "best_loss": self.best_loss,
            "wandb_run_id": getattr(self, 'wandb_run_id', None),
            "normalization_stats": {
                "action_q01": model_to_save.action_q01.cpu(),
                "action_q99": model_to_save.action_q99.cpu(),
                "state_q01": model_to_save.state_q01.cpu(),
                "state_q99": model_to_save.state_q99.cpu(),
            },
        }
        torch.save(metadata, tmp_checkpoint_path / "metadata.pt")
        
        # Atomically move temp to final location
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
        tmp_checkpoint_path.rename(checkpoint_path)
        
        # Also save as latest for easy resuming
        latest_path = checkpoint_dir / "latest"
        if latest_path.exists():
            shutil.rmtree(latest_path)
        shutil.copytree(checkpoint_path, latest_path)
        
    
    def _load_checkpoint_if_exists(self) -> bool:
        """
        Load the latest checkpoint if it exists.
        
        Prioritizes epoch_{number} checkpoints over 'latest'.
        
        Returns:
            True if checkpoint was loaded, False otherwise
        """
        cfg = self.cfg
        if not cfg.training.get('resume', True):
            return False
        
        checkpoint_dir = self._get_checkpoint_dir()
        if not checkpoint_dir.exists():
            return False
        
        # Find the highest numbered checkpoint
        checkpoints = []
        for d in checkpoint_dir.iterdir():
            if d.is_dir():
                if d.name.isdigit():
                    checkpoints.append((int(d.name), d))
                elif d.name.startswith("epoch_"):
                    try:
                        epoch_num = int(d.name.split("_")[1])
                        checkpoints.append((epoch_num * 10000, d))  # Higher priority
                    except:
                        pass
                elif d.name == "latest":
                    checkpoints.append((-1, d))  # Lower priority fallback
        
        if not checkpoints:
            if self._is_main_process:
                print(f"No checkpoint found in {checkpoint_dir}, starting fresh")
            return False
        
        # Sort by priority and pick the best
        latest_checkpoint = max(checkpoints, key=lambda x: x[0])[1]
        
        if self._is_main_process:
            print(f"Loading checkpoint from {latest_checkpoint}")
        
        # Load model using safetensors
        import safetensors.torch
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        state_dict = safetensors.torch.load_file(latest_checkpoint / "model.safetensors", device='cpu')
        # Use strict=False for tied weights (PaLI-Gemma)
        model_to_load.model.load_state_dict(state_dict, strict=False)
        
        # Load optimizer
        if (latest_checkpoint / "optimizer.pt").exists():
            optimizer_state = torch.load(latest_checkpoint / "optimizer.pt", map_location='cpu')
            self.optimizer.load_state_dict(optimizer_state)
        
        # Load scheduler
        if self.scheduler is not None and (latest_checkpoint / "scheduler.pt").exists():
            scheduler_state = torch.load(latest_checkpoint / "scheduler.pt", map_location='cpu')
            self.scheduler.load_state_dict(scheduler_state)
        
        # Load metadata
        if (latest_checkpoint / "metadata.pt").exists():
            metadata = torch.load(latest_checkpoint / "metadata.pt")
            self.global_step = metadata.get("global_step", 0)
            self.epoch = metadata.get("epoch", 0)
            self.best_loss = metadata.get("best_loss", float('inf'))
            self.wandb_run_id = metadata.get("wandb_run_id", None)
        
        if self._is_main_process:
            print(f"Resumed from: epoch={self.epoch}, step={self.global_step}")
        
        return True
