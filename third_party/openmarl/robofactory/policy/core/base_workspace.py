"""
Abstract base workspace for VLA policy training.

This module provides the BaseVLAWorkspace class that handles common training
functionality. All policy-specific workspaces should inherit from this class
and implement the abstract methods.

Patterns extracted from:
- Pi0Workspace: DDP setup, training loop, wandb logging
- OpenVLAWorkspace: Logging, simulation evaluation, checkpoint management
- BaseWorkspace (Diffusion-Policy): Checkpoint save/load with dill
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
from tqdm import tqdm

from .distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
)
from .logging_utils import init_wandb, log_metrics, finish_wandb
from .checkpoint_manager import CheckpointManager


class BaseVLAWorkspace(ABC):
    """
    Abstract base workspace for VLA policy training.
    
    This class handles:
    - Distributed training setup (DDP)
    - Optimizer and scheduler initialization
    - Training and validation loops
    - WandB logging integration
    - Checkpoint saving/loading with resumption support
    
    Subclasses must implement:
    - _init_model(): Initialize the policy-specific model
    - _init_dataset(): Create dataset instances
    - _compute_loss(batch): Compute training loss from a batch
    
    Optional overrides:
    - _get_checkpoint_dir(): Customize checkpoint directory
    - _get_collate_fn(): Provide custom collate function
    - _on_epoch_end(epoch, metrics): Hook for end-of-epoch actions
    - _validate_batch(batch): Custom validation logic
    """
    
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        """
        Initialize workspace.
        
        Args:
            cfg: Hydra/OmegaConf configuration object
            output_dir: Output directory for logs and checkpoints
        """
        self.cfg = cfg
        
        # Set output directory
        if output_dir is None:
            output_dir = cfg.hydra.run.dir if hasattr(cfg, 'hydra') else 'outputs'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup distributed training
        self.distributed, self.local_rank, self.device = setup_distributed()
        self.rank = get_rank()
        self.world_size = get_world_size()
        self._is_main_process = is_main_process()
        
        # Set random seed (different per rank for data augmentation diversity)
        seed = cfg.training.seed + self.rank
        self._set_seed(seed)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.wandb_run_id = None
        
        # Components (initialized in run())
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Step-level metrics for logging
        self.step_losses = []
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self._get_checkpoint_dir(),
            is_main_process=self._is_main_process,
        )
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    @abstractmethod
    def _init_model(self) -> nn.Module:
        """
        Initialize the policy-specific model.
        
        Must be implemented by subclass.
        
        Returns:
            Initialized model (not yet wrapped with DDP)
        """
        pass
    
    @abstractmethod
    def _init_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        Create training and validation datasets.
        
        Must be implemented by subclass.
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        pass
    
    @abstractmethod
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute training loss from a batch.
        
        Must be implemented by subclass.
        
        Args:
            batch: Batch dictionary from dataloader (already on device)
            
        Returns:
            Loss tensor (scalar)
        """
        pass
    
    def _get_checkpoint_dir(self) -> Path:
        """
        Get checkpoint directory path.
        
        Can be overridden by subclass for custom checkpoint locations.
        
        Returns:
            Path to checkpoint directory
        """
        policy_type = self.cfg.get('exp_name', 'policy')
        task_name = self.cfg.task.name if hasattr(self.cfg.task, 'name') else 'unknown'
        agent_id = self.cfg.get('agent_id', 0)
        return Path('robofactory/checkpoints') / policy_type / f"{task_name}_Agent{agent_id}"
    
    def _get_collate_fn(self):
        """
        Get collate function for dataloaders.
        
        Can be overridden by subclass.
        
        Returns:
            Collate function or None for default
        """
        return None
    
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        cfg = self.cfg.training
        
        # Get trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Create optimizer
        self.optimizer = AdamW(
            params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )
        
        # Create learning rate scheduler
        if cfg.use_scheduler:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.num_epochs,
                eta_min=cfg.get('min_learning_rate', 1e-6),
            )
    
    def _init_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset):
        """
        Create dataloaders with distributed sampling support.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        cfg = self.cfg
        collate_fn = self._get_collate_fn()
        
        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None
        if self.distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=cfg.training.seed,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.dataloader.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            collate_fn=collate_fn,
            drop_last=True,
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.val_dataloader.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=cfg.val_dataloader.num_workers,
            pin_memory=cfg.val_dataloader.pin_memory,
            collate_fn=collate_fn,
        )
        
        if self._is_main_process:
            print(f"Created dataloaders:")
            print(f"  Train: {len(train_dataset)} samples, {len(self.train_dataloader)} batches")
            print(f"  Val: {len(val_dataset)} samples, {len(self.val_dataloader)} batches")
    
    def run(self):
        """
        Main training loop.
        
        This method orchestrates the entire training process:
        1. Initialize model, optimizer, and dataloaders
        2. Resume from checkpoint if available
        3. Run training epochs
        4. Save checkpoints and log metrics
        """
        cfg = self.cfg
        
        # Initialize components
        if self._is_main_process:
            print("=" * 80)
            print("Initializing Training")
            print("=" * 80)
        
        # Initialize model
        self.model = self._init_model()
        if self._is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Initialize optimizer
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
        
        # Resume from checkpoint if exists
        resuming = False
        if cfg.training.get('resume', True):
            checkpoint_data = self.checkpoint_manager.load_if_exists(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            if checkpoint_data is not None:
                self.global_step = checkpoint_data.get('global_step', 0)
                self.epoch = checkpoint_data.get('epoch', 0)
                self.best_loss = checkpoint_data.get('best_loss', float('inf'))
                self.wandb_run_id = checkpoint_data.get('wandb_run_id', None)
                resuming = True
                if self._is_main_process:
                    print(f"Resumed from epoch {self.epoch}, step {self.global_step}")
        
        # Initialize wandb
        self.wandb_run_id = init_wandb(
            cfg=cfg,
            resuming=resuming,
            is_main_process=self._is_main_process,
            run_id=self.wandb_run_id,
        )
        
        # Training loop
        num_epochs = cfg.training.num_epochs
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
                    log_metrics({
                        'val_loss': val_loss,
                        'epoch': epoch + 1,
                    }, self.global_step, self._is_main_process)
            
            # Track best loss and save best checkpoint
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.checkpoint_manager.save_best(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    global_step=self.global_step,
                    best_loss=self.best_loss,
                    wandb_run_id=self.wandb_run_id,
                )
            
            # Periodic checkpoint
            if (epoch + 1) % cfg.training.checkpoint_every == 0:
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    global_step=self.global_step,
                    best_loss=self.best_loss,
                    wandb_run_id=self.wandb_run_id,
                    name=f"epoch_{epoch + 1}",
                )
            
            # End of epoch hook
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': self.best_loss,
            }
            self._on_epoch_end(epoch, metrics)
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Final checkpoint
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=num_epochs,
            global_step=self.global_step,
            best_loss=self.best_loss,
            wandb_run_id=self.wandb_run_id,
            name="final",
        )
        
        # Cleanup
        cleanup_distributed()
        
        # Finish wandb
        summary = {
            'final_train_loss': train_loss,
            'best_loss': self.best_loss,
            'total_epochs': num_epochs,
        }
        finish_wandb(summary, self._is_main_process)
        
        if self._is_main_process:
            print("\n" + "=" * 80)
            print("Training completed!")
            print("=" * 80)
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        cfg = self.cfg
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        self.step_losses = []
        
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
            self.step_losses.append(step_loss)
            total_loss += step_loss
            num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                # Gradient clipping
                if cfg.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        cfg.training.max_grad_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Periodic logging
                if self.global_step % log_every == 0 and self._is_main_process:
                    log_metrics({
                        'train/step_loss': step_loss,
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'train/global_step': self.global_step,
                    }, self.global_step, self._is_main_process)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{step_loss:.4e}'})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate(self) -> float:
        """
        Run validation.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
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
                loss = self._validate_batch(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Validate a single batch.
        
        Can be overridden for custom validation logic.
        Default implementation uses _compute_loss.
        
        Args:
            batch: Batch dictionary (already on device)
            
        Returns:
            Loss tensor
        """
        return self._compute_loss(batch)
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch tensors to device.
        
        Args:
            batch: Batch dictionary from dataloader
            
        Returns:
            Batch with tensors moved to device
        """
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                # Handle nested dicts (e.g., multi-view images)
                result[key] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                result[key] = value
        return result
    
    def _on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """
        Hook called at the end of each epoch.
        
        Can be overridden for custom end-of-epoch logic.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics from the epoch
        """
        pass

