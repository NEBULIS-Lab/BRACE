"""
Unified checkpoint management for VLA policies.

This module provides the CheckpointManager class that handles saving and loading
checkpoints across all policy implementations, with support for:
- Epoch-based checkpointing
- Best model tracking
- Resume from latest checkpoint
- Atomic writes to prevent corruption

Patterns extracted from:
- TopKCheckpointManager (Diffusion-Policy): Best checkpoint tracking
- BaseWorkspace.save_checkpoint: State dict saving with dill
- Pi0Workspace._save_checkpoint: Atomic writes, safetensors support
- OpenVLAWorkspace.save_checkpoint: Metadata JSON, latest pointer
"""

import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class CheckpointManager:
    """
    Unified checkpoint manager for VLA policies.
    
    Handles:
    - Saving model, optimizer, and scheduler state
    - Tracking best checkpoint based on metric
    - Finding and loading latest checkpoint for resume
    - Atomic writes to prevent corruption
    - Metadata storage for training state
    
    Directory structure:
        checkpoint_dir/
        ├── epoch_10/           # Periodic checkpoint
        │   ├── model.pt
        │   ├── optimizer.pt
        │   └── metadata.json
        ├── epoch_20/
        ├── best/               # Best checkpoint
        ├── latest/             # Symlink or copy to most recent
        └── training_state.json # Overall training metadata
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        is_main_process: bool = True,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        use_safetensors: bool = False,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            is_main_process: Whether this is the main process (only main saves)
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
            use_safetensors: Whether to use safetensors format for model weights
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.is_main_process = is_main_process
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.use_safetensors = use_safetensors
        
        # Create directory
        if is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        best_loss: float = float('inf'),
        wandb_run_id: Optional[str] = None,
        name: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save checkpoint.
        
        Args:
            model: Model to save (can be DDP wrapped)
            optimizer: Optimizer to save
            scheduler: Learning rate scheduler to save
            epoch: Current epoch
            global_step: Current global step
            best_loss: Best loss so far
            wandb_run_id: WandB run ID for resume
            name: Checkpoint name (default: epoch_{epoch})
            extra_metadata: Additional metadata to save
        """
        if not self.is_main_process:
            return
        
        # Determine checkpoint name
        if name is None:
            name = f"epoch_{epoch}"
        
        checkpoint_path = self.checkpoint_dir / name
        tmp_checkpoint_path = self.checkpoint_dir / f"tmp_{name}"
        
        # Clean up temp directory if exists
        if tmp_checkpoint_path.exists():
            shutil.rmtree(tmp_checkpoint_path)
        tmp_checkpoint_path.mkdir(parents=True)
        
        # Get model state dict (unwrap DDP if needed)
        model_to_save = model.module if isinstance(model, DDP) else model
        
        # Save model
        if self.use_safetensors:
            try:
                import safetensors.torch
                safetensors.torch.save_model(
                    model_to_save, 
                    tmp_checkpoint_path / "model.safetensors"
                )
            except ImportError:
                torch.save(
                    model_to_save.state_dict(),
                    tmp_checkpoint_path / "model.pt"
                )
        else:
            torch.save(
                model_to_save.state_dict(),
                tmp_checkpoint_path / "model.pt"
            )
        
        # Save optimizer
        if self.save_optimizer and optimizer is not None:
            torch.save(
                optimizer.state_dict(),
                tmp_checkpoint_path / "optimizer.pt"
            )
        
        # Save scheduler
        if self.save_scheduler and scheduler is not None:
            torch.save(
                scheduler.state_dict(),
                tmp_checkpoint_path / "scheduler.pt"
            )
        
        # Save metadata
        metadata = {
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "wandb_run_id": wandb_run_id,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        
        with open(tmp_checkpoint_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Atomic move: delete old, rename temp
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
        tmp_checkpoint_path.rename(checkpoint_path)
        
        # Update latest pointer
        self._update_latest(checkpoint_path)
        
        # Update training state
        self._save_training_state(epoch, global_step, best_loss, wandb_run_id)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def save_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        best_loss: float = float('inf'),
        wandb_run_id: Optional[str] = None,
    ):
        """
        Save best checkpoint.
        
        Args:
            Same as save()
        """
        self.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_loss=best_loss,
            wandb_run_id=wandb_run_id,
            name="best",
        )
    
    def load_if_exists(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_name: str = "latest",
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint if it exists.
        
        Args:
            model: Model to load weights into (can be DDP wrapped)
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            checkpoint_name: Name of checkpoint to load ("latest", "best", or specific)
            
        Returns:
            Metadata dict if checkpoint was loaded, None otherwise
        """
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_name)
        
        if checkpoint_path is None or not checkpoint_path.exists():
            return None
        
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # Get model to load into (unwrap DDP if needed)
        model_to_load = model.module if isinstance(model, DDP) else model
        
        # Load model
        model_file = checkpoint_path / "model.safetensors"
        if not model_file.exists():
            model_file = checkpoint_path / "model.pt"
        
        if model_file.suffix == ".safetensors":
            try:
                import safetensors.torch
                state_dict = safetensors.torch.load_file(model_file, device='cpu')
                model_to_load.load_state_dict(state_dict)
            except ImportError:
                raise ImportError("safetensors required to load this checkpoint")
        else:
            state_dict = torch.load(model_file, map_location='cpu')
            model_to_load.load_state_dict(state_dict)
        
        # Load optimizer
        optimizer_file = checkpoint_path / "optimizer.pt"
        if optimizer is not None and optimizer_file.exists():
            optimizer_state = torch.load(optimizer_file, map_location='cpu')
            optimizer.load_state_dict(optimizer_state)
        
        # Load scheduler
        scheduler_file = checkpoint_path / "scheduler.pt"
        if scheduler is not None and scheduler_file.exists():
            scheduler_state = torch.load(scheduler_file, map_location='cpu')
            scheduler.load_state_dict(scheduler_state)
        
        # Load metadata
        metadata_file = checkpoint_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        print(f"Loaded checkpoint: epoch={metadata.get('epoch', 'N/A')}, "
              f"step={metadata.get('global_step', 'N/A')}")
        
        return metadata
    
    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """
        Get path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint directory, or None if no checkpoints exist
        """
        return self._resolve_checkpoint_path("latest")
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """
        Get path to the best checkpoint.
        
        Returns:
            Path to best checkpoint directory, or None if no best checkpoint exists
        """
        best_path = self.checkpoint_dir / "best"
        return best_path if best_path.exists() else None
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint names
        """
        if not self.checkpoint_dir.exists():
            return []
        
        checkpoints = []
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                checkpoints.append(item.name)
        
        return sorted(checkpoints)
    
    def _resolve_checkpoint_path(self, name: str) -> Optional[Path]:
        """
        Resolve checkpoint name to path.
        
        Handles "latest" by finding the most recent checkpoint.
        
        Args:
            name: Checkpoint name
            
        Returns:
            Path to checkpoint directory, or None if not found
        """
        if name == "latest":
            # Try latest symlink/directory first
            latest_path = self.checkpoint_dir / "latest"
            if latest_path.exists():
                # Handle symlink
                if latest_path.is_symlink():
                    return latest_path.resolve()
                return latest_path
            
            # Fall back to finding highest epoch
            return self._find_highest_epoch_checkpoint()
        else:
            checkpoint_path = self.checkpoint_dir / name
            return checkpoint_path if checkpoint_path.exists() else None
    
    def _find_highest_epoch_checkpoint(self) -> Optional[Path]:
        """
        Find checkpoint with highest epoch number.
        
        Returns:
            Path to highest epoch checkpoint, or None if none found
        """
        if not self.checkpoint_dir.exists():
            return None
        
        max_epoch = -1
        best_checkpoint = None
        
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("epoch_"):
                try:
                    epoch = int(item.name.split("_")[1])
                    if epoch > max_epoch:
                        max_epoch = epoch
                        best_checkpoint = item
                except ValueError:
                    continue
        
        return best_checkpoint
    
    def _update_latest(self, checkpoint_path: Path):
        """
        Update 'latest' pointer to the given checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint to point to
        """
        latest_path = self.checkpoint_dir / "latest"
        
        # Remove existing latest
        if latest_path.exists() or latest_path.is_symlink():
            if latest_path.is_dir() and not latest_path.is_symlink():
                shutil.rmtree(latest_path)
            else:
                latest_path.unlink()
        
        # Create copy (safer than symlink for cloud storage)
        shutil.copytree(checkpoint_path, latest_path)
    
    def _save_training_state(
        self,
        epoch: int,
        global_step: int,
        best_loss: float,
        wandb_run_id: Optional[str],
    ):
        """
        Save overall training state metadata.
        
        Args:
            epoch: Current epoch
            global_step: Current global step
            best_loss: Best loss achieved
            wandb_run_id: WandB run ID
        """
        state_file = self.checkpoint_dir / "training_state.json"
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "wandb_run_id": wandb_run_id,
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

