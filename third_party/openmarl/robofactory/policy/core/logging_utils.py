"""
Logging utilities for VLA policy training.

This module provides common utilities for WandB logging that all policy
implementations can use, ensuring consistent logging behavior and metrics
across the codebase.

Patterns extracted from:
- Pi0Workspace._init_wandb: Resume support, run ID tracking
- OpenVLAWorkspace.run: Config logging, summary metrics
- robotworkspace.py: Step-level logging, image logging
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from omegaconf import OmegaConf

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def init_wandb(
    cfg: OmegaConf,
    resuming: bool = False,
    is_main_process: bool = True,
    run_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    Initialize WandB logging.
    
    Args:
        cfg: Configuration object containing logging settings
        resuming: Whether resuming from a checkpoint
        is_main_process: Whether this is the main process
        run_id: Optional run ID for resumption
        output_dir: Optional output directory for WandB files
        checkpoint_dir: Optional checkpoint directory for wandb_id file
        
    Returns:
        WandB run ID if initialized, None otherwise
        
    Configuration expected:
        cfg.logging.project: Project name
        cfg.logging.name: Run name
        cfg.logging.mode: "online", "offline", or "disabled"
        cfg.logging.tags: Optional list of tags
        cfg.logging.wandb_enabled: Optional boolean to enable/disable
    """
    if not is_main_process:
        return None
    
    if not WANDB_AVAILABLE:
        print("Warning: wandb not installed, logging disabled")
        return None
    
    # Check if wandb is enabled in config
    logging_cfg = cfg.get('logging', {})
    if not logging_cfg.get('wandb_enabled', True):
        wandb.init(mode="disabled")
        return None
    
    mode = logging_cfg.get('mode', 'online')
    if mode == "disabled":
        wandb.init(mode="disabled")
        return None
    
    # Try to load run_id from checkpoint_dir if resuming
    wandb_id_file = None
    if checkpoint_dir is not None:
        wandb_id_file = Path(checkpoint_dir) / "wandb_id.txt"
        if resuming and run_id is None and wandb_id_file.exists():
            run_id = wandb_id_file.read_text().strip()
            print(f"Found wandb ID from checkpoint: {run_id}")
    
    # Prepare wandb init kwargs
    wandb_kwargs = {
        "project": logging_cfg.get('project', 'openmarl'),
        "name": logging_cfg.get('name', None),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "mode": mode,
    }
    
    # Add tags if specified
    tags = logging_cfg.get('tags', None)
    if tags:
        wandb_kwargs["tags"] = list(tags) if not isinstance(tags, list) else tags
    
    # Add output directory
    if output_dir:
        wandb_kwargs["dir"] = str(output_dir)
    
    # Handle resume
    if resuming and run_id:
        wandb_kwargs["id"] = run_id
        wandb_kwargs["resume"] = "allow"
        print(f"Resuming wandb run: {run_id}")
    
    # Initialize
    try:
        wandb.init(**wandb_kwargs)
        new_run_id = wandb.run.id
        
        # Save run ID to checkpoint_dir for future resumption
        if wandb_id_file is not None and new_run_id:
            try:
                wandb_id_file.parent.mkdir(parents=True, exist_ok=True)
                wandb_id_file.write_text(new_run_id)
            except Exception as e:
                print(f"Warning: Could not save wandb ID to {wandb_id_file}: {e}")
        
        if resuming and run_id:
            print(f"Resumed wandb run: {new_run_id}")
        else:
            print(f"Started new wandb run: {new_run_id}")
        
        return new_run_id
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        return None


def log_metrics(
    metrics: Dict[str, Any],
    step: int,
    is_main_process: bool = True,
):
    """
    Log metrics to WandB.
    
    Args:
        metrics: Dictionary of metric names to values
        step: Global step for x-axis
        is_main_process: Whether this is the main process
        
    Example:
        log_metrics({
            'train/loss': 0.5,
            'train/lr': 1e-4,
        }, step=1000)
    """
    if not is_main_process:
        return
    
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    try:
        wandb.log(metrics, step=step)
    except Exception as e:
        print(f"Warning: Failed to log metrics: {e}")


def log_images(
    images: Dict[str, Any],
    step: int,
    is_main_process: bool = True,
    caption_prefix: str = "",
):
    """
    Log images to WandB.
    
    Args:
        images: Dictionary mapping names to images
                Images can be numpy arrays (HWC or CHW), PIL Images, or paths
        step: Global step for x-axis
        is_main_process: Whether this is the main process
        caption_prefix: Optional prefix for image captions
        
    Example:
        log_images({
            'observation': obs_image,  # numpy array (H, W, 3)
            'action_plot': plot_image,
        }, step=1000)
    """
    if not is_main_process:
        return
    
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    try:
        wandb_images = {}
        for name, img in images.items():
            if isinstance(img, np.ndarray):
                # Handle CHW format
                if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                    img = np.transpose(img, (1, 2, 0))
                
                # Normalize to [0, 255] if needed
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                elif img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                
                caption = f"{caption_prefix}{name}" if caption_prefix else name
                wandb_images[name] = wandb.Image(img, caption=caption)
            else:
                # Assume it's already a wandb.Image or PIL Image
                wandb_images[name] = wandb.Image(img)
        
        wandb.log(wandb_images, step=step)
    except Exception as e:
        print(f"Warning: Failed to log images: {e}")


def log_video(
    video: np.ndarray,
    name: str,
    step: int,
    is_main_process: bool = True,
    fps: int = 10,
):
    """
    Log video to WandB.
    
    Args:
        video: Video array with shape (T, H, W, C) or (T, C, H, W)
        name: Name for the video
        step: Global step for x-axis
        is_main_process: Whether this is the main process
        fps: Frames per second
    """
    if not is_main_process:
        return
    
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    try:
        # Handle THWC vs TCHW format
        if video.ndim == 4 and video.shape[-1] == 3:
            # THWC -> TCHW
            video = np.transpose(video, (0, 3, 1, 2))
        
        wandb.log({
            name: wandb.Video(video, fps=fps, format="mp4")
        }, step=step)
    except Exception as e:
        print(f"Warning: Failed to log video: {e}")


def log_histogram(
    data: np.ndarray,
    name: str,
    step: int,
    is_main_process: bool = True,
):
    """
    Log histogram to WandB.
    
    Args:
        data: Data array to create histogram from
        name: Name for the histogram
        step: Global step for x-axis
        is_main_process: Whether this is the main process
    """
    if not is_main_process:
        return
    
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    try:
        wandb.log({name: wandb.Histogram(data)}, step=step)
    except Exception as e:
        print(f"Warning: Failed to log histogram: {e}")


def finish_wandb(
    summary: Optional[Dict[str, Any]] = None,
    is_main_process: bool = True,
):
    """
    Finish WandB logging and upload summary.
    
    Args:
        summary: Optional dictionary of summary metrics
        is_main_process: Whether this is the main process
        
    Example:
        finish_wandb({
            'final_loss': 0.1,
            'best_loss': 0.08,
            'total_epochs': 100,
        })
    """
    if not is_main_process:
        return
    
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    try:
        # Log summary metrics
        if summary:
            for key, value in summary.items():
                wandb.run.summary[key] = value
        
        # Finish run
        wandb.finish()
    except Exception as e:
        print(f"Warning: Failed to finish wandb: {e}")


def update_config(
    updates: Dict[str, Any],
    is_main_process: bool = True,
):
    """
    Update WandB config.
    
    Args:
        updates: Dictionary of config updates
        is_main_process: Whether this is the main process
    """
    if not is_main_process:
        return
    
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    try:
        wandb.config.update(updates, allow_val_change=True)
    except Exception as e:
        print(f"Warning: Failed to update wandb config: {e}")

