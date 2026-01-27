"""
Core framework components for VLA policies.

This module provides base classes and utilities that all VLA policy implementations
should inherit from or use, ensuring consistent interfaces and reducing code duplication.

Usage:
    from robofactory.policy.core import BaseVLAWorkspace, BaseVLAPolicy, BaseVLADataset
    from robofactory.policy.core import setup_distributed, is_main_process
    from robofactory.policy.core import CheckpointManager, init_wandb
"""

from .base_workspace import BaseVLAWorkspace
from .base_policy import BaseVLAPolicy
from .base_dataset import BaseVLADataset
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    reduce_tensor,
)
from .logging_utils import (
    init_wandb,
    log_metrics,
    log_images,
    finish_wandb,
)
from .checkpoint_manager import CheckpointManager

__all__ = [
    # Base classes
    "BaseVLAWorkspace",
    "BaseVLAPolicy",
    "BaseVLADataset",
    # Distributed utilities
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "reduce_tensor",
    # Logging utilities
    "init_wandb",
    "log_metrics",
    "log_images",
    "finish_wandb",
    # Checkpoint management
    "CheckpointManager",
]

