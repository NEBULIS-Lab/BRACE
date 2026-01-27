"""
Shared utilities for VLA policies.

This module provides common utilities that all VLA policy implementations
can use, reducing code duplication across policies.

Usage:
    from robofactory.policy.shared import get_task_instruction
    from robofactory.policy.shared import normalize_action, denormalize_action
    from robofactory.policy.shared import process_image, hwc_to_chw
"""

from .task_instructions import (
    get_task_instruction,
    get_all_task_instructions,
    TASK_INSTRUCTIONS_DETAILED,
    TASK_INSTRUCTIONS_SIMPLE,
)
from .action_utils import (
    normalize_action,
    denormalize_action,
    normalize_quantile,
    denormalize_quantile,
    compute_action_statistics,
)
from .image_utils import (
    process_image,
    hwc_to_chw,
    chw_to_hwc,
    normalize_image,
    denormalize_image,
    random_crop,
    center_crop,
)

__all__ = [
    # Task instructions
    "get_task_instruction",
    "get_all_task_instructions",
    "TASK_INSTRUCTIONS_DETAILED",
    "TASK_INSTRUCTIONS_SIMPLE",
    # Action utilities
    "normalize_action",
    "denormalize_action",
    "normalize_quantile",
    "denormalize_quantile",
    "compute_action_statistics",
    # Image utilities
    "process_image",
    "hwc_to_chw",
    "chw_to_hwc",
    "normalize_image",
    "denormalize_image",
    "random_crop",
    "center_crop",
]

