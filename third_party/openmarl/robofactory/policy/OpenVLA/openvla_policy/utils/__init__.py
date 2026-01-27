"""
Utility functions for OpenVLA policy.

All utilities are now in shared locations:
- Data conversion: robofactory.dataset.converters
- Action utils: robofactory.policy.shared.action_utils
- Task instructions: robofactory.policy.shared.task_instructions
"""

# Re-export from shared locations for backward compatibility
from robofactory.dataset.converters import (
    convert_zarr_to_rlds,
    convert_zarr_to_rlds_global,
)
from robofactory.dataset.converters.zarr_to_rlds import create_dataset_statistics

from robofactory.policy.shared.action_utils import (
    normalize_action,
    denormalize_action,
)

from robofactory.policy.shared.task_instructions import (
    get_task_instruction,
)

__all__ = [
    "convert_zarr_to_rlds",
    "convert_zarr_to_rlds_global",
    "create_dataset_statistics",
    "normalize_action",
    "denormalize_action",
    "get_task_instruction",
]
