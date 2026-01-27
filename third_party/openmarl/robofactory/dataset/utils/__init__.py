"""
Dataset utilities for RoboFactory data conversion.

This module provides common utilities used by data converters,
including ZARR loading, statistics computation, and image processing.
"""

from .zarr_utils import load_zarr_data, get_episode_boundaries
from .statistics import create_dataset_statistics, compute_action_statistics
from .image_utils import process_image, hwc_to_chw, chw_to_hwc

__all__ = [
    # ZARR utilities
    'load_zarr_data',
    'get_episode_boundaries',
    # Statistics
    'create_dataset_statistics',
    'compute_action_statistics',
    # Image processing
    'process_image',
    'hwc_to_chw',
    'chw_to_hwc',
]

