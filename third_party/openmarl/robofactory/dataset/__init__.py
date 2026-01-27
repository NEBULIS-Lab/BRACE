"""
RoboFactory Dataset Module - Unified data conversion utilities.

This module provides a centralized location for all data format conversion
utilities used by the VLA policies in RoboFactory.

Supported conversions:
- ZARR → LeRobot (for Pi0/Pi0.5)
- ZARR → RLDS (for OpenVLA)
- ZARR → ZARR-DP (for Diffusion-Policy)

Usage:
    from robofactory.dataset import ZarrToLeRobotConverter, ZarrToRLDSConverter
    from robofactory.dataset import convert_data
    
    # Using converter classes
    converter = ZarrToLeRobotConverter()
    converter.convert(input_path, output_path, task_name='LiftBarrier-rf', agent_id=0)
    
    # Using convenience functions
    convert_data(input_path, output_path, format='lerobot', task_name='LiftBarrier-rf')
"""

from .converters import (
    BaseDataConverter,
    ZarrToLeRobotConverter,
    ZarrToRLDSConverter,
)

from .utils import (
    load_zarr_data,
    create_dataset_statistics,
    process_image,
)

__all__ = [
    # Converter classes
    'BaseDataConverter',
    'ZarrToLeRobotConverter',
    'ZarrToRLDSConverter',
    # Utility functions
    'load_zarr_data',
    'create_dataset_statistics',
    'process_image',
]

