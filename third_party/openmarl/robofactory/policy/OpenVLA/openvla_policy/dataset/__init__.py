"""Dataset module for OpenVLA."""

# Import BaseVLADataset from core (shared across all policies)
from robofactory.policy.core import BaseVLADataset

# Import OpenVLA-specific dataset
from .robot_rlds_dataset import RobotRLDSDataset

__all__ = [
    "BaseVLADataset",
    "RobotRLDSDataset",
]
