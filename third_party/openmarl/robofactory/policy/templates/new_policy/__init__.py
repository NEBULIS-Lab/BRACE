"""
Template for a new VLA policy implementation.

This package provides a minimal working example that can be copied
and modified to add a new VLA policy to OpenMARL.
"""

from .policy import NewPolicy
from .workspace import NewPolicyWorkspace

__all__ = ["NewPolicy", "NewPolicyWorkspace"]

