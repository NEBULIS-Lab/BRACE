"""
Shared evaluation framework for VLA policies.

This module provides common evaluation utilities that work with all
VLA policy implementations.

Usage:
    from robofactory.policy.shared.evaluation import BaseEvaluator, BasePolicyWrapper
    from robofactory.policy.shared.evaluation import run_evaluation, detect_policy_type
"""

from .base_evaluator import BaseEvaluator, EvalConfig, EvalResult
from .policy_wrapper import (
    BasePolicyWrapper,
    ActionChunkingWrapper,
    ObservationHistoryWrapper,
)
from .unified_eval import (
    detect_policy_type,
    create_policy_wrapper,
    run_evaluation,
    OpenVLAEvalWrapper,
    Pi0EvalWrapper,
    DiffusionPolicyEvalWrapper,
)

__all__ = [
    # Base classes
    "BaseEvaluator",
    "EvalConfig",
    "EvalResult",
    "BasePolicyWrapper",
    "ActionChunkingWrapper",
    "ObservationHistoryWrapper",
    # Unified evaluation
    "detect_policy_type",
    "create_policy_wrapper",
    "run_evaluation",
    # Policy-specific wrappers
    "OpenVLAEvalWrapper",
    "Pi0EvalWrapper",
    "DiffusionPolicyEvalWrapper",
]

