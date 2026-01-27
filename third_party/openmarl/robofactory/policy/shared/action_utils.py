"""
Action normalization and processing utilities for VLA policies.

This module provides common utilities for action normalization/denormalization
used by all VLA policies, supporting both Gaussian and quantile normalization.

Consolidated from:
- OpenVLA action_utils.py: Gaussian normalization
- Pi0 pi0_wrapper.py: Quantile normalization (openpi convention)
"""

from typing import Dict, Any, Union, Optional
import numpy as np
import torch


# Type alias for numeric arrays
ArrayLike = Union[np.ndarray, torch.Tensor]


def normalize_action(
    action: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    eps: float = 1e-8,
) -> ArrayLike:
    """
    Normalize actions using Gaussian (mean/std) normalization.
    
    z = (x - mean) / (std + eps)
    
    Args:
        action: Action array/tensor to normalize
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Normalized action (same type as input)
        
    Example:
        >>> action = np.array([1.0, 2.0, 3.0])
        >>> mean = np.array([0.0, 1.0, 2.0])
        >>> std = np.array([1.0, 1.0, 1.0])
        >>> normalize_action(action, mean, std)
        array([1., 1., 1.])
    """
    return (action - mean) / (std + eps)


def denormalize_action(
    action: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
) -> ArrayLike:
    """
    Denormalize actions from Gaussian normalization back to original scale.
    
    x = z * std + mean
    
    Args:
        action: Normalized action array/tensor
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        Denormalized action (same type as input)
    """
    return action * std + mean


def normalize_quantile(
    data: ArrayLike,
    q01: ArrayLike,
    q99: ArrayLike,
    eps: float = 1e-8,
) -> ArrayLike:
    """
    Normalize data using quantile normalization (openpi convention).
    
    Maps [q01, q99] to [0, 1], with values potentially outside this range.
    
    z = (x - q01) / (q99 - q01 + eps)
    
    Args:
        data: Data array/tensor to normalize
        q01: 1st percentile values
        q99: 99th percentile values
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Normalized data with values approximately in [0, 1]
        
    Example:
        >>> data = torch.tensor([0.5, 1.0, 1.5])
        >>> q01 = torch.tensor([0.0, 0.0, 0.0])
        >>> q99 = torch.tensor([2.0, 2.0, 2.0])
        >>> normalize_quantile(data, q01, q99)
        tensor([0.25, 0.5, 0.75])
    """
    return (data - q01) / (q99 - q01 + eps)


def denormalize_quantile(
    data: ArrayLike,
    q01: ArrayLike,
    q99: ArrayLike,
) -> ArrayLike:
    """
    Denormalize data from quantile normalization back to original scale.
    
    x = z * (q99 - q01) + q01
    
    Args:
        data: Normalized data array/tensor
        q01: 1st percentile values used for normalization
        q99: 99th percentile values used for normalization
        
    Returns:
        Denormalized data (same type as input)
    """
    return data * (q99 - q01) + q01


def compute_action_statistics(
    actions: np.ndarray,
    compute_quantiles: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute statistics from action array.
    
    Computes mean, std, min, max, and optionally quantiles (q01, q99)
    for use in normalization.
    
    Args:
        actions: Array of actions with shape (N, action_dim) or (N, horizon, action_dim)
        compute_quantiles: Whether to compute q01 and q99 quantiles
        
    Returns:
        Dictionary containing:
            - 'mean': Mean values [action_dim]
            - 'std': Standard deviation values [action_dim]
            - 'min': Minimum values [action_dim]
            - 'max': Maximum values [action_dim]
            - 'q01': 1st percentile values [action_dim] (if compute_quantiles)
            - 'q99': 99th percentile values [action_dim] (if compute_quantiles)
    """
    # Flatten actions if they have horizon dimension
    if actions.ndim == 3:
        # (N, horizon, action_dim) -> (N * horizon, action_dim)
        actions = actions.reshape(-1, actions.shape[-1])
    
    statistics = {
        'mean': actions.mean(axis=0).astype(np.float32),
        'std': actions.std(axis=0).astype(np.float32),
        'min': actions.min(axis=0).astype(np.float32),
        'max': actions.max(axis=0).astype(np.float32),
    }
    
    if compute_quantiles:
        statistics['q01'] = np.percentile(actions, 1, axis=0).astype(np.float32)
        statistics['q99'] = np.percentile(actions, 99, axis=0).astype(np.float32)
    
    return statistics


def compute_state_statistics(
    states: np.ndarray,
    compute_quantiles: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute statistics from state array.
    
    Equivalent to compute_action_statistics but for proprioceptive state.
    
    Args:
        states: Array of states with shape (N, state_dim)
        compute_quantiles: Whether to compute q01 and q99 quantiles
        
    Returns:
        Dictionary of statistics
    """
    return compute_action_statistics(states, compute_quantiles)


def action_to_7dof(action: np.ndarray) -> np.ndarray:
    """
    Convert action to 7-DoF format (6 joint angles + 1 gripper state).
    
    Some robots use 8-DoF actions while others use 7-DoF. This function
    converts from 8-DoF to 7-DoF by taking the first 7 dimensions.
    
    Args:
        action: Action array, potentially with 8 DoF
        
    Returns:
        7-DoF action array
        
    Raises:
        ValueError: If action dimension is not 7 or 8
    """
    if len(action) == 8:
        # Assume last element is gripper, take first 7 DoF
        return action[:7]
    elif len(action) == 7:
        return action
    else:
        raise ValueError(f"Unexpected action dimension: {len(action)}")


def clip_action(
    action: ArrayLike,
    min_val: Optional[ArrayLike] = None,
    max_val: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Clip action values to specified range.
    
    Args:
        action: Action array/tensor
        min_val: Minimum values (per-dimension or scalar)
        max_val: Maximum values (per-dimension or scalar)
        
    Returns:
        Clipped action
    """
    if isinstance(action, torch.Tensor):
        if min_val is not None or max_val is not None:
            action = torch.clamp(action, min=min_val, max=max_val)
    else:
        if min_val is not None or max_val is not None:
            action = np.clip(action, min_val, max_val)
    return action


def smooth_action(
    action: np.ndarray,
    prev_action: Optional[np.ndarray] = None,
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Apply exponential smoothing to action.
    
    Useful for reducing jitter in predicted actions.
    
    smoothed = alpha * action + (1 - alpha) * prev_action
    
    Args:
        action: Current action
        prev_action: Previous action (if None, returns action unchanged)
        alpha: Smoothing factor (higher = more weight on current action)
        
    Returns:
        Smoothed action
    """
    if prev_action is None:
        return action
    return alpha * action + (1 - alpha) * prev_action

