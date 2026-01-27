"""
Dataset statistics computation utilities.

Provides functions for computing normalization statistics from datasets.
"""

import json
import os
from typing import Dict, Any, Optional

import numpy as np

from .zarr_utils import load_zarr_data


def create_dataset_statistics(
    zarr_path: str,
    output_path: str,
    action_key: str = 'action',
    state_key: str = 'state'
) -> Dict[str, Any]:
    """
    Create dataset statistics JSON file for normalization.
    
    Computes mean, std, min, max, and quantiles for action and state data.
    
    Args:
        zarr_path: Path to ZARR dataset
        output_path: Path to save statistics JSON
        action_key: Key for action data
        state_key: Key for state/proprio data
        
    Returns:
        Dictionary containing dataset statistics
    """
    data = load_zarr_data(zarr_path)
    
    statistics = {}
    
    # Compute action statistics
    if action_key in data:
        statistics['action'] = compute_action_statistics(data[action_key])
    
    # Compute proprio/state statistics  
    if state_key in data:
        statistics['proprio'] = compute_action_statistics(data[state_key])
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"Saved dataset statistics to {output_path}")
    return statistics


def compute_action_statistics(
    actions: np.ndarray,
    compute_quantiles: bool = True,
) -> Dict[str, Any]:
    """
    Compute statistics from action/state array.
    
    Args:
        actions: Array of actions with shape (N, action_dim) or (N, horizon, action_dim)
        compute_quantiles: Whether to compute q01 and q99 quantiles
        
    Returns:
        Dictionary containing:
            - 'mean': Mean values
            - 'std': Standard deviation values
            - 'min': Minimum values
            - 'max': Maximum values
            - 'q01': 1st percentile values (if compute_quantiles)
            - 'q99': 99th percentile values (if compute_quantiles)
    """
    # Flatten actions if they have horizon dimension
    if actions.ndim == 3:
        actions = actions.reshape(-1, actions.shape[-1])
    
    statistics = {
        'mean': actions.mean(axis=0).tolist(),
        'std': actions.std(axis=0).tolist(),
        'min': actions.min(axis=0).tolist(),
        'max': actions.max(axis=0).tolist(),
    }
    
    if compute_quantiles:
        statistics['q01'] = np.percentile(actions, 1, axis=0).tolist()
        statistics['q99'] = np.percentile(actions, 99, axis=0).tolist()
    
    return statistics


def load_statistics(stats_path: str) -> Dict[str, Any]:
    """
    Load statistics from JSON file.
    
    Args:
        stats_path: Path to statistics JSON file
        
    Returns:
        Dictionary containing statistics
    """
    with open(stats_path, 'r') as f:
        return json.load(f)


def merge_statistics(stats_list: list) -> Dict[str, Any]:
    """
    Merge statistics from multiple datasets.
    
    Uses weighted average for mean, pooled std, and overall min/max.
    
    Args:
        stats_list: List of (statistics_dict, num_samples) tuples
        
    Returns:
        Merged statistics dictionary
    """
    if not stats_list:
        return {}
    
    # Extract keys from first statistics dict
    first_stats, _ = stats_list[0]
    merged = {}
    
    for key in first_stats.keys():
        if key not in ['action', 'proprio']:
            continue
            
        # Collect values
        means = []
        stds = []
        mins = []
        maxs = []
        weights = []
        q01s = []
        q99s = []
        
        for stats, n_samples in stats_list:
            if key in stats:
                means.append(np.array(stats[key]['mean']))
                stds.append(np.array(stats[key]['std']))
                mins.append(np.array(stats[key]['min']))
                maxs.append(np.array(stats[key]['max']))
                weights.append(n_samples)
                if 'q01' in stats[key]:
                    q01s.append(np.array(stats[key]['q01']))
                if 'q99' in stats[key]:
                    q99s.append(np.array(stats[key]['q99']))
        
        if not means:
            continue
        
        weights = np.array(weights)
        total_weight = weights.sum()
        
        # Weighted average for mean
        merged_mean = sum(m * w for m, w in zip(means, weights)) / total_weight
        
        # Pooled standard deviation (approximate)
        merged_std = np.sqrt(
            sum(((s**2 + (m - merged_mean)**2) * w) 
                for s, m, w in zip(stds, means, weights)) / total_weight
        )
        
        merged[key] = {
            'mean': merged_mean.tolist(),
            'std': merged_std.tolist(),
            'min': np.min(mins, axis=0).tolist(),
            'max': np.max(maxs, axis=0).tolist(),
        }
        
        if q01s and q99s:
            merged[key]['q01'] = np.min(q01s, axis=0).tolist()
            merged[key]['q99'] = np.max(q99s, axis=0).tolist()
    
    return merged

