"""
ZARR loading and handling utilities.

Common utilities for loading and working with RoboFactory ZARR datasets.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import zarr


def load_zarr_data(zarr_path: str) -> Dict[str, np.ndarray]:
    """
    Load data from ZARR file.
    
    Args:
        zarr_path: Path to ZARR dataset
        
    Returns:
        Dictionary containing loaded data arrays including:
        - 'episode_ends': Episode boundary indices
        - Camera data: 'head_camera', 'wrist_camera', or 'img'
        - 'action': Action data
        - 'state' or 'agent_pos': State data
    """
    root = zarr.open(zarr_path, mode='r')
    
    data = {}
    
    # Load metadata
    if 'meta' in root:
        meta = root['meta']
        if 'episode_ends' in meta:
            data['episode_ends'] = np.array(meta['episode_ends'])
    
    # Load data arrays
    if 'data' in root:
        data_group = root['data']
        for key in data_group.keys():
            data[key] = np.array(data_group[key])
    
    return data


def get_episode_boundaries(
    zarr_path: str,
    default_length: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Get episode boundaries from ZARR dataset.
    
    Args:
        zarr_path: Path to ZARR dataset
        default_length: Default total length if episode_ends not found
        
    Returns:
        Tuple of (episode_ends array, total_steps)
    """
    root = zarr.open(zarr_path, mode='r')
    
    if 'meta' in root and 'episode_ends' in root['meta']:
        episode_ends = np.array(root['meta']['episode_ends'])
        total_steps = int(episode_ends[-1])
    elif default_length is not None:
        episode_ends = np.array([default_length])
        total_steps = default_length
    else:
        # Try to infer from data
        if 'data' in root:
            data_group = root['data']
            # Find first array to get length
            for key in data_group.keys():
                total_steps = len(data_group[key])
                break
        else:
            raise ValueError("Cannot determine dataset length")
        episode_ends = np.array([total_steps])
    
    return episode_ends, total_steps


def get_zarr_info(zarr_path: str) -> Dict:
    """
    Get information about a ZARR dataset.
    
    Args:
        zarr_path: Path to ZARR dataset
        
    Returns:
        Dictionary with dataset information
    """
    root = zarr.open(zarr_path, mode='r')
    
    info = {
        'path': zarr_path,
        'groups': list(root.keys()),
    }
    
    if 'meta' in root:
        meta = root['meta']
        if 'episode_ends' in meta:
            episode_ends = np.array(meta['episode_ends'])
            info['num_episodes'] = len(episode_ends)
            info['total_steps'] = int(episode_ends[-1])
    
    if 'data' in root:
        data_group = root['data']
        info['data_keys'] = list(data_group.keys())
        
        # Get shapes
        info['shapes'] = {}
        for key in data_group.keys():
            info['shapes'][key] = data_group[key].shape
    
    return info


def iter_episodes(zarr_path: str):
    """
    Iterate over episodes in a ZARR dataset.
    
    Args:
        zarr_path: Path to ZARR dataset
        
    Yields:
        Tuple of (episode_index, start_idx, end_idx, episode_data)
    """
    data = load_zarr_data(zarr_path)
    
    episode_ends = data.get('episode_ends', None)
    if episode_ends is None:
        # Assume single episode
        first_key = [k for k in data.keys() if k != 'episode_ends'][0]
        total_steps = len(data[first_key])
        episode_ends = np.array([total_steps])
    
    start_idx = 0
    for ep_idx, end_idx in enumerate(episode_ends):
        episode_data = {
            key: val[start_idx:end_idx] 
            for key, val in data.items() 
            if key != 'episode_ends'
        }
        yield ep_idx, start_idx, end_idx, episode_data
        start_idx = end_idx

