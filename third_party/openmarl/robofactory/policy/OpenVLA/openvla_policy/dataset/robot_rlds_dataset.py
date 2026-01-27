"""RLDS dataset loader for RoboFactory tasks with OpenVLA - Multi-view support."""

import json
import os
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import tensorflow as tf
from robofactory.policy.core import BaseVLADataset


class RobotRLDSDataset(BaseVLADataset):
    """
    RLDS dataset loader with multi-view image support.
    
    Supports loading multiple camera views:
    - primary: head/side camera (third-person view)
    - secondary: global camera (overhead view) - loaded from separate global folder
    - wrist: wrist/gripper camera (end-effector view)
    """
    
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
        augment_crop_ratio: float = 0.9,
        val_split: float = 0.1,
        seed: int = 42,
        use_multi_view: bool = False,
        image_views: List[str] = None,  # ['primary', 'secondary', 'wrist']
        global_data_dir: Optional[str] = None,  # Path to global camera RLDS data
        auto_detect_global: bool = True,  # Auto-detect global folder from agent path
        use_cache: bool = True,  # Enable dataset caching for faster distributed loading
    ):
        """
        Initialize RLDS dataset with multi-view support.
        
        Args:
            data_dir: Directory containing RLDS dataset (agent-specific)
            train: Whether to load training or validation split
            image_size: Target image size (H, W)
            augment: Whether to apply image augmentation
            augment_crop_ratio: Crop ratio for augmentation
            val_split: Validation split ratio
            seed: Random seed for split
            use_multi_view: Whether to load multiple camera views
            image_views: List of view names to load ['primary', 'secondary', 'wrist']
            global_data_dir: Optional path to global camera RLDS data folder
            auto_detect_global: If True, auto-detect global folder from agent path
                               (e.g., LiftBarrier-rf_Agent0 -> LiftBarrier-rf_global)
            use_cache: If True, cache merged dataset to disk for faster subsequent loads
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.train = train
        self.image_size = image_size
        self.augment = augment and train
        self.augment_crop_ratio = augment_crop_ratio
        self.val_split = val_split
        self.seed = seed
        self.use_multi_view = use_multi_view
        self.image_views = image_views or ['primary', 'secondary', 'wrist']
        self.use_cache = use_cache
        
        # Get distributed training info (must be set early for logging control)
        self._rank = int(os.environ.get("RANK", 0))
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._is_distributed = self._world_size > 1
        
        # Handle global data directory
        self.global_data_dir = None
        if global_data_dir:
            self.global_data_dir = Path(global_data_dir)
        elif auto_detect_global and use_multi_view and 'secondary' in self.image_views:
            # Auto-detect global folder from agent path
            # e.g., data/rlds_data/LiftBarrier-rf_Agent0 -> data/rlds_data/LiftBarrier-rf_global
            self.global_data_dir = self._auto_detect_global_dir()
        
        # Map view names to TFRecord keys
        self.view_to_key = {
            'primary': 'observation/side_image',      # Head camera (side view)
            'secondary': 'observation/global_image',  # Global camera
            'wrist': 'observation/wrist_image',       # Wrist camera (gripper view)
            'image': 'observation/image',             # Fallback single image
        }
        
        # Generate cache path
        self._cache_path = self._get_cache_path() if use_cache else None
        
        # Load dataset info and statistics (always needed)
        self.info = self._load_dataset_info()
        self.statistics = self._load_statistics()
        
        # Try to load from cache first
        if self.use_cache and self._load_from_cache():
            # Successfully loaded from cache
            pass
        else:
            # No cache exists - use rank-0-first pattern to avoid OOM
            # Only rank 0 loads from TFRecords, others wait for cache
            if self._is_distributed and dist.is_initialized():
                if self._rank == 0:
                    # Rank 0 loads and creates cache
                    self.episodes = self._load_episodes()
                    if self.global_data_dir and self.global_data_dir.exists():
                        self._load_and_merge_global_images()
                    self.indices = self._create_split()
                    if self.use_cache:
                        self._save_to_cache()
                
                # All ranks wait for rank 0 to finish saving cache
                dist.barrier()
                
                # Non-rank-0 processes load from cache
                if self._rank != 0:
                    if not self._load_from_cache():
                        raise RuntimeError(f"Rank {self._rank}: Failed to load cache created by rank 0")
            else:
                # Single process - load normally
                self.episodes = self._load_episodes()
                if self.global_data_dir and self.global_data_dir.exists():
                    self._load_and_merge_global_images()
                self.indices = self._create_split()
                if self.use_cache:
                    self._save_to_cache()
        
        if self._rank == 0:
            print(f"Loaded {len(self.indices)} samples ({'train' if train else 'val'})")
            if self.use_multi_view:
                print(f"  Multi-view enabled: {self.image_views}")
                if self.global_data_dir:
                    print(f"  Global camera data: {self.global_data_dir}")
    
    def _auto_detect_global_dir(self) -> Optional[Path]:
        """Auto-detect global data directory from agent path."""
        # Parse agent path: e.g., LiftBarrier-rf_Agent0 or LiftBarrier-rf_Agent0_160
        dir_name = self.data_dir.name
        
        # Extract task name (everything before _Agent)
        if '_Agent' in dir_name:
            task_name = dir_name.split('_Agent')[0]
            global_dir = self.data_dir.parent / f"{task_name}_global"
            
            if global_dir.exists():
                if self._rank == 0:
                    print(f"Auto-detected global data directory: {global_dir}")
                return global_dir
            else:
                if self._rank == 0:
                    print(f"Warning: Global data directory not found at {global_dir}")
        
        return None
    
    def _get_cache_path(self) -> Path:
        """Generate unique cache file path based on dataset configuration."""
        # Create hash of config for unique cache key
        # Added 'v2' to invalidate old caches that used decoded images
        config_str = f"{self.data_dir}_{self.global_data_dir}_{self.val_split}_{self.seed}_{self.image_views}_v2"
        cache_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        split = "train" if self.train else "val"
        return self.data_dir / f".cache_{split}_{cache_hash}.pkl"
    
    def _load_from_cache(self) -> bool:
        """
        Load dataset from cache if available.
        
        Returns:
            True if successfully loaded from cache, False otherwise
        """
        if self._cache_path is None or not self._cache_path.exists():
            return False
        
        try:
            if self._rank == 0:
                print(f"Loading dataset from cache: {self._cache_path}")
            
            with open(self._cache_path, 'rb') as f:
                cached = pickle.load(f)
            
            self.episodes = cached['episodes']
            self.indices = cached['indices']
            
            if self._rank == 0:
                print(f"Loaded {len(self.indices)} samples from cache")
            return True
        except Exception as e:
            if self._rank == 0:
                print(f"Cache load failed: {e}, will reload from TFRecords")
            return False
    
    def _save_to_cache(self):
        """
        Save dataset to cache file.
        Only called by rank 0 in distributed training.
        """
        if self._cache_path is None:
            return
        
        try:
            print(f"Saving dataset cache to: {self._cache_path}")
            cache_data = {
                'episodes': self.episodes,
                'indices': self.indices,
            }
            with open(self._cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cache saved successfully ({self._cache_path.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            print(f"Cache save failed: {e}")
    
    def _load_dataset_info(self) -> Dict:
        """Load dataset info from JSON file."""
        info_path = self.data_dir / 'dataset_info.json'
        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_statistics(self) -> Dict:
        """Load dataset statistics for normalization."""
        stats_path = self.data_dir / 'statistics.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                for key in stats:
                    for subkey in stats[key]:
                        stats[key][subkey] = np.array(stats[key][subkey], dtype=np.float32)
                return stats
        else:
            if self._rank == 0:
                print(f"Warning: Statistics file not found at {stats_path}")
            return {}
    
    def _load_episodes(self):
        """Load all episodes from TFRecord files with multi-view support."""
        tfrecord_path = self.data_dir / 'train.tfrecord'
        
        if not tfrecord_path.exists():
            raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")
        
        episodes = []
        
        # Feature description with multi-view support
        example_keys = {
            'action': tf.io.VarLenFeature(tf.float32),
            'is_first': tf.io.FixedLenFeature([], tf.int64),
            'is_last': tf.io.FixedLenFeature([], tf.int64),
            'is_terminal': tf.io.FixedLenFeature([], tf.int64),
            'language_instruction': tf.io.FixedLenFeature([], tf.string),
            # Multi-view images (with defaults for missing keys)
            'observation/image': tf.io.FixedLenFeature([], tf.string, default_value=b''),
            'observation/wrist_image': tf.io.FixedLenFeature([], tf.string, default_value=b''),
            'observation/side_image': tf.io.FixedLenFeature([], tf.string, default_value=b''),
            'observation/global_image': tf.io.FixedLenFeature([], tf.string, default_value=b''),
            'observation/proprio': tf.io.VarLenFeature(tf.float32),
        }
        
        dataset = tf.data.TFRecordDataset(str(tfrecord_path))
        current_episode = []
        
        for raw_record in dataset:
            example = tf.io.parse_single_example(raw_record, example_keys)
            
            action = tf.sparse.to_dense(example['action']).numpy().astype(np.float32)
            proprio = tf.sparse.to_dense(example['observation/proprio']).numpy().astype(np.float32)
            instruction = example['language_instruction'].numpy().decode('utf-8')
            
            # Load all available images as COMPRESSED BYTES (not decoded)
            # This reduces memory by ~10x - images are decoded on-demand in __getitem__
            images = {}
            
            # Primary: side/head camera
            side_bytes = example['observation/side_image'].numpy()
            if side_bytes:
                images['primary'] = side_bytes  # Store compressed bytes
            
            # Secondary: global camera (may be empty in agent data, will be filled from global folder)
            global_bytes = example['observation/global_image'].numpy()
            if global_bytes:
                images['secondary'] = global_bytes  # Store compressed bytes
            
            # Wrist: gripper camera
            wrist_bytes = example['observation/wrist_image'].numpy()
            if wrist_bytes:
                images['wrist'] = wrist_bytes  # Store compressed bytes
            
            # Fallback: single image
            image_bytes = example['observation/image'].numpy()
            if image_bytes:
                images['image'] = image_bytes  # Store compressed bytes
            
            step = {
                'images': images,
                'proprio': proprio,
                'action': action,
                'instruction': instruction,
                'is_first': bool(example['is_first'].numpy()),
                'is_last': bool(example['is_last'].numpy()),
            }
            
            current_episode.append(step)
            
            if step['is_last']:
                episodes.append(current_episode)
                current_episode = []
        
        # Log available views from first episode (rank 0 only)
        if self._rank == 0:
            if episodes and episodes[0]:
                available_views = list(episodes[0][0]['images'].keys())
                print(f"Available camera views in agent dataset: {available_views}")
            print(f"Loaded {len(episodes)} episodes from {tfrecord_path}")
        
        return episodes
    
    def _load_and_merge_global_images(self):
        """Load global camera images from separate folder and merge with agent data."""
        global_tfrecord = self.global_data_dir / 'train.tfrecord'
        
        if not global_tfrecord.exists():
            if self._rank == 0:
                print(f"Warning: Global TFRecord not found at {global_tfrecord}")
            return
        
        if self._rank == 0:
            print(f"Loading global camera images from {global_tfrecord}...")
        
        # Feature description for global data (no actions)
        global_keys = {
            'is_first': tf.io.FixedLenFeature([], tf.int64),
            'is_last': tf.io.FixedLenFeature([], tf.int64),
            'is_terminal': tf.io.FixedLenFeature([], tf.int64),
            'language_instruction': tf.io.FixedLenFeature([], tf.string),
            'observation/image': tf.io.FixedLenFeature([], tf.string, default_value=b''),
            'observation/global_image': tf.io.FixedLenFeature([], tf.string, default_value=b''),
        }
        
        global_dataset = tf.data.TFRecordDataset(str(global_tfrecord))
        
        # Load global images organized by episode/step
        global_episodes = []
        current_global_episode = []
        
        for raw_record in global_dataset:
            example = tf.io.parse_single_example(raw_record, global_keys)
            
            # Try global_image first, then fallback to image
            # Store as compressed bytes (not decoded) to reduce memory
            global_bytes = example['observation/global_image'].numpy()
            if not global_bytes:
                global_bytes = example['observation/image'].numpy()
            
            step = {
                'global_image': global_bytes if global_bytes else None,  # Store compressed bytes
                'is_first': bool(example['is_first'].numpy()),
                'is_last': bool(example['is_last'].numpy()),
            }
            
            current_global_episode.append(step)
            
            if step['is_last']:
                global_episodes.append(current_global_episode)
                current_global_episode = []
        
        if self._rank == 0:
            print(f"Loaded {len(global_episodes)} global episodes")
        
        # Merge global images into agent episodes
        merged_count = 0
        mismatch_episodes = 0
        
        for ep_idx, (agent_ep, global_ep) in enumerate(zip(self.episodes, global_episodes)):
            if len(agent_ep) != len(global_ep):
                if self._rank == 0:
                    print(f"Warning: Episode {ep_idx} length mismatch - agent: {len(agent_ep)}, global: {len(global_ep)}")
                mismatch_episodes += 1
                continue
            
            for step_idx, (agent_step, global_step) in enumerate(zip(agent_ep, global_ep)):
                if global_step['global_image'] is not None:
                    agent_step['images']['secondary'] = global_step['global_image']
                    merged_count += 1
        
        if self._rank == 0:
            print(f"Merged {merged_count} global images into agent data")
            if mismatch_episodes > 0:
                print(f"Warning: {mismatch_episodes} episodes had length mismatches")
            
            # Verify merge
            if self.episodes and self.episodes[0]:
                available_views = list(self.episodes[0][0]['images'].keys())
                print(f"Available camera views after merge: {available_views}")

    def _create_split(self):
        """Create train/val split indices."""
        all_indices = []
        for ep_idx, episode in enumerate(self.episodes):
            for step_idx in range(len(episode)):
                all_indices.append((ep_idx, step_idx))
        
        rng = np.random.RandomState(self.seed)
        rng.shuffle(all_indices)
        
        n_val = int(len(all_indices) * self.val_split)
        if self.train:
            return all_indices[n_val:]
        else:
            return all_indices[:n_val]
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample with multi-view support.
        
        Returns:
            Dictionary containing:
                - 'image': Dict[str, Tensor] for multi-view or Tensor for single view
                - 'proprio': Proprioceptive state tensor
                - 'action': Action tensor
                - 'instruction': Language instruction string
        """
        ep_idx, step_idx = self.indices[idx]
        step = self.episodes[ep_idx][step_idx]
        
        # Process images
        if self.use_multi_view:
            # Multi-view: return dict of processed images
            processed_images = {}
            for view in self.image_views:
                if view in step['images']:
                    processed_images[view] = torch.from_numpy(
                        self._process_image(step['images'][view])
                    )
                elif 'image' in step['images']:
                    # Fallback to single image for missing views
                    processed_images[view] = torch.from_numpy(
                        self._process_image(step['images']['image'])
                    )
                else:
                    # Create placeholder if no image available
                    processed_images[view] = torch.zeros(3, self.image_size[0], self.image_size[1])
            image_output = processed_images
        else:
            # Single image mode (backward compatible)
            if 'image' in step['images']:
                img = step['images']['image']
            elif 'primary' in step['images']:
                img = step['images']['primary']
            elif step['images']:
                img = list(step['images'].values())[0]
            else:
                # Placeholder if no image
                img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            image_output = torch.from_numpy(self._process_image(img))
        
        # Process proprio
        proprio = step['proprio'].astype(np.float32)
        if 'proprio' in self.statistics:
            proprio = self._normalize(
                proprio,
                self.statistics['proprio']['mean'],
                self.statistics['proprio']['std']
            )
        
        # Process action - DO NOT normalize here, let the model handle tokenization
        # FIX: Actions should be passed RAW to the model, which will handle
        # normalization internally during tokenization (to avoid double normalization)
        action = step['action'].astype(np.float32)
        # NOTE: Normalization is now done in the model's _tokenize_actions()
        # using action_min and action_max derived from dataset statistics
        
        return {
            'image': image_output,  # Dict[str, Tensor] or Tensor
            'proprio': torch.from_numpy(proprio),
            'action': torch.from_numpy(action),
            'instruction': step['instruction'],
        }
    
    def _decode_image(self, image_data) -> np.ndarray:
        """
        Decode image from compressed bytes or return as-is if already numpy array.
        
        Args:
            image_data: Either compressed bytes (PNG) or numpy array
            
        Returns:
            Decoded numpy array (H, W, C)
        """
        if isinstance(image_data, bytes):
            # Decode compressed PNG bytes
            return tf.io.decode_png(image_data).numpy()
        elif isinstance(image_data, np.ndarray):
            # Already decoded (legacy cache)
            return image_data
        else:
            raise ValueError(f"Unknown image data type: {type(image_data)}")
    
    def _process_image(self, image_data) -> np.ndarray:
        """
        Process image: decode if needed, resize, crop, normalize.
        
        Args:
            image_data: Input image - either compressed bytes or numpy array (H, W, C) in [0, 255]
            
        Returns:
            Processed image (C, H, W) in [0, 1]
        """
        # Decode if compressed bytes
        image = self._decode_image(image_data)
        
        # Convert to PIL Image
        image_pil = Image.fromarray(image)
        
        # Apply augmentation (random crop) if enabled
        if self.augment:
            w, h = image_pil.size
            crop_w = int(w * self.augment_crop_ratio)
            crop_h = int(h * self.augment_crop_ratio)
            left = np.random.randint(0, w - crop_w + 1)
            top = np.random.randint(0, h - crop_h + 1)
            image_pil = image_pil.crop((left, top, left + crop_w, top + crop_h))
        else:
            # Center crop for validation
            w, h = image_pil.size
            crop_w = int(w * self.augment_crop_ratio)
            crop_h = int(h * self.augment_crop_ratio)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            image_pil = image_pil.crop((left, top, left + crop_w, top + crop_h))
        
        # Resize to target size
        image_pil = image_pil.resize(self.image_size, Image.BILINEAR)
        
        # Convert to numpy array
        image_np = np.array(image_pil).astype(np.float32) / 255.0
        
        # Convert to CHW format
        image_np = np.transpose(image_np, (2, 0, 1))
        
        return image_np
    
    def _normalize(self, data: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalize data using mean and std."""
        return (data - mean) / (std + eps)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.statistics
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset info."""
        return self.info


def collate_fn(batch):
    """
    Collate function for DataLoader with multi-view support.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched dictionary with images as dict or tensor
    """
    first_image = batch[0]['image']
    
    if isinstance(first_image, dict):
        # Multi-view: stack each view separately
        images = {}
        for view in first_image.keys():
            images[view] = torch.stack([item['image'][view] for item in batch])
    else:
        # Single image: backward compatible
        images = torch.stack([item['image'] for item in batch])
    
    proprios = torch.stack([item['proprio'] for item in batch])
    actions = torch.stack([item['action'] for item in batch])
    instructions = [item['instruction'] for item in batch]
    
    return {
        'image': images,
        'proprio': proprios,
        'action': actions,
        'instruction': instructions,
    }


if __name__ == "__main__":
    # Test dataset loading with multi-view
    print("Testing multi-view dataset loading...")
    
    dataset = RobotRLDSDataset(
        data_dir="data/rlds_data/LiftBarrier-rf_Agent0",
        train=True,
        augment=True,
        use_multi_view=True,
        image_views=['primary', 'secondary', 'wrist'],
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset info: {dataset.get_info()}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    
    if isinstance(sample['image'], dict):
        print("Multi-view images:")
        for view, img in sample['image'].items():
            print(f"  {view}: shape={img.shape}")
    else:
        print(f"Single image shape: {sample['image'].shape}")
    
    print(f"Proprio shape: {sample['proprio'].shape}")
    print(f"Action shape: {sample['action'].shape}")
    print(f"Instruction: {sample['instruction']}")
