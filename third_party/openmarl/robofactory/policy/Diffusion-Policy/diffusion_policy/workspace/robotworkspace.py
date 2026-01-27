"""
Diffusion Policy training workspace with DDP and enhanced logging support.

This module implements the training loop for Diffusion Policy models using PyTorch DDP
for distributed training across multiple GPUs, with comprehensive wandb logging
aligned with OpenVLA patterns.

Now inherits from BaseVLAWorkspace for consistent behavior across all policies
while maintaining full backward compatibility with existing Diffusion-Policy
training scripts and checkpoints.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
import pathlib
import copy
import random
import dill
import threading
import tqdm
import numpy as np
from typing import Dict, Any, Tuple
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
import wandb

# Import base workspace and shared utilities
from robofactory.policy.core import (
    BaseVLAWorkspace,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def is_distributed():
    """Check if running in distributed mode - wrapper for backward compatibility."""
    return dist.is_available() and dist.is_initialized()


# ============================================================================
# Distributed-aware BatchSampler
# ============================================================================

class DistributedBatchSampler:
    """Batch sampler that supports distributed training."""
    
    def __init__(self, data_size: int, batch_size: int, shuffle: bool = False, 
                 seed: int = 0, drop_last: bool = True, rank: int = 0, world_size: int = 1):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        
        # Calculate batches per process
        self.total_batches = data_size // batch_size
        self.batches_per_rank = self.total_batches // world_size
        self.discard = data_size - batch_size * self.total_batches
        
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling (important for distributed training)."""
        self.epoch = epoch
        
    def __iter__(self):
        # Use epoch-dependent seed for shuffling
        rng = np.random.default_rng(self.seed + self.epoch)
        
        if self.shuffle:
            perm = rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
            
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.total_batches, self.batch_size)
        
        # Each rank gets a subset of batches
        start_batch = self.rank * self.batches_per_rank
        end_batch = start_batch + self.batches_per_rank
        
        for i in range(start_batch, end_batch):
            yield perm[i]
            
    def __len__(self):
        return self.batches_per_rank


class BatchSampler:
    """Original batch sampler for single-GPU training."""
    
    def __init__(self, data_size: int, batch_size: int, shuffle: bool = False, 
                 seed: int = 0, drop_last: bool = True):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.rng = np.random.default_rng(seed) if shuffle else None

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling (for API compatibility with DistributedBatchSampler)."""
        self.epoch = epoch
        # Update RNG with epoch-dependent seed for better shuffling each epoch
        if self.shuffle:
            self.rng = np.random.default_rng(self.seed + epoch)

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


# ============================================================================
# RobotWorkspace with DDP Support and Enhanced Logging
# ============================================================================

class RobotWorkspace(BaseVLAWorkspace):
    """
    Training workspace for Diffusion Policy models.
    
    Inherits from BaseVLAWorkspace and implements Diffusion-Policy-specific:
    - Model initialization via Hydra
    - EMA model support
    - U-Net based diffusion policy
    - Custom batch samplers
    - Simulation evaluation
    
    Features:
    - Multi-GPU training with DDP
    - Comprehensive wandb logging (aligned with OpenVLA/Pi0)
    - Gradient norm tracking
    - Loss statistics
    - Visual logging (observations, action predictions)
    - Evaluation video logging
    - Best model checkpointing
    """
    
    include_keys = ['global_step', 'epoch', 'wandb_run_id']
    exclude_keys = tuple()  # For checkpoint compatibility
    
    def __init__(self, cfg: OmegaConf, output_dir=None):
        # Note: Don't call super().__init__() as we override most functionality
        # but keep the interface compatible
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # Setup distributed training if available
        self.distributed, self.local_rank, _ = setup_distributed()
        self.rank = get_rank()
        self.world_size = get_world_size()
        
        # Set instance variables for BaseVLAWorkspace compatibility
        self.device = torch.device(f'cuda:{self.local_rank}') if self.distributed else torch.device(cfg.training.device)
        self._is_main_process = is_main_process()
        
        # Set seed (different seed per rank for data augmentation diversity)
        seed = cfg.training.seed + self.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.wandb_run_id = None  # Will be set after wandb.init() and saved for resumption
        
        # Best loss tracking (aligned with OpenVLA)
        self.best_loss = float('inf')
        
        # Step-level loss tracking for statistics
        self.step_losses = []
        
        # Evaluation environment (lazy init)
        self.eval_env = None
        self.eval_env_id = None

        # Deterministic checkpoint directory based on zarr dataset name
        # Use absolute path - parents[4] is robofactory/
        project_root = pathlib.Path(__file__).resolve().parents[4]
        zarr_stem = pathlib.Path(cfg.task.dataset.zarr_path).stem  # e.g., "PassShoe-rf_Agent0_160"
        self.checkpoint_base_dir = project_root / "checkpoints" / f"{cfg.exp_name}" / zarr_stem
    
    # =========================================================================
    # Property for output_dir (from original BaseWorkspace)
    # =========================================================================
    
    @property
    def output_dir(self):
        from hydra.core.hydra_config import HydraConfig
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    # =========================================================================
    # Abstract method implementations for BaseVLAWorkspace interface
    # =========================================================================
    
    def _get_checkpoint_dir(self) -> pathlib.Path:
        """Get checkpoint directory for Diffusion Policy."""
        return self.checkpoint_base_dir
    
    def _init_model(self) -> nn.Module:
        """Initialize Diffusion Policy model via Hydra."""
        # Model is already initialized in __init__ via Hydra
        return self.model
    
    def _init_dataset(self) -> Tuple[Dataset, Dataset]:
        """Initialize datasets via Hydra."""
        cfg = self.cfg
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        val_dataset = dataset.get_validation_dataset()
        return dataset, val_dataset
    
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute diffusion policy loss."""
        # The model's compute_loss is called in the training loop
        raw_loss = self.model.compute_loss(batch)
        loss = raw_loss.mean()
        return loss
    
    # =========================================================================
    # Checkpoint save/load (from original BaseWorkspace for compatibility)
    # =========================================================================
    
    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None, include_keys=None, use_thread=True):
        """Save checkpoint with full training state."""
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, include_keys=None, **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, exclude_keys=None, include_keys=None, **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys, **kwargs)
        return instance

    # =========================================================================
    # Checkpoint discovery methods
    # =========================================================================

    def get_latest_checkpoint_path(self):
        """Find the checkpoint with the highest epoch number in the deterministic checkpoint dir."""
        ckpt_dir = self.checkpoint_base_dir
        
        if not ckpt_dir.exists():
            return None
        
        # Find all numbered checkpoint files (supports both epoch_{N}.ckpt and {N}.ckpt formats)
        max_epoch = -1
        best_ckpt = None
        
        for ckpt_file in ckpt_dir.glob('*.ckpt'):
            name = ckpt_file.stem  # e.g., "epoch_100", "100", "best"
            try:
                # Try new format: epoch_{N}
                if name.startswith('epoch_'):
                    epoch = int(name.split('_')[1])
                else:
                    # Legacy format: {N}
                    epoch = int(name)
                if epoch > max_epoch:
                    max_epoch = epoch
                    best_ckpt = ckpt_file
            except (ValueError, IndexError):
                # Not a numbered checkpoint (e.g., "best"), skip
                pass
        
        return best_ckpt

    def _compute_grad_norm(self) -> float:
        """Compute the total gradient norm (aligned with OpenVLA)."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _log_sample_images(self, batch, step):
        """Log sample observation images to wandb."""
        try:
            obs = batch['obs']
            if 'head_cam' in obs:
                images = obs['head_cam']  # (B, T, C, H, W)
                # Take first sample, last timestep
                img = images[0, -1].cpu().numpy()
                if img.shape[0] == 3:  # CHW -> HWC
                    img = np.transpose(img, (1, 2, 0))
                # Denormalize if needed (ImageNet normalization)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                wandb.log({
                    "samples/input_observation": wandb.Image(img, caption=f"Step {step}")
                }, step=step)
        except Exception as e:
            pass  # Silent fail for image logging

    def _log_action_predictions(self, gt_action, pred_action, step):
        """Log action prediction comparison as a plot."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from io import BytesIO
            from PIL import Image as PILImage
            
            gt = gt_action[0].cpu().numpy()  # First sample
            pred = pred_action[0].cpu().numpy()
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 6))
            
            # Plot each action dimension
            time_steps = np.arange(gt.shape[0])
            
            # Ground truth
            axes[0].set_title('Ground Truth Actions')
            for i in range(min(gt.shape[1], 8)):
                axes[0].plot(time_steps, gt[:, i], label=f'dim{i}', alpha=0.7)
            axes[0].legend(loc='upper right', fontsize=8)
            axes[0].set_ylabel('Action Value')
            axes[0].grid(True, alpha=0.3)
            
            # Predicted
            axes[1].set_title('Predicted Actions')
            for i in range(min(pred.shape[1], 8)):
                axes[1].plot(time_steps, pred[:, i], label=f'dim{i}', alpha=0.7)
            axes[1].legend(loc='upper right', fontsize=8)
            axes[1].set_xlabel('Time Step')
            axes[1].set_ylabel('Action Value')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to wandb image
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = PILImage.open(buf)
            
            wandb.log({
                "samples/action_prediction": wandb.Image(img, caption=f"Step {step}")
            }, step=step)
            
            plt.close(fig)
        except Exception as e:
            pass  # Silent fail

    def _init_eval_env(self):
        """Lazy initialization of evaluation environment with GPU backend for parallel envs."""
        if self.eval_env is not None:
            return True
        
        cfg = self.cfg
        
        # Setup headless rendering
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["DISPLAY"] = ""
        
        try:
            import gymnasium as gym
            import robofactory.tasks
        except (ImportError, FileNotFoundError, OSError) as e:
            print(f"Warning: Could not import simulation dependencies: {e}")
            return False
        
        # Determine environment ID from task
        env_id = cfg.task.name if hasattr(cfg.task, 'name') else None
        if env_id and not env_id.endswith('-rf'):
            env_id += '-rf'
        
        if env_id is None:
            print("Warning: Could not determine environment ID")
            return False
        
        # Get parallel env count from config
        num_parallel = getattr(cfg.training, 'eval_sim_num_envs', 4)
        
        # Get expected observation shapes from config
        self.eval_n_obs_steps = getattr(cfg, 'n_obs_steps', 3)
        self.eval_img_height = 240
        self.eval_img_width = 320
        self.eval_state_dim = 8
        
        # Try GPU backend first (required for parallel envs in ManiSkill)
        try:
            env_kwargs = dict(
                obs_mode='rgbd',  # Get RGB images and depth (includes camera observations)
                control_mode='pd_joint_pos',
                render_mode='rgb_array',
                num_envs=num_parallel,
                sim_backend='gpu',  # GPU backend for parallel environments
            )
            
            self.eval_env = gym.make(env_id, **env_kwargs)
            self.eval_env_id = env_id
            self.eval_num_parallel = num_parallel
            
            # Initialize observation history buffers for each parallel env
            self.obs_history = {
                'images': [[] for _ in range(num_parallel)],
                'states': [[] for _ in range(num_parallel)],
            }
            
            print(f"Initialized eval env: {env_id} with {num_parallel} parallel envs (GPU)")
            print(f"  Observation format: {self.eval_n_obs_steps} timesteps, {self.eval_img_height}x{self.eval_img_width} images")
            return True
            
        except Exception as e:
            print(f"GPU parallel env failed ({e}), falling back to single CPU env...")
            
            # Fallback to single CPU env
            try:
                env_kwargs = dict(
                    obs_mode='rgbd',  # Get RGB images and depth
                    control_mode='pd_joint_pos',
                    render_mode='rgb_array',
                    num_envs=1,
                    sim_backend='cpu',
                )
                
                self.eval_env = gym.make(env_id, **env_kwargs)
                self.eval_env_id = env_id
                self.eval_num_parallel = 1
                
                # Initialize observation history buffers
                self.obs_history = {
                    'images': [[]],
                    'states': [[]],
                }
                
                print(f"Initialized eval env: {env_id} with 1 env (CPU fallback)")
                print(f"  Observation format: {self.eval_n_obs_steps} timesteps, {self.eval_img_height}x{self.eval_img_width} images")
                return True
                
            except Exception as e2:
                print(f"Warning: Could not create eval environment: {e2}")
                import traceback
                traceback.print_exc()
                return False

    @torch.no_grad()
    def _evaluate_in_simulation(self, policy) -> Dict[str, Any]:
        """
        Run model evaluation in simulation and log videos to wandb.
        
        Properly handles observation format matching:
        - Maintains observation history buffer (n_obs_steps=3)
        - Resizes images to training dimensions (240x320)
        - Normalizes images to [0, 1]
        - Extracts proper agent state
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not is_main_process():
            return {}
        
        cfg = self.cfg
        
        # Check if eval is enabled
        if not getattr(cfg.training, 'eval_in_sim', False):
            return {}
        
        # Lazy initialize environment
        if not self._init_eval_env():
            print("Skipping simulation evaluation - environment not available")
            return {}
        
        env = self.eval_env
        num_parallel = self.eval_num_parallel
        n_obs_steps = self.eval_n_obs_steps
        
        # Get evaluation settings
        num_episodes = getattr(cfg.training, 'eval_sim_episodes', 4)
        max_steps = getattr(cfg.training, 'eval_sim_max_steps', 200)
        log_video = getattr(cfg.logging, 'log_eval_video', True)
        
        print(f"\nRunning simulation evaluation: {num_episodes} episodes with {num_parallel} parallel envs")
        
        try:
            policy.eval()
            device = next(policy.parameters()).device
            
            all_successes = []
            all_rewards = []
            all_frames = []
            
            num_batches = (num_episodes + num_parallel - 1) // num_parallel
            
            for batch_idx in range(num_batches):
                obs, info = env.reset()
                episode_rewards = np.zeros(num_parallel)
                batch_frames = []
                
                # Reset observation history for new episodes
                self.obs_history = {
                    'images': [[] for _ in range(num_parallel)],
                    'states': [[] for _ in range(num_parallel)],
                }
                
                for step in range(max_steps):
                    # Update observation history and get formatted observations
                    obs_dict = self._update_obs_history_and_prepare(obs, num_parallel, device)
                    if obs_dict is None:
                        print(f"Warning: Failed to prepare observations at step {step}")
                        break
                    
                    # Predict actions for all parallel environments
                    actions = []
                    for i in range(num_parallel):
                        # Extract single env observation
                        single_obs = {
                            'head_cam': obs_dict['head_cam'][i:i+1],  # (1, T, C, H, W)
                            'agent_pos': obs_dict['agent_pos'][i:i+1],  # (1, T, D)
                        }
                        result = policy.predict_action(single_obs)
                        # Get first action from predicted sequence
                        action = result['action'][0, 0].cpu().numpy()  # (action_dim,)
                        actions.append(action)
                    
                    # Format actions for environment
                    action_batch = self._format_actions_for_env(actions, env, num_parallel)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action_batch)
                    
                    # Accumulate rewards
                    reward_np = self._to_numpy(reward)
                    if np.isscalar(reward_np):
                        episode_rewards[0] += reward_np
                    else:
                        episode_rewards += np.array(reward_np).flatten()[:num_parallel]
                    
                    # Capture frames for video (last batch only, every 2nd frame)
                    if log_video and batch_idx == num_batches - 1 and step % 2 == 0:
                        try:
                            frame = env.render()
                            if frame is not None:
                                if isinstance(frame, np.ndarray) and frame.ndim == 4:
                                    frame = frame[0]
                                elif isinstance(frame, torch.Tensor):
                                    frame = frame[0].cpu().numpy() if frame.ndim == 4 else frame.cpu().numpy()
                                batch_frames.append(frame)
                        except:
                            pass
                    
                    # Check if done
                    terminated_np = self._to_numpy(terminated)
                    truncated_np = self._to_numpy(truncated)
                    done = np.logical_or(terminated_np, truncated_np)
                    if np.isscalar(done):
                        if done:
                            break
                    elif np.all(done):
                        break
                
                # Collect success info
                if isinstance(info, dict):
                    success = info.get('success', False)
                    success_np = self._to_numpy(success) if not isinstance(success, bool) else success
                    if np.isscalar(success_np):
                        all_successes.append(success_np)
                    else:
                        all_successes.extend(np.array(success_np).flatten()[:num_parallel].tolist())
                
                all_rewards.extend(episode_rewards[:num_parallel].tolist())
                
                if batch_idx == num_batches - 1:
                    all_frames = batch_frames
            
            # Compute metrics
            successes = all_successes[:num_episodes]
            rewards = all_rewards[:num_episodes]
            
            success_rate = sum(successes) / len(successes) if successes else 0.0
            avg_reward = np.mean(rewards) if rewards else 0.0
            
            metrics = {
                'eval/sim_success_rate': float(success_rate),
                'eval/sim_avg_reward': float(avg_reward),
                'eval/sim_episodes': num_episodes,
            }
            
            print(f"  Success rate: {success_rate:.2%}, Avg reward: {avg_reward:.4f}")
            
            # Log to wandb
            if cfg.logging.mode == "online":
                wandb.log(metrics, step=self.global_step)
                
                # Log video
                if log_video and all_frames and len(all_frames) > 0:
                    try:
                        video_array = np.stack(all_frames, axis=0)
                        # Ensure correct format: (T, C, H, W)
                        if video_array.ndim == 4 and video_array.shape[-1] == 3:
                            video_array = np.transpose(video_array, (0, 3, 1, 2))
                        wandb.log({
                            "eval/sim_video": wandb.Video(video_array, fps=10, format="mp4")
                        }, step=self.global_step)
                        print(f"  Logged evaluation video ({len(all_frames)} frames)")
                    except Exception as e:
                        print(f"Warning: Could not log video: {e}")
            
            policy.train()
            return metrics
            
        except Exception as e:
            print(f"Warning: Simulation evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            policy.train()
            return {}
    
    def _update_obs_history_and_prepare(self, obs, num_parallel, device):
        """
        Update observation history with new observation and prepare formatted obs dict.
        
        Maintains a sliding window of n_obs_steps observations for each parallel env.
        
        Args:
            obs: Raw observation from environment
            num_parallel: Number of parallel environments
            device: Target device for tensors
            
        Returns:
            Dict with 'head_cam' (B, T, C, H, W) and 'agent_pos' (B, T, D)
        """
        try:
            import cv2
            
            n_obs_steps = self.eval_n_obs_steps
            target_h, target_w = self.eval_img_height, self.eval_img_width
            
            # Extract images and states from observation
            for i in range(num_parallel):
                # Get image for this env
                img = self._extract_single_image(obs, i)
                if img is None:
                    return None
                
                # Resize to training dimensions
                if img.shape[0] != target_h or img.shape[1] != target_w:
                    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                
                # Normalize to [0, 1] and convert to CHW
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                
                # Get state for this env
                state = self._extract_single_state(obs, i)
                if state is None:
                    state = np.zeros(self.eval_state_dim, dtype=np.float32)
                
                # Add to history
                self.obs_history['images'][i].append(img)
                self.obs_history['states'][i].append(state)
                
                # Keep only last n_obs_steps
                if len(self.obs_history['images'][i]) > n_obs_steps:
                    self.obs_history['images'][i] = self.obs_history['images'][i][-n_obs_steps:]
                    self.obs_history['states'][i] = self.obs_history['states'][i][-n_obs_steps:]
            
            # Prepare observation dict with proper padding
            batch_images = []
            batch_states = []
            
            for i in range(num_parallel):
                imgs = self.obs_history['images'][i]
                states = self.obs_history['states'][i]
                
                # Pad with first observation if not enough history
                while len(imgs) < n_obs_steps:
                    imgs = [imgs[0]] + imgs
                    states = [states[0]] + states
                
                # Stack into arrays: (T, C, H, W) and (T, D)
                img_stack = np.stack(imgs, axis=0)
                state_stack = np.stack(states, axis=0)
                
                batch_images.append(img_stack)
                batch_states.append(state_stack)
            
            # Convert to tensors: (B, T, C, H, W) and (B, T, D)
            head_cam = torch.from_numpy(np.stack(batch_images, axis=0)).float().to(device)
            agent_pos = torch.from_numpy(np.stack(batch_states, axis=0)).float().to(device)
            
            return {
                'head_cam': head_cam,
                'agent_pos': agent_pos,
            }
            
        except Exception as e:
            print(f"Warning: Failed to prepare observations: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_single_image(self, obs, env_idx):
        """Extract image for a single environment from batched observation."""
        try:
            image = None
            
            if isinstance(obs, dict):
                # Try different observation keys for ManiSkill
                if 'sensor_data' in obs:
                    sensor_data = obs['sensor_data']
                    # Find first camera with rgb - try known camera names first
                    for cam_name in ['head_camera_agent0', 'head_camera', 'base_camera', 'hand_camera']:
                        if cam_name in sensor_data:
                            cam_data = sensor_data[cam_name]
                            if isinstance(cam_data, dict) and 'rgb' in cam_data:
                                image = cam_data['rgb']
                                break
                    
                    # Fallback: find any camera with rgb
                    if image is None:
                        for cam_key in sensor_data.keys():
                            cam_data = sensor_data[cam_key]
                            if isinstance(cam_data, dict) and 'rgb' in cam_data:
                                image = cam_data['rgb']
                                break
                            elif isinstance(cam_data, (np.ndarray, torch.Tensor)):
                                if hasattr(cam_data, 'shape') and len(cam_data.shape) >= 3:
                                    image = cam_data
                                    break
                
                # Try head_camera directly
                if image is None and 'head_camera' in obs:
                    cam_data = obs['head_camera']
                    if isinstance(cam_data, dict) and 'rgb' in cam_data:
                        image = cam_data['rgb']
                    else:
                        image = cam_data
                
                # Direct rgb/image keys
                if image is None and 'rgb' in obs:
                    image = obs['rgb']
                if image is None and 'image' in obs:
                    image = obs['image']
            
            if image is None:
                return None
            
            # Handle tensor
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            # Extract single env from batch
            if image.ndim == 4:  # (B, H, W, C) or (B, C, H, W)
                image = image[env_idx]
            
            # Ensure HWC format
            if image.ndim == 3:
                if image.shape[0] in [3, 4]:  # CHW
                    image = np.transpose(image, (1, 2, 0))
                if image.shape[-1] == 4:  # RGBA -> RGB
                    image = image[..., :3]
            
            # Ensure uint8 for resize
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            return image
            
        except Exception as e:
            return None
    
    def _extract_single_state(self, obs, env_idx):
        """Extract agent state for a single environment from batched observation."""
        try:
            state = None
            
            if isinstance(obs, dict):
                # Try to find agent state/proprio
                if 'agent' in obs:
                    agent_obs = obs['agent']
                    if isinstance(agent_obs, dict):
                        # Prefer qpos for joint positions
                        if 'qpos' in agent_obs:
                            state = agent_obs['qpos']
                        elif 'joint_pos' in agent_obs:
                            state = agent_obs['joint_pos']
                        elif len(agent_obs) > 0:
                            # Take first available state
                            first_key = list(agent_obs.keys())[0]
                            state = agent_obs[first_key]
                    else:
                        state = agent_obs
                elif 'extra' in obs and 'agent' in obs['extra']:
                    state = obs['extra']['agent']
                elif 'state' in obs:
                    state = obs['state']
                elif 'proprio' in obs:
                    state = obs['proprio']
            
            if state is None:
                # Return zeros if we can't extract state - it's less critical than images
                return np.zeros(self.eval_state_dim, dtype=np.float32)
            
            # Handle tensor
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            
            # Extract single env from batch
            if state.ndim == 2:  # (B, D)
                state = state[env_idx]
            
            # Ensure correct dimension (take first 8 dims if longer)
            state = state.flatten()[:self.eval_state_dim].astype(np.float32)
            
            # Pad if too short
            if len(state) < self.eval_state_dim:
                state = np.pad(state, (0, self.eval_state_dim - len(state)))
            
            return state
            
        except Exception as e:
            return None

    def _format_actions_for_env(self, actions, env, num_parallel):
        """Format actions for the environment."""
        try:
            env_unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
            is_multi_agent = hasattr(env_unwrapped, 'agent') and hasattr(env_unwrapped.agent, 'agents')
            
            if is_multi_agent:
                num_agents = len(env_unwrapped.agent.agents)
                if num_parallel > 1:
                    action_batch = {}
                    stacked_actions = np.stack(actions, axis=0)
                    for agent_idx in range(num_agents):
                        agent_uid = f'panda-{agent_idx}'
                        if agent_idx == 0:
                            action_batch[agent_uid] = stacked_actions
                        else:
                            action_batch[agent_uid] = np.zeros_like(stacked_actions)
                else:
                    action_batch = {}
                    for agent_idx in range(num_agents):
                        agent_uid = f'panda-{agent_idx}'
                        if agent_idx == 0:
                            action_batch[agent_uid] = actions[0]
                        else:
                            action_batch[agent_uid] = np.zeros_like(actions[0])
            else:
                if num_parallel > 1:
                    action_batch = np.stack(actions, axis=0)
                else:
                    action_batch = actions[0]
            
            return action_batch
        except Exception as e:
            print(f"Warning: Could not format actions: {e}")
            return actions[0] if len(actions) == 1 else np.stack(actions, axis=0)

    def _to_numpy(self, x):
        """Convert tensor to numpy."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x) if not np.isscalar(x) else x

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # Print config at start (aligned with Pi0/OpenVLA - using OmegaConf)
        if is_main_process():
            print("="*80)
            print("Initializing Diffusion Policy Training")
            print("="*80)
            print(OmegaConf.to_yaml(cfg))
            print("="*80)
        
        # Resume training - automatically find highest numbered checkpoint
        if cfg.training.resume:
            latest_ckpt_path = self.get_latest_checkpoint_path()
            if latest_ckpt_path is not None and latest_ckpt_path.is_file():
                if is_main_process():
                    print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)
                
                # Use checkpoint filename to determine completed epochs
                # Supports both epoch_{N}.ckpt and {N}.ckpt formats
                try:
                    name = latest_ckpt_path.stem
                    if name.startswith('epoch_'):
                        completed_epochs = int(name.split('_')[1])
                    else:
                        completed_epochs = int(name)
                    if self.epoch < completed_epochs:
                        if is_main_process():
                            print(f"Adjusting epoch from {self.epoch} to {completed_epochs} (from checkpoint name)")
                        self.epoch = completed_epochs
                except (ValueError, IndexError):
                    pass  # Not a numbered checkpoint (e.g., "best.ckpt")
            else:
                if is_main_process():
                    print(f"No checkpoint found in {self.checkpoint_base_dir}, starting fresh")

        # Configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        
        # Create dataloader with distributed support
        train_dataloader, train_sampler = self._create_dataloader(
            dataset, 
            batch_size=cfg.dataloader.batch_size,
            shuffle=cfg.dataloader.shuffle,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=cfg.dataloader.persistent_workers,
            seed=cfg.training.seed
        )
        normalizer = dataset.get_normalizer()

        # Configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader, _ = self._create_dataloader(
            val_dataset,
            batch_size=cfg.val_dataloader.batch_size,
            shuffle=False,
            num_workers=cfg.val_dataloader.num_workers,
            pin_memory=cfg.val_dataloader.pin_memory,
            persistent_workers=cfg.val_dataloader.persistent_workers,
            seed=cfg.training.seed
        )

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        # Configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        env_runner = None

        # Check if training should be skipped (already complete)
        training_complete = self.epoch >= cfg.training.num_epochs
        if training_complete and is_main_process():
            print(f"Training already complete ({self.epoch}/{cfg.training.num_epochs} epochs). Skipping.")
        
        # Configure logging (only on main process AND if training will run)
        if is_main_process() and not training_complete:
            # Determine run ID: use saved ID from checkpoint if available, else from config
            run_id = self.wandb_run_id or (cfg.logging.id if hasattr(cfg.logging, 'id') else None)
            
            # Filter out custom logging options that wandb.init() doesn't accept
            wandb_kwargs = {
                'project': cfg.logging.project,
                'name': cfg.logging.name,
                'mode': cfg.logging.mode,
                'resume': 'allow' if run_id else False,  # Only resume if we have a run ID
                'tags': list(cfg.logging.tags) if hasattr(cfg.logging, 'tags') and cfg.logging.tags else None,
                'id': run_id,
                'group': cfg.logging.group if hasattr(cfg.logging, 'group') else None,
            }
            # Remove None values
            wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
            
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **wandb_kwargs
            )
            
            # Save the run ID for future resumption (will be saved in checkpoint)
            self.wandb_run_id = wandb_run.id
            if run_id:
                print(f"Resumed wandb run: {wandb_run.id}")
            else:
                print(f"Started new wandb run: {wandb_run.id}")
            
            # Log model architecture info (aligned with OpenVLA)
            wandb.config.update({
                "output_dir": self.output_dir,
                "distributed": self.distributed,
                "world_size": self.world_size,
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }, allow_val_change=True)

        # Configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # Device transfer
        if self.distributed:
            device = torch.device(f'cuda:{self.local_rank}')
        else:
            device = torch.device(cfg.training.device)
            
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Wrap model with DDP if distributed
        model_for_training = self.model
        if self.distributed:
            model_for_training = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            if is_main_process():
                print(f"Using DistributedDataParallel with {self.world_size} GPUs")

        # Save batch for sampling
        train_sampling_batch = None

        # Get logging config
        log_every = getattr(cfg.logging, 'log_every_n_steps', 50)
        log_gradients = getattr(cfg.logging, 'log_gradients', True)
        log_lr = getattr(cfg.logging, 'log_learning_rate', True)
        log_images = getattr(cfg.logging, 'log_images', True)
        log_action_pred = getattr(cfg.logging, 'log_action_predictions', True)
        max_grad_norm = getattr(cfg.training, 'max_grad_norm', 1.0)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # Training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        json_logger = JsonLogger(log_path) if is_main_process() else None
        
        final_train_loss = 0.0
        
        try:
            if json_logger:
                json_logger.__enter__()
            
            # Skip training if already complete
            if training_complete:
                return
                
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
                step_log = dict()
                
                # Reset step losses for epoch
                self.step_losses = []
                
                # Set epoch for distributed sampler
                if train_sampler is not None:
                    train_sampler.set_epoch(self.epoch)
                
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    if self.distributed:
                        model_for_training.module.obs_encoder.eval()
                        model_for_training.module.obs_encoder.requires_grad_(False)
                    else:
                        self.model.obs_encoder.eval()
                        self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                
                # Only show progress bar on main process
                dataloader_iter = train_dataloader
                if is_main_process():
                    dataloader_iter = tqdm.tqdm(
                        train_dataloader, 
                        desc=f"Training epoch {self.epoch}", 
                        leave=False, 
                        mininterval=cfg.training.tqdm_interval_sec
                    )
                
                for batch_idx, batch in enumerate(dataloader_iter):
                    batch = dataset.postprocess(batch, device)
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # Compute loss
                    raw_loss = model_for_training.compute_loss(batch) if not self.distributed else model_for_training.module.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # Step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        # Compute gradient norm before clipping
                        grad_norm = None
                        if is_main_process() and log_gradients:
                            grad_norm = self._compute_grad_norm()
                        
                        # Gradient clipping
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_grad_norm
                            )
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    
                    # Update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)

                    # Logging
                    raw_loss_cpu = raw_loss.item()
                    
                    # Reduce loss across all processes
                    if self.distributed:
                        loss_tensor = torch.tensor([raw_loss_cpu], device=device)
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                        raw_loss_cpu = loss_tensor.item() / self.world_size
                    
                    if is_main_process():
                        dataloader_iter.set_postfix(loss=raw_loss_cpu, refresh=False)
                    
                    # Track step losses
                    self.step_losses.append(raw_loss_cpu)
                    train_losses.append(raw_loss_cpu)
                    
                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        if is_main_process():
                            # Always log to JSON
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }
                            json_logger.log(step_log)
                            
                            # Log to wandb every N steps (aligned with OpenVLA)
                            if self.global_step % log_every == 0:
                                wandb_log = {
                                    'train/step_loss': raw_loss_cpu,
                                    'train/global_step': self.global_step,
                                }
                                if log_lr:
                                    wandb_log['train/learning_rate'] = lr_scheduler.get_last_lr()[0]
                                if grad_norm is not None:
                                    wandb_log['train/grad_norm'] = grad_norm
                                
                                wandb.log(wandb_log, step=self.global_step)
                                
                                # Log sample images periodically
                                if log_images and self.global_step % (log_every * 10) == 0:
                                    self._log_sample_images(batch, self.global_step)
                        
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

                # At the end of each epoch
                train_loss = np.mean(train_losses)
                final_train_loss = train_loss
                
                # Epoch metrics with statistics (aligned with OpenVLA)
                epoch_metrics = {
                    'epoch': self.epoch,
                    'train/epoch_loss': train_loss,
                    'train/epoch_loss_std': np.std(self.step_losses) if self.step_losses else 0.0,
                    'train/epoch_loss_min': min(self.step_losses) if self.step_losses else 0.0,
                    'train/epoch_loss_max': max(self.step_losses) if self.step_losses else 0.0,
                }
                
                # Track and save best loss
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    if is_main_process():
                        save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                        self.save_checkpoint(f'checkpoints/{self.cfg.exp_name}/{save_name}/best.ckpt')

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # Run validation (only on main process)
                if is_main_process() and (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            epoch_metrics['eval/val_loss'] = val_loss

                # Run diffusion sampling on a training batch (only on main process)
                if is_main_process() and (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        epoch_metrics['eval/train_action_mse'] = mse.item()
                        
                        # Log action predictions visualization
                        if log_action_pred:
                            self._log_action_predictions(gt_action, pred_action, self.global_step)
                        
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # Run simulation evaluation
                eval_every = getattr(cfg.training, 'eval_sim_every_n_epochs', 50)
                if is_main_process() and getattr(cfg.training, 'eval_in_sim', False):
                    if (self.epoch + 1) % eval_every == 0:
                        sim_metrics = self._evaluate_in_simulation(policy)
                        if sim_metrics:
                            epoch_metrics.update(sim_metrics)
                            print(f"Simulation eval: {sim_metrics}")
                
                # End of epoch logging
                if is_main_process():
                    json_logger.log(epoch_metrics)
                    wandb.log(epoch_metrics, step=self.global_step)
                
                # Increment counters FIRST
                self.global_step += 1
                self.epoch += 1

                # Checkpoint (only on main process) - now self.epoch is already incremented
                if is_main_process() and (self.epoch % cfg.training.checkpoint_every) == 0:
                    save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                    ckpt_path = f'checkpoints/{self.cfg.exp_name}/{save_name}/epoch_{self.epoch}.ckpt'
                    self.save_checkpoint(ckpt_path)
                
                # Synchronize all processes before next epoch
                if self.distributed:
                    dist.barrier()
                
                # ========= eval end for this epoch ==========
                policy.train()

            # Save final checkpoint if not already saved
            if is_main_process():
                save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                final_ckpt_path = f'checkpoints/{self.cfg.exp_name}/{save_name}/epoch_{self.epoch}.ckpt'
                # Check if the final epoch checkpoint already exists
                if not pathlib.Path(final_ckpt_path).exists():
                    self.save_checkpoint(final_ckpt_path)
                    print(f"Saved final checkpoint at epoch_{self.epoch} (step {self.global_step}) -> {final_ckpt_path}")
                
                print("="*80)
                print("Training completed!")
                print("="*80)
            
            # Final summary (aligned with OpenVLA)
            if is_main_process():
                if cfg.logging.mode == "online":
                    wandb.run.summary["final_train_loss"] = final_train_loss
                    wandb.run.summary["best_loss"] = self.best_loss
                    wandb.run.summary["total_epochs"] = self.epoch
                    wandb.finish()
                
        finally:
            if json_logger:
                json_logger.__exit__(None, None, None)
            
            # Cleanup eval environment
            if self.eval_env is not None:
                try:
                    self.eval_env.close()
                    self.eval_env = None
                except:
                    pass
            
            cleanup_distributed()

    def _create_dataloader(self, dataset, *, batch_size: int, shuffle: bool, 
                           num_workers: int, pin_memory: bool, 
                           persistent_workers: bool, seed: int = 0):
        """Create dataloader with optional distributed support."""
        
        def collate(x):
            assert len(x) == 1
            return x[0]
        
        sampler = None
        
        if self.distributed and len(dataset) > 0:
            # Use distributed batch sampler
            batch_sampler = DistributedBatchSampler(
                len(dataset), 
                batch_size, 
                shuffle=shuffle, 
                seed=seed, 
                drop_last=True,
                rank=self.rank,
                world_size=self.world_size
            )
            sampler = batch_sampler
        else:
            # Use original batch sampler for single GPU
            batch_sampler = BatchSampler(
                len(dataset), 
                batch_size, 
                shuffle=shuffle, 
                seed=seed, 
                drop_last=True
            )
            sampler = batch_sampler
        
        dataloader = DataLoader(
            dataset, 
            collate_fn=collate, 
            sampler=sampler, 
            num_workers=num_workers, 
            pin_memory=pin_memory, 
            persistent_workers=persistent_workers if num_workers > 0 else False
        )
        
        return dataloader, sampler


def create_dataloader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, 
                      pin_memory: bool, persistent_workers: bool, seed: int = 0):
    """Legacy function for backward compatibility."""
    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)
    def collate(x):
        assert len(x) == 1
        return x[0]
    dataloader = DataLoader(
        dataset, 
        collate_fn=collate, 
        sampler=batch_sampler, 
        num_workers=num_workers, 
        pin_memory=False, 
        persistent_workers=persistent_workers
    )
    return dataloader


def _copy_to_cpu(x):
    """Helper function for checkpoint saving - copy tensors to CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
