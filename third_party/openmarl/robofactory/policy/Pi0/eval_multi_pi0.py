"""
Multi-agent evaluation script for Pi0/Pi0.5 policies.

This script evaluates trained Pi0/Pi0.5 models on multi-agent RoboFactory tasks.
Follows the pattern from OpenVLA's evaluation script.
"""

import sys
sys.path.append('./')
sys.path.insert(0, './policy/Pi0')

import torch
import os
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser

import gymnasium as gym
import sapien

from robofactory.tasks import *
from mani_skill.envs.sapien_env import BaseEnv
from robofactory.utils.wrappers.record import RecordEpisodeMA
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

from pi0_policy.policy.pi0_policy import Pi0Policy
from robofactory.policy.shared import get_task_instruction
from robofactory.policy.shared.evaluation import ActionChunkingWrapper


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Evaluate Pi0/Pi0.5 multi-agent policies")
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to task config file'
    )
    parser.add_argument(
        '--policy_type',
        type=str,
        default='pi0',
        choices=['pi0', 'pi05'],
        help='Policy type: pi0 or pi05'
    )
    parser.add_argument(
        '--data_num',
        type=int,
        default=150,
        help='Number of training samples used'
    )
    parser.add_argument(
        '--checkpoint_step',
        type=int,
        default=5000,
        help='Checkpoint step number'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=10000,
        help='Random seed'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=250,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '--debug',
        type=int,
        default=0,
        help='Debug mode (0=off, 1=on)'
    )
    parser.add_argument(
        '--record_dir',
        type=str,
        default='../../eval_video/{policy_type}/{env_id}',
        help='Directory to save evaluation videos'
    )
    parser.add_argument(
        '--render_mode',
        type=str,
        default='rgb_array',
        help='Render mode'
    )
    parser.add_argument(
        '--obs_mode',
        type=str,
        default='rgb',
        help='Observation mode'
    )
    parser.add_argument(
        '--control_mode',
        type=str,
        default='pd_joint_pos',
        help='Control mode'
    )
    parser.add_argument(
        '--num_envs',
        type=int,
        default=1,
        help='Number of parallel environments'
    )
    parser.add_argument(
        '--num_eval_episodes',
        type=int,
        default=50,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device for inference'
    )
    
    return parser.parse_args()


def get_model_input(observation, agent_pos, agent_id, num_cameras=3):
    """
    Extract model input from observation.
    
    Pi0 requires 3 camera views (matching training data):
    - base_0_rgb: head_camera_agent{id} (side view)
    - left_wrist_0_rgb: head_camera_global (overhead view)
    - right_wrist_0_rgb: wrist_camera_agent{id} (gripper view, if available)
    
    Args:
        observation: Environment observation
        agent_pos: Agent joint positions
        agent_id: Agent ID
        num_cameras: Number of camera views
        
    Returns:
        Dictionary with 'images' and 'state'
    """
    images = []
    sensor_data = observation['sensor_data']
    
    # Camera mapping following training data convention:
    # [0] base_0_rgb: per-agent head camera (side view)
    head_key = f'head_camera_agent{agent_id}'
    if head_key in sensor_data:
        img = sensor_data[head_key]['rgb'].squeeze(0).cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        images.append(img)
    else:
        raise ValueError(f"Missing required camera: {head_key}")
    
    # [1] left_wrist_0_rgb: global camera (overhead view)
    global_key = 'head_camera_global'
    if global_key in sensor_data:
        img = sensor_data[global_key]['rgb'].squeeze(0).cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        images.append(img)
    else:
        # Fallback: use head camera if no global camera
        images.append(images[0].copy())
    
    # [2] right_wrist_0_rgb: wrist camera (gripper view)
    wrist_key = f'wrist_camera_agent{agent_id}'
    if wrist_key in sensor_data:
        img = sensor_data[wrist_key]['rgb'].squeeze(0).cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        images.append(img)
    else:
        # Fallback: use head camera if no wrist camera
        images.append(images[0].copy())
    
    # Stack images: [num_cameras, H, W, 3]
    images = np.array(images)
    
    return {
        'images': images,
        'state': agent_pos,
    }


def load_task_instruction(task_name: str) -> str:
    """Get language instruction for a task."""
    return get_task_instruction(task_name)


class Pi0PolicyWrapper(ActionChunkingWrapper):
    """Wrapper for Pi0 policy to match evaluation interface."""
    
    def __init__(
        self, 
        task_name: str, 
        checkpoint_step: int, 
        data_num: int, 
        agent_id: int = 0,
        policy_type: str = 'pi0',
        device: str = 'cuda:0'
    ):
        """Initialize policy wrapper."""
        # Initialize base class
        # action_repeat=2: sim_freq=20Hz, policy runs at ~10Hz → 2 sim steps per policy step
        super().__init__(
            task_name=task_name,
            device=device,
            action_horizon=50,  # Default, will be updated after loading policy
            action_repeat=2,
        )
        
        self.checkpoint_step = checkpoint_step
        self.data_num = data_num
        self.agent_id = agent_id
        self.policy_type = policy_type
        
        # Find and load checkpoint
        checkpoint_dir = self._find_checkpoint_dir(checkpoint_step)
        self.load_policy(str(checkpoint_dir))
        
        print(f"Initialized {policy_type} policy for {task_name} (Agent {agent_id})")
        print(f"Action horizon: {self.action_horizon}")
    
    def _find_checkpoint_dir(self, checkpoint_step: int) -> Path:
        """Find checkpoint directory with fallback to best/latest."""
        # Script runs from robofactory/policy/Pi0/, so go up to robofactory/checkpoints/
        # Path: ../../checkpoints/{policy_type}/{task_name}_Agent{agent_id}/epoch_{checkpoint_step}
        base_dir = Path('../../checkpoints') / self.policy_type / f'{self.task_name}_Agent{self.agent_id}'
        
        # Try specified epoch first, then fallback to best, then latest
        candidates = [
            base_dir / f'epoch_{checkpoint_step}',
            base_dir / 'best',
            base_dir / 'latest',
        ]
        
        for checkpoint_dir in candidates:
            if checkpoint_dir.exists():
                if checkpoint_dir != candidates[0]:
                    print(f"Note: epoch_{checkpoint_step} not found for Agent{self.agent_id}, using {checkpoint_dir.name}")
                return checkpoint_dir
        
        raise FileNotFoundError(f"Checkpoint not found at {base_dir} (tried epoch_{checkpoint_step}, best, latest)")
    
    def load_policy(self, checkpoint_path: str, **kwargs):
        """Load policy from checkpoint."""
        print(f"Loading Pi0 checkpoint from {checkpoint_path}")
        
        # Get config path based on policy type
        script_dir = Path(__file__).parent
        if self.policy_type == 'pi05':
            config_path = script_dir / 'pi0_policy' / 'config' / 'robot_pi05.yaml'
        else:
            config_path = script_dir / 'pi0_policy' / 'config' / 'robot_pi0.yaml'
        
        # Create policy
        self.policy = Pi0Policy(
            checkpoint_path=checkpoint_path,
            config_path=str(config_path),
            task_name=self.task_name,
            device=self.device,
        )
        
        # Update action horizon from loaded config
        self.action_horizon = self.policy.cfg.model.action_horizon
    
    def _predict_action_sequence(self, observation):
        """Predict action sequence from policy."""
        return self.policy.predict_action(observation)


def main(args):
    """Main evaluation function."""
    np.set_printoptions(suppress=True, precision=5)
    verbose = args.debug == 1
    
    # Set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Config uses task_name, construct env_id from it
    task_name = config.get('task_name', config.get('env_id', '').replace('-rf', ''))
    env_id = f"{task_name}-rf"
    
    print(f"Evaluating {args.policy_type} on {task_name}")
    print(f"Data num: {args.data_num}, Checkpoint: {args.checkpoint_step}")
    
    # Get number of agents from config (count agents list or use num_agents key)
    if 'agents' in config:
        num_agents = len(config['agents'])
    else:
        num_agents = config.get('num_agents', 1)
    
    # Create environment with CPU backend for video recording support
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack='default'),
        human_render_camera_configs=dict(shader_pack='default'),
        viewer_camera_configs=dict(shader_pack='default'),
        num_envs=args.num_envs,
        sim_backend='cpu',  # CPU backend required for RecordEpisodeMA video recording
        enable_shadow=True,
    )
    
    env = gym.make(env_id, **env_kwargs)
    
    # Wrap with video recording - use unique directory per seed to avoid file locking conflicts
    record_dir = args.record_dir.format(policy_type=args.policy_type, env_id=env_id)
    record_dir = f"{record_dir}/seed_{args.seed}"  # Unique per seed for parallel evaluation
    env = RecordEpisodeMA(
        env,
        output_dir=record_dir,
        save_trajectory=True,
        save_video=True,
        info_on_video=True,
        max_steps_per_video=args.max_steps,
    )
    
    # Initial reset to get agent poses for motion planner
    env.reset(seed=args.seed)
    
    # Get agent base poses for motion planner
    env_unwrapped = env.unwrapped
    is_multi_agent = num_agents > 1
    if is_multi_agent:
        agents_list = env_unwrapped.agent.agents
    else:
        agents_list = [env_unwrapped.agent]
    base_pose = [agent.robot.pose for agent in agents_list]
    
    # Create motion planner (for invalid action handling)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base_pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
        is_multi_agent=is_multi_agent,
    )
    
    # Load policies for each agent
    # Use env_id (with -rf suffix) since checkpoint directories use that format
    policies = []
    for agent_id in range(num_agents):
        policy = Pi0PolicyWrapper(
            task_name=env_id,  # Use env_id (e.g., "LiftBarrier-rf") for checkpoint path
            checkpoint_step=args.checkpoint_step,
            data_num=args.data_num,
            agent_id=agent_id,
            policy_type=args.policy_type,
            device=args.device,
        )
        policies.append(policy)
    
    # Evaluation loop
    success_count = 0
    episode_rewards = []
    episode_lengths = []
    
    for episode_idx in range(args.num_eval_episodes):
        print(f"\n{'='*80}")
        print(f"Episode {episode_idx + 1}/{args.num_eval_episodes}")
        print(f"{'='*80}")
        
        # Reset environment
        obs, info = env.reset(seed=args.seed + episode_idx)
        
        # Reset policies
        for policy in policies:
            policy.obs_buffer = []
            policy.action_buffer = []
            policy.action_idx = 0
        
        episode_reward = 0
        step_count = 0
        done = False
        episode_success = False  # Track if success was ever achieved during episode
        
        while not done and step_count < args.max_steps:
            # Get action sequences from each agent's policy
            # Each sequence contains action_repeat actions to execute consecutively
            action_sequences = {}
            
            if is_multi_agent:
                for agent_id in range(num_agents):
                    agent_uid = f'panda-{agent_id}'
                    qpos = obs['agent'][agent_uid]['qpos'].squeeze(0).cpu().numpy()
                    
                    # Convert 9-dim qpos to 8-dim state (7 joints + 1 gripper command)
                    # Panda qpos: [j1, j2, j3, j4, j5, j6, j7, gripper_left, gripper_right]
                    # Training data: [j1, j2, j3, j4, j5, j6, j7, gripper_command]
                    # Remove 2 gripper fingers, append planner's gripper command state
                    agent_pos = np.append(qpos[:-2], planner.gripper_state[agent_id])
                    
                    model_input = get_model_input(obs, agent_pos, agent_id)
                    
                    # Update policy observation buffer
                    policies[agent_id].update_obs(model_input)
                    
                    # Get action sequence (list of action_repeat actions)
                    action_sequences[agent_uid] = policies[agent_id].get_action(model_input)
            else:
                qpos = obs['agent']['qpos'].squeeze(0).cpu().numpy()
                # Convert 9-dim qpos to 8-dim state (7 joints + 1 gripper command)
                # Remove 2 gripper fingers, append planner's gripper command state
                agent_pos = np.append(qpos[:-2], planner.gripper_state[0])
                model_input = get_model_input(obs, agent_pos, 0)
                policies[0].update_obs(model_input)
                action_sequences['single'] = policies[0].get_action(model_input)
            
            # Execute all actions in the sequence (action_repeat steps per policy prediction)
            num_repeat = len(list(action_sequences.values())[0])
            for action_idx in range(num_repeat):
                if done or step_count >= args.max_steps:
                    break
                    
                # Collect current action for each agent
                if is_multi_agent:
                    actions = {}
                    for agent_id in range(num_agents):
                        agent_uid = f'panda-{agent_id}'
                        actions[agent_uid] = action_sequences[agent_uid][action_idx]
                else:
                    actions = action_sequences['single'][action_idx]
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(actions)
                
                # Convert tensors to Python types (for GPU backend compatibility)
                if isinstance(terminated, torch.Tensor):
                    terminated = terminated.item()
                if isinstance(truncated, torch.Tensor):
                    truncated = truncated.item()
                if isinstance(reward, torch.Tensor):
                    reward = reward.item()
                
                # Check success every step (success is a transient condition)
                step_success = info.get('success', False)
                if isinstance(step_success, torch.Tensor):
                    step_success = step_success.item()
                if step_success:
                    episode_success = True  # Remember success was achieved
                
                done = terminated or truncated
                episode_reward += reward
                step_count += 1
                
                if verbose:
                    print(f"Step {step_count}: reward={reward:.4f}, done={done}, success={step_success}")
        
        # Episode results - use tracked success (True if success ever achieved)
        if episode_success:
            success_count += 1
        
        episode_rewards.append(float(episode_reward))
        episode_lengths.append(step_count)
        
        print(f"Episode {episode_idx + 1} finished:")
        print(f"  Success: {episode_success}")
        print(f"  Reward: {episode_reward:.4f}")
        print(f"  Length: {step_count}")
        print(f"  Success rate so far: {success_count}/{episode_idx + 1} ({100*success_count/(episode_idx+1):.1f}%)")
    
    # Final statistics
    print(f"\n{'='*80}")
    print(f"Evaluation Results")
    print(f"{'='*80}")
    print(f"Task: {task_name}")
    print(f"Policy: {args.policy_type}")
    print(f"Episodes: {args.num_eval_episodes}")
    print(f"Success rate: {success_count}/{args.num_eval_episodes} ({100*success_count/args.num_eval_episodes:.1f}%)")
    print(f"Average reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"{'='*80}")
    
    env.close()
    
    return {
        'success_rate': success_count / args.num_eval_episodes,
        'avg_reward': np.mean(episode_rewards),
        'avg_length': np.mean(episode_lengths),
    }


if __name__ == "__main__":
    args = parse_args()
    main(args)

