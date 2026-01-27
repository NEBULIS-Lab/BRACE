"""
Multi-agent evaluation script for OpenVLA policies.

This script evaluates trained OpenVLA models on multi-agent RoboFactory tasks.
"""

import sys
sys.path.append('./')
sys.path.insert(0, './policy/OpenVLA')

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

from openvla_policy.policy.openvla_policy import OpenVLAPolicy
from robofactory.policy.shared import get_task_instruction
from robofactory.policy.shared.evaluation import BasePolicyWrapper


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Evaluate OpenVLA multi-agent policies")
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to task config file'
    )
    parser.add_argument(
        '--data_num',
        type=int,
        default=150,
        help='Number of training samples used'
    )
    parser.add_argument(
        '--checkpoint_num',
        type=int,
        default=300,
        help='Checkpoint epoch number'
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
        default='./eval_video/openvla/{env_id}',
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
    
    return parser.parse_args()


def get_model_input(observation, agent_pos, agent_id):
    """
    Extract model input from observation.
    
    Args:
        observation: Environment observation
        agent_pos: Agent joint positions
        agent_id: Agent ID
        
    Returns:
        Dictionary with 'image' and 'agent_pos'
    """
    camera_name = f'head_camera_agent{agent_id}'
    head_cam = observation['sensor_data'][camera_name]['rgb'].squeeze(0).numpy()
    
    # Convert to HWC format if needed
    if head_cam.shape[0] == 3:
        head_cam = np.transpose(head_cam, (1, 2, 0))
    
    return {
        'image': head_cam,
        'agent_pos': agent_pos,
    }


def load_task_instruction(task_name: str) -> str:
    """Get language instruction for a task (uses centralized task_instructions module)."""
    return get_task_instruction(task_name)


class OpenVLAPolicyWrapper(BasePolicyWrapper):
    """Wrapper for OpenVLA policy to match evaluation interface."""
    
    def __init__(self, task_name: str, checkpoint_num: int, data_num: int, agent_id: int = 0):
        """Initialize policy wrapper."""
        # Initialize base class
        super().__init__(task_name=task_name, device='cuda:0')
        
        self.checkpoint_num = checkpoint_num
        self.data_num = data_num
        self.agent_id = agent_id
        
        # Build checkpoint path
        checkpoint_dir = f'checkpoints/{task_name}_Agent{agent_id}_{data_num}/epoch_{checkpoint_num}'
        
        if not Path(checkpoint_dir).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
        
        # Load statistics
        stats_path = f'data/rlds_data/{task_name}_Agent{agent_id}_{data_num}/statistics.json'
        import json
        if Path(stats_path).exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
        else:
            stats = None
        
        # Load policy
        self.load_policy(checkpoint_dir, action_statistics=stats.get('action') if stats else None)
        
        # Get instruction
        self.instruction = load_task_instruction(task_name)
        
    def load_policy(self, checkpoint_path: str, **kwargs):
        """Load policy from checkpoint."""
        action_statistics = kwargs.get('action_statistics')
        
        self.policy = OpenVLAPolicy(
            checkpoint_path=checkpoint_path,
            device=self.device,
            action_statistics=action_statistics
        )
    
    def get_action(self, observation=None):
        """Get action from policy."""
        if observation is None and len(self.obs_buffer) > 0:
            observation = self.obs_buffer[-1]
        
        # Predict action
        action = self.policy.predict(observation, self.instruction)
        
        # Return as list of actions (for compatibility with multi-step execution)
        return [action for _ in range(6)]


def main(args):
    """Main evaluation function."""
    np.set_printoptions(suppress=True, precision=5)
    verbose = args.debug == 1
    
    # Set seed
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        env_id = config['task_name']
        if not env_id.endswith('-rf'):
            env_id += '-rf'
    
    # Create environment
    env_kwargs = dict(
        config=args.config,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack='default'),
        human_render_camera_configs=dict(shader_pack='default'),
        viewer_camera_configs=dict(shader_pack='default'),
        num_envs=args.num_envs,
        sim_backend='auto',
        enable_shadow=True,
        parallel_in_single_scene=False,
    )
    
    env: BaseEnv = gym.make(env_id, **env_kwargs)
    
    # Setup recording
    record_dir = args.record_dir.format(env_id=env_id)
    record_dir = f"{record_dir}/{args.seed}_{args.data_num}_{args.checkpoint_num}"
    if record_dir:
        env = RecordEpisodeMA(
            env,
            record_dir,
            info_on_video=False,
            save_trajectory=False,
            max_steps_per_video=30000
        )
    
    # Reset environment
    raw_obs, _ = env.reset(seed=args.seed)
    
    # Initialize motion planner
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=verbose,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=verbose,
        print_env_info=False,
        is_multi_agent=True
    )
    
    # Load OpenVLA policies for all agents
    agent_num = planner.agent_num
    openvla_models = []
    
    print(f"\nLoading OpenVLA policies for {agent_num} agents...")
    for i in range(agent_num):
        try:
            model = OpenVLAPolicyWrapper(
                env_id,
                args.checkpoint_num,
                args.data_num,
                agent_id=i
            )
            openvla_models.append(model)
            print(f"  ✓ Loaded policy for agent {i}")
        except FileNotFoundError as e:
            print(f"  ✗ Failed to load policy for agent {i}: {e}")
            env.close()
            return
    
    # Initialize observations
    for agent_id in range(agent_num):
        initial_qpos = raw_obs['agent'][f'panda-{agent_id}']['qpos'].squeeze(0)[:-2].numpy()
        initial_qpos = np.append(initial_qpos, planner.gripper_state[agent_id])
        obs = get_model_input(raw_obs, initial_qpos, agent_id)
        openvla_models[agent_id].update_obs(obs)
    
    # Setup viewer
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = False
        env.render()
    
    # Evaluation loop
    cnt = 0
    print(f"\n{'='*60}")
    print(f"Starting evaluation...")
    print(f"Task: {env_id}")
    print(f"Seed: {args.seed}")
    print(f"Max steps: {args.max_steps}")
    print(f"{'='*60}\n")
    
    while True:
        if verbose:
            print(f"Iteration: {cnt}")
        cnt += 1
        
        if cnt > args.max_steps:
            print(f"\nMax steps ({args.max_steps}) reached")
            break
        
        # Collect actions from all agents
        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)
        
        for agent_id in range(agent_num):
            action_list = openvla_models[agent_id].get_action()
            
            for i in range(6):
                now_action = action_list[i]
                raw_obs = env.get_obs()
                
                if i == 0:
                    current_qpos = raw_obs['agent'][f'panda-{agent_id}']['qpos'].squeeze(0)[:-2].numpy()
                else:
                    current_qpos = action_list[i - 1][:-1]
                
                path = np.vstack((current_qpos, now_action[:-1]))
                
                try:
                    times, position, right_vel, acc, duration = planner.planner[agent_id].TOPP(
                        path, 0.05, verbose=verbose
                    )
                except Exception as e:
                    if verbose:
                        print(f"Error in motion planning: {e}")
                    action_now = np.hstack([current_qpos, now_action[-1]])
                    action_dict[f'panda-{agent_id}'].append(action_now)
                    action_step_dict[f'panda-{agent_id}'].append(1)
                    continue
                
                n_step = position.shape[0]
                action_step_dict[f'panda-{agent_id}'].append(n_step)
                gripper_state = now_action[-1]
                
                if n_step == 0:
                    action_now = np.hstack([current_qpos, gripper_state])
                    action_dict[f'panda-{agent_id}'].append(action_now)
                
                for j in range(n_step):
                    true_action = np.hstack([position[j], gripper_state])
                    action_dict[f'panda-{agent_id}'].append(true_action)
        
        # Execute actions
        start_idx = [0] * agent_num
        
        for i in range(6):
            max_step = max(action_step_dict[f'panda-{j}'][i] for j in range(agent_num))
            
            for j in range(max_step):
                true_action = {}
                for agent_id in range(agent_num):
                    now_step = min(j, action_step_dict[f'panda-{agent_id}'][i] - 1)
                    true_action[f'panda-{agent_id}'] = action_dict[f'panda-{agent_id}'][
                        start_idx[agent_id] + now_step
                    ]
                
                observation, reward, terminated, truncated, info = env.step(true_action)
                
                if verbose:
                    env.render_human()
            
            # Update observations
            for agent_id in range(agent_num):
                start_idx[agent_id] += action_step_dict[f'panda-{agent_id}'][i]
                if action_step_dict[f'panda-{agent_id}'][i] > 0:
                    obs = get_model_input(observation, true_action[f'panda-{agent_id}'], agent_id)
                    openvla_models[agent_id].update_obs(obs)
        
        # Render
        if args.render_mode is not None:
            env.render()
        
        # Check success
        if info.get('success', False):
            env.close()
            if record_dir:
                print(f"\n✓ SUCCESS! Video saved to {record_dir}")
            else:
                print(f"\n✓ SUCCESS!")
            return
    
    # Failed
    env.close()
    if record_dir:
        print(f"\n✗ FAILED. Video saved to {record_dir}")
    else:
        print(f"\n✗ FAILED")


if __name__ == "__main__":
    args = parse_args()
    main(args)

