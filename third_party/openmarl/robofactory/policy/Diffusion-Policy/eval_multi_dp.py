# Setup headless rendering for parallel evaluation
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_GL"] = "egl" 
os.environ["DISPLAY"] = ""

import sys
import pathlib

# Add robofactory parent directory to path
_script_dir = pathlib.Path(__file__).parent.resolve()
_robofactory_dir = _script_dir.parent.parent  # policy/Diffusion-Policy -> policy -> robofactory
_robofactory_parent = _robofactory_dir.parent  # robofactory -> parent (workspace)
sys.path.insert(0, str(_robofactory_parent))
sys.path.insert(0, str(_script_dir))

# Clear DDP environment variables for single-GPU evaluation
# This prevents RobotWorkspace from trying to initialize distributed training
for _key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
    if _key in os.environ:
        del os.environ[_key]

import torch

import hydra
from pathlib import Path
from collections import deque, defaultdict
from robofactory.tasks import *
import traceback

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.dp_runner import DPRunner
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver


import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from robofactory.utils.wrappers.record import RecordEpisodeMA

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = ""
    """The environment ID of the task you want to simulate"""

    config: str = "${CONFIG_DIR}/robocasa/take_photo.yaml"
    """Configuration to build scenes, assets and agents."""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    # Seed arguments - support both single seed and range
    seed: Annotated[Optional[int], tyro.conf.arg(aliases=["-s"])] = 1000
    """Starting seed for evaluation"""
    
    num_episodes: int = 1
    """Number of episodes to run (seeds will be seed, seed+1, ..., seed+num_episodes-1)"""

    data_num: int = 100
    """The number of episode data used for training the policy"""

    checkpoint_num: int = 300
    """The number of training epoch of the checkpoint"""

    record_dir: Optional[str] = './eval_video/diffusion_policy/{env_id}'
    """Directory to save recordings"""

    max_steps: int = 250
    """Maximum number of steps to run the simulation"""

def get_policy(checkpoint, output_dir, device):
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy


class DP:
    def __init__(self, task_name, checkpoint_num: int, data_num: int, id: int = 0):
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
        checkpoint_path = os.path.join(project_root, 'robofactory', 'checkpoints', 'diffusion_policy', f'{task_name}_Agent{id}_{data_num}', f'{checkpoint_num}.ckpt')
        self.policy = get_policy(checkpoint_path, None, 'cuda:0')
        self.runner = DPRunner(output_dir=None)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def reset(self):
        """Reset the observation buffer for a new episode"""
        self.runner = DPRunner(output_dir=None)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

def get_model_input(observation, agent_pos, agent_id, is_multi_agent=True):
    # Camera is always named head_camera_agent{id}, even for single-agent envs
    camera_name = f'head_camera_agent{agent_id}'
    head_cam = np.moveaxis(observation['sensor_data'][camera_name]['rgb'].squeeze(0).cpu().numpy(), -1, 0) / 255
    return dict(
        head_cam=head_cam,
        agent_pos=agent_pos,
    )

def run_episode(env, dp_models, planner, seed, args, env_id, verbose=False):
    """Run a single evaluation episode and return success status"""
    is_multi_agent = planner.is_multi_agent
    agent_num = planner.agent_num
    
    # Subsample TOPP trajectory to reduce video frames
    # Execute every Nth step instead of all interpolated steps
    TOPP_SUBSAMPLE = 10  # Execute every 10th TOPP step (reduces frames by 10x)
    
    # Reset environment with new seed
    raw_obs, _ = env.reset(seed=seed)
    
    # Reset all DP models for new episode
    for model in dp_models:
        model.reset()
    
    # Reset planner gripper states
    planner.gripper_state = [1] * agent_num
    
    if env.action_space is not None:
        env.action_space.seed(seed)
    
    # Initialize observations for all agents
    for id in range(agent_num):
        if is_multi_agent:
            initial_qpos = raw_obs['agent'][f'panda-{id}']['qpos'].squeeze(0)[:-2].cpu().numpy()
        else:
            initial_qpos = raw_obs['agent']['qpos'].squeeze(0)[:-2].cpu().numpy()
        initial_qpos = np.append(initial_qpos, planner.gripper_state[id])
        obs = get_model_input(raw_obs, initial_qpos, id, is_multi_agent)
        dp_models[id].update_obs(obs)
    
    cnt = 0
    total_steps = 0
    while True:
        if verbose:
            print(f"Iteration: {cnt}, Total steps: {total_steps}")
        cnt = cnt + 1
        # Exit based on ACTUAL env.step() calls, not iterations
        # Each iteration does 6 actions × ~10 TOPP steps = ~60 env.step() calls
        if total_steps >= args.max_steps:
            break
        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)
        for id in range(agent_num):
            action_list = dp_models[id].get_action()
            for i in range(6):
                now_action = action_list[i]
                raw_obs = env.get_obs()
                if i == 0:
                    if is_multi_agent:
                        current_qpos = raw_obs['agent'][f'panda-{id}']['qpos'].squeeze(0)[:-2].cpu().numpy()
                    else:
                        current_qpos = raw_obs['agent']['qpos'].squeeze(0)[:-2].cpu().numpy()
                else:
                    current_qpos = action_list[i - 1][:-1]
                path = np.vstack((current_qpos, now_action[:-1]))
                try:
                    times, position, right_vel, acc, duration = planner.planner[id].TOPP(path, 0.05, verbose=False)
                except Exception as e:
                    if verbose:
                        print(f"Error occurred: {e}")
                    action_now = np.hstack([current_qpos, now_action[-1]])
                    action_dict[f'panda-{id}'].append(action_now)
                    action_step_dict[f'panda-{id}'].append(1)
                    continue
                
                n_step = position.shape[0]
                gripper_state = now_action[-1]
                
                if n_step == 0:
                    action_now = np.hstack([current_qpos, gripper_state])
                    action_dict[f'panda-{id}'].append(action_now)
                    action_step_dict[f'panda-{id}'].append(1)
                else:
                    # Subsample TOPP trajectory: take every Nth step + always include last step
                    subsampled_indices = list(range(0, n_step, TOPP_SUBSAMPLE))
                    if (n_step - 1) not in subsampled_indices:
                        subsampled_indices.append(n_step - 1)
                    
                    action_step_dict[f'panda-{id}'].append(len(subsampled_indices))
                    for j in subsampled_indices:
                        true_action = np.hstack([position[j], gripper_state])
                        action_dict[f'panda-{id}'].append(true_action)
        
        start_idx = []
        for id in range(agent_num):
            start_idx.append(0)
        for i in range(6):
            max_step = 0
            for id in range(agent_num):
                max_step = max(max_step, action_step_dict[f'panda-{id}'][i])
            for j in range(max_step):
                if is_multi_agent:
                    true_action = dict()
                    for id in range(agent_num):
                        now_step = min(j, action_step_dict[f'panda-{id}'][i] - 1)
                        true_action[f'panda-{id}'] = action_dict[f'panda-{id}'][start_idx[id] + now_step]
                else:
                    # Single agent: use plain array, not dict
                    now_step = min(j, action_step_dict['panda-0'][i] - 1)
                    true_action = action_dict['panda-0'][start_idx[0] + now_step]
                observation, reward, terminated, truncated, info = env.step(true_action)
                total_steps += 1
            if verbose:
                print(true_action)
                print("max_step", max_step)
            for id in range(agent_num):
                start_idx[id] += action_step_dict[f'panda-{id}'][i]
                if action_step_dict[f'panda-{id}'][i] == 0:
                    continue
                if is_multi_agent:
                    action_for_obs = true_action[f'panda-{id}']
                else:
                    action_for_obs = true_action
                obs = get_model_input(observation, action_for_obs, id, is_multi_agent)
                dp_models[id].update_obs(obs)
        if verbose:
            print("info", info)
        if info['success'] == True:
            if verbose:
                print(f"Success after {total_steps} env steps")
            return True
    if verbose:
        print(f"Failed after {total_steps} env steps")
    return False

def main(args: Args):
    np.set_printoptions(suppress=True, precision=5)
    verbose = not args.quiet
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    
    env_id = args.env_id
    if env_id == "":
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            env_id = config['task_name'] + '-rf'
    
    print(f"=== Diffusion Policy Evaluation ===")
    print(f"Task: {env_id}")
    print(f"Seeds: {args.seed} to {args.seed + args.num_episodes - 1} ({args.num_episodes} episodes)")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"===================================")
    
    # Create environment once
    env_kwargs = dict(
        config=args.config, 
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    
    env: BaseEnv = gym.make(env_id, **env_kwargs)
    
    # Initial reset to setup the environment
    raw_obs, _ = env.reset(seed=args.seed)
    
    # Check if it's multi-agent environment
    env_unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
    is_multi_agent = hasattr(env_unwrapped, 'agent') and hasattr(env_unwrapped.agent, 'agents')
    
    if is_multi_agent:
        agents_list = env_unwrapped.agent.agents
    else:
        agents_list = [env_unwrapped.agent]
    
    base_pose = [agent.robot.pose for agent in agents_list]
    
    # Create planner once
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=verbose,
        base_pose=base_pose,
        visualize_target_grasp_pose=verbose,
        print_env_info=False,
        is_multi_agent=is_multi_agent
    )
    
    # Load models once
    agent_num = planner.agent_num
    print(f"Loading {agent_num} DP model(s)...")
    dp_models = []
    for i in range(agent_num):
        dp_models.append(DP(env_id, args.checkpoint_num, args.data_num, id=i))
    print("Models loaded!")
    
    # Run evaluation episodes
    results = []
    success_count = 0
    
    for episode_idx in range(args.num_episodes):
        current_seed = args.seed + episode_idx
        
        # Setup recording for this episode
        if args.record_dir:
            record_dir = args.record_dir.format(env_id=env_id)
            record_dir = f"{record_dir}/[{current_seed}]_{args.data_num}_{args.checkpoint_num}"
            # Wrap env with new recorder for this episode
            record_env = RecordEpisodeMA(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=args.max_steps)
        else:
            record_env = env
        
        # Run episode
        success = run_episode(record_env, dp_models, planner, current_seed, args, env_id, verbose=verbose)
        
        # Close recorder to save video
        if args.record_dir and hasattr(record_env, '_video_recorder'):
            record_env.close()
        
        if success:
            success_count += 1
            print(f"Episode {episode_idx + 1}/{args.num_episodes} (seed={current_seed}): success")
        else:
            print(f"Episode {episode_idx + 1}/{args.num_episodes} (seed={current_seed}): failed")
        
        results.append((current_seed, 1 if success else 0))
    
    # Print summary
    print(f"\n=== Results ===")
    print(f"Success: {success_count}/{args.num_episodes} ({100*success_count/args.num_episodes:.2f}%)")
    
    # Print results in format that bash script can parse
    for seed, success in results:
        print(f"RESULT:{seed},{success}")
    
    env.close() 

if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
