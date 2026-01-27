"""
Base evaluator for VLA policies in simulation.

This module provides a unified evaluation framework that works with all
VLA policy implementations through the policy wrapper interface.

Consolidated patterns from:
- eval_multi_pi0.py: Environment setup, episode loop, success tracking
- eval_multi_openvla.py: Video recording, metrics computation
- eval_multi_dp.py: Multi-agent support, action execution
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import yaml

try:
    import gymnasium as gym
except ImportError:
    import gym

from .policy_wrapper import BasePolicyWrapper


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    # Environment
    config_path: str = ""
    task_name: str = ""
    obs_mode: str = "rgbd"
    control_mode: str = "pd_joint_delta_pos"
    render_mode: str = "cameras"
    num_envs: int = 1
    
    # Evaluation
    num_episodes: int = 100
    max_steps: int = 300
    seed: int = 42
    
    # Recording
    record_dir: str = "eval_records/{env_id}"
    save_video: bool = True
    save_trajectory: bool = False
    
    # Debug
    verbose: bool = False
    debug: bool = False


@dataclass
class EvalResult:
    """Result of evaluation run."""
    task_name: str
    num_episodes: int
    success_rate: float
    mean_reward: float
    std_reward: float
    mean_steps: float
    episode_rewards: List[float] = field(default_factory=list)
    episode_successes: List[bool] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'task_name': self.task_name,
            'num_episodes': self.num_episodes,
            'success_rate': self.success_rate,
            'mean_reward': self.mean_reward,
            'std_reward': self.std_reward,
            'mean_steps': self.mean_steps,
            'episode_rewards': self.episode_rewards,
            'episode_successes': self.episode_successes,
            'episode_steps': self.episode_steps,
        }


class BaseEvaluator:
    """
    Unified evaluator for VLA policies.
    
    This class provides a common framework for evaluating any VLA policy
    in simulation environments. It handles:
    - Environment setup and management
    - Episode execution with multi-agent support
    - Metrics tracking and aggregation
    - Video recording
    
    Usage:
        evaluator = BaseEvaluator(config, policy_wrappers)
        result = evaluator.run_evaluation()
    """
    
    def __init__(
        self,
        config: Union[EvalConfig, Dict[str, Any]],
        policy_wrappers: List[BasePolicyWrapper],
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
            policy_wrappers: List of policy wrappers, one per agent
        """
        if isinstance(config, dict):
            self.config = EvalConfig(**config)
        else:
            self.config = config
        
        self.policy_wrappers = policy_wrappers
        self.env = None
        self.num_agents = len(policy_wrappers)
        
        # Load task config if provided
        if self.config.config_path and not self.config.task_name:
            with open(self.config.config_path, 'r') as f:
                task_config = yaml.safe_load(f)
                self.config.task_name = task_config.get('task_name', '')
    
    def _setup_env(self):
        """
        Setup simulation environment.
        
        Creates gymnasium environment with recording wrapper if needed.
        """
        # Construct environment ID
        env_id = self.config.task_name
        if env_id and not env_id.endswith('-rf'):
            env_id += '-rf'
        
        # Environment kwargs
        env_kwargs = {
            'config': self.config.config_path,
            'obs_mode': self.config.obs_mode,
            'control_mode': self.config.control_mode,
            'render_mode': self.config.render_mode,
            'num_envs': self.config.num_envs,
            'sim_backend': 'auto',
            'enable_shadow': True,
            'parallel_in_single_scene': False,
        }
        
        # Create environment
        self.env = gym.make(env_id, **env_kwargs)
        
        # Add recording wrapper if needed
        if self.config.save_video and self.config.record_dir:
            try:
                from mani_skill.utils.wrappers.record import RecordEpisode
                
                record_dir = self.config.record_dir.format(env_id=env_id)
                record_dir = f"{record_dir}/{self.config.seed}"
                
                self.env = RecordEpisode(
                    self.env,
                    record_dir,
                    info_on_video=False,
                    save_trajectory=self.config.save_trajectory,
                    max_steps_per_video=30000,
                )
            except ImportError:
                print("Warning: RecordEpisode not available, skipping video recording")
        
        return self.env
    
    def _reset_policies(self):
        """Reset all policy wrappers."""
        for wrapper in self.policy_wrappers:
            wrapper.reset()
    
    def _get_agent_observations(
        self,
        raw_obs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract per-agent observations from raw environment observation.
        
        Args:
            raw_obs: Raw observation from environment
            
        Returns:
            List of observation dictionaries, one per agent
        """
        # Implementation depends on environment observation structure
        # This is a base implementation that should be overridden for specific envs
        
        observations = []
        for agent_id in range(self.num_agents):
            agent_obs = {}
            
            # Try to extract agent-specific sensor data
            if 'sensor_data' in raw_obs:
                sensor_data = raw_obs['sensor_data']
                agent_obs['sensor_data'] = {}
                
                # Look for agent-specific cameras
                for key, value in sensor_data.items():
                    if f'agent{agent_id}' in key or (agent_id == 0 and 'agent' not in key):
                        agent_obs['sensor_data'][key] = value
            
            # Copy other relevant fields
            for key in ['state', 'agent_pos', 'extra']:
                if key in raw_obs:
                    agent_obs[key] = raw_obs[key]
            
            observations.append(agent_obs)
        
        return observations
    
    def _execute_actions(
        self,
        actions: List[List[np.ndarray]],
    ) -> tuple:
        """
        Execute actions in the environment.
        
        Args:
            actions: List of action lists, one per agent
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        # Combine actions from all agents
        # Assumes actions are returned as [action] * repeat from wrappers
        combined_actions = []
        for agent_actions in actions:
            combined_actions.append(agent_actions[0])  # Take first action
        
        # Stack actions for multi-agent
        if self.num_agents > 1:
            action = np.stack(combined_actions, axis=0)
        else:
            action = combined_actions[0]
        
        # Step environment
        obs, reward, done, truncated, info = self.env.step(action)
        
        return obs, reward, done, truncated, info
    
    def _run_episode(self) -> Dict[str, Any]:
        """
        Run a single evaluation episode.
        
        Returns:
            Dictionary with episode results:
                - 'reward': Total episode reward
                - 'steps': Number of steps
                - 'success': Whether task was successful
        """
        # Reset environment and policies
        obs, info = self.env.reset()
        self._reset_policies()
        
        total_reward = 0.0
        steps = 0
        success = False
        
        # Episode loop
        for step in range(self.config.max_steps):
            # Get per-agent observations
            agent_obs = self._get_agent_observations(obs)
            
            # Get actions from each policy
            actions = []
            for i, wrapper in enumerate(self.policy_wrappers):
                wrapper.update_obs(agent_obs[i])
                action = wrapper.get_action(agent_obs[i])
                actions.append(action)
            
            # Execute actions
            obs, reward, done, truncated, info = self._execute_actions(actions)
            
            # Accumulate reward
            if isinstance(reward, (list, np.ndarray)):
                total_reward += float(np.sum(reward))
            else:
                total_reward += float(reward)
            
            steps += 1
            
            # Check termination
            if isinstance(done, (list, np.ndarray)):
                done = np.any(done)
            if isinstance(truncated, (list, np.ndarray)):
                truncated = np.any(truncated)
            
            if done or truncated:
                # Check success
                if 'success' in info:
                    if isinstance(info['success'], (list, np.ndarray)):
                        success = np.all(info['success'])
                    else:
                        success = bool(info['success'])
                break
        
        return {
            'reward': total_reward,
            'steps': steps,
            'success': success,
        }
    
    def run_evaluation(self) -> EvalResult:
        """
        Run full evaluation over multiple episodes.
        
        Returns:
            EvalResult with aggregated metrics
        """
        # Setup environment
        self._setup_env()
        
        # Set seed
        np.random.seed(self.config.seed)
        if hasattr(self.env, 'action_space') and self.env.action_space is not None:
            self.env.action_space.seed(self.config.seed)
        
        # Run episodes
        episode_rewards = []
        episode_successes = []
        episode_steps = []
        
        print(f"Running evaluation for {self.config.task_name}")
        print(f"  Episodes: {self.config.num_episodes}")
        print(f"  Max steps: {self.config.max_steps}")
        print("-" * 50)
        
        for ep in range(self.config.num_episodes):
            result = self._run_episode()
            
            episode_rewards.append(result['reward'])
            episode_successes.append(result['success'])
            episode_steps.append(result['steps'])
            
            if self.config.verbose:
                print(f"Episode {ep + 1}: reward={result['reward']:.2f}, "
                      f"steps={result['steps']}, success={result['success']}")
        
        # Compute aggregated metrics
        success_rate = np.mean(episode_successes)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_steps = np.mean(episode_steps)
        
        print("-" * 50)
        print(f"Results:")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Mean steps: {mean_steps:.1f}")
        
        # Cleanup
        self.env.close()
        
        return EvalResult(
            task_name=self.config.task_name,
            num_episodes=self.config.num_episodes,
            success_rate=success_rate,
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_steps=mean_steps,
            episode_rewards=episode_rewards,
            episode_successes=episode_successes,
            episode_steps=episode_steps,
        )
    
    def save_results(
        self,
        result: EvalResult,
        output_path: str,
    ):
        """
        Save evaluation results to JSON file.
        
        Args:
            result: EvalResult to save
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"Saved results to {output_path}")

