#!/usr/bin/env python3
"""
Unified evaluation script for all VLA policies.

This script provides a single entry point for evaluating any VLA policy
(Diffusion-Policy, OpenVLA, Pi0, or custom policies) in simulation.

Usage:
    # Auto-detect policy type from checkpoint
    python -m robofactory.policy.shared.evaluation.unified_eval \
        --checkpoint /path/to/checkpoint \
        --config configs/task.yaml
    
    # Specify policy type explicitly
    python -m robofactory.policy.shared.evaluation.unified_eval \
        --policy_type openvla \
        --checkpoint /path/to/checkpoint \
        --task_name LiftBarrier-rf \
        --num_episodes 100
    
    # Evaluate multiple agents
    python -m robofactory.policy.shared.evaluation.unified_eval \
        --policy_type pi0 \
        --checkpoints /path/to/agent0 /path/to/agent1 \
        --config configs/task.yaml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml

from .base_evaluator import BaseEvaluator, EvalConfig, EvalResult
from .policy_wrapper import BasePolicyWrapper, ActionChunkingWrapper


# Policy type detection patterns
POLICY_PATTERNS = {
    'openvla': ['openvla', 'vla'],
    'pi0': ['pi0', 'pi05', 'openpi'],
    'diffusion': ['diffusion', 'dp', 'diffusion_policy'],
}


def detect_policy_type(checkpoint_path: str) -> str:
    """
    Auto-detect policy type from checkpoint path.
    
    Args:
        checkpoint_path: Path to checkpoint directory or file
        
    Returns:
        Detected policy type ('openvla', 'pi0', 'diffusion', or 'unknown')
    """
    path_lower = str(checkpoint_path).lower()
    
    for policy_type, patterns in POLICY_PATTERNS.items():
        for pattern in patterns:
            if pattern in path_lower:
                return policy_type
    
    # Check for specific files that indicate policy type
    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.is_dir():
        # Pi0 uses safetensors
        if (checkpoint_dir / "model.safetensors").exists():
            return 'pi0'
        # Diffusion Policy uses .ckpt files
        if any(checkpoint_dir.glob("*.ckpt")):
            return 'diffusion'
        # OpenVLA typically has adapter_model files
        if (checkpoint_dir / "adapter_model.safetensors").exists():
            return 'openvla'
    
    return 'unknown'


def create_policy_wrapper(
    policy_type: str,
    task_name: str,
    checkpoint_path: str,
    agent_id: int = 0,
    device: str = 'cuda:0',
    **kwargs,
) -> BasePolicyWrapper:
    """
    Create appropriate policy wrapper based on policy type.
    
    Args:
        policy_type: Type of policy ('openvla', 'pi0', 'diffusion')
        task_name: Name of the task
        checkpoint_path: Path to checkpoint
        agent_id: Agent ID for multi-agent setups
        device: Device for inference
        **kwargs: Additional arguments for specific wrappers
        
    Returns:
        Appropriate policy wrapper instance
    """
    if policy_type == 'openvla':
        return OpenVLAEvalWrapper(
            task_name=task_name,
            checkpoint_path=checkpoint_path,
            agent_id=agent_id,
            device=device,
            **kwargs,
        )
    elif policy_type == 'pi0':
        return Pi0EvalWrapper(
            task_name=task_name,
            checkpoint_path=checkpoint_path,
            agent_id=agent_id,
            device=device,
            **kwargs,
        )
    elif policy_type == 'diffusion':
        return DiffusionPolicyEvalWrapper(
            task_name=task_name,
            checkpoint_path=checkpoint_path,
            agent_id=agent_id,
            device=device,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


class OpenVLAEvalWrapper(BasePolicyWrapper):
    """Evaluation wrapper for OpenVLA policies."""
    
    def __init__(
        self,
        task_name: str,
        checkpoint_path: str,
        agent_id: int = 0,
        device: str = 'cuda:0',
        statistics_path: Optional[str] = None,
    ):
        super().__init__(task_name=task_name, device=device)
        self.checkpoint_path = checkpoint_path
        self.agent_id = agent_id
        self.statistics_path = statistics_path
        self.policy = None
    
    def load_policy(self, checkpoint_path: str = None, **kwargs):
        """Load OpenVLA policy from checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        
        try:
            from robofactory.policy.OpenVLA.openvla_policy.policy.openvla_policy import OpenVLAPolicy
            from robofactory.policy.shared import get_task_instruction
            
            self.policy = OpenVLAPolicy.from_checkpoint(
                checkpoint_path=checkpoint_path,
                device=self.device,
                statistics_path=self.statistics_path,
            )
            self.instruction = get_task_instruction(self.task_name, policy_type='openvla')
            self.policy.set_instruction(self.instruction)
        except ImportError as e:
            raise ImportError(f"OpenVLA policy not available: {e}")
    
    def get_action(self, observation=None):
        """Get action from OpenVLA policy."""
        if self.policy is None:
            self.load_policy()
        
        if observation is None and len(self.obs_buffer) > 0:
            observation = self.obs_buffer[-1]
        
        action = self.policy.predict_action(observation, self.instruction)
        return [action for _ in range(6)]


class Pi0EvalWrapper(ActionChunkingWrapper):
    """Evaluation wrapper for Pi0 policies with action chunking."""
    
    def __init__(
        self,
        task_name: str,
        checkpoint_path: str,
        agent_id: int = 0,
        device: str = 'cuda:0',
        action_horizon: int = 50,
    ):
        super().__init__(
            task_name=task_name,
            device=device,
            action_horizon=action_horizon,
        )
        self.checkpoint_path = checkpoint_path
        self.agent_id = agent_id
        self.policy = None
    
    def load_policy(self, checkpoint_path: str = None, **kwargs):
        """Load Pi0 policy from checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        
        try:
            from robofactory.policy.Pi0.pi0_policy.policy.pi0_policy import Pi0Policy
            
            self.policy = Pi0Policy.from_checkpoint(
                checkpoint_path=checkpoint_path,
                device=self.device,
                task_name=self.task_name,
            )
            self.action_horizon = self.policy.action_horizon
        except ImportError as e:
            raise ImportError(f"Pi0 policy not available: {e}")
    
    def _predict_action_sequence(self, observation):
        """Predict action sequence from Pi0 policy."""
        if self.policy is None:
            self.load_policy()
        
        # Pi0 returns action sequence directly
        action_sequence = self.policy.predict_action(observation)
        return action_sequence


class DiffusionPolicyEvalWrapper(BasePolicyWrapper):
    """Evaluation wrapper for Diffusion Policy."""
    
    def __init__(
        self,
        task_name: str,
        checkpoint_path: str,
        agent_id: int = 0,
        device: str = 'cuda:0',
        n_obs_steps: int = 3,
    ):
        super().__init__(task_name=task_name, device=device)
        self.checkpoint_path = checkpoint_path
        self.agent_id = agent_id
        self.n_obs_steps = n_obs_steps
        self.policy = None
        self.runner = None
    
    def load_policy(self, checkpoint_path: str = None, **kwargs):
        """Load Diffusion Policy from checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        
        try:
            from robofactory.policy.Diffusion_Policy.diffusion_policy.workspace.robotworkspace import RobotWorkspace
            from robofactory.policy.Diffusion_Policy.diffusion_policy.env_runner.dp_runner import DPRunner
            import torch
            import dill
            
            # Load checkpoint
            payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
            cfg = payload['cfg']
            
            # Create workspace and load weights
            workspace = RobotWorkspace(cfg)
            workspace.load_payload(payload)
            
            self.policy = workspace.model
            self.policy.eval()
            
            # Create runner for observation history management
            self.runner = DPRunner(
                output_dir=None,
                n_obs_steps=self.n_obs_steps,
            )
        except ImportError as e:
            raise ImportError(f"Diffusion Policy not available: {e}")
    
    def get_action(self, observation=None):
        """Get action from Diffusion Policy."""
        if self.policy is None:
            self.load_policy()
        
        if observation is not None:
            self.runner.update_obs(observation)
        
        action = self.runner.get_action(self.policy)
        return action


def run_evaluation(
    policy_type: str,
    checkpoints: List[str],
    config: EvalConfig,
    output_path: Optional[str] = None,
    **wrapper_kwargs,
) -> EvalResult:
    """
    Run evaluation with specified configuration.
    
    Args:
        policy_type: Type of policy to evaluate
        checkpoints: List of checkpoint paths (one per agent)
        config: Evaluation configuration
        output_path: Optional path to save results
        **wrapper_kwargs: Additional arguments for policy wrappers
        
    Returns:
        Evaluation results
    """
    # Create policy wrappers for each agent
    policy_wrappers = []
    for i, ckpt in enumerate(checkpoints):
        # Determine device for this agent
        device = f'cuda:{i % 8}' if len(checkpoints) > 1 else 'cuda:0'
        
        wrapper = create_policy_wrapper(
            policy_type=policy_type,
            task_name=config.task_name,
            checkpoint_path=ckpt,
            agent_id=i,
            device=device,
            **wrapper_kwargs,
        )
        wrapper.load_policy()
        policy_wrappers.append(wrapper)
    
    # Create evaluator
    evaluator = BaseEvaluator(config, policy_wrappers)
    
    # Run evaluation
    result = evaluator.run_evaluation()
    
    # Save results if path provided
    if output_path:
        evaluator.save_results(result, output_path)
    
    return result


def main():
    """Main entry point for unified evaluation."""
    parser = argparse.ArgumentParser(description="Unified VLA Policy Evaluation")
    
    # Policy arguments
    parser.add_argument('--policy_type', type=str, default='auto',
                       choices=['auto', 'openvla', 'pi0', 'diffusion'],
                       help='Policy type (auto-detected if not specified)')
    parser.add_argument('--checkpoint', '--checkpoints', type=str, nargs='+',
                       required=True, help='Checkpoint path(s), one per agent')
    
    # Task/Environment arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Task configuration YAML file')
    parser.add_argument('--task_name', type=str, default=None,
                       help='Task name (inferred from config if not provided)')
    parser.add_argument('--obs_mode', type=str, default='rgbd',
                       help='Observation mode')
    parser.add_argument('--control_mode', type=str, default='pd_joint_delta_pos',
                       help='Control mode')
    
    # Evaluation arguments
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=300,
                       help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output arguments
    parser.add_argument('--record_dir', type=str, default='eval_records/{env_id}',
                       help='Directory to save evaluation videos')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results JSON')
    parser.add_argument('--no_video', action='store_true',
                       help='Disable video recording')
    
    # Debug arguments
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Auto-detect policy type if needed
    policy_type = args.policy_type
    if policy_type == 'auto':
        policy_type = detect_policy_type(args.checkpoint[0])
        if policy_type == 'unknown':
            print("Could not auto-detect policy type. Please specify with --policy_type")
            return
        print(f"Auto-detected policy type: {policy_type}")
    
    # Get task name from config or argument
    task_name = args.task_name
    if task_name is None and args.config:
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
            task_name = config_data.get('task_name', '')
    
    if not task_name:
        print("Task name not specified. Use --task_name or provide a config file.")
        return
    
    # Create evaluation config
    eval_config = EvalConfig(
        config_path=args.config or '',
        task_name=task_name,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        record_dir=args.record_dir,
        save_video=not args.no_video,
        verbose=args.verbose,
        debug=args.debug,
    )
    
    # Run evaluation
    result = run_evaluation(
        policy_type=policy_type,
        checkpoints=args.checkpoint,
        config=eval_config,
        output_path=args.output,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Task: {result.task_name}")
    print(f"Policy Type: {policy_type}")
    print(f"Episodes: {result.num_episodes}")
    print(f"Success Rate: {result.success_rate:.2%}")
    print(f"Mean Reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}")
    print(f"Mean Steps: {result.mean_steps:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

