"""
Cooperative Multi-Agent Planner with E-RECAP Integration.

This module implements the cooperative multi-agent planning setting where multiple
agents operate sequentially, each receiving a shared planning context that has been
pruned by E-RECAP's cost-aware token pruning module.
"""

import time
import random
import torch
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from .context_buffer import SharedPlanningContextBuffer, AgentContribution
from .structured_output import StructuredAgentOutput, build_structured_prompt
from .agent_config import AgentConfig, load_agent_configs
from .task_definitions import get_task_steps

# Import E-RECAP inference functions
# Note: This assumes cooperative_planner.py is in src/multi_agent/
# and inference_erecap.py is in src/
import sys
import os
# Add src/ to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from inference_erecap import load_model_and_pruners, prune_context_only
from inference_erecap import prune_context_only_ids, MIN_HEAD_TOKENS, MIN_TAIL_RATIO


def _base_keep_indices(seq_len: int, *, min_head_tokens: int, min_tail_ratio: float) -> List[int]:
    seq_len = max(0, int(seq_len))
    base_keep = set(range(min(int(min_head_tokens), seq_len)))
    min_tail_tokens = max(16, int(seq_len * float(min_tail_ratio)))
    for i in range(max(0, seq_len - min_tail_tokens), seq_len):
        base_keep.add(i)
    return sorted(base_keep)


def _cap_to_budget(
    *,
    input_ids: List[int],
    token_budget: int,
    min_head_tokens: int,
    min_tail_ratio: float,
    rng: random.Random,
    method: str,
    summary_head_tokens: int,
    summary_tail_tokens: int,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Tuple[List[int], Dict]:
    """Apply a budget cap to `input_ids` using a given strategy.

    Returns:
      pruned_ids: list of token ids with length == token_budget (when token_budget < len(input_ids))
      stats: dict with selection metadata (summary tokens, etc.)
    """
    seq_len = len(input_ids)
    token_budget = int(token_budget)
    if token_budget <= 0 or token_budget >= seq_len:
        return list(input_ids), {"budget_binding": False, "budget": token_budget}

    base_keep = _base_keep_indices(seq_len, min_head_tokens=min_head_tokens, min_tail_ratio=min_tail_ratio)
    if token_budget < len(base_keep):
        raise ValueError(
            f"token_budget ({token_budget}) < mandatory head/tail tokens ({len(base_keep)}); "
            "increase budget or reduce tail/head constraints."
        )
    remaining = token_budget - len(base_keep)
    prunable = [i for i in range(seq_len) if i not in set(base_keep)]

    method = str(method or "recency").strip().lower()
    summary_tokens = 0
    summary_time_s = 0.0

    if method in ("recency", "recent", "tail"):
        chosen = prunable[-remaining:] if remaining > 0 else []
        kept = sorted(base_keep + chosen)
        pruned = [input_ids[i] for i in kept]
        return pruned, {"budget_binding": True, "budget": token_budget, "method": method}

    if method in ("random", "rand"):
        chosen = rng.sample(prunable, k=remaining) if remaining > 0 else []
        kept = sorted(base_keep + chosen)
        pruned = [input_ids[i] for i in kept]
        return pruned, {"budget_binding": True, "budget": token_budget, "method": method}

    if method in ("structured_summary", "summary", "head_tail_summary"):
        # Deterministic: keep larger head/tail regions, and fill the middle with summary placeholders.
        head_keep = max(int(min_head_tokens), int(summary_head_tokens))
        head_keep = min(head_keep, seq_len, token_budget)
        tail_keep = max(max(16, int(seq_len * float(min_tail_ratio))), int(summary_tail_tokens))
        tail_keep = min(tail_keep, max(0, seq_len - head_keep), max(0, token_budget - head_keep))
        summary_tokens = max(0, token_budget - head_keep - tail_keep)

        summary_ids: List[int] = []
        if summary_tokens > 0:
            t0 = time.perf_counter()
            if tokenizer is not None:
                template = "\n[SUMMARY]\n"
                template_ids = tokenizer.encode(template, add_special_tokens=False)
            else:
                template_ids = []
            if not template_ids:
                # Fallback to a stable token id if encoding fails.
                template_ids = [input_ids[min_head_tokens - 1] if seq_len > 0 else 0]
            reps = (summary_tokens + len(template_ids) - 1) // len(template_ids)
            summary_ids = (template_ids * reps)[:summary_tokens]
            summary_time_s = float(time.perf_counter() - t0)

        pruned = list(input_ids[:head_keep]) + summary_ids + (list(input_ids[-tail_keep:]) if tail_keep > 0 else [])
        return pruned, {
            "budget_binding": True,
            "budget": token_budget,
            "method": method,
            "summary_tokens": int(summary_tokens),
            "summary_time_s": float(summary_time_s),
            "head_keep": int(head_keep),
            "tail_keep": int(tail_keep),
        }

    raise ValueError(f"Unknown context_compress_method: {method}")


class CooperativeMultiAgentPlanner:
    """
    Cooperative multi-agent planner with E-RECAP token pruning.
    
    Implements the cooperative multi-agent setting where:
    - Multiple agents operate sequentially
    - Each agent receives a pruned shared planning context
    - Agent outputs are structured (observations, conflicts, plan patches)
    - Context accumulates over time and is pruned before each agent invocation
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        pruning_modules: torch.nn.ModuleDict,
        keep_ratio: float = 0.7,
        prune_layers: Optional[List[int]] = None,
        max_new_tokens: int = 128,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the cooperative multi-agent planner.
        
        Args:
            model: Language model instance.
            tokenizer: Tokenizer instance.
            pruning_modules: Dictionary of pruning modules.
            keep_ratio: Fraction of tokens to keep per layer during pruning.
            prune_layers: List of layer indices to prune. If None, uses default.
            max_new_tokens: Maximum number of tokens to generate per agent.
            device: Device to run inference on. If None, uses model's device.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.pruning_modules = pruning_modules
        self.keep_ratio = keep_ratio
        self.prune_layers = prune_layers
        self.max_new_tokens = max_new_tokens
        self.device = device or next(model.parameters()).device
        
        self.context_buffer = SharedPlanningContextBuffer()
        self.agents: List[AgentConfig] = []
        self.planning_history: List[Dict] = []
    
    def add_agent(self, agent_config: AgentConfig):
        """Add an agent configuration to the planner."""
        self.agents.append(agent_config)
    
    def load_agents_from_config(self, config_path: Optional[str] = None):
        """Load agent configurations from file or use defaults."""
        self.agents = load_agent_configs(config_path)
    
    def _prune_context(
        self,
        context_text: str,
        use_pruning: bool = True,
        *,
        context_compress_method: Optional[str] = None,
        token_budget: Optional[int] = None,
        random_seed: Optional[int] = None,
        summary_head_tokens: int = 40,
        summary_tail_tokens: int = 80,
    ) -> Tuple[str, Dict]:
        """
        Prune context using E-RECAP token pruning, or return original if baseline.
        
        Args:
            context_text: Full context text to be pruned.
            use_pruning: If True, apply E-RECAP pruning. If False, return original (baseline).
        
        Returns:
            pruned_text: Pruned context text (or original if baseline).
            pruning_stats: Pruning statistics (empty dict if baseline).
        """
        method = (
            str(context_compress_method).strip().lower()
            if context_compress_method is not None
            else ("erecap" if use_pruning else "none")
        )
        if method in ("", "baseline"):
            method = "none"

        token_budget_int = int(token_budget) if token_budget is not None and int(token_budget) > 0 else None
        seed = int(random_seed) if random_seed is not None else 0
        rng = random.Random(seed)

        # Always compute input token length for accounting.
        try:
            tok = self.tokenizer(context_text, add_special_tokens=False)
            input_ids = tok.get("input_ids", [])
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            input_ids_list = [int(x) for x in input_ids]
        except Exception:
            input_ids_list = []

        input_tokens = int(len(input_ids_list)) if input_ids_list else None

        if method in ("none", "noprune", "full"):
            return context_text, {
                "context_compress_method": method,
                "token_budget": token_budget_int,
                "random_seed": seed,
                "pruning_applied": False,
                "original_length_chars": len(context_text),
                "input_tokens": input_tokens,
                "output_tokens": input_tokens,
            }

        if method in ("erecap", "e-recap", "prune"):
            pruned_input_ids, pruning_stats = prune_context_only_ids(
                model=self.model,
                tokenizer=self.tokenizer,
                pruning_modules=self.pruning_modules,
                input_text=context_text,
                keep_ratio=self.keep_ratio,
                prune_layers=self.prune_layers,
                token_select_strategy="erecap",
                random_seed=seed,
            )
            pruned_ids_list = [int(x) for x in pruned_input_ids[0].detach().cpu().tolist()]

            cap_meta: Dict = {}
            if token_budget_int is not None:
                pruned_ids_list, cap_meta = _cap_to_budget(
                    input_ids=pruned_ids_list,
                    token_budget=token_budget_int,
                    min_head_tokens=int(MIN_HEAD_TOKENS),
                    min_tail_ratio=float(MIN_TAIL_RATIO),
                    rng=rng,
                    method="recency",
                    summary_head_tokens=int(summary_head_tokens),
                    summary_tail_tokens=int(summary_tail_tokens),
                    tokenizer=self.tokenizer,
                )

            pruned_text = self.tokenizer.decode(pruned_ids_list, skip_special_tokens=True)
            pruning_stats = dict(pruning_stats or {})
            pruning_stats.update(
                {
                    "context_compress_method": "erecap",
                    "token_budget": token_budget_int,
                    "random_seed": seed,
                    "budget_cap": cap_meta,
                    "output_tokens": int(len(pruned_ids_list)),
                }
            )
            return pruned_text, pruning_stats

        # Paper baselines (progressive layer-wise pruning; token-count matched by construction).
        if method in ("random_layerwise", "recency_layerwise"):
            token_select_strategy = "random" if method == "random_layerwise" else "recency"
            pruned_input_ids, pruning_stats = prune_context_only_ids(
                model=self.model,
                tokenizer=self.tokenizer,
                pruning_modules=self.pruning_modules,
                input_text=context_text,
                keep_ratio=self.keep_ratio,
                prune_layers=self.prune_layers,
                token_select_strategy=token_select_strategy,
                random_seed=seed,
            )
            pruned_ids_list = [int(x) for x in pruned_input_ids[0].detach().cpu().tolist()]
            pruned_text = self.tokenizer.decode(pruned_ids_list, skip_special_tokens=True)
            pruning_stats = dict(pruning_stats or {})
            pruning_stats.update(
                {
                    "context_compress_method": method,
                    "token_budget": token_budget_int,
                    "random_seed": seed,
                    "output_tokens": int(len(pruned_ids_list)),
                }
            )
            return pruned_text, pruning_stats

        # Budget-matched baselines (token-level selection on the context).
        if token_budget_int is None:
            # Without a token_budget, keep baseline semantics identical to "none".
            return context_text, {
                "context_compress_method": method,
                "token_budget": None,
                "random_seed": seed,
                "pruning_applied": False,
                "original_length_chars": len(context_text),
                "input_tokens": input_tokens,
                "output_tokens": input_tokens,
            }

        pruned_ids_list, cap_meta = _cap_to_budget(
            input_ids=input_ids_list,
            token_budget=token_budget_int,
            min_head_tokens=int(MIN_HEAD_TOKENS),
            min_tail_ratio=float(MIN_TAIL_RATIO),
            rng=rng,
            method=method,
            summary_head_tokens=int(summary_head_tokens),
            summary_tail_tokens=int(summary_tail_tokens),
            tokenizer=self.tokenizer,
        )
        pruned_text = self.tokenizer.decode(pruned_ids_list, skip_special_tokens=True)
        out_stats = {
            "context_compress_method": method,
            "token_budget": token_budget_int,
            "random_seed": seed,
            "pruning_applied": True,
            "original_length_chars": len(context_text),
            "input_tokens": input_tokens,
            "output_tokens": int(len(pruned_ids_list)),
            "budget_cap": cap_meta,
        }
        if isinstance(cap_meta, dict) and cap_meta.get("summary_tokens") is not None:
            out_stats["summary_tokens"] = cap_meta.get("summary_tokens")
            out_stats["summary_time_s"] = cap_meta.get("summary_time_s")
        return pruned_text, out_stats
    
    def _call_agent_llm(
        self,
        prompt: str,
        agent_config: AgentConfig,
    ) -> str:
        """
        Call the language model for a single agent.
        
        Note: The context has already been pruned, so we use standard generation
        without additional pruning.
        
        Args:
            prompt: Input prompt for the agent.
            agent_config: Agent configuration.
        
        Returns:
            Generated text output from the agent.
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=10,  # Ensure minimum output length
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output (exclude input tokens)
        generated_ids = outputs[0][input_ids.shape[1]:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return output_text
    
    def _build_agent_prompt(
        self,
        agent_config: AgentConfig,
        pruned_context: str,
        task_step: Optional[str] = None,
    ) -> str:
        """
        Build the prompt for an agent based on pruned context.
        
        Args:
            agent_config: Agent configuration.
            pruned_context: Pruned context from previous agents.
            task_step: Optional specific task step description.
        
        Returns:
            Formatted prompt string.
        """
        return build_structured_prompt(
            agent_role=agent_config.role,
            agent_goal=agent_config.goal,
            agent_backstory=agent_config.backstory,
            pruned_context=pruned_context,
            task_step=task_step,
        )
    
    def run_planning_cycle(
        self,
        task_description: str,
        task_steps: Optional[List[Dict[str, str]]] = None,
        task_type: str = "cooperative",
        use_pruning: bool = True,
        context_compress_method: Optional[str] = None,
        token_budget: Optional[int] = None,
        random_seed: Optional[int] = None,
        summary_head_tokens: int = 40,
        summary_tail_tokens: int = 80,
    ) -> Dict:
        """
        Execute a complete planning cycle with all agents.
        
        Args:
            task_description: Initial task description.
            task_steps: Optional list of task steps. If None, uses default.
            task_type: Type of task ("cooperative" or "embodied").
        
        Returns:
            Dictionary with planning results and statistics.
        """
        # Initialize context buffer
        self.context_buffer = SharedPlanningContextBuffer()
        self.context_buffer.set_task_description(task_description)
        self.planning_history = []
        
        # Load task steps if not provided
        if task_steps is None:
            task_steps = get_task_steps(task_type)
        
        # Ensure we have agents
        if not self.agents:
            self.load_agents_from_config()
        
        # If we don't have enough agents for all task steps, create agents dynamically from task steps
        if len(self.agents) < len(task_steps):
            # Create agent configs from task step roles
            for step_idx in range(len(self.agents), len(task_steps)):
                task_step = task_steps[step_idx]
                agent_role = task_step.get("agent_role", f"Agent {step_idx}")
                # Create a simple agent config for this role
                agent_config = AgentConfig(
                    agent_id=step_idx,
                    name=agent_role.replace(" ", ""),
                    role=agent_role,
                    goal=f"Complete the task step: {task_step.get('description', '')[:100]}",
                    backstory=f"You are a {agent_role} responsible for this planning step."
                )
                self.agents.append(agent_config)
        
        # Ensure number of agents matches task steps
        num_agents = min(len(self.agents), len(task_steps))
        
        start_time = time.time()
        total_pruning_time = 0.0
        total_inference_time = 0.0
        
        # Sequential agent execution
        for step_idx in range(num_agents):
            agent_config = self.agents[step_idx]
            task_step = task_steps[step_idx]
            
            step_start_time = time.time()
            
            # Get current context
            full_context = self.context_buffer.to_text()
            context_length_before = len(full_context)
            
            # Prune context using E-RECAP (or skip for baseline)
            prune_start = time.time()
            step_seed = int(random_seed) + int(step_idx) if random_seed is not None else None
            pruned_context, pruning_stats = self._prune_context(
                full_context,
                use_pruning=use_pruning,
                context_compress_method=context_compress_method,
                token_budget=token_budget,
                random_seed=step_seed,
                summary_head_tokens=summary_head_tokens,
                summary_tail_tokens=summary_tail_tokens,
            )
            prune_time = time.time() - prune_start
            total_pruning_time += prune_time
            
            context_length_after = len(pruned_context)
            compression_ratio = context_length_after / context_length_before if context_length_before > 0 else 1.0
            
            # Build agent prompt
            agent_prompt = self._build_agent_prompt(
                agent_config=agent_config,
                pruned_context=pruned_context,
                task_step=task_step.get("description", None),
            )
            
            # Call agent LLM
            inference_start = time.time()
            llm_output = self._call_agent_llm(agent_prompt, agent_config)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # Parse structured output
            structured_output = StructuredAgentOutput.from_llm_output(llm_output)
            
            # Create agent contribution
            contribution = AgentContribution(
                agent_id=agent_config.agent_id,
                agent_role=agent_config.role,
                observations=structured_output.observations,
                conflicts=structured_output.conflicts,
                plan_patches=structured_output.plan_patches,
            )
            
            # Update context buffer
            self.context_buffer.add_agent_contribution(contribution)
            
            step_time = time.time() - step_start_time
            
            # Record planning history
            self.planning_history.append({
                "step_id": step_idx,
                "agent_id": agent_config.agent_id,
                "agent_role": agent_config.role,
                "context_length_before": context_length_before,
                "context_length_after": context_length_after,
                "compression_ratio": compression_ratio,
                "pruning_time": prune_time,
                "inference_time": inference_time,
                "step_time": step_time,
                "pruning_stats": pruning_stats,
                "structured_output": structured_output.to_dict(),
            })
            
            print(f"[Step {step_idx}] {agent_config.role}: "
                  f"Context {context_length_before} -> {context_length_after} chars "
                  f"({compression_ratio:.2%}), "
                  f"Pruning {prune_time:.3f}s, Inference {inference_time:.3f}s")
        
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            "task_description": task_description,
            "num_agents": num_agents,
            "total_time": total_time,
            "total_pruning_time": total_pruning_time,
            "total_inference_time": total_inference_time,
            "planning_history": self.planning_history,
            "final_context_summary": self.context_buffer.get_summary(),
        }
        
        return results
    
    def get_final_context(self) -> str:
        """Get the final context buffer as text."""
        return self.context_buffer.to_text()
    
    def get_planning_summary(self) -> Dict:
        """Get a summary of the planning cycle."""
        return {
            "num_steps": len(self.planning_history),
            "total_time": sum(step["step_time"] for step in self.planning_history),
            "total_pruning_time": sum(step["pruning_time"] for step in self.planning_history),
            "total_inference_time": sum(step["inference_time"] for step in self.planning_history),
            "context_growth": [
                {
                    "step": step["step_id"],
                    "length_before": step["context_length_before"],
                    "length_after": step["context_length_after"],
                    "compression_ratio": step["compression_ratio"],
                }
                for step in self.planning_history
            ],
        }


def create_planner(
    model_path: str = "checkpoints/qwen2-7b-instruct",
    pruning_ckpt: str = "checkpoints/pruning_module.pt",
    keep_ratio: float = 0.7,
    prune_layers: Optional[List[int]] = None,
    max_new_tokens: int = 128,
) -> CooperativeMultiAgentPlanner:
    """
    Create a CooperativeMultiAgentPlanner instance with loaded model and pruners.
    
    Args:
        model_path: Path to the language model.
        pruning_ckpt: Path to the pruning module checkpoint.
        keep_ratio: Fraction of tokens to keep per layer.
        prune_layers: List of layer indices to prune. If None, uses default.
        max_new_tokens: Maximum tokens to generate per agent.
    
    Returns:
        Initialized CooperativeMultiAgentPlanner instance.
    """
    # Load model and pruning modules
    model, tokenizer, pruning_modules = load_model_and_pruners(prune_layers=prune_layers)
    
    # Create planner
    planner = CooperativeMultiAgentPlanner(
        model=model,
        tokenizer=tokenizer,
        pruning_modules=pruning_modules,
        keep_ratio=keep_ratio,
        prune_layers=prune_layers,
        max_new_tokens=max_new_tokens,
    )
    
    # Load default agents
    planner.load_agents_from_config()
    
    return planner
