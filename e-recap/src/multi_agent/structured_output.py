"""
Structured Output Parser for Agent Responses.

This module handles parsing agent outputs into structured format (observations,
conflicts, plan patches) as required by the cooperative multi-agent setting.
"""

import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .context_buffer import AgentContribution


@dataclass
class StructuredAgentOutput:
    """
    Structured output from an agent.
    
    Ensures pruning stability and interpretability by constraining agent outputs
    to structured format rather than free-form natural language reasoning.
    """
    observations: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    plan_patches: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "observations": self.observations,
            "conflicts": self.conflicts,
            "plan_patches": self.plan_patches
        }
    
    @classmethod
    def from_llm_output(cls, llm_output: str) -> "StructuredAgentOutput":
        """
        Parse LLM output into structured format.
        
        Attempts to extract JSON structure from the output. If JSON parsing fails,
        falls back to pattern matching.
        
        Args:
            llm_output: Raw output from the language model.
        
        Returns:
            StructuredAgentOutput instance.
        """
        output = cls()
        
        # Try to extract JSON from the output (more robust pattern)
        # Look for complete JSON objects that may span multiple lines
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"observations"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_match = re.search(json_pattern, llm_output, re.DOTALL)
        if json_match:
            try:
                # Try to parse the matched JSON
                json_str = json_match.group(0)
                # Clean up common JSON issues
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                data = json.loads(json_str)
                output.observations = data.get("observations", [])
                output.conflicts = data.get("conflicts", [])
                output.plan_patches = data.get("plan_patches", [])
                if output.observations or output.conflicts or output.plan_patches:
                    return output
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, try to extract partial JSON
                pass
        
        # Fallback: pattern matching for structured content
        # Look for sections marked with keywords
        observations_pattern = r'(?:observations?|new observations?):\s*(.*?)(?=\n(?:conflicts?|plan|$))'
        conflicts_pattern = r'(?:conflicts?|failures?|errors?):\s*(.*?)(?=\n(?:plan|patches?|$))'
        patches_pattern = r'(?:plan patches?|patches?|updates?):\s*(.*?)(?=\n|$)'
        
        obs_match = re.search(observations_pattern, llm_output, re.IGNORECASE | re.DOTALL)
        if obs_match:
            obs_text = obs_match.group(1).strip()
            output.observations = [line.strip() for line in obs_text.split('\n') if line.strip()]
        
        conflict_match = re.search(conflicts_pattern, llm_output, re.IGNORECASE | re.DOTALL)
        if conflict_match:
            conflict_text = conflict_match.group(1).strip()
            output.conflicts = [line.strip() for line in conflict_text.split('\n') if line.strip()]
        
        patch_match = re.search(patches_pattern, llm_output, re.IGNORECASE | re.DOTALL)
        if patch_match:
            patch_text = patch_match.group(1).strip()
            output.plan_patches = [line.strip() for line in patch_text.split('\n') if line.strip()]
        
        # If no structured content found, treat entire output as a single observation
        if not output.observations and not output.conflicts and not output.plan_patches:
            if llm_output.strip():
                output.observations = [llm_output.strip()]
        
        return output


def build_structured_prompt(
    agent_role: str,
    agent_goal: str,
    agent_backstory: str,
    pruned_context: str,
    task_step: Optional[str] = None
) -> str:
    """
    Build a prompt that encourages structured output from the agent.
    
    Args:
        agent_role: Role of the agent.
        agent_goal: Goal of the agent.
        agent_backstory: Backstory/context for the agent.
        pruned_context: Pruned context from previous agents.
        task_step: Optional specific task step.
    
    Returns:
        Formatted prompt string.
    """
    prompt_parts = [
        f"You are a {agent_role}.",
        f"Your goal: {agent_goal}",
        f"Background: {agent_backstory}",
        "",
        "Based on the following context, provide your output in JSON format:",
        "{",
        '    "observations": ["observation1", "observation2", ...],',
        '    "conflicts": ["conflict1", "conflict2", ...],',
        '    "plan_patches": ["patch1", "patch2", ...]',
        "}",
        "",
        "Guidelines:",
        "- observations: New information or insights you've discovered (be specific and reference previous steps when relevant)",
        "- conflicts: Any conflicts, failures, or issues you've detected (provide detailed analysis)",
        "- plan_patches: Specific updates or modifications to the plan (with clear rationale)",
        "",
        "Important: When referencing previous decisions or observations, be specific about which step or agent you're referring to.",
        "",
        "Context from previous planning steps:",
        "---",
        pruned_context,
        "---",
    ]
    
    if task_step:
        prompt_parts.append("")
        prompt_parts.append(f"Current task step: {task_step}")
    
    prompt_parts.append("")
    prompt_parts.append("Your response (JSON format only):")
    
    return "\n".join(prompt_parts)

