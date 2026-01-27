"""
Shared Planning Context Buffer for Cooperative Multi-Agent Planning.

This module implements a shared context buffer that accumulates task descriptions,
current plans, constraints, and agent contributions over time. The buffer provides
interfaces for E-RECAP token pruning and context reconstruction.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class AgentContribution:
    """Structured output from a single agent."""
    agent_id: int
    agent_role: str
    observations: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    plan_patches: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "observations": self.observations,
            "conflicts": self.conflicts,
            "plan_patches": self.plan_patches
        }


class SharedPlanningContextBuffer:
    """
    Shared planning context buffer for cooperative multi-agent planning.
    
    Accumulates task descriptions, current plans, constraints, and agent contributions
    over time. Provides interfaces for E-RECAP token pruning.
    """
    
    def __init__(self):
        """Initialize an empty context buffer."""
        self.task_description: str = ""
        self.current_plan: str = ""
        self.constraints: List[str] = []
        self.agent_contributions: List[AgentContribution] = []
    
    def set_task_description(self, description: str):
        """Set the initial task description."""
        self.task_description = description
    
    def update_plan(self, plan: str):
        """Update the current plan."""
        self.current_plan = plan
    
    def add_constraint(self, constraint: str):
        """Add a constraint to the buffer."""
        self.constraints.append(constraint)
    
    def add_agent_contribution(self, contribution: AgentContribution):
        """Add an agent's structured output to the buffer."""
        self.agent_contributions.append(contribution)
    
    def to_text(self) -> str:
        """
        Convert the context buffer to text format for E-RECAP pruning.
        
        Returns:
            Full context text ready for tokenization and pruning.
        """
        parts = []
        
        # Task description
        if self.task_description:
            parts.append(f"Task Description:\n{self.task_description}\n")
        
        # Current plan
        if self.current_plan:
            parts.append(f"Current Plan:\n{self.current_plan}\n")
        
        # Constraints
        if self.constraints:
            parts.append("Constraints:")
            for constraint in self.constraints:
                parts.append(f"  - {constraint}")
            parts.append("")
        
        # Agent contributions (most recent first)
        if self.agent_contributions:
            parts.append("Agent Contributions (in reverse chronological order):")
            for contrib in reversed(self.agent_contributions):
                parts.append(f"\n--- {contrib.agent_role} (Agent {contrib.agent_id}) ---")
                
                if contrib.observations:
                    parts.append("Observations:")
                    for obs in contrib.observations:
                        parts.append(f"  - {obs}")
                
                if contrib.conflicts:
                    parts.append("Detected Conflicts/Failures:")
                    for conflict in contrib.conflicts:
                        parts.append(f"  - {conflict}")
                
                if contrib.plan_patches:
                    parts.append("Plan Patches:")
                    for patch in contrib.plan_patches:
                        parts.append(f"  - {patch}")
        
        return "\n".join(parts)
    
    def get_context_length(self) -> int:
        """Get the approximate length of the context in characters."""
        return len(self.to_text())
    
    def clear_contributions(self):
        """Clear all agent contributions (keep task description and plan)."""
        self.agent_contributions = []
    
    def get_summary(self) -> Dict:
        """Get a summary of the buffer state."""
        return {
            "task_description_length": len(self.task_description),
            "current_plan_length": len(self.current_plan),
            "num_constraints": len(self.constraints),
            "num_contributions": len(self.agent_contributions),
            "total_context_length": self.get_context_length()
        }

