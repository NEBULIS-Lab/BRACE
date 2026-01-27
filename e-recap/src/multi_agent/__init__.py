"""
Cooperative Multi-Agent Planning Module for E-RECAP.

This module implements cooperative multi-agent planning with E-RECAP token pruning
for context management in multi-agent replanning scenarios.
"""

from .cooperative_planner import (
    CooperativeMultiAgentPlanner,
    create_planner,
)
from .context_buffer import (
    SharedPlanningContextBuffer,
    AgentContribution,
)
from .structured_output import (
    StructuredAgentOutput,
    build_structured_prompt,
)
from .agent_config import (
    AgentConfig,
    load_agent_configs,
    DEFAULT_AGENT_CONFIGS,
)
from .task_definitions import (
    define_cooperative_planning_steps,
    define_embodied_replanning_steps,
    get_task_steps,
)

__all__ = [
    "CooperativeMultiAgentPlanner",
    "create_planner",
    "SharedPlanningContextBuffer",
    "AgentContribution",
    "StructuredAgentOutput",
    "build_structured_prompt",
    "AgentConfig",
    "load_agent_configs",
    "DEFAULT_AGENT_CONFIGS",
    "define_cooperative_planning_steps",
    "define_embodied_replanning_steps",
    "get_task_steps",
]

