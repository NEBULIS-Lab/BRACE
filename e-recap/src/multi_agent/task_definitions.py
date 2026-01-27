"""
Task Definitions for Cooperative Multi-Agent Planning.

This module defines task steps for cooperative multi-agent planning scenarios
and replanning scenarios.
"""

from typing import List, Dict, Optional


def define_cooperative_planning_steps() -> List[Dict[str, str]]:
    """
    Define cooperative planning steps for multi-agent replanning.
    
    Returns:
        List of task step dictionaries with 'step_id', 'agent_role', and 'description'.
    """
    return [
        {
            "step_id": 0,
            "agent_role": "Product Manager",
            "description": "Define the core features, target audience, and user journey for the planning task."
        },
        {
            "step_id": 1,
            "agent_role": "System Architect",
            "description": "Based on the requirements, design the overall system architecture, including component interaction and technology stack."
        },
        {
            "step_id": 2,
            "agent_role": "AI Researcher",
            "description": "Propose the specific AI models and training algorithms to be supported, focusing on distributed training efficiency."
        },
        {
            "step_id": 3,
            "agent_role": "Data Engineer",
            "description": "Design the data pipeline architecture for handling large-scale datasets, including storage and preprocessing."
        },
        {
            "step_id": 4,
            "agent_role": "Backend Developer",
            "description": "Outline the API design and microservices structure to support job scheduling and resource management."
        },
        {
            "step_id": 5,
            "agent_role": "Frontend Developer",
            "description": "Design the user interface for the dashboard, focusing on monitoring and job submission workflows."
        },
        {
            "step_id": 6,
            "agent_role": "Security Specialist",
            "description": "Review the architecture and designs to identify security risks and propose mitigation strategies."
        },
        {
            "step_id": 7,
            "agent_role": "DevOps Engineer",
            "description": "Plan the deployment strategy, including containerization (Docker), orchestration (Kubernetes), and CI/CD pipelines."
        },
    ]


def define_embodied_replanning_steps() -> List[Dict[str, str]]:
    """
    Define task steps for embodied AI replanning scenario.
    
    This is an alternative task definition focused on embodied AI replanning,
    which may be more relevant to E-RECAP's core application.
    
    Returns:
        List of task step dictionaries.
    """
    return [
        {
            "step_id": 0,
            "agent_role": "Task Planner",
            "description": "Analyze the current task and break it down into sub-goals and action sequences."
        },
        {
            "step_id": 1,
            "agent_role": "Environment Observer",
            "description": "Observe the current environment state and identify any changes or obstacles."
        },
        {
            "step_id": 2,
            "agent_role": "Failure Analyzer",
            "description": "Identify any failures or conflicts in the current plan based on environment observations."
        },
        {
            "step_id": 3,
            "agent_role": "Replanner",
            "description": "Generate plan patches to address identified failures and adapt to environment changes."
        },
        {
            "step_id": 4,
            "agent_role": "Constraint Validator",
            "description": "Validate that the replanned actions satisfy all constraints and safety requirements."
        },
        {
            "step_id": 5,
            "agent_role": "Execution Monitor",
            "description": "Monitor the execution of the replanned actions and prepare for next replanning cycle."
        },
    ]


def get_task_steps(task_type: str = "iterative_replanning") -> List[Dict[str, str]]:
    """
    Get task steps based on task type.
    
    Args:
        task_type: Type of task. Options:
            - "iterative_replanning": 15-step iterative replanning (default, better for showcasing E-RECAP)
            - "embodied": 6-step embodied replanning
            - "cooperative": 8-step cooperative planning (legacy)
    
    Returns:
        List of task step dictionaries.
    """
    if task_type == "iterative_replanning":
        # Import enhanced task definitions
        try:
            from .task_definitions_enhanced import get_enhanced_task_steps
            return get_enhanced_task_steps()
        except ImportError:
            # Fallback to embodied if enhanced version not available
            return define_embodied_replanning_steps()
    elif task_type == "embodied":
        return define_embodied_replanning_steps()
    elif task_type == "cooperative":
        return define_cooperative_planning_steps()
    else:
        # Default to iterative_replanning
        try:
            from .task_definitions_enhanced import get_enhanced_task_steps
            return get_enhanced_task_steps()
        except ImportError:
            return define_embodied_replanning_steps()

