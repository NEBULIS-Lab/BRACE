"""
Enhanced Task Definitions for Cooperative Multi-Agent Planning.

This module provides an iterative replanning scenario (15 steps) designed to better
showcase E-RECAP's advantages in controlling context explosion in long sequences.
"""

from typing import List, Dict


def define_iterative_replanning_steps() -> List[Dict[str, str]]:
    """
    Define an iterative replanning scenario (15 steps) that accumulates context rapidly.
    
    This task simulates an embodied AI replanning scenario where the agent needs to
    continuously reference failure history and adapt plans.
    
    Returns:
        List of task step dictionaries.
    """
    return [
        {
            "step_id": 0,
            "agent_role": "Initial Planner",
            "description": "Create an initial plan for the task: navigate to the kitchen, pick up a cup, fill it with water, and bring it to the living room."
        },
        {
            "step_id": 1,
            "agent_role": "Environment Observer",
            "description": "Observe the current environment state. Check if the path to the kitchen is clear and if the cup is accessible."
        },
        {
            "step_id": 2,
            "agent_role": "Failure Detector",
            "description": "Based on observations, identify any obstacles or failures in the initial plan. Report any blocked paths or missing objects."
        },
        {
            "step_id": 3,
            "agent_role": "Replanner (Round 1)",
            "description": "Generate plan patches to address the failures detected. Modify the navigation path or object retrieval strategy. Reference the initial plan and detected failures."
        },
        {
            "step_id": 4,
            "agent_role": "Execution Monitor",
            "description": "Monitor the execution of the replanned actions. Observe if the new plan is working or if new failures occur."
        },
        {
            "step_id": 5,
            "agent_role": "Failure Analyzer",
            "description": "Analyze any new failures from the execution. Identify root causes and patterns. Reference all previous failures and replanning attempts."
        },
        {
            "step_id": 6,
            "agent_role": "Replanner (Round 2)",
            "description": "Generate new plan patches based on the failure analysis. Consider all previous replanning attempts to avoid repeating mistakes. Reference complete failure history."
        },
        {
            "step_id": 7,
            "agent_role": "Constraint Validator",
            "description": "Validate that the new replanned actions satisfy safety constraints and physical limitations. Check against all previous constraint violations."
        },
        {
            "step_id": 8,
            "agent_role": "Execution Monitor (Round 2)",
            "description": "Monitor the second round of execution. Track progress and identify any remaining issues. Reference all previous monitoring results."
        },
        {
            "step_id": 9,
            "agent_role": "Failure Pattern Analyzer",
            "description": "Analyze patterns across all failures encountered so far. Identify systemic issues that need fundamental plan changes. Reference complete failure history from steps 2, 5."
        },
        {
            "step_id": 10,
            "agent_role": "Strategic Replanner",
            "description": "Based on failure patterns, propose a fundamentally revised plan strategy. This should address root causes identified in step 9. Reference all previous plans and failures."
        },
        {
            "step_id": 11,
            "agent_role": "Constraint Validator (Round 2)",
            "description": "Validate the strategic replan against all constraints and previous violations. Ensure the new strategy doesn't repeat past mistakes. Reference all constraint checks from step 7."
        },
        {
            "step_id": 12,
            "agent_role": "Execution Monitor (Round 3)",
            "description": "Monitor the strategic replan execution. Compare results with previous attempts. Reference all monitoring results from steps 4, 8."
        },
        {
            "step_id": 13,
            "agent_role": "Success Evaluator",
            "description": "Evaluate if the current plan is successful or if further replanning is needed. Compare with all previous attempts. Reference complete execution history."
        },
        {
            "step_id": 14,
            "agent_role": "Final Planner",
            "description": "If not successful, generate final plan patches incorporating all lessons learned. If successful, document the final working plan. Reference all previous steps 0-13."
        },
    ]


def get_enhanced_task_steps() -> List[Dict[str, str]]:
    """
    Get enhanced task steps for iterative replanning scenario.
    
    Returns:
        List of task step dictionaries (15 steps).
    """
    return define_iterative_replanning_steps()

