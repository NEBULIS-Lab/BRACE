"""
Centralized task instruction mappings for all VLA policies.

This module provides language instructions for RoboFactory tasks in different
formats suitable for different VLA architectures:
- DETAILED: Full descriptive instructions (Pi0-style)
- SIMPLE: Concise verb phrase instructions (OpenVLA-style)

Consolidated from:
- Pi0 task_instructions.py: Detailed descriptive format
- OpenVLA task_instructions.py: Simple verb phrase format
"""

from typing import Dict, Optional, Literal


# Detailed task instructions (Pi0-style)
# Full descriptive sentences for complex multi-agent tasks
TASK_INSTRUCTIONS_DETAILED: Dict[str, str] = {
    'LiftBarrier-rf': 'Lift the barrier together with the other robot',
    'TwoRobotsStackCube-rf': 'Stack the cubes together with the other robot',
    'ThreeRobotsStackCube-rf': 'Stack the cubes together with the other robots',
    'CameraAlignment-rf': 'Align the camera with the target object',
    'LongPipelineDelivery-rf': 'Pass the object along the robot chain',
    'TakePhoto-rf': 'Take a photo of the target object',
    'PassShoe-rf': 'Pass the shoe to the other robot',
    'PlaceFood-rf': 'Place the food on the plate',
    'StackCube-rf': 'Stack the cube on top of the other cube',
    'StrikeCube-rf': 'Strike the cube to the target location',
    'PickMeat-rf': 'Pick up the meat from the grill',
}


# Simple task instructions (OpenVLA-style)
# Concise verb phrases compatible with OpenVLA prompt format:
#   "What action should the robot take to {instruction}?"
# Key principles:
#   1. Lowercase - Use lowercase (the template capitalizes the sentence)
#   2. Simple verb phrase - Start with verb infinitive: "pick up", "place", "lift"
#   3. Concise - Avoid extra context like "together with the other robot"
#   4. Object-focused - Mention the main object: "the cube", "the barrier"
TASK_INSTRUCTIONS_SIMPLE: Dict[str, str] = {
    'LiftBarrier-rf': 'lift the barrier',
    'TwoRobotsStackCube-rf': 'stack the cube',
    'ThreeRobotsStackCube-rf': 'stack the cubes',
    'StackCube-rf': 'stack the cube',
    'TakePhoto-rf': 'take a photo',
    'PassShoe-rf': 'pass the shoe',
    'PlaceFood-rf': 'place the food on the plate',
    'CameraAlignment-rf': 'align the camera',
    'LongPipelineDelivery-rf': 'deliver the object',
    'StrikeCube-rf': 'strike the cube',
    'PickMeat-rf': 'pick up the meat',
}


# Global view instructions for observation-only data
GLOBAL_VIEW_INSTRUCTIONS: Dict[str, str] = {
    task: f'observe the {task.replace("-rf", "").replace("_", " ").lower()} task'
    for task in TASK_INSTRUCTIONS_SIMPLE.keys()
}


PolicyType = Literal['detailed', 'simple', 'pi0', 'openvla', 'auto']


def get_task_instruction(
    task_name: str,
    policy_type: PolicyType = 'auto',
    is_global_view: bool = False,
) -> str:
    """
    Get language instruction for a task.
    
    Args:
        task_name: Task name (e.g., 'LiftBarrier-rf')
        policy_type: Instruction format to use:
            - 'detailed' or 'pi0': Full descriptive instructions
            - 'simple' or 'openvla': Concise verb phrase instructions
            - 'auto': Returns simple format by default
        is_global_view: If True, return observation-style instruction for global camera
        
    Returns:
        Language instruction string
        
    Examples:
        >>> get_task_instruction('LiftBarrier-rf', 'detailed')
        'Lift the barrier together with the other robot'
        >>> get_task_instruction('LiftBarrier-rf', 'simple')
        'lift the barrier'
        >>> get_task_instruction('LiftBarrier-rf', is_global_view=True)
        'observe the liftbarrier task'
    """
    # Handle global view
    if is_global_view:
        return GLOBAL_VIEW_INSTRUCTIONS.get(
            task_name,
            f'observe the {task_name.replace("-rf", "").replace("_", " ").lower()} task'
        )
    
    # Normalize policy type
    if policy_type in ('detailed', 'pi0'):
        instructions = TASK_INSTRUCTIONS_DETAILED
        default_fn = lambda t: f"Complete the {t.replace('-rf', '')} task"
    else:  # 'simple', 'openvla', 'auto'
        instructions = TASK_INSTRUCTIONS_SIMPLE
        default_fn = lambda t: t.replace('-rf', '').replace('_', ' ').lower()
    
    return instructions.get(task_name, default_fn(task_name))


def get_all_task_instructions(
    policy_type: PolicyType = 'simple',
) -> Dict[str, str]:
    """
    Get all task instructions in the specified format.
    
    Args:
        policy_type: Instruction format to use
        
    Returns:
        Dictionary mapping task names to instructions
    """
    if policy_type in ('detailed', 'pi0'):
        return TASK_INSTRUCTIONS_DETAILED.copy()
    return TASK_INSTRUCTIONS_SIMPLE.copy()


def get_supported_tasks() -> list:
    """
    Get list of all supported task names.
    
    Returns:
        List of task name strings
    """
    return list(TASK_INSTRUCTIONS_SIMPLE.keys())


if __name__ == "__main__":
    # Print all instructions for verification
    print("Task Instructions - Comparison")
    print("=" * 80)
    for task in TASK_INSTRUCTIONS_SIMPLE.keys():
        print(f"\n{task}:")
        print(f"  Simple:   '{TASK_INSTRUCTIONS_SIMPLE[task]}'")
        print(f"  Detailed: '{TASK_INSTRUCTIONS_DETAILED.get(task, 'N/A')}'")
        print(f"  Global:   '{GLOBAL_VIEW_INSTRUCTIONS[task]}'")

