"""
Agent Configuration for Cooperative Multi-Agent Planning.

This module defines agent configurations (roles, goals, backstories) used in
the cooperative multi-agent setting for planning tasks.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    agent_id: int
    name: str
    role: str
    goal: str
    backstory: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory
        }


# Default agent configurations for cooperative planning
DEFAULT_AGENT_CONFIGS = [
    AgentConfig(
        agent_id=0,
        name="ProductManager",
        role="Product Manager",
        goal="Define requirements and roadmap for the planning task",
        backstory="You are a visionary Product Manager with a background in AI. You define the features and user experience goals."
    ),
    AgentConfig(
        agent_id=1,
        name="SystemArchitect",
        role="System Architect",
        goal="Design the high-level architecture and select technology stack",
        backstory="You are a Chief System Architect with deep expertise in distributed systems and high-performance computing."
    ),
    AgentConfig(
        agent_id=2,
        name="AIResearcher",
        role="AI Researcher",
        goal="Select models and optimize training algorithms",
        backstory="You are a Lead AI Scientist specializing in large language models and parallel training techniques."
    ),
    AgentConfig(
        agent_id=3,
        name="DataEngineer",
        role="Data Engineer",
        goal="Design data ingestion, storage and preprocessing pipelines",
        backstory="You are a Senior Data Engineer expert in big data technologies, ETL pipelines, and vector databases."
    ),
    AgentConfig(
        agent_id=4,
        name="BackendDeveloper",
        role="Backend Developer",
        goal="Develop APIs and microservices for job scheduling and management",
        backstory="You are a Backend Tech Lead experienced in Go, Python, and building scalable microservices."
    ),
    AgentConfig(
        agent_id=5,
        name="FrontendDeveloper",
        role="Frontend Developer",
        goal="Create intuitive UI for monitoring training jobs and system status",
        backstory="You are a Senior Frontend Developer passionate about UX and data visualization using React and D3."
    ),
    AgentConfig(
        agent_id=6,
        name="SecuritySpecialist",
        role="Security Specialist",
        goal="Ensure system security, data privacy and compliance",
        backstory="You are a Cyber Security Expert focused on cloud infrastructure security and data protection regulations."
    ),
    AgentConfig(
        agent_id=7,
        name="DevOpsEngineer",
        role="DevOps Engineer",
        goal="Implement CI/CD pipelines and container orchestration",
        backstory="You are a DevOps Engineer specializing in Kubernetes, Docker, and automated deployment pipelines."
    ),
]


def load_agent_configs(config_path: str = None) -> List[AgentConfig]:
    """
    Load agent configurations.
    
    Args:
        config_path: Optional path to JSON configuration file.
                    If None, returns default configurations.
    
    Returns:
        List of AgentConfig instances.
    """
    if config_path is None:
        return DEFAULT_AGENT_CONFIGS.copy()
    
    import json
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    configs = []
    for agent_data in data.get("agents", []):
        configs.append(AgentConfig(
            agent_id=agent_data.get("index", len(configs)),
            name=agent_data.get("name", f"Agent{len(configs)}"),
            role=agent_data.get("role", ""),
            goal=agent_data.get("goal", ""),
            backstory=agent_data.get("backstory", "")
        ))
    
    return configs

