"""
Optional Framework Support for CrewAI/LangChain.

This module provides optional integration with CrewAI and LangChain frameworks.
When enabled, these frameworks are used strictly as scheduling/role-assignment
layers, while all prompt construction, context buffering, and pruning decisions
remain under explicit control of the E-RECAP pipeline.
"""

from typing import Optional, Dict, Any, List
import os

# Optional imports - will be None if frameworks are not installed
try:
    from crewai import Agent, Task, Crew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Agent = None
    Task = None
    Crew = None

try:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LLM = None
    CallbackManagerForLLMRun = None


def is_crewai_available() -> bool:
    """Check if CrewAI is available."""
    return CREWAI_AVAILABLE


def is_langchain_available() -> bool:
    """Check if LangChain is available."""
    return LANGCHAIN_AVAILABLE


def load_agents_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load agent configuration from JSON file.
    
    This function loads the agents_config.json file from the framework_optional
    directory if it exists. The file is not tracked in Git.
    
    Args:
        config_path: Optional path to config file. If None, uses default location.
    
    Returns:
        Dictionary with agent configurations.
    """
    if config_path is None:
        # Default location: framework_optional/agents_config.json
        optional_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "framework_optional"
        )
        config_path = os.path.join(optional_dir, "agents_config.json")
    
    if not os.path.exists(config_path):
        return {}
    
    import json
    with open(config_path, 'r') as f:
        return json.load(f)


def create_crewai_agent(
    agent_config: Dict[str, Any],
    llm: Any,
    device_id: int = 0,
) -> Optional[Any]:
    """
    Create a CrewAI Agent from configuration.
    
    This is an optional wrapper that allows using CrewAI Agent objects
    while maintaining E-RECAP control over context and pruning.
    
    Args:
        agent_config: Agent configuration dictionary.
        llm: Language model instance (should be E-RECAP compatible).
        device_id: GPU device ID.
    
    Returns:
        CrewAI Agent instance, or None if CrewAI is not available.
    """
    if not CREWAI_AVAILABLE:
        return None
    
    return Agent(
        name=agent_config.get("name", ""),
        role=agent_config.get("role", ""),
        goal=agent_config.get("goal", ""),
        backstory=agent_config.get("backstory", ""),
        llm=llm,
        function_calling_llm=llm,
        verbose=False,  # Disable automatic logging
        allow_delegation=False,  # Disable automatic delegation
    )


def create_crewai_task(
    agent: Any,
    description: str,
    expected_output: str = "Structured output in JSON format",
) -> Optional[Any]:
    """
    Create a CrewAI Task from description.
    
    Note: The actual prompt construction and context pruning are handled
    by E-RECAP, not by CrewAI. This function is only for compatibility.
    
    Args:
        agent: CrewAI Agent instance.
        description: Task description (may be overridden by E-RECAP).
        expected_output: Expected output format.
    
    Returns:
        CrewAI Task instance, or None if CrewAI is not available.
    """
    if not CREWAI_AVAILABLE:
        return None
    
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )


class ERECAPLangChainWrapper:
    """
    Wrapper to make E-RECAP model compatible with LangChain interface.
    
    This allows using E-RECAP models with LangChain while maintaining
    full control over pruning and context management.
    """
    
    def __init__(self, model, tokenizer, pruning_modules, keep_ratio: float = 0.7):
        """
        Initialize the wrapper.
        
        Args:
            model: E-RECAP model instance.
            tokenizer: Tokenizer instance.
            pruning_modules: Pruning modules dictionary.
            keep_ratio: Token keep ratio for pruning.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.pruning_modules = pruning_modules
        self.keep_ratio = keep_ratio
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Generate response using E-RECAP with pruning.
        
        Note: This wrapper applies E-RECAP pruning to the input prompt
        before generation, ensuring context control.
        
        Args:
            prompt: Input prompt.
            **kwargs: Additional arguments (ignored for compatibility).
        
        Returns:
            Generated text.
        """
        # Import here to avoid circular dependencies
        from inference_erecap import prune_context_only
        
        # Prune context before generation
        pruned_prompt, _ = prune_context_only(
            model=self.model,
            tokenizer=self.tokenizer,
            pruning_modules=self.pruning_modules,
            input_text=prompt,
            keep_ratio=self.keep_ratio,
        )
        
        # Generate with pruned prompt
        inputs = self.tokenizer(pruned_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        
        with self.model.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )
        
        generated_ids = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def get_framework_status() -> Dict[str, bool]:
    """Get status of optional frameworks."""
    return {
        "crewai_available": CREWAI_AVAILABLE,
        "langchain_available": LANGCHAIN_AVAILABLE,
    }

