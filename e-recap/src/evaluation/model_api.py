"""
Unified Model API for E-RECAP Evaluation

This module provides a unified inference interface that works for:
- LongBench evaluation (Phase C2)
- lm-eval harness integration (Phase D)

IMPORTANT: This is a setup/placeholder implementation.
- Does NOT execute real inference
- Does NOT load any actual model
- Does NOT require GPU
- Only defines structure, classes, and method signatures

TODO: In actual evaluation phases, implement real model loading and inference.
"""
from typing import Optional, List, Union
import json


class ModelAPI:
    """
    Unified inference API for Baseline and E-RECAP models.
    
    This class provides a consistent interface for:
    1. LongBench evaluation (via generate() method)
    2. lm-eval harness (via generate_until() method)
    
    Current implementation is a placeholder that does not perform
    actual inference. Real implementation will be added in later phases.
    
    Attributes:
        model_name: Model name or path (e.g., "checkpoints/qwen2-7b-instruct")
        pruning_module_path: Path to pruning module checkpoint (None for baseline)
        base_model: Placeholder for base model (not actually loaded)
        pruner: Placeholder for pruning module (not actually loaded)
        device: Device string (e.g., "cuda", "cpu")
        is_loaded: Whether model is "loaded" (always False in setup phase)
    """
    
    def __init__(
        self,
        model_name: str,
        pruning_module_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize ModelAPI (no actual model loading).
        
        Args:
            model_name: Model name or path
            pruning_module_path: Path to pruning module checkpoint (None for baseline)
            device: Device to use (default: "cuda")
        """
        self.model_name = model_name
        self.pruning_module_path = pruning_module_path
        self.device = device
        
        # Placeholder attributes (not actually loaded)
        self.base_model = None
        self.pruner = None
        self.tokenizer = None
        self.is_loaded = False
        
        print(f"[ModelAPI] Initialized (no model loaded)")
        print(f"  Model: {self.model_name}")
        print(f"  Pruning module: {self.pruning_module_path or 'None (baseline)'}")
        print(f"  Device: {self.device}")
    
    def load_model(self):
        """
        Prepare loading logic but do NOT actually load any model.
        
        This method is a placeholder that will be implemented in actual
        evaluation phases. It should:
        1. Load base model from self.model_name
        2. Load tokenizer
        3. If pruning_module_path is set, load pruning module
        4. Move models to self.device
        5. Set self.is_loaded = True
        
        Current implementation only prints preparation messages.
        """
        print(f"[ModelAPI] Preparing model loading (not executed)")
        print(f"  Would load model from: {self.model_name}")
        
        if self.pruning_module_path:
            print(f"  Would load pruning module from: {self.pruning_module_path}")
            print(f"  Mode: E-RECAP (with token pruning)")
        else:
            print(f"  Mode: Baseline (no pruning)")
        
        print(f"  Would use device: {self.device}")
        print(f"[ModelAPI] Model loading is disabled in setup phase.")
        
        # TODO: Actual implementation should:
        # - Load model using AutoModelForCausalLM.from_pretrained()
        # - Load tokenizer using AutoTokenizer.from_pretrained()
        # - If pruning_module_path exists, load pruning module weights
        # - Set self.base_model, self.tokenizer, self.pruner
        # - Set self.is_loaded = True
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        top_p: float = 1.0
    ) -> str:
        """
        Unified generation API for LongBench.
        
        This method is used by LongBench evaluation framework.
        Do NOT perform real inference. Just define structure.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text (placeholder string in setup phase)
            
        TODO: Actual implementation should:
        1. Tokenize prompt
        2. Run forward pass with E-RECAP pruning (if enabled)
        3. Generate tokens using model.generate() or custom generation
        4. Decode and return generated text
        """
        if not self.is_loaded:
            print(f"[ModelAPI] generate() called but model not loaded (setup phase)")
            print(f"  Prompt length: {len(prompt)} chars")
            print(f"  Max new tokens: {max_new_tokens}")
            print(f"  Temperature: {temperature}, Top-p: {top_p}")
        
        # Return placeholder output
        return "[DUMMY OUTPUT — INFERENCE DISABLED]"
    
    def generate_until(
        self,
        prompt: str,
        stop: Optional[Union[str, List[str]]] = None,
        max_new_tokens: int = 128
    ) -> str:
        """
        Required by lm-eval harness.
        
        Generate text until stop sequences are encountered.
        NO real inference in setup phase.
        
        Args:
            prompt: Input prompt text
            stop: Stop sequences (string or list of strings)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text (placeholder string in setup phase)
            
        TODO: Actual implementation should:
        1. Tokenize prompt
        2. Generate tokens until stop sequence is found
        3. Handle multiple stop sequences
        4. Return generated text (excluding stop sequences)
        """
        if not self.is_loaded:
            print(f"[ModelAPI] generate_until() called but model not loaded (setup phase)")
            print(f"  Prompt length: {len(prompt)} chars")
            print(f"  Stop sequences: {stop}")
            print(f"  Max new tokens: {max_new_tokens}")
        
        # Return placeholder output
        return "[DUMMY OUTPUT — INFERENCE DISABLED]"
    
    def set_device(self, device: str):
        """
        Optional method for future GPU assignment.
        
        Args:
            device: Device string (e.g., "cuda", "cuda:0", "cpu")
        """
        old_device = self.device
        self.device = device
        print(f"[ModelAPI] Device changed: {old_device} -> {self.device}")
        
        # TODO: If model is loaded, move it to new device
        # if self.is_loaded and self.base_model is not None:
        #     self.base_model = self.base_model.to(self.device)
    
    def is_erecap(self) -> bool:
        """
        Check if this API is configured for E-RECAP (with pruning).
        
        Returns:
            True if pruning_module_path is set, False otherwise
        """
        return self.pruning_module_path is not None
    
    def get_config(self) -> dict:
        """
        Get current configuration as dictionary.
        
        Returns:
            Dict with model configuration
        """
        return {
            "model_name": self.model_name,
            "pruning_module_path": self.pruning_module_path,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "is_erecap": self.is_erecap()
        }

