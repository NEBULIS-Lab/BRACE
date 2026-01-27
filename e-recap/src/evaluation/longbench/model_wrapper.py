import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import E-RECAP functions from inference_erecap
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from src.inference_erecap import (
    TokenPruningModule,
    prefill_with_pruning,
    PRUNE_LAYERS,
    MIN_HEAD_TOKENS,
    MIN_TAIL_RATIO,
)

class ModelWrapper:
    """
    Unified wrapper for Baseline & E-RECAP model.

    - setup 阶段：只打印信息（real_load=False）

    - 推理阶段：加载 HuggingFace 模型并执行 generate

    """

    def __init__(
        self,
        model_name: str,
        pruning_module_path: str = None,
        mode: str = "baseline",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        device: str = None,
        keep_ratio: float = 1.0,
    ):
        self.model_name = model_name
        self.pruning_module_path = pruning_module_path
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.keep_ratio = keep_ratio
        self.tokenizer = None
        self.model = None
        self.pruning_modules = None
        # Default to cuda:0 for single GPU inference
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_model(self, real_load: bool = False):
        print(f"[Init] Preparing model loading: {self.model_name}")
        if self.pruning_module_path:
            print(f"[Init] Pruning module: {self.pruning_module_path}")
        print(f"[Init] Mode: {self.mode}")
        print(f"[Init] Keep ratio: {self.keep_ratio}")

        if not real_load:
            print("[Init] Model loading is disabled in setup stage.")
            return

        print("[Init] >>> Real model loading START <<<")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False,
        )

        # Load model on single GPU (cuda:0) - do NOT use device_map="auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=None,  # Force single GPU
        )
        
        # Move model to single GPU explicitly
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[Init] Model device: {self.device}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load pruning modules if E-RECAP mode
        if self.mode == "erecap" and self.pruning_module_path:
            print("[Init] Loading pruning modules...")
            hidden_size = self.model.config.hidden_size
            
            # Build pruning modules for selected layers
            self.pruning_modules = nn.ModuleDict(
                {str(i): TokenPruningModule(hidden_size) for i in PRUNE_LAYERS}
            )
            
            # Load trained pruning weights
            state_dict = torch.load(self.pruning_module_path, map_location="cpu")
            self.pruning_modules.load_state_dict(state_dict)
            
            # Convert to half precision and set to eval mode
            # Pruning modules stay on CPU initially, will be moved to correct device
            # dynamically in apply_token_pruning() when needed
            self.pruning_modules.half()
            self.pruning_modules.eval()
            for p in self.pruning_modules.parameters():
                p.requires_grad = False
            
            print(f"[Init] Pruning modules loaded for layers: {PRUNE_LAYERS}")
            print("[Init] Note: Pruning modules will be moved to correct device during inference")

        print("[Init] >>> Real model loading DONE <<<")

    def infer(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model is not loaded. Call load_model(real_load=True) before infer()."
            )

        self.model.eval()

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        # Move to model device (cuda:0)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            if self.mode == "erecap" and self.pruning_modules is not None:
                # E-RECAP mode: use prefill_with_pruning for prefill phase
                # Then use pruned input_ids for decode phase to benefit from reduced KV cache
                
                # Prefill with pruning and request pruned input_ids
                result = prefill_with_pruning(
                    model=self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pruning_modules=self.pruning_modules,
                    keep_ratio=self.keep_ratio,
                    prune_layers=PRUNE_LAYERS,
                    min_head_tokens=MIN_HEAD_TOKENS,
                    min_tail_ratio=MIN_TAIL_RATIO,
                    return_pruned_input_ids=True,
                )
                
                # Handle both old (2-tuple) and new (3-tuple) return formats for backward compatibility
                if len(result) == 3:
                    logits, pruning_stats, pruned_input_ids = result
                else:
                    logits, pruning_stats = result
                    pruned_input_ids = None
                
                # Ensure logits are on the correct device
                logits = logits.to(self.device)
                
                # Use pruned input_ids for decode phase if available, otherwise fall back to original
                # This ensures decode benefits from reduced KV cache size
                decode_input_ids = pruned_input_ids if pruned_input_ids is not None else input_ids
                decode_input_ids = decode_input_ids.to(self.device)
                decode_attention_mask = torch.ones_like(decode_input_ids, dtype=torch.long, device=self.device)
                
                # Generate tokens starting from pruned sequence
                # This benefits from the reduced KV cache size from prefill pruning
                output_ids = self.model.generate(
                    input_ids=decode_input_ids,
                    attention_mask=decode_attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=(self.temperature > 0),
                    temperature=self.temperature if self.temperature > 0 else 1.0,
                    top_p=self.top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                # Decode only the newly generated tokens
                # Use decode_input_ids length (pruned) instead of original input_ids length
                new_tokens = output_ids[:, decode_input_ids.shape[-1]:]
                out = self.tokenizer.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]
                
            else:
                # Baseline mode: standard generation
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=(self.temperature > 0),
                    temperature=self.temperature if self.temperature > 0 else 1.0,
                    top_p=self.top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                new_tokens = output_ids[:, input_ids.shape[-1]:]

                out = self.tokenizer.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]

        return out.strip()
