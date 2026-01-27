# E-RECAP 推理（单机单卡的完整推理脚本）

import argparse
import time
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import end2end benchmarking
try:
    from benchmark_end2end import run_end2end_latency
except ImportError:
    run_end2end_latency = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Local model and pruning ckpt
# ============================
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = os.environ.get("BRACE_E_RECAP_MODEL_DIR") or str(_REPO_ROOT / "checkpoints" / "qwen2-7b-instruct")
_DEFAULT_PRUNING_CKPT = os.environ.get("BRACE_E_RECAP_PRUNER_PATH") or str(_REPO_ROOT / "checkpoints" / "pruning_module.pt")
MODEL_PATH = _DEFAULT_MODEL_PATH
PRUNING_CKPT = _DEFAULT_PRUNING_CKPT

# ============================
# Pruning config (must match Stage2)
# ============================
MAX_NEW_TOKENS = 128

# Configuration presets
KEEP09_CONFIG = {
    "prune_layers": [4, 7, 10, 13, 16, 19, 22, 25],  # 8 layers
    "keep_ratio": 0.9,  # Keep 90% tokens per layer (applied to CURRENT sequence length, NOT cumulative)
    "min_head_tokens": 4,
    "min_tail_ratio": 0.1,  # Keep 10% of tokens at tail (as in paper)
    # NOTE: cumulative_keep_ratio is for reference only - actual pruning applies keep_ratio per layer
    # Each layer prunes based on its current sequence length, not the original length
    "cumulative_keep_ratio": 0.9 ** 8,  # ~0.43 (reference only, not used in pruning logic)
}

KEEP08_CONFIG = {
    "prune_layers": [4, 7, 10, 13, 16, 19, 22, 25],  # 8 layers
    "keep_ratio": 0.8,  # Keep 80% tokens per layer (applied to CURRENT sequence length, NOT cumulative)
    "min_head_tokens": 4,
    "min_tail_ratio": 0.1,  # Keep 10% of tokens at tail (as in paper)
    # NOTE: cumulative_keep_ratio is for reference only - actual pruning applies keep_ratio per layer
    "cumulative_keep_ratio": 0.8 ** 8,  # ~0.17 (reference only, not used in pruning logic)
}

KEEP07_CONFIG = {
    "prune_layers": [4, 7, 10, 13, 16, 19, 22, 25],  # 8 layers
    "keep_ratio": 0.7,  # Keep 70% tokens per layer (applied to CURRENT sequence length, NOT cumulative)
    "min_head_tokens": 4,
    "min_tail_ratio": 0.1,  # Keep 10% of tokens at tail (as in paper)
    # NOTE: cumulative_keep_ratio is for reference only - actual pruning applies keep_ratio per layer
    "cumulative_keep_ratio": 0.7 ** 8,  # ~0.058 (reference only, not used in pruning logic)
}

# Default config (keep07)
PRUNE_LAYERS = KEEP07_CONFIG["prune_layers"]
KEEP_RATIO = KEEP07_CONFIG["keep_ratio"]
MIN_HEAD_TOKENS = KEEP07_CONFIG["min_head_tokens"]
MIN_TAIL_RATIO = KEEP07_CONFIG["min_tail_ratio"]


# ============================
# TokenPruningModule
# ============================
class TokenPruningModule(nn.Module):
    """Small MLP that outputs an importance score per token."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),  # Use GELU as in paper (was ReLU)
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [seq_len, hidden] or [batch, seq_len, hidden]
        returns: [seq_len]
        """
        return self.scorer(hidden_states).squeeze(-1)


# ============================
# Load model + pruning modules
# ============================
def load_model_and_pruners(prune_layers=None):
    """
    Load model and pruning modules.
    
    Args:
        prune_layers: List of layer indices to prune. If None, uses PRUNE_LAYERS.
    
    Returns:
        model, tokenizer, pruning_modules
    """
    if prune_layers is None:
        prune_layers = PRUNE_LAYERS
    
    # Load Qwen2 model in float16 on GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=None,
        local_files_only=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    hidden_size = model.config.hidden_size

    # Build pruning modules for selected layers
    pruning_modules = nn.ModuleDict(
        {str(i): TokenPruningModule(hidden_size) for i in prune_layers}
    )

    # Load trained pruning weights from Stage2
    state_dict = torch.load(PRUNING_CKPT, map_location="cpu")
    pruning_modules.load_state_dict(state_dict)

    pruning_modules.to(device)
    pruning_modules.half()
    pruning_modules.eval()
    for p in pruning_modules.parameters():
        p.requires_grad = False

    return model, tokenizer, pruning_modules


# ============================
# Pruning logic
# ============================
def apply_token_pruning(
    hidden_states: torch.Tensor,
    pruning_module: nn.Module,
    keep_ratio: float,
    min_head_tokens: int = None,
    min_tail_ratio: float = None,
    layer_idx: int = -1,
    debug_log: bool = False,
    token_select_strategy: str = "erecap",
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    hidden_states: [1, seq_len, hidden]
    pruning_module: TokenPruningModule (required only when token_select_strategy="erecap")
    keep_ratio: fraction of tokens to keep (applied to CURRENT sequence length, NOT cumulative)
    min_head_tokens: minimum tokens to keep at head (default: MIN_HEAD_TOKENS)
    min_tail_ratio: ratio of tokens to keep at tail (default: MIN_TAIL_RATIO, as in paper: 10%)
    layer_idx: layer index for debug logging
    debug_log: whether to log debug information

    Returns:
        pruned_hidden_states: [1, kept_len, hidden]
        index_tensor: [kept_len] indices kept
        stats: dict with pruning statistics
    """
    import logging
    import json
    import os
    import random
    from datetime import datetime
    
    if min_head_tokens is None:
        min_head_tokens = MIN_HEAD_TOKENS
    if min_tail_ratio is None:
        min_tail_ratio = MIN_TAIL_RATIO
    
    seq_len = hidden_states.size(1)
    device = hidden_states.device
    dtype = hidden_states.dtype

    # CRITICAL FIX A: Ensure we never prune to zero length
    if seq_len == 0:
        raise ValueError(f"apply_token_pruning: input sequence length is 0!")

    # [seq_len, hidden]
    hs_flat = hidden_states.squeeze(0)

    token_select_strategy = str(token_select_strategy).strip().lower()
    if token_select_strategy not in {"erecap", "random", "recency"}:
        raise ValueError(
            f"apply_token_pruning: invalid token_select_strategy={token_select_strategy!r} "
            "(expected 'erecap'|'random'|'recency')"
        )

    scores = None
    scores_abs_sum = None
    scores_min = None
    scores_max = None
    if token_select_strategy == "erecap":
        # Ensure pruning module is on the same device as hidden_states
        # This ensures device compatibility (pruning modules start on CPU, move to GPU when needed)
        pruning_module = pruning_module.to(device=device, dtype=dtype)

        # importance scores: [seq_len]
        scores = pruning_module(hs_flat)
        scores = scores.squeeze(-1) if scores.dim() > 1 else scores

        # CRITICAL FIX D: Validate saliency scores before sorting
        scores_abs_sum = float(scores.abs().sum().item())
        scores_min = float(scores.min().item())
        scores_max = float(scores.max().item())

        if scores_abs_sum == 0 or (scores_max - scores_min) < 1e-8:
            # All scores are zero or identical - add small random noise to break ties
            if debug_log:
                print(f"[WARNING] Layer {layer_idx}: All saliency scores are zero/identical, adding noise")
            scores = scores + 1e-6 * torch.randn_like(scores)
            scores_abs_sum = float(scores.abs().sum().item())
            scores_min = float(scores.min().item())
            scores_max = float(scores.max().item())

    # Always keep the first min_head_tokens and last min_tail_ratio * seq_len tokens (as in paper)
    base_keep = set(range(min(min_head_tokens, seq_len)))
    min_tail_tokens = max(16, int(seq_len * min_tail_ratio))  # At least 16, or 10% of sequence
    for i in range(max(0, seq_len - min_tail_tokens), seq_len):
        base_keep.add(i)

    # CRITICAL FIX A: Compute keep_k with safeguards
    # Each layer prunes based on CURRENT sequence length (not cumulative)
    keep_k = int(seq_len * keep_ratio)
    # Guarantee minimum: at least 1 token
    keep_k = max(1, keep_k)
    # Ensure we don't keep all tokens (must prune at least 1)
    if keep_k >= seq_len:
        keep_k = max(1, seq_len - 1)
    # Ensure we keep at least all mandatory tokens
    keep_k = max(keep_k, len(base_keep))
    # Final safeguard: ensure keep_k < seq_len
    keep_k = min(keep_k, seq_len - 1) if seq_len > 1 else 1

    # How many tokens to keep in total
    target_keep = keep_k

    # Select tokens according to strategy, under the SAME token-count budget.
    selected = list(sorted(base_keep))
    remaining_need = max(0, int(target_keep) - len(selected))

    if remaining_need > 0:
        remaining_indices = [i for i in range(seq_len) if i not in base_keep]
        if token_select_strategy == "erecap":
            assert scores is not None
            _, sorted_idx = torch.topk(scores, k=seq_len)
            for idx in sorted_idx.tolist():
                if idx in base_keep:
                    continue
                selected.append(idx)
                if len(selected) >= target_keep:
                    break
        elif token_select_strategy == "recency":
            for idx in reversed(remaining_indices):
                selected.append(idx)
                if len(selected) >= target_keep:
                    break
        else:
            seed_base = int(random_seed) if random_seed is not None else 0
            layer_seed = seed_base + int(layer_idx + 1) * 1000003
            rng = random.Random(layer_seed)
            if remaining_need >= len(remaining_indices):
                picked = remaining_indices
            else:
                picked = rng.sample(remaining_indices, k=remaining_need)
            selected.extend(picked)

    # Final safeguard: ensure we have at least 1 token
    if len(selected) == 0:
        selected = [0]  # Keep at least the first token
    
    selected = sorted(selected)
    index_tensor = torch.tensor(
        selected,
        device=hidden_states.device,
        dtype=torch.long,
    )
    pruned_hidden = hidden_states[:, index_tensor, :]

    # CRITICAL FIX A: Final validation - ensure output is not empty
    if pruned_hidden.size(1) == 0:
        raise ValueError(f"apply_token_pruning: Output sequence length is 0! seq_len={seq_len}, keep_k={keep_k}, selected={len(selected)}")

    # Calculate statistics
    tokens_kept = len(selected)
    tokens_pruned = seq_len - tokens_kept
    pruning_ratio = tokens_pruned / seq_len if seq_len > 0 else 0.0

    stats = {
        "tokens_kept": tokens_kept,
        "tokens_pruned": tokens_pruned,
        "pruning_ratio": pruning_ratio,
        "original_length": seq_len,
        "keep_k": keep_k,
        "token_select_strategy": token_select_strategy,
    }
    if token_select_strategy == "erecap":
        stats["scores_min"] = scores_min
        stats["scores_max"] = scores_max
        stats["scores_abs_sum"] = scores_abs_sum
    if token_select_strategy == "random":
        seed_base = int(random_seed) if random_seed is not None else 0
        stats["random_seed_base"] = seed_base
        stats["random_seed_layer"] = seed_base + int(layer_idx + 1) * 1000003
    
    # CRITICAL FIX E: Debug logging
    if debug_log:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "layer": layer_idx,
            "input_seq_len": seq_len,
            "computed_keep_k": keep_k,
            "tokens_kept": tokens_kept,
            "tokens_pruned": tokens_pruned,
            "pruning_ratio": pruning_ratio,
            "token_select_strategy": token_select_strategy,
        }
        if token_select_strategy == "erecap":
            log_entry.update(
                {
                    "scores_min": scores_min,
                    "scores_max": scores_max,
                    "scores_abs_sum": scores_abs_sum,
                    "fallback_applied": bool(scores_abs_sum == 0 or (scores_max - scores_min) < 1e-8),
                }
            )
        if token_select_strategy == "random":
            log_entry.update(
                {
                    "random_seed_base": stats.get("random_seed_base"),
                    "random_seed_layer": stats.get("random_seed_layer"),
                }
            )
        os.makedirs("debug", exist_ok=True)
        with open("debug/prune_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        print(
            f"[PRUNE] Layer {layer_idx}: seq_len={seq_len}, keep_k={keep_k}, "
            f"kept={tokens_kept}, pruned={tokens_pruned}, ratio={pruning_ratio:.2%}, "
            f"strategy={token_select_strategy}"
        )

    return pruned_hidden, index_tensor, stats


# ============================
# Prefill with pruning (E-RECAP)
# ============================
def prefill_with_pruning(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,  # not used internally, kept for API symmetry
    pruning_modules: nn.ModuleDict,
    keep_ratio: float,
    prune_layers: list = None,
    min_head_tokens: int = None,
    min_tail_ratio: float = None,
    return_pruned_input_ids: bool = False,
    token_select_strategy: str = "erecap",
    random_seed: Optional[int] = None,
) -> Union[Tuple[torch.Tensor, Dict], Tuple[torch.Tensor, Dict, torch.Tensor]]:
    """
    Manual forward over Transformer layers with token pruning applied
    at selected layers. Mirrors Stage2 training:

    - Start from embed_tokens(input_ids).
    - For each layer:
        * Build position_ids explicitly.
        * Call layer(hidden_states, position_ids=..., attention_mask=None).
        * Optionally prune tokens at this layer.
    - Apply final norm + lm_head to get logits.
    
    Args:
        return_pruned_input_ids: If True, also return pruned input_ids for decode phase
    
    Returns:
        logits: Model output logits
        pruning_stats: Dict with aggregated pruning statistics
        pruned_input_ids: (Optional) Pruned input_ids [batch, pruned_seq_len] if return_pruned_input_ids=True
    """
    if prune_layers is None:
        prune_layers = PRUNE_LAYERS
    if min_head_tokens is None:
        min_head_tokens = MIN_HEAD_TOKENS
    if min_tail_ratio is None:
        min_tail_ratio = MIN_TAIL_RATIO

    # Embed tokens: [1, seq_len, hidden]
    hidden_states = model.model.embed_tokens(input_ids)
    original_length = hidden_states.size(1)
    
    # Track pruning statistics and indices for reconstructing pruned input_ids
    pruning_stats = {
        "total_pruning_steps": 0,
        "total_tokens_pruned": 0,
        "final_length": original_length,
        "layer_stats": [],
        "token_select_strategy": str(token_select_strategy).strip().lower(),
    }
    if str(token_select_strategy).strip().lower() == "random":
        pruning_stats["random_seed_base"] = int(random_seed) if random_seed is not None else 0
    
    # Track cumulative indices to reconstruct pruned input_ids
    # Start with all indices: [0, 1, 2, ..., seq_len-1]
    current_indices = torch.arange(original_length, device=input_ids.device, dtype=torch.long)

    for layer_idx, layer in enumerate(model.model.layers):
        # CRITICAL: Check sequence length before layer forward
        current_seq_len = hidden_states.size(1)
        if current_seq_len == 0:
            raise ValueError(
                f"prefill_with_pruning: hidden_states sequence length is 0 at layer {layer_idx}! "
                f"This should never happen after apply_token_pruning fixes."
            )
        
        # Build position ids: [1, seq_len]
        position_ids = torch.arange(
            0,
            current_seq_len,
            dtype=torch.long,
            device=hidden_states.device,
        ).unsqueeze(0)

        # Use internal causal mask: attention_mask=None
        try:
            outputs = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                use_cache=False,
            )
            hidden_states = outputs[0]
        except Exception as e:
            raise RuntimeError(
                f"prefill_with_pruning: Layer {layer_idx} forward failed. "
                f"Input seq_len={current_seq_len}, error={e}"
            ) from e
        
        # CRITICAL: Check sequence length after layer forward
        if hidden_states.size(1) == 0:
            raise ValueError(
                f"prefill_with_pruning: Layer {layer_idx} output sequence length is 0! "
                f"Input was {current_seq_len}."
            )

        # Apply token pruning on selected layers
        if layer_idx in prune_layers:
            pruner = pruning_modules[str(layer_idx)]
            # CRITICAL FIX B: Each layer prunes based on CURRENT sequence length
            # keep_ratio is applied per layer, NOT cumulatively
            hidden_states, kept_indices, stats = apply_token_pruning(
                hidden_states,
                pruner,
                keep_ratio,  # Applied to current seq_len, not cumulative
                min_head_tokens,
                min_tail_ratio,
                layer_idx=layer_idx,
                debug_log=False,
                token_select_strategy=token_select_strategy,
                random_seed=random_seed,
            )
            
            # CRITICAL: Verify pruning didn't produce zero-length output
            if hidden_states.size(1) == 0:
                raise ValueError(
                    f"prefill_with_pruning: After pruning at layer {layer_idx}, "
                    f"hidden_states sequence length is 0! "
                    f"Input was {current_seq_len}, stats={stats}"
                )
            
            # Update cumulative indices: map current layer's kept indices to original input_ids indices
            if return_pruned_input_ids:
                # kept_indices: [kept_len] indices relative to current sequence
                # current_indices: [current_seq_len] indices relative to original input_ids
                # Map kept_indices to original indices
                current_indices = current_indices[kept_indices]
            
            pruning_stats["total_pruning_steps"] += 1
            pruning_stats["total_tokens_pruned"] += stats["tokens_pruned"]
            pruning_stats["final_length"] = stats["tokens_kept"]
            pruning_stats["layer_stats"].append({
                "layer": layer_idx,
                **stats
            })

    # CRITICAL: Final check before output
    if hidden_states.size(1) == 0:
        raise ValueError(
            f"prefill_with_pruning: Final hidden_states sequence length is 0! "
            f"Original length was {original_length}, final_length in stats={pruning_stats.get('final_length')}"
        )

    # Final RMSNorm + LM head to get logits
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    
    # CRITICAL: Verify logits shape
    if logits.size(1) == 0:
        raise ValueError(
            f"prefill_with_pruning: Output logits sequence length is 0! "
            f"hidden_states shape was {hidden_states.shape}"
        )
    
    # Reconstruct pruned input_ids if requested
    pruned_input_ids = None
    if return_pruned_input_ids:
        # current_indices now contains the indices of original input_ids that were kept
        # Extract the pruned input_ids
        pruned_input_ids = input_ids[:, current_indices]
    
    if return_pruned_input_ids:
        return logits, pruning_stats, pruned_input_ids
    else:
        return logits, pruning_stats


# ============================
# Context pruning only (for multi-agent cooperative planning)
# ============================
def prune_context_only(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pruning_modules: nn.ModuleDict,
    input_text: str,
    keep_ratio: float = None,
    prune_layers: list = None,
    min_head_tokens: int = None,
    min_tail_ratio: float = None,
    token_select_strategy: str = "erecap",
    random_seed: Optional[int] = None,
) -> Tuple[str, Dict]:
    """
    Prune context text using E-RECAP token pruning, returning pruned text.
    
    This function is designed for cooperative multi-agent planning where the
    shared context buffer needs to be pruned before each agent invocation.
    
    Args:
        model: Language model instance.
        tokenizer: Tokenizer instance.
        pruning_modules: Dictionary of pruning modules.
        input_text: Input context text to be pruned.
        keep_ratio: Fraction of tokens to keep per layer. If None, uses KEEP_RATIO.
        prune_layers: List of layer indices to prune. If None, uses PRUNE_LAYERS.
        min_head_tokens: Minimum tokens to keep at head. If None, uses MIN_HEAD_TOKENS.
        min_tail_ratio: Ratio of tokens to keep at tail. If None, uses MIN_TAIL_RATIO.
    
    Returns:
        pruned_text: Pruned context text.
        pruning_stats: Dictionary with pruning statistics.
    """
    if keep_ratio is None:
        keep_ratio = KEEP_RATIO
    if prune_layers is None:
        prune_layers = PRUNE_LAYERS
    if min_head_tokens is None:
        min_head_tokens = MIN_HEAD_TOKENS
    if min_tail_ratio is None:
        min_tail_ratio = MIN_TAIL_RATIO
    
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=False)
    input_ids = inputs["input_ids"].to(model.device)
    input_tokens = int(input_ids.size(1))
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    # Apply E-RECAP pruning and get pruned input_ids
    logits, pruning_stats, pruned_input_ids = prefill_with_pruning(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pruning_modules=pruning_modules,
        keep_ratio=keep_ratio,
        prune_layers=prune_layers,
        min_head_tokens=min_head_tokens,
        min_tail_ratio=min_tail_ratio,
        return_pruned_input_ids=True,  # Return pruned input_ids
        token_select_strategy=token_select_strategy,
        random_seed=random_seed,
    )
    
    # Decode pruned input_ids back to text
    pruned_text = tokenizer.decode(pruned_input_ids[0], skip_special_tokens=True)

    # Add simple token accounting for downstream integrations (independent of caller cwd).
    pruning_stats = dict(pruning_stats or {})
    pruning_stats["input_tokens"] = input_tokens
    pruning_stats["output_tokens"] = int(pruned_input_ids.size(1))
    
    return pruned_text, pruning_stats


# ============================
# Context pruning only (return ids)
# ============================
def prune_context_only_ids(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pruning_modules: nn.ModuleDict,
    input_text: str,
    keep_ratio: float = None,
    prune_layers: list = None,
    min_head_tokens: int = None,
    min_tail_ratio: float = None,
    token_select_strategy: str = "erecap",
    random_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Prune context text using E-RECAP token pruning, returning pruned input_ids.

    This is the token-level companion to `prune_context_only()` for downstream
    integrations that need exact token accounting or post-pruning budget capping.

    Returns:
        pruned_input_ids: [1, kept_len] token ids after E-RECAP pruning
        pruning_stats: dict with pruning statistics + input_tokens/output_tokens
    """
    if keep_ratio is None:
        keep_ratio = KEEP_RATIO
    if prune_layers is None:
        prune_layers = PRUNE_LAYERS
    if min_head_tokens is None:
        min_head_tokens = MIN_HEAD_TOKENS
    if min_tail_ratio is None:
        min_tail_ratio = MIN_TAIL_RATIO

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=False)
    input_ids = inputs["input_ids"].to(model.device)
    input_tokens = int(input_ids.size(1))
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    _, pruning_stats, pruned_input_ids = prefill_with_pruning(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pruning_modules=pruning_modules,
        keep_ratio=keep_ratio,
        prune_layers=prune_layers,
        min_head_tokens=min_head_tokens,
        min_tail_ratio=min_tail_ratio,
        return_pruned_input_ids=True,
        token_select_strategy=token_select_strategy,
        random_seed=random_seed,
    )

    pruning_stats = dict(pruning_stats or {})
    pruning_stats["input_tokens"] = input_tokens
    pruning_stats["output_tokens"] = int(pruned_input_ids.size(1))
    return pruned_input_ids, pruning_stats


# ============================
# Baseline prefill (no pruning)
# ============================
def baseline_prefill(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Baseline reference: normal model forward using
    prepare_inputs_for_generation and full sequence.
    """
    model_inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    with torch.no_grad():
        outputs = model(**model_inputs)
    return outputs.logits


# ============================
# Timing utilities
# ============================
def measure_latency(fn, *args, warmup: int = 1, runs: int = 3) -> float:
    """
    Measure average latency of fn(*args).
    """
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = fn(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


def build_dummy_input(tokenizer: AutoTokenizer, length: int):
    """
    Build a fake long sequence of given length by repeating a short prompt.
    """
    base_ids = tokenizer("Hello, this is a test.", return_tensors="pt")[
        "input_ids"
    ][0]
    if base_ids.size(0) >= length:
        ids = base_ids[:length]
    else:
        repeat = (length + base_ids.size(0) - 1) // base_ids.size(0)
        ids = base_ids.repeat(repeat)[:length]

    input_ids = ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return input_ids, attention_mask


# ============================
# Profiling
# ============================
def get_hardware_info():
    """Collect hardware information."""
    info = {
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["gpu_model"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["cuda_version"] = torch.version.cuda
    
    return info


def profile_lengths(lengths, config_name: str = "keep07", save_json: bool = True, benchmark_mode: str = "prefill"):
    """
    Profile baseline vs E-RECAP for given sequence lengths.
    
    Args:
        lengths: List of sequence lengths to test
        config_name: Configuration name ("keep09", "keep08", or "keep07")
        save_json: Whether to save results to JSON
        benchmark_mode: "prefill" for prefill-only, "end2end" for full E2E
    """
    # Select configuration
    if config_name == "keep09":
        config = KEEP09_CONFIG
    elif config_name == "keep08":
        config = KEEP08_CONFIG
    else:
        config = KEEP07_CONFIG
    
    print(f"[Config] Using {config_name} configuration:")
    print(f"  Prune layers: {config['prune_layers']}")
    print(f"  Keep ratio: {config['keep_ratio']}")
    print(f"  Cumulative keep ratio: {config['cumulative_keep_ratio']:.4f}")
    
    # Load model with selected configuration
    model, tokenizer, pruners = load_model_and_pruners(prune_layers=config["prune_layers"])
    model.eval()

    print("Profiling lengths:", lengths)
    
    # Store results in new format
    results_data = {
        "metadata": {
            "config_name": config_name,
            "model": os.path.basename(MODEL_PATH),
            "hardware": get_hardware_info(),
            "pruning_config": {
                "prune_layers": config["prune_layers"],
                "keep_ratio": config["keep_ratio"],
                "min_head_tokens": config["min_head_tokens"],
                "min_tail_ratio": config["min_tail_ratio"],
                "cumulative_keep_ratio": config["cumulative_keep_ratio"],
            },
            "run_info": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "script": "inference_erecap.py",
                "mode": "profile",
                "benchmark_mode": benchmark_mode,
            }
        },
        "results": {},
        "summary": {
            "tested_lengths": [],
            "avg_speedup": 0.0,
            "avg_memory_reduction": 0.0,
        }
    }
    
    speedups = []
    memory_reductions = []
    
    for L in lengths:
        # Build new input for each length
        input_ids, attention_mask = build_dummy_input(tokenizer, L)

        try:
            if benchmark_mode == "end2end":
                # End2End benchmarking
                if run_end2end_latency is None:
                    raise ImportError("benchmark_end2end module not found. Cannot run end2end benchmark.")
                
                # Reset memory tracking
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                
                # Baseline End2End - with error handling
                try:
                    baseline_result = run_end2end_latency(
                        model, tokenizer, input_ids, attention_mask,
                        use_erecap=False,
                    )
                    baseline_prefill_t = baseline_result["prefill_time"]
                    baseline_decode_t = baseline_result["decode_time"]
                    baseline_total_t = baseline_result["total_time"]
                    baseline_kv_lens = baseline_result["kv_lens_after_prefill"]
                except Exception as e:
                    print(f"[Length {L}] Baseline End2End failed: {e}")
                    raise  # Re-raise to be caught by outer try-except
                
                baseline_memory = 0.0
                if device.type == "cuda":
                    baseline_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                    torch.cuda.reset_peak_memory_stats()
                
                # E-RECAP End2End - with error handling
                try:
                    erecap_result = run_end2end_latency(
                        model, tokenizer, input_ids, attention_mask,
                        use_erecap=True,
                        pruning_modules=pruners,
                        keep_ratio=config["keep_ratio"],
                        prune_layers=config["prune_layers"],
                    )
                    erecap_prefill_t = erecap_result["prefill_time"]
                    erecap_decode_t = erecap_result["decode_time"]
                    erecap_total_t = erecap_result["total_time"]
                    erecap_kv_lens = erecap_result["kv_lens_after_prefill"]
                    pruning_stats = erecap_result.get("pruning_stats", {})
                except Exception as e:
                    print(f"[Length {L}] E-RECAP End2End failed: {e}")
                    raise  # Re-raise to be caught by outer try-except
                
                erecap_memory = 0.0
                if device.type == "cuda":
                    erecap_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                
                # Calculate speedups
                prefill_speedup = baseline_prefill_t / erecap_prefill_t if erecap_prefill_t > 0 else float("inf")
                decode_speedup = baseline_decode_t / erecap_decode_t if erecap_decode_t > 0 else float("inf")
                total_speedup = baseline_total_t / erecap_total_t if erecap_total_t > 0 else float("inf")
                
                # Calculate KV cache reduction
                if baseline_kv_lens and erecap_kv_lens:
                    baseline_kv_avg = sum(baseline_kv_lens) / len(baseline_kv_lens) if baseline_kv_lens else L
                    erecap_kv_avg = sum(erecap_kv_lens) / len(erecap_kv_lens) if erecap_kv_lens else L
                    kv_reduction = (baseline_kv_avg - erecap_kv_avg) / baseline_kv_avg if baseline_kv_avg > 0 else 0.0
                else:
                    kv_reduction = 0.0
                
                memory_reduction = (baseline_memory - erecap_memory) / baseline_memory if baseline_memory > 0 else 0.0
                
                print(
                    f"[Length {L}] End2End Results:\n"
                    f"  Baseline: prefill={baseline_prefill_t:.4f}s, decode={baseline_decode_t:.4f}s, "
                    f"total={baseline_total_t:.4f}s\n"
                    f"  E-RECAP:  prefill={erecap_prefill_t:.4f}s, decode={erecap_decode_t:.4f}s, "
                    f"total={erecap_total_t:.4f}s\n"
                    f"  Speedup:  prefill={prefill_speedup:.2f}x, decode={decode_speedup:.2f}x, "
                    f"total={total_speedup:.2f}x\n"
                    f"  KV Cache: baseline={baseline_kv_lens[0] if baseline_kv_lens else L}, "
                    f"erecap={erecap_kv_lens[0] if erecap_kv_lens else L}, "
                    f"reduction={kv_reduction:.2%}"
                )
                
                # Store detailed results
                results_data["results"][str(L)] = {
                    "baseline": {
                        "prefill_latency_seconds": baseline_prefill_t,
                        "decode_latency_seconds": baseline_decode_t,
                        "total_latency_seconds": baseline_total_t,
                        "memory_gb": baseline_memory,
                        "kv_lens_after_prefill": baseline_kv_lens,
                    },
                    "erecap": {
                        "prefill_latency_seconds": erecap_prefill_t,
                        "decode_latency_seconds": erecap_decode_t,
                        "total_latency_seconds": erecap_total_t,
                        "memory_gb": erecap_memory,
                        "kv_lens_after_prefill": erecap_kv_lens,
                        "tokens_kept": pruning_stats.get("final_length", L),
                        "tokens_pruned": pruning_stats.get("total_tokens_pruned", 0),
                        "pruning_ratio": pruning_stats.get("total_tokens_pruned", 0) / L if L > 0 else 0.0,
                        "pruning_steps": pruning_stats.get("total_pruning_steps", 0),
                    },
                    "speedup": {
                        "prefill": prefill_speedup,
                        "decode": decode_speedup,
                        "total": total_speedup,
                    },
                    "kv_reduction": kv_reduction,
                    "memory_reduction": memory_reduction,
                }
                
                speedups.append(total_speedup)
                memory_reductions.append(memory_reduction)
                results_data["summary"]["tested_lengths"].append(L)
                
            else:
                # Prefill-only benchmarking (original behavior)
                # Reset memory tracking
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                
                # Baseline: no pruning
                baseline_t = measure_latency(
                    lambda x, m: baseline_prefill(model, x, m),
                    input_ids,
                    attention_mask,
                )
                
                baseline_memory = 0.0
                if device.type == "cuda":
                    baseline_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                    torch.cuda.reset_peak_memory_stats()

                # E-RECAP: manual forward + pruning
                def erecap_fn(x, m):
                    logits, stats = prefill_with_pruning(
                        model, x, m, pruners, 
                        config["keep_ratio"],
                        config["prune_layers"],
                        config["min_head_tokens"],
                        config["min_tail_ratio"],
                    )
                    return logits, stats
                
                erecap_t = measure_latency(
                    lambda x, m: erecap_fn(x, m)[0],  # Only measure latency, not stats
                    input_ids,
                    attention_mask,
                )
                
                # Get pruning stats from a separate run
                _, pruning_stats = erecap_fn(input_ids, attention_mask)
                
                erecap_memory = 0.0
                if device.type == "cuda":
                    erecap_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB

                speedup = baseline_t / erecap_t if erecap_t > 0 else float("inf")
                memory_reduction = (baseline_memory - erecap_memory) / baseline_memory if baseline_memory > 0 else 0.0
                
                print(
                    f"[Length {L}] baseline={baseline_t:.4f}s ({baseline_memory:.2f}GB)  "
                    f"erecap={erecap_t:.4f}s ({erecap_memory:.2f}GB)  "
                    f"speedup={speedup:.2f}x  memory_reduction={memory_reduction:.2%}"
                )
                
                # Store detailed results
                results_data["results"][str(L)] = {
                    "baseline": {
                        "latency_seconds": baseline_t,
                        "memory_gb": baseline_memory,
                    },
                    "erecap": {
                        "latency_seconds": erecap_t,
                        "memory_gb": erecap_memory,
                        "tokens_kept": pruning_stats["final_length"],
                        "tokens_pruned": pruning_stats["total_tokens_pruned"],
                        "pruning_ratio": pruning_stats["total_tokens_pruned"] / L if L > 0 else 0.0,
                        "pruning_steps": pruning_stats["total_pruning_steps"],
                    },
                    "speedup": speedup,
                    "memory_reduction": memory_reduction,
                }
                
                speedups.append(speedup)
                memory_reductions.append(memory_reduction)
                results_data["summary"]["tested_lengths"].append(L)

        except torch.cuda.OutOfMemoryError:
            print(f"[Length {L}] OOM on GPU, skipping this length.")
        except Exception as e:
            print(f"[Length {L}] Error: {e}")
        finally:
            # Explicitly free tensors and clear cache to avoid accumulation
            if 'input_ids' in locals():
                del input_ids, attention_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
    # Calculate summary statistics
    if speedups:
        results_data["summary"]["avg_speedup"] = sum(speedups) / len(speedups)
    if memory_reductions:
        results_data["summary"]["avg_memory_reduction"] = sum(memory_reductions) / len(memory_reductions)
    
    # Save results to JSON
    if save_json and results_data["results"]:
        os.makedirs("results", exist_ok=True)
        
        # Save combined results (new format)
        combined_path = f"results/latency_results_{config_name}.json"
        with open(combined_path, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"[OK] Results saved to {combined_path}")
        
        # Also save backward-compatible separate files
        if benchmark_mode == "prefill":
            baseline_results = {
                k: v["baseline"]["latency_seconds"]
                for k, v in results_data["results"].items()
            }
            erecap_results = {
                k: v["erecap"]["latency_seconds"]
                for k, v in results_data["results"].items()
            }
        else:  # "end2end"
            baseline_results = {
                k: v["baseline"]["total_latency_seconds"]
                for k, v in results_data["results"].items()
            }
            erecap_results = {
                k: v["erecap"]["total_latency_seconds"]
                for k, v in results_data["results"].items()
            }
        
        baseline_path = f"results/latency_baseline_{config_name}.json"
        erecap_path = f"results/latency_erecap_{config_name}.json"
        
        with open(baseline_path, "w") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"[OK] Baseline results saved to {baseline_path}")
        
        with open(erecap_path, "w") as f:
            json.dump(erecap_results, f, indent=2)
        print(f"[OK] E-RECAP results saved to {erecap_path}")


# ============================
# Text generation (baseline)
# ============================
def generate_text(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


# ============================
# CLI
# ============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="profile",
        choices=["profile", "generate"],
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, E-RECAP!",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=None,
        help="Token keep ratio for pruning (overrides config if set)",
    )
    parser.add_argument(
        "--config",
        choices=["keep09", "keep08", "keep07"],
        default="keep07",
        help="Configuration preset: keep09 (0.9), keep08 (0.8), or keep07 (0.7) keep ratio",
    )
    parser.add_argument(
        "--benchmark_mode",
        choices=["prefill", "end2end"],
        default="prefill",
        help="Benchmark mode: 'prefill' for prefill-only, 'end2end' for full end-to-end (prefill + decode)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=_DEFAULT_MODEL_PATH,
        help="Local HuggingFace model directory (default: BRACE_E_RECAP_MODEL_DIR or checkpoints/qwen2-7b-instruct).",
    )
    parser.add_argument(
        "--pruning_ckpt",
        type=str,
        default=_DEFAULT_PRUNING_CKPT,
        help="E-RECAP pruning module checkpoint (default: BRACE_E_RECAP_PRUNER_PATH or checkpoints/pruning_module.pt).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global MODEL_PATH, PRUNING_CKPT
    MODEL_PATH = str(args.model_path)
    PRUNING_CKPT = str(args.pruning_ckpt)

    if args.mode == "profile":
        profile_lengths(
            args.lengths, 
            config_name=args.config,
            benchmark_mode=args.benchmark_mode,
        )

    elif args.mode == "generate":
        model, tokenizer, _ = load_model_and_pruners()
        model.eval()
        text = generate_text(model, tokenizer, args.prompt)
        print(text)


if __name__ == "__main__":
    main()
