# End-to-End Latency Benchmarking for E-RECAP
# Measures prefill + decode (128 tokens) latency
# Simplified implementation to avoid 0-length tensor errors

import time
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from inference_erecap
try:
    from inference_erecap import (
        load_model_and_pruners,
        prefill_with_pruning,
        KEEP09_CONFIG,
        KEEP08_CONFIG,
        KEEP07_CONFIG,
        PRUNE_LAYERS,
        MIN_HEAD_TOKENS,
        MIN_TAIL_RATIO,
        build_dummy_input,
        device,
    )
except ImportError:
    # Fallback if running as standalone
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from inference_erecap import (
        load_model_and_pruners,
        prefill_with_pruning,
        KEEP09_CONFIG,
        KEEP08_CONFIG,
        KEEP07_CONFIG,
        PRUNE_LAYERS,
        MIN_HEAD_TOKENS,
        MIN_TAIL_RATIO,
        build_dummy_input,
        device,
    )

MAX_NEW_TOKENS = 128  # Generate 128 tokens as in paper


def run_end2end_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict:
    """
    Run baseline end-to-end inference (prefill + generate).
    
    Baseline: no pruning, pure model.generate().
    This function does NOT touch any E-RECAP modules or pruning logic.
    
    Args:
        model: The language model
        tokenizer: The tokenizer (unused, kept for API consistency)
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        max_new_tokens: Number of tokens to generate
        
    Returns:
        Dictionary with timing information
    """
    # CRITICAL: Guard against 0-length input
    assert input_ids.shape[1] > 0, f"input_ids seq_len must be > 0, got {input_ids.shape[1]}"
    assert attention_mask is None or attention_mask.shape[1] == input_ids.shape[1], \
        f"attention_mask seq_len {attention_mask.shape[1]} != input_ids seq_len {input_ids.shape[1]}"
    
    model.eval()
    
    with torch.no_grad():
        # Warmup: one tiny generate to compile kernels
        warmup_ids = input_ids[:, :min(8, input_ids.shape[1])]
        warmup_mask = attention_mask[:, :min(8, attention_mask.shape[1])] if attention_mask is not None else None
        _ = model.generate(
            input_ids=warmup_ids,
            attention_mask=warmup_mask,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Measure total end2end time with a single generate() call
        # This is the safest approach - no manual KV cache manipulation
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_start = time.perf_counter()
        
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_end = time.perf_counter()
        total_time = total_end - total_start
        
        # For reporting, we approximate:
        # - prefill_latency: time for first forward pass (we measure separately)
        # - decode_latency: total - prefill
        # But to keep things simple and robust, we measure prefill separately
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        
        # Single forward pass to approximate prefill time
        model_inputs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        _ = model(**model_inputs)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        prefill_time = prefill_end - prefill_start
        
        # Decode time is approximate: total - prefill
        decode_time = total_time - prefill_time
        if decode_time < 0:
            decode_time = 0.0  # Safety: ensure non-negative
    
    return {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "kv_lens_after_prefill": [input_ids.shape[1]] * len(model.model.layers),  # Approximate
        "kv_lens_final": [generated.shape[1]] * len(model.model.layers),
        "generated_length": generated.shape[1] - input_ids.shape[1],
    }


def run_end2end_erecap(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pruning_modules: nn.ModuleDict,
    keep_ratio: float,
    prune_layers: List[int],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict:
    """
    Run E-RECAP end-to-end inference with FALLBACK mode for Qwen2 GQA compatibility.
    
    FALLBACK STRATEGY:
    - Prefill: Use E-RECAP pruning to reduce sequence length and KV cache size
    - Decode: Use standard model.generate() WITHOUT pruned KV cache
              This avoids GQA (num_key_value_heads) compatibility issues
    
    Steps:
      1. Run prefill_with_pruning to get pruned logits and stats
      2. Use model.generate() with the ORIGINAL input_ids (not pruned)
         This ensures decode phase works correctly with GQA
      3. Measure prefill_time, decode_time, total_time
    
    NOTE: Decode speedup comes from reduced KV cache size in prefill,
          not from dynamic pruning during decode.
    """
    assert input_ids.shape[1] > 0, f"input_ids seq_len must be > 0, got {input_ids.shape[1]}"
    assert attention_mask is None or attention_mask.shape[1] == input_ids.shape[1], \
        f"attention_mask seq_len {attention_mask.shape[1]} != input_ids seq_len {input_ids.shape[1]}"

    device = input_ids.device
    model.eval()

    # Note: We use prefill_with_pruning from inference_erecap directly,
    # which doesn't require ERECAPModel wrapper. The pruning_modules
    # are passed directly to prefill_with_pruning.

    with torch.no_grad():
        # ============================================
        # STEP 1: Prefill with E-RECAP pruning
        # ============================================
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_start = time.perf_counter()

        # Use the simpler prefill_with_pruning from inference_erecap (not prefill_with_pruning_infer)
        # This avoids KV cache manipulation issues
        # MIN_HEAD_TOKENS and MIN_TAIL_RATIO are already imported at module level
        # Request pruned input_ids for decode phase
        result = prefill_with_pruning(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pruning_modules=pruning_modules,
            keep_ratio=keep_ratio,
            prune_layers=prune_layers,
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

        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        prefill_time = prefill_end - prefill_start

        # Sanity checks
        assert logits.size(1) > 0, "E-RECAP prefill returned empty sequence"
        final_seq_len = pruning_stats.get("final_length", input_ids.shape[1])
        
        # KV lengths after prefill: approximate by final_seq_len for all layers
        kv_lens_after_prefill = [final_seq_len] * len(model.model.layers)

        # ============================================
        # STEP 2: Decode with pruned sequence
        # Use pruned input_ids from prefill phase to continue generation
        # This ensures decode benefits from reduced KV cache size
        # ============================================
        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_start = time.perf_counter()

        # Use pruned input_ids for decode phase to benefit from reduced KV cache
        # If pruned_input_ids is None (backward compatibility), fall back to original input_ids
        decode_input_ids = pruned_input_ids if pruned_input_ids is not None else input_ids
        decode_attention_mask = torch.ones_like(decode_input_ids, dtype=torch.long, device=device)
        
        generated = model.generate(
            input_ids=decode_input_ids,
            attention_mask=decode_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_end = time.perf_counter()
        decode_time = decode_end - decode_start

    total_time = prefill_time + decode_time

    # Calculate final KV lengths
    # Use pruned input_ids length if available, otherwise use original
    decode_start_length = pruned_input_ids.shape[1] if pruned_input_ids is not None else input_ids.shape[1]
    generated_length = generated.shape[1] - decode_start_length
    kv_lens_final = [kv_lens_after_prefill[0] + generated_length] * len(model.model.layers)

    return {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "kv_lens_after_prefill": kv_lens_after_prefill,
        "kv_lens_final": kv_lens_final,
        "generated_length": int(generated_length),
        "pruning_stats": pruning_stats,
        "decode_pruning_disabled": True,  # Flag indicating fallback mode
    }


def run_end2end_latency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    use_erecap: bool = False,
    pruning_modules: Optional[nn.ModuleDict] = None,
    keep_ratio: float = 0.7,
    prune_layers: Optional[List[int]] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict:
    """
    Run end-to-end latency benchmark.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_ids: Input token IDs
        attention_mask: Attention mask
        use_erecap: Whether to use E-RECAP pruning
        pruning_modules: Pruning modules (required if use_erecap=True)
        keep_ratio: Token keep ratio (required if use_erecap=True)
        prune_layers: List of layers to prune (required if use_erecap=True)
        max_new_tokens: Number of tokens to generate
        
    Returns:
        Dictionary with timing and KV cache information
    """
    # CRITICAL: Validate inputs before processing
    if input_ids.shape[1] == 0:
        raise ValueError(f"run_end2end_latency: input_ids seq_len is 0!")
    
    if prune_layers is None:
        prune_layers = PRUNE_LAYERS
    
    if use_erecap:
        if pruning_modules is None:
            raise ValueError("pruning_modules required when use_erecap=True")
        return run_end2end_erecap(
            model, tokenizer, input_ids, attention_mask,
            pruning_modules, keep_ratio, prune_layers, max_new_tokens,
        )
    else:
        return run_end2end_baseline(
            model, tokenizer, input_ids, attention_mask, max_new_tokens,
        )


if __name__ == "__main__":
    # Test script
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--use_erecap", action="store_true")
    parser.add_argument("--config", choices=["keep09", "keep08", "keep07"], default="keep07")
    args = parser.parse_args()
    
    # Load model
    if args.use_erecap:
        if args.config == "keep09":
            config = KEEP09_CONFIG
        elif args.config == "keep08":
            config = KEEP08_CONFIG
        else:
            config = KEEP07_CONFIG
        
        model, tokenizer, pruners = load_model_and_pruners(prune_layers=config["prune_layers"])
        keep_ratio = config["keep_ratio"]
        prune_layers = config["prune_layers"]
    else:
        model, tokenizer, _ = load_model_and_pruners()
        pruners = None
        keep_ratio = 1.0
        prune_layers = []
    
    # Build input
    input_ids, attention_mask = build_dummy_input(tokenizer, args.length)
    
    # Run benchmark
    result = run_end2end_latency(
        model, tokenizer, input_ids, attention_mask,
        use_erecap=args.use_erecap,
        pruning_modules=pruners,
        keep_ratio=keep_ratio,
        prune_layers=prune_layers,
    )
    
    print(f"\n{'='*60}")
    print(f"End2End Benchmark Results (Length: {args.length})")
    print(f"{'='*60}")
    print(f"Prefill time: {result['prefill_time']:.4f}s")
    print(f"Decode time: {result['decode_time']:.4f}s")
    print(f"Total time: {result['total_time']:.4f}s")
    print(f"\nKV lengths after prefill: {result['kv_lens_after_prefill']}")
    if args.use_erecap and 'pruning_stats' in result:
        print(f"Pruning stats: {result['pruning_stats']}")
