# Multi-GPU E-RECAP 推理（自动分片）


import argparse
import time
import json
import os
import torch
import torch.nn as nn
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================
# Config
# ============================
MODEL_PATH = "checkpoints/qwen2-7b-instruct"
PRUNING_CKPT = "checkpoints/pruning_module.pt"

PRUNE_LAYERS = [4, 7, 10, 13, 16, 19, 22, 25]
KEEP_RATIO = 0.7
MIN_HEAD_TOKENS = 4
MIN_TAIL_RATIO = 0.1  # Keep 10% of tokens at tail (as in paper)
MAX_NEW_TOKENS = 128


# ============================
# Multi-GPU helper
# ============================
def print_gpu_usage(tag=""):
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    print(f"[{tag}] GPU0: used={info.used//1024**2}MB total={info.total//1024**2}MB")


# ============================
# Pruning MLP
# ============================
class TokenPruningModule(nn.Module):
    """Small token-scoring MLP"""
    def __init__(self, hidden_size):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),  # Use GELU as in paper (was ReLU)
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states):
        return self.scorer(hidden_states).squeeze(-1)


# ============================
# Load model + pruners
# ============================
def load_model_and_pruners():
    print("[Loading model] device_map=auto ...")

    # Qwen2 model automatically split across GPUs
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,     # ensures half precision across devices
        device_map="auto",
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    hidden_size = model.config.hidden_size

    # Build pruning modules
    pruning_modules = nn.ModuleDict({
        str(i): TokenPruningModule(hidden_size)
        for i in PRUNE_LAYERS
    })

    # Load trained weights
    state_dict = torch.load(PRUNING_CKPT, map_location="cpu")
    pruning_modules.load_state_dict(state_dict)

    # *** KEY FIX: convert pruning modules to FP16 ***
    pruning_modules.half()
    pruning_modules.eval()
    for p in pruning_modules.parameters():
        p.requires_grad = False

    print("[Model + pruners loaded]")
    return model, tokenizer, pruning_modules


# ============================
# Token pruning logic
# ============================
def apply_token_pruning(hidden_states, pruning_module, keep_ratio):
    """
    hidden_states: [1, seq, hidden]
    """
    device = hidden_states.device
    dtype = hidden_states.dtype

    # ensure pruning module matches dtype
    pruning_module = pruning_module.to(dtype=dtype, device=device)

    seq_len = hidden_states.size(1)

    # flatten to [seq, hidden]
    hs_flat = hidden_states.squeeze(0)   # [seq, hidden]

    # scores: [seq]
    scores = pruning_module(hs_flat)
    scores = scores.squeeze(-1) if scores.dim() > 1 else scores
    
    # CRITICAL FIX A: Ensure we never prune to zero length
    if seq_len == 0:
        raise ValueError(f"apply_token_pruning: input sequence length is 0!")
    
    # CRITICAL FIX D: Validate saliency scores
    scores_abs_sum = scores.abs().sum().item()
    scores_min = scores.min().item()
    scores_max = scores.max().item()
    
    if scores_abs_sum == 0 or (scores_max - scores_min) < 1e-8:
        # All scores are zero or identical - add small random noise
        scores = scores + 1e-6 * torch.randn_like(scores)
        scores_min = scores.min().item()
        scores_max = scores.max().item()

    # mandatory keep: head + tail (10% of sequence, as in paper)
    base_keep = set(range(min(MIN_HEAD_TOKENS, seq_len)))
    min_tail_tokens = max(16, int(seq_len * MIN_TAIL_RATIO))  # At least 16, or 10% of sequence
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
    
    target_keep = keep_k

    # sort by score
    _, sorted_idx = torch.topk(scores, k=seq_len)

    selected = []
    for idx in sorted_idx.tolist():
        if idx in base_keep:
            selected.append(idx)
    for idx in sorted_idx.tolist():
        if idx not in base_keep and len(selected) < target_keep:
            selected.append(idx)
    
    # Final safeguard: ensure we have at least 1 token
    if len(selected) == 0:
        selected = [0]  # Keep at least the first token

    selected = sorted(selected)
    index_tensor = torch.tensor(selected, dtype=torch.long, device=device)

    pruned_hidden = hidden_states[:, index_tensor, :]
    
    # CRITICAL FIX A: Final validation - ensure output is not empty
    if pruned_hidden.size(1) == 0:
        raise ValueError(f"apply_token_pruning: Output sequence length is 0! seq_len={seq_len}, keep_k={keep_k}, selected={len(selected)}")

    return pruned_hidden, index_tensor


# ============================
# Manual E-RECAP forward
# ============================
def prefill_with_pruning(model, input_ids, attention_mask, pruners, keep_ratio):
    """
    Manual forward with pruning at selected layers.
    Supports multi-GPU (device_map=auto)
    """

    # embed tokens will be on the correct GPU automatically
    hidden_states = model.model.embed_tokens(input_ids)

    for layer_idx, layer in enumerate(model.model.layers):

        device = hidden_states.device
        seq_len = hidden_states.size(1)

        # build position ids
        position_ids = torch.arange(
            0, seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)

        # transformer block forward
        outputs = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = outputs[0]

        # pruning
        if layer_idx in PRUNE_LAYERS:
            pruner = pruners[str(layer_idx)]
            hidden_states, _ = apply_token_pruning(
                hidden_states,
                pruner,
                keep_ratio,
            )

    # final norm + head
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits


# ============================
# Baseline (normal forward)
# ============================
def baseline_prefill(model, input_ids, attention_mask):
    inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits


# ============================
# Timing utility
# ============================
def measure_latency(fn, *args, warmup=1, runs=3):
    for _ in range(warmup):
        _ = fn(*args)
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)


# ============================
# Dummy input
# ============================
def build_dummy_input(tokenizer, length, device="cuda"):
    base_ids = tokenizer("Hello, this is a test.", return_tensors="pt")["input_ids"][0]
    if base_ids.size(0) >= length:
        ids = base_ids[:length]
    else:
        repeat = (length + base_ids.size(0) - 1) // base_ids.size(0)
        ids = base_ids.repeat(repeat)[:length]

    input_ids = ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return input_ids, attention_mask


# ============================
# Profiling multi-GPU
# ============================
def profile_lengths(lengths, keep_ratio, save_json: bool = True):
    model, tokenizer, pruners = load_model_and_pruners()
    model.eval()

    print("Profiling lengths:", lengths)
    
    # Store results
    baseline_results = {}
    erecap_results = {}

    for L in lengths:
        input_ids = None
        attention_mask = None
        try:
            print("\n======================================")
            print(f"Testing length {L}")
            print("======================================")

            # Find first device holding first layer (model.model.layers[0])
            first_device = next(model.model.layers[0].parameters()).device

            input_ids, attention_mask = build_dummy_input(tokenizer, L, device=first_device)

            # Test baseline
            try:
                baseline_t = measure_latency(
                    lambda x, m: baseline_prefill(model, x, m),
                    input_ids,
                    attention_mask,
                )
                baseline_results[str(L)] = baseline_t
                print(f"[Length {L}] baseline={baseline_t:.4f}s")
            except Exception as e:
                print(f"[Length {L}] Baseline test failed: {e}")
            
            # Test E-RECAP
            try:
                erecap_t = measure_latency(
                    lambda x, m: prefill_with_pruning(model, x, m, pruners, keep_ratio),
                    input_ids,
                    attention_mask,
                )
                erecap_results[str(L)] = erecap_t
                print(f"[Length {L}] erecap={erecap_t:.4f}s")
                
                if str(L) in baseline_results:
                    speedup = baseline_results[str(L)] / erecap_t if erecap_t > 0 else float("inf")
                    print(f"[Length {L}] speedup={speedup:.2f}x")
            except Exception as e:
                print(f"[Length {L}] E-RECAP test failed: {e}")
                import traceback
                traceback.print_exc()

        except torch.cuda.OutOfMemoryError as e:
            print(f"[Length {L}] OOM, skipped: {e}")
        except Exception as e:
            print(f"[Length {L}] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if input_ids is not None:
                del input_ids, attention_mask
                torch.cuda.empty_cache()
    
    # Save results to JSON
    if save_json:
        os.makedirs("results", exist_ok=True)
        
        baseline_path = "results/latency_baseline_multigpu.json"
        erecap_path = "results/latency_erecap_multigpu.json"
        
        # Save baseline results if available
        if baseline_results:
            with open(baseline_path, "w") as f:
                json.dump(baseline_results, f, indent=2)
            print(f"[OK] Baseline results saved to {baseline_path}")
        else:
            print(f"[Warning] No baseline results to save")
        
        # Save E-RECAP results if available
        if erecap_results:
            with open(erecap_path, "w") as f:
                json.dump(erecap_results, f, indent=2)
            print(f"[OK] E-RECAP results saved to {erecap_path}")
        else:
            print(f"[Warning] No E-RECAP results to save (all tests may have failed)")


# ============================
# Text generation (multi-GPU)
# ============================
def generate_text(prompt):
    model, tokenizer, pruners = load_model_and_pruners()
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ============================
# Entry
# ============================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["profile", "generate"], default="profile")
    p.add_argument("--prompt", type=str, default="Hello E-RECAP multi GPU!")
    p.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--keep_ratio", type=float, default=KEEP_RATIO)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "profile":
        profile_lengths(args.lengths, args.keep_ratio)
    else:
        print(generate_text(args.prompt))


if __name__ == "__main__":
    main()
