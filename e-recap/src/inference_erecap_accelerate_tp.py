# src/inference_erecap_accelerate_tp.py 
# 一个不依赖 DeepSpeed 的 TP 版本，用 accelerate 做多卡并行。
"""
E-RECAP + HuggingFace Accelerate Tensor-Parallel skeleton.

This script:
  - builds a Qwen2-7B model with Accelerate device_map="auto";
  - keeps E-RECAP pruning logic (similar to single-GPU version);
  - is designed for profiling only, no training.
"""

import argparse
import os
import time
from typing import Tuple, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

MODEL_PATH = "checkpoints/qwen2-7b-instruct"
PRUNING_CKPT = "checkpoints/pruning_module.pt"

MAX_NEW_TOKENS = 128
PRUNE_LAYERS = [4, 7, 10, 13, 16, 19, 22, 25]
KEEP_RATIO = 0.7
MIN_HEAD_TOKENS = 4
MIN_TAIL_TOKENS = 16


# ------------------------
# TokenPruningModule
# ------------------------
class TokenPruningModule(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)
        return self.scorer(hidden_states).squeeze(-1)


# ------------------------
# Accelerate TP helpers
# ------------------------
def load_accelerate_model_and_pruners():
    """
    Build Qwen2-7B under Accelerate with device_map="auto".

    Note:
      - This assumes a HF-style checkpoint layout is available.
      - For your current project, using from_pretrained with device_map="auto"
        might be enough; here we show the more explicit pattern.
    """
    # Empty-init prevents allocating full weights on CPU first
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map=None,
            local_files_only=True,
        )

    # Dispatch across all visible GPUs
    device_map = "auto"
    model = load_checkpoint_and_dispatch(
        model,
        MODEL_PATH,
        device_map=device_map,
        dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    hidden_size = model.config.hidden_size
    pruning_modules = nn.ModuleDict(
        {str(i): TokenPruningModule(hidden_size) for i in PRUNE_LAYERS}
    )
    state_dict = torch.load(PRUNING_CKPT, map_location="cpu")
    pruning_modules.load_state_dict(state_dict)

    # For simplicity we keep pruners on cuda:0
    device0 = torch.device("cuda:0")
    pruning_modules.to(device0)
    pruning_modules.half()
    pruning_modules.eval()
    for p in pruning_modules.parameters():
        p.requires_grad = False

    return model, tokenizer, pruning_modules, device0


# ------------------------
# Pruning logic
# ------------------------
def apply_token_pruning(
    hidden_states: torch.Tensor,
    pruning_module: nn.Module,
    keep_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = hidden_states.size(1)
    scores = pruning_module(hidden_states.to(pruning_module.scorer[0].weight.device))

    base_keep = set(range(min(MIN_HEAD_TOKENS, seq_len)))
    for i in range(max(0, seq_len - MIN_TAIL_TOKENS), seq_len):
        base_keep.add(i)

    target_keep = max(int(seq_len * keep_ratio), len(base_keep))
    target_keep = min(target_keep, seq_len)

    _, sorted_idx = torch.topk(scores, k=seq_len)
    selected: List[int] = []

    for idx in sorted_idx.tolist():
        if idx in base_keep:
            selected.append(idx)
    for idx in sorted_idx.tolist():
        if idx not in base_keep and len(selected) < target_keep:
            selected.append(idx)

    selected = sorted(selected)
    index_tensor = torch.tensor(
        selected,
        device=hidden_states.device,
        dtype=torch.long,
    )
    pruned_hidden = hidden_states[:, index_tensor, :]
    return pruned_hidden, index_tensor


# ------------------------
# Prefill with Accelerate TP + E-RECAP
# ------------------------
def prefill_with_erecap_accel(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    pruning_modules: nn.ModuleDict,
    keep_ratio: float,
) -> torch.Tensor:
    """
    input_ids is placed on cuda:0, but model is sharded across devices.
    Accelerate will route tensors as needed internally.
    """
    # embed on the first device
    first_device = next(model.parameters()).device
    input_ids = input_ids.to(first_device)

    hidden_states = model.model.embed_tokens(input_ids)

    for layer_idx, layer in enumerate(model.model.layers):
        device_layer = next(layer.parameters()).device
        hidden_states = hidden_states.to(device_layer)

        position_ids = torch.arange(
            0,
            hidden_states.size(1),
            dtype=torch.long,
            device=device_layer,
        ).unsqueeze(0)

        outputs = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = outputs[0]

        if layer_idx in PRUNE_LAYERS:
            pruner = pruning_modules[str(layer_idx)]
            hidden_states, _ = apply_token_pruning(
                hidden_states,
                pruner,
                keep_ratio,
            )

    last_device = next(model.lm_head.parameters()).device
    hidden_states = hidden_states.to(last_device)
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits


def baseline_prefill_accel(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    first_device = next(model.parameters()).device
    input_ids = input_ids.to(first_device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, device=input_ids.device),
            use_cache=False,
        )
    return outputs.logits


# ------------------------
# Timing utilities
# ------------------------
def measure_latency(fn, *args, warmup: int = 1, runs: int = 3) -> float:
    for _ in range(warmup):
        _ = fn(*args)
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = fn(*args)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / len(times)


def build_dummy_input(tokenizer: AutoTokenizer, length: int, device: torch.device):
    base_ids = tokenizer("Hello from E-RECAP + Accelerate TP.", return_tensors="pt")["input_ids"][0]
    if base_ids.size(0) >= length:
        ids = base_ids[:length]
    else:
        repeat = (length + base_ids.size(0) - 1) // base_ids.size(0)
        ids = base_ids.repeat(repeat)[:length]

    input_ids = ids.unsqueeze(0).to(device)
    return input_ids


# ------------------------
# Profiling
# ------------------------
def profile_lengths(args):
    model, tokenizer, pruners, device0 = load_accelerate_model_and_pruners()
    model.eval()

    if int(os.environ.get("RANK", "0")) == 0:
        print("[Accel-E-RECAP] Profiling lengths:", args.lengths)

    for L in args.lengths:
        input_ids = build_dummy_input(tokenizer, L, device0)
        try:
            baseline_t = measure_latency(
                lambda x: baseline_prefill_accel(model, x),
                input_ids,
            )
            erecap_t = measure_latency(
                lambda x: prefill_with_erecap_accel(model, x, pruners, args.keep_ratio),
                input_ids,
            )
            speedup = baseline_t / erecap_t if erecap_t > 0 else float("inf")
            print(
                f"[Length {L}] Accel-baseline={baseline_t:.4f}s  "
                f"Accel-E-RECAP={erecap_t:.4f}s  speedup={speedup:.2f}x"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[Length {L}] OOM under Accelerate TP, skipped.")
        finally:
            del input_ids
            torch.cuda.empty_cache()


# ------------------------
# CLI
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="profile",
        choices=["profile"],
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32768],
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=KEEP_RATIO,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "profile":
        profile_lengths(args)
    else:
        print(f"Mode {args.mode} is not implemented yet.")


if __name__ == "__main__":
    main()
