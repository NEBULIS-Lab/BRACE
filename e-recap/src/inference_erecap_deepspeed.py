# src/inference_erecap_deepspeed.py
# 真正把 Qwen2-7B + E-RECAP 跑到 64K/128K 级别上下文
# 64K tokens ≈ 5 万字中文 / 6〜7 万字英文
# 128K tokens ≈ 10 万字中文 / 13〜14 万字英文

"""
E-RECAP + DeepSpeed ZeRO-3 + Tensor Parallel skeleton.

This file assumes:
  - It is launched with `deepspeed` CLI;
  - A deepspeed config JSON is provided via --deepspeed_config;
  - Qwen2-7B weights are the same as in other scripts.
The E-RECAP pruning logic follows the single-GPU implementation,
but the model is wrapped inside a DeepSpeed engine.
"""

import argparse
import json
import os
import time
from typing import Tuple, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import deepspeed  # requires deepspeed installed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# DeepSpeed init helpers
# ------------------------
def load_pruning_modules(hidden_size: int) -> nn.ModuleDict:
    modules = nn.ModuleDict(
        {str(i): TokenPruningModule(hidden_size) for i in PRUNE_LAYERS}
    )
    state_dict = torch.load(PRUNING_CKPT, map_location="cpu")
    modules.load_state_dict(state_dict)
    modules.to(DEVICE)
    modules.half()
    modules.eval()
    for p in modules.parameters():
        p.requires_grad = False
    return modules


def init_deepspeed_engine(args, model: nn.Module):
    """
    Initialize DeepSpeed ZeRO-3 + (optional) TP on the given model.
    The exact behaviour is controlled by the JSON config.
    """
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    # No optimizer / scheduler needed for inference-only engine
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    return engine


# ------------------------
# Pruning logic
# ------------------------
def apply_token_pruning(
    hidden_states: torch.Tensor,
    pruning_module: nn.Module,
    keep_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = hidden_states.size(1)

    scores = pruning_module(hidden_states)  # [seq_len]
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
# Prefill with E-RECAP on DeepSpeed engine
# ------------------------
def prefill_with_erecap_deepspeed(
    engine,
    input_ids: torch.Tensor,
    pruning_modules: nn.ModuleDict,
    keep_ratio: float,
) -> torch.Tensor:
    """
    engine.module is the underlying Qwen2-7B model (possibly TP-sharded).
    We keep the same E-RECAP logic, but all forward passes go through engine.module.
    """
    model = engine.module

    hidden_states = model.model.embed_tokens(input_ids)

    for layer_idx, layer in enumerate(model.model.layers):
        position_ids = torch.arange(
            0,
            hidden_states.size(1),
            dtype=torch.long,
            device=hidden_states.device,
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

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits


def baseline_prefill_deepspeed(engine, input_ids: torch.Tensor) -> torch.Tensor:
    model = engine.module
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
    base_ids = tokenizer("Hello from E-RECAP + DeepSpeed.", return_tensors="pt")["input_ids"][0]
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
    # Local rank is managed by DeepSpeed launcher
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=None,  # DeepSpeed handles device placement
        local_files_only=True,
    )

    engine = init_deepspeed_engine(args, model)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    hidden_size = engine.module.config.hidden_size
    pruning_modules = load_pruning_modules(hidden_size)

    if engine.global_rank == 0:
        print("[DS-E-RECAP] Profiling lengths:", args.lengths)

    for L in args.lengths:
        input_ids = build_dummy_input(tokenizer, L, device=engine.local_rank)

        try:
            baseline_t = measure_latency(
                lambda x: baseline_prefill_deepspeed(engine, x),
                input_ids,
            )
            erecap_t = measure_latency(
                lambda x: prefill_with_erecap_deepspeed(
                    engine,
                    x,
                    pruning_modules,
                    args.keep_ratio,
                ),
                input_ids,
            )

            if engine.global_rank == 0:
                speedup = baseline_t / erecap_t if erecap_t > 0 else float("inf")
                print(
                    f"[Length {L}] DS-baseline={baseline_t:.4f}s  "
                    f"DS-E-RECAP={erecap_t:.4f}s  speedup={speedup:.2f}x"
                )

        except torch.cuda.OutOfMemoryError:
            if engine.global_rank == 0:
                print(f"[Length {L}] OOM under DeepSpeed, skipped.")
        finally:
            del input_ids
            torch.cuda.empty_cache()


# ------------------------
# CLI
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        required=True,
        help="Path to DeepSpeed ZeRO-3 + TP config JSON.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="profile",
        choices=["profile"],
        help="Only 'profile' is defined for now.",
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
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "profile":
        profile_lengths(args)
    else:
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Mode {args.mode} not implemented yet.")


if __name__ == "__main__":
    main()
