# FlashAttention2 + Flash-Decoding 版本（单机单卡）

import argparse
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local paths
MODEL_PATH = "checkpoints/qwen2-7b-instruct"
PRUNING_CKPT = "checkpoints/pruning_module.pt"

# E-RECAP config (should match Stage2)
MAX_NEW_TOKENS = 128
PRUNE_LAYERS = [4, 7, 10, 13, 16, 19, 22, 25]
KEEP_RATIO = 0.7
MIN_HEAD_TOKENS = 4
MIN_TAIL_TOKENS = 16


class TokenPruningModule(nn.Module):
    """Small MLP that outputs a scalar importance score per token."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [seq_len, hidden] or [batch, seq_len, hidden]
        return: [seq_len]
        """
        return self.scorer(hidden_states).squeeze(-1)


def load_model_and_pruners_flash():
    """
    Load Qwen2 with FlashAttention2 enabled and attach E-RECAP pruning modules.
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",  # enable FlashAttention2
        device_map=None,
        local_files_only=True,
    ).to(device)

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

    pruning_modules.to(device)
    pruning_modules.half()
    pruning_modules.eval()
    for p in pruning_modules.parameters():
        p.requires_grad = False

    return model, tokenizer, pruning_modules


def apply_token_pruning(
    hidden_states: torch.Tensor,
    pruning_module: nn.Module,
    keep_ratio: float,
):
    """
    hidden_states: [1, seq_len, hidden]
    return: pruned_hidden_states, kept_indices
    """
    seq_len = hidden_states.size(1)

    flat = hidden_states.squeeze(0)
    scores = pruning_module(flat)

    base_keep = set(range(min(MIN_HEAD_TOKENS, seq_len)))
    for i in range(max(0, seq_len - MIN_TAIL_TOKENS), seq_len):
        base_keep.add(i)

    target_keep = max(int(seq_len * keep_ratio), len(base_keep))
    target_keep = min(target_keep, seq_len)

    _, sorted_idx = torch.topk(scores, k=seq_len)

    selected = []
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


def prefill_with_pruning_flash(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    pruning_modules: nn.ModuleDict,
    keep_ratio: float,
):
    """
    Manual forward over all transformer layers with E-RECAP pruning.
    FlashAttention2 is used internally by each layer (through attn_implementation).
    """
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


def baseline_prefill_flash(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    model_inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    with torch.no_grad():
        outputs = model(**model_inputs)
    return outputs.logits


def measure_latency(fn, *args, warmup: int = 1, runs: int = 3) -> float:
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
    base_ids = tokenizer("Hello, this is a test.", return_tensors="pt")["input_ids"][0]
    if base_ids.size(0) >= length:
        ids = base_ids[:length]
    else:
        repeat = (length + base_ids.size(0) - 1) // base_ids.size(0)
        ids = base_ids.repeat(repeat)[:length]

    input_ids = ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return input_ids, attention_mask


def profile_lengths_flash(lengths, keep_ratio: float):
    model, tokenizer, pruners = load_model_and_pruners_flash()
    model.eval()

    print("[Flash E-RECAP] Profiling lengths:", lengths)
    for L in lengths:
        input_ids, attention_mask = build_dummy_input(tokenizer, L)
        try:
            base_t = measure_latency(
                lambda x, m: baseline_prefill_flash(model, x, m),
                input_ids,
                attention_mask,
            )
            erecap_t = measure_latency(
                lambda x: prefill_with_pruning_flash(model, x, pruners, keep_ratio),
                input_ids,
            )
            speed = base_t / erecap_t if erecap_t > 0 else float("inf")
            print(
                f"[Length {L}] baseline={base_t:.4f}s  "
                f"flash_erecap={erecap_t:.4f}s  speedup={speed:.2f}x"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[Length {L}] OOM on GPU, skip.")
        finally:
            del input_ids, attention_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()


def generate_text_flash(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="profile",
        choices=["profile", "generate"],
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
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, E-RECAP with FlashAttention2!",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "profile":
        profile_lengths_flash(args.lengths, args.keep_ratio)
        return

    if args.mode == "generate":
        model, tokenizer, pruners = load_model_and_pruners_flash()
        model.eval()
        text = generate_text_flash(model, tokenizer, args.prompt)
        print(text)
        return


if __name__ == "__main__":
    main()