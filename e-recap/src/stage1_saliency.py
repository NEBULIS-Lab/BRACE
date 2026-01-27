# Pruning Module Training（Stage 1） 只是尝试，后续还需要拓展

import random
from typing import Dict, List
import argparse
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "checkpoints/qwen2-7b-instruct"
DATA_PATH = "data/raw/dolly15k"

MAX_LEN = 512
BATCH_SIZE = 1
PRUNE_LAYERS = [4, 7, 10, 13, 16, 19, 22, 25]

def build_dataloader(*, tokenizer: AutoTokenizer, data_path: str, num_samples: int) -> DataLoader:
    dataset = load_from_disk(str(data_path))["train"]
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_samples]

    examples = []
    for idx in indices:
        item = dataset[idx]
        # Include instruction, context, and response for complete semantic information
        parts = [item.get('instruction', ''), item.get('context', ''), item['response']]
        text = '\n'.join([p for p in parts if p])  # Only join non-empty parts
        enc = tokenizer(
            text,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        examples.append(
            {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }
        )
    def collate_fn(batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return DataLoader(examples, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--out_path", type=str, default="checkpoints/saliency.pt")
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Local HuggingFace model directory (default: checkpoints/qwen2-7b-instruct).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help="Dataset directory for `datasets.load_from_disk` (default: data/raw/dolly15k).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    # for p in model.parameters():
        # p.requires_grad = False

    dataloader = build_dataloader(tokenizer=tokenizer, data_path=str(args.data_path), num_samples=int(args.num_samples))

    forward_cache = {}
    backward_cache = {}
    saliency_results = {k: [] for k in PRUNE_LAYERS}

    def create_hooks(layer_idx):
        def forward_hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, (tuple, list)) else out
            forward_cache[layer_idx] = hidden.detach()

        def backward_hook(_module, grad_in, grad_out):
            grad_hidden = grad_out[0] if isinstance(grad_out, (tuple, list)) else grad_out
            backward_cache[layer_idx] = grad_hidden.detach()

        return forward_hook, backward_hook

    hooks = []
    try:
        for idx in PRUNE_LAYERS:
            layer = model.model.layers[idx]
            f_hook, b_hook = create_hooks(idx)
            hooks.append(layer.register_forward_hook(f_hook))
            hooks.append(layer.register_full_backward_hook(b_hook))

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            model.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                use_cache=False,
            )
            loss = outputs.loss
            loss.backward()

            for idx in PRUNE_LAYERS:
                hidden = forward_cache.get(idx)
                grad = backward_cache.get(idx)
                if hidden is None or grad is None:
                    continue

                sal = (hidden * grad).sum(dim=-1)
                saliency_results[idx].append(sal.float().cpu().squeeze(0))

            forward_cache.clear()
            backward_cache.clear()

    finally:
        for h in hooks:
            h.remove()

    torch.save(saliency_results, str(args.out_path))
    print(f"[OK] Saliency saved to {args.out_path}")


if __name__ == "__main__":
    main()
