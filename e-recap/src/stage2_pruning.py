# Pruning Module Training（Stage 2）

import random
from typing import Dict, List

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# =======================
# Local model & data paths
# =======================
MODEL_NAME = "checkpoints/qwen2-7b-instruct"
DATA_PATH = "data/raw/dolly15k"
SAL_PATH = "checkpoints/saliency.pt"

# =======================
# Hyperparameters
# =======================
MAX_LEN = 512
BATCH_SIZE = 1
LR = 1e-4
EPOCHS = 2
KEEP_RATIO = 0.7
TEMPERATURE = 1.0

# Must match Stage1
PRUNE_LAYERS = [4, 7, 10, 13, 16, 19, 22, 25]


# =======================
# Token Pruning Module
# =======================
class TokenPruningModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),  # Use GELU as in paper (was ReLU)
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states):
        logits = self.scorer(hidden_states).squeeze(-1)
        return logits


# =======================
# Ranking loss (logistic loss form as in paper)
# =======================
def ranking_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Logistic ranking loss as in the paper:
    L_r = sum_{i<j} log(1 + exp(-(π_i - π_j) * sign(π̂_i - π̂_j)))
    """
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)  # [N, N]
    diff_target = target.unsqueeze(1) - target.unsqueeze(0)  # [N, N]
    sign_target = torch.sign(diff_target)
    # Logistic loss: log(1 + exp(-diff_pred * sign_target))
    loss = torch.log(1 + torch.exp(-diff_pred * sign_target))
    return loss.mean()


# =======================
# Data loader
# =======================
def build_dataloader(tokenizer: AutoTokenizer, dataset, num_examples: int) -> DataLoader:
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_examples]

    examples: List[Dict[str, torch.Tensor]] = []
    for idx in indices:
        item = dataset[idx]
        # Include instruction, context, and response for complete semantic information
        parts = [item.get('instruction', ''), item.get('context', ''), item['response']]
        text = '\n'.join([p for p in parts if p])  # Only join non-empty parts
        encoding = tokenizer(
            text,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        examples.append(
            {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
        )

    def collate_fn(batch):
        input_ids = torch.stack([example["input_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return DataLoader(examples, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


# =======================
# Main training loop
# =======================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_NAME)
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--saliency_path", type=str, default=SAL_PATH)
    parser.add_argument("--output_path", type=str, default="checkpoints/pruning_module.pt")
    parser.add_argument("--learning_rate", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load local Qwen2 model
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    # Freeze Qwen
    for p in model.parameters():
        p.requires_grad = False

    # Init pruning modules
    hidden_size = model.config.hidden_size
    pruning_modules = nn.ModuleDict()
    for layer_idx in PRUNE_LAYERS:
        pruning_modules[str(layer_idx)] = TokenPruningModule(hidden_size).to(device)

    # <<< NEW: convert pruning modules to fp16 >>>
    pruning_modules.half()

    # Load saliency & dataset
    saliency_data = torch.load(str(args.saliency_path))
    dataset = load_from_disk(str(args.data_path))["train"]

    num_examples = min(len(next(iter(saliency_data.values()))), len(dataset))
    dataloader = build_dataloader(tokenizer, dataset, num_examples)

    optimizer = torch.optim.Adam(pruning_modules.parameters(), lr=float(args.learning_rate))

    # =======================
    # Training
    # =======================
    for epoch in range(int(args.epochs)):
        print(f"[Epoch {epoch+1}/{int(args.epochs)}]")
        for batch_idx, batch in enumerate(dataloader):

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # embedding
            hidden_states = model.model.embed_tokens(input_ids)

            sample_index = batch_idx * BATCH_SIZE

            # forward through each layer
            for layer_idx, block in enumerate(model.model.layers):

                # mandatory: position_ids
                position_ids = torch.arange(
                    0, hidden_states.size(1),
                    dtype=torch.long,
                    device=device
                ).unsqueeze(0)

                block_outputs = block(
                    hidden_states,
                    attention_mask=None,   # DO NOT pass 2D mask
                    position_ids=position_ids,
                    use_cache=False,
                )
                hidden_states = block_outputs[0]

                if layer_idx in PRUNE_LAYERS:
                    module = pruning_modules[str(layer_idx)]

                    logits = module(hidden_states.squeeze(0))

                    mask_logits = torch.stack(
                        [logits, torch.zeros_like(logits)],
                        dim=-1,
                    )
                    soft_mask = F.gumbel_softmax(
                        mask_logits,
                        tau=TEMPERATURE,
                        hard=False,
                        dim=-1,
                    )[:, 0]
                    hidden_states = hidden_states * soft_mask.unsqueeze(0).unsqueeze(-1)

                    target_list = saliency_data[layer_idx]
                    target_sal = target_list[sample_index].to(device)

                    pred_sal = logits
                    mse = F.mse_loss(pred_sal, target_sal)
                    rank = ranking_loss(pred_sal, target_sal)
                else:
                    mse = torch.tensor(0.0, device=device)
                    rank = torch.tensor(0.0, device=device)

            # LM loss
            logits = model.lm_head(hidden_states)
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            loss = lm_loss + mse + rank
            loss.backward()
            optimizer.step()

    out_path = str(args.output_path)
    torch.save(pruning_modules.state_dict(), out_path)
    print(f"[OK] pruning module saved to {out_path}")


if __name__ == "__main__":
    main()
