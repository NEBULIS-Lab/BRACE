# `checkpoints/` (local models & pruning weights; not committed)

This repository may require large model weights and pruning checkpoints (e.g., E-RECAP pruning modules).
To keep the repo open-source friendly, `checkpoints/` is **git-ignored** by default.

Suggested layout (examples only):

```
checkpoints/
  <hf-model-name>/              # HuggingFace model directory (config.json, tokenizer, weights, ...)
  pruning_module.pt             # E-RECAP pruning module (Stage 2 output)
  saliency.pt                   # E-RECAP saliency cache (Stage 1 output; optional)
```

BRACE runners and utilities should reference local assets via env vars (recommended) or these ignored folders.
