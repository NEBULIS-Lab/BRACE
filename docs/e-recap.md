# E-RECAP in BRACE-code

BRACE can optionally use **E-RECAP** as a learned pruning module to reduce replanning context cost.
This repo vendors the E-RECAP code under `e-recap/` and standardizes all assets/deps at the BRACE project level.

## 1) Dependencies

Install BRACE dependencies:

```bash
pip install -r requirements.txt
```

E-RECAP also requires **PyTorch**. Install a Torch build that matches your CUDA / platform.

## 2) Local assets (not committed)

Put large assets under the project-wide `checkpoints/` (git-ignored). Suggested layout:

```
checkpoints/
  <hf-model-name>/
  saliency.pt
  pruning_module.pt
```

Optional env vars (used by `e-recap/src/inference_erecap.py`):
- `BRACE_E_RECAP_MODEL_DIR`: overrides the default model dir
- `BRACE_E_RECAP_PRUNER_PATH`: overrides the default pruning checkpoint path

## 3) Run E-RECAP entrypoints (BRACE-level scripts)

### Stage 1 (saliency cache)

```bash
scripts/run_e_recap_stage1.sh --model checkpoints/<hf-model-name> --data data/raw/dolly15k --out checkpoints/saliency.pt --num-samples 1000
```

### Stage 2 (train pruning module)

```bash
scripts/run_e_recap_stage2.sh --model checkpoints/<hf-model-name> --data data/raw/dolly15k --saliency checkpoints/saliency.pt --out checkpoints/pruning_module.pt --lr 1e-4 --epochs 2
```

### Inference / profiling

```bash
scripts/run_e_recap_inference.sh --mode profile --config keep07 --model_path checkpoints/<hf-model-name> --pruning_ckpt checkpoints/pruning_module.pt
```

## 4) Using E-RECAP from BRACE runners

BRACE experiment configs typically toggle pruning via `pruning_enabled` / `context_strategy`.
For learned pruning, point the runner to the local model + pruning checkpoint via env vars or config knobs (runner-dependent).

