# Pi0/Pi0.5 Training Pipeline

## Model Overview

**Pi0 (π₀)** is a Vision-Language-Action (VLA) model developed by Physical Intelligence that uses **flow matching** for continuous action prediction. It combines a vision-language model (PaliGemma) with an action expert network to predict robot actions from visual observations and language instructions.

### Key Innovations
- **Flow Matching**: Continuous action prediction without discretization
- **Action Chunking**: Predicts 50 future actions at once for temporal consistency
- **Multi-Camera**: Leverages 3 camera views for comprehensive scene understanding

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Pi0 Model Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                       │
│   │  Side View  │   │Overhead View│   │ Wrist View  │                       │
│   │  (224×224)  │   │  (224×224)  │   │  (224×224)  │                       │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                       │
│          │                 │                 │                              │
│          └─────────────────┼─────────────────┘                              │
│                            ▼                                                │
│                 ┌─────────────────────┐                                     │
│                 │     SigLIP ViT      │  Vision Encoder                     │
│                 │  (400M parameters)  │  - Patch size: 14×14                │
│                 │                     │  - Output: 256 tokens/image         │
│                 └──────────┬──────────┘                                     │
│                            │                                                │
│                            ▼                                                │
│          ┌─────────────────────────────────────┐                            │
│          │         Visual Token Projection     │                            │
│          │         (Vision → LLM space)        │                            │
│          └─────────────────┬───────────────────┘                            │
│                            │                                                │
│   ┌────────────────────────┼────────────────────────┐                       │
│   │                        ▼                        │                       │
│   │  ┌─────────────────────────────────────────┐    │                       │
│   │  │            PaliGemma (Gemma 2B)         │    │                       │
│   │  │         Vision-Language Model           │    │                       │
│   │  │                                         │    │                       │
│   │  │  Input: Visual Tokens + Language Prompt │    │                       │
│   │  │  Output: Multimodal Embeddings          │    │                       │
│   │  └─────────────────────┬───────────────────┘    │                       │
│   │                        │                        │                       │
│   └────────────────────────┼────────────────────────┘                       │
│                            │                                                │
│   ┌────────────────────────┼────────────────────────┐                       │
│   │                        ▼                        │                       │
│   │  ┌─────────────────────────────────────────┐    │                       │
│   │  │         Action Expert (Gemma 300M)      │    │                       │
│   │  │                                         │    │                       │
│   │  │  Input: Multimodal Embeddings + State   │    │  Robot State (8-dim)  │
│   │  │  Output: Action Embeddings              │◄──-┼───────────────────────│
│   │  └─────────────────────┬───────────────────┘    │                       │
│   │                        │                        │                       │
│   └────────────────────────┼────────────────────────┘                       │
│                            │                                                │
│                            ▼                                                │
│                 ┌─────────────────────┐                                     │
│                 │  Flow Matching Head │                                     │
│                 │                     │                                     │
│                 │  Predicts velocity  │                                     │
│                 │  field for actions  │                                     │
│                 └──────────┬──────────┘                                     │
│                            │                                                │
│                            ▼                                                │
│                 ┌─────────────────────┐                                     │
│                 │   Action Output     │                                     │
│                 │   (50 × 8-dim)      │                                     │
│                 │                     │                                     │
│                 │  50 future actions  │                                     │
│                 │  8 DoF per action   │                                     │
│                 └─────────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Vision Encoder (SigLIP)

| Property | Value |
|----------|-------|
| Architecture | Vision Transformer (ViT) |
| Parameters | ~400M |
| Input Resolution | 224 × 224 |
| Patch Size | 14 × 14 |
| Output Tokens | 256 per image |
| Training | Contrastive (image-text pairs) |

**Purpose**: Extract rich visual features from each camera view.

### 2. PaliGemma (Gemma 2B)

| Property | Value |
|----------|-------|
| Architecture | Decoder-only Transformer |
| Parameters | 2B |
| Context Length | 48 (Pi0) / 200 (Pi0.5) |
| Vocabulary | 256k tokens |

**Purpose**: Fuse visual features with language instructions for multimodal understanding.

### 3. Action Expert (Gemma 300M)

| Property | Value |
|----------|-------|
| Architecture | Decoder-only Transformer |
| Parameters | 300M |
| Input | Multimodal embeddings + robot state |
| Output | Action embeddings |

**Purpose**: Specialized network for action prediction, conditioned on scene understanding.

### 4. Flow Matching Head

**Flow matching** predicts actions by learning a velocity field that transforms noise into actions:

```
Noise (t=0) ────────────────────────► Actions (t=1)
             ↑
             │ Learned velocity field v(x, t)
             │ dx/dt = v(x, t)
```

Benefits over discrete action prediction:
- Continuous action space (no discretization artifacts)
- Smooth trajectories
- Better for fine manipulation

## Model Variants

```
┌─────────────────────────────────────────────────────────────┐
│                    Pi0 vs Pi0.5 Comparison                  │
├─────────────────────┬───────────────────┬───────────────────┤
│     Feature         │       Pi0         │      Pi0.5        │
├─────────────────────┼───────────────────┼───────────────────┤
│ Token Length        │        48         │       200         │
│ Context Window      │      Short        │      Long         │
│ Flow Matching       │      Basic        │    Enhanced       │
│ Memory Usage        │      Lower        │     Higher        │
│ Training Speed      │      Faster       │     Slower        │
│ Performance         │       Good        │     Better        │
│ Checkpoint          │    pi0_base       │   pi05_base       │
└─────────────────────┴───────────────────┴───────────────────┘
```

## Input/Output Processing

### Input Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Processing                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  IMAGES (3 views)                                           │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                    │
│  │HWC uint8│ → │ Resize  │ → │Normalize│ → float32          │
│  │240×320×3│   │224×224×3│   │ [-1,1]  │                    │
│  └─────────┘   └─────────┘   └─────────┘                    │
│                                                             │
│  STATE (8-dim)                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                    │
│  │ 8-dim   │ → │Pad to 32│ → │ bfloat16│ → Model            │
│  │ float32 │   │  zeros  │   │         │                    │
│  └─────────┘   └─────────┘   └─────────┘                    │
│                                                             │
│  LANGUAGE                                                   │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                    │
│  │  Text   │ → │Tokenize │ → │  Embed  │ → Token IDs        │
│  │ String  │   │PaliGemma│   │         │                    │
│  └─────────┘   └─────────┘   └─────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Output Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     Output Processing                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  MODEL OUTPUT                                               │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐              │
│  │ 50 × 32   │ → │Slice [:8] │ → │Denormalize│ → Actions    │
│  │ (padded)  │   │ 50 × 8    │   │  q01/q99  │              │
│  └───────────┘   └───────────┘   └───────────┘              │
│                                                             │
│  ACTION SPACE                                               │
│  ┌─────────────────────────────────────────────┐            │
│  │ Dim 0-6: Joint positions (continuous)       │            │
│  │ Dim 7:   Gripper state [-1, 1]              │            │
│  └─────────────────────────────────────────────┘            │
│                                                             │
│  NORMALIZATION (Quantile-based)                             │
│  ┌─────────────────────────────────────────────┐            │
│  │ action_normalized = (action - q01) / (q99 - q01)         │
│  │ action_denorm = action_norm * (q99 - q01) + q01          │
│  └─────────────────────────────────────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Training

### Commands

```bash
# Single GPU
bash train.sh --policy pi0 --task LiftBarrier-rf --agent_id 0

# Multi-GPU (4 GPUs)
bash train.sh --policy pi0 --task LiftBarrier-rf --agent_id 0 --gpus 4

# Pi0.5 variant
bash train.sh --policy pi05 --task LiftBarrier-rf --agent_id 0
```

### Configuration (`robot_pi0.yaml`)

```yaml
model:
  model_variant: "pi0"              # or "pi05"
  action_dim: 8
  action_horizon: 50
  max_token_len: 48                 # 200 for pi05
  pretrained_checkpoint: "gs://openpi-assets/checkpoints/pi0_base"
  use_gradient_checkpointing: true

training:
  num_epochs: 300
  learning_rate: 5.0e-4
  weight_decay: 1.0e-6
  checkpoint_every: 1
```

### Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Loop                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  for epoch in range(300):                                   │
│      for batch in dataloader:                               │
│          ┌─────────────────────────────────────────┐        │
│          │ 1. Load batch (images, state, actions)  │        │
│          └─────────────────┬───────────────────────┘        │
│                            ▼                                │
│          ┌─────────────────────────────────────────┐        │
│          │ 2. Forward pass with autocast(bfloat16) │        │
│          │    loss = model(obs, actions)           │        │
│          └─────────────────┬───────────────────────┘        │
│                            ▼                                │
│          ┌─────────────────────────────────────────┐        │
│          │ 3. Backward pass                        │        │
│          │    loss.backward()                      │        │
│          └─────────────────┬───────────────────────┘        │
│                            ▼                                │
│          ┌─────────────────────────────────────────┐        │
│          │ 4. Gradient clipping (max_norm=1.0)     │        │
│          │    optimizer.step()                     │        │
│          └─────────────────────────────────────────┘        │
│                                                             │
│      save_checkpoint(f"epoch_{epoch+1}")                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Checkpoints

```
robofactory/checkpoints/pi0/{task}_Agent{id}_{num}/
├── epoch_1/
│   ├── model.safetensors    # Model weights
│   ├── optimizer.pt         # Optimizer state
│   └── metadata.pt          # Training info
├── epoch_2/
│   └── ...
└── latest/                  # Symlink to most recent
```

## Memory & Performance

| Config | GPU Memory | Batch Size | Time/Epoch |
|--------|------------|------------|------------|
| Pi0 (A100 40GB) | ~28GB | 8 | ~3 min |
| Pi0.5 (A100 40GB) | ~35GB | 4 | ~5 min |
| Pi0 (4×A100) | ~28GB each | 32 total | ~1 min |

### OOM Solutions

```yaml
# Reduce batch size
dataloader:
  batch_size: 4

# Add gradient accumulation
training:
  gradient_accumulate_every: 2  # Effective batch = 4 × 2 = 8
```

## Evaluation

```bash
bash eval.sh --policy pi0 \
  --task LiftBarrier-rf \
  --config robofactory/configs/table/lift_barrier.yaml \
  --checkpoint epoch_300
```

## Wandb Logging

| Metric | Description |
|--------|-------------|
| `train_loss` | Flow matching loss |
| `val_loss` | Validation loss |
| `lr` | Learning rate |
| `epoch` | Current epoch |
| `input_images` | Sample observations |
