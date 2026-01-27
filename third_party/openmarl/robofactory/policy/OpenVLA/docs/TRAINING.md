# OpenVLA Training Pipeline

## Model Overview

**OpenVLA** (Open Vision-Language-Action) is a 7B parameter VLA model that combines a Prismatic vision-language model with action prediction capabilities. It uses **LoRA fine-tuning** for efficient adaptation to new tasks while preserving pretrained knowledge.

### Key Features
- **Large-scale pretraining**: Trained on 970k robot demonstrations
- **LoRA fine-tuning**: Only 1.05% parameters trained (efficient)
- **Multi-view support**: Token-level fusion of multiple camera views
- **Language-conditioned**: Natural language task instructions

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OpenVLA Model Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                       │
│   │  Side View  │   │Overhead View│   │ Wrist View  │                       │
│   │  (224×224)  │   │  (224×224)  │   │  (224×224)  │                       │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                       │
│          │                 │                 │                              │
│          ▼                 ▼                 ▼                              │
│   ┌─────────────────────────────────────────────────────────┐               │
│   │              Fused Vision Backbone                      │               │
│   │  ┌───────────────────┐  ┌───────────────────┐           │               │
│   │  │     DINOv2        │  │      SigLIP       │           │               │
│   │  │   (ViT-L/14)      │  │  (ViT-SO400M/14)  │           │               │
│   │  │                   │  │                   │           │               │
│   │  │ Self-supervised   │  │   Contrastive     │           │               │
│   │  │ spatial features  │  │ semantic features │           │               │
│   │  └─────────┬─────────┘  └─────────┬─────────┘           │               │
│   │            │                      │                     │               │
│   │            └──────────┬───────────┘                     │               │
│   │                       ▼                                 │               │
│   │            ┌─────────────────────┐                      │               │
│   │            │  Feature Concat     │                      │               │
│   │            │  (1024 + 1024)      │                      │               │
│   │            └─────────┬───────────┘                      │               │
│   │                      │                                  │               │
│   └──────────────────────┼──────────────────────────────────┘               │
│                          │                                                  │
│                          ▼                                                  │
│   ┌──────────────────────────────────────────────────────────┐              │
│   │                    MLP Projector                         │              │
│   │                  2048 → 4096 dim                         │              │
│   │           (Maps vision to LLM embedding space)           │              │
│   └────────────────────────┬─────────────────────────────────┘              │
│                            │                                                │
│                            ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │                     Token Sequence Construction                     │   │
│   │                                                                     │   │
│   │  ┌─────┬─────────────────────────────────────────┬──────────────┐   │   │
│   │  │ BOS │        Visual Tokens (771)              │  Text Tokens │   │   │
│   │  │ (1) │  [View1: 257] [View2: 257] [View3: 257] │    (~20)     │   │   │
│   │  └─────┴─────────────────────────────────────────┴──────────────┘   │   │
│   │                                                                     │   │
│   │              Total: ~792 tokens                                     │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                            │                                                │
│                            ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │                   Llama-2 7B Language Model                         │   │
│   │                                                                     │   │
│   │   ┌──────────────────────────────────────────────────────────────┐  │   │
│   │   │                    Transformer Layers                        │  │   │
│   │   │  ┌────────────────────────────────────────────────────────┐  │  │   │
│   │   │  │  Self-Attention + LoRA Adapters (rank=32)              │  │  │   │
│   │   │  │  • Query/Key/Value projections with LoRA               │  │  │   │
│   │   │  │  • Output projection with LoRA                         │  │  │   │
│   │   │  └────────────────────────────────────────────────────────┘  │  │   │
│   │   │  ┌────────────────────────────────────────────────────────┐  │  │   │
│   │   │  │  Feed-Forward Network                                  │  │  │   │
│   │   │  └────────────────────────────────────────────────────────┘  │  │   │
│   │   │                         × 32 layers                          │  │   │
│   │   └──────────────────────────────────────────────────────────────┘  │   │
│   │                                                                     │   │
│   └────────────────────────────────────────────────────────────────┬────┘   │
│                                                                    │        │
│                                                                    ▼        │
│   ┌──────────────────────────────────────────────────────────────-──────┐   │
│   │                      Action Token Head                              │   │
│   │                                                                     │   │
│   │   LLM Output → Linear → Discretized Action Bins → Action (8-dim)    │   │
│   │                                                                     │   │
│   └────────────────────────────────────────────────────────────────────-┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Fused Vision Backbone

OpenVLA uses **two complementary vision encoders**:

```
┌─────────────────────────────────────────────────────────────┐
│                   Dual Vision Encoder                       │
├──────────────────────────┬──────────────────────────────────┤
│        DINOv2            │           SigLIP                 │
├──────────────────────────┼──────────────────────────────────┤
│ • Self-supervised        │ • Contrastive (image-text)       │
│ • Spatial understanding  │ • Semantic understanding         │
│ • Object localization    │ • Concept recognition            │
│ • 1024-dim features      │ • 1024-dim features              │
├──────────────────────────┴──────────────────────────────────┤
│              Concatenated: 2048-dim features                │
└─────────────────────────────────────────────────────────────┘
```

| Property | DINOv2 | SigLIP |
|----------|--------|--------|
| Architecture | ViT-L/14 | ViT-SO400M/14 |
| Training | Self-supervised | Contrastive |
| Strength | Spatial features | Semantic features |
| Output dim | 1024 | 1024 |

### 2. Multi-View Token Processing

**Critical**: Each view is processed **separately** at full resolution, then tokens are concatenated:

```
┌─────────────────────────────────────────────────────────────┐
│                Multi-View Token Concatenation               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ❌ WRONG: Pixel-space concatenation                        │
│  ┌────────┬────────┬────────┐                               │
│  │ View 1 │ View 2 │ View 3 │ → Resize 224×672 → 224×224    │
│  └────────┴────────┴────────┘   (67% information loss!)     │
│                                                             │
│  ✅ CORRECT: Token-space concatenation                      │
│  View 1 (224×224) → Vision Encoder → 257 tokens             │
│  View 2 (224×224) → Vision Encoder → 257 tokens             │
│  View 3 (224×224) → Vision Encoder → 257 tokens             │
│                            ↓                                │
│              Concatenate: 771 visual tokens                 │
│              (Zero information loss!)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. LoRA Adaptation

```
┌─────────────────────────────────────────────────────────────┐
│                    LoRA Configuration                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Original Weight: W ∈ R^(d×k)                               │
│                                                             │
│  LoRA Decomposition:                                        │
│  W' = W + BA   where B ∈ R^(d×r), A ∈ R^(r×k)               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Parameter     │  Value                             │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  Rank (r)      │  32                                │    │
│  │  Alpha         │  64                                │    │
│  │  Dropout       │  0.0                               │    │
│  │  Target        │  All attention layers              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Total Parameters:     7,621,191,104 (7.6B)                 │
│  Trainable (LoRA):        79,953,920 (79.9M)                │
│  Trainable Ratio:              1.05%                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. Action Prediction

OpenVLA uses **discretized action tokens**:

```
┌─────────────────────────────────────────────────────────────┐
│                  Action Token Prediction                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Continuous Action Space:                                   │
│  ┌───────────────────────────────────────────────────┐      │
│  │ Dim 0-6: Joint positions (7 DoF)                  │      │
│  │ Dim 7:   Gripper command [-1, 1]                  │      │
│  └───────────────────────────────────────────────────┘      │
│                                                             │
│  Discretization:                                            │
│  • Each action dimension → 256 bins                         │
│  • Predicted as next-token in vocabulary                    │
│  • Cross-entropy loss for training                          │
│                                                             │
│  Normalization:                                             │
│  • Statistics computed from training data                   │
│  • Mean/std normalization before discretization             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Input/Output Processing

### Input Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Processing                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  IMAGES (3 views, processed separately)                     │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐      │
│  │HWC uint8│ → │ Resize  │ → │  Crop   │ → │Normalize│      │
│  │Variable │   │224×224  │   │  (90%)  │   │ [0, 1]  │      │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘      │ 
│                     │              │              │         │
│                     └──────────────┴──────────────┘         │
│                                    ▼                        │
│                          ┌─────────────────┐                │
│                          │ Vision Backbone │                │
│                          │ (per view)      │                │
│                          └────────┬────────┘                │
│                                   ▼                         │
│                          257 tokens × 3 views               │
│                          = 771 visual tokens                │
│                                                             │
│  LANGUAGE                                                   │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                    │
│  │  Task   │ → │Tokenize │ → │  Embed  │ → ~20 tokens       │
│  │Instruction│  │ (Llama) │   │         │                   │
│  └─────────┘   └─────────┘   └─────────┘                    │
│                                                             │
│  PROPRIO (optional)                                         │
│  ┌─────────┐   ┌─────────┐                                  │
│  │ 8-dim   │ → │Normalize│ → Concatenated with visual       │
│  └─────────┘   └─────────┘                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Output Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     Output Processing                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LLM Output Logits                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────┐                                        │
│  │ Action Token    │  8 tokens (one per action dim)         │
│  │ Extraction      │                                        │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Bin → Value     │  256 bins → continuous value           │
│  │ Conversion      │                                        │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Denormalize     │  Apply dataset statistics              │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  Final Action (8-dim float)                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Training

### Commands

```bash
# Single GPU
bash train.sh --policy openvla --task LiftBarrier-rf --agent_id 0

# Multi-GPU (8 GPUs)
bash train.sh --policy openvla --task LiftBarrier-rf --agent_id 0 --gpus 8
```

### Configuration (`robot_openvla.yaml`)

```yaml
model:
  model_name: "openvla/openvla-7b"
  use_lora: true
  lora_rank: 32
  lora_alpha: 64
  torch_dtype: "bfloat16"

training:
  num_epochs: 300
  learning_rate: 5.0e-4
  batch_size: 16
  image_aug: true
  augment_crop_ratio: 0.9
  checkpoint_every: 50
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
│          │ 1. Process multi-view images            │        │
│          │    (token-space concatenation)          │        │
│          └─────────────────┬───────────────────────┘        │
│                            ▼                                │
│          ┌─────────────────────────────────────────┐        │
│          │ 2. Construct multimodal sequence        │        │
│          │    [BOS][Visual][Text][Action]          │        │
│          └─────────────────┬───────────────────────┘        │
│                            ▼                                │
│          ┌─────────────────────────────────────────┐        │
│          │ 3. Forward through LLM with LoRA        │        │
│          │    Cross-entropy loss on action tokens  │        │
│          └─────────────────┬───────────────────────┘        │
│                            ▼                                │
│          ┌─────────────────────────────────────────┐        │
│          │ 4. Backward + optimizer step            │        │
│          │    Only LoRA parameters updated         │        │
│          └─────────────────────────────────────────┘        │
│                                                             │
│      if epoch % 50 == 0: save_checkpoint()                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Checkpoints

```
robofactory/checkpoints/openvla/{task}_Agent{id}_{num}/
├── epoch_50/
│   ├── adapter_model.safetensors   # LoRA weights only
│   ├── adapter_config.json         # LoRA config
│   └── training_state.pt           # Optimizer, scheduler
├── epoch_100/
│   └── ...
└── latest/
```

## Memory & Performance

| Config | GPU Memory | Batch Size | Time/Epoch |
|--------|------------|------------|------------|
| Single A100 40GB | ~26GB | 16 | ~30 min |
| 8× A100 40GB | ~26GB each | 8/GPU | ~4 min |

### OOM Solutions

```yaml
# Reduce batch size
dataloader:
  batch_size: 8

# Add gradient accumulation  
training:
  gradient_accumulate_every: 2  # Effective batch = 8 × 2 = 16
```

## Evaluation

```bash
bash eval.sh --policy openvla \
  --task LiftBarrier-rf \
  --config robofactory/configs/table/lift_barrier.yaml \
  --checkpoint epoch_300
```

## Wandb Logging

| Metric | Description |
|--------|-------------|
| `train_loss` | Cross-entropy loss on action tokens |
| `val_loss` | Validation loss |
| `lr` | Learning rate (cosine schedule) |
| `epoch` | Current epoch |
| `train/input_images` | Multi-view sample images |
