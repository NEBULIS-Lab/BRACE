# OpenVLA Integration for RoboFactory

Integration of **OpenVLA** (Vision-Language-Action) model for multi-agent robotic manipulation with LoRA fine-tuning.

## Docker

```bash
docker pull christianlin0420/openmarl:openvla
docker run --gpus all -it -v $(pwd):/workspace/OpenMARL christianlin0420/openmarl:openvla
```

## Quick Start

```bash
# 1. Prepare data (inside Docker)
cd /workspace/OpenMARL/robofactory
bash prepare_all_data.sh --num 150 --task LiftBarrier

# 2. Train
bash train.sh --policy openvla --task LiftBarrier-rf --agent_id 0

# 3. Evaluate
bash eval.sh --policy openvla --task LiftBarrier-rf --config robofactory/configs/table/lift_barrier.yaml
```

## Model Features

- **Base Model**: OpenVLA-7B from HuggingFace
- **Fine-tuning**: LoRA (rank=32, 1.05% trainable params)
- **Precision**: BFloat16
- **Vision**: Multi-view (3 cameras, token-level concatenation)
- **Action Space**: 8-DoF

## Camera Mapping

| OpenVLA Key | Source | View |
|-------------|--------|------|
| `primary` | Agent `head_camera` | Side |
| `secondary` | Global `head_camera` | Overhead |
| `wrist` | Agent `wrist_camera` | Gripper |

## Training Options

```bash
# Multi-GPU (8 GPUs)
bash train.sh --policy openvla --task LiftBarrier-rf --agent_id 0 --gpus 8

# Custom settings
bash train.sh --policy openvla --task LiftBarrier-rf --agent_id 0 \
  --batch_size 8 --seed 42 --wandb online
```

## Documentation

- **[docs/DATA_PROCESSING.md](docs/DATA_PROCESSING.md)** - ZARR to RLDS conversion
- **[docs/TRAINING.md](docs/TRAINING.md)** - Model architecture & training

## Directory Structure

```
OpenVLA/
├── train.py                      # Training entry point
├── eval_multi_openvla.py        # Evaluation script
├── openvla_policy/
│   ├── config/                  # Hydra configs
│   ├── dataset/                 # RLDS dataset loader
│   ├── model/openvla_wrapper.py # OpenVLA model with LoRA
│   ├── utils/data_conversion.py
│   └── workspace/               # Training loop
└── docs/
    ├── DATA_PROCESSING.md
    └── TRAINING.md
```

## Memory Requirements

| Config | GPU Memory | Batch Size |
|--------|------------|------------|
| Single A100 40GB | ~26GB | 16 |
| 8× A100 40GB | ~26GB each | 8/GPU |

## Citation

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={Kim, Moo Jin and others},
    journal={arXiv preprint arXiv:2406.09246},
    year={2024}
}
```
