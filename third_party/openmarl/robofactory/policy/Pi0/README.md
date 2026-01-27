# Pi0/Pi0.5 Integration for RoboFactory

Integration of Physical Intelligence's **Pi0** and **Pi0.5** Vision-Language-Action models for multi-agent robotic manipulation.

## Docker

```bash
docker pull christianlin0420/openmarl:pi
docker run --gpus all -it -v $(pwd):/workspace/OpenMARL christianlin0420/openmarl:pi
```

## Quick Start

```bash
# 1. Prepare data (inside Docker)
cd /workspace/OpenMARL/robofactory
bash prepare_all_data.sh --num 150 --task LiftBarrier

# 2. Train
bash train.sh --policy pi0 --task LiftBarrier-rf --agent_id 0

# 3. Evaluate
bash eval.sh --policy pi0 --task LiftBarrier-rf --config robofactory/configs/table/lift_barrier.yaml
```

## Model Variants

| Model | Token Length | Best For |
|-------|--------------|----------|
| **Pi0** | 48 | Faster training, lower memory |
| **Pi0.5** | 200 | Better performance |

## Camera Mapping

| Pi0 Key | Source | View |
|---------|--------|------|
| `base_0_rgb` | Agent ZARR `head_camera` | Side |
| `left_wrist_0_rgb` | Global ZARR `head_camera` | Overhead |
| `right_wrist_0_rgb` | Agent ZARR `wrist_camera` | Gripper |

## Training Options

```bash
# Multi-GPU
bash train.sh --policy pi0 --task LiftBarrier-rf --agent_id 0 --gpus 4

# Pi0.5
bash train.sh --policy pi05 --task LiftBarrier-rf --agent_id 0

# Custom settings
bash train.sh --policy pi0 --task LiftBarrier-rf --agent_id 0 \
  --batch_size 16 --seed 42 --wandb online
```

## Documentation

- **[docs/DATA_PROCESSING.md](docs/DATA_PROCESSING.md)** - ZARR to LeRobot conversion
- **[docs/TRAINING.md](docs/TRAINING.md)** - Model architecture & training

## Directory Structure

```
Pi0/
├── train.py                    # Training entry point
├── eval_multi_pi0.py          # Evaluation script
├── pi0_policy/
│   ├── config/                # Hydra configs
│   ├── dataset/               # LeRobot dataset loader
│   ├── model/pi0_wrapper.py   # OpenPI model wrapper
│   ├── utils/data_conversion.py
│   └── workspace/             # Training loop
└── docs/
    ├── DATA_PROCESSING.md
    └── TRAINING.md
```

## Citation

```bibtex
@article{black2024pi0,
  title={$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control},
  author={Black, Kevin and others},
  journal={arXiv preprint arXiv:2410.24164},
  year={2024}
}
```
