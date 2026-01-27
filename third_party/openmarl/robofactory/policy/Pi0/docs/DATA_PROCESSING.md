# Pi0 Data Processing

Convert RoboFactory ZARR data to LeRobot format for Pi0/Pi0.5 training.

## Data Format

### Input: ZARR

**Agent ZARR** (`{TASK}_Agent{ID}_{NUM}.zarr`):
- `head_camera`: Side view
- `wrist_camera`: Gripper view
- `state`: Robot state (8-dim)
- `action`: Actions (8-dim)

**Global ZARR** (`{TASK}_global_{NUM}.zarr`):
- `head_camera`: Overhead view

### Output: LeRobot

| Key | Source | Description |
|-----|--------|-------------|
| `base_0_rgb` | Agent `head_camera` | Side view |
| `left_wrist_0_rgb` | Global `head_camera` | Overhead view |
| `right_wrist_0_rgb` | Agent `wrist_camera` | Gripper view |
| `state` | Agent `state` | 8-dim float32 |
| `actions` | Agent `action` | 8-dim float32 |
| `task` | Auto-generated | Language instruction |

## Conversion

### Automatic (Recommended)

```bash
cd /workspace/OpenMARL/robofactory
bash prepare_all_data.sh --num 150 --task LiftBarrier
```

### Manual

```bash
cd /workspace/OpenMARL
PYTHONPATH=/workspace/OpenMARL python robofactory/policy/Pi0/pi0_policy/utils/data_conversion.py \
    --zarr_path=robofactory/data/zarr_data/LiftBarrier-rf_Agent0_150.zarr \
    --output_dir=robofactory/data/lerobot_data \
    --task_name=LiftBarrier-rf \
    --agent_id=0 \
    --num_episodes=150
```

## Output Structure

```
robofactory/data/lerobot_data/
└── LiftBarrier-rf_Agent0_150/
    ├── meta/
    │   ├── info.json
    │   └── episodes/chunk-000/file-000.parquet
    └── data/chunk-000/file-000.parquet
```

## Language Instructions

| Task | Instruction |
|------|-------------|
| LiftBarrier-rf | "Lift the barrier together with the other robot" |
| TwoRobotsStackCube-rf | "Stack the cube on the target location" |
| PassShoe-rf | "Pass the shoe to the other robot" |
| PlaceFood-rf | "Place the food on the plate" |

## Verification

```bash
python << 'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="LiftBarrier-rf_Agent0_150", root="robofactory/data/lerobot_data")
print(f"Frames: {len(ds)}, Keys: {list(ds[0].keys())}")
EOF
```

## Troubleshooting

**Parquet corruption:**
```bash
rm -rf robofactory/data/lerobot_data/LiftBarrier-rf_Agent0_*
rm -rf ~/.cache/huggingface/lerobot/*
bash prepare_all_data.sh --num 150 --task LiftBarrier
```

**Missing global view:** Ensure `{TASK}_global_{NUM}.zarr` exists in `zarr_data/`.

