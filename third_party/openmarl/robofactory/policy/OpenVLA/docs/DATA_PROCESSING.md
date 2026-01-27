# OpenVLA Data Processing

Convert RoboFactory ZARR data to RLDS format for OpenVLA training.

## Data Format

### Input: ZARR

**Agent ZARR** (`{TASK}_Agent{ID}_{NUM}.zarr`):
- `head_camera`: Side view
- `wrist_camera`: Gripper view
- `state`: Robot state (8-dim)
- `action`: Actions (8-dim)

**Global ZARR** (`{TASK}_global_{NUM}.zarr`):
- `head_camera`: Overhead view

### Output: RLDS (TFRecord)

| Key | Description |
|-----|-------------|
| `observation/image_primary` | Side view (224×224) |
| `observation/image_secondary` | Overhead view (224×224) |
| `observation/image_wrist` | Gripper view (224×224) |
| `observation/proprio` | Robot state (8-dim) |
| `action` | Actions (8-dim) |
| `language_instruction` | Task description |

## Conversion

### Automatic (Recommended)

```bash
cd /workspace/OpenMARL/robofactory
bash prepare_all_data.sh --num 150 --task LiftBarrier
```

### Manual

```bash
python robofactory/policy/OpenVLA/openvla_policy/utils/data_conversion.py \
    --zarr_path robofactory/data/zarr_data/LiftBarrier-rf_Agent0_150.zarr \
    --output_dir robofactory/data/rlds_data

# Batch conversion
python robofactory/policy/OpenVLA/openvla_policy/utils/data_conversion.py \
    --zarr_path robofactory/data/zarr_data \
    --output_dir robofactory/data/rlds_data \
    --batch
```

## Output Structure

```
robofactory/data/rlds_data/
└── LiftBarrier-rf_Agent0_150/
    ├── 0.0.0/
    │   └── train.tfrecord-*
    └── dataset_info.json
```

## Language Instructions

| Task | Instruction |
|------|-------------|
| LiftBarrier-rf | "Lift the barrier together with the other robot" |
| TwoRobotsStackCube-rf | "Stack the cube on the target location" |
| PassShoe-rf | "Pass the shoe to the other robot" |

## Normalization Statistics

Generated automatically during conversion:
- Action mean/std
- Proprio mean/std

Stored in `dataset_statistics.json`.

## Verification

```bash
# Check dataset exists
ls robofactory/data/rlds_data/LiftBarrier-rf_Agent0_150/

# Test loading
python -c "
import tensorflow_datasets as tfds
ds = tfds.load('LiftBarrier-rf_Agent0_150', data_dir='robofactory/data/rlds_data')
print(f'Loaded: {ds}')
"
```

## Troubleshooting

**Dataset not found:**
```bash
# Regenerate
rm -rf robofactory/data/rlds_data/LiftBarrier-rf_Agent0_*
bash prepare_all_data.sh --num 150 --task LiftBarrier
```

**Symlink issues:**
```bash
cd robofactory/data/rlds_data
ln -s LiftBarrier-rf_Agent0 LiftBarrier-rf_Agent0_150
```

