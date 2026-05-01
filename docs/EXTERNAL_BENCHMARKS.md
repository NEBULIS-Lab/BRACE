# External benchmark integration notes

This repository now vendors lightweight helper entrypoints for benchmark-native rendering and inspection workflows that are useful when working with BRACE qualitative assets.

The goal is narrow:
- keep the public repo path-safe and self-contained,
- expose the thin wrappers we actually use, and
- avoid vendoring full simulator installations, datasets, or checkpoints.

## RoboSuite

Repository entrypoints:
- Python helpers:
  - `experiments/robosuite/render_offscreen_video.py`
  - `experiments/robosuite/playback_npz_demo_to_video.py`
  - `experiments/robosuite/collect_random_demos_npz.py`
- Shell wrappers:
  - `scripts/run_robosuite_render.sh`
  - `scripts/run_robosuite_playback.sh`
  - `scripts/run_robosuite_collect.sh`

Supported use cases:
- render a short offscreen rollout for a RoboSuite task
- play back a saved `ep_*` demonstration directory into MP4
- collect random `DataCollectionWrapper` episodes into `ep_*` folders

Requirements:
- a working `robosuite` install in the active Python environment
- MuJoCo headless rendering support for offscreen rendering

Example:

```bash
scripts/run_robosuite_render.sh \
  --env TwoArmPegInHole \
  --robots Panda Panda \
  --camera frontview \
  --steps 240 \
  --out artifacts/robosuite/twoarm_peginhole.mp4
```

## LIBERO

Repository entrypoints:
- Python helpers:
  - `experiments/libero/render_single_task.py`
  - `experiments/libero/check_task_suites.py`
- Shell wrappers:
  - `scripts/run_libero_render.sh`
  - `scripts/run_libero_check.sh`

Supported use cases:
- inspect benchmark suites and task names
- verify that referenced BDDL / init-state / demo assets exist
- render a single task snapshot panel

Requirements:
- either a working `libero` install in the active Python environment, or
- `BRACE_LIBERO_ROOT` / `LIBERO_ROOT` pointing to a local LIBERO workspace

Example:

```bash
scripts/run_libero_check.sh --verify-files
```

Task note:
- tasks such as `Moka Pot` are handled as suite members; enumerate the suite first, then render or evaluate the selected `task_id`.

## PushT

This repository does not vendor PushT-specific helpers.

Current boundary:
- BRACE keeps controller logic, configs, and analysis code here
- benchmark-native PushT evaluation remains an external workflow

## Qwen-backed assets

This repository does not vendor model weights or backbone workspaces.

Current boundary:
- BRACE exposes controller-side experiments and wrappers
- backbone checkpoints, evaluation harnesses, and large model assets stay outside git

## Integration boundary

What lives here:
- BRACE controller logic
- curated configs
- platform runners
- benchmark helper wrappers
- public docs and media

What stays external:
- simulator installs
- benchmark datasets
- checkpoints / weights
- large local outputs
