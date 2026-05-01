# External benchmark and backbone notes

This repository keeps the BRACE controller, configs, and analysis code in one place. Some benchmark-specific demo scripts and local benchmark workspaces live outside this repo and are referenced here for reproducibility.

## RoboSuite (MuJoCo)

Primary local workspace:
- `/data/private/user2/workspace/robosuite_learning`

Reference note:
- `/data/private/user2/ref/ROBOSUITE.md`

Useful local scripts:
- `/data/private/user2/workspace/robosuite_learning/06_scripts/render_offscreen_video.py`
- `/data/private/user2/workspace/robosuite_learning/06_scripts/playback_npz_demo_to_video.py`
- `/data/private/user2/workspace/robosuite_learning/06_scripts/collect_random_demos_npz.py`
- `/data/private/user2/workspace/robosuite_learning/06_scripts/capture_mujoco_viewer_ui.py`

Relevant local task assets and demos:
- `TwoArmPegInHole`
  - `/data/private/user2/workspace/robosuite_learning/07_demos/github_demo_video_recording__TwoArmPegInHole__PandaPanda__frontview.mp4`
  - `/data/private/user2/workspace/robosuite_learning/07_demos/npz_playback__TwoArmPegInHole__PandaPanda__frontview.mp4`
  - `/data/private/user2/workspace/robosuite_learning/05_related/assets/robosuite_twoarm_peginhole_expert_v0.h5`
- `TwoArmHandover`
  - `/data/private/user2/workspace/robosuite_learning/07_demos/github_demo_segmentation__TwoArmHandover__frontview__element.mp4`
  - `/data/private/user2/workspace/robosuite_learning/07_demos/twoarmhandover_random_v152.mp4`

Practical note:
- These scripts are not vendored into `BRACE-code`; they remain in the dedicated RoboSuite workspace because they depend on that local MuJoCo / robosuite setup.

## LIBERO

Primary local workspace:
- `/data/private/user2/workspace/embodied_platforms/libero_env`

Reference notes:
- `/data/private/user2/ref/LIBERO_BENCHMARKS.md`
- `/data/private/user2/ref/LIBERO_DATASETS.md`

LIBERO code root:
- `/data/private/user2/workspace/embodied_platforms/libero_env/src/LIBERO`

Useful local scripts:
- `/data/private/user2/workspace/embodied_platforms/libero_env/src/LIBERO/benchmark_scripts/check_task_suites.py`
- `/data/private/user2/workspace/embodied_platforms/libero_env/src/LIBERO/benchmark_scripts/render_single_task.py`
- `/data/private/user2/workspace/embodied_platforms/libero_env/src/LIBERO/scripts/collect_demonstration.py`
- `/data/private/user2/workspace/embodied_platforms/libero_env/src/LIBERO/scripts/create_libero_task_example.py`

Available local suites:
- `libero_10`
- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_90`
- `libero_100`

Task-level note:
- `Moka Pot` is handled as a task inside the relevant LIBERO suite rather than through a standalone script. In practice, the usual workflow is:
  1. enumerate the suite task list, then
  2. render or evaluate the selected task with the benchmark scripts above.

## PushT

Primary local workspace:
- `/data/private/user2/workspace/plato_workspace`

Reference note:
- `/data/private/user2/ref/PUSHT.md`

Main local entrypoint:
- `/data/private/user2/workspace/plato_workspace/scripts/lerobot_eval.py --env-type pusht`

Supporting local launcher:
- `/data/private/user2/workspace/plato_workspace/local/slurm/eval_hf_jubba_smolvla_pusht_seed1_n20_nas4.sbatch`

Practical note:
- PushT is maintained in a separate local evaluation workspace and is not wired into the `BRACE-code` shell wrappers.

## Qwen-backed local assets

Primary local asset root:
- `/data/private/user2/workspace/VLA/qwen3`

Related local assets and evaluation code:
- `/data/private/user2/workspace/RoboFactory_workspace/rf_run_tools/run_openmarl_openvla_eval_one.sh`
- `/data/private/user2/workspace/VLA/baseline_assets/official/vla0/eval/eval_libero.py`
- `/data/private/user2/workspace/VLA/baseline_assets/official/openvla-oft/LIBERO.md`

Practical note:
- These assets are useful when tracing local VLA / backbone dependencies, but the BRACE repo does not vendor the Qwen workspace itself.

## Integration boundary

What lives in `BRACE-code`:
- controller logic
- curated experiment configs
- platform wrappers
- postprocess / audit utilities
- project docs and public media

What stays external:
- benchmark-native workspaces and demo helpers
- heavyweight simulator installs
- model checkpoints and large datasets

This split keeps the public repository focused while still documenting where benchmark-native tooling lives locally.
