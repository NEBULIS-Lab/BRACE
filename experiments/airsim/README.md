# Domain C: AirSim (Vehicles / Drones)

This folder contains the **Domain C** runner for AirSim-based vehicle/drone tasks.

## Requirements (not shipped in this repo)

- AirSim UE binary environments on your machine
- Python deps installed (see repo `requirements.txt`)

Recommended env vars:
- `BRACE_AIRSIM_ENVS_ROOT`: root folder that contains your AirSim UE environments

## Run

Minimal entrypoint (uses a small demo config):

```bash
scripts/run_domain_c_airsim.sh --config configs/smoke/airsim_multidrone_demo.json --run-name airsim_demo
```

You can switch to curated configs under `configs/experiments/` for showcase/ablation runs.

## Outputs

- Run logs: `runs/<run_id>/{run.json,events.jsonl,episode_metrics.jsonl}`
- Demo media (if enabled by config): `artifacts/demos/airsim/<run_id>/`

