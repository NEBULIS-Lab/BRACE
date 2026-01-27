# Domain B: RoboFactory

This folder contains the **Domain B** runner for RoboFactory tasks.

## Requirements (not shipped in this repo)

- A working RoboFactory installation/workspace (env + assets)
- Python deps installed (see repo `requirements.txt`)

Recommended env vars:
- `BRACE_ROBOFACTORY_DATA_ROOT`: where RoboFactory/OpenMARL assets/checkpoints/caches live (machine-specific)

## Run

Minimal entrypoint (uses a small smoke config):

```bash
scripts/run_domain_b_robofactory.sh --config configs/smoke/robofactory_lift_barrier.json --run-name robofactory_smoke
```

You can switch to curated configs under `configs/experiments/` for demos/ablations.

## Outputs

- Run logs: `runs/<run_id>/{run.json,events.jsonl,episode_metrics.jsonl}`
- Aggregate tables: `scripts/postprocess_run.sh runs/<run_id>` (writes to `artifacts/tables/`, git-ignored)

