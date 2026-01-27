# Local outputs (generated at runtime; not tracked in git)

This repo is **code/configs/docs only**. The following directories are created **after you run experiments or utilities locally** and are intentionally **not committed** to GitHub.

## `runs/` (local logs; not committed)

Experiment runners write **append-only** outputs under `runs/<run_id>/`:

- `run.json`: config + environment metadata (small)
- `events.jsonl`: per-replan (and optional phase) events
- `episode_metrics.jsonl`: per-episode outcomes
- `summary.json`: optional free-form summary (keep small)

Typical workflow:

```bash
# Run a tiny local smoke (no simulators)
scripts/smoke_local.sh

# Postprocess an existing run into paper-facing tables
scripts/postprocess_run.sh runs/<run_id>
```

## `artifacts/` (derived outputs; not committed)

Postprocessing writes **append-only** derived outputs under `artifacts/`:

- `artifacts/tables/`: markdown + JSON summaries produced by `analysis/*_table.py` and `analysis/aggregate_runs.py`
- `artifacts/demos/`: optional demo exports (videos/screenshots) *outside git*
- `artifacts/figures/`: optional figures *outside git*

Policy: keep committed website/media assets under `docs/static/` only.

## `data/` (local intermediate files; optional; not committed)

Some utilities may write small intermediate files under `data/` (still considered **local-only**).

Example:
- `experiments/robofactory/instructions/generate_robofactory_instruction_manifest.py` defaults to
  `data/robofactory_instructions/*` for generated JSONL manifests.

