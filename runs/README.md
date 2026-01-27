# `runs/` (local outputs; not committed)

All experiment runners write **append-only** outputs under `runs/<run_id>/`:

- `run.json`: config + environment metadata (small)
- `events.jsonl`: per-replan (and optional phase) events
- `episode_metrics.jsonl`: per-episode outcomes
- `summary.json`: optional free-form summary (keep small)

This directory is **git-ignored** by default to avoid accidentally committing large logs.

Typical workflow:

```bash
# Run a tiny local smoke (no simulators)
scripts/smoke_local.sh

# Postprocess an existing run into paper-facing tables
scripts/postprocess_run.sh runs/<run_id>
```
