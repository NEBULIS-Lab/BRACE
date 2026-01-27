# `artifacts/` (derived outputs; not committed)

Postprocessing writes **append-only** derived outputs under `artifacts/`:

- `artifacts/tables/`: markdown + JSON summaries produced by `analysis/*_table.py` and `analysis/aggregate_runs.py`
- `artifacts/demos/`: optional demo exports (videos/screenshots) *outside git*
- `artifacts/figures/`: optional figures *outside git*

This directory is **git-ignored** by default; keep committed assets under `docs/static/` only.
