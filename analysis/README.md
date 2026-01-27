# Analysis utilities

## Logging schema

See `analysis/logging_schema.md` for the authoritative field definitions for:
- `runs/<run_id>/events.jsonl`
- `runs/<run_id>/episode_metrics.jsonl`

## Aggregation

Aggregate a **single run**:
```bash
python analysis/aggregate_runs.py runs/<run_id> --write_tables
```

Aggregate a **runs root** (auto-discover subfolders containing `run.json`):
```bash
python analysis/aggregate_runs.py runs --write_tables
python analysis/aggregate_runs.py runs --pattern "20260122_*" --limit 5 --write_tables
```

Outputs are append-only under `artifacts/tables/` when using `--write_tables`.

If a run contains `event_type="phase"` events (see schema), `analysis/aggregate_runs.py` appends a small
**Phase latency breakdown** section to the per-run markdown output.
If a run contains `phase="vla_policy_call"`, `analysis/aggregate_runs.py` reports VLA-aware **control-loop latency** in the main
Lat columns (and adds a “VLA-aware latency accounting” audit section).

## Budget matching tables

Generate “Quality @ matched tokens” + “Systems @ matched quality” tables for a **single run**:
```bash
python analysis/budget_match_table.py runs/<run_id> --write_tables
```

## Domain tables (per-domain)

- RoboFactory/RoboCasa:
```bash
python analysis/domainb_table.py runs/<run_id> --write_tables
```

- Microsoft AirSim:
```bash
python analysis/domainc_table.py runs/<run_id> --write_tables
```

## Paper claim check (optional)

If you have table JSONs (e.g., `*__agg__*.json`) and want a quick numeric sanity check across comparisons:

1) Edit `configs/experiments/claim_check_manifest.json` to point to your table JSON(s).
2) Run:
```bash
python analysis/claim_check.py --write_tables
```
