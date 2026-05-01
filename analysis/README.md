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

## Table and audit utility index

Use the following entrypoints when you want a specific paper-facing table or audit artifact:

- `analysis/phase_breakdown_table.py`: phase-latency breakdown tables from `event_type="phase"` logs
- `analysis/trigger_audit_table.py`: trigger suppression / override audit tables
- `analysis/clarification_table.py`: compact cross-domain summary tables for public-facing docs
- `analysis/rag_prune_2x2_table.py`: retrieval/pruning comparison tables
- `analysis/schema_check.py`: strict schema validation for a run or runs root
- `analysis/schema_coverage.py`: field-presence summary across a runs root
- `analysis/trigger_field_coverage.py`: trigger-controller field coverage audit
- `analysis/controller_field_coverage.py`: controller accounting field coverage audit
- `analysis/export_icml2026.py`: export helper for paper-facing artifacts

## Per-environment tables

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

## Curated experiment manifests

These configs provide stable, named entrypoints for the main public examples and analysis slices:

- `configs/experiments/habitat_llm_tails_slo.json`
- `configs/experiments/habitat_budgetmatch_baselines_oracle30ep.json`
- `configs/experiments/habitat_llm_2x2_oracle30ep.json`
- `configs/experiments/robofactory_pass_shoe_2x2.json`
- `configs/experiments/robofactory_budgetmatch_baselines_pass_shoe_10ep_b128.json`
- `configs/experiments/robofactory_rag_prune_2x2_pass_shoe_10ep_b128.json`
- `configs/experiments/robofactory_demo_pass_shoe_baseline_vs_brace_erecap.json`
- `configs/experiments/airsim_k1_showcase.json`
- `configs/experiments/airsim_k8_ablation_2x2.json`
- `configs/experiments/proxy_frequency_sweep_steps.json`
- `configs/experiments/proxy_stability_sweep_p0.json`

For consistency, prefer these manifests over ad hoc JSON edits when generating public tables, screenshots, or demo assets.
