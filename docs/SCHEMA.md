# Run Logging Schema (BRACE-code)

Paper-facing schema docs:
- `analysis/logging_schema.md` (authoritative field definitions)
- `analysis/logging_schema_quickref.md` (1-page quick reference)

This repo uses a lightweight, auditable on-disk run format:

```
runs/<run_id>/
  run.json
  events.jsonl
  episode_metrics.jsonl
  summary.json            # optional
```

The schema is designed to support:
- strict completeness checks (`analysis/schema_check.py`)
- best-effort aggregation into paper-style tables (`analysis/aggregate_runs.py`)
- phase-level cost accounting (e.g., VLM/VLA/prompt-compress overhead) via `event_type="phase"`

## Versioning

`run.json` stores schema versions and a human-readable freeze tag:
- `schema.events_jsonl`
- `schema.episode_metrics_jsonl`
- `schema_tag`

See `experiments/common/logging.py` (`EVENT_SCHEMA_VERSION`, `EPISODE_SCHEMA_VERSION`, `SCHEMA_TAG`).

## `events.jsonl`

Each line is a JSON dict. Two record types share the same file:

### 1) Replanning events (`event_type="replan"`)

These are the main per-(episode,timestep) replanning calls used for token/latency/SLO accounting.

**Strict required fields** (used by `analysis/schema_check.py`):
- `run_id`, `time_utc`, `domain`, `variant`, `episode_id`, `t`
- `tokens_in`, `tokens_after_prune`
- `lat_total_ms`

**Common optional fields** (recommended for paper tables / auditing):
- toggles: `brace_enabled`, `pruning_enabled`, `rag_enabled`, `summary_compress_enabled`
- budgets: `slo_ms`, `token_budget`, `clarification_budget_turns`
- mode: `mode` (e.g., `partial_replan`, `full_replan`, `reuse_subplan`, `defer_replan`)
- latency breakdown: `lat_prune_ms`, `lat_retrieval_ms`, `lat_prefill_ms`, `lat_decode_ms`
- derived tail/SLO fields: `slo_violation`, `slo_over_ms` (auto-derived when `lat_total_ms` and `slo_ms` exist; see `experiments/common/logging.py`)
- stability signals: `deadlock_flag`, `wait_time_ms`, `plan_hash`, `plan_churn_score`

**Minimal example**
```json
{"event_type":"replan","run_id":"...","time_utc":"...","domain":"robofactory","variant":"brace_prune_r0.7","episode_id":"ep0000","t":12,"tokens_in":1800,"tokens_after_prune":1200,"lat_total_ms":230.5}
```

### 2) Phase events (`event_type="phase"`)

Phase events are used to split a single replanning call into sub-phases (e.g., summarize → compress → planner → VLA policy),
enabling **auditable overhead breakdown** without conflating phases into a single `lat_total_ms`.

Phase events are identified by either:
- `event_type="phase"`, or
- a record containing `phase` and no timestep-like keys (back-compat inference in `analysis/schema_check.py` / `analysis/aggregate_runs.py`)

**Common fields**
- `phase` (string; e.g., `vlm_summarize`, `context_compress`, `planner_call`, `vla_policy_call`)
- `lat_total_ms` (phase duration in ms; convention used by the analysis scripts)

**Optional VLM/VLA accounting fields**
- `vlm_model`, `vlm_tokens_in`, `vlm_tokens_out`, `lat_vlm_ms`

**Minimal example**
```json
{"event_type":"phase","run_id":"...","time_utc":"...","domain":"robofactory","variant":"openvla","episode_id":"ep0000","t":12,"phase":"vla_policy_call","lat_total_ms":45.1,"vlm_model":"OpenVLA","vlm_tokens_in":18,"vlm_tokens_out":3}
```

## `episode_metrics.jsonl`

Each line is a JSON dict summarizing one episode.

**Strict required fields** (used by `analysis/schema_check.py`):
- `run_id`, `time_utc`, `domain`, `variant`, `episode_id`
- `success`, `step_count`, `replan_cycles`

**Minimal example**
```json
{"run_id":"...","time_utc":"...","domain":"robofactory","variant":"brace_prune_r0.7","episode_id":"ep0000","success":1.0,"step_count":312,"replan_cycles":28}
```

## Audit tools

- `analysis/schema_check.py`: strict required-field checks (replan + episode rows)
- `analysis/schema_coverage.py`: field coverage stats across a run (best-effort)
- `analysis/aggregate_runs.py`: aggregated tables + phase breakdown (best-effort; warns on missing keys)
