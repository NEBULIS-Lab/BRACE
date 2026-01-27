# Logging schema quick reference (for paper integration)

This is a 1-page summary of the most frequently used fields. Full details live in `analysis/logging_schema.md`.

## 1) `events.jsonl` (replanning-call events; `event_type="replan"`)

**Identifiers**
- `run_id`, `time_utc`, `domain`, `task`, `variant`, `episode_id`, `t`

**Budgets / controller**
- `slo_ms`, `token_budget`, `clarification_budget_turns`, `mode`
- toggles: `brace_enabled`, `pruning_enabled`, `rag_enabled`, `summary_compress_enabled`
- retrieval selector (optional): `rag_source`

**Tokens (planner input tokens)**
- `tokens_in` (before pruning/compress)
- `tokens_after_prune` (actually sent to planner)
- optional per-block: `tokens_task/tokens_state/tokens_safety/tokens_coord/tokens_history`
- retrieval (when `rag_enabled=1`): `retrieved_tokens`, `kept_tokens`
- retrieval audit (optional): `retrieval_note`

**Latency (ms)**
- `lat_total_ms` (inclusive replanning latency)
- optional split: `lat_prune_ms`, `lat_retrieval_ms`, `lat_prefill_ms`, `lat_decode_ms`
- derived: `slo_violation`, `slo_over_ms`

**Frequency / triggers**
- `replan_interval_steps`, `trigger_cooldown_steps`, `replan_trigger_type`, optional `trigger` dict

**Stability / multi-agent**
- `plan_hash`, `plan_churn_score`
- `deadlock_flag`, `wait_time_ms`, `locks_held`, `locks_waiting`

## 2) `events.jsonl` phase events (`event_type="phase"`)

Purpose: split accounting for overheads and VLM/VLA tracks without polluting replanning-call tails.

**Core**
- same identifiers (`run_id/domain/variant/episode_id/t`) when joinable
- `phase` (string), plus optional `phase_idx`, `phase_parent`
- `lat_total_ms` is the phase duration
- VLA runs: if `phase="vla_policy_call"` exists, `analysis/aggregate_runs.py` reports VLA-aware **control-loop** latency by using
  `vla_policy_call` as the per-step anchor and adding replanning latency since the previous VLA call.

**Recommended phase names (canonical)**
- `context_compress`, `planner_call`, `clarification`, `vla_policy_call`
- demo artifacts: `demo_start/demo_end/demo_compare_end/demo_manifest_written`

**VLM/VLA accounting (optional)**
- `vlm_model`, `vlm_tokens_in`, `vlm_tokens_out`, `lat_vlm_ms`

**Demo artifact metadata (optional)**
- `path` (mp4/png/json), optional `fps`, `frame_wh`, `probe`

## 3) `episode_metrics.jsonl` (one line per episode)

**Identifiers/outcomes**
- `run_id`, `time_utc`, `domain`, `variant`, `episode_id`, `success`, `step_count`, `replan_cycles`
- `spl` is Habitat-only and may be null in non-Habitat domains.

**Optional per-episode summaries (runner- or analysis-computed)**
- `lat_p50_ms/lat_p95_ms/lat_p99_ms`
- `slo_violation_rate`
- `tokens_in_mean/tokens_in_p95/tokens_in_p99`
- `tokens_after_prune_mean/tokens_after_prune_p95/tokens_after_prune_p99`
