# BRACE Logging Schema (v1)

This document defines the **audit-friendly, aggregatable** logging schema for each run under `runs/<run_id>/`.

- Event schema version: `EVENT_SCHEMA_VERSION=1` (stored per-event as `schema_version`)
- Episode schema version: `EPISODE_SCHEMA_VERSION=1` (stored per-episode as `schema_version`)
- Schema freeze tag: `SCHEMA_TAG="2026-01-26"` (stored per-run in `run.json["schema_tag"]`)
- Source of truth for defaults/normalization: `experiments/common/logging.py`

## Per-run files

| File | Granularity | Purpose |
|---|---|---|
| `run.json` | run | Config + environment metadata + schema versions |
| `events.jsonl` | replanning-call / phase | One JSON object per line (append-only) |
| `episode_metrics.jsonl` | episode | One JSON object per line (append-only) |
| `summary.json` | run | Optional final summary (free-form, but should stay small) |

## `events.jsonl` (one line per replanning call; plus optional phase events)

### Core identifiers (should always be present for `event_type="replan"`)

| Field | Type | Unit | Default | Meaning / measurement point |
|---|---|---:|---|---|
| `schema_version` | int | - | `1` | Schema version for this record |
| `time_utc` | str | ISO8601 | auto | Event timestamp |
| `run_id` | str | - | auto | Run identifier (must match directory name) |
| `event_type` | str | - | inferred | `"replan"` or `"phase"` |
| `domain` | str | - | from `run.json` | e.g. `"habitat"`, `"robofactory"` |
| `task` | str\|null | - | from `run.json` | Optional task name within domain |
| `variant` | str | - | - | Experimental condition label |
| `episode_id` | str\|int | - | - | Episode identifier |
| `t` | int | step | - | Environment timestep when replanning is invoked |

### `event_type="phase"` (optional; per-replan accounting / overhead breakdown)

Phase events are optional records that allow **split accounting** for VLM/VLA tracks and budget-matched baselines:
instead of collapsing everything into the replanning-call event, record separate phase timings (and tokens when applicable)
into the same `events.jsonl`.

Recommended practice:
- Include the same identifiers as replanning events (`run_id/domain/variant/episode_id/t`) so phases can be joined.
- Use `lat_total_ms` as the **phase duration** (you may also fill specialized fields like `lat_vlm_ms` for readability).
- Phase events are typically summarized separately by `analysis/aggregate_runs.py` (see “Phase latency breakdown”).
  - Special-case (VLA runs): if `phase="vla_policy_call"` exists, `analysis/aggregate_runs.py` reports **VLA-aware control-loop**
    latency by using `vla_policy_call` as the per-step anchor and adding any replanning latency since the previous VLA call.

| Field | Type | Unit | Default | Meaning |
|---|---|---:|---|---|
| `phase` | str\|null | - | `null` | Phase name, e.g. `vlm_summarize/context_compress/planner_call/executor_step` |
| `phase_idx` | int\|null | - | `null` | Optional within-replan ordering index (0,1,2,…) |
| `phase_parent` | str\|int\|null | - | `null` | Optional link to a parent replanning call (e.g., a `replan_cycle` id if the runner uses it) |

#### Optional VLM/VLA accounting fields (recommended when using a VLM/VLA module)

| Field | Type | Unit | Default | Meaning |
|---|---|---:|---|---|
| `vlm_model` | str\|null | - | `null` | VLM/VLA model id/name (free-form but stable) |
| `vlm_tokens_in` | int\|null | tokens | `null` | VLM/VLA input token count (or proxy; record method in `run.json`) |
| `vlm_tokens_out` | int\|null | tokens | `null` | VLM/VLA output token count (or proxy) |
| `lat_vlm_ms` | float\|null | ms | `null` | VLM/VLA call latency (should usually match `lat_total_ms` for this phase) |

#### Optional demo artifact fields (recommended when videos/screenshots are part of the experiment)

| Field | Type | Unit | Default | Meaning |
|---|---|---:|---|---|
| `path` | str\|null | - | `null` | Artifact path (e.g., demo mp4/png/json) |
| `fps` | int\|float\|null | - | `null` | Video FPS (if applicable) |
| `frame_wh` | list\|null | px | `null` | Frame width/height as `[w,h]` (if applicable) |
| `probe` | dict\|null | - | `null` | Optional probe/validator output (e.g., ffprobe summary) |

Recommended phase names for this purpose: `demo_start/demo_end/demo_compare_end/demo_manifest_written`, and for VLA
executor split-accounting: `vla_policy_call` (or an equivalent stable name).

### Condition toggles (for 2×2 and budget-matched baselines)

| Field | Type | Unit | Default | Meaning |
|---|---|---:|---|---|
| `brace_enabled` | bool\|null | - | `null` | Whether BRACE controller is enabled |
| `pruning_enabled` | bool\|null | - | `null` | Whether E-RECAP (or other pruning) is enabled |
| `rag_enabled` | bool\|null | - | `null` | Whether retrieval/RAG is enabled |
| `summary_compress_enabled` | bool\|null | - | `null` | Whether structured summary compression is enabled |
| `rag_source` | str\|null | - | `null` | Retrieval source selector, e.g. `static/memory_then_static/memory_only` (runner-defined) |

### Budgets + mode

| Field | Type | Unit | Default | Meaning |
|---|---|---:|---|---|
| `slo_ms` | float\|int\|null | ms | from `run.json` | Replanning latency SLO threshold |
| `token_budget` | int\|null | tokens | from `run.json` | Budget on *input* prompt tokens for the planner call |
| `clarification_budget_turns` | int\|null | turns | from `run.json` | Allowed clarification turns (budget) |
| `mode` | str\|null | - | `null` | BRACE mode, e.g. `full_replan/partial_replan/reuse_subplan/defer_replan` |

### Replanning schedule + trigger metadata (optional, but recommended when studying replanning frequency)

| Field | Type | Unit | Default | Meaning |
|---|---|---:|---|---|
| `replan_interval_steps` | int\|null | steps | from `run.json` | Periodic replanning interval (if applicable) |
| `trigger_cooldown_steps` | int\|null | steps | from `run.json` | Cooldown between replans; triggers may be suppressed during cooldown |
| `replan_trigger_type` | str\|null | - | `null` | Primary trigger type for this replan, e.g. `periodic/failure/deadlock/unsafe/unknown` |
| `trigger` | dict\|null | - | `null` | Optional structured trigger dict (recommended keys: `periodic/deadlock/unsafe/types`) |

### Token accounting (planner-call *input* tokens)

| Field | Type | Unit | Default | Meaning / measurement point |
|---|---|---:|---|---|
| `tokens_in` | int\|null | tokens | `null` | Input tokens **before** pruning/compression |
| `tokens_after_prune` | int\|null | tokens | `null` | Input tokens actually passed to the planner **after** pruning/compression |
| `tokens_task` | int\|null | tokens | `null` | Optional block-level accounting (task) |
| `tokens_state` | int\|null | tokens | `null` | Optional block-level accounting (state) |
| `tokens_safety` | int\|null | tokens | `null` | Optional block-level accounting (safety) |
| `tokens_coord` | int\|null | tokens | `null` | Optional block-level accounting (coordination) |
| `tokens_history` | int\|null | tokens | `null` | Optional block-level accounting (history) |
| `retrieved_tokens` | int\|null | tokens | `null` | Retrieval payload size (tokens; proxy allowed if exact tokenization is unavailable) |
| `kept_tokens` | int\|null | tokens | `null` | Retrieval payload retained after pruning/compression (tokens; proxy allowed) |
| `retrieval_note` | str\|null | - | `null` | Optional retrieval audit note (e.g., corpus id, k, picked) |

Notes:
- Clarification tokens/latency should be logged **separately** (do not merge into `tokens_in`/`tokens_after_prune` for replanning calls).
- If true LLM tokenization is unavailable, you may log a proxy in `tokens_in`/`tokens_after_prune`, but **record the method** in `run.json["config"]`.

### Latency accounting (milliseconds)

| Field | Type | Unit | Default | Meaning / measurement point |
|---|---|---:|---|---|
| `lat_total_ms` | float\|null | ms | `null` | End-to-end replanning-call latency (inclusive) |
| `lat_prune_ms` | float\|null | ms | `null` | Pruning/compression time |
| `lat_retrieval_ms` | float\|null | ms | `null` | Retrieval/RAG time |
| `lat_prefill_ms` | float\|null | ms | `null` | LLM prefill time (if available) |
| `lat_decode_ms` | float\|null | ms | `null` | LLM decode time (if available) |

### Tail/SLO derived fields (can be computed in logging or analysis)

| Field | Type | Unit | Default | Meaning |
|---|---|---:|---|---|
| `slo_violation` | bool\|null | - | `null` | Whether `lat_total_ms > slo_ms` |
| `slo_over_ms` | float\|null | ms | `null` | `max(0, lat_total_ms - slo_ms)` |

### Stability + multi-agent (optional; may be null in single-agent domains)

| Field | Type | Unit | Default | Meaning |
|---|---|---:|---|---|
| `plan_hash` | str\|null | - | `null` | Hash of the current selected plan (stability proxy) |
| `plan_churn_score` | float\|null | - | `null` | Plan churn proxy (definition must be documented in config/paper) |
| `deadlock_flag` | bool\|null | - | `null` | Whether deadlock is detected at/after this event |
| `wait_time_ms` | float\|null | ms | `null` | Waiting time due to coordination/locks |
| `locks_held` | list\|dict\|null | - | `null` | Locks currently held (if applicable) |
| `locks_waiting` | list\|dict\|null | - | `null` | Locks currently waited on (if applicable) |

## `episode_metrics.jsonl` (one line per episode)

### Core identifiers + outcomes

| Field | Type | Unit | Default | Meaning |
|---|---|---:|---|---|
| `schema_version` | int | - | `1` | Schema version for this record |
| `time_utc` | str | ISO8601 | auto | Timestamp |
| `run_id` | str | - | auto | Run identifier |
| `domain` | str | - | from `run.json` | Domain name |
| `task` | str\|null | - | from `run.json` | Optional task name |
| `variant` | str | - | - | Experimental condition label |
| `episode_id` | str\|int | - | - | Episode identifier |
| `success` | float\|int\|bool | - | - | Episode success flag/score |
| `spl` | float\|null | - | - | SPL (Habitat) |
| `step_count` | int\|null | steps | - | Steps executed |
| `replan_cycles` | int\|float\|null | count | - | Number of replanning cycles in the episode |

### Optional derived per-episode summaries (may be computed by analysis)

| Field | Type | Unit | Default | Meaning |
|---|---|---:|---|---|
| `lat_p50_ms` | float\|null | ms | `null` | Per-episode replanning latency P50 (over replans in this episode) |
| `lat_p95_ms` | float\|null | ms | `null` | Per-episode replanning latency P95 |
| `lat_p99_ms` | float\|null | ms | `null` | Per-episode replanning latency P99 |
| `slo_violation_rate` | float\|null | - | `null` | Fraction of replans violating SLO |
| `tokens_in_mean` | float\|null | tokens | `null` | Mean `tokens_in` over replans in this episode |
| `tokens_in_p95` | float\|null | tokens | `null` | P95 `tokens_in` over replans |
| `tokens_in_p99` | float\|null | tokens | `null` | P99 `tokens_in` over replans |
| `tokens_after_prune_mean` | float\|null | tokens | `null` | Mean `tokens_after_prune` over replans |
| `tokens_after_prune_p95` | float\|null | tokens | `null` | P95 `tokens_after_prune` |
| `tokens_after_prune_p99` | float\|null | tokens | `null` | P99 `tokens_after_prune` |
| `deadlock_flag` | bool\|null | - | `null` | Episode-level deadlock indicator (if applicable) |
| `wait_time_ms` | float\|null | ms | `null` | Episode-level total/mean wait time (definition must be documented) |

## Quick checks

- Schema defaults/normalization: `experiments/common/logging.py`
- Aggregation (runs root or a single run): `python analysis/aggregate_runs.py --write_tables`
