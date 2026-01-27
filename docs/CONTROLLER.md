# BRACE Controller (v0) — Interface + Decision Logic (Proxy-Ready)

This document specifies the **minimal, auditable BRACE controller** implemented in `brace/controller.py`. It is designed to be:

- **Domain-agnostic** (works in proxy; can be wired into Habitat/RoboFactory runners later).
- **Budget-aware** (token/time budgets are first-class inputs/outputs).
- **Stability-aware** (anti-churn cooldown + minimum commit window).
- **Loggable** (hazard flags and decisions are explicitly surfaced for `events.jsonl`).

## 1) Inputs / Outputs

### Inputs (per replanning trigger)

- `state: BraceState`
  - `cooldown_timer`: active cooldown remaining steps (integer).
  - `commit_timer`: minimum commitment remaining steps (integer).
  - `consecutive_defers`: consecutive `defer_replan` count.
  - `no_progress_steps`: consecutive “low progress” count.
  - `churn_ema`: EMA of observed churn events (0–1).
  - `last_plan_hash`: optional last plan id (string).
- `trigger: dict`
  - `unsafe: bool`: safety-critical trigger.
  - `deadlock: bool`: explicit deadlock trigger (optional; can be inferred from telemetry too).
  - `periodic: bool`: periodic schedule trigger (optional).
  - `types: list[str]`: free-form trigger types (optional).
- `telemetry: dict`
  - `progress: float | None`: progress since last plan (domain-defined; proxy uses distance reduction).
  - `lat_total_ms: float | None`: last replanning latency (ms) for SLO hazard detection.
  - `churn: bool`: churn indicator (e.g., “replanned but negligible progress”).
  - `clarification_budget_turns: int`: remaining clarification turns (plumbed; may be forced to 0).
- `remaining_budget: int | None`
  - Interpreted as a **token budget upper bound** for the (potential) next planner call.

### Outputs (per replanning trigger)

- `decision: BraceDecision`
  - `mode ∈ {full_replan, partial_replan, reuse_subplan, defer_replan}`
  - `token_budget: int | None` (0 for reuse/defer; ratio for partial)
  - `time_budget_ms: int | None` (derived from `slo_ms * slo_guard_ratio`; 0 for reuse/defer)
  - `clarification_budget_turns: int` (forced to 0 in reuse/defer and under SLO hazard)
  - `protected_blocks: tuple[str,...]` (currently `("A","B","C","D")`)
  - `reason: str` (first-match decision reason: `unsafe/deadlock/cooldown/churn/commit_window/slo/default`)
  - hazards/logging flags:
    - `hazard_slo, hazard_churn, hazard_deadlock, hazard_unsafe`
    - `cooldown_active, rollback_flag, min_commit_window`
- `state_next: BraceState`

## 2) Hazard definitions (v0)

- `hazard_unsafe := trigger.unsafe`
- `hazard_deadlock := trigger.deadlock OR (no_progress_steps >= deadlock_window)`
- `hazard_slo := (telemetry.lat_total_ms > slo_ms * slo_guard_ratio)`
- `hazard_churn := telemetry.churn OR (churn_ema > churn_threshold)`
- `cooldown_active := (cooldown_timer > 0)`

Notes:
- `no_progress_steps` is updated whenever `telemetry.progress` is provided:
  - if `progress < progress_epsilon`: increment; else reset to 0.
- `churn_ema` is updated from `telemetry.churn` as an EMA with `churn_ema_alpha`.

## 3) Mode selection policy (v0)

Priority order (first match wins):

1. If `hazard_unsafe OR hazard_deadlock` → `full_replan`
2. Else if `cooldown_active` → `defer_replan`
3. Else if `hazard_churn` → `defer_replan`
4. Else if `commit_timer > 0` → `reuse_subplan`
5. Else if `hazard_slo` → `partial_replan`
6. Else → `partial_replan`

Deferral guard:
- If `mode == defer_replan` for `>= max_consecutive_defers` (and `max_consecutive_defers > 0`) → force `partial_replan`

## 4) Budget assignment (v0)

Let `B := remaining_budget` (if provided).

- `time_budget_ms := round(slo_ms * slo_guard_ratio)` (0 for reuse/defer)
- `full_replan`: `token_budget = B`
- `partial_replan`: `token_budget = round(B * partial_budget_ratio)` (clamped to ≥1 if `B>0`)
- `reuse_subplan / defer_replan`: `token_budget = 0`

Important: BRACE **outputs budgets**; the runner/pruner must enforce them.

## 5) State update rules (v0)

- Timers:
  - `cooldown_timer := max(0, cooldown_timer-1)`
  - `commit_timer := max(0, commit_timer-1)`
  - if `hazard_churn` and `cooldown_steps>0`: set `cooldown_timer := cooldown_steps`
  - if `mode ∈ {full_replan, partial_replan}` and `min_commit_window>0`: set `commit_timer := min_commit_window`
- Counters:
  - if `mode == defer_replan`: `consecutive_defers += 1`; else if plan-changing: reset to 0
  - `no_progress_steps` updated from telemetry as above
- `churn_ema` updated from `telemetry.churn` via EMA

## 6) Proxy experiments (reproduce)

- Self-check (no deps):
  - `python brace/selfcheck_controller.py`

- 2×2 smoke (BRACE on/off × pruning on/off):
  - `python experiments/proxy/brace_controller_proxy_runner.py --config configs/smoke/proxy_controller.json --runs_root runs --run_name brace_controller_proxy_smoke`
  - or `scripts/run_proxy.sh --config configs/smoke/proxy_controller.json --run-name brace_controller_proxy_smoke`

- Controller ablation sweep (cooldown/commit/deadlock window):
  - `python experiments/proxy/brace_controller_proxy_runner.py --config configs/paper/proxy_stability_sweep_p0.json --runs_root runs --run_name brace_controller_proxy_stability_sweep`

- Frequency axis sweep (replan interval × cooldown):
  - `python experiments/proxy/brace_controller_proxy_runner.py --config configs/paper/proxy_frequency_sweep_steps.json --runs_root runs --run_name brace_controller_proxy_freq_sweep`

Outputs:
- runs are written under `runs/<run_id>/`
- tables are appended under `artifacts/tables/<run_id>.md` (+ `.json`)
