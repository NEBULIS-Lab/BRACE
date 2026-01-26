from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from brace.controller import BraceController, BraceHyperparams, BraceState
from experiments.common.context_compress.baselines import (
    extra_overhead_ms,
    normalize_method,
    quality_multiplier,
)
from experiments.common.logging import RunContext
from experiments.common import replan_schedule as rs


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    if q <= 0:
        return float(sorted(vals)[0])
    if q >= 1:
        return float(sorted(vals)[-1])
    s = sorted(vals)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return float(d0 + d1)


def _pct(x: float) -> str:
    if x != x:  # NaN
        return "-"
    return f"{100.0 * x:.1f}%"


def _f(x: float) -> str:
    if x != x:
        return "-"
    return f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}"


def _expand_variants(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants = base_cfg.get("variants", [])
    if variants:
        return variants

    grid = base_cfg.get("variant_grid")
    if not grid:
        raise ValueError("Config must define either `variants` or `variant_grid`.")

    brace_opts = list(grid.get("brace_enabled", [False, True]))
    prune_opts = list(grid.get("pruning_enabled", [False, True]))
    keep_ratios = list(grid.get("keep_ratio", [1.0]))
    token_budgets = list(grid.get("token_budget", [base_cfg.get("token_budget", 0)]))

    out: List[Dict[str, Any]] = []
    for brace_enabled in brace_opts:
        for pruning_enabled in prune_opts:
            for token_budget in token_budgets:
                if pruning_enabled:
                    kr_list = keep_ratios
                else:
                    kr_list = [1.0]
                for keep_ratio in kr_list:
                    name = (
                        f"{'brace' if brace_enabled else 'nobrace'}_"
                        f"{'prune' if pruning_enabled else 'noprune'}"
                        f"__r{keep_ratio}"
                        f"__B{token_budget}"
                    )
                    out.append(
                        {
                            "name": name,
                            "brace_enabled": bool(brace_enabled),
                            "pruning_enabled": bool(pruning_enabled),
                            "keep_ratio": float(keep_ratio),
                            "token_budget": int(token_budget),
                        }
                    )
    return out


def _apply_context_compress(
    *,
    tokens_in: int,
    tokens_protected: int,
    method: str,
    keep_ratio: float,
    token_budget: Optional[int],
) -> int:
    tokens_in = max(0, int(tokens_in))
    tokens_protected = max(0, int(tokens_protected))
    if method == "none":
        return tokens_in
    if method == "erecap":
        keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
        prunable = max(0, tokens_in - tokens_protected)
        kept_prunable = max(1, int(round(prunable * keep_ratio))) if prunable > 0 else 0
        tokens_after = tokens_protected + kept_prunable
        if token_budget is not None and token_budget > 0:
            tokens_after = min(tokens_after, int(token_budget))
        return max(0, int(tokens_after))

    # Budget-matched baselines (random / recency / structured_summary): in this proxy,
    # we model them as budget-bound token selection/compression with method-dependent quality.
    if token_budget is not None and token_budget > 0:
        return max(0, int(min(tokens_in, int(token_budget))))
    return tokens_in


def _plan_hash(
    *,
    episode_id: str,
    planner_calls: int,
    mode: str,
    tokens_after: int,
    dist: float,
    churn_flag: bool,
) -> str:
    tokens_bucket = int(tokens_after // 50)
    dist_bucket = int(max(0.0, min(1.0, dist)) * 20)
    if churn_flag:
        # When the agent isn't making progress, treat consecutive replans as a churny regime:
        # the planner may emit different plans even when the world state hasn't changed.
        key = f"{episode_id}|{mode}|{tokens_bucket}|thrash|{planner_calls}"
    else:
        # Otherwise, treat plans as mostly state-determined, stable within a coarse state bucket.
        key = f"{episode_id}|{mode}|{tokens_bucket}|{dist_bucket}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return h[:12]


def run_variant(ctx: RunContext, variant: Dict[str, Any], base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    rng = random.Random(int(base_cfg.get("random_seed", 0)) + abs(hash(variant["name"])) % (2**31))

    slo_ms = int(base_cfg.get("slo_ms", 250))
    ms_per_token = float(base_cfg.get("ms_per_token", 0.8))
    overhead_ms = float(base_cfg.get("overhead_ms", 0.0))
    summary_overhead_ms = float(base_cfg.get("summary_overhead_ms", 0.0))

    episodes = int(base_cfg.get("episodes", 10))
    max_steps = int(base_cfg.get("max_steps", 200))
    replan_interval = int(
        variant.get(
            "replan_interval_steps",
            variant.get("replan_interval", base_cfg.get("replan_interval_steps", base_cfg.get("replan_interval", 10))),
        )
    )
    trigger_cooldown_steps = int(
        variant.get("trigger_cooldown_steps", base_cfg.get("trigger_cooldown_steps", 0))
    )

    dist_start = float(base_cfg.get("dist_start", 1.0))
    dist_goal_threshold = float(base_cfg.get("dist_goal_threshold", 0.05))
    base_progress = float(base_cfg.get("base_progress", 0.02))
    failure_prob = float(base_cfg.get("failure_prob", 0.03))
    progress_deadlock_epsilon = float(base_cfg.get("progress_deadlock_epsilon", 0.001))
    env_step_ms = float(base_cfg.get("env_step_ms", 50.0))
    max_frozen_steps = int(base_cfg.get("max_frozen_steps", 250))
    conflict_prob = float(base_cfg.get("conflict_prob", 0.0))
    conflict_min_steps = int(base_cfg.get("conflict_min_steps", 0))
    conflict_max_steps = int(base_cfg.get("conflict_max_steps", conflict_min_steps))
    conflict_resolve_on_full_replan = bool(base_cfg.get("conflict_resolve_on_full_replan", True))

    tokens_protected = int(base_cfg.get("tokens_protected", 200))
    tokens_base = int(base_cfg.get("tokens_base", 600))
    tokens_growth_per_step = int(base_cfg.get("tokens_growth_per_step", 8))

    brace_enabled = bool(variant.get("brace_enabled", False))
    pruning_enabled = bool(variant.get("pruning_enabled", False))
    keep_ratio = float(variant.get("keep_ratio", 1.0))
    token_budget = int(variant.get("token_budget", base_cfg.get("token_budget", 0)))
    token_budget = token_budget if token_budget > 0 else None
    context_compress_method = normalize_method(
        variant.get("context_compress_method", variant.get("baseline_method")),
        pruning_enabled=pruning_enabled,
    )
    pruning_enabled_event = context_compress_method in ("erecap", "random", "recency")
    summary_compress_enabled_event = context_compress_method == "structured_summary"
    method_q = quality_multiplier(context_compress_method)

    deadlock_trigger_edge = bool(base_cfg.get("deadlock_trigger_edge", False))
    reset_stuck_on_planner_call = bool(base_cfg.get("reset_stuck_on_planner_call", True))

    hparams_dict = dict(base_cfg.get("brace_hparams", {}) or {})
    hparams_dict.update(dict(variant.get("brace_hparams", {}) or {}))
    hparams = (
        BraceHyperparams(slo_ms=slo_ms, **hparams_dict) if hparams_dict else BraceHyperparams(slo_ms=slo_ms)
    )
    controller = BraceController(hparams)

    sim_deadlock_window_steps = int(base_cfg.get("sim_deadlock_window_steps", int(hparams.deadlock_window)))

    # Aggregation buffers (per variant)
    lat_ms_all: List[float] = []
    tok_in_all: List[float] = []
    tok_after_all: List[float] = []

    success_sum = 0.0
    steps_sum = 0.0
    replans_sum = 0.0
    planner_calls_sum = 0.0
    plan_changes_sum = 0.0
    deadlocks_sum = 0.0
    stall_steps_sum = 0.0
    suppressed_triggers_sum = 0.0
    slo_viol_sum = 0.0
    trigger_counts = {"periodic": 0.0, "failure": 0.0, "deadlock": 0.0}

    for ep_i in range(episodes):
        episode_id = f"proxy_ep{ep_i:04d}"

        dist = dist_start
        plan_quality = 1.0
        last_plan_hash = None
        last_replan_step = -10**9
        last_planner_latency_ms = None

        state = BraceState()
        replans = 0
        planner_calls = 0
        plan_changes = 0
        deadlocks = 0
        stall_steps = 0
        suppressed_triggers = 0
        slo_viol = 0

        dist_at_last_replan = dist
        stuck_steps = 0
        frozen_steps = 0
        conflict_steps = 0

        # Main simulation loop
        for t in range(max_steps):
            # Dynamics (progress is affected by plan quality; failures create sudden stalls)
            failure = False
            conflict_active_step = False
            if frozen_steps > 0:
                frozen_steps -= 1
                progress_step = 0.0
            else:
                conflict_active = conflict_steps > 0
                if conflict_active:
                    conflict_steps -= 1
                    progress_step = 0.0
                    conflict_active_step = True
                else:
                    if conflict_prob > 0.0 and conflict_min_steps > 0 and rng.random() < conflict_prob:
                        conflict_steps = rng.randint(
                            int(min(conflict_min_steps, conflict_max_steps)),
                            int(max(conflict_min_steps, conflict_max_steps)),
                        )
                        conflict_steps = max(0, int(conflict_steps) - 1)
                        progress_step = 0.0
                        conflict_active_step = True
                    else:
                        failure = rng.random() < failure_prob
                        noise = 0.7 + 0.6 * rng.random()
                        progress_step = base_progress * plan_quality * noise
                        if failure:
                            progress_step = 0.0

                # Deadlock proxy: only count "stuck" while the agent is actively executing.
                if progress_step < progress_deadlock_epsilon:
                    stuck_steps += 1
                else:
                    stuck_steps = 0

            dist = max(0.0, dist - progress_step)
            if progress_step < progress_deadlock_epsilon:
                stall_steps += 1

            done = dist <= dist_goal_threshold

            periodic_trigger = rs.periodic_trigger(
                t=t, interval_steps=int(replan_interval), last_replan_step=int(last_replan_step)
            )
            if sim_deadlock_window_steps > 0:
                if deadlock_trigger_edge:
                    deadlock_trigger = stuck_steps == sim_deadlock_window_steps
                else:
                    deadlock_trigger = stuck_steps >= sim_deadlock_window_steps
            else:
                deadlock_trigger = False
            failure_trigger = failure

            allow_trigger = rs.allow_trigger(
                t=t, last_replan_step=int(last_replan_step), trigger_cooldown_steps=int(trigger_cooldown_steps)
            )

            if (periodic_trigger or failure_trigger or deadlock_trigger) and not allow_trigger:
                suppressed_triggers += 1

            if (periodic_trigger or failure_trigger or deadlock_trigger) and allow_trigger:
                last_replan_step = t
                replans += 1

                conflict_steps_before_replan = int(conflict_steps)

                if periodic_trigger:
                    trigger_counts["periodic"] += 1.0
                if failure_trigger:
                    trigger_counts["failure"] += 1.0
                if deadlock_trigger:
                    trigger_counts["deadlock"] += 1.0

                trigger_dict = {
                    "unsafe": False,
                    "deadlock": bool(deadlock_trigger),
                    "periodic": bool(periodic_trigger),
                    "types": [x for x, ok in [("failure", failure_trigger), ("deadlock", deadlock_trigger)] if ok],
                }
                replan_trigger_type = rs.trigger_type_primary(
                    periodic=bool(periodic_trigger), failure=bool(failure_trigger), deadlock=bool(deadlock_trigger)
                )

                progress_since_last_replan = float(dist_at_last_replan - dist)
                churn_flag = bool(planner_calls > 0 and progress_since_last_replan < float(hparams.progress_epsilon))
                telemetry = {
                    "progress": progress_since_last_replan,
                    "lat_total_ms": last_planner_latency_ms,
                    "churn": churn_flag,
                    "clarification_budget_turns": int(base_cfg.get("clarification_budget_turns", 0)),
                }

                if brace_enabled:
                    decision, state = controller.step(
                        state=state,
                        trigger=trigger_dict,
                        telemetry=telemetry,
                        remaining_budget=token_budget,
                        num_agents=int(base_cfg.get("num_agents", 1)),
                    )
                    mode = decision.mode
                    budget = decision.token_budget
                    time_budget_ms = decision.time_budget_ms
                    brace_reason = str(decision.reason)
                    clarification_budget_turns = int(decision.clarification_budget_turns)
                    hazards = {
                        "hazard_slo": bool(decision.hazard_slo),
                        "hazard_churn": bool(decision.hazard_churn),
                        "hazard_deadlock": bool(decision.hazard_deadlock),
                        "hazard_unsafe": bool(decision.hazard_unsafe),
                        "cooldown_active": bool(decision.cooldown_active),
                        "rollback_flag": bool(decision.rollback_flag),
                        "min_commit_window": int(decision.min_commit_window),
                    }
                else:
                    mode = "partial_replan"
                    budget = token_budget
                    time_budget_ms = (
                        max(0, int(round(float(slo_ms) * float(hparams.slo_guard_ratio)))) if slo_ms > 0 else None
                    )
                    brace_reason = "nobrace"
                    clarification_budget_turns = int(base_cfg.get("clarification_budget_turns", 0))
                    hazards = {
                        "hazard_slo": False,
                        "hazard_churn": False,
                        "hazard_deadlock": bool(deadlock_trigger),
                        "hazard_unsafe": False,
                        "cooldown_active": False,
                        "rollback_flag": False,
                        "min_commit_window": 0,
                    }

                planner_called = mode in ("full_replan", "partial_replan")
                budget_log = int(budget) if budget is not None else 0

                if planner_called:
                    planner_calls += 1

                    # Proxy multi-agent/resource-conflict resolution: full replanning can clear conflicts.
                    if conflict_resolve_on_full_replan and mode == "full_replan":
                        conflict_steps = 0

                    tokens_in = tokens_base + tokens_growth_per_step * t
                    tokens_after = _apply_context_compress(
                        tokens_in=tokens_in,
                        tokens_protected=tokens_protected,
                        method=context_compress_method,
                        keep_ratio=keep_ratio,
                        token_budget=budget,
                    )
                    extra_ms = extra_overhead_ms(
                        context_compress_method,
                        summary_overhead_ms=summary_overhead_ms,
                    )
                    lat_total_ms = overhead_ms + ms_per_token * float(tokens_after) + extra_ms
                    last_planner_latency_ms = lat_total_ms

                    if env_step_ms > 0:
                        freeze = int(round(lat_total_ms / env_step_ms))
                        if freeze > 0:
                            frozen_steps = min(int(max_frozen_steps), int(frozen_steps) + freeze)

                    tok_in_all.append(float(tokens_in))
                    tok_after_all.append(float(tokens_after))
                    lat_ms_all.append(float(lat_total_ms))

                    if slo_ms > 0 and lat_total_ms > float(slo_ms):
                        slo_viol += 1

                    ph = _plan_hash(
                        episode_id=episode_id,
                        planner_calls=planner_calls,
                        mode=mode,
                        tokens_after=tokens_after,
                        dist=float(dist),
                        churn_flag=bool(churn_flag),
                    )
                    plan_changed = last_plan_hash is not None and ph != last_plan_hash
                    if plan_changed:
                        plan_changes += 1
                    last_plan_hash = ph
                    state.last_plan_hash = ph

                    # Update plan quality: full > partial; pruning reduces quality based on retained context.
                    base_q = 1.0 if mode == "full_replan" else 0.9
                    if tokens_in > 0:
                        retain = float(tokens_after) / float(tokens_in)
                        base_q *= 0.70 + 0.30 * retain
                    base_q *= float(method_q)
                    plan_quality = base_q

                    dist_at_last_replan = dist
                    if reset_stuck_on_planner_call:
                        stuck_steps = 0
                else:
                    tokens_in = 0
                    tokens_after = 0
                    lat_total_ms = 0.0
                    plan_changed = False

                    # If we defer/reuse, the environment gradually drifts away from the last plan.
                    if mode == "defer_replan":
                        plan_quality *= 0.98
                    else:
                        plan_quality *= 0.995

                if deadlock_trigger:
                    deadlocks += 1

                slo_violation = bool(slo_ms > 0 and lat_total_ms > float(slo_ms))
                slo_over_ms = max(0.0, float(lat_total_ms) - float(slo_ms)) if slo_ms > 0 else 0.0

                ctx.append_event(
                    {
                        "domain": "proxy",
                        "variant": variant["name"],
                        "episode_id": episode_id,
                        "t": t,
                        "brace_enabled": brace_enabled,
                        "pruning_enabled": bool(pruning_enabled_event),
                        "summary_compress_enabled": bool(summary_compress_enabled_event),
                        "context_compress_method": context_compress_method,
                        "keep_ratio": keep_ratio,
                        "clarification_budget_turns": int(clarification_budget_turns),
                        "mode": mode,
                        "brace_reason": str(brace_reason),
                        "planner_called": bool(planner_called),
                        "replan_interval_steps": int(replan_interval),
                        "trigger_cooldown_steps": int(trigger_cooldown_steps),
                        "replan_trigger_type": str(replan_trigger_type),
                        "slo_ms": slo_ms,
                        "time_budget_ms": time_budget_ms,
                        "token_budget": budget_log,
                        "tokens_in": tokens_in,
                        "tokens_after_prune": tokens_after,
                        "lat_total_ms": lat_total_ms,
                        "lat_prune_ms": float(extra_ms) if planner_called else 0.0,
                        "slo_violation": bool(slo_violation),
                        "slo_over_ms": float(slo_over_ms),
                        "progress_since_last_replan": progress_since_last_replan,
                        "plan_hash": state.last_plan_hash,
                        "plan_changed": bool(plan_changed),
                        "plan_churn_score": float(1.0 if plan_changed else 0.0),
                        "conflict_active": bool(conflict_active_step),
                        "conflict_steps_before_replan": int(conflict_steps_before_replan),
                        "conflict_steps_after_replan": int(conflict_steps),
                        "state": {
                            "cooldown_timer": int(state.cooldown_timer),
                            "commit_timer": int(state.commit_timer),
                            "consecutive_defers": int(state.consecutive_defers),
                            "no_progress_steps": int(state.no_progress_steps),
                            "churn_ema": float(state.churn_ema),
                        },
                        "trigger": trigger_dict,
                        **hazards,
                    }
                )

            if done:
                break

        success = float(dist <= dist_goal_threshold)
        success_sum += success
        steps_sum += float(t + 1)
        replans_sum += float(replans)
        planner_calls_sum += float(planner_calls)
        plan_changes_sum += float(plan_changes)
        deadlocks_sum += float(deadlocks)
        stall_steps_sum += float(stall_steps)
        suppressed_triggers_sum += float(suppressed_triggers)
        slo_viol_sum += float(slo_viol)

        ctx.append_episode(
            {
                "domain": "proxy",
                "variant": variant["name"],
                "episode_id": episode_id,
                "success": success,
                "spl": 0.0,
                "step_count": float(t + 1),
                "replan_cycles": float(replans),
                "planner_calls": float(planner_calls),
                "plan_changes": float(plan_changes),
                "deadlocks": float(deadlocks),
                "stall_steps": float(stall_steps),
                "suppressed_triggers": float(suppressed_triggers),
                "slo_violations": float(slo_viol),
                "replan_interval_steps": float(replan_interval),
                "trigger_cooldown_steps": float(trigger_cooldown_steps),
                "effective_replans_per_episode": float(replans),
                "episode_wall_time_ms": float((t + 1) * env_step_ms) if env_step_ms > 0 else None,
                "effective_replans_per_min": float(replans) / (float((t + 1) * env_step_ms) / 60000.0)
                if env_step_ms > 0
                else None,
            }
        )

    n = float(episodes) if episodes > 0 else float("nan")
    avg_success = success_sum / n
    avg_steps = steps_sum / n
    avg_replans = replans_sum / n
    avg_planner_calls = planner_calls_sum / n
    avg_plan_changes = plan_changes_sum / n
    churn_rate = avg_plan_changes / avg_planner_calls if avg_planner_calls > 0 else 0.0
    deadlocks_per_ep = deadlocks_sum / n
    stall_steps_per_ep = stall_steps_sum / n
    suppressed_triggers_per_ep = suppressed_triggers_sum / n
    slo_viol_rate = slo_viol_sum / planner_calls_sum if planner_calls_sum > 0 else 0.0

    tok_in_mean = float(mean(tok_in_all)) if tok_in_all else float("nan")
    tok_after_mean = float(mean(tok_after_all)) if tok_after_all else float("nan")
    tok_red = 1.0 - (tok_after_mean / tok_in_mean) if tok_in_mean == tok_in_mean and tok_in_mean > 0 else float("nan")

    lat_mean = float(mean(lat_ms_all)) if lat_ms_all else float("nan")
    lat_p95 = _percentile(lat_ms_all, 0.95)
    lat_p99 = _percentile(lat_ms_all, 0.99)

    trig_total = sum(trigger_counts.values())
    trig_periodic_frac = trigger_counts["periodic"] / trig_total if trig_total > 0 else float("nan")
    trig_failure_frac = trigger_counts["failure"] / trig_total if trig_total > 0 else float("nan")
    trig_deadlock_frac = trigger_counts["deadlock"] / trig_total if trig_total > 0 else float("nan")

    return {
        "variant": variant["name"],
        "episodes": int(episodes),
        "success": avg_success,
        "steps": avg_steps,
        "replans": avg_replans,
        "replan_interval_steps": int(replan_interval),
        "trigger_cooldown_steps": int(trigger_cooldown_steps),
        "planner_calls": avg_planner_calls,
        "plan_changes": avg_plan_changes,
        "churn_rate": churn_rate,
        "deadlocks_per_ep": deadlocks_per_ep,
        "stall_steps_per_ep": stall_steps_per_ep,
        "suppressed_triggers_per_ep": suppressed_triggers_per_ep,
        "trigger_periodic_frac": trig_periodic_frac,
        "trigger_failure_frac": trig_failure_frac,
        "trigger_deadlock_frac": trig_deadlock_frac,
        "slo_viol_rate": slo_viol_rate,
        "tok_in": tok_in_mean,
        "tok_after": tok_after_mean,
        "tok_red": tok_red,
        "lat_mean_ms": lat_mean,
        "lat_p95_ms": lat_p95,
        "lat_p99_ms": lat_p99,
        "brace_hparams": asdict(hparams),
    }


def _render_table(run_id: str, rows: List[Dict[str, Any]]) -> str:
    md: List[str] = []
    md.append(f"# BRACE controller proxy summary: `{run_id}`\n")
    md.append(
        "Token/latency are synthetic proxies; churn/deadlock/stall are mechanism proxies from the simulation runner.\n\n"
    )
    md.append(
        "| Variant | Episodes | Success | Steps | Replans | Interval | Trigger cooldown | Trigger mix (P/F/D) | Suppressed/ep | Planner calls | Plan changes | Churn (chg/call) | Deadlocks/ep | Stall steps/ep | SLO viol/call | Tokens in | Tokens after | Token reduction | Lat mean (ms) | Lat P95 | Lat P99 |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        md.append(
            "| {variant} | {n} | {succ} | {steps} | {replans} | {intv} | {tcd} | {mix} | {sup} | {calls} | {chg} | {churn} | {dl} | {stall} | {slo} | {ti} | {ta} | {tr} | {lm} | {p95} | {p99} |\n".format(
                variant=r["variant"],
                n=r["episodes"],
                succ=_pct(float(r["success"])),
                steps=_f(float(r["steps"])),
                replans=_f(float(r["replans"])),
                intv=str(r.get("replan_interval_steps", "-")),
                tcd=str(r.get("trigger_cooldown_steps", "-")),
                mix="{}/{}/{}".format(
                    _pct(float(r.get("trigger_periodic_frac", float("nan")))),
                    _pct(float(r.get("trigger_failure_frac", float("nan")))),
                    _pct(float(r.get("trigger_deadlock_frac", float("nan")))),
                ),
                sup=_f(float(r.get("suppressed_triggers_per_ep", float("nan")))),
                calls=_f(float(r["planner_calls"])),
                chg=_f(float(r["plan_changes"])),
                churn=_f(float(r["churn_rate"])),
                dl=_f(float(r.get("deadlocks_per_ep", float("nan")))),
                stall=_f(float(r.get("stall_steps_per_ep", float("nan")))),
                slo=_pct(float(r["slo_viol_rate"])),
                ti=_f(float(r["tok_in"])),
                ta=_f(float(r["tok_after"])),
                tr=_pct(float(r["tok_red"])),
                lm=_f(float(r["lat_mean_ms"])),
                p95=_f(float(r["lat_p95_ms"])),
                p99=_f(float(r["lat_p99_ms"])),
            )
        )
    return "".join(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke/proxy_controller.json")
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--run_name", default="brace_controller_proxy_smoke")
    args = ap.parse_args()

    base_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ctx = RunContext.create(args.runs_root, args.run_name, base_cfg)

    rows: List[Dict[str, Any]] = []
    for variant in _expand_variants(base_cfg):
        rows.append(run_variant(ctx, variant, base_cfg))

    rows = sorted(rows, key=lambda r: r["variant"])

    # Persist summary + a paper-ready small table.
    ctx.write_summary({"rows": rows, "note": "brace_controller_proxy_runner (synthetic simulation)"})

    tables_dir = _PROJ_ROOT / "artifacts" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_md = tables_dir / f"{ctx.run_id}.md"
    out_json = tables_dir / f"{ctx.run_id}.json"
    out_md.write_text(_render_table(ctx.run_id, rows), encoding="utf-8")
    out_json.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")

    print(ctx.run_dir)
    print(out_md)


if __name__ == "__main__":
    main()
