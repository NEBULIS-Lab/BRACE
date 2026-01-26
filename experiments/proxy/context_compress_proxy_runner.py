from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from experiments.common.context_compress import budget_target_history_keep, select_history_tokens
from experiments.common.logging import RunContext
from experiments.common import replan_schedule as rs


def _safe_mean(vals: List[float]) -> float:
    return float(mean(vals)) if vals else float("nan")


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

    strategies = list(grid.get("compress_strategy", ["recency", "random", "structured_summary"]))
    token_budgets = list(grid.get("token_budget", [base_cfg.get("token_budget", 0)]))

    summary_head_tokens = list(grid.get("summary_head_tokens", [32]))
    summary_tail_tokens = list(grid.get("summary_tail_tokens", [64]))
    summary_quality = list(grid.get("summary_quality", [0.8]))
    summary_efficiency = list(grid.get("summary_efficiency", [3.0]))

    out: List[Dict[str, Any]] = []
    for strategy in strategies:
        for token_budget in token_budgets:
            if str(strategy).lower() in ("structured_summary", "summary", "head_tail_summary"):
                for h in summary_head_tokens:
                    for t in summary_tail_tokens:
                        for q in summary_quality:
                            for e in summary_efficiency:
                                name = f"{strategy}__B{int(token_budget)}__H{int(h)}__T{int(t)}__Q{float(q):.2f}__E{float(e):.2f}"
                                out.append(
                                    {
                                        "name": name,
                                        "compress_strategy": str(strategy),
                                        "token_budget": int(token_budget),
                                        "summary_head_tokens": int(h),
                                        "summary_tail_tokens": int(t),
                                        "summary_quality": float(q),
                                        "summary_efficiency": float(e),
                                    }
                                )
            else:
                name = f"{strategy}__B{int(token_budget)}"
                out.append(
                    {
                        "name": name,
                        "compress_strategy": str(strategy),
                        "token_budget": int(token_budget),
                    }
                )
    return out


def _history_weights(history_len: int) -> List[float]:
    history_len = max(0, int(history_len))
    if history_len <= 1:
        return [1.0 for _ in range(history_len)]
    # Recency-biased weight profile: later tokens matter more.
    return [0.2 + 0.8 * (i / float(history_len - 1)) for i in range(history_len)]


def _info_retention(
    *,
    weights: List[float],
    selection_indices: List[int],
    summary_tokens: int,
    summary_quality: float,
    summary_efficiency: float,
    head_kept: int,
    tail_kept: int,
) -> float:
    if not weights:
        return 1.0
    total = float(sum(weights))
    if total <= 0:
        return 1.0

    kept_weight = float(sum(weights[i] for i in selection_indices if 0 <= i < len(weights)))
    if summary_tokens <= 0:
        return max(0.0, min(1.0, kept_weight / total))

    history_len = len(weights)
    head_kept = max(0, min(int(head_kept), history_len))
    tail_kept = max(0, min(int(tail_kept), history_len - head_kept))
    mid_start = head_kept
    mid_end = history_len - tail_kept
    mid_weights = weights[mid_start:mid_end] if mid_end > mid_start else []
    mid_total = float(sum(mid_weights))
    if mid_total > 0 and mid_weights:
        mid_len = len(mid_weights)
        capture_frac = min(1.0, max(0.0, float(summary_efficiency) * float(summary_tokens) / float(mid_len)))
        kept_weight += float(summary_quality) * mid_total * capture_frac
    return max(0.0, min(1.0, kept_weight / total))


def run_variant(ctx: RunContext, variant: Dict[str, Any], base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    rng = random.Random(int(base_cfg.get("random_seed", 0)) + abs(hash(variant["name"])) % (2**31))

    episodes = int(base_cfg.get("episodes", 20))
    max_steps = int(base_cfg.get("max_steps", 200))
    replan_interval = int(variant.get("replan_interval", base_cfg.get("replan_interval", 10)))
    trigger_cooldown_steps = int(
        variant.get("trigger_cooldown_steps", base_cfg.get("trigger_cooldown_steps", 0))
    )

    dist_start = float(base_cfg.get("dist_start", 1.0))
    dist_goal_threshold = float(base_cfg.get("dist_goal_threshold", 0.05))
    base_progress = float(base_cfg.get("base_progress", 0.02))
    failure_prob = float(base_cfg.get("failure_prob", 0.03))

    tokens_protected = int(base_cfg.get("tokens_protected", 200))
    tokens_base = int(base_cfg.get("tokens_base", 800))
    tokens_growth_per_step = int(base_cfg.get("tokens_growth_per_step", 8))

    slo_ms = int(base_cfg.get("slo_ms", 250))
    ms_per_token = float(base_cfg.get("ms_per_token", 0.8))
    overhead_ms = float(base_cfg.get("overhead_ms", 0.0))
    ms_per_compress_token = float(base_cfg.get("ms_per_compress_token", 0.0))
    ms_per_summary_token = float(base_cfg.get("ms_per_summary_token", 0.0))

    strategy = str(variant.get("compress_strategy", "recency"))
    token_budget = int(variant.get("token_budget", base_cfg.get("token_budget", 0)))
    token_budget = token_budget if token_budget > 0 else None

    summary_head_tokens = int(variant.get("summary_head_tokens", base_cfg.get("summary_head_tokens", 32)))
    summary_tail_tokens = int(variant.get("summary_tail_tokens", base_cfg.get("summary_tail_tokens", 64)))
    summary_quality = float(variant.get("summary_quality", base_cfg.get("summary_quality", 0.8)))
    summary_efficiency = float(variant.get("summary_efficiency", base_cfg.get("summary_efficiency", 3.0)))

    # Aggregation buffers (per variant)
    lat_all: List[float] = []
    tok_in_all: List[float] = []
    tok_after_all: List[float] = []
    retention_all: List[float] = []

    success_sum = 0.0
    steps_sum = 0.0
    replans_sum = 0.0

    for ep_i in range(episodes):
        episode_id = f"proxy_ep{ep_i:04d}"
        dist = dist_start
        plan_quality = 1.0
        replans = 0
        last_replan_step = -10**9

        for t in range(max_steps):
            failure = rng.random() < failure_prob
            noise = 0.7 + 0.6 * rng.random()
            progress_step = 0.0 if failure else base_progress * plan_quality * noise
            dist = max(0.0, dist - progress_step)
            done = dist <= dist_goal_threshold

            periodic_trigger = rs.periodic_trigger(
                t=t, interval_steps=int(replan_interval), last_replan_step=int(last_replan_step)
            )
            allow_trigger = rs.allow_trigger(
                t=t, last_replan_step=int(last_replan_step), trigger_cooldown_steps=int(trigger_cooldown_steps)
            )

            if periodic_trigger and allow_trigger:
                last_replan_step = t
                replans += 1

                tokens_in = max(0, tokens_base + tokens_growth_per_step * t)
                history_len = max(0, tokens_in - tokens_protected)
                target_keep = budget_target_history_keep(
                    tokens_protected=tokens_protected, tokens_in=tokens_in, token_budget=token_budget
                )
                sel_t0 = time.time()
                selection = select_history_tokens(
                    strategy=strategy,
                    history_len=history_len,
                    target_keep=target_keep,
                    rng=rng,
                    summary_head_tokens=summary_head_tokens,
                    summary_tail_tokens=summary_tail_tokens,
                )
                lat_prune_ms = (time.time() - sel_t0) * 1000.0

                tokens_history_after = len(selection.kept_indices) + int(selection.summary_tokens)
                tokens_after = int(tokens_protected + tokens_history_after)

                # Synthetic info retention proxy.
                weights = _history_weights(history_len)
                info_ret = _info_retention(
                    weights=weights,
                    selection_indices=selection.kept_indices,
                    summary_tokens=int(selection.summary_tokens),
                    summary_quality=summary_quality,
                    summary_efficiency=summary_efficiency,
                    head_kept=int(selection.meta.get("head_kept", 0)),
                    tail_kept=int(selection.meta.get("tail_kept", 0)),
                )
                # Convert retention into a "plan quality" knob (bounded).
                plan_quality = max(0.3, min(1.0, 0.65 + 0.35 * float(info_ret)))

                # Latency accounting: planner inference scales with tokens_after; compression/summarization overheads are
                # logged separately but included in lat_total_ms for SLO accounting.
                lat_infer_ms = overhead_ms + ms_per_token * float(tokens_after)
                lat_compress_ms = ms_per_compress_token * float(tokens_in)
                lat_summary_ms = ms_per_summary_token * float(selection.summary_tokens)
                lat_total_ms = float(lat_infer_ms + lat_compress_ms + lat_summary_ms + lat_prune_ms)

                tok_in_all.append(float(tokens_in))
                tok_after_all.append(float(tokens_after))
                lat_all.append(float(lat_total_ms))
                retention_all.append(float(info_ret))

                ctx.append_event(
                    {
                        "domain": "proxy",
                        "variant": variant["name"],
                        "episode_id": episode_id,
                        "t": t,
                        "brace_enabled": False,
                        "pruning_enabled": True,
                        "rag_enabled": False,
                        "summary_compress_enabled": str(strategy).lower()
                        in ("structured_summary", "summary", "head_tail_summary"),
                        "mode": "partial_replan",
                        "token_budget": token_budget,
                        "replan_interval_steps": int(replan_interval),
                        "trigger_cooldown_steps": int(trigger_cooldown_steps),
                        "replan_trigger_type": "periodic",
                        "compress_strategy": str(strategy),
                        "summary_head_tokens": summary_head_tokens,
                        "summary_tail_tokens": summary_tail_tokens,
                        "summary_tokens": int(selection.summary_tokens),
                        "summary_quality": float(summary_quality),
                        "summary_efficiency": float(summary_efficiency),
                        "info_retention": float(info_ret),
                        "tokens_in": int(tokens_in),
                        "tokens_after_prune": int(tokens_after),
                        "tokens_task": int(base_cfg.get("tokens_task", 80)),
                        "tokens_state": int(base_cfg.get("tokens_state", 60)),
                        "tokens_safety": int(base_cfg.get("tokens_safety", 40)),
                        "tokens_coord": int(base_cfg.get("tokens_coord", 20)),
                        "tokens_history": int(history_len),
                        "lat_total_ms": float(lat_total_ms),
                        "lat_prune_ms": float(lat_prune_ms),
                        "lat_prefill_ms": float(lat_infer_ms),
                        "lat_decode_ms": 0.0,
                        "clarification_budget_turns": 0,
                        "slo_ms": int(slo_ms),
                    }
                )

            if done:
                break

        success = float(dist <= dist_goal_threshold)
        success_sum += success
        steps_sum += float(t + 1)
        replans_sum += float(replans)

        ctx.append_episode(
            {
                "domain": "proxy",
                "variant": variant["name"],
                "episode_id": episode_id,
                "success": success,
                "spl": success,  # proxy: SPL ~= success (no path-length notion)
                "step_count": float(t + 1),
                "replan_cycles": float(replans),
                "effective_replans_per_episode": float(replans),
                "compress_strategy": str(strategy),
                "token_budget": token_budget,
            }
        )

    n = float(episodes) if episodes > 0 else float("nan")
    tok_in_mean = _safe_mean(tok_in_all)
    tok_after_mean = _safe_mean(tok_after_all)
    tok_red = 1.0 - (tok_after_mean / tok_in_mean) if tok_in_mean == tok_in_mean and tok_in_mean > 0 else float("nan")

    return {
        "variant": variant["name"],
        "compress_strategy": str(strategy),
        "token_budget": token_budget,
        "episodes": int(episodes),
        "success": success_sum / n,
        "steps": steps_sum / n,
        "replans": replans_sum / n,
        "tok_in": tok_in_mean,
        "tok_after": tok_after_mean,
        "tok_red": tok_red,
        "lat_mean_ms": _safe_mean(lat_all),
        "lat_p95_ms": _percentile(lat_all, 0.95),
        "lat_p99_ms": _percentile(lat_all, 0.99),
        "info_retention_mean": _safe_mean(retention_all),
    }


def _render_tables(run_id: str, rows: List[Dict[str, Any]], report_budget: int, target_success: float) -> str:
    md: List[str] = []
    md.append(f"# Budget-matched baselines (proxy): `{run_id}`\n\n")
    md.append(
        "This is a synthetic proxy runner for *token-budget matched* context compression baselines.\n"
        "Use it to validate fairness plumbing + logging/aggregation; numbers are not claims about real domains.\n\n"
    )

    # Table A: Quality @ matched tokens (fixed budget).
    md.append(f"## A) Quality @ matched tokens (`token_budget={report_budget}`)\n\n")
    md.append(
        "| Strategy | Episodes | Success | Steps | Replans | Tokens after (mean) | Lat P95 (ms) | Lat P99 (ms) | Info retention |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in sorted([x for x in rows if int(x.get("token_budget") or 0) == int(report_budget)], key=lambda x: x["compress_strategy"]):
        md.append(
            "| {s} | {n} | {succ} | {steps} | {replans} | {ta} | {p95} | {p99} | {ir} |\n".format(
                s=r.get("compress_strategy", r["variant"]),
                n=r["episodes"],
                succ=_pct(float(r["success"])),
                steps=_f(float(r["steps"])),
                replans=_f(float(r["replans"])),
                ta=_f(float(r["tok_after"])),
                p95=_f(float(r["lat_p95_ms"])),
                p99=_f(float(r["lat_p99_ms"])),
                ir=_f(float(r.get("info_retention_mean", float("nan")))),
            )
        )

    # Table B: Systems @ matched quality (fixed success threshold).
    md.append("\n")
    md.append(f"## B) Systems @ matched quality (`target_success≥{target_success:.2f}`)\n\n")
    md.append("| Strategy | Chosen budget | Episodes | Success | Tokens after (mean) | Lat P95 (ms) | Lat P99 (ms) |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")

    by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_strategy.setdefault(str(r.get("compress_strategy", "unknown")), []).append(r)

    for strategy, sr in sorted(by_strategy.items(), key=lambda kv: kv[0]):
        sr_sorted = sorted(sr, key=lambda x: float(x.get("tok_after", float("inf"))))
        chosen = None
        for cand in sr_sorted:
            if float(cand.get("success", 0.0)) >= float(target_success):
                chosen = cand
                break
        if chosen is None and sr_sorted:
            chosen = max(sr_sorted, key=lambda x: float(x.get("success", 0.0)))

        if not chosen:
            continue
        md.append(
            "| {s} | {b} | {n} | {succ} | {ta} | {p95} | {p99} |\n".format(
                s=strategy,
                b=int(chosen.get("token_budget") or 0),
                n=int(chosen.get("episodes") or 0),
                succ=_pct(float(chosen.get("success", float("nan")))),
                ta=_f(float(chosen.get("tok_after", float("nan")))),
                p95=_f(float(chosen.get("lat_p95_ms", float("nan")))),
                p99=_f(float(chosen.get("lat_p99_ms", float("nan")))),
            )
        )

    return "".join(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke/proxy_context_compress.json")
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--run_name", default="context_compress_proxy_ws4_smoke")
    args = ap.parse_args()

    base_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ctx = RunContext.create(args.runs_root, args.run_name, base_cfg)

    rows: List[Dict[str, Any]] = []
    for variant in _expand_variants(base_cfg):
        rows.append(run_variant(ctx, variant, base_cfg))

    rows = sorted(rows, key=lambda r: (str(r.get("compress_strategy")), int(r.get("token_budget") or 0)))
    ctx.write_summary({"rows": rows, "note": "context_compress_proxy_runner (synthetic)"})

    report_budget = int(base_cfg.get("report_budget", 400))
    target_success = float(base_cfg.get("report_target_success", 0.6))

    tables_dir = _PROJ_ROOT / "artifacts" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_md = tables_dir / f"{ctx.run_id}__budget_match.md"
    out_json = tables_dir / f"{ctx.run_id}__budget_match.json"
    out_md.write_text(_render_tables(ctx.run_id, rows, report_budget, target_success), encoding="utf-8")
    out_json.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")

    print(ctx.run_dir)
    print(out_md)


if __name__ == "__main__":
    main()
