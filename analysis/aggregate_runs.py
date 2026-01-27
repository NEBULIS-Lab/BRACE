from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]


EVENT_REQUIRED_KEYS_V1 = (
    "schema_version",
    "run_id",
    "event_type",
    "domain",
    "task",
    "variant",
    "episode_id",
    "t",
    "brace_enabled",
    "pruning_enabled",
    "rag_enabled",
    "summary_compress_enabled",
    "slo_ms",
    "token_budget",
    "clarification_budget_turns",
    "mode",
    "tokens_in",
    "tokens_after_prune",
    "lat_total_ms",
    "slo_violation",
    "slo_over_ms",
)

EPISODE_REQUIRED_KEYS_V1 = (
    "schema_version",
    "run_id",
    "domain",
    "task",
    "variant",
    "episode_id",
    "success",
    "spl",
    "step_count",
    "replan_cycles",
)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_mean(vals: List[float]) -> float:
    clean = [v for v in vals if v == v]  # drop NaNs
    return float(mean(clean)) if clean else float("nan")


def _percentile(vals: List[float], q: float) -> float:
    vals = [v for v in vals if v == v]
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

def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def _union_keys(records: List[Dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for r in records:
        keys.update(r.keys())
    return keys

def _time_key(ev: Dict[str, Any]) -> Optional[str]:
    for k in ("t", "step", "replan_cycle"):
        v = ev.get(k, None)
        if v is None:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            fv = float(v)
            if fv.is_integer():
                return str(int(fv))
            return str(fv)
        return str(v)
    return None

def _time_value(ev: Dict[str, Any]) -> Optional[float]:
    tk = _time_key(ev)
    if tk is None:
        return None
    try:
        return float(tk)
    except Exception:
        return None


def _is_replan_event(ev: Dict[str, Any]) -> bool:
    et = ev.get("event_type")
    if et == "phase":
        return False
    if et == "replan":
        return True
    # Back-compat: treat anything with replanning stats as a replanning call.
    return any(k in ev for k in ("tokens_in", "tokens_after_prune", "lat_total_ms", "replan_cycle", "t", "step"))

def _is_phase_event(ev: Dict[str, Any]) -> bool:
    et = ev.get("event_type")
    if et == "phase":
        return True
    if et == "replan":
        return False
    # Back-compat: treat records with a `phase` marker and no timestep as phase events.
    return "phase" in ev and "t" not in ev and "step" not in ev and "replan_cycle" not in ev


def _discover_run_dirs(runs_root: Path, pattern: str = "*", limit: int = 0) -> List[Path]:
    run_dirs = sorted([p for p in runs_root.glob(pattern) if (p / "run.json").exists()])
    if limit and limit > 0:
        run_dirs = run_dirs[-limit:]
    return run_dirs


def _pick_domain(run_json: Dict[str, Any], episodes: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Optional[str]:
    dom = (run_json.get("config") or {}).get("domain")
    if dom:
        return str(dom)
    for r in episodes:
        if r.get("domain"):
            return str(r["domain"])
    for r in events:
        if r.get("domain"):
            return str(r["domain"])
    return None


def aggregate_run(run_dir: Path) -> Tuple[str, Dict[str, Any]]:
    run_json = _read_json(run_dir / "run.json")
    run_id = str(run_json.get("run_id", run_dir.name))
    run_cfg = run_json.get("config") or {}
    slo_ms_default = run_cfg.get("slo_ms", None)

    episodes = list(_read_jsonl(run_dir / "episode_metrics.jsonl"))
    raw_events = list(_read_jsonl(run_dir / "events.jsonl"))
    events = [e for e in raw_events if _is_replan_event(e)]
    phase_events = [e for e in raw_events if _is_phase_event(e)]

    # Schema checks (warn only; keep analysis best-effort).
    missing_event_keys = [k for k in EVENT_REQUIRED_KEYS_V1 if k not in _union_keys(raw_events)]
    if missing_event_keys:
        _warn(f"{run_id}: events.jsonl missing keys: {missing_event_keys}")
    missing_episode_keys = [k for k in EPISODE_REQUIRED_KEYS_V1 if k not in _union_keys(episodes)]
    if missing_episode_keys:
        _warn(f"{run_id}: episode_metrics.jsonl missing keys: {missing_episode_keys}")

    domain = _pick_domain(run_json, episodes, events)

    variants = set()
    for r in episodes:
        variants.add(r.get("variant", None))
    for r in events:
        variants.add(r.get("variant", None))
    variants.discard(None)
    if not variants:
        variants = {"unknown"}

    by_variant_ep: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_variant_ep[str(ep.get("variant", "unknown"))].append(ep)

    # Per-variant replanning stats.
    ctx_stats = defaultdict(lambda: {"before": [], "after": []})
    tok_in_stats = defaultdict(list)
    tok_after_stats = defaultdict(list)
    lat_stats = defaultdict(list)
    slo_violation_stats = defaultdict(list)
    slo_over_stats = defaultdict(list)
    # Per-episode (within a variant) SLO bookkeeping: mark if any replanning call violates SLO.
    ep_any_slo_violation: Dict[str, Dict[str, bool]] = defaultdict(dict)

    missing_required = defaultdict(lambda: defaultdict(int))
    required_event_keys = (
        "run_id",
        "domain",
        "variant",
        "episode_id",
        "t",
        "mode",
        "brace_enabled",
        "pruning_enabled",
        "slo_ms",
        "token_budget",
        "clarification_budget_turns",
        "tokens_in",
        "tokens_after_prune",
        "lat_total_ms",
        "slo_violation",
        "slo_over_ms",
    )

    for ev in events:
        v = str(ev.get("variant", "unknown"))
        ep_id = ev.get("episode_id")
        ep_id_s = str(ep_id) if ep_id is not None else None
        for k in required_event_keys:
            if k == "t":
                if ev.get("t", None) is None and ev.get("step", None) is None:
                    missing_required[v][k] += 1
                continue
            if ev.get(k, None) is None:
                missing_required[v][k] += 1

        b = ev.get("context_length_before_chars")
        a = ev.get("context_length_after_chars")
        if isinstance(b, (int, float)) and isinstance(a, (int, float)) and float(b) > 0:
            ctx_stats[v]["before"].append(float(b))
            ctx_stats[v]["after"].append(float(a))

        ti = ev.get("tokens_in")
        if isinstance(ti, (int, float)) and float(ti) > 0:
            tok_in_stats[v].append(float(ti))
        ta = ev.get("tokens_after_prune")
        if isinstance(ta, (int, float)) and float(ta) > 0:
            tok_after_stats[v].append(float(ta))

        lt = ev.get("lat_total_ms")
        if isinstance(lt, (int, float)) and float(lt) >= 0:
            lat_stats[v].append(float(lt))

        slo_ms = ev.get("slo_ms", slo_ms_default)
        if isinstance(lt, (int, float)) and isinstance(slo_ms, (int, float)) and float(slo_ms) > 0:
            violation = float(lt) > float(slo_ms)
            slo_violation_stats[v].append(1.0 if violation else 0.0)
            slo_over_stats[v].append(max(0.0, float(lt) - float(slo_ms)))
            if ep_id_s is not None and violation:
                ep_any_slo_violation[v][ep_id_s] = True

    # Optional: per-phase overhead breakdown (VLM/VLA tracks, budget-matched baselines, etc.).
    # Convention: for `event_type="phase"` records, `lat_total_ms` is the phase duration.
    phase_lat_stats: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    vla_policy_calls_by_variant_ep: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
    vla_policy_events_by_variant: Dict[str, int] = defaultdict(int)
    vla_policy_events_skipped: int = 0
    for ev in phase_events:
        v = str(ev.get("variant", "unknown"))
        ph = ev.get("phase", None)
        ph_s = str(ph) if ph is not None else "unknown"

        lt = ev.get("lat_total_ms", None)
        if lt is None:
            lt = ev.get("lat_vlm_ms", None)
        if isinstance(lt, (int, float)) and float(lt) >= 0:
            phase_lat_stats[(v, ph_s)].append(float(lt))

        # VLA-aware latency accounting: treat `phase=vla_policy_call` as the control-loop latency anchor for VLA runs.
        if ph_s == "vla_policy_call":
            ep_id = ev.get("episode_id", None)
            tv = _time_value(ev)
            if ep_id is None or tv is None:
                vla_policy_events_skipped += 1
            elif isinstance(lt, (int, float)) and float(lt) >= 0:
                vla_policy_calls_by_variant_ep[(v, str(ep_id))].append((tv, float(lt)))
                vla_policy_events_by_variant[v] += 1

    phase_rows: List[Dict[str, Any]] = []
    for (variant, phase), vals in sorted(phase_lat_stats.items()):
        phase_rows.append(
            {
                "variant": variant,
                "phase": phase,
                "n": int(len(vals)),
                "lat_mean_ms": _safe_mean(vals),
                "lat_p95_ms": _percentile(vals, 0.95),
                "lat_p99_ms": _percentile(vals, 0.99),
            }
        )

    # If VLA policy calls were logged as separate `phase` events, report VLA-aware control-loop latency:
    # for each VLA call, add all replanning latency since the previous VLA call (per-episode timeline).
    lat_stats_effective = defaultdict(list)
    slo_violation_stats_effective = defaultdict(list)
    slo_over_stats_effective = defaultdict(list)
    ep_any_slo_violation_effective: Dict[str, Dict[str, bool]] = defaultdict(dict)
    vla_assign_stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"vla_calls": 0.0, "replans_total": 0.0, "replans_assigned": 0.0, "replans_unassigned": 0.0}
    )

    if vla_policy_calls_by_variant_ep:
        replans_by_variant_ep: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
        for ev in events:
            v = str(ev.get("variant", "unknown"))
            if vla_policy_events_by_variant.get(v, 0) <= 0:
                continue
            ep_id = ev.get("episode_id", None)
            tv = _time_value(ev)
            lt = ev.get("lat_total_ms", None)
            if ep_id is None or tv is None or not (isinstance(lt, (int, float)) and float(lt) >= 0):
                continue
            replans_by_variant_ep[(v, str(ep_id))].append((tv, float(lt)))
            vla_assign_stats[v]["replans_total"] += 1.0

        for (v, ep_id), vla_calls in vla_policy_calls_by_variant_ep.items():
            vla_calls.sort(key=lambda x: x[0])
            replans = replans_by_variant_ep.get((v, ep_id), [])
            replans.sort(key=lambda x: x[0])

            i = 0
            pending_replan_ms = 0.0
            for tv, vla_ms in vla_calls:
                assigned_this = 0
                while i < len(replans) and replans[i][0] <= tv:
                    pending_replan_ms += replans[i][1]
                    assigned_this += 1
                    i += 1

                lt_eff = float(vla_ms) + float(pending_replan_ms)
                pending_replan_ms = 0.0

                lat_stats_effective[v].append(lt_eff)
                vla_assign_stats[v]["vla_calls"] += 1.0
                vla_assign_stats[v]["replans_assigned"] += float(assigned_this)

                slo_ms = slo_ms_default
                if isinstance(slo_ms, (int, float)) and float(slo_ms) > 0:
                    violation = lt_eff > float(slo_ms)
                    slo_violation_stats_effective[v].append(1.0 if violation else 0.0)
                    slo_over_stats_effective[v].append(max(0.0, lt_eff - float(slo_ms)))
                    if violation:
                        ep_any_slo_violation_effective[v][ep_id] = True

            # Any replans after the last VLA call cannot be assigned in this heuristic.
            if i < len(replans):
                vla_assign_stats[v]["replans_unassigned"] += float(len(replans) - i)

    rows: List[Dict[str, Any]] = []
    for variant in sorted(variants):
        eps = by_variant_ep.get(variant, [])
        n = len(eps)
        success_rate = sum(1.0 for e in eps if float(e.get("success", 0.0)) > 0.0) / n if n else float("nan")
        use_effective_slo = bool(lat_stats_effective.get(variant))
        ep_any_slo = ep_any_slo_violation_effective if use_effective_slo else ep_any_slo_violation

        # "SLO+Success": success AND no SLO violations within the episode.
        slo_success_n = 0
        for e in eps:
            ep_id = e.get("episode_id")
            ep_id_s = str(ep_id) if ep_id is not None else None
            has_violation = bool(ep_id_s is not None and ep_any_slo.get(variant, {}).get(ep_id_s, False))
            is_success = float(e.get("success", 0.0)) > 0.0
            if is_success and (not has_violation):
                slo_success_n += 1
        slo_success_rate = (float(slo_success_n) / float(n)) if n else float("nan")

        executor = str(eps[0].get("executor", "-")) if eps else "-"
        instruction_style = str(eps[0].get("instruction_style", "-")) if eps else "-"
        ambiguity_type = str(eps[0].get("ambiguity_type", "-")) if eps else "-"
        clarif_turn_vals = [
            float(e["clarification_budget_turns"])
            for e in eps
            if e.get("clarification_budget_turns", None) is not None
            and isinstance(e.get("clarification_budget_turns"), (int, float))
        ]
        clarif_tok_vals = [
            float(e["clarification_tokens"])
            for e in eps
            if e.get("clarification_tokens", None) is not None and isinstance(e.get("clarification_tokens"), (int, float))
        ]
        clarif_lat_vals = [
            float(e["clarification_lat_ms"])
            for e in eps
            if e.get("clarification_lat_ms", None) is not None and isinstance(e.get("clarification_lat_ms"), (int, float))
        ]
        deadlock_vals = [
            float(e["deadlock_flag"])
            for e in eps
            if e.get("deadlock_flag", None) is not None and isinstance(e.get("deadlock_flag"), (int, float, bool))
        ]
        wait_vals = [
            float(e["wait_time_ms"])
            for e in eps
            if e.get("wait_time_ms", None) is not None and isinstance(e.get("wait_time_ms"), (int, float, bool))
        ]

        ctx_before = _safe_mean(ctx_stats[variant]["before"])
        ctx_after = _safe_mean(ctx_stats[variant]["after"])
        ctx_reduction = float("nan")
        if ctx_before == ctx_before and ctx_after == ctx_after and ctx_before > 0:
            ctx_reduction = 1.0 - (ctx_after / ctx_before)

        tok_in_mean = _safe_mean(tok_in_stats[variant])
        tok_after_mean = _safe_mean(tok_after_stats[variant])
        tok_keep_mean = float("nan")
        tok_reduction_mean = float("nan")
        if tok_in_mean == tok_in_mean and tok_after_mean == tok_after_mean and tok_in_mean > 0:
            tok_keep_mean = tok_after_mean / tok_in_mean
            tok_reduction_mean = 1.0 - tok_keep_mean

        lat_list = lat_stats_effective[variant] if lat_stats_effective.get(variant) else lat_stats[variant]
        slo_violation_list = (
            slo_violation_stats_effective[variant] if slo_violation_stats_effective.get(variant) else slo_violation_stats[variant]
        )
        slo_over_list = slo_over_stats_effective[variant] if slo_over_stats_effective.get(variant) else slo_over_stats[variant]

        lat_mean = _safe_mean(lat_list)
        rows.append(
            {
                "run_id": run_id,
                "domain": domain,
                "variant": variant,
                "episodes": n,
                "success": success_rate,
                "slo_success": slo_success_rate,
                "executor": executor,
                "instruction_style": instruction_style,
                "ambiguity_type": ambiguity_type,
                "clarification_turns": _safe_mean(clarif_turn_vals),
                "clarification_tokens": _safe_mean(clarif_tok_vals),
                "clarification_lat_ms": _safe_mean(clarif_lat_vals),
                "spl": _safe_mean([float(e.get("spl")) if e.get("spl") is not None else float("nan") for e in eps])
                if n
                else float("nan"),
                "steps": _safe_mean([float(e.get("step_count")) if e.get("step_count") is not None else float("nan") for e in eps])
                if n
                else float("nan"),
                "replans": _safe_mean([float(e.get("replan_cycles")) if e.get("replan_cycles") is not None else float("nan") for e in eps])
                if n
                else float("nan"),
                "deadlock_rate": _safe_mean(deadlock_vals),
                "wait_time_ms_mean": _safe_mean(wait_vals),
                "wait_time_ms_p95": _percentile(wait_vals, 0.95),
                "tok_in_mean": tok_in_mean,
                "tok_after_mean": tok_after_mean,
                "tok_keep_mean": tok_keep_mean,
                "tok_reduction_mean": tok_reduction_mean,
                "tok_after_p95": _percentile(tok_after_stats[variant], 0.95),
                "tok_after_p99": _percentile(tok_after_stats[variant], 0.99),
                "lat_mean_ms": lat_mean,
                "lat_p50_ms": _percentile(lat_list, 0.50),
                "lat_p95_ms": _percentile(lat_list, 0.95),
                "lat_p99_ms": _percentile(lat_list, 0.99),
                "slo_ms": slo_ms_default,
                "slo_violation_rate": _safe_mean(slo_violation_list),
                "slo_over_mean_ms": _safe_mean(slo_over_list),
                "ctx_before_chars": ctx_before,
                "ctx_after_chars": ctx_after,
                "ctx_reduction": ctx_reduction,
                "missing_event_fields": dict(missing_required.get(variant, {})),
                "lat_accounting": "control_loop_vla_policy_call_plus_replan" if lat_stats_effective.get(variant) else "replan_only",
                "vla_policy_call_events": int(vla_policy_events_by_variant.get(variant, 0)),
                "vla_policy_call_replans_total": int(vla_assign_stats.get(variant, {}).get("replans_total", 0.0)),
                "vla_policy_call_replans_assigned": int(vla_assign_stats.get(variant, {}).get("replans_assigned", 0.0)),
                "vla_policy_call_replans_unassigned": int(vla_assign_stats.get(variant, {}).get("replans_unassigned", 0.0)),
            }
        )

    for variant, miss in sorted(missing_required.items()):
        if any(v > 0 for v in miss.values()):
            _warn(f"{run_id} / {variant}: missing event fields (null) counts: {dict(miss)}")

    md: List[str] = []
    md.append(f"# Run summary: `{run_id}`\n")
    md.append("| Variant | Episodes | Executor | Style | Ambiguity | Clarif turns | Clarif tokens | Clarif lat (ms) | Success | SLO+Success | SPL | Steps | Replans | Deadlock rate | Wait mean (ms) | Wait P95 (ms) | Tokens in | Tokens after | Token reduction | Lat mean (ms) | Lat P50 | Lat P95 | Lat P99 | SLO (ms) | SLO viol. | Avg ctx before (chars) | Avg ctx after (chars) | Ctx reduction |\n")
    md.append("|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        md.append(
            "| {variant} | {n} | {executor} | {style} | {amb} | {ct} | {ctok} | {clat} | {success} | {slo_succ} | {spl} | {steps} | {replans} | {deadlock} | {waitm} | {wait95} | {toki} | {toka} | {tokred} | {lm} | {p50} | {p95} | {p99} | {slo} | {sv} | {ctxb} | {ctxa} | {red} |\n".format(
                variant=r["variant"],
                n=r["episodes"],
                executor=str(r.get("executor", "-") or "-"),
                style=r.get("instruction_style", "-"),
                amb=r.get("ambiguity_type", "-"),
                ct=_f(float(r.get("clarification_turns", float("nan")))),
                ctok=_f(float(r.get("clarification_tokens", float("nan")))),
                clat=_f(float(r.get("clarification_lat_ms", float("nan")))),
                success=_pct(r["success"]),
                slo_succ=_pct(float(r.get("slo_success", float("nan")))),
                spl=_f(r["spl"]),
                steps=_f(r["steps"]),
                replans=_f(r["replans"]),
                deadlock=_pct(r["deadlock_rate"]),
                waitm=_f(r["wait_time_ms_mean"]),
                wait95=_f(r["wait_time_ms_p95"]),
                toki=_f(r["tok_in_mean"]),
                toka=_f(r["tok_after_mean"]),
                tokred=_pct(r["tok_reduction_mean"]),
                lm=_f(r["lat_mean_ms"]),
                p50=_f(r["lat_p50_ms"]),
                p95=_f(r["lat_p95_ms"]),
                p99=_f(r["lat_p99_ms"]),
                slo=_f(float(r["slo_ms"])) if isinstance(r["slo_ms"], (int, float)) else "-",
                sv=_pct(r["slo_violation_rate"]),
                ctxb=_f(r["ctx_before_chars"]),
                ctxa=_f(r["ctx_after_chars"]),
                red=_pct(r["ctx_reduction"]),
            )
        )

    if vla_policy_calls_by_variant_ep:
        md.append("\n## VLA-aware latency accounting\n\n")
        md.append(
            "If `phase=vla_policy_call` is present, this run-level report treats it as the control-loop latency anchor "
            "and adds all replanning latency since the previous VLA call (per-episode timeline). "
            "Lat metrics above use this VLA-aware control-loop latency.\n\n"
        )
        md.append("| Variant | VLA phase events | VLA calls used | Replans total | Replans assigned | Replans unassigned | Skipped VLA events |\n")
        md.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            variant = str(r.get("variant", "-"))
            md.append(
                "| {variant} | {vlae} | {vlac} | {rt} | {ra} | {ru} | {skipped} |\n".format(
                    variant=variant,
                    vlae=int(r.get("vla_policy_call_events", 0)),
                    vlac=int(vla_assign_stats.get(variant, {}).get("vla_calls", 0.0)),
                    rt=int(r.get("vla_policy_call_replans_total", 0)),
                    ra=int(r.get("vla_policy_call_replans_assigned", 0)),
                    ru=int(r.get("vla_policy_call_replans_unassigned", 0)),
                    skipped=int(vla_policy_events_skipped),
                )
            )

    if phase_rows:
        md.append("\n## Phase latency breakdown\n\n")
        md.append("| Variant | Phase | N | Lat mean (ms) | Lat P95 | Lat P99 |\n")
        md.append("|---|---|---:|---:|---:|---:|\n")
        for r in phase_rows:
            md.append(
                "| {variant} | {phase} | {n} | {mean} | {p95} | {p99} |\n".format(
                    variant=r.get("variant", "-"),
                    phase=r.get("phase", "-"),
                    n=int(r.get("n", 0)),
                    mean=_f(float(r.get("lat_mean_ms", float("nan")))),
                    p95=_f(float(r.get("lat_p95_ms", float("nan")))),
                    p99=_f(float(r.get("lat_p99_ms", float("nan")))),
                )
            )

    payload = {
        "run_id": run_id,
        "domain": domain,
        "rows": rows,
        "phase_latency_breakdown": phase_rows,
        "vla_policy_events_skipped": vla_policy_events_skipped,
    }
    return "".join(md), payload


def aggregate_runs_root(runs_root: Path, pattern: str = "*", limit: int = 0) -> Tuple[str, Dict[str, Any]]:
    run_dirs = _discover_run_dirs(runs_root, pattern=pattern, limit=limit)
    all_rows: List[Dict[str, Any]] = []

    for rd in run_dirs:
        _md, payload = aggregate_run(rd)
        all_rows.extend(payload.get("rows", []))

    md: List[str] = []
    md.append(f"# Runs summary: `{runs_root}` ({len(run_dirs)} runs)\n")
    md.append("| Run | Domain | Variant | Episodes | Success | SPL | Replans | Tokens in (mean) | Tokens after (mean) | Token reduction | Lat P50 (ms) | Lat P95 (ms) | Lat P99 (ms) | SLO (ms) | SLO viol. |\n")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in all_rows:
        slo_ms = r.get("slo_ms", None)
        slo_s = _f(float(slo_ms)) if isinstance(slo_ms, (int, float)) else "-"
        md.append(
            "| {run_id} | {domain} | {variant} | {n} | {success} | {spl} | {replans} | {toki} | {toka} | {tokred} | {p50} | {p95} | {p99} | {slo} | {sv} |\n".format(
                run_id=r.get("run_id", "-"),
                domain=r.get("domain", "-") or "-",
                variant=r.get("variant", "-"),
                n=r.get("episodes", 0),
                success=_pct(float(r.get("success", float("nan")))),
                spl=_f(float(r.get("spl", float("nan")))),
                replans=_f(float(r.get("replans", float("nan")))),
                toki=_f(float(r.get("tok_in_mean", float("nan")))),
                toka=_f(float(r.get("tok_after_mean", float("nan")))),
                tokred=_pct(float(r.get("tok_reduction_mean", float("nan")))),
                p50=_f(float(r.get("lat_p50_ms", float("nan")))),
                p95=_f(float(r.get("lat_p95_ms", float("nan")))),
                p99=_f(float(r.get("lat_p99_ms", float("nan")))),
                slo=slo_s,
                sv=_pct(float(r.get("slo_violation_rate", float("nan")))),
            )
        )

    payload = {"runs_root": str(runs_root), "rows": all_rows}
    return "".join(md), payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="runs", help="Run dir (contains run.json) or runs root")
    ap.add_argument("--pattern", default="*", help="Glob under runs root (root mode only)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only aggregate latest N runs (root mode)")
    ap.add_argument("--out_md", default=None, help="Write markdown to this path")
    ap.add_argument("--out_json", default=None, help="Write JSON to this path")
    ap.add_argument(
        "--write_tables",
        action="store_true",
        help="Auto-write outputs under artifacts/tables/ with a timestamp (append-only).",
    )
    args = ap.parse_args()

    path = Path(args.path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if (path / "run.json").exists():
        md, payload = aggregate_run(path)
        default_stem = f"{path.name}__agg"
    else:
        md, payload = aggregate_runs_root(path, pattern=args.pattern, limit=args.limit)
        default_stem = "runs__agg"

    out_md = args.out_md
    out_json = args.out_json
    if args.write_tables and not out_md and not out_json:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = _REPO_ROOT / "artifacts" / "tables"
        out_md = str(out_dir / f"{default_stem}__{ts}.md")
        out_json = str(out_dir / f"{default_stem}__{ts}.json")

    if out_md:
        out_md_p = Path(out_md)
        if not out_md_p.is_absolute():
            out_md_p = _REPO_ROOT / out_md_p
        out_md_p.parent.mkdir(parents=True, exist_ok=True)
        out_md_p.write_text(md, encoding="utf-8")
    else:
        print(md)

    if out_json:
        out_json_p = Path(out_json)
        if not out_json_p.is_absolute():
            out_json_p = _REPO_ROOT / out_json_p
        out_json_p.parent.mkdir(parents=True, exist_ok=True)
        out_json_p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
