from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return float(x.item())
    except Exception:
        pass
    return None


def _to_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, (int, float)):
        return bool(x)
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return bool(x.item())
    except Exception:
        pass
    return None


def _is_replan_event(ev: Dict[str, Any]) -> bool:
    et = ev.get("event_type")
    if et == "phase":
        return False
    if et == "replan":
        return True
    return "t" in ev and ("tokens_in" in ev or "lat_total_ms" in ev)


def aggregate_domainc(run_dir: Path) -> Tuple[str, Dict[str, Any]]:
    run_json = _read_json(run_dir / "run.json")
    run_id = str(run_json.get("run_id", run_dir.name))
    run_cfg = run_json.get("config") or {}
    slo_ms_default = run_cfg.get("slo_ms", None)
    if slo_ms_default is None:
        slo_ms_default = (run_cfg.get("brace_hparams", {}) or {}).get("slo_ms")

    episodes = list(_read_jsonl(run_dir / "episode_metrics.jsonl"))
    events_all = list(_read_jsonl(run_dir / "events.jsonl"))
    events = [e for e in events_all if _is_replan_event(e)]

    variants = sorted({str(r.get("variant", "unknown")) for r in (episodes + events)})
    if not variants:
        variants = ["unknown"]

    rows: List[Dict[str, Any]] = []
    for v in variants:
        eps = [e for e in episodes if str(e.get("variant", "unknown")) == v]
        evs = [e for e in events if str(e.get("variant", "unknown")) == v]

        n_eps = len(eps)
        success_rate = (
            sum(1.0 for e in eps if float(e.get("success", 0.0) or 0.0) > 0.0) / n_eps if n_eps else float("nan")
        )
        deadlock_rate = (
            sum(1.0 for e in eps if float(e.get("deadlock_flag", 0.0) or 0.0) > 0.0) / n_eps
            if n_eps
            else float("nan")
        )

        collisions_mean = _safe_mean([float(e.get("collisions_total", 0.0) or 0.0) for e in eps]) if n_eps else float("nan")
        near_misses_mean = _safe_mean([float(e.get("near_misses_total", 0.0) or 0.0) for e in eps]) if n_eps else float("nan")
        wait_time_mean = _safe_mean([float(e.get("wait_time_ms", 0.0) or 0.0) for e in eps]) if n_eps else float("nan")

        min_dist_vals = [x for x in (_to_float(e.get("min_interagent_dist_m")) for e in evs) if x is not None and x > 0]
        min_dist_min = min(min_dist_vals) if min_dist_vals else float("nan")

        tok_in = [x for x in (_to_float(e.get("tokens_in")) for e in evs) if x is not None and x > 0]
        tok_after = [x for x in (_to_float(e.get("tokens_after_prune")) for e in evs) if x is not None and x > 0]
        lat = [x for x in (_to_float(e.get("lat_total_ms")) for e in evs) if x is not None and x >= 0]

        tok_in_mean = _safe_mean(tok_in)
        tok_after_mean = _safe_mean(tok_after)
        tok_reduction = float("nan")
        if tok_in_mean == tok_in_mean and tok_after_mean == tok_after_mean and tok_in_mean > 0:
            tok_reduction = 1.0 - (tok_after_mean / tok_in_mean)

        lat_p50 = _percentile(lat, 0.50)
        lat_p95 = _percentile(lat, 0.95)
        lat_p99 = _percentile(lat, 0.99)

        slo_viol: List[float] = []
        for e in evs:
            sv = _to_bool(e.get("slo_violation"))
            slo_ms_f = _to_float(e.get("slo_ms"))
            if slo_ms_f is None:
                slo_ms_f = _to_float(slo_ms_default)
            if sv is not None:
                slo_viol.append(1.0 if sv else 0.0)
                continue
            lt = _to_float(e.get("lat_total_ms"))
            if lt is None or slo_ms_f is None or slo_ms_f <= 0:
                continue
            slo_viol.append(1.0 if (lt > slo_ms_f) else 0.0)
        slo_viol_rate = _safe_mean(slo_viol)

        rows.append(
            {
                "run_id": run_id,
                "variant": v,
                "episodes": n_eps,
                "success": success_rate,
                "deadlock_rate": deadlock_rate,
                "wait_time_ms_mean": wait_time_mean,
                "collisions_mean": collisions_mean,
                "near_misses_mean": near_misses_mean,
                "min_interagent_dist_min": min_dist_min,
                "tok_in_mean": tok_in_mean,
                "tok_after_mean": tok_after_mean,
                "tok_after_p95": _percentile(tok_after, 0.95),
                "tok_after_p99": _percentile(tok_after, 0.99),
                "tok_reduction_mean": tok_reduction,
                "lat_p50_ms": lat_p50,
                "lat_p95_ms": lat_p95,
                "lat_p99_ms": lat_p99,
                "slo_ms": _to_float(slo_ms_default),
                "slo_violation_rate": slo_viol_rate,
            }
        )

    md: List[str] = []
    md.append(f"# Domain C summary: `{run_id}`\n\n")
    md.append(
        "| Variant | Episodes | Success | Deadlock | Wait (ms/ep) | Collisions/ep | Near-miss/ep | Min dist (m) | Tokens in (mean) | Tokens after (mean) | Token reduction | Tokens after P95 | Tokens after P99 | Lat P50 (ms) | Lat P95 (ms) | Lat P99 (ms) | SLO (ms) | SLO viol. |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        md.append(
            "| {variant} | {n} | {succ} | {dl} | {wt} | {col} | {nm} | {mind} | {ti} | {ta} | {tr} | {tap95} | {tap99} | {lp50} | {lp95} | {lp99} | {slo} | {sv} |\n".format(
                variant=r["variant"],
                n=r["episodes"],
                succ=_pct(float(r["success"])),
                dl=_pct(float(r["deadlock_rate"])),
                wt=_f(float(r["wait_time_ms_mean"])),
                col=_f(float(r["collisions_mean"])),
                nm=_f(float(r["near_misses_mean"])),
                mind=_f(float(r["min_interagent_dist_min"])),
                ti=_f(float(r["tok_in_mean"])),
                ta=_f(float(r["tok_after_mean"])),
                tr=_pct(float(r["tok_reduction_mean"])),
                tap95=_f(float(r["tok_after_p95"])),
                tap99=_f(float(r["tok_after_p99"])),
                lp50=_f(float(r["lat_p50_ms"])),
                lp95=_f(float(r["lat_p95_ms"])),
                lp99=_f(float(r["lat_p99_ms"])),
                slo=_f(float(r["slo_ms"])) if r["slo_ms"] is not None else "-",
                sv=_pct(float(r["slo_violation_rate"])),
            )
        )

    payload = {"run_id": run_id, "rows": rows}
    return "".join(md), payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Run directory (contains run.json)")
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument(
        "--write_tables",
        action="store_true",
        help="Auto-write outputs under artifacts/tables/ with a timestamp (append-only).",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _REPO_ROOT / run_dir
    if not (run_dir / "run.json").exists():
        print(f"[ERROR] Not a run dir: {run_dir}", file=sys.stderr)
        sys.exit(2)

    md, payload = aggregate_domainc(run_dir)

    out_md = args.out_md
    out_json = args.out_json
    if args.write_tables and not out_md and not out_json:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = _REPO_ROOT / "artifacts" / "tables"
        stem = f"{run_dir.name}__domainc"
        out_md = str(out_dir / f"{stem}__{ts}.md")
        out_json = str(out_dir / f"{stem}__{ts}.json")

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
