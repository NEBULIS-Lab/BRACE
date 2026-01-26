from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _classify_event(ev: Dict[str, Any]) -> str:
    et = ev.get("event_type")
    if et in ("replan", "phase"):
        return str(et)
    if "phase" in ev and "t" not in ev and "step" not in ev and "replan_cycle" not in ev:
        return "phase"
    return "replan"


def _na_rate(records: List[Dict[str, Any]], fields: Tuple[str, ...]) -> Dict[str, float]:
    if not records:
        return {k: 1.0 for k in fields}
    n = len(records)
    out: Dict[str, float] = {}
    for k in fields:
        miss = 0
        for r in records:
            v = r.get(k, None)
            if v is None:
                miss += 1
        out[k] = miss / n
    return out


def _count_missing(records: List[Dict[str, Any]], fields: Tuple[str, ...]) -> Counter:
    c: Counter = Counter()
    for r in records:
        for k in fields:
            if r.get(k, None) is None:
                c[k] += 1
    return c


def _ts() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


REPLAN_CORE_FIELDS = (
    "run_id",
    "time_utc",
    "domain",
    "variant",
    "episode_id",
    "t",
    "tokens_in",
    "tokens_after_prune",
    "lat_total_ms",
)

REPLAN_RECOMMENDED_FIELDS = (
    "task",
    "brace_enabled",
    "pruning_enabled",
    "rag_enabled",
    "summary_compress_enabled",
    "slo_ms",
    "token_budget",
    "clarification_budget_turns",
    "mode",
    "replan_interval_steps",
    "trigger_cooldown_steps",
    "replan_trigger_type",
    "lat_prune_ms",
    "lat_retrieval_ms",
    "lat_prefill_ms",
    "lat_decode_ms",
    "slo_violation",
    "slo_over_ms",
    "plan_hash",
    "plan_churn_score",
    "deadlock_flag",
    "wait_time_ms",
)

PHASE_CORE_FIELDS = (
    "run_id",
    "time_utc",
    "domain",
    "variant",
    "episode_id",
    "phase",
)

PHASE_RECOMMENDED_FIELDS = (
    "t",
    "lat_total_ms",
    "vlm_model",
    "vlm_tokens_in",
    "vlm_tokens_out",
    "lat_vlm_ms",
    "path",
    "fps",
    "frame_wh",
)

EPISODE_CORE_FIELDS = (
    "run_id",
    "time_utc",
    "domain",
    "variant",
    "episode_id",
    "success",
    "step_count",
    "replan_cycles",
)

EPISODE_RECOMMENDED_FIELDS = (
    "task",
    "spl",  # Habitat-only (may be null in other domains; reported as NA rate)
    "lat_p50_ms",
    "lat_p95_ms",
    "lat_p99_ms",
    "slo_violation_rate",
    "tokens_in_mean",
    "tokens_in_p95",
    "tokens_in_p99",
    "tokens_after_prune_mean",
    "tokens_after_prune_p95",
    "tokens_after_prune_p99",
    "deadlock_flag",
    "wait_time_ms",
)


CANON_PHASES = ("context_compress", "planner_call", "clarification", "vla_policy_call")


def _phase_bucket(phase: str) -> str:
    if phase.startswith("demo_") or phase == "demo":
        return "demo"
    return phase


def coverage_report(run_dir: Path) -> Dict[str, Any]:
    events = list(_read_jsonl(run_dir / "events.jsonl"))
    episodes = list(_read_jsonl(run_dir / "episode_metrics.jsonl"))

    replan_events = [e for e in events if _classify_event(e) == "replan"]
    phase_events = [e for e in events if _classify_event(e) == "phase"]

    phase_counts: Dict[str, int] = defaultdict(int)
    for e in phase_events:
        p = e.get("phase", None)
        if isinstance(p, str) and p:
            phase_counts[_phase_bucket(p)] += 1

    by_phase: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in phase_events:
        p = e.get("phase", None)
        if isinstance(p, str) and p:
            by_phase[_phase_bucket(p)].append(e)
        else:
            by_phase["<null>"].append(e)

    phase_tables = {}
    for p, rows in sorted(by_phase.items(), key=lambda kv: kv[0]):
        phase_tables[p] = _na_rate(rows, PHASE_RECOMMENDED_FIELDS)

    payload = {
        "run_id": run_dir.name,
        "paths": {
            "run_dir": str(run_dir),
            "run_json": str(run_dir / "run.json"),
            "events_jsonl": str(run_dir / "events.jsonl"),
            "episode_metrics_jsonl": str(run_dir / "episode_metrics.jsonl"),
        },
        "counts": {
            "replan_events": len(replan_events),
            "phase_events": len(phase_events),
            "episode_rows": len(episodes),
        },
        "replan": {
            "missing_core": dict(_count_missing(replan_events, REPLAN_CORE_FIELDS)),
            "na_rate_recommended": _na_rate(replan_events, REPLAN_RECOMMENDED_FIELDS),
        },
        "episode": {
            "missing_core": dict(_count_missing(episodes, EPISODE_CORE_FIELDS)),
            "na_rate_recommended": _na_rate(episodes, EPISODE_RECOMMENDED_FIELDS),
        },
        "phase": {
            "phase_counts": dict(phase_counts),
            "canon_phase_presence": {p: int(phase_counts.get(p, 0) > 0) for p in CANON_PHASES},
            "phase_na_rate": phase_tables,
        },
    }
    return payload


def _md_table_na_rates(title: str, na: Dict[str, float]) -> str:
    lines = [f"## {title}", "", "| Field | NA rate |", "|---|---:|"]
    for k, v in sorted(na.items(), key=lambda kv: kv[0]):
        lines.append(f"| `{k}` | {v:.1%} |")
    lines.append("")
    return "\n".join(lines)


def write_markdown(rep: Dict[str, Any]) -> str:
    run_id = rep["run_id"]
    lines: List[str] = []
    lines.append(f"# Schema coverage: `{run_id}`")
    lines.append("")
    c = rep["counts"]
    lines.append(f"- Replan events: {c['replan_events']}")
    lines.append(f"- Phase events: {c['phase_events']}")
    lines.append(f"- Episode rows: {c['episode_rows']}")
    lines.append("")

    miss_replan = {k: v for k, v in rep["replan"]["missing_core"].items() if v}
    miss_ep = {k: v for k, v in rep["episode"]["missing_core"].items() if v}
    lines.append("## Missing core fields (must-fix)")
    lines.append("")
    lines.append(f"- Replan core missing: {json.dumps(miss_replan, sort_keys=True)}")
    lines.append(f"- Episode core missing: {json.dumps(miss_ep, sort_keys=True)}")
    lines.append("")

    lines.append(_md_table_na_rates("Replan recommended fields (NA rate)", rep["replan"]["na_rate_recommended"]))
    lines.append(_md_table_na_rates("Episode recommended fields (NA rate)", rep["episode"]["na_rate_recommended"]))

    lines.append("## Phase presence (canonical)")
    lines.append("")
    lines.append("| Phase | Present | Count |")
    lines.append("|---|---:|---:|")
    for p in CANON_PHASES:
        present = rep["phase"]["canon_phase_presence"].get(p, 0)
        cnt = rep["phase"]["phase_counts"].get(p, 0)
        lines.append(f"| `{p}` | {present} | {cnt} |")
    demo_cnt = rep["phase"]["phase_counts"].get("demo", 0)
    lines.append(f"| `demo` | {int(demo_cnt > 0)} | {demo_cnt} |")
    lines.append("")

    lines.append("## Phase fields (NA rate by phase)")
    lines.append("")
    lines.append("| Phase | Field | NA rate |")
    lines.append("|---|---|---:|")
    for phase_name, na_map in sorted(rep["phase"]["phase_na_rate"].items(), key=lambda kv: kv[0]):
        for field, rate in sorted(na_map.items(), key=lambda kv: kv[0]):
            lines.append(f"| `{phase_name}` | `{field}` | {rate:.1%} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="runs", help="Run dir (contains run.json) or runs root")
    ap.add_argument("--run", action="append", default=[], help="Specific run dir name under runs root (repeatable)")
    ap.add_argument("--write_tables", action="store_true", help="Write append-only markdown/json under artifacts/tables/")
    args = ap.parse_args()

    path = Path(args.path)
    run_dirs: List[Path] = []
    if (path / "run.json").exists():
        run_dirs = [path]
    else:
        runs_root = path
        if args.run:
            run_dirs = [runs_root / r for r in args.run]
        else:
            raise SystemExit("In runs-root mode, pass at least one --run <run_id>.")

    out_root = Path("artifacts/tables")
    out_root.mkdir(parents=True, exist_ok=True)

    for rd in run_dirs:
        rep = coverage_report(rd)
        md = write_markdown(rep)
        print(md)
        if args.write_tables:
            ts = _ts()
            md_path = out_root / f"{rd.name}__schema_coverage__{ts}.md"
            js_path = out_root / f"{rd.name}__schema_coverage__{ts}.json"
            md_path.write_text(md, encoding="utf-8")
            js_path.write_text(json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

