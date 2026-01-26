from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _pct(x: float) -> str:
    if x != x:  # NaN
        return "-"
    return f"{100.0 * x:.1f}%"


def _classify_event(ev: Dict[str, Any]) -> str:
    et = ev.get("event_type")
    if et in ("replan", "phase"):
        return str(et)
    if "phase" in ev and "t" not in ev and "step" not in ev and "replan_cycle" not in ev:
        return "phase"
    return "replan"


def _mode_mix(counts: Counter, total: int) -> str:
    if total <= 0:
        return "-/-/-/-"
    return "{}/{}/{}/{}".format(
        _pct(counts.get("full_replan", 0) / total),
        _pct(counts.get("partial_replan", 0) / total),
        _pct(counts.get("reuse_subplan", 0) / total),
        _pct(counts.get("defer_replan", 0) / total),
    )


def _top_reasons(counts: Counter, total: int, k: int = 3) -> str:
    if total <= 0 or not counts:
        return "-"
    parts = []
    for reason, n in counts.most_common(k):
        parts.append(f"{reason} {_pct(n / total)}")
    return ", ".join(parts)


def build_table(run_dir: Path) -> Tuple[str, Dict[str, Any]]:
    run_json = _read_json(run_dir / "run.json")
    run_id = str(run_json.get("run_id", run_dir.name))

    events = [e for e in _read_jsonl(run_dir / "events.jsonl") if _classify_event(e) == "replan"]
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        by_variant[str(ev.get("variant", "unknown"))].append(ev)

    rows: List[Dict[str, Any]] = []
    for variant, evs in sorted(by_variant.items()):
        n = len(evs)
        mode_counts: Counter = Counter()
        reason_counts: Counter = Counter()
        planner_called = []
        hazard_slo = []
        hazard_churn = []
        hazard_deadlock = []

        for e in evs:
            mode = e.get("mode")
            if isinstance(mode, str):
                mode_counts[mode] += 1
            reason = e.get("brace_reason", e.get("reason", "unknown"))
            if reason is None:
                reason = "unknown"
            reason_counts[str(reason)] += 1
            planner_called.append(1.0 if bool(e.get("planner_called", False)) else 0.0)
            hazard_slo.append(1.0 if bool(e.get("hazard_slo", False)) else 0.0)
            hazard_churn.append(1.0 if bool(e.get("hazard_churn", False)) else 0.0)
            hazard_deadlock.append(1.0 if bool(e.get("hazard_deadlock", False)) else 0.0)

        rows.append(
            {
                "variant": variant,
                "replans": n,
                "planner_called_rate": _safe_mean(planner_called),
                "mode_counts": dict(mode_counts),
                "reason_counts": dict(reason_counts),
                "hazard_slo_rate": _safe_mean(hazard_slo),
                "hazard_churn_rate": _safe_mean(hazard_churn),
                "hazard_deadlock_rate": _safe_mean(hazard_deadlock),
                "mode_mix": _mode_mix(mode_counts, n),
                "top_reasons": _top_reasons(reason_counts, n),
            }
        )

    md: List[str] = []
    md.append(f"# BRACE mechanism summary (modes/reasons/hazards): `{run_id}`\n\n")
    md.append(
        "| Variant | Replans | Planner-called | Mode mix (F/P/R/D) | Top reasons | Hazards (SLO/Churn/Deadlock) |\n"
    )
    md.append("|---|---:|---:|---:|---|---:|\n")
    for r in rows:
        md.append(
            "| {v} | {n} | {pc} | {mix} | {reasons} | {hz} |\n".format(
                v=r["variant"],
                n=r["replans"],
                pc=_pct(float(r["planner_called_rate"])),
                mix=r["mode_mix"],
                reasons=r["top_reasons"],
                hz="{}/{}/{}".format(
                    _pct(float(r["hazard_slo_rate"])),
                    _pct(float(r["hazard_churn_rate"])),
                    _pct(float(r["hazard_deadlock_rate"])),
                ),
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

    md, payload = build_table(run_dir)

    out_md = args.out_md
    out_json = args.out_json
    if args.write_tables and not out_md and not out_json:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = _REPO_ROOT / "artifacts" / "tables"
        stem = f"{run_dir.name}__brace_mechanism"
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

