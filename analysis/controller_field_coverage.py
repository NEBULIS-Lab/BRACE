from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]

FIELDS = (
    "mode",
    "token_budget",
    "hazard_slo",
    "hazard_churn",
    "hazard_deadlock",
    "hazard_unsafe",
    "cooldown_active",
    "rollback_flag",
    "min_commit_window",
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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_event_type(event: Dict[str, Any]) -> str:
    if event.get("event_type") in ("replan", "phase"):
        return str(event["event_type"])
    if "phase" in event and "t" not in event and "step" not in event and "replan_cycle" not in event:
        return "phase"
    return "replan"


def _pct(x: float) -> str:
    if x != x:
        return "-"
    return f"{100.0 * x:.1f}%"


def _coverage_row(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(events)
    missing: Dict[str, int] = {k: 0 for k in FIELDS}
    absent: Dict[str, int] = {k: 0 for k in FIELDS}
    for e in events:
        for k in FIELDS:
            if k not in e:
                absent[k] += 1
                missing[k] += 1
            elif e.get(k, None) is None:
                missing[k] += 1
    return {"n": n, "missing": missing, "absent": absent}


def audit_run(run_dir: Path) -> Tuple[str, Dict[str, Any]]:
    run_json = _read_json(run_dir / "run.json")
    run_id = str(run_json.get("run_id", run_dir.name))

    events = [e for e in _read_jsonl(run_dir / "events.jsonl") if _infer_event_type(e) == "replan"]
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_variant[str(e.get("variant", "unknown"))].append(e)

    rows: List[Dict[str, Any]] = []
    for variant, evs in sorted(by_variant.items()):
        stats = _coverage_row(evs)
        n = int(stats["n"])
        missing = stats["missing"]
        absent = stats["absent"]

        missing_rate = {k: (missing[k] / n if n > 0 else float("nan")) for k in FIELDS}
        absent_rate = {k: (absent[k] / n if n > 0 else float("nan")) for k in FIELDS}
        fully_missing = [k for k in FIELDS if n > 0 and missing[k] == n]
        rows.append(
            {
                "variant": variant,
                "replan_events": n,
                "missing_rate": missing_rate,
                "absent_rate": absent_rate,
                "fully_missing": fully_missing,
            }
        )

    # Overall row across variants.
    stats_all = _coverage_row(events)
    n_all = int(stats_all["n"])
    missing_all = stats_all["missing"]
    missing_rate_all = {k: (missing_all[k] / n_all if n_all > 0 else float("nan")) for k in FIELDS}
    fully_missing_all = [k for k in FIELDS if n_all > 0 and missing_all[k] == n_all]

    md: List[str] = []
    md.append(f"# Controller field coverage: `{run_id}`\n\n")
    md.append(f"- Run dir: `{run_dir}`\n")
    md.append(f"- Replan events: {n_all}\n\n")

    md.append("## Overall (all variants)\n\n")
    md.append("| Field | NA rate |\n")
    md.append("|---|---:|\n")
    for k in FIELDS:
        md.append(f"| `{k}` | {_pct(float(missing_rate_all[k]))} |\n")
    if fully_missing_all:
        md.append(f"\nFully missing in all events: {', '.join(f'`{k}`' for k in fully_missing_all)}\n")

    md.append("\n## By variant\n\n")
    header = (
        "| Variant | Replans | "
        + " | ".join(f"`{k}` NA" for k in ("mode", "token_budget"))
        + " | `hazard_*` NA | "
        + " | ".join(f"`{k}` NA" for k in ("cooldown_active", "rollback_flag", "min_commit_window"))
        + " | Fully missing |\n"
    )
    md.append(header)
    md.append("|---|---:|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        mr = r["missing_rate"]
        hazard_na = max(float(mr["hazard_slo"]), float(mr["hazard_churn"]), float(mr["hazard_deadlock"]), float(mr["hazard_unsafe"]))
        md.append(
            "| {v} | {n} | {mode} | {tb} | {hz} | {cd} | {rb} | {mcw} | {fm} |\n".format(
                v=r["variant"],
                n=int(r["replan_events"]),
                mode=_pct(float(mr["mode"])),
                tb=_pct(float(mr["token_budget"])),
                hz=_pct(hazard_na),
                cd=_pct(float(mr["cooldown_active"])),
                rb=_pct(float(mr["rollback_flag"])),
                mcw=_pct(float(mr["min_commit_window"])),
                fm=",".join(r["fully_missing"]) if r["fully_missing"] else "-",
            )
        )

    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "fields": list(FIELDS),
        "overall": {"replan_events": n_all, "missing_rate": missing_rate_all, "fully_missing": fully_missing_all},
        "by_variant": rows,
    }
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

    md, payload = audit_run(run_dir)

    out_md = args.out_md
    out_json = args.out_json
    if args.write_tables and not out_md and not out_json:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = _REPO_ROOT / "artifacts" / "tables"
        stem = f"{run_dir.name}__controller_field_coverage"
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

