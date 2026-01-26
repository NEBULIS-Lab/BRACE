from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(x: float) -> str:
    if x != x:  # NaN
        return "-"
    return f"{100.0 * x:.1f}%"


def _f(x: float) -> str:
    if x != x:
        return "-"
    return f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}"


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return int(x)
    return None


def build_table(run_dir: Path) -> Tuple[str, Dict[str, Any]]:
    run_json = _read_json(run_dir / "run.json")
    run_id = str(run_json.get("run_id", run_dir.name))
    summary = _read_json(run_dir / "summary.json")
    rows_in: List[Dict[str, Any]] = list(summary.get("rows") or [])

    rows: List[Dict[str, Any]] = []
    for r in rows_in:
        hp = dict(r.get("brace_hparams") or {})
        variant = str(r.get("variant", "unknown"))
        brace = bool(variant.startswith("brace_"))
        cd = _to_int(hp.get("cooldown_steps")) if brace else None
        commit = _to_int(hp.get("min_commit_window")) if brace else None
        dlw = _to_int(hp.get("deadlock_window")) if brace else None
        rows.append(
            {
                "variant": variant,
                "brace": brace,
                "cooldown_steps": cd,
                "min_commit_window": commit,
                "deadlock_window": dlw,
                "episodes": int(r.get("episodes", 0) or 0),
                "success": float(r.get("success", float("nan"))),
                "planner_calls": float(r.get("planner_calls", float("nan"))),
                "plan_changes": float(r.get("plan_changes", float("nan"))),
                "churn_rate": float(r.get("churn_rate", float("nan"))),
                "deadlocks_per_ep": float(r.get("deadlocks_per_ep", float("nan"))),
                "stall_steps_per_ep": float(r.get("stall_steps_per_ep", float("nan"))),
                "slo_viol_rate": float(r.get("slo_viol_rate", float("nan"))),
            }
        )

    def _sort_key(rr: Dict[str, Any]) -> Tuple[int, int, int, str]:
        brace_rank = 0 if rr.get("brace") else 1
        cd_v = rr.get("cooldown_steps")
        cm_v = rr.get("min_commit_window")
        dlw_v = rr.get("deadlock_window")
        cd_i = cd_v if isinstance(cd_v, int) else 10**9
        cm_i = cm_v if isinstance(cm_v, int) else 10**9
        dlw_i = dlw_v if isinstance(dlw_v, int) else 10**9
        return (brace_rank, cd_i, cm_i, dlw_i, rr["variant"])

    rows = sorted(rows, key=_sort_key)

    md: List[str] = []
    md.append(f"# BRACE stability sweep (proxy): `{run_id}`\n\n")
    md.append(
        "| Variant | BRACE | cooldown | commit | deadlock_window | Episodes | Success | Planner calls/ep | Plan changes/ep | Churn (chg/call) | Deadlocks/ep | Stall steps/ep | SLO viol/call |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        md.append(
            "| {v} | {b} | {cd} | {cm} | {dlw} | {n} | {succ} | {pc} | {chg} | {churn} | {dl} | {stall} | {slo} |\n".format(
                v=r["variant"],
                b="yes" if r["brace"] else "no",
                cd=str(r["cooldown_steps"]) if r["cooldown_steps"] is not None else "-",
                cm=str(r["min_commit_window"]) if r["min_commit_window"] is not None else "-",
                dlw=str(r["deadlock_window"]) if r["deadlock_window"] is not None else "-",
                n=r["episodes"],
                succ=_pct(float(r["success"])),
                pc=_f(float(r["planner_calls"])),
                chg=_f(float(r["plan_changes"])),
                churn=_f(float(r["churn_rate"])),
                dl=_f(float(r["deadlocks_per_ep"])),
                stall=_f(float(r["stall_steps_per_ep"])),
                slo=_pct(float(r["slo_viol_rate"])),
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
        stem = f"{run_dir.name}__stability_sweep"
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
