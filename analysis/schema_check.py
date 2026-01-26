from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPLAN_REQUIRED_FIELDS = (
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

EPISODE_REQUIRED_FIELDS = (
    "run_id",
    "time_utc",
    "domain",
    "variant",
    "episode_id",
    "success",
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


def _classify_event(ev: Dict[str, Any]) -> str:
    et = ev.get("event_type")
    if et in ("replan", "phase"):
        return str(et)
    if "phase" in ev and "t" not in ev and "step" not in ev and "replan_cycle" not in ev:
        return "phase"
    return "replan"


def _missing_fields(records: Iterable[Dict[str, Any]], required: Tuple[str, ...]) -> Counter:
    c: Counter = Counter()
    for r in records:
        for k in required:
            if k == "t" and r.get("t", None) is None and r.get("step", None) is not None:
                continue
            v = r.get(k, None)
            if v is None:
                c[k] += 1
    return c


def _discover_runs(runs_root: Path, pattern: str = "*", limit: int = 0) -> List[Path]:
    run_dirs = sorted([p for p in runs_root.glob(pattern) if (p / "run.json").exists()])
    if limit and limit > 0:
        run_dirs = run_dirs[-limit:]
    return run_dirs


def check_run(run_dir: Path) -> Dict[str, Any]:
    events = list(_read_jsonl(run_dir / "events.jsonl"))
    episodes = list(_read_jsonl(run_dir / "episode_metrics.jsonl"))

    replan_events = [e for e in events if _classify_event(e) == "replan"]
    phase_events = [e for e in events if _classify_event(e) == "phase"]

    miss_replan = _missing_fields(replan_events, REPLAN_REQUIRED_FIELDS)
    miss_episode = _missing_fields(episodes, EPISODE_REQUIRED_FIELDS)

    return {
        "run_id": run_dir.name,
        "replan_events": len(replan_events),
        "phase_events": len(phase_events),
        "episode_rows": len(episodes),
        "missing_replan_required": dict(miss_replan),
        "missing_episode_required": dict(miss_episode),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="runs", help="Run dir (contains run.json) or runs root")
    ap.add_argument("--pattern", default="*", help="Glob under runs root (root mode only)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only check latest N runs (root mode)")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if any required fields are missing")
    args = ap.parse_args()

    path = Path(args.path)
    if (path / "run.json").exists():
        run_dirs = [path]
    else:
        run_dirs = _discover_runs(path, pattern=args.pattern, limit=args.limit)

    any_missing = False
    for rd in run_dirs:
        rep = check_run(rd)
        mr = {k: v for k, v in rep["missing_replan_required"].items() if v}
        me = {k: v for k, v in rep["missing_episode_required"].items() if v}
        if mr or me:
            any_missing = True
        print(
            json.dumps(
                {
                    "run_id": rep["run_id"],
                    "replan_events": rep["replan_events"],
                    "phase_events": rep["phase_events"],
                    "episode_rows": rep["episode_rows"],
                    "missing_replan_required": mr,
                    "missing_episode_required": me,
                },
                indent=2,
                sort_keys=True,
            )
        )

    if args.strict and any_missing:
        sys.exit(2)


if __name__ == "__main__":
    main()
