from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]

# Top-level fields expected on *replan* events for WS3 auditing.
FIELDS = (
    "trigger",
    "replan_trigger_type",
    "replan_interval_steps",
    "trigger_cooldown_steps",
)

# Nested trigger keys that are commonly used downstream.
TRIGGER_KEYS = (
    "types",
    "periodic",
    "failure",
    "deadlock",
    "unsafe",
    "cooldown_active",
)

INFERABLE_TRIGGER_KEYS = (
    "types",
    "periodic",
    "failure",
    "deadlock",
    "unsafe",
    "cooldown_active",
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
    trig_missing: Dict[str, int] = {k: 0 for k in TRIGGER_KEYS}
    trig_absent: Dict[str, int] = {k: 0 for k in TRIGGER_KEYS}
    trig_missing_effective: Dict[str, int] = {k: 0 for k in INFERABLE_TRIGGER_KEYS}

    for e in events:
        for k in FIELDS:
            if k not in e:
                absent[k] += 1
                missing[k] += 1
                continue
            v = e.get(k, None)
            if v is None:
                missing[k] += 1
            if k == "trigger" and v is not None and not isinstance(v, dict):
                # Treat non-dict triggers as missing for downstream auditing.
                missing[k] += 1

        trig = e.get("trigger")
        trig = trig if isinstance(trig, dict) else None
        for k in TRIGGER_KEYS:
            if trig is None:
                trig_absent[k] += 1
                trig_missing[k] += 1
                continue
            if k not in trig:
                trig_absent[k] += 1
                trig_missing[k] += 1
                continue
            if trig.get(k, None) is None:
                trig_missing[k] += 1

        # Effective coverage (best-effort inference): count missing only if we truly cannot infer.
        # This is meant to avoid "paper narrative" inconsistency due to redundant keys not being
        # explicitly written in `trigger` dicts. Raw coverage remains the source-of-truth.
        replan_trigger_type = e.get("replan_trigger_type")
        replan_trigger_type = str(replan_trigger_type) if replan_trigger_type is not None else None
        types: List[str] = []
        if trig is not None and isinstance(trig.get("types"), list):
            types = [str(x) for x in trig.get("types", []) if x is not None]
        # If `types` is missing, we can still infer a minimal list from the primary trigger.
        inferred_types = list(types)
        if not inferred_types and replan_trigger_type in ("periodic", "failure", "deadlock", "unsafe"):
            inferred_types = [replan_trigger_type]

        for k in INFERABLE_TRIGGER_KEYS:
            if k == "types":
                # Inferable if either present as list or inferred from primary trigger.
                if (trig is not None and isinstance(trig.get("types"), list)) or bool(inferred_types):
                    continue
                trig_missing_effective[k] += 1
                continue

            if k == "cooldown_active":
                # Replan events are only logged when the trigger is allowed; so cooldown_active is effectively false.
                # If a runner logs a top-level cooldown flag, respect it.
                if e.get("cooldown_active") is not None:
                    continue
                # Otherwise, infer false (not missing).
                continue

            if k in ("periodic", "failure", "deadlock", "unsafe"):
                # Inferable from either explicit bool fields or membership in `types` or primary trigger.
                if trig is not None and trig.get(k) is not None:
                    continue
                if k in inferred_types:
                    continue
                if replan_trigger_type == k:
                    continue
                # If none of the above, infer false (not missing).
                continue

    return {
        "n": n,
        "missing": missing,
        "absent": absent,
        "trigger_missing": trig_missing,
        "trigger_absent": trig_absent,
        "trigger_missing_effective": trig_missing_effective,
    }


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
        trig_missing = stats["trigger_missing"]
        trig_absent = stats["trigger_absent"]
        trig_missing_eff = stats["trigger_missing_effective"]

        missing_rate = {k: (missing[k] / n if n > 0 else float("nan")) for k in FIELDS}
        absent_rate = {k: (absent[k] / n if n > 0 else float("nan")) for k in FIELDS}
        trig_missing_rate = {k: (trig_missing[k] / n if n > 0 else float("nan")) for k in TRIGGER_KEYS}
        trig_absent_rate = {k: (trig_absent[k] / n if n > 0 else float("nan")) for k in TRIGGER_KEYS}
        trig_missing_rate_eff = {k: (trig_missing_eff[k] / n if n > 0 else float("nan")) for k in INFERABLE_TRIGGER_KEYS}

        fully_missing = [k for k in FIELDS if n > 0 and missing[k] == n]
        fully_missing_trig = [k for k in TRIGGER_KEYS if n > 0 and trig_missing[k] == n]
        rows.append(
            {
                "variant": variant,
                "replan_events": n,
                "missing_rate": missing_rate,
                "absent_rate": absent_rate,
                "trigger_missing_rate": trig_missing_rate,
                "trigger_absent_rate": trig_absent_rate,
                "trigger_missing_rate_effective": trig_missing_rate_eff,
                "fully_missing": fully_missing,
                "fully_missing_trigger": fully_missing_trig,
            }
        )

    stats_all = _coverage_row(events)
    n_all = int(stats_all["n"])
    missing_all = stats_all["missing"]
    trig_missing_all = stats_all["trigger_missing"]
    trig_missing_eff_all = stats_all["trigger_missing_effective"]
    missing_rate_all = {k: (missing_all[k] / n_all if n_all > 0 else float("nan")) for k in FIELDS}
    trig_missing_rate_all = {k: (trig_missing_all[k] / n_all if n_all > 0 else float("nan")) for k in TRIGGER_KEYS}
    trig_missing_rate_eff_all = {
        k: (trig_missing_eff_all[k] / n_all if n_all > 0 else float("nan")) for k in INFERABLE_TRIGGER_KEYS
    }

    fully_missing_all = [k for k in FIELDS if n_all > 0 and missing_all[k] == n_all]
    fully_missing_trig_all_raw = [k for k in TRIGGER_KEYS if n_all > 0 and trig_missing_all[k] == n_all]
    fully_missing_trig_all_effective = [
        k for k in INFERABLE_TRIGGER_KEYS if n_all > 0 and trig_missing_eff_all.get(k, 0) == n_all
    ]

    md: List[str] = []
    md.append(f"# Trigger field coverage: `{run_id}`\n\n")
    md.append(f"- Run dir: `{run_dir}`\n")
    md.append(f"- Replan events: {n_all}\n\n")

    md.append("## Overall (all variants)\n\n")
    md.append("| Field | NA rate |\n")
    md.append("|---|---:|\n")
    for k in FIELDS:
        md.append(f"| `{k}` | {_pct(float(missing_rate_all[k]))} |\n")
    if fully_missing_all:
        md.append(f"\nFully missing in all events: {', '.join(f'`{k}`' for k in fully_missing_all)}\n")

    md.append("\n### `trigger` nested keys (overall)\n\n")
    md.append("| Key | NA rate (defaults-applied) | NA rate (explicit) |\n")
    md.append("|---|---:|---:|\n")
    for k in TRIGGER_KEYS:
        eff = trig_missing_rate_eff_all.get(k, float("nan"))
        md.append(f"| `{k}` | {_pct(float(eff))} | {_pct(float(trig_missing_rate_all[k]))} |\n")
    if fully_missing_trig_all_effective:
        md.append(
            "\nDefaults-applied fully-missing nested keys (cannot be inferred): "
            + ", ".join(f"`{k}`" for k in fully_missing_trig_all_effective)
            + ".\n"
        )

    md.append("\n## By variant\n\n")
    md.append(
        "| Variant | Replans | "
        + " | ".join(f"`{k}` NA" for k in FIELDS)
        + " | `trigger.types` NA (default/explicit) | `trigger.periodic` NA (default/explicit) | `trigger.failure` NA (default/explicit) | `trigger.cooldown_active` NA (default/explicit) | Explicitly missing |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        mr = r["missing_rate"]
        tmr = r["trigger_missing_rate"]
        tmr_eff = r["trigger_missing_rate_effective"]
        md.append(
            "| {v} | {n} | {tr} | {tt} | {ri} | {cd} | {types} | {per} | {fail} | {cda} | {fm} |\n".format(
                v=r["variant"],
                n=int(r["replan_events"]),
                tr=_pct(float(mr["trigger"])),
                tt=_pct(float(mr["replan_trigger_type"])),
                ri=_pct(float(mr["replan_interval_steps"])),
                cd=_pct(float(mr["trigger_cooldown_steps"])),
                types=f"{_pct(float(tmr_eff.get('types', float('nan'))))}/{_pct(float(tmr['types']))}",
                per=f"{_pct(float(tmr_eff.get('periodic', float('nan'))))}/{_pct(float(tmr['periodic']))}",
                fail=f"{_pct(float(tmr_eff.get('failure', float('nan'))))}/{_pct(float(tmr['failure']))}",
                cda=f"{_pct(float(tmr_eff.get('cooldown_active', float('nan'))))}/{_pct(float(tmr['cooldown_active']))}",
                fm=",".join(r["fully_missing"] + [f"trigger.{k}" for k in r["fully_missing_trigger"]])
                if (r["fully_missing"] or r["fully_missing_trigger"])
                else "-",
            )
        )

    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "fields": list(FIELDS),
        "trigger_keys": list(TRIGGER_KEYS),
        "overall": {
            "replan_events": n_all,
            "missing_rate": missing_rate_all,
            "trigger_missing_rate": trig_missing_rate_all,
            "trigger_missing_rate_effective": trig_missing_rate_eff_all,
            "fully_missing": fully_missing_all,
            "fully_missing_trigger_raw": fully_missing_trig_all_raw,
            "fully_missing_trigger_effective": fully_missing_trig_all_effective,
        },
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
        stem = f"{run_dir.name}__trigger_field_coverage"
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
