from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
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


def _safe_mean(vals: List[float]) -> float:
    return float(mean(vals)) if vals else float("nan")


def _pct(x: float) -> str:
    if x != x:  # NaN
        return "-"
    return f"{100.0 * x:.1f}%"


def _f(x: float) -> str:
    if x != x:
        return "-"
    return f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}"


def build_table(run_dir: Path) -> Tuple[str, Dict[str, Any]]:
    episodes = list(_read_jsonl(run_dir / "episode_metrics.jsonl"))

    by_variant = defaultdict(list)
    for ep in episodes:
        by_variant[str(ep.get("variant", "unknown"))].append(ep)

    rows: List[Dict[str, Any]] = []
    for variant, eps in sorted(by_variant.items()):
        n = len(eps)
        success_rate = sum(1.0 for e in eps if float(e.get("success", 0.0)) > 0.0) / n if n else float("nan")
        rows.append(
            {
                "variant": variant,
                "n": n,
                "instruction_style": str(eps[0].get("instruction_style", "-")) if eps else "-",
                "ambiguity_type": str(eps[0].get("ambiguity_type", "-")) if eps else "-",
                "clarification_budget_turns": _safe_mean([float(e.get("clarification_budget_turns", 0.0)) for e in eps]),
                "clarification_tokens": _safe_mean([float(e.get("clarification_tokens", 0.0)) for e in eps]),
                "clarification_lat_ms": _safe_mean([float(e.get("clarification_lat_ms", 0.0)) for e in eps]),
                "success": success_rate,
                "steps": _safe_mean([float(e.get("step_count", 0.0)) for e in eps]),
                "replans": _safe_mean([float(e.get("replan_cycles", 0.0)) for e in eps]),
            }
        )

    md: List[str] = []
    md.append(f"# Clarification summary: `{run_dir.name}`\n")
    md.append(
        "| Variant | Style | Ambiguity | Clarif turns | Clarif tokens | Clarif lat (ms) | Episodes | Success | Steps | Replans |\n"
    )
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        md.append(
            "| {variant} | {style} | {amb} | {ct} | {tok} | {lat} | {n} | {succ} | {steps} | {rep} |\n".format(
                variant=r["variant"],
                style=r["instruction_style"],
                amb=r.get("ambiguity_type", "-"),
                ct=_f(float(r["clarification_budget_turns"])),
                tok=_f(float(r["clarification_tokens"])),
                lat=_f(float(r["clarification_lat_ms"])),
                n=r["n"],
                succ=_pct(float(r["success"])),
                steps=_f(float(r["steps"])),
                rep=_f(float(r["replans"])),
            )
        )

    return "".join(md), {"rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument(
        "--write_tables",
        action="store_true",
        help="Write outputs under artifacts/tables/ with a timestamp (append-only).",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    md, payload = build_table(run_dir)

    out_md = Path(args.out_md) if args.out_md else None
    out_json = Path(args.out_json) if args.out_json else None
    if args.write_tables:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_md = Path("artifacts/tables") / f"{run_dir.name}__clarification__{ts}.md"
        out_json = Path("artifacts/tables") / f"{run_dir.name}__clarification__{ts}.json"

    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
    else:
        print(md)

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.write_tables and out_md:
        print(str(out_md))


if __name__ == "__main__":
    main()
