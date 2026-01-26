#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe(v: Any) -> str:
    if v is None:
        return "-"
    try:
        if isinstance(v, float):
            if v != v:  # NaN
                return "-"
            return f"{v:.2f}"
        return str(v)
    except Exception:
        return "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg_json", required=True, help="Path to an aggregate json produced by analysis/aggregate_runs.py")
    ap.add_argument("--out_md", required=True, help="Output markdown path")
    args = ap.parse_args()

    agg_path = Path(args.agg_json)
    payload: Dict[str, Any] = json.loads(agg_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = list(payload.get("phase_latency_breakdown") or [])

    out_lines: List[str] = []
    out_lines.append(f"# Phase breakdown: `{payload.get('run_id', '-')}`\n")
    out_lines.append(f"- Source: `{agg_path}`\n\n")
    out_lines.append("| Variant | Phase | N | Lat mean (ms) | Lat P95 | Lat P99 |\n")
    out_lines.append("|---|---|---:|---:|---:|---:|\n")
    for r in rows:
        out_lines.append(
            f"| {r.get('variant','-')} | {r.get('phase','-')} | {int(r.get('n',0) or 0)} | "
            f"{_safe(r.get('lat_mean_ms'))} | {_safe(r.get('lat_p95_ms'))} | {_safe(r.get('lat_p99_ms'))} |\n"
        )

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(out_lines), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()

