#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class VariantRow:
    name: str
    row: dict[str, Any]

    def get(self, key: str) -> Any:
        return self.row.get(key, None)


def _pct(x: Any) -> str:
    try:
        xf = float(x)
    except Exception:
        return "-"
    if not math.isfinite(xf):
        return "-"
    return f"{100.0 * xf:.1f}%"


def _num(x: Any, digits: int = 2) -> str:
    try:
        xf = float(x)
    except Exception:
        return "-"
    if not math.isfinite(xf):
        return "-"
    return f"{xf:.{digits}f}"


def _ms(x: Any) -> str:
    try:
        xf = float(x)
    except Exception:
        return "-"
    if not math.isfinite(xf):
        return "-"
    return f"{xf:.0f}"


def _delta(a: Any, b: Any, digits: int = 2) -> str:
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return "-"
    if not (math.isfinite(af) and math.isfinite(bf)):
        return "-"
    return f"{(af - bf):.{digits}f}"


def _delta_pct(a: Any, b: Any) -> str:
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return "-"
    if not (math.isfinite(af) and math.isfinite(bf)):
        return "-"
    return f"{100.0 * (af - bf):.1f}pp"


def load_rows(table_paths: list[str]) -> dict[str, VariantRow]:
    by_variant: dict[str, VariantRow] = {}
    for p in table_paths:
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
        for r in obj.get("rows", []):
            v = r.get("variant")
            if not v:
                continue
            by_variant[str(v)] = VariantRow(name=str(v), row=r)
    return by_variant


def summarize_one(
    name: str,
    by_variant: dict[str, VariantRow],
    baseline_variant: str,
    primary_method_variant: str,
    secondary_variants: list[str],
) -> dict[str, Any]:
    baseline = by_variant.get(baseline_variant)
    primary = by_variant.get(primary_method_variant)

    missing: list[str] = []
    for v in [baseline_variant, primary_method_variant, *secondary_variants]:
        if v not in by_variant:
            missing.append(v)

    def pick(v: VariantRow | None, key: str) -> Any:
        return None if v is None else v.get(key)

    out: dict[str, Any] = {
        "name": name,
        "baseline_variant": baseline_variant,
        "primary_method_variant": primary_method_variant,
        "missing_variants": missing,
        "baseline": baseline.row if baseline else None,
        "primary": primary.row if primary else None,
        "secondary": {v: by_variant[v].row for v in secondary_variants if v in by_variant},
        "deltas": {},
    }

    # Common metrics across tables in this repo (Domain A/B/C + proxy).
    metrics = [
        ("success", "success", "pp"),
        ("slo_violation_rate", "slo_violation_rate", "pp"),
        ("tok_after_mean", "tok_after_mean", "abs"),
        ("tok_reduction_mean", "tok_reduction_mean", "pp"),
        ("lat_p95_ms", "lat_p95_ms", "abs"),
        ("wait_time_ms_mean", "wait_time_ms_mean", "abs"),
        ("deadlock_rate", "deadlock_rate", "pp"),
        ("near_misses_mean", "near_misses_mean", "abs"),
        ("min_interagent_dist_min", "min_interagent_dist_min", "abs"),
        ("spl", "spl", "abs"),
        ("slo_success_rate", "slo_success_rate", "pp"),
    ]

    for key, src, kind in metrics:
        a = pick(primary, src)
        b = pick(baseline, src)
        if kind == "pp":
            out["deltas"][key] = _delta_pct(a, b)
        else:
            out["deltas"][key] = _delta(a, b)

    return out


def write_md(out_path: Path, summaries: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Claim check (numeric, cross-domain)\n\n")
    lines.append(
        "This is a **sanity check**: do we observe improvements in the runs we have (tables JSON), "
        "under the comparison pairs described by the manifest.\n\n"
    )

    lines.append("| Comparison | Baseline | Primary | Δ Success | Δ SLO viol | Δ Tokens after | Δ Lat P95 | Δ SLO+Success | Missing variants |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|\n")
    for s in summaries:
        d = s.get("deltas", {}) or {}
        lines.append(
            "| {name} | `{b}` | `{p}` | {ds} | {dv} | {dt} | {dl} | {dss} | {miss} |\n".format(
                name=s.get("name", "-"),
                b=s.get("baseline_variant", "-"),
                p=s.get("primary_method_variant", "-"),
                ds=str(d.get("success", "-")),
                dv=str(d.get("slo_violation_rate", "-")),
                dt=str(d.get("tok_after_mean", "-")),
                dl=str(d.get("lat_p95_ms", "-")),
                dss=str(d.get("slo_success_rate", "-")),
                miss=", ".join(s.get("missing_variants", []) or []) if (s.get("missing_variants") or []) else "-",
            )
        )

    lines.append("\n## Details\n\n")
    for s in summaries:
        lines.append(f"### {s.get('name','-')}\n\n")
        lines.append(f"- baseline: `{s.get('baseline_variant','-')}`\n")
        lines.append(f"- primary: `{s.get('primary_method_variant','-')}`\n")
        if s.get("missing_variants"):
            lines.append(f"- missing: {', '.join('`'+v+'`' for v in s['missing_variants'])}\n")

        d = s.get("deltas", {}) or {}
        lines.append("\n| Metric | Primary | Baseline | Δ |\n")
        lines.append("|---|---:|---:|---:|\n")

        p = s.get("primary") or {}
        b = s.get("baseline") or {}

        def add_row(label: str, key: str, fmt):
            lines.append(
                "| {label} | {pv} | {bv} | {dv} |\n".format(
                    label=label,
                    pv=fmt(p.get(key)),
                    bv=fmt(b.get(key)),
                    dv=str(d.get(key, "-")),
                )
            )

        add_row("Success", "success", _pct)
        add_row("SLO violation rate", "slo_violation_rate", _pct)
        add_row("SLO+Success", "slo_success_rate", _pct)
        add_row("Tokens after (mean)", "tok_after_mean", _num)
        add_row("Token reduction (mean)", "tok_reduction_mean", _pct)
        add_row("Latency P95 (ms)", "lat_p95_ms", _ms)
        add_row("Deadlock rate", "deadlock_rate", _pct)
        add_row("Wait time mean (ms)", "wait_time_ms_mean", _ms)

        # Domain-specific (best-effort; may be missing).
        add_row("Near-miss mean", "near_misses_mean", _num)
        add_row("Min inter-agent dist (min)", "min_interagent_dist_min", _num)
        add_row("SPL", "spl", _num)

        lines.append("\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="configs/paper/claim_check_manifest.json",
        help="Manifest JSON describing which table JSONs and variants to compare.",
    )
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument(
        "--write_tables",
        action="store_true",
        help="Write outputs under artifacts/tables/ with a timestamp (append-only).",
    )
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    comps = manifest.get("comparisons", [])

    summaries: list[dict[str, Any]] = []
    for c in comps:
        by_variant = load_rows(c["tables"])
        summaries.append(
            summarize_one(
                name=c["name"],
                by_variant=by_variant,
                baseline_variant=c["baseline_variant"],
                primary_method_variant=c["primary_method_variant"],
                secondary_variants=c.get("secondary_variants", []),
            )
        )

    payload = {
        "generated_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "manifest": args.manifest,
        "summaries": summaries,
    }

    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    out_md = Path(args.out_md) if args.out_md else None
    out_json = Path(args.out_json) if args.out_json else None
    if args.write_tables:
        out_md = Path("artifacts/tables") / f"{ts}__claim_check__paper_summary.md"
        out_json = Path("artifacts/tables") / f"{ts}__claim_check__paper_summary.json"

    if out_md:
        write_md(out_md, summaries)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if out_md:
        print(str(out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

