from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


_DEFAULT_FREQ_MAP_STEPS = {20: "1×", 10: "2×", 5: "5×", 2: "10×"}
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


def _i(x: float) -> str:
    if x != x:
        return "-"
    return str(int(round(float(x))))


def _format_freq_label(*, interval_steps: float, base_interval_steps: int) -> str | None:
    """Return a paper-friendly frequency label.

    - If interval_steps matches the default map exactly, use its canonical label.
    - Otherwise, fall back to a best-effort ratio label relative to base_interval_steps.
    """

    if interval_steps != interval_steps:  # NaN
        return None
    try:
        interval_i = int(round(float(interval_steps)))
    except Exception:
        return None

    if interval_i in _DEFAULT_FREQ_MAP_STEPS:
        return _DEFAULT_FREQ_MAP_STEPS[interval_i]

    if base_interval_steps <= 0 or interval_i <= 0:
        return None

    ratio = float(base_interval_steps) / float(interval_i)
    if ratio <= 0 or ratio != ratio:
        return None
    nearest = round(ratio)
    if abs(ratio - nearest) < 0.05:
        return f"{int(nearest)}×"
    return f"{ratio:.1f}×"


def _primary_trigger_from_event(event: Dict[str, Any]) -> str:
    if event.get("replan_trigger_type") is not None:
        return str(event["replan_trigger_type"])

    trig = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
    types = trig.get("types") if isinstance(trig.get("types"), list) else []
    types = [str(t) for t in types if t is not None]

    unsafe = bool(trig.get("unsafe", False))
    deadlock = bool(trig.get("deadlock", False)) or ("deadlock" in types)
    failure = bool(trig.get("failure", False)) or ("failure" in types)
    periodic = bool(trig.get("periodic", False))

    if unsafe:
        return "unsafe"
    if deadlock:
        return "deadlock"
    if failure:
        return "failure"

    # Keep the first non-(failure/deadlock) extra trigger type if any.
    for t in types:
        if t not in ("failure", "deadlock"):
            return t

    if periodic:
        return "periodic"
    return "unknown"


def _format_counter(counter: Counter[str], *, top_k: int = 3) -> Tuple[str, Dict[str, float]]:
    total = float(sum(counter.values()))
    if total <= 0:
        return "-", {}

    frac: Dict[str, float] = {k: float(v) / total for k, v in counter.items()}
    items = counter.most_common(max(1, int(top_k)))
    shown = 0
    parts: List[str] = []
    for k, v in items:
        shown += int(v)
        parts.append(f"{k}:{_pct(float(v) / total)}")
    rest = int(total) - shown
    if rest > 0:
        parts.append(f"other:{_pct(float(rest) / total)}")
    return " ".join(parts), frac


def _read_run_config(run_dir: Path) -> Dict[str, Any]:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return {}
    try:
        payload = json.loads(run_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cfg = payload.get("config")
    return cfg if isinstance(cfg, dict) else {}


def build_table(run_dir: Path, *, top_k: int = 3, base_interval_steps: int = 20) -> Tuple[str, Dict[str, Any]]:
    run_cfg = _read_run_config(run_dir)
    env_step_ms = run_cfg.get("env_step_ms")
    try:
        env_step_ms = float(env_step_ms) if env_step_ms is not None else None
    except Exception:
        env_step_ms = None

    episodes = list(_read_jsonl(run_dir / "episode_metrics.jsonl"))
    events = list(_read_jsonl(run_dir / "events.jsonl"))

    by_variant_eps: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_variant_eps[str(ep.get("variant", "unknown"))].append(ep)

    trig_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for ev in events:
        if str(ev.get("event_type", "replan")) != "replan":
            continue
        variant = str(ev.get("variant", "unknown"))
        trig_counts[variant][_primary_trigger_from_event(ev)] += 1

    rows: List[Dict[str, Any]] = []
    for variant, eps in sorted(by_variant_eps.items()):
        n = len(eps)

        replans = _safe_mean(
            [float(e.get("effective_replans_per_episode", e.get("replan_cycles", 0.0))) for e in eps]
        )

        replans_per_min_vals: List[float] = []
        for e in eps:
            if e.get("effective_replans_per_min") is not None:
                try:
                    replans_per_min_vals.append(float(e.get("effective_replans_per_min")))
                    continue
                except Exception:
                    pass
            if e.get("episode_wall_time_ms") is not None:
                try:
                    wall_ms = float(e.get("episode_wall_time_ms"))
                    rep = float(e.get("effective_replans_per_episode", e.get("replan_cycles", 0.0)))
                    if wall_ms > 0:
                        replans_per_min_vals.append(rep / (wall_ms / 60000.0))
                        continue
                except Exception:
                    pass
            if env_step_ms is not None and env_step_ms > 0 and e.get("step_count") is not None:
                try:
                    steps = float(e.get("step_count"))
                    rep = float(e.get("effective_replans_per_episode", e.get("replan_cycles", 0.0)))
                    wall_ms = steps * float(env_step_ms)
                    if wall_ms > 0:
                        replans_per_min_vals.append(rep / (wall_ms / 60000.0))
                        continue
                except Exception:
                    pass
        replans_per_min = _safe_mean(replans_per_min_vals)

        interval_steps = _safe_mean(
            [
                float(
                    e.get(
                        "replan_interval_steps",
                        e.get("replan_interval", float("nan")),
                    )
                )
                for e in eps
            ]
        )
        cooldown_steps = _safe_mean([float(e.get("trigger_cooldown_steps", float("nan"))) for e in eps])
        freq_label = _format_freq_label(interval_steps=interval_steps, base_interval_steps=int(base_interval_steps))

        suppressed_vals: List[float] = []
        for e in eps:
            if e.get("suppressed_triggers") is not None:
                suppressed_vals.append(float(e.get("suppressed_triggers", 0.0)))
        suppressed = _safe_mean(suppressed_vals)

        mix_str, mix_frac = _format_counter(trig_counts.get(variant, Counter()), top_k=top_k)

        rows.append(
            {
                "variant": variant,
                "episodes": n,
                "replan_interval_steps": interval_steps,
                "freq_label": freq_label,
                "trigger_cooldown_steps": cooldown_steps,
                "effective_replans_per_episode": replans,
                "effective_replans_per_min": replans_per_min,
                "suppressed_triggers_per_episode": suppressed,
                "trigger_counts": dict(trig_counts.get(variant, Counter())),
                "trigger_mix": mix_frac,
                "trigger_mix_str": mix_str,
            }
        )

    md: List[str] = []
    md.append(f"# Trigger audit summary: `{run_dir.name}`\n")
    md.append(
        "Primary trigger types come from `replan_trigger_type` when present, otherwise a best-effort parse of `trigger`.\n\n"
    )
    md.append(
        "Frequency mapping (steps → label): "
        + ", ".join([f"{k}→{v}" for k, v in sorted(_DEFAULT_FREQ_MAP_STEPS.items(), reverse=True)])
        + f" (fallback: {int(base_interval_steps)} / interval_steps).\n\n"
    )
    md.append(
        "| Variant | Episodes | Freq | Interval (steps) | Cooldown (steps) | Replans/ep | Replans/min | Trigger mix (primary) | Suppressed/ep |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---|---:|\n")
    for r in rows:
        md.append(
            "| {variant} | {n} | {freq} | {intv} | {cd} | {rep} | {rpm} | {mix} | {sup} |\n".format(
                variant=r["variant"],
                n=r["episodes"],
                freq=str(r.get("freq_label") or "-"),
                intv=_i(float(r["replan_interval_steps"])),
                cd=_i(float(r["trigger_cooldown_steps"])),
                rep=_f(float(r["effective_replans_per_episode"])),
                rpm=_f(float(r["effective_replans_per_min"])),
                mix=str(r["trigger_mix_str"]),
                sup=_f(float(r["suppressed_triggers_per_episode"])),
            )
        )

    return "".join(md), {"rows": rows, "freq_map_steps": dict(_DEFAULT_FREQ_MAP_STEPS), "freq_base_interval_steps": int(base_interval_steps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument(
        "--freq_base_interval_steps",
        type=int,
        default=20,
        help="Base interval used for fallback frequency labels when interval_steps is not in the canonical map.",
    )
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

    md, payload = build_table(
        run_dir,
        top_k=int(args.top_k),
        base_interval_steps=int(args.freq_base_interval_steps),
    )

    out_md = args.out_md
    out_json = args.out_json
    if args.write_tables and not out_md and not out_json:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = _REPO_ROOT / "artifacts" / "tables"
        stem = f"{run_dir.name}__trigger_audit"
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
