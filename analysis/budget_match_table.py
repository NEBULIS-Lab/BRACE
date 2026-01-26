from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    if q <= 0:
        return float(sorted(vals)[0])
    if q >= 1:
        return float(sorted(vals)[-1])
    s = sorted(vals)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return float(d0 + d1)


def _pct(x: float) -> str:
    if x != x:  # NaN
        return "-"
    return f"{100.0 * x:.1f}%"


def _f(x: float) -> str:
    if x != x:
        return "-"
    return f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}"


def _is_replan_event(ev: Dict[str, Any]) -> bool:
    et = ev.get("event_type")
    if et == "phase":
        return False
    if et == "replan":
        return True
    return any(k in ev for k in ("tokens_in", "tokens_after_prune", "lat_total_ms", "replan_cycle", "t", "step"))


def _variant_token_budget(
    *, variant: str, run_cfg: Dict[str, Any], episodes: List[Dict[str, Any]], events: List[Dict[str, Any]]
) -> Optional[int]:
    for ev in events:
        if str(ev.get("variant", "unknown")) != variant:
            continue
        b = ev.get("token_budget", None)
        if isinstance(b, (int, float)) and int(b) > 0:
            return int(b)
    for ep in episodes:
        if str(ep.get("variant", "unknown")) != variant:
            continue
        b = ep.get("token_budget", None)
        if isinstance(b, (int, float)) and int(b) > 0:
            return int(b)
    b = run_cfg.get("token_budget", None)
    if isinstance(b, (int, float)) and int(b) > 0:
        return int(b)
    return None


def _variant_group(
    *,
    group_key: str,
    variant: str,
    episodes: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> str:
    for ev in events:
        if str(ev.get("variant", "unknown")) != variant:
            continue
        g = ev.get(group_key, None)
        if g is not None and str(g).strip():
            return str(g).strip()
    for ep in episodes:
        if str(ep.get("variant", "unknown")) != variant:
            continue
        g = ep.get(group_key, None)
        if g is not None and str(g).strip():
            return str(g).strip()
    return variant


def summarize_run(run_dir: Path, group_key: str) -> Tuple[str, List[Dict[str, Any]]]:
    run_json = _read_json(run_dir / "run.json")
    run_id = str(run_json.get("run_id", run_dir.name))
    run_cfg = run_json.get("config") or {}

    episodes = list(_read_jsonl(run_dir / "episode_metrics.jsonl"))
    raw_events = list(_read_jsonl(run_dir / "events.jsonl"))
    events = [e for e in raw_events if _is_replan_event(e)]

    variants = set()
    for r in episodes:
        variants.add(r.get("variant", None))
    for r in events:
        variants.add(r.get("variant", None))
    variants.discard(None)
    if not variants:
        variants = {"unknown"}

    by_variant_eps: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_variant_eps[str(ep.get("variant", "unknown"))].append(ep)

    # Episode-set audit (P1): hash the order-sensitive episode_id sequence per variant.
    episode_set_hash: Dict[str, str | None] = {}
    episode_set_preview: Dict[str, str] = {}
    for v, eps in by_variant_eps.items():
        ids = [str(e.get("episode_id", "")).strip() for e in eps]
        ids = [x for x in ids if x]
        if not ids:
            episode_set_hash[v] = None
            episode_set_preview[v] = "-"
            continue
        h = hashlib.sha1(("|".join(ids)).encode("utf-8")).hexdigest()[:12]
        episode_set_hash[v] = h
        head = ", ".join(ids[:3])
        tail = ", ".join(ids[-3:]) if len(ids) > 3 else ""
        episode_set_preview[v] = head if not tail else f"{head} … {tail}"

    tok_after = defaultdict(list)
    tok_after_binding = defaultdict(list)
    lat_total = defaultdict(list)
    lat_total_binding = defaultdict(list)
    binding_counts = defaultdict(int)
    total_replans = defaultdict(int)

    for ev in events:
        v = str(ev.get("variant", "unknown"))
        total_replans[v] += 1

        ta = ev.get("tokens_after_prune", None)
        if isinstance(ta, (int, float)) and float(ta) >= 0:
            tok_after[v].append(float(ta))

        lt = ev.get("lat_total_ms", None)
        if isinstance(lt, (int, float)) and float(lt) >= 0:
            lat_total[v].append(float(lt))

        b = ev.get("token_budget", None)
        ti = ev.get("tokens_in", None)
        is_binding = (
            isinstance(b, (int, float))
            and isinstance(ti, (int, float))
            and float(b) > 0
            and float(ti) >= float(b)
        )
        if is_binding:
            binding_counts[v] += 1
            if isinstance(ta, (int, float)) and float(ta) >= 0:
                tok_after_binding[v].append(float(ta))
            if isinstance(lt, (int, float)) and float(lt) >= 0:
                lat_total_binding[v].append(float(lt))

    rows: List[Dict[str, Any]] = []
    for v in sorted(variants):
        eps = by_variant_eps.get(v, [])
        n = len(eps)
        success = sum(1.0 for e in eps if float(e.get("success", 0.0)) > 0.0) / n if n else float("nan")
        budget = _variant_token_budget(variant=v, run_cfg=run_cfg, episodes=episodes, events=events)
        group = _variant_group(group_key=group_key, variant=v, episodes=episodes, events=events)
        total = total_replans.get(v, 0)
        bind = binding_counts.get(v, 0)
        bind_rate = float(bind) / float(total) if total > 0 else float("nan")
        rows.append(
            {
                "variant": v,
                "group": group,
                "token_budget": budget,
                "episodes": n,
                "episode_set_hash": episode_set_hash.get(v),
                "episode_set_preview": episode_set_preview.get(v, "-"),
                "success": success,
                "spl": _safe_mean([float(e.get("spl", 0.0)) for e in eps]) if n else float("nan"),
                "replans": _safe_mean([float(e.get("replan_cycles", 0.0)) for e in eps]) if n else float("nan"),
                "tok_after_mean": _safe_mean(tok_after[v]),
                "tok_after_mean_binding": _safe_mean(tok_after_binding[v]),
                "lat_p95_ms": _percentile(lat_total[v], 0.95),
                "lat_p99_ms": _percentile(lat_total[v], 0.99),
                "lat_p95_ms_binding": _percentile(lat_total_binding[v], 0.95),
                "lat_p99_ms_binding": _percentile(lat_total_binding[v], 0.99),
                "budget_binding_rate": bind_rate,
                "replan_events": int(total),
                "replan_events_binding": int(bind),
            }
        )

    return run_id, rows


def _render(
    run_id: str,
    rows: List[Dict[str, Any]],
    group_key: str,
    budget: int,
    target_success: float,
    binding_only: bool,
) -> str:
    md: List[str] = []
    md.append(f"# Budget-matched table: `{run_id}`\n\n")
    md.append(f"- Group key: `{group_key}`\n")
    md.append(f"- Budget for Table A: `{budget}`\n")
    md.append(f"- Target success for Table B: `{target_success:.2f}`\n\n")
    md.append(f"- Tokens/latency events: `{'binding-only' if binding_only else 'all replans'}`\n\n")

    md.append("## A) Quality @ matched tokens\n\n")
    md.append(
        "| Group | Variant | Episodes | Success | SPL | Replans | Tokens after (mean) | Lat P95 (ms) | Lat P99 (ms) | Budget-binding |\n"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in sorted(
        [x for x in rows if int(x.get("token_budget") or 0) == int(budget)],
        key=lambda x: (str(x.get("group", "")), str(x.get("variant", ""))),
    ):
        md.append(
            "| {g} | {v} | {n} | {succ} | {spl} | {repl} | {ta} | {p95} | {p99} | {bind} |\n".format(
                g=r.get("group", r["variant"]),
                v=r["variant"],
                n=r["episodes"],
                succ=_pct(float(r["success"])),
                spl=_f(float(r["spl"])),
                repl=_f(float(r["replans"])),
                ta=_f(float(r["tok_after_mean_binding"] if binding_only else r["tok_after_mean"])),
                p95=_f(float(r["lat_p95_ms_binding"] if binding_only else r["lat_p95_ms"])),
                p99=_f(float(r["lat_p99_ms_binding"] if binding_only else r["lat_p99_ms"])),
                bind=_pct(float(r.get("budget_binding_rate", float("nan")))),
            )
        )

    md.append("\n")
    md.append("## C) Episode-set match audit\n\n")
    md.append("| Variant | Episodes | Episode-set hash | Episode-id preview (order-sensitive) |\n")
    md.append("|---|---:|---|---|\n")
    for r in sorted(rows, key=lambda x: str(x.get("variant", ""))):
        h = r.get("episode_set_hash")
        md.append(
            "| {v} | {n} | {h} | {p} |\n".format(
                v=str(r.get("variant", "-")),
                n=int(r.get("episodes") or 0),
                h=str(h) if h is not None else "-",
                p=str(r.get("episode_set_preview") or "-").replace("|", "\\|"),
            )
        )
    hashes = [str(r.get("episode_set_hash")) for r in rows if r.get("episode_set_hash") is not None]
    if hashes and len(set(hashes)) == 1:
        md.append("\n- Episode-set check: ✅ all variants share the same episode-id sequence.\n\n")
    elif hashes:
        md.append("\n- Episode-set check: ⚠️ variants differ in episode-id sequence (check seeding/env reset).\n\n")
    else:
        md.append("\n- Episode-set check: (episode_id missing; cannot audit)\n\n")

    md.append("## B) Systems @ matched quality\n\n")
    md.append("| Group | Chosen budget | Variant | Episodes | Success | Tokens after (mean) | Lat P95 (ms) | Lat P99 (ms) |\n")
    md.append("|---|---:|---|---:|---:|---:|---:|---:|\n")

    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[str(r.get("group", r["variant"]))].append(r)

    for group, gr in sorted(by_group.items(), key=lambda kv: kv[0]):
        # Prefer minimal tokens_after among those meeting the target success.
        candidates = [x for x in gr if float(x.get("success", 0.0)) >= float(target_success)]
        if candidates:
            chosen = min(candidates, key=lambda x: float(x.get("tok_after_mean", float("inf"))))
        else:
            # Fall back to best success if nothing meets threshold.
            chosen = max(gr, key=lambda x: float(x.get("success", 0.0)))

        md.append(
            "| {g} | {b} | {v} | {n} | {succ} | {ta} | {p95} | {p99} |\n".format(
                g=group,
                b=int(chosen.get("token_budget") or 0),
                v=str(chosen.get("variant", "-")),
                n=int(chosen.get("episodes") or 0),
                succ=_pct(float(chosen.get("success", float("nan")))),
                ta=_f(
                    float(
                        chosen.get(
                            "tok_after_mean_binding" if binding_only else "tok_after_mean", float("nan")
                        )
                    )
                ),
                p95=_f(
                    float(
                        chosen.get(
                            "lat_p95_ms_binding" if binding_only else "lat_p95_ms", float("nan")
                        )
                    )
                ),
                p99=_f(
                    float(
                        chosen.get(
                            "lat_p99_ms_binding" if binding_only else "lat_p99_ms", float("nan")
                        )
                    )
                ),
            )
        )

    return "".join(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Path to a single run directory (contains run.json)")
    ap.add_argument("--group_key", default="compress_strategy", help="Field used to group variants (default: compress_strategy)")
    ap.add_argument("--budget", type=int, default=0, help="Token budget used in Table A (0 means infer from run config)")
    ap.add_argument("--target_success", type=float, default=0.6, help="Success threshold for Table B")
    ap.add_argument(
        "--binding_only",
        action="store_true",
        help="Compute token/latency stats from budget-binding replans only (success is still episode-level).",
    )
    ap.add_argument("--out_md", default=None, help="Write markdown to this path")
    ap.add_argument("--out_json", default=None, help="Write JSON to this path")
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
        raise FileNotFoundError(f"Not a run dir (missing run.json): {run_dir}")

    run_json = _read_json(run_dir / "run.json")
    run_cfg = run_json.get("config") or {}
    inferred_budget = int(run_cfg.get("token_budget", 0)) if int(run_cfg.get("token_budget", 0)) > 0 else 0
    budget = int(args.budget) if int(args.budget) > 0 else inferred_budget

    run_id, rows = summarize_run(run_dir, group_key=str(args.group_key))
    md = _render(
        run_id,
        rows,
        group_key=str(args.group_key),
        budget=budget,
        target_success=float(args.target_success),
        binding_only=bool(args.binding_only),
    )

    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "group_key": str(args.group_key),
        "budget": int(budget),
        "target_success": float(args.target_success),
        "binding_only": bool(args.binding_only),
        "rows": rows,
    }

    out_md = args.out_md
    out_json = args.out_json
    if args.write_tables and not out_md and not out_json:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = _REPO_ROOT / "artifacts" / "tables"
        out_md = str(out_dir / f"{run_id}__budget_match__{ts}.md")
        out_json = str(out_dir / f"{run_id}__budget_match__{ts}.json")

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
