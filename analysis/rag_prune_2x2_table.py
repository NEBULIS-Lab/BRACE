from __future__ import annotations

import argparse
import datetime as _dt
import json
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
    if x != x:
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
    return any(k in ev for k in ("tokens_in", "tokens_after_prune", "lat_total_ms", "t", "step"))


def _token_budget_for_variant(run_cfg: Dict[str, Any], eps: List[Dict[str, Any]], evs: List[Dict[str, Any]]) -> Optional[int]:
    for ev in evs:
        b = ev.get("token_budget", None)
        if isinstance(b, (int, float)) and int(b) > 0:
            return int(b)
    for ep in eps:
        b = ep.get("token_budget", None)
        if isinstance(b, (int, float)) and int(b) > 0:
            return int(b)
    b = run_cfg.get("token_budget", None)
    if isinstance(b, (int, float)) and int(b) > 0:
        return int(b)
    return None


def summarize_run(run_dir: Path, *, binding_only: bool) -> Tuple[str, List[Dict[str, Any]]]:
    run_json = _read_json(run_dir / "run.json")
    run_id = str(run_json.get("run_id", run_dir.name))
    run_cfg = run_json.get("config") or {}

    episodes = list(_read_jsonl(run_dir / "episode_metrics.jsonl"))
    raw_events = list(_read_jsonl(run_dir / "events.jsonl"))
    events = [e for e in raw_events if _is_replan_event(e)]

    by_variant_eps: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_variant_eps[str(ep.get("variant", "unknown"))].append(ep)

    by_variant_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        by_variant_events[str(ev.get("variant", "unknown"))].append(ev)

    rows: List[Dict[str, Any]] = []
    for variant, evs in sorted(by_variant_events.items(), key=lambda kv: kv[0]):
        eps = by_variant_eps.get(variant, [])
        budget = _token_budget_for_variant(run_cfg, eps, evs)
        budget_f = float(budget or 0)

        selected = evs
        binding = []
        if budget is not None and budget > 0:
            binding = [
                e
                for e in evs
                if isinstance(e.get("tokens_in"), (int, float))
                and isinstance(e.get("token_budget"), (int, float))
                and float(e["token_budget"]) > 0
                and float(e["tokens_in"]) >= float(e["token_budget"])
            ]
        if binding_only and binding:
            selected = binding

        lat_vals = [float(e["lat_total_ms"]) for e in selected if isinstance(e.get("lat_total_ms"), (int, float))]
        tok_vals = [
            float(e["tokens_after_prune"]) for e in selected if isinstance(e.get("tokens_after_prune"), (int, float))
        ]
        lat_ret_vals = [
            float(e["lat_retrieval_ms"])
            for e in selected
            if isinstance(e.get("lat_retrieval_ms"), (int, float)) and float(e.get("lat_retrieval_ms")) >= 0
        ]
        ret_tok_vals = [
            float(e["retrieved_tokens"]) for e in selected if isinstance(e.get("retrieved_tokens"), (int, float))
        ]
        kept_tok_vals = [float(e["kept_tokens"]) for e in selected if isinstance(e.get("kept_tokens"), (int, float))]

        slo_ms = None
        for e in evs:
            if isinstance(e.get("slo_ms"), (int, float)) and float(e.get("slo_ms")) > 0:
                slo_ms = float(e.get("slo_ms"))
                break
        slo_viol = 0.0
        slo_den = 0.0
        for e in selected:
            lt = e.get("lat_total_ms")
            if not isinstance(lt, (int, float)) or slo_ms is None:
                continue
            slo_den += 1.0
            if e.get("slo_violation") is True:
                slo_viol += 1.0
            elif e.get("slo_violation") is False:
                pass
            else:
                slo_viol += 1.0 if float(lt) > float(slo_ms) else 0.0
        slo_rate = (slo_viol / slo_den) if slo_den > 0 else float("nan")

        success = sum(1.0 for e in eps if float(e.get("success", 0.0)) > 0.0) / len(eps) if eps else float("nan")

        rag_enabled = None
        pruning_enabled = None
        method = None
        rag_source = None
        if evs:
            rag_enabled = evs[0].get("rag_enabled", None)
            pruning_enabled = evs[0].get("pruning_enabled", None)
            method = evs[0].get("context_compress_method", None) or evs[0].get("context_strategy", None)
            rag_source = evs[0].get("rag_source", None)

        rows.append(
            {
                "variant": variant,
                "context_method": method,
                "rag_enabled": rag_enabled,
                "rag_source": rag_source,
                "pruning_enabled": pruning_enabled,
                "token_budget": budget,
                "episodes": len(eps),
                "success": success,
                "replan_events": len(evs),
                "binding_events": len(binding),
                "binding_rate": (float(len(binding)) / float(len(evs))) if evs else float("nan"),
                "tokens_after_mean": _safe_mean(tok_vals),
                "lat_p95_ms": _percentile(lat_vals, 0.95),
                "lat_p99_ms": _percentile(lat_vals, 0.99),
                "slo_violation_rate": slo_rate,
                "lat_retrieval_ms_mean": _safe_mean(lat_ret_vals),
                "retrieved_tokens_mean": _safe_mean(ret_tok_vals),
                "kept_tokens_mean": _safe_mean(kept_tok_vals),
            }
        )

    return run_id, rows


def render_md(run_id: str, rows: List[Dict[str, Any]], *, binding_only: bool) -> str:
    md: List[str] = []
    md.append(f"# RAG×Prune 2×2 table: `{run_id}`\n\n")
    md.append(f"- Event subset: `{'binding-only' if binding_only else 'all replans'}`\n\n")

    md.append(
        "| Variant | RAG | RagSrc | Prune | Method | Episodes | Success | Binding | Tokens after | LatRet mean (ms) | RetTok mean | KeptTok mean | Lat P95 (ms) | Lat P99 (ms) | SLO viol. |\n"
    )
    md.append("|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in sorted(
        rows,
        key=lambda x: (
            0 if bool(x.get("rag_enabled")) else 1,
            str(x.get("rag_source") or ""),
            0 if bool(x.get("pruning_enabled")) else 1,
            str(x.get("variant", "")),
        ),
    ):
        md.append(
            "| {v} | {rag} | {rs} | {pr} | {m} | {n} | {succ} | {bind} | {ta} | {lr} | {rt} | {kt} | {p95} | {p99} | {slo} |\n".format(
                v=str(r.get("variant", "-")),
                rag="1" if bool(r.get("rag_enabled")) else "0",
                rs=str(r.get("rag_source") or "-"),
                pr="1" if bool(r.get("pruning_enabled")) else "0",
                m=str(r.get("context_method", "-")),
                n=int(r.get("episodes") or 0),
                succ=_pct(float(r.get("success", float("nan")))),
                bind=_pct(float(r.get("binding_rate", float("nan")))),
                ta=_f(float(r.get("tokens_after_mean", float("nan")))),
                lr=_f(float(r.get("lat_retrieval_ms_mean", float("nan")))),
                rt=_f(float(r.get("retrieved_tokens_mean", float("nan")))),
                kt=_f(float(r.get("kept_tokens_mean", float("nan")))),
                p95=_f(float(r.get("lat_p95_ms", float("nan")))),
                p99=_f(float(r.get("lat_p99_ms", float("nan")))),
                slo=_pct(float(r.get("slo_violation_rate", float("nan")))),
            )
        )

    md.append("\n")
    md.append("Notes:\n")
    md.append("- `LatRet mean` is the auditable retrieval time proxy (`lat_retrieval_ms`).\n")
    md.append("- `RetTok mean`/`KeptTok mean` are retrieval payload tokens before/after pruning (`retrieved_tokens`, `kept_tokens`; proxy allowed).\n")
    md.append("- Binding is computed as `tokens_in >= token_budget` on replanning events.\n")
    return "".join(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Path to a single run directory (contains run.json)")
    ap.add_argument("--binding_only", action="store_true", help="Compute token/latency stats from binding replans only.")
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

    run_id, rows = summarize_run(run_dir, binding_only=bool(args.binding_only))
    md = render_md(run_id, rows, binding_only=bool(args.binding_only))

    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "binding_only": bool(args.binding_only),
        "rows": rows,
    }

    out_md = args.out_md
    out_json = args.out_json
    if args.write_tables and not out_md and not out_json:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = _REPO_ROOT / "artifacts" / "tables"
        out_md = str(out_dir / f"{run_id}__rag_prune_2x2__{ts}.md")
        out_json = str(out_dir / f"{run_id}__rag_prune_2x2__{ts}.json")

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
