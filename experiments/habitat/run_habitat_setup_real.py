from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from brace.controller import BraceController, BraceHyperparams, BraceState
from experiments.common.clarification import build_pointnav_instruction
from experiments.common import replan_schedule as rs
from experiments.common.logging import RunContext


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_habitat_setup(script_path: Path, args: List[str], env: Dict[str, str]) -> int:
    cmd = [env.get("BRACE_HABITAT_PY", sys.executable), str(script_path), *args]
    p = subprocess.run(cmd, env=env)
    return p.returncode


def _env_without_proxies() -> Dict[str, str]:
    env = dict(os.environ)
    for k in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        env.pop(k, None)
    return env


def _pick_habitat_python() -> str:
    candidates = [
        os.path.expanduser("~/miniconda3/envs/habitat/bin/python"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return sys.executable


def _append_runs_index(runs_root: str, payload: Dict[str, Any]) -> None:
    path = Path(runs_root) / "index.jsonl"
    payload = dict(payload)
    payload.setdefault("time_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_habitat_setup_modules(habitat_setup_root: Path):
    src = habitat_setup_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from habitat_multi_agent_wrapper import HabitatMultiAgentWrapper  # type: ignore
    from habitat_erecap_integration import HabitatERECAPIntegration  # type: ignore

    return HabitatMultiAgentWrapper, HabitatERECAPIntegration


def _tokenize_ids(tokenizer: Any, text: str) -> List[int]:
    tok = tokenizer(text, add_special_tokens=False)
    input_ids = tok.get("input_ids", [])
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids or [])


def _decode_ids(tokenizer: Any, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def _count_tokens(tokenizer: Any, text: str) -> int:
    try:
        return len(_tokenize_ids(tokenizer, text))
    except Exception:
        return max(1, len(str(text).split()))


def _truncate_recency(tokenizer: Any, text: str, token_budget: int) -> Tuple[str, Dict[str, Any]]:
    ids = _tokenize_ids(tokenizer, text)
    tokens_in = len(ids)
    token_budget = max(1, int(token_budget))
    kept = ids[-token_budget:] if tokens_in > token_budget else ids
    pruned_text = _decode_ids(tokenizer, kept)
    return pruned_text, {"pruning_applied": True, "input_tokens": tokens_in, "output_tokens": len(kept)}

def _filler_ids(tokenizer: Any, token_budget: int, *, token: str = "SUMMARY") -> List[int]:
    token_budget = max(0, int(token_budget))
    if token_budget <= 0:
        return []
    # Create enough tokens to fill `token_budget` deterministically.
    base_text = (token + " ") * max(1, token_budget)
    ids = _tokenize_ids(tokenizer, base_text.strip())
    if not ids:
        return []
    if len(ids) >= token_budget:
        return ids[:token_budget]
    # Repeat if tokenizer produces fewer ids than requested.
    reps = (token_budget // len(ids)) + 1
    out = (ids * reps)[:token_budget]
    return out


def _install_context_pruner(planner: Any) -> None:
    """Monkeypatch CooperativeMultiAgentPlanner._prune_context for extra baselines.

    This keeps default behavior when no extra strategy is set.
    """
    if getattr(planner, "_brace_patched", False):
        return

    orig = planner._prune_context
    tokenizer = planner.tokenizer

    def _patched(context_text: str, use_pruning: bool = True, **kwargs) -> Tuple[str, Dict]:
        strategy = getattr(planner, "_brace_context_strategy", None)
        token_budget = getattr(planner, "_brace_token_budget", None)
        if strategy is None:
            try:
                return orig(context_text, use_pruning=use_pruning, **kwargs)
            except TypeError:
                return orig(context_text, use_pruning=use_pruning)

        summary_head_tokens = int(getattr(planner, "_brace_summary_head_tokens", 40))
        summary_tail_tokens = int(getattr(planner, "_brace_summary_tail_tokens", 80))
        base_seed = getattr(planner, "_brace_random_seed", None)
        base_seed_int = int(base_seed) if base_seed is not None else 0
        method = str(strategy).strip().lower()

        # Prefer native E-RECAP pruning path for learned pruning only; baselines are handled locally
        # to avoid budget<mandatory-token failures when the context grows large.
        if method in ("erecap", "none"):
            try:
                pruned_text, stats = orig(
                    context_text,
                    use_pruning=(method == "erecap"),
                    context_compress_method=method,
                    token_budget=int(token_budget) if (method == "erecap" and token_budget is not None) else None,
                    random_seed=base_seed_int,
                    summary_head_tokens=summary_head_tokens,
                    summary_tail_tokens=summary_tail_tokens,
                    **kwargs,
                )
                stats = dict(stats or {})
                stats.setdefault("context_strategy", str(strategy))
                stats.setdefault("token_budget", int(token_budget) if token_budget is not None else None)
                return pruned_text, stats
            except Exception:
                pass

        if strategy == "erecap":
            try:
                pruned_text, stats = orig(context_text, use_pruning=True, **kwargs)
            except TypeError:
                pruned_text, stats = orig(context_text, use_pruning=True)
            stats = dict(stats or {})
            stats["context_strategy"] = "erecap"
            stats["token_budget"] = int(token_budget) if token_budget is not None else None
            return pruned_text, stats

        if strategy == "none":
            try:
                pruned_text, stats = orig(context_text, use_pruning=False, **kwargs)
            except TypeError:
                pruned_text, stats = orig(context_text, use_pruning=False)
            stats = dict(stats or {})
            stats["context_strategy"] = "none"
            stats["token_budget"] = int(token_budget) if token_budget is not None else None
            return pruned_text, stats

        # Paper-faithful baselines: progressive layer-wise pruning, token-count matched by construction.
        if strategy in ("random_layerwise", "recency_layerwise"):
            token_select_strategy = "random" if strategy == "random_layerwise" else "recency"
            try:
                pruned_text, stats = orig(
                    context_text,
                    use_pruning=True,
                    token_select_strategy=token_select_strategy,
                    random_seed=base_seed_int,
                    **kwargs,
                )
            except TypeError:
                pruned_text, stats = orig(
                    context_text,
                    use_pruning=True,
                    token_select_strategy=token_select_strategy,
                    random_seed=base_seed_int,
                )
            stats = dict(stats or {})
            stats["context_strategy"] = str(strategy)
            stats["token_budget"] = None
            return pruned_text, stats

        if strategy == "recency":
            if token_budget is None:
                return orig(context_text, use_pruning=False)
            pruned_text, stats = _truncate_recency(tokenizer, context_text, int(token_budget))
            stats = dict(stats)
            stats["context_strategy"] = "recency"
            stats["token_budget"] = int(token_budget)
            return pruned_text, stats

        if strategy == "random":
            if token_budget is None:
                return orig(context_text, use_pruning=False)

            token_budget_int = max(1, int(token_budget))
            ids = _tokenize_ids(tokenizer, context_text)
            tokens_in = len(ids)
            if tokens_in <= token_budget_int:
                return context_text, {
                    "pruning_applied": False,
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_in,
                    "context_strategy": "random",
                    "token_budget": token_budget_int,
                }

            head = min(max(0, summary_head_tokens), token_budget_int, tokens_in)
            remaining = max(0, token_budget_int - head)
            tail = min(max(0, summary_tail_tokens), remaining, max(0, tokens_in - head))
            rand_k = max(0, token_budget_int - head - tail)

            mid_start = head
            mid_end = max(mid_start, tokens_in - tail)
            mid_indices = list(range(mid_start, mid_end))

            rng = random.Random(base_seed_int)
            chosen = rng.sample(mid_indices, k=min(rand_k, len(mid_indices))) if rand_k > 0 else []
            kept_indices = sorted(list(range(head)) + chosen + (list(range(tokens_in - tail, tokens_in)) if tail else []))
            pruned_ids = [ids[i] for i in kept_indices][:token_budget_int]
            pruned_text = _decode_ids(tokenizer, pruned_ids)
            return pruned_text, {
                "pruning_applied": True,
                "input_tokens": tokens_in,
                "output_tokens": len(pruned_ids),
                "context_strategy": "random",
                "token_budget": token_budget_int,
                "random_seed": base_seed_int,
                "head_tokens": head,
                "tail_tokens": tail,
                "random_tokens": rand_k,
                "kept_indices_head": kept_indices[:5],
                "kept_indices_tail": kept_indices[-5:],
            }

        if strategy == "structured_summary":
            if token_budget is None:
                return orig(context_text, use_pruning=False)

            token_budget_int = max(1, int(token_budget))
            ids = _tokenize_ids(tokenizer, context_text)
            tokens_in = len(ids)
            if tokens_in <= token_budget_int:
                return context_text, {
                    "pruning_applied": False,
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_in,
                    "context_strategy": "structured_summary",
                    "token_budget": token_budget_int,
                    "head_tokens": min(int(getattr(planner, "_brace_summary_head_tokens", 8)), tokens_in),
                    "tail_tokens": 0,
                    "summary_tokens": 0,
                }

            head = min(max(0, summary_head_tokens), token_budget_int, tokens_in)
            remaining = max(0, token_budget_int - head)
            tail = min(max(0, summary_tail_tokens), remaining, max(0, tokens_in - head))
            summary_tokens = max(0, token_budget_int - head - tail)

            mid_start = head
            mid_end = max(mid_start, tokens_in - tail)
            mid_len = max(0, mid_end - mid_start)

            t0 = time.perf_counter()
            summary_ids: List[int] = []
            if summary_tokens > 0 and mid_len > 0:
                if summary_tokens >= mid_len:
                    summary_ids = list(ids[mid_start:mid_end])
                else:
                    stride = float(mid_len) / float(summary_tokens)
                    for i in range(summary_tokens):
                        idx = mid_start + int((i + 0.5) * stride)
                        idx = min(max(mid_start, idx), mid_end - 1)
                        summary_ids.append(int(ids[idx]))

            filler = (
                _filler_ids(tokenizer, summary_tokens - len(summary_ids), token="SUMMARY")
                if summary_tokens > len(summary_ids)
                else []
            )
            summary_time_s = float(time.perf_counter() - t0) if summary_tokens > 0 else 0.0
            tail_ids = list(ids[-tail:]) if tail > 0 else []
            pruned_ids = list(ids[:head]) + summary_ids + filler + tail_ids
            pruned_text = _decode_ids(tokenizer, pruned_ids)
            return pruned_text, {
                "pruning_applied": True,
                "input_tokens": tokens_in,
                "output_tokens": len(pruned_ids),
                "context_strategy": "structured_summary",
                "token_budget": token_budget_int,
                "head_tokens": head,
                "tail_tokens": tail,
                "summary_tokens": summary_tokens,
                "summary_time_s": float(summary_time_s),
                "summary_head_tokens": summary_head_tokens,
                "summary_tail_tokens": summary_tail_tokens,
            }

        if strategy == "erecap_clamp":
            try:
                pruned_text, stats = orig(context_text, use_pruning=True, **kwargs)
            except TypeError:
                pruned_text, stats = orig(context_text, use_pruning=True)
            if token_budget is None:
                return pruned_text, stats
            pruned_text2, clamp_stats = _truncate_recency(tokenizer, pruned_text, int(token_budget))
            merged = dict(stats or {})
            # Make the top-level output token count reflect the final text actually fed to the model.
            # Keep E-RECAP's pre-clamp length under clamp_* fields for auditing.
            try:
                merged["output_tokens"] = int(clamp_stats.get("output_tokens"))  # type: ignore[arg-type]
            except Exception:
                pass
            merged.update({f"clamp_{k}": v for k, v in (clamp_stats or {}).items()})
            merged["context_strategy"] = "erecap_clamp"
            merged["token_budget"] = int(token_budget)
            return pruned_text2, merged

        # "none" or unknown: fall back to baseline.
        try:
            pruned_text, stats = orig(context_text, use_pruning=False, **kwargs)
        except TypeError:
            pruned_text, stats = orig(context_text, use_pruning=False)
        stats = dict(stats or {})
        stats["context_strategy"] = "none"
        stats["token_budget"] = int(token_budget) if token_budget is not None else None
        return pruned_text, stats

    planner._prune_context = _patched
    planner._brace_patched = True


def _fallback_action(agent_state: Dict[str, Any], *, stop_distance_m: float = 0.2) -> Dict[str, str]:
    pointgoal = agent_state.get("pointgoal")
    if pointgoal is not None and len(pointgoal) >= 2:
        goal_distance = float(pointgoal[0])
        goal_angle = float(pointgoal[1])
        if goal_distance < float(stop_distance_m):
            return {"action": "stop"}
        if abs(goal_angle) > 0.3:
            return {"action": "turn_left" if goal_angle > 0 else "turn_right"}
        return {"action": "move_forward"}
    return {"action": "move_forward"}

def _goal_distance_m(agent_state: Dict[str, Any]) -> float:
    pointgoal = agent_state.get("pointgoal")
    if pointgoal is not None and len(pointgoal) >= 1 and isinstance(pointgoal[0], (int, float)):
        return float(pointgoal[0])
    dist = agent_state.get("distance_to_goal")
    if isinstance(dist, (int, float)):
        return float(dist)
    return float("inf")


def _sanitize_action(
    action: Dict[str, str], agent_state: Dict[str, Any], *, stop_distance_m: float = 0.2
) -> Dict[str, str]:
    """Prevent pathological early STOP from LLM outputs (common in nav prompts)."""
    if action.get("action") != "stop":
        return action
    # Only allow STOP when we're plausibly near the goal.
    if _goal_distance_m(agent_state) < float(stop_distance_m):
        return action
    return _fallback_action(agent_state, stop_distance_m=float(stop_distance_m))


def _extract_success_distance(env: Any) -> Optional[float]:
    try:
        cfg = getattr(env, "config", None)
        succ = cfg.habitat.task.measurements.success
        return float(getattr(succ, "success_distance"))
    except Exception:
        return None


def _try_make_shortest_path_follower(env: Any, goal_radius: Optional[float]) -> Tuple[Optional[Any], Optional[Any]]:
    try:
        from habitat.sims.habitat_simulator.actions import HabitatSimActions  # type: ignore
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower  # type: ignore

        if goal_radius is None:
            goal_radius = _extract_success_distance(env) or 0.2
        follower = ShortestPathFollower(env.env.sim, goal_radius=float(goal_radius), return_one_hot=False)
        return follower, HabitatSimActions
    except Exception:
        return None, None


def _demo_obs_rgb_uint8(observations: Any):
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None

    rgb = None
    if isinstance(observations, dict):
        rgb = observations.get("rgb")
    else:
        try:
            rgb = dict(observations).get("rgb")
        except Exception:
            rgb = None
    if rgb is None:
        return None

    arr = np.asarray(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        try:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        except Exception:
            arr = arr.astype(np.uint8)
    return arr


def _vlm_heuristic_rgb_stats_text(observations: Any) -> str:
    """Lightweight Track-S placeholder: summarize RGB into a short text block.

    This is NOT a real VLM. It exists to exercise the accounting + interfaces without introducing
    a heavy dependency or large model downloads.
    """
    rgb = _demo_obs_rgb_uint8(observations)
    if rgb is None:
        return "rgb: (missing)"
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(rgb).astype(np.float32)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        mean = arr.reshape(-1, 3).mean(axis=0)
        std = arr.reshape(-1, 3).std(axis=0)
        mn = arr.reshape(-1, 3).min(axis=0)
        mx = arr.reshape(-1, 3).max(axis=0)
        return (
            f"rgb_hw={h}x{w}; mean={mean.round(1).tolist()}; std={std.round(1).tolist()}; "
            f"min={mn.round(0).astype(int).tolist()}; max={mx.round(0).astype(int).tolist()}"
        )
    except Exception:
        return "rgb: (stats_failed)"


def _demo_put_text(img, text: str, org: Tuple[int, int], *, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    import cv2  # type: ignore

    # Scale text to be readable for both 256p and 512p+ demo exports.
    try:
        h = int(getattr(img, "shape", [256])[0])
    except Exception:
        h = 256
    scale = max(0.5, 0.5 * (float(h) / 256.0))
    thickness = max(1, int(round(scale * 2)))
    outline = max(thickness + 1, thickness * 3)

    # Outline for readability.
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, tuple(int(x) for x in color), thickness, cv2.LINE_AA)


def _demo_render_frame(
    env: Any,
    observations: Any,
    *,
    rgb_hw: Tuple[int, int],
    map_hw: Tuple[int, int],
    overlay_lines: List[str],
):
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    out_h = int(max(int(rgb_hw[0]), int(map_hw[0])))

    rgb = _demo_obs_rgb_uint8(observations)
    if rgb is None:
        rgb = np.zeros((out_h, int(rgb_hw[1]), 3), dtype=np.uint8)
    rgb = cv2.resize(rgb, (int(rgb_hw[1]), out_h), interpolation=cv2.INTER_AREA)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    try:
        top_down = env.get_top_down_map(output_size=map_hw)
        top_down = cv2.resize(top_down, (int(map_hw[1]), out_h), interpolation=cv2.INTER_NEAREST)
        map_bgr = cv2.cvtColor(top_down, cv2.COLOR_RGB2BGR)
    except Exception:
        map_bgr = np.zeros((out_h, int(map_hw[1]), 3), dtype=np.uint8)

    frame = np.concatenate([rgb_bgr, map_bgr], axis=1)

    y = int(max(18, round(18 * (float(out_h) / 256.0))))
    y_step = int(max(18, round(18 * (float(out_h) / 256.0))))
    for line in overlay_lines[:10]:
        color = (255, 255, 255)
        if str(line).startswith("SLO_VIOL"):
            color = (0, 0, 255)
        _demo_put_text(frame, line, (10, y), color=color)
        y += y_step

    return frame


def _demo_open_writer(path: Path, *, fps: int, frame_wh: Tuple[int, int]):
    import cv2  # type: ignore

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), frame_wh)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path}")
    return writer


def _run_v2(cfg: Dict[str, Any], ctx: RunContext, habitat_setup_root: Path) -> None:
    HabitatMultiAgentWrapper, HabitatERECAPIntegration = _load_habitat_setup_modules(habitat_setup_root)

    base_seed = int(cfg.get("seed", 0))
    if base_seed:
        _seed_everything(base_seed)

    demo_cfg: Dict[str, Any] = dict(cfg.get("demo", {}) or {})
    demo_enabled = bool(demo_cfg.get("enabled", False))
    demo_variants = set([str(x) for x in (demo_cfg.get("variants", []) or [])])
    demo_episodes_per_variant = int(demo_cfg.get("episodes_per_variant", 1))
    demo_fps = int(demo_cfg.get("fps", 10))
    demo_rgb_hw = tuple(demo_cfg.get("rgb_hw", demo_cfg.get("rgb_size", [256, 256])))
    demo_map_hw = tuple(demo_cfg.get("map_hw", demo_cfg.get("map_size", [256, 256])))
    demo_out_dir = _PROJ_ROOT / "artifacts" / "demos" / "habitat" / ctx.run_id

    max_steps_default = int(cfg.get("max_steps", 60))

    def _make_env(*, max_episode_steps: int) -> Any:
        env = HabitatMultiAgentWrapper(
            config_path=None,
            scene_id=None,
            max_episode_steps=int(max_episode_steps),
        )
        # Ensure deterministic episode ordering across variants.
        try:
            if base_seed:
                env.env.seed(int(base_seed))
        except Exception:
            pass
        return env

    # Load the real model/pruners ONCE; variants will toggle pruning/budgets without re-loading.
    # IMPORTANT: we will create a fresh Habitat env per variant (same seed) to guarantee identical episode sets.
    env0 = _make_env(max_episode_steps=max_steps_default + 50)
    integration = HabitatERECAPIntegration(
        habitat_wrapper=env0,
        keep_ratio=float(cfg.get("keep_ratio", 0.7)),
        num_agents=int(cfg.get("num_agents", 1)),
        use_pruning=True,
        max_new_tokens=int(cfg.get("max_new_tokens", 128)),
        min_new_tokens=int(cfg.get("min_new_tokens", 10)),
    )
    if getattr(integration, "planner", None) is None:
        raise RuntimeError("E-RECAP planner is unavailable (model/pruners failed to load); cannot run real LLM mode.")

    planner = integration.planner
    _install_context_pruner(planner)
    try:
        env0.close()
    except Exception:
        pass

    variants = list(cfg.get("variants", []) or [])
    if not variants:
        raise ValueError("v2 config requires a non-empty `variants` list")

    slo_ms_default = cfg.get("slo_ms", None)
    token_budget_default = cfg.get("token_budget", None)

    for variant in variants:
        # Fresh env per variant to guarantee identical episode sets across methods (paper fairness requirement).
        # We re-seed everything to keep the env's episode iterator deterministic.
        if base_seed:
            _seed_everything(base_seed)

        vname = str(variant.get("name", "variant"))
        episodes = int(variant.get("episodes", cfg.get("episodes", cfg.get("num_episodes", 1))))
        max_steps = int(variant.get("max_steps", max_steps_default))
        env = _make_env(max_episode_steps=max_steps + 50)

        replan_interval = int(
            variant.get(
                "replan_interval_steps",
                variant.get("replan_interval", cfg.get("replan_interval_steps", cfg.get("replan_interval", 20))),
            )
        )
        min_replans = int(variant.get("min_replan_cycles", cfg.get("min_replan_cycles", 3)))
        trigger_cooldown_steps = int(variant.get("trigger_cooldown_steps", cfg.get("trigger_cooldown_steps", 0)))
        use_env_triggers = bool(variant.get("use_env_triggers", cfg.get("use_env_triggers", True)))

        num_agents = int(variant.get("num_agents", cfg.get("num_agents", 1)))
        executor = str(variant.get("executor", cfg.get("executor", "llm_patch_actions")))
        sp_noise_prob = None
        if executor == "shortest_path_noise":
            try:
                sp_noise_prob = float(variant.get("sp_noise_prob", cfg.get("sp_noise_prob", 0.1)))
            except Exception:
                sp_noise_prob = None

        brace_enabled = bool(variant.get("brace_enabled", False))
        pruning_enabled = bool(variant.get("pruning_enabled", variant.get("use_pruning", False)))
        keep_ratio = float(variant.get("keep_ratio", cfg.get("keep_ratio", 0.7)))

        context_strategy = str(variant.get("context_strategy", "erecap" if pruning_enabled else "none"))
        token_budget = variant.get("token_budget", token_budget_default)
        token_budget = int(token_budget) if isinstance(token_budget, (int, float)) and int(token_budget) > 0 else None
        summary_head_tokens = int(variant.get("summary_head_tokens", cfg.get("summary_head_tokens", 40)))
        summary_tail_tokens = int(variant.get("summary_tail_tokens", cfg.get("summary_tail_tokens", 80)))
        context_pad_tokens = int(variant.get("context_pad_tokens", cfg.get("context_pad_tokens", 0)) or 0)
        context_pad_text = ""
        context_pad_tokens_actual = 0
        if context_pad_tokens > 0:
            try:
                pad_ids = _filler_ids(planner.tokenizer, int(context_pad_tokens), token="CONTEXT_PAD")
                context_pad_text = _decode_ids(planner.tokenizer, pad_ids)
                context_pad_tokens_actual = len(_tokenize_ids(planner.tokenizer, context_pad_text))
            except Exception:
                context_pad_text = ("CONTEXT_PAD " * int(context_pad_tokens)).strip()
                context_pad_tokens_actual = int(context_pad_tokens)

        instruction_style = str(variant.get("instruction_style", cfg.get("instruction_style", "oracle"))).strip().lower()
        if instruction_style not in ("oracle", "coarsened"):
            instruction_style = "oracle"
        ambiguity_type = str(variant.get("ambiguity_type", cfg.get("ambiguity_type", "goal"))).strip().lower()
        if ambiguity_type not in ("goal", "process", "success"):
            ambiguity_type = "goal"
        clarification_budget_turns = int(
            variant.get(
                "clarification_budget_turns",
                variant.get(
                    "clarification_turns",
                    cfg.get("clarification_budget_turns", cfg.get("clarification_turns", 0)),
                ),
            )
        )
        clarification_budget_turns = max(0, clarification_budget_turns)
        clarification_overhead_ms = float(
            variant.get("clarification_overhead_ms", cfg.get("clarification_overhead_ms", 0.0))
        )
        clarification_ms_per_token = float(
            variant.get("clarification_ms_per_token", cfg.get("clarification_ms_per_token", 0.0))
        )

        vlm_cfg: Dict[str, Any] = dict(cfg.get("vlm", {}) or {})
        vlm_cfg.update(dict(variant.get("vlm", {}) or {}))
        vlm_enabled = bool(vlm_cfg.get("enabled", False))
        vlm_method = str(vlm_cfg.get("method", "none")).strip().lower()
        if not vlm_enabled:
            vlm_method = "none"

        slo_ms = variant.get("slo_ms", slo_ms_default)
        slo_ms = int(slo_ms) if isinstance(slo_ms, (int, float)) and int(slo_ms) > 0 else None

        hparams_dict: Dict[str, Any] = dict(cfg.get("brace_hparams", {}) or {})
        hparams_dict.update(dict(variant.get("brace_hparams", {}) or {}))
        if slo_ms is not None and "slo_ms" not in hparams_dict:
            hparams_dict["slo_ms"] = int(slo_ms)
        if "slo_ms" not in hparams_dict:
            hparams_dict["slo_ms"] = int(cfg.get("slo_ms", 2500))
        controller = BraceController(BraceHyperparams(**hparams_dict))

        # Re-seed + recreate the env per variant so variants iterate over (approximately) the same episode stream.
        if base_seed:
            _seed_everything(base_seed)
        try:
            env.close()
        except Exception:
            pass
        env = HabitatMultiAgentWrapper(
            config_path=None,
            scene_id=None,
            max_episode_steps=max_steps + 50,
        )
        integration.habitat_wrapper = env
        stop_distance_m = float(variant.get("goal_radius") or _extract_success_distance(env) or 0.2)
        if stop_distance_m <= 0:
            stop_distance_m = 0.2

        # Slice embodied task steps to the desired agent count (otherwise the planner will auto-expand).
        try:
            from multi_agent.task_definitions import get_task_steps  # type: ignore

            task_steps_full = get_task_steps("embodied")
            task_steps = task_steps_full[: min(num_agents, len(task_steps_full))]
        except Exception:
            task_steps = None

        ctx.append_event(
            {
                "phase": "variant_start",
                "variant": vname,
                "brace_enabled": brace_enabled,
                "pruning_enabled": pruning_enabled,
                "context_strategy": context_strategy,
                "keep_ratio": keep_ratio,
                "token_budget": token_budget,
                "slo_ms": slo_ms,
                "episodes": episodes,
                "num_agents": num_agents,
                "max_steps": max_steps,
                "replan_interval": replan_interval,
                "replan_interval_steps": int(replan_interval),
                "min_replan_cycles": min_replans,
                "trigger_cooldown_steps": trigger_cooldown_steps,
                "use_env_triggers": use_env_triggers,
                "executor": executor,
                "sp_noise_prob": sp_noise_prob,
                "context_pad_tokens": int(context_pad_tokens),
                "context_pad_tokens_actual": int(context_pad_tokens_actual),
                "instruction_style": instruction_style,
                "ambiguity_type": ambiguity_type,
                "clarification_budget_turns": clarification_budget_turns,
                "clarification_overhead_ms": clarification_overhead_ms,
                "clarification_ms_per_token": clarification_ms_per_token,
            }
        )

        record_demo = bool(demo_enabled and (not demo_variants or vname in demo_variants))

        for ep_i in range(episodes):
            st = env.reset()
            episode_id = st["episode_info"]["episode_id"]
            start_pos = tuple(st["episode_info"]["start_position"])
            goal_pos = tuple(st["episode_info"]["goal_position"])

            task_description = (
                f"Navigate from start position ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}) "
                f"to goal position ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f})."
            )

            clarif_tokens = 0
            clarif_lat_ms = 0.0
            if instruction_style == "coarsened":
                clarif = build_pointnav_instruction(
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    style="coarsened",
                    clarification_budget_turns=int(clarification_budget_turns),
                    ambiguity_type=ambiguity_type,  # type: ignore[arg-type]
                    success_distance_m=float(stop_distance_m),
                    token_counter=lambda s: _count_tokens(planner.tokenizer, s),
                )
                task_description = str(clarif.instruction)
                clarif_tokens = int(clarif.clarification_tokens)
                clarif_lat_ms = float(
                    float(clarification_overhead_ms) + float(clarification_ms_per_token) * float(clarif_tokens)
                    if clarif.clarification_transcript
                    else 0.0
                )
                if clarif.clarification_transcript:
                    lines: List[str] = []
                    for turn in clarif.transcript_dicts():
                        role = str(turn.get("role", "user")).upper()
                        text = str(turn.get("text", ""))
                        lines.append(f"{role}: {text}")
                    task_description = f"{task_description}\n\n[CLARIFICATION]\n" + "\n".join(lines)
                ctx.log_phase(
                    phase="clarification",
                    variant=vname,
                    episode_id=episode_id,
                    instruction_style=instruction_style,
                    ambiguity_type=ambiguity_type,
                    clarification_budget_turns=int(clarification_budget_turns),
                    clarification_tokens=int(clarif_tokens),
                    clarification_lat_ms=float(clarif_lat_ms),
                    tokens_in=int(clarif_tokens),
                    tokens_after_prune=int(clarif_tokens),
                    lat_total_ms=float(clarif_lat_ms),
                    clarification_transcript=clarif.transcript_dicts(),
                )
            else:
                ctx.log_phase(
                    phase="clarification",
                    variant=vname,
                    episode_id=episode_id,
                    instruction_style=instruction_style,
                    ambiguity_type=ambiguity_type,
                    clarification_budget_turns=int(clarification_budget_turns),
                    clarification_tokens=0,
                    clarification_lat_ms=0.0,
                    tokens_in=0,
                    tokens_after_prune=0,
                    lat_total_ms=0.0,
                    clarification_transcript=[],
                )

            observations = st["observations"]
            agent_state = st["agent_state"]

            # Per-episode controller/planner state.
            brace_state = BraceState()
            ep_t0 = time.time()
            replan_count = 0
            suppressed_triggers = 0
            last_replan_step = -1
            last_plan_latency_ms: Optional[float] = None
            dist_at_last_replan: Optional[float] = None
            executed_actions: List[Dict[str, str]] = []
            history_blocks: List[str] = []

            follower = None
            habitat_actions = None
            noise_prob = float(cfg.get("sp_noise_prob", 0.1))
            noise_rng = None
            if executor in ("oracle_shortest_path", "shortest_path_noise"):
                follower, habitat_actions = _try_make_shortest_path_follower(env, stop_distance_m)
                if follower is None:
                    ctx.append_event(
                        {
                            "phase": "executor_init_failed",
                            "variant": vname,
                            "episode_id": episode_id,
                            "executor": executor,
                        }
                    )
                if executor == "shortest_path_noise":
                    noise_prob = float(variant.get("sp_noise_prob", cfg.get("sp_noise_prob", 0.1)))
                    try:
                        import hashlib

                        h = hashlib.sha1(f"{episode_id}|{vname}".encode("utf-8")).hexdigest()[:8]
                        noise_rng = random.Random(int(base_seed) + int(h, 16))
                    except Exception:
                        noise_rng = random.Random(int(base_seed))

            demo_writer = None
            demo_frame_wh = None
            demo_path = None
            if record_demo and ep_i < demo_episodes_per_variant:
                try:
                    demo_out_dir.mkdir(parents=True, exist_ok=True)
                    demo_path = demo_out_dir / f"{vname}__ep{episode_id}.mp4"
                    demo_frame_wh = (
                        int(demo_rgb_hw[1]) + int(demo_map_hw[1]),
                        int(max(int(demo_rgb_hw[0]), int(demo_map_hw[0]))),
                    )
                    demo_writer = _demo_open_writer(demo_path, fps=demo_fps, frame_wh=demo_frame_wh)
                    ctx.append_event(
                        {
                            "phase": "demo_start",
                            "variant": vname,
                            "episode_id": episode_id,
                            "path": str(demo_path),
                            "fps": int(demo_fps),
                            "frame_wh": list(demo_frame_wh),
                            "executor": executor,
                        }
                    )
                except Exception:
                    demo_writer = None

            last_replan_tokens_after: Optional[int] = None
            last_replan_lat_ms: Optional[float] = None
            last_replan_slo_over_ms: Optional[float] = None

            for t in range(max_steps):
                cur_dist = agent_state.get("distance_to_goal")
                if dist_at_last_replan is None and isinstance(cur_dist, (int, float)):
                    dist_at_last_replan = float(cur_dist)

                periodic_trigger = rs.periodic_trigger(
                    t=t, interval_steps=int(replan_interval), last_replan_step=int(last_replan_step)
                )
                triggers = env.check_replanning_trigger() if use_env_triggers else []
                allow_trigger = rs.allow_trigger(
                    t=t,
                    last_replan_step=int(last_replan_step),
                    trigger_cooldown_steps=int(trigger_cooldown_steps),
                )

                any_trigger = bool(periodic_trigger) or bool(triggers)
                if any_trigger and not allow_trigger:
                    suppressed_triggers += 1

                do_replan = bool(any_trigger and allow_trigger)
                if do_replan:
                    replan_count += 1
                    last_replan_step = t

                    raw_types: List[str] = []
                    for tr in triggers or []:
                        if isinstance(tr, dict) and tr.get("type") is not None:
                            raw_types.append(str(tr.get("type")))
                        else:
                            raw_types.append(str(tr))
                    deadlock_trigger = any(t == "stuck" for t in raw_types)
                    replan_trigger_type = rs.trigger_type_primary(
                        periodic=bool(periodic_trigger),
                        failure=False,
                        deadlock=bool(deadlock_trigger),
                    )

                    progress_since_last = None
                    if isinstance(cur_dist, (int, float)) and dist_at_last_replan is not None:
                        progress_since_last = float(dist_at_last_replan) - float(cur_dist)
                    churn = any(str(tr.get("type")) == "stuck" for tr in (triggers or []))

                    telemetry = {
                        "clarification_budget_turns": int(clarification_budget_turns),
                        "churn": bool(churn),
                        "progress": progress_since_last,
                        "lat_total_ms": last_plan_latency_ms,
                    }
                    trigger_dict = rs.build_trigger_dict(
                        periodic=bool(periodic_trigger),
                        failure=False,
                        deadlock=bool(deadlock_trigger),
                        unsafe=False,
                        extra_types=raw_types,
                    )

                    if brace_enabled:
                        decision, brace_state = controller.step(
                            state=brace_state,
                            trigger=trigger_dict,
                            telemetry=telemetry,
                            remaining_budget=token_budget,
                            num_agents=num_agents,
                        )
                        mode = decision.mode
                        eff_budget = decision.token_budget
                        hazards = {
                            "hazard_slo": bool(decision.hazard_slo),
                            "hazard_churn": bool(decision.hazard_churn),
                            "hazard_deadlock": bool(decision.hazard_deadlock),
                            "hazard_unsafe": bool(decision.hazard_unsafe),
                            "cooldown_active": bool(decision.cooldown_active),
                            "rollback_flag": bool(decision.rollback_flag),
                        }
                    else:
                        mode = "partial_replan"
                        eff_budget = token_budget
                        hazards = {
                            "hazard_slo": False,
                            "hazard_churn": False,
                            "hazard_deadlock": False,
                            "hazard_unsafe": False,
                            "cooldown_active": False,
                            "rollback_flag": False,
                        }

                    planner_called = mode in ("full_replan", "partial_replan")

                    tokens_in = 0
                    tokens_after = 0
                    lat_total_ms = 0.0
                    ctx_before_chars = None
                    ctx_after_chars = None
                    lat_prune_ms = None
                    lat_summary_ms = None
                    lat_infer_ms = None

                    if planner_called:
                        integration.keep_ratio = keep_ratio
                        planner.keep_ratio = keep_ratio
                        planner._brace_context_strategy = context_strategy
                        planner._brace_token_budget = eff_budget
                        planner._brace_summary_head_tokens = int(summary_head_tokens)
                        planner._brace_summary_tail_tokens = int(summary_tail_tokens)
                        try:
                            import hashlib

                            h = hashlib.sha1(str(episode_id).encode("utf-8")).hexdigest()[:8]
                            planner._brace_random_seed = int(base_seed) + int(replan_count) * 1000 + int(h, 16)
                        except Exception:
                            planner._brace_random_seed = int(base_seed) + int(replan_count) * 1000

                        # Track-S (optional): produce an extra state summary block and log it as a separate phase.
                        obs_text = integration.observation_to_text(observations, agent_state, st["episode_info"])
                        if vlm_method != "none":
                            t0_vlm = time.perf_counter()
                            if vlm_method in ("heuristic_rgb_stats", "rgb_stats"):
                                vlm_text = _vlm_heuristic_rgb_stats_text(observations)
                            else:
                                vlm_text = "(vlm_disabled_or_unknown_method)"
                            lat_vlm_ms = (time.perf_counter() - t0_vlm) * 1000.0
                            try:
                                vlm_tokens_out = int(_count_tokens(planner.tokenizer, str(vlm_text)))
                            except Exception:
                                vlm_tokens_out = 0
                            ctx.log_phase(
                                phase="vlm_summarize",
                                variant=vname,
                                episode_id=episode_id,
                                t=t,
                                replan_cycle=replan_count,
                                vlm_model=str(vlm_method),
                                vlm_tokens_in=0,
                                vlm_tokens_out=int(vlm_tokens_out),
                                lat_total_ms=float(lat_vlm_ms),
                                lat_vlm_ms=float(lat_vlm_ms),
                            )
                            obs_text = f"{obs_text}\n\n[VLM_SUMMARY]\n{vlm_text}\n"
                        hist_text = "\n".join(history_blocks) if history_blocks else "(none)"
                        pad_block = ""
                        if context_pad_text:
                            # Insert padding early so recency-based methods naturally prefer newer information.
                            pad_block = f"\n\n[CONTEXT_PAD]\n{context_pad_text}\n"
                        full_task_description = (
                            f"{task_description}{pad_block}\n\n[REPLAN_HISTORY]\n{hist_text}\n\n[CURRENT_STATE]\n{obs_text}\n"
                        )

                        try:
                            import torch  # type: ignore

                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            planning_result = planner.run_planning_cycle(
                                task_description=full_task_description,
                                task_steps=task_steps,
                                task_type="embodied",
                                use_pruning=bool(pruning_enabled),
                            )
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            lat_total_ms = (time.perf_counter() - t0) * 1000.0
                        except Exception:
                            # If torch isn't available for sync, still run best-effort.
                            t0 = time.perf_counter()
                            planning_result = planner.run_planning_cycle(
                                task_description=full_task_description,
                                task_steps=task_steps,
                                task_type="embodied",
                                use_pruning=bool(pruning_enabled),
                            )
                            lat_total_ms = (time.perf_counter() - t0) * 1000.0

                        last_plan_latency_ms = float(lat_total_ms)
                        if isinstance(cur_dist, (int, float)):
                            dist_at_last_replan = float(cur_dist)

                        ph = planning_result.get("planning_history", []) if isinstance(planning_result, dict) else []
                        if ph:
                            ctx_before_chars = ph[0].get("context_length_before")
                            ctx_after_chars = ph[0].get("context_length_after")
                            try:
                                lat_prune_ms = float(ph[0].get("pruning_time", 0.0)) * 1000.0
                            except Exception:
                                lat_prune_ms = None
                            try:
                                lat_infer_ms = float(ph[0].get("inference_time", 0.0)) * 1000.0
                            except Exception:
                                lat_infer_ms = None
                            gen_prompt_tokens = None
                            gen_generated_tokens = None
                            gen_total_tokens = None
                            gen_time_ms = None
                            gen_max_new_tokens = None
                            try:
                                if ph[0].get("prompt_tokens") is not None:
                                    gen_prompt_tokens = int(ph[0].get("prompt_tokens"))
                                if ph[0].get("generated_tokens") is not None:
                                    gen_generated_tokens = int(ph[0].get("generated_tokens"))
                                if ph[0].get("total_tokens") is not None:
                                    gen_total_tokens = int(ph[0].get("total_tokens"))
                                if ph[0].get("gen_time_ms") is not None:
                                    gen_time_ms = float(ph[0].get("gen_time_ms"))
                                if ph[0].get("max_new_tokens") is not None:
                                    gen_max_new_tokens = int(ph[0].get("max_new_tokens"))
                            except Exception:
                                gen_prompt_tokens = None
                                gen_generated_tokens = None
                                gen_total_tokens = None
                                gen_time_ms = None
                                gen_max_new_tokens = None
                            ps = ph[0].get("pruning_stats", {}) or {}
                            try:
                                tokens_in = int(ps.get("input_tokens", 0) or 0)
                                tokens_after = int(ps.get("output_tokens", 0) or 0)
                            except Exception:
                                tokens_in, tokens_after = 0, 0
                            try:
                                summary_time_s = ps.get("summary_time_s")
                                if summary_time_s is None and isinstance(ps.get("budget_cap"), dict):
                                    summary_time_s = (ps.get("budget_cap") or {}).get("summary_time_s")
                                if summary_time_s is not None:
                                    lat_summary_ms = float(summary_time_s) * 1000.0
                            except Exception:
                                lat_summary_ms = None

                        # Extract plan patches and convert to actions.
                        plan_patches: List[str] = []
                        for contrib in planner.context_buffer.agent_contributions:
                            plan_patches.extend(list(contrib.plan_patches or []))
                        if executor not in ("oracle_shortest_path", "shortest_path_noise"):
                            if executor == "llm_patch_actions_strict":
                                actions = integration.plan_to_actions(
                                    plan_patches,
                                    agent_state,
                                    allow_pointgoal_fallback=False,
                                    fallback_action="stop",
                                )
                            else:
                                actions = integration.plan_to_actions(plan_patches, agent_state)
                            if actions:
                                executed_actions.extend(actions)
                        patch_snippets = "; ".join([p.strip().replace("\n", " ")[:80] for p in plan_patches[:2]])
                        history_blocks.append(
                            f"- replan={replan_count} t={t} trigger={replan_trigger_type} "
                            f"mode={mode} tokens={tokens_after} lat_ms={lat_total_ms:.1f} "
                            f"patches={patch_snippets}"
                        )
                    else:
                        # BRACE chose to defer/reuse: treat as zero-cost replanning decision.
                        planner._brace_context_strategy = context_strategy
                        planner._brace_token_budget = eff_budget
                        fallback = _fallback_action(agent_state, stop_distance_m=stop_distance_m)
                        executed_actions.append(fallback)

                    ctx.append_event(
                        {
                            "domain": "habitat",
                            "variant": vname,
                            "episode_id": episode_id,
                            "t": t,
                            "brace_enabled": brace_enabled,
                            "pruning_enabled": pruning_enabled,
                            "mode": mode,
                            "token_budget": eff_budget if eff_budget is not None else 0,
                            "keep_ratio": keep_ratio,
                            "context_strategy": context_strategy,
                            "replan_cycle": replan_count,
                            "replan_trigger_type": replan_trigger_type,
                            "replan_interval_steps": int(replan_interval),
                            "trigger_cooldown_steps": int(trigger_cooldown_steps),
                            "trigger": trigger_dict,
                            "instruction_style": instruction_style,
                            "ambiguity_type": ambiguity_type,
                            "clarification_budget_turns": int(clarification_budget_turns),
                            "clarification_tokens": int(clarif_tokens),
                            "clarification_lat_ms": float(clarif_lat_ms),
                            "tokens_in": tokens_in,
                            "tokens_after_prune": tokens_after,
                            "gen_prompt_tokens": gen_prompt_tokens,
                            "gen_generated_tokens": gen_generated_tokens,
                            "gen_total_tokens": gen_total_tokens,
                            "gen_time_ms": gen_time_ms,
                            "gen_max_new_tokens": gen_max_new_tokens,
                            "lat_total_ms": float(lat_total_ms),
                            "lat_prune_ms": lat_prune_ms,
                            "lat_summary_ms": lat_summary_ms,
                            "lat_infer_ms": lat_infer_ms,
                            "context_length_before_chars": ctx_before_chars,
                            "context_length_after_chars": ctx_after_chars,
                            **hazards,
                        }
                    )

                    last_replan_tokens_after = int(tokens_after) if isinstance(tokens_after, int) else None
                    last_replan_lat_ms = float(lat_total_ms) if isinstance(lat_total_ms, (int, float)) else None
                    if slo_ms is not None and last_replan_lat_ms is not None:
                        last_replan_slo_over_ms = max(0.0, float(last_replan_lat_ms) - float(slo_ms))
                    else:
                        last_replan_slo_over_ms = None

                action = None
                action_str = None
                if executor in ("oracle_shortest_path", "shortest_path_noise") and follower is not None:
                    try:
                        action = follower.get_next_action(goal_pos)
                    except Exception:
                        action = None
                    if (
                        executor == "shortest_path_noise"
                        and noise_rng is not None
                        and habitat_actions is not None
                        and action is not None
                        and action != habitat_actions.stop
                    ):
                        try:
                            if float(noise_prob) > 0.0 and noise_rng.random() < float(noise_prob):
                                action = noise_rng.choice(
                                    [
                                        habitat_actions.move_forward,
                                        habitat_actions.turn_left,
                                        habitat_actions.turn_right,
                                    ]
                                )
                        except Exception:
                            pass
                    # Keep episodes long enough to collect replanning tails.
                    if habitat_actions is not None and action == habitat_actions.stop and replan_count < min_replans:
                        action = habitat_actions.turn_left

                if action is None:
                    if executed_actions:
                        action = _sanitize_action(
                            executed_actions.pop(0), agent_state, stop_distance_m=stop_distance_m
                        )
                    else:
                        action = _fallback_action(agent_state, stop_distance_m=stop_distance_m)

                if isinstance(action, dict):
                    action_str = str(action.get("action"))
                else:
                    action_str = str(action)

                if demo_writer is not None:
                    try:
                        overlay = [
                            f"variant={vname}",
                            f"ep={episode_id} t={t} dist={_goal_distance_m(agent_state):.2f}",
                            f"replans={replan_count} exec={executor}",
                            f"action={action_str}",
                        ]
                        if last_replan_tokens_after is not None and last_replan_lat_ms is not None:
                            overlay.append(f"last_replan: tok={last_replan_tokens_after} lat={last_replan_lat_ms:.0f}ms")
                            if last_replan_slo_over_ms is not None:
                                if float(last_replan_slo_over_ms) > 0.0:
                                    overlay.append(f"SLO_VIOLATION +{float(last_replan_slo_over_ms):.0f}ms")
                                else:
                                    overlay.append("SLO_OK")
                        frame = _demo_render_frame(
                            env,
                            observations,
                            rgb_hw=(int(demo_rgb_hw[0]), int(demo_rgb_hw[1])),
                            map_hw=(int(demo_map_hw[0]), int(demo_map_hw[1])),
                            overlay_lines=overlay,
                        )
                        demo_writer.write(frame)
                    except Exception:
                        pass

                observations, reward, done, info = env.step(action)
                agent_state = info.get("agent_state", agent_state)
                if done:
                    break

            if demo_writer is not None:
                try:
                    demo_writer.release()
                    if demo_path is not None:
                        ctx.append_event(
                            {
                                "phase": "demo_end",
                                "variant": vname,
                                "episode_id": episode_id,
                                "path": str(demo_path),
                                "frame_wh": list(demo_frame_wh) if demo_frame_wh is not None else None,
                            }
                        )
                except Exception:
                    pass

            metrics = env.env.get_metrics()
            ep_wall_ms = (time.time() - ep_t0) * 1000.0
            ctx.append_episode(
                {
                    "domain": "habitat",
                    "variant": vname,
                    "episode_id": episode_id,
                    "success": float(bool(metrics.get("success", False))),
                    "spl": float(metrics.get("spl", 0.0)),
                    "step_count": float(env.step_count),
                    "replan_cycles": float(replan_count),
                    "effective_replans_per_episode": float(replan_count),
                    "suppressed_triggers": float(suppressed_triggers),
                    "replan_interval_steps": float(replan_interval),
                    "replan_interval": float(replan_interval),
                    "trigger_cooldown_steps": float(trigger_cooldown_steps),
                    "episode_wall_time_ms": float(ep_wall_ms),
                    "effective_replans_per_min": float(replan_count) / (float(ep_wall_ms) / 60000.0)
                    if ep_wall_ms > 0
                    else None,
                    "executor": executor,
                    "instruction_style": instruction_style,
                    "ambiguity_type": ambiguity_type,
                    "clarification_budget_turns": float(clarification_budget_turns),
                    "clarification_tokens": float(clarif_tokens),
                    "clarification_lat_ms": float(clarif_lat_ms),
                }
            )

        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke/habitat_setup.json")
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--run_name", default="habitat_setup_smoke")
    ap.add_argument("--habitat_setup_root", default="habitat-setup")
    args = ap.parse_args()

    cfg = _load_json(Path(args.config))
    cfg.setdefault("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES"))
    ctx = RunContext.create(args.runs_root, args.run_name, cfg)
    _append_runs_index(
        args.runs_root,
        {
            "run_id": ctx.run_id,
            "run_dir": ctx.run_dir,
            "run_name": args.run_name,
            "runner": "experiments/habitat/run_habitat_setup_real.py",
            "config_path": str(args.config),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cmd": " ".join([sys.executable, *sys.argv]),
        },
    )

    # v2: in-process runner with variants (loads model once; supports BRACE + baselines).
    if cfg.get("variants"):
        _run_v2(cfg, ctx, Path(args.habitat_setup_root))
        ctx.write_summary({"note": "habitat real runner v2 (in-process, model loaded once)"})
        print(ctx.run_dir)
        return

    habitat_root = Path(args.habitat_setup_root)
    script_path = habitat_root / "scripts" / "run_real_erecap_experiment.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing habitat-setup script: {script_path}")

    env = _env_without_proxies()
    env["BRACE_HABITAT_PY"] = _pick_habitat_python()
    ctx.append_event({"phase": "selected_habitat_python", "python": env["BRACE_HABITAT_PY"]})

    all_episode_rows: List[Dict[str, Any]] = []

    for variant in cfg.get("configs", []):
        name = variant["name"]
        out_dir = Path(ctx.run_dir) / "habitat_setup" / name
        out_dir.mkdir(parents=True, exist_ok=True)

        args_list = [
            "--num_episodes",
            str(cfg["num_episodes"]),
            "--num_agents",
            str(cfg["num_agents"]),
            "--keep_ratio",
            str(variant["keep_ratio"]),
            *(["--use_pruning"] if variant.get("use_pruning") else []),
            "--max_steps",
            str(cfg["max_steps"]),
            "--replan_interval",
            str(cfg["replan_interval"]),
            "--min_replan_cycles",
            str(cfg["min_replan_cycles"]),
            "--output_dir",
            str(out_dir),
        ]
        if cfg.get("random_seed") is not None:
            args_list.extend(["--random_seed", str(int(cfg.get("random_seed", 0)))])
        if variant.get("context_compress_method") is not None:
            args_list.extend(["--context_compress_method", str(variant.get("context_compress_method"))])
        if variant.get("token_budget") is not None:
            args_list.extend(["--token_budget", str(int(variant.get("token_budget") or 0))])
        if variant.get("summary_head_tokens") is not None:
            args_list.extend(["--summary_head_tokens", str(int(variant.get("summary_head_tokens") or 0))])
        if variant.get("summary_tail_tokens") is not None:
            args_list.extend(["--summary_tail_tokens", str(int(variant.get("summary_tail_tokens") or 0))])

        rc = _run_habitat_setup(script_path, args_list, env)

        ctx.append_event(
            {
                "domain": "habitat",
                "phase": "habitat_setup_completed",
                "variant": name,
                "returncode": rc,
                "output_dir": str(out_dir),
            }
        )

        summary_path = out_dir / "experiment_summary.json"
        if not summary_path.exists():
            ctx.append_event(
                {
                    "domain": "habitat",
                    "phase": "missing_summary",
                    "variant": name,
                    "path": str(summary_path),
                }
            )
            continue

        summary = _load_json(summary_path)
        for ep in summary.get("results", []):
            # habitat-setup uses character lengths for context; log them as such.
            metrics = ep.get("metrics", {})
            row = {
                "domain": "habitat",
                "source": "habitat-setup",
                "variant": name,
                "episode_id": ep.get("episode_id"),
                "success": float(metrics.get("success", ep.get("success", 0.0))),
                "spl": float(metrics.get("spl", 0.0)),
                "step_count": int(ep.get("step_count", 0)),
                "replan_cycles": int(ep.get("replan_cycles", 0)),
            }
            all_episode_rows.append(row)
            ctx.append_episode(row)

            # Per-replan context stats (token-based + latency breakdown).
            for h in ep.get("context_history", []) or []:
                ps = h.get("pruning_stats", {}) or {}
                lat_prune_ms = None
                lat_summary_ms = None
                lat_infer_ms = None
                lat_total_ms = None
                try:
                    if ps.get("pruning_time_s") is not None:
                        lat_prune_ms = float(ps.get("pruning_time_s")) * 1000.0
                    if ps.get("summary_time_s") is not None:
                        lat_summary_ms = float(ps.get("summary_time_s")) * 1000.0
                    if ps.get("inference_time_s") is not None:
                        lat_infer_ms = float(ps.get("inference_time_s")) * 1000.0
                    if lat_prune_ms is not None and lat_infer_ms is not None:
                        lat_total_ms = lat_prune_ms + lat_infer_ms
                except Exception:
                    pass
                ctx.append_event(
                    {
                        "domain": "habitat",
                        "source": "habitat-setup",
                        "variant": name,
                        "episode_id": ep.get("episode_id"),
                        "replan_cycle": h.get("replan_cycle"),
                        "step": h.get("step"),
                        "context_length_chars": h.get("context_length"),
                        "context_length_before_chars": ps.get("context_length_before"),
                        "context_length_after_chars": ps.get("context_length_after"),
                        "context_compress_method": ps.get("context_compress_method"),
                        "tokens_in": ps.get("tokens_before"),
                        "tokens_after_prune": ps.get("tokens_after"),
                        "lat_prune_ms": lat_prune_ms,
                        "lat_summary_ms": lat_summary_ms,
                        "lat_infer_ms": lat_infer_ms,
                        "lat_total_ms": lat_total_ms,
                    }
                )

    ctx.write_summary(
        {
            "episodes": all_episode_rows,
            "note": "This runner wraps habitat-setup real_experiment scripts; token counts are NOT yet standardized (context_length is char-based in habitat-setup).",
        }
    )

    print(ctx.run_dir)


if __name__ == "__main__":
    main()
