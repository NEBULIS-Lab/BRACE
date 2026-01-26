from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from brace.controller import BraceController, BraceHyperparams, BraceState
from experiments.common.context_compress import budget_target_history_keep, select_history_tokens
from experiments.common.clarification import build_pointnav_instruction
from experiments.common.logging import RunContext
from experiments.common import replan_schedule as rs


def _approx_tokens(text: str) -> int:
    # Deterministic, dependency-free token proxy.
    return max(1, len(text.split()))


def _serialize_context(
    *,
    task: str,
    agent_state: Dict[str, Any],
    history: List[Dict[str, Any]],
    protected_blocks: Tuple[str, ...],
    token_budget: Optional[int],
    pruning_enabled: bool,
    keep_ratio: float,
    compress_strategy: Optional[str] = None,
    summary_head_tokens: int = 40,
    summary_tail_tokens: int = 80,
    rng: Optional[random.Random] = None,
) -> Tuple[str, Dict[str, Any]]:
    # Typed blocks A-F (minimal)
    blocks: Dict[str, str] = {}
    blocks["A"] = f"[A] TASK\n{task}\n"
    blocks["B"] = (
        "[B] STATE\n"
        f"distance_to_goal={agent_state.get('distance_to_goal')}\n"
        f"pointgoal={agent_state.get('pointgoal')}\n"
    )
    blocks["C"] = "[C] CONSTRAINTS\nnone\n"
    blocks["D"] = "[D] COORDINATION\nnone\n"

    # History: last N steps
    recent = history[-20:]
    hist_lines = []
    for h in recent:
        hist_lines.append(
            f"step={h.get('step')} action={h.get('action')} dist={h.get('distance_to_goal')}"
        )
    blocks["E"] = "[E] HISTORY\n" + "\n".join(hist_lines) + "\n"

    protected_text = "".join(blocks[k] for k in ["A", "B", "C", "D"] if k in blocks)
    prunable_text = blocks["E"]

    tokens_protected = _approx_tokens(protected_text)
    tokens_prunable = _approx_tokens(prunable_text)

    tokens_in = tokens_protected + tokens_prunable

    tokens_after = tokens_in
    pruned_prunable = prunable_text

    compress_strategy_norm = None
    if compress_strategy is not None and str(compress_strategy).strip():
        compress_strategy_norm = str(compress_strategy).strip().lower()

    target_prunable_keep: Optional[int] = max(1, int(tokens_prunable * keep_ratio)) if pruning_enabled else None
    target_prunable_budget: Optional[int] = (
        max(1, token_budget - tokens_protected) if pruning_enabled and token_budget is not None else None
    )
    summary_tokens = 0

    if pruning_enabled:
        if compress_strategy_norm is not None and compress_strategy_norm not in (
            "erecap_like",
            "keep_ratio",
            "truncate",
        ):
            # Baseline token-budget matched strategies (random / recency / structured_summary).
            if rng is None:
                rng = random.Random(0)

            header_text = "[E] HISTORY\n"
            body_text = "\n".join(hist_lines) + "\n"
            header_tokens = _approx_tokens(header_text)

            if token_budget is not None and token_budget > 0 and token_budget < (tokens_protected + header_tokens):
                raise ValueError(
                    f"token_budget ({token_budget}) < protected+history_header ({tokens_protected + header_tokens}); "
                    "cannot satisfy protected blocks for budget-matched baselines."
                )

            target_prunable_total = budget_target_history_keep(
                tokens_protected=tokens_protected,
                tokens_in=tokens_in,
                token_budget=token_budget,
            )
            target_keep_body = max(0, int(target_prunable_total) - int(header_tokens))

            body_tokens = body_text.split()
            selection = select_history_tokens(
                strategy=compress_strategy_norm,
                history_len=len(body_tokens),
                target_keep=target_keep_body,
                rng=rng,
                summary_head_tokens=int(summary_head_tokens),
                summary_tail_tokens=int(summary_tail_tokens),
            )
            summary_tokens = int(selection.summary_tokens)

            pruned_body_tokens: List[str]
            if (
                compress_strategy_norm in ("structured_summary", "summary", "head_tail_summary")
                and selection.meta.get("head_kept") is not None
            ):
                head = int(selection.meta.get("head_kept") or 0)
                tail = int(selection.meta.get("tail_kept") or 0)
                head_tokens_list = body_tokens[:head] if head > 0 else []
                tail_tokens_list = body_tokens[-tail:] if tail > 0 else []
                summary_tokens_list = ["<SUM>"] * int(selection.summary_tokens)
                pruned_body_tokens = head_tokens_list + summary_tokens_list + tail_tokens_list
            else:
                pruned_body_tokens = [
                    body_tokens[i] for i in selection.kept_indices if 0 <= i < len(body_tokens)
                ]

            pruned_prunable = header_text + " ".join(pruned_body_tokens) + "\n"
            tokens_after = tokens_protected + _approx_tokens(pruned_prunable)
            target_prunable_keep = None
            target_prunable_budget = int(target_prunable_total)
        else:
            # Keep-ratio is a process knob on the prunable region; token_budget is an absolute
            # cap (if provided). Apply keep-ratio first, then enforce the budget.
            keep_ratio = max(0.0, min(1.0, keep_ratio))
            target_prunable_keep = max(1, int(tokens_prunable * keep_ratio))
            target_prunable_budget = (
                max(1, token_budget - tokens_protected) if token_budget is not None else None
            )
            target_prunable = (
                min(target_prunable_keep, target_prunable_budget)
                if target_prunable_budget is not None
                else target_prunable_keep
            )
            # Approximate pruning by truncating lines.
            lines = prunable_text.splitlines()
            kept_lines = []
            for ln in lines:
                if _approx_tokens("\n".join(kept_lines + [ln])) > target_prunable:
                    break
                kept_lines.append(ln)
            pruned_prunable = "\n".join(kept_lines) + "\n"
            tokens_after = tokens_protected + _approx_tokens(pruned_prunable)

    ctx = protected_text + pruned_prunable
    stats = {
        "tokens_in": tokens_in,
        "tokens_after": tokens_after,
        "tokens_protected": tokens_protected,
        "tokens_prunable": tokens_prunable,
        "target_prunable_keep": target_prunable_keep,
        "target_prunable_budget": target_prunable_budget,
        "compress_strategy": compress_strategy_norm,
        "summary_tokens": summary_tokens,
        "protected_blocks": protected_blocks,
        "token_budget": token_budget,
    }
    return ctx, stats


def _policy_action(
    agent_state: Dict[str, Any], observations: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    # Try to STOP when near goal; Habitat PointNav success typically requires STOP.
    try:
        if agent_state.get("distance_to_goal") is not None and float(agent_state["distance_to_goal"]) <= 0.2:
            return {"action": "stop"}
    except Exception:
        pass

    pointgoal = None
    if observations and isinstance(observations, dict):
        pointgoal = observations.get("pointgoal_with_gps_compass")
    if pointgoal is None:
        pointgoal = agent_state.get("pointgoal")

    if pointgoal is not None and len(pointgoal) >= 2:
        try:
            goal_angle = float(pointgoal[1])
        except Exception:
            goal_angle = None
        if goal_angle is not None:
            if abs(goal_angle) > 0.3:
                return {"action": "turn_left" if goal_angle > 0 else "turn_right"}
            return {"action": "move_forward"}

    return {"action": "move_forward"}


def _load_habitat_wrapper():
    # Import habitat-setup wrapper (requires running inside conda env habitat).
    import sys

    proj_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(proj_root / "habitat-setup" / "src"))
    from habitat_multi_agent_wrapper import HabitatMultiAgentWrapper

    return HabitatMultiAgentWrapper


def _expand_variants(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants = base_cfg.get("variants", [])
    if variants:
        return variants

    grid = base_cfg.get("variant_grid")
    if not grid:
        return []

    brace_opts = list(grid.get("brace_enabled", [False, True]))
    prune_opts = list(grid.get("pruning_enabled", [False, True]))
    keep_ratios = list(grid.get("keep_ratio", [1.0]))
    clarification_turns = list(grid.get("clarification_turns", [0]))
    replan_intervals = list(
        grid.get(
            "replan_interval_steps",
            grid.get("replan_interval", [base_cfg.get("replan_interval_steps", base_cfg.get("replan_interval", 20))]),
        )
    )

    token_budget_opts = grid.get("token_budget")
    if token_budget_opts is None:
        token_budget_opts = [base_cfg.get("token_budget", 0)]
    token_budget_opts = list(token_budget_opts)

    out: List[Dict[str, Any]] = []
    for brace_enabled in brace_opts:
        for pruning_enabled in prune_opts:
            for ct in clarification_turns:
                for replan_interval in replan_intervals:
                    for token_budget in token_budget_opts:
                        if pruning_enabled:
                            kr_list = keep_ratios
                        else:
                            kr_list = [1.0]
                        for keep_ratio in kr_list:
                            name = (
                                f"{'brace' if brace_enabled else 'nobrace'}_"
                                f"{'prune' if pruning_enabled else 'noprune'}"
                                f"__r{keep_ratio}"
                                f"__cl{ct}"
                                f"__int{replan_interval}"
                                f"__B{token_budget}"
                            )
                            out.append(
                                {
                                    "name": name,
                                    "brace_enabled": bool(brace_enabled),
                                    "pruning_enabled": bool(pruning_enabled),
                                    "keep_ratio": float(keep_ratio),
                                    "clarification_turns": int(ct),
                                    "replan_interval": int(replan_interval),
                                    "token_budget": int(token_budget),
                                }
                            )
    return out


def run_variant(ctx: RunContext, env: Any, variant: Dict[str, Any], base_cfg: Dict[str, Any]) -> None:
    rng_base = int(base_cfg.get("random_seed", 0)) + abs(hash(str(variant.get("name", "")))) % (2**31)

    hparams_dict: Dict[str, Any] = dict(base_cfg.get("brace_hparams", {}) or {})
    hparams_dict.update(dict(variant.get("brace_hparams", {}) or {}))
    hparams = (
        BraceHyperparams(slo_ms=int(base_cfg.get("slo_ms", 250)), **hparams_dict)
        if hparams_dict
        else BraceHyperparams(slo_ms=int(base_cfg.get("slo_ms", 250)))
    )
    controller = BraceController(hparams)

    episodes = int(variant.get("episodes", base_cfg.get("episodes", 1)))
    max_steps = int(variant.get("max_steps", base_cfg.get("max_steps", 200)))
    replan_interval = int(
        variant.get(
            "replan_interval_steps",
            variant.get("replan_interval", base_cfg.get("replan_interval_steps", base_cfg.get("replan_interval", 20))),
        )
    )
    min_replans = int(variant.get("min_replans", base_cfg.get("min_replans", 3)))
    trigger_cooldown_steps = int(
        variant.get("trigger_cooldown_steps", base_cfg.get("trigger_cooldown_steps", 0))
    )

    token_budget = int(variant.get("token_budget", base_cfg.get("token_budget", 0)))
    token_budget = token_budget if token_budget > 0 else None

    brace_enabled = bool(variant.get("brace_enabled", False))
    pruning_enabled = bool(variant.get("pruning_enabled", False))
    keep_ratio = float(variant.get("keep_ratio", 1.0))
    env_triggers_enabled = bool(variant.get("env_triggers_enabled", base_cfg.get("env_triggers_enabled", True)))
    compress_strategy = variant.get("compress_strategy", base_cfg.get("compress_strategy", None))
    summary_head_tokens = int(variant.get("summary_head_tokens", base_cfg.get("summary_head_tokens", 40)))
    summary_tail_tokens = int(variant.get("summary_tail_tokens", base_cfg.get("summary_tail_tokens", 80)))
    clarification_budget_turns = int(
        variant.get(
            "clarification_budget_turns",
            variant.get("clarification_turns", base_cfg.get("clarification_turns", 0)),
        )
    )
    instruction_style = str(variant.get("instruction_style", base_cfg.get("instruction_style", "coarsened")))
    if instruction_style not in ("oracle", "coarsened"):
        instruction_style = "coarsened"
    ambiguity_type = str(variant.get("ambiguity_type", base_cfg.get("ambiguity_type", "goal")))
    if ambiguity_type not in ("goal", "process", "success"):
        ambiguity_type = "goal"
    success_distance_m = float(variant.get("success_distance_m", base_cfg.get("success_distance_m", 0.2)))
    clarif_overhead_ms = float(
        variant.get("clarification_overhead_ms", base_cfg.get("clarification_overhead_ms", 0.0))
    )
    clarif_ms_per_token = float(
        variant.get("clarification_ms_per_token", base_cfg.get("clarification_ms_per_token", 0.0))
    )
    slo_ms = int(base_cfg.get("slo_ms", 0))

    for ep_i in range(episodes):
        st = env.reset()
        last_obs = st.get("observations")
        episode_id = st["episode_info"]["episode_id"]
        ep_rng = random.Random(rng_base + abs(hash(str(episode_id))) % (2**31))
        start_pos = tuple(st["episode_info"]["start_position"])
        goal_pos = tuple(st["episode_info"]["goal_position"])

        clarif = build_pointnav_instruction(
            start_pos=start_pos,
            goal_pos=goal_pos,
            style=instruction_style,  # type: ignore[arg-type]
            clarification_budget_turns=clarification_budget_turns,
            ambiguity_type=ambiguity_type,  # type: ignore[arg-type]
            success_distance_m=success_distance_m,
            token_counter=_approx_tokens,
        )
        clarif_lat_ms = 0.0
        if clarif.clarification_transcript:
            clarif_lat_ms = float(
                clarif_overhead_ms + clarif_ms_per_token * float(clarif.clarification_tokens)
            )
        task = clarif.instruction
        clarif_transcript = clarif.transcript_dicts()
        clarif_tokens = int(clarif.clarification_tokens)

        state = BraceState()
        history: List[Dict[str, Any]] = []
        replan_count = 0
        dist_at_last_replan: Optional[float] = None
        last_plan_latency_ms: Optional[float] = None
        suppressed_triggers = 0
        ep_t0 = time.time()

        # Main loop
        last_replan_step = -1
        for t in range(max_steps):
            agent_state = env.get_agent_state()
            cur_dist = agent_state.get("distance_to_goal")
            if dist_at_last_replan is None and isinstance(cur_dist, (int, float)):
                dist_at_last_replan = float(cur_dist)

            periodic_trigger = rs.periodic_trigger(
                t=t, interval_steps=int(replan_interval), last_replan_step=int(last_replan_step)
            )
            triggers = (env.check_replanning_trigger() or []) if env_triggers_enabled else []

            allow_trigger = rs.allow_trigger(
                t=t, last_replan_step=int(last_replan_step), trigger_cooldown_steps=int(trigger_cooldown_steps)
            )

            any_trigger = bool(periodic_trigger) or bool(triggers)
            if any_trigger and not allow_trigger:
                suppressed_triggers += 1

            if any_trigger and allow_trigger:
                last_replan_step = t
                replan_count += 1

                trigger_types: List[str] = []
                for tr in triggers:
                    if isinstance(tr, dict) and tr.get("type") is not None:
                        trigger_types.append(str(tr.get("type")))
                    else:
                        trigger_types.append(str(tr))
                trigger_dict: Dict[str, Any] = {
                    "unsafe": False,
                    "deadlock": False,
                    "types": trigger_types,
                    "periodic": periodic_trigger,
                }
                replan_trigger_type = (
                    str(trigger_types[0]) if trigger_types else ("periodic" if periodic_trigger else "unknown")
                )

                progress_since_last_replan = None
                if isinstance(cur_dist, (int, float)) and dist_at_last_replan is not None:
                    progress_since_last_replan = float(dist_at_last_replan) - float(cur_dist)

                telemetry = {
                    "clarification_budget_turns": clarification_budget_turns,
                    "churn": False,
                    "progress": progress_since_last_replan,
                    "lat_total_ms": last_plan_latency_ms,
                }

                if brace_enabled:
                    decision, state = controller.step(
                        state=state,
                        trigger=trigger_dict,
                        telemetry=telemetry,
                        remaining_budget=token_budget,
                        num_agents=int(base_cfg.get("num_agents", 1)),
                    )
                    protected_blocks = decision.protected_blocks
                    budget = decision.token_budget
                    mode = decision.mode
                    hazards = {
                        "hazard_slo": bool(decision.hazard_slo),
                        "hazard_churn": bool(decision.hazard_churn),
                        "hazard_deadlock": bool(decision.hazard_deadlock),
                        "hazard_unsafe": bool(decision.hazard_unsafe),
                        "cooldown_active": bool(decision.cooldown_active),
                        "rollback_flag": bool(decision.rollback_flag),
                        "min_commit_window": int(decision.min_commit_window),
                    }
                else:
                    protected_blocks = ("A", "B", "C", "D")
                    budget = token_budget
                    mode = "partial_replan"
                    hazards = {
                        "hazard_slo": False,
                        "hazard_churn": False,
                        "hazard_deadlock": False,
                        "hazard_unsafe": False,
                        "cooldown_active": False,
                        "rollback_flag": False,
                        "min_commit_window": 0,
                    }

                budget_log = int(budget) if budget is not None else 0
                planner_called = mode in ("full_replan", "partial_replan")
                if planner_called:
                    t0 = time.time()
                    ctx_text, tok_stats = _serialize_context(
                        task=task,
                        agent_state=agent_state,
                        history=history,
                        protected_blocks=protected_blocks,
                        token_budget=budget,
                        pruning_enabled=pruning_enabled,
                        keep_ratio=keep_ratio,
                        compress_strategy=compress_strategy,
                        summary_head_tokens=summary_head_tokens,
                        summary_tail_tokens=summary_tail_tokens,
                        rng=ep_rng,
                    )
                    plan_latency_ms = (time.time() - t0) * 1000.0
                    last_plan_latency_ms = plan_latency_ms
                    if isinstance(cur_dist, (int, float)):
                        dist_at_last_replan = float(cur_dist)
                else:
                    tok_stats = {
                        "tokens_in": 0,
                        "tokens_after": 0,
                        "tokens_protected": 0,
                        "tokens_prunable": 0,
                        "target_prunable_keep": None,
                        "target_prunable_budget": None,
                    }
                    plan_latency_ms = 0.0

                compress_strategy_norm = (
                    str(compress_strategy).strip().lower() if compress_strategy is not None else None
                )
                summary_enabled = (
                    compress_strategy_norm in ("structured_summary", "summary", "head_tail_summary")
                    if compress_strategy_norm is not None
                    else None
                )

                slo_violation = bool(planner_called and slo_ms > 0 and plan_latency_ms > float(slo_ms))
                slo_over_ms = max(0.0, float(plan_latency_ms) - float(slo_ms)) if slo_ms > 0 else 0.0

                ctx.append_event(
                    {
                        "domain": "habitat",
                        "variant": variant["name"],
                        "episode_id": episode_id,
                        "t": t,
                        "brace_enabled": brace_enabled,
                        "pruning_enabled": pruning_enabled,
                        "keep_ratio": keep_ratio,
                        "compress_strategy": compress_strategy_norm,
                        "summary_head_tokens": summary_head_tokens if summary_enabled else None,
                        "summary_tail_tokens": summary_tail_tokens if summary_enabled else None,
                        "summary_tokens": tok_stats.get("summary_tokens", 0),
                        "summary_compress_enabled": summary_enabled,
                        "instruction_style": instruction_style,
                        "ambiguity_type": ambiguity_type,
                        "clarification_turns": clarification_budget_turns,
                        "clarification_budget_turns": clarification_budget_turns,
                        "clarification_tokens": clarif_tokens,
                        "clarification_lat_ms": clarif_lat_ms,
                        "mode": mode,
                        "planner_called": bool(planner_called),
                        "slo_ms": int(slo_ms),
                        "slo_violation": bool(slo_violation),
                        "slo_over_ms": float(slo_over_ms),
                        "token_budget": int(budget_log),
                        "replan_interval_steps": int(replan_interval),
                        "replan_interval": replan_interval,
                        "trigger_cooldown_steps": trigger_cooldown_steps,
                        "replan_trigger_type": str(replan_trigger_type),
                        "tokens_in": tok_stats["tokens_in"],
                        "tokens_after_prune": tok_stats["tokens_after"],
                        "tokens_protected": tok_stats["tokens_protected"],
                        "tokens_prunable": tok_stats["tokens_prunable"],
                        "target_prunable_keep": tok_stats.get("target_prunable_keep"),
                        "target_prunable_budget": tok_stats.get("target_prunable_budget"),
                        "lat_total_ms": plan_latency_ms,
                        "trigger": trigger_dict,
                        **hazards,
                    }
                )

            action = _policy_action(agent_state, last_obs if isinstance(last_obs, dict) else None)
            obs, reward, done, info = env.step(action)
            last_obs = obs
            history.append(
                {
                    "step": t,
                    "action": action.get("action"),
                    "distance_to_goal": info.get("agent_state", {}).get("distance_to_goal"),
                }
            )

            if done:
                metrics = env.env.get_metrics()
                success = bool(metrics.get("success", False))
                if success and replan_count >= min_replans:
                    break

        metrics = env.env.get_metrics()
        ep_wall_ms = (time.time() - ep_t0) * 1000.0
        compress_strategy_norm = (
            str(compress_strategy).strip().lower() if compress_strategy is not None else None
        )
        summary_enabled = (
            compress_strategy_norm in ("structured_summary", "summary", "head_tail_summary")
            if compress_strategy_norm is not None
            else None
        )
        ctx.append_episode(
            {
                "domain": "habitat",
                "variant": variant["name"],
                "episode_id": episode_id,
                "success": float(bool(metrics.get("success", False))),
                "spl": float(metrics.get("spl", 0.0)),
                "step_count": float(env.step_count),
                "replan_cycles": float(replan_count),
                "effective_replans_per_episode": float(replan_count),
                "suppressed_triggers": float(suppressed_triggers),
                "episode_wall_time_ms": float(ep_wall_ms),
                "effective_replans_per_min": float(replan_count) / (float(ep_wall_ms) / 60000.0)
                if ep_wall_ms > 0
                else None,
                "instruction_style": instruction_style,
                "ambiguity_type": ambiguity_type,
                "clarification_budget_turns": float(clarification_budget_turns),
                "clarification_tokens": float(clarif_tokens),
                "clarification_lat_ms": float(clarif_lat_ms),
                "replan_interval": float(replan_interval),
                "replan_interval_steps": float(replan_interval),
                "trigger_cooldown_steps": float(trigger_cooldown_steps),
                "compress_strategy": compress_strategy_norm,
                "summary_head_tokens": summary_head_tokens if summary_enabled else None,
                "summary_tail_tokens": summary_tail_tokens if summary_enabled else None,
            }
        )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/habitat_brace_smoke.json")
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--run_name", default="habitat_brace_smoke")
    args = ap.parse_args()

    base_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ctx = RunContext.create(args.runs_root, args.run_name, base_cfg)

    HabitatMultiAgentWrapper = _load_habitat_wrapper()
    max_steps = int(base_cfg.get("max_steps", 200))
    env = HabitatMultiAgentWrapper(config_path=None, scene_id=None, max_episode_steps=max_steps + 50)

    for variant in _expand_variants(base_cfg):
        run_variant(ctx, env, variant, base_cfg)

    try:
        env.close()
    except Exception:
        pass

    ctx.write_summary({"note": "brace_pointnav_runner (token counts are proxy word counts)"})
    print(ctx.run_dir)


if __name__ == "__main__":
    main()
