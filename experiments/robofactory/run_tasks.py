from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from brace.controller import BraceController, BraceHyperparams, BraceState
from experiments.common.clarification import build_text_instruction
from experiments.common.logging import RunContext
from experiments.common import replan_schedule as rs
from experiments.common.context_compress.baselines import extra_overhead_ms, normalize_method
from experiments.robofactory.serialize import approx_tokens, plan_hash, serialize_context
from experiments.robofactory.retrieval import retrieve_snippets


def _resolve_path(p: str) -> str:
    pp = Path(p)
    if not pp.is_absolute():
        pp = _PROJ_ROOT / pp
    return str(pp.resolve())


def _expand_variants(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants = base_cfg.get("variants", [])
    if variants:
        return variants

    grid = base_cfg.get("variant_grid")
    if not grid:
        raise ValueError("Config must define either `variants` or `variant_grid`.")

    brace_opts = list(grid.get("brace_enabled", [False, True]))
    prune_opts = list(grid.get("pruning_enabled", [False, True]))
    keep_ratios = list(grid.get("keep_ratio", [1.0]))
    token_budgets = list(grid.get("token_budget", [base_cfg.get("token_budget", 0)]))
    replan_intervals = list(grid.get("replan_interval_steps", [base_cfg.get("replan_interval_steps", 10)]))
    if "context_compress_method" in grid:
        method_opts = list(grid.get("context_compress_method", []))
    else:
        method_opts = list(grid.get("context_strategy", [])) if "context_strategy" in grid else []

    out: List[Dict[str, Any]] = []
    for brace_enabled in brace_opts:
        # New-style grid: explicit compression methods (erecap/recency/random/structured_summary/none).
        if method_opts:
            for method in method_opts:
                # `normalize_method` gates on pruning_enabled; for grid expansion we always want normalization.
                method = normalize_method(str(method), pruning_enabled=True)
                for replan_interval in replan_intervals:
                    for token_budget in token_budgets:
                        kr_list = keep_ratios if method == "erecap" else [1.0]
                        for keep_ratio in kr_list:
                            name = (
                                f"{'brace' if brace_enabled else 'nobrace'}"
                                f"__m{method}"
                                f"__r{keep_ratio}"
                                f"__int{replan_interval}"
                                f"__B{token_budget}"
                            )
                            out.append(
                                {
                                    "name": name,
                                    "brace_enabled": bool(brace_enabled),
                                    "context_compress_method": method,
                                    "pruning_enabled": bool(method == "erecap"),
                                    "keep_ratio": float(keep_ratio),
                                    "token_budget": int(token_budget),
                                    "replan_interval_steps": int(replan_interval),
                                }
                            )
            continue

        # Back-compat grid: pruning_enabled + keep_ratio.
        for pruning_enabled in prune_opts:
            method = "erecap" if bool(pruning_enabled) else "none"
            for replan_interval in replan_intervals:
                for token_budget in token_budgets:
                    if pruning_enabled:
                        kr_list = keep_ratios
                    else:
                        kr_list = [1.0]
                    for keep_ratio in kr_list:
                        name = (
                            f"{'brace' if brace_enabled else 'nobrace'}_"
                            f"{'prune' if pruning_enabled else 'noprune'}"
                            f"__r{keep_ratio}"
                            f"__int{replan_interval}"
                            f"__B{token_budget}"
                        )
                        out.append(
                            {
                                "name": name,
                                "brace_enabled": bool(brace_enabled),
                                "context_compress_method": method,
                                "pruning_enabled": bool(pruning_enabled),
                                "keep_ratio": float(keep_ratio),
                                "token_budget": int(token_budget),
                                "replan_interval_steps": int(replan_interval),
                            }
                        )
    return out


def _action_norms(action: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(action, dict):
        for k, v in action.items():
            try:
                import numpy as np

                arr = np.asarray(v, dtype=float).reshape(-1)
                out[str(k)] = float(np.linalg.norm(arr))
            except Exception:
                out[str(k)] = 0.0
    return out


def _obs_delta(prev_obs: Any, obs: Any) -> Optional[float]:
    if prev_obs is None or obs is None:
        return None
    try:
        import torch

        if isinstance(prev_obs, torch.Tensor) and isinstance(obs, torch.Tensor):
            d = torch.norm(obs.detach().cpu().float() - prev_obs.detach().cpu().float()).item()
            return float(d)
    except Exception:
        pass
    try:
        import numpy as np

        a = np.asarray(prev_obs, dtype=float).reshape(-1)
        b = np.asarray(obs, dtype=float).reshape(-1)
        return float(np.linalg.norm(a - b))
    except Exception:
        return None


def _coerce_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return bool(x.item())
    except Exception:
        pass
    if isinstance(x, (bool, int, float)):
        return bool(x)
    return None


def _load_env(rf_cfg: Dict[str, Any]) -> Any:
    repo_dir = _resolve_path(str(rf_cfg.get("repo_dir", "RoboFactory_workspace/RoboFactory")))
    run_dir = _resolve_path(str(rf_cfg.get("run_dir", "RoboFactory_workspace/RoboFactory__run")))
    os.environ.setdefault("ROBOFACTORY_RUN_DIR", run_dir)

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    import gymnasium as gym  # type: ignore
    import robofactory  # noqa: F401  # triggers env registration

    env_id = str(rf_cfg["env_id"])
    config_path = _resolve_path(str(rf_cfg["config_path"]))

    obs_mode = str(rf_cfg.get("obs_mode", "state"))
    control_mode = str(rf_cfg.get("control_mode", "pd_joint_pos"))
    render_mode = str(rf_cfg.get("render_mode", "rgb_array"))
    reward_mode = str(rf_cfg.get("reward_mode", "dense"))
    sim_backend = str(rf_cfg.get("sim_backend", "cpu"))
    shader = str(rf_cfg.get("shader", "default"))

    env = gym.make(
        env_id,
        config=config_path,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        reward_mode=reward_mode,
        sensor_configs=dict(shader_pack=shader),
        human_render_camera_configs=dict(shader_pack=shader),
        viewer_camera_configs=dict(shader_pack=shader),
        sim_backend=sim_backend,
    )
    return env


def _maybe_wrap_recording(env: Any, *, output_dir: str, base_cfg: Dict[str, Any]) -> Any:
    """Optionally wrap env with RoboFactory's RecordEpisodeMA for MP4 demos."""
    if not bool(base_cfg.get("record_video", False)):
        return env

    try:
        from robofactory.utils.wrappers.record import RecordEpisodeMA  # type: ignore
    except Exception:
        return env

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_traj = bool(base_cfg.get("record_trajectory", False))
    video_fps = int(base_cfg.get("video_fps", 30))

    return RecordEpisodeMA(
        env,
        output_dir=str(out_dir),
        save_trajectory=save_traj,
        trajectory_name=None,
        save_video=True,
        info_on_video=False,
        save_on_reset=False,
        clean_on_close=True,
        record_reward=False,
        record_env_state=False,
        record_observation=False,
        video_fps=video_fps,
        avoid_overwriting_video=True,
        source_type="brace_domainb",
        source_desc="BRACE Domain B demo recording",
    )


def _simulate_latency_ms(
    *,
    rng: random.Random,
    tokens_after: int,
    ms_per_token: float,
    overhead_ms: float,
    tail_lognorm_sigma: float,
) -> float:
    tokens_after = max(0, int(tokens_after))
    ms_per_token = max(0.0, float(ms_per_token))
    overhead_ms = max(0.0, float(overhead_ms))
    tail_lognorm_sigma = max(0.0, float(tail_lognorm_sigma))
    noise = rng.lognormvariate(0.0, tail_lognorm_sigma) if tail_lognorm_sigma > 0 else 1.0
    return float(overhead_ms + ms_per_token * tokens_after * noise)


def _http_post_json(url: str, payload: Dict[str, Any], *, timeout_s: float = 30.0) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        raise RuntimeError(f"HTTPError {e.code} {e.reason}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"HTTP request failed: {e}") from e


def _llm_count_tokens(llm_server_url: str, text: str) -> int:
    resp = _http_post_json(f"{llm_server_url.rstrip('/')}/count_tokens", {"text": str(text)}, timeout_s=30.0)
    if not bool(resp.get("ok", False)):
        raise RuntimeError(f"llm_count_tokens failed: {resp}")
    return int(resp.get("tokens", 0) or 0)


def _llm_generate(llm_server_url: str, prompt: str, *, max_new_tokens: int, temperature: float) -> Dict[str, Any]:
    resp = _http_post_json(
        f"{llm_server_url.rstrip('/')}/generate",
        {"prompt": str(prompt), "max_new_tokens": int(max_new_tokens), "temperature": float(temperature)},
        timeout_s=300.0,
    )
    if not bool(resp.get("ok", False)):
        raise RuntimeError(f"llm_generate failed: {resp}")
    return dict(resp)


def _load_mp_solve(env_id: str) -> Optional[Callable[..., Any]]:
    """Return RoboFactory motion-planning solve() for env_id if available."""
    env_id = str(env_id)
    try:
        if env_id == "LiftBarrier-rf":
            from robofactory.planner.solutions.lift_barrier import solve  # type: ignore

            return solve
        if env_id == "PassShoe-rf":
            from robofactory.planner.solutions.pass_shoe import solve  # type: ignore

            return solve
        if env_id == "CameraAlignment-rf":
            from robofactory.planner.solutions.camera_alignment import solve  # type: ignore

            return solve
        if env_id == "TakePhoto-rf":
            from robofactory.planner.solutions.take_photo import solve  # type: ignore

            return solve
    except Exception:
        return None
    return None


class ReplanLoggingWrapper:
    """Intercepts env.step/reset to run a replanning loop + log schema-aligned events.

    Works for:
      - random policy (the runner calls step())
      - RoboFactory motion-planning solutions (they call step() internally)
    """

    def __init__(
        self,
        *,
        env: Any,
        ctx: RunContext,
        base_cfg: Dict[str, Any],
        variant: Dict[str, Any],
        controller: BraceController,
        env_id: str,
        task_text: str,
        instruction_style: str,
        ambiguity_type: str,
        clarification_budget_turns: int,
        clarification_tokens: int,
        clarification_lat_ms: float,
        clarification_transcript: List[Dict[str, Any]],
    ) -> None:
        self.env = env
        self.ctx = ctx
        self.base_cfg = base_cfg
        self.variant = variant
        self.controller = controller
        self.env_id = str(env_id)
        self.task_text = str(task_text)
        self.instruction_style = str(instruction_style)
        self.ambiguity_type = str(ambiguity_type)
        self.clarification_budget_turns = int(clarification_budget_turns)
        self.clarification_tokens = int(clarification_tokens)
        self.clarification_lat_ms = float(clarification_lat_ms)
        self.clarification_transcript = list(clarification_transcript or [])

        self.episode_index: int = 0
        self.seed: int = 0
        self.episode_id: str = ""

        self.t: int = 0
        self.max_history: int = int(base_cfg.get("max_history", 20))
        self.hard_step_limit: Optional[int] = (
            int(base_cfg["hard_step_limit"]) if base_cfg.get("hard_step_limit") is not None else None
        )

        self.obs: Any = None
        self.info: Dict[str, Any] = {}

        self.state = BraceState()
        self.history: List[Dict[str, Any]] = []
        self.replan_cycles: int = 0
        self.last_replan_step: int = -1
        self.suppressed_triggers: int = 0
        self.episode_t0: float = time.time()

        self.last_plan_latency_ms: Optional[float] = None  # lat_total_ms (simulated or measured)
        self.last_plan_hash: Optional[str] = None

        self.stall_steps: int = 0
        self.deadlock_flag: bool = False
        self.wait_time_ms: float = 0.0
        self.virtual_time_ms: float = 0.0
        self.slo_violation_streak: int = 0
        self.replan_max_calls_per_episode: Optional[int] = (
            int(base_cfg["replan_max_calls_per_episode"])
            if base_cfg.get("replan_max_calls_per_episode") is not None
            else None
        )

        self.episode_reward_sum: float = 0.0
        self.reward_sum_at_last_replan: float = 0.0

        self.error: Optional[str] = None

        # Deterministic RNG for latency simulation (per wrapper instance).
        self.rng = random.Random(int(base_cfg.get("random_seed", 0)) + 1337)

    def set_episode(self, episode_index: int, seed: int) -> None:
        self.episode_index = int(episode_index)
        self.seed = int(seed)
        # Mix in episode-specific entropy for simulation noise.
        self.rng = random.Random(int(self.seed) + abs(hash(self.variant.get("name", ""))) % (2**31))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
        self.t = 0
        self.episode_id = f"{self.env_id}__seed{self.seed}__ep{self.episode_index:04d}"

        self.state = BraceState()
        self.history = []
        self.replan_cycles = 0
        self.last_replan_step = -1
        self.suppressed_triggers = 0
        self.episode_t0 = time.time()
        self.last_plan_latency_ms = None
        self.last_plan_hash = None
        self.stall_steps = 0
        self.deadlock_flag = False
        self.wait_time_ms = 0.0
        self.virtual_time_ms = 0.0
        self.slo_violation_streak = 0
        self.episode_reward_sum = 0.0
        self.reward_sum_at_last_replan = 0.0
        self.error = None

        # Audit: clarification is a separate phase (tokens/latency are NOT replanning tokens/latency).
        self.ctx.log_phase(
            phase="clarification",
            variant=self.variant.get("name"),
            episode_id=self.episode_id,
            instruction_style=self.instruction_style,
            ambiguity_type=self.ambiguity_type,
            clarification_budget_turns=int(self.clarification_budget_turns),
            clarification_tokens=int(self.clarification_tokens),
            clarification_lat_ms=float(self.clarification_lat_ms),
            tokens_in=int(self.clarification_tokens),
            tokens_after_prune=int(self.clarification_tokens),
            lat_total_ms=float(self.clarification_lat_ms),
            clarification_transcript=self.clarification_transcript,
        )

        obs, info = self.env.reset(seed=self.seed, options=options)
        self.obs = obs
        self.info = dict(info) if isinstance(info, dict) else {"info": info}
        return obs, self.info

    def _maybe_replan(self) -> None:
        slo_ms = int(self.base_cfg.get("slo_ms", 250))
        deadlock_window_steps = int(self.base_cfg.get("deadlock_window_steps", 50))

        replan_interval_steps = int(
            self.variant.get("replan_interval_steps", self.base_cfg.get("replan_interval_steps", 10))
        )
        trigger_cooldown_steps = int(
            self.variant.get("trigger_cooldown_steps", self.base_cfg.get("trigger_cooldown_steps", 0))
        )

        brace_enabled = bool(self.variant.get("brace_enabled", False))
        raw_method = self.variant.get(
            "context_compress_method",
            self.variant.get("context_strategy", self.variant.get("baseline_method")),
        )
        rag_enabled = bool(self.variant.get("rag_enabled", self.base_cfg.get("rag_enabled", False)))
        rag_source = str(self.variant.get("rag_source", self.base_cfg.get("rag_source", "static"))).strip().lower()
        method_enabled = raw_method is not None and str(raw_method).strip().lower() not in (
            "",
            "none",
            "identity",
            "no",
            "off",
        )
        pruning_enabled_variant = bool(self.variant.get("pruning_enabled", method_enabled))
        keep_ratio = float(self.variant.get("keep_ratio", 1.0))
        method = normalize_method(
            raw_method,
            pruning_enabled=pruning_enabled_variant,
        )
        pruning_enabled_event = method in ("erecap", "random", "recency")
        summary_compress_enabled_event = method == "structured_summary"
        compress_enabled = method != "none"
        # For budget-matched baselines (e.g., recency/random/structured_summary), allow using keep_ratio as the knob.
        keep_ratio_eff = keep_ratio if method in ("erecap", "recency", "random", "structured_summary") else 1.0

        token_budget = int(self.variant.get("token_budget", self.base_cfg.get("token_budget", 0)))
        token_budget = token_budget if token_budget > 0 else None

        periodic = rs.periodic_trigger(
            t=self.t, interval_steps=replan_interval_steps, last_replan_step=self.last_replan_step
        )
        allow = rs.allow_trigger(
            t=self.t,
            last_replan_step=self.last_replan_step,
            trigger_cooldown_steps=trigger_cooldown_steps,
        )

        deadlock = deadlock_window_steps > 0 and self.stall_steps >= deadlock_window_steps
        failure = False
        any_trigger = bool(periodic or failure or deadlock)
        if any_trigger and not allow:
            self.suppressed_triggers += 1
            return
        should_replan = bool(any_trigger and allow)
        if not should_replan:
            return

        self.last_replan_step = int(self.t)
        self.replan_cycles += 1
        if self.replan_max_calls_per_episode is not None and int(self.replan_cycles) > int(self.replan_max_calls_per_episode):
            return

        replan_trigger_type = rs.trigger_type_primary(periodic=periodic, failure=failure, deadlock=deadlock)
        progress_since_last_replan = self.episode_reward_sum - self.reward_sum_at_last_replan
        churn_flag = bool(self.last_plan_latency_ms is not None and float(self.last_plan_latency_ms) > float(slo_ms))
        telemetry = {
            "clarification_budget_turns": int(self.clarification_budget_turns),
            "churn": bool(churn_flag),
            "progress": float(progress_since_last_replan),
            "lat_total_ms": self.last_plan_latency_ms,
        }

        trigger_dict = rs.build_trigger_dict(periodic=periodic, failure=failure, deadlock=deadlock, unsafe=False)

        if brace_enabled:
            decision, self.state = self.controller.step(
                state=self.state,
                trigger=trigger_dict,
                telemetry=telemetry,
                remaining_budget=token_budget,
                num_agents=int(self.base_cfg.get("num_agents", 2)),
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

        planner_called = mode in ("full_replan", "partial_replan")
        lat_serialize_ms = 0.0
        lat_prune_ms = 0.0
        lat_summary_ms = None
        lat_retrieval_ms = None
        lat_llm_ms: Optional[float] = None
        tokens_out_llm: Optional[int] = None
        if planner_called:
            t0 = time.time()
            summary_overhead_ms = float(self.base_cfg.get("summary_overhead_ms", 0.0))
            if summary_compress_enabled_event:
                lat_summary_ms = float(extra_overhead_ms(method, summary_overhead_ms=summary_overhead_ms))

            summary_head_tokens = int(
                self.variant.get("summary_head_tokens", self.base_cfg.get("summary_head_tokens", 32))
            )
            summary_tail_tokens = int(
                self.variant.get("summary_tail_tokens", self.base_cfg.get("summary_tail_tokens", 64))
            )

            llm_server_url = self.base_cfg.get("llm_server_url")
            token_counter = None
            if llm_server_url:
                token_counter = lambda txt: _llm_count_tokens(str(llm_server_url), txt)

            retrieval_lines = []
            retrieval_note = None
            if rag_enabled:
                t_r0 = time.perf_counter()
                rag_k = int(self.variant.get("rag_k", self.base_cfg.get("rag_k", 4)))
                use_memory = rag_source in ("memory", "memory_only", "memory_then_static", "episodic")
                fallback_static = rag_source not in ("memory_only",)
                memory_jsonl = None
                if use_memory:
                    memory_root = self.base_cfg.get("rag_memory_root")
                    if not memory_root:
                        memory_root = os.path.join(str(self.ctx.run_dir), "rag_memory")
                    variant_mem_dir = os.path.join(str(memory_root), str(self.variant.get("name", "unknown")))
                    os.makedirs(variant_mem_dir, exist_ok=True)
                    memory_jsonl = os.path.join(variant_mem_dir, f"{self.env_id}.jsonl")
                retrieval_lines, retrieval_note = retrieve_snippets(
                    env_id=self.env_id,
                    query=str(self.task_text),
                    k=rag_k,
                    rng=self.rng,
                    memory_jsonl=memory_jsonl,
                    max_memory_items=int(self.base_cfg.get("rag_memory_max_items", 200)),
                    fallback_static=bool(fallback_static),
                )
                # Auditable retrieval overhead (ms): measured + configurable synthetic components.
                retrieved_tokens_sim = sum(approx_tokens(x) for x in retrieval_lines) if retrieval_lines else 0
                lat_retrieval_ms = float((time.perf_counter() - t_r0) * 1000.0)
                lat_retrieval_ms += float(self.base_cfg.get("retrieval_overhead_ms", 0.0))
                lat_retrieval_ms += float(self.base_cfg.get("retrieval_ms_per_token", 0.0)) * float(retrieved_tokens_sim)

            ctx_text, tok_stats, ctx_before = serialize_context(
                task=self.task_text,
                env_id=self.env_id,
                obs=self.obs,
                info=dict(self.info),
                history=self.history,
                retrieval_lines=retrieval_lines,
                protected_blocks=protected_blocks,
                token_budget=budget,
                context_compress_method=method,
                pruning_enabled=compress_enabled,
                keep_ratio=keep_ratio_eff,
                token_counter=token_counter,
                rng=self.rng,
                summary_head_tokens=summary_head_tokens,
                summary_tail_tokens=summary_tail_tokens,
                max_history=self.max_history,
            )
            lat_serialize_ms = (time.time() - t0) * 1000.0
            self.reward_sum_at_last_replan = float(self.episode_reward_sum)

            ph = plan_hash(ctx_text)
            churn_score = 1.0 if (self.last_plan_hash is not None and ph != self.last_plan_hash) else 0.0
            self.last_plan_hash = ph

            # Optional: real tokenizer + real LLM latency via local HTTP service (runs in habitat env).
            if llm_server_url:
                try:
                    tok_stats["tokens_in_hf"] = int(_llm_count_tokens(str(llm_server_url), ctx_before))
                    tok_stats["tokens_after_hf"] = int(_llm_count_tokens(str(llm_server_url), ctx_text))
                except Exception:
                    tok_stats["tokens_in_hf"] = None
                    tok_stats["tokens_after_hf"] = None

                if bool(self.base_cfg.get("llm_generate", False)):
                    out = _llm_generate(
                        str(llm_server_url),
                        ctx_text,
                        max_new_tokens=int(self.base_cfg.get("llm_max_new_tokens", 32)),
                        temperature=float(self.base_cfg.get("llm_temperature", 0.0)),
                    )
                    lat_llm_ms = float(out.get("lat_total_ms", 0.0) or 0.0)
                    tokens_out_llm = int(out.get("tokens_out", 0) or 0)
                    try:
                        self.last_plan_hash = plan_hash(str(out.get("output_text", "")))
                    except Exception:
                        pass
            if retrieval_note is not None:
                tok_stats["retrieval_note"] = str(retrieval_note)
        else:
            tok_stats = {
                "tokens_in": 0,
                "tokens_after": 0,
                "tokens_task": 0,
                "tokens_state": 0,
                "tokens_safety": 0,
                "tokens_coord": 0,
                "tokens_history": 0,
                "tokens_protected": 0,
                "tokens_prunable": 0,
                "target_prunable_keep": None,
                "target_prunable_budget": None,
                "context_length_before_chars": 0,
                "context_length_after_chars": 0,
            }
            ph = self.last_plan_hash
            churn_score = 0.0

        tokens_after = int(tok_stats.get("tokens_after", 0) or 0)

        simulate = bool(self.base_cfg.get("simulate_planner_latency", False))
        if planner_called and lat_llm_ms is not None:
            lat_total_ms = float(lat_llm_ms) + float(lat_prune_ms) + float(lat_serialize_ms)
            if lat_summary_ms is not None:
                lat_total_ms += float(lat_summary_ms)
            if lat_retrieval_ms is not None:
                lat_total_ms += float(lat_retrieval_ms)
        elif simulate and planner_called:
            lat_compute_ms = _simulate_latency_ms(
                rng=self.rng,
                tokens_after=tokens_after,
                ms_per_token=float(self.base_cfg.get("ms_per_token", 1.0)),
                overhead_ms=float(self.base_cfg.get("overhead_ms", 10.0)),
                tail_lognorm_sigma=float(self.base_cfg.get("tail_lognorm_sigma", 0.25)),
            )
            lat_total_ms = float(lat_compute_ms) + float(lat_prune_ms) + float(lat_serialize_ms)
            if lat_summary_ms is not None:
                lat_total_ms += float(lat_summary_ms)
            if lat_retrieval_ms is not None:
                lat_total_ms += float(lat_retrieval_ms)
        else:
            lat_total_ms = float(lat_prune_ms) + float(lat_serialize_ms)
            if lat_summary_ms is not None:
                lat_total_ms += float(lat_summary_ms)
            if lat_retrieval_ms is not None:
                lat_total_ms += float(lat_retrieval_ms)

        self.last_plan_latency_ms = float(lat_total_ms)
        self.deadlock_flag = bool(self.deadlock_flag or deadlock)
        if planner_called:
            self.wait_time_ms += float(lat_total_ms)
            self.virtual_time_ms += float(lat_total_ms)

        # Time budget proxy: treat planning latency as "lost wall-time" that can cause timeouts.
        time_budget_ms = self.base_cfg.get("episode_time_budget_ms")
        if planner_called and time_budget_ms is not None:
            budget_f = float(time_budget_ms)
            if budget_f > 0 and float(self.virtual_time_ms) > budget_f:
                self.deadlock_flag = True
                self.error = (
                    f"TIME_BUDGET_EXCEEDED: virtual_time_ms={float(self.virtual_time_ms):.1f} > "
                    f"budget_ms={budget_f:.1f}"
                )
                raise RuntimeError(self.error)

        # Track SLO violations for stability heuristics.
        if planner_called and slo_ms > 0 and float(lat_total_ms) > float(slo_ms):
            self.slo_violation_streak += 1
        else:
            self.slo_violation_streak = 0

        self.ctx.append_event(
            {
                "domain": "robofactory",
                "task": self.base_cfg.get("task", self.env_id),
                "variant": self.variant["name"],
                "episode_id": self.episode_id,
                "t": int(self.t),
                "brace_enabled": brace_enabled,
                "pruning_enabled": bool(pruning_enabled_event),
                "rag_enabled": bool(rag_enabled),
                "rag_source": str(rag_source) if rag_enabled else None,
                "summary_compress_enabled": bool(summary_compress_enabled_event),
                "keep_ratio": keep_ratio,
                "context_compress_method": method,
                "mode": mode,
                "planner_called": bool(planner_called),
                "slo_ms": slo_ms,
                "token_budget": budget,
                "instruction_style": self.instruction_style,
                "ambiguity_type": self.ambiguity_type,
                "clarification_budget_turns": int(self.clarification_budget_turns),
                "clarification_tokens": int(self.clarification_tokens),
                "clarification_lat_ms": float(self.clarification_lat_ms),
                "replan_interval_steps": int(replan_interval_steps),
                "trigger_cooldown_steps": int(trigger_cooldown_steps),
                "replan_trigger_type": str(replan_trigger_type),
                "trigger": trigger_dict,
                "tokens_in": tok_stats.get("tokens_in_hf") if tok_stats.get("tokens_in_hf") is not None else tok_stats["tokens_in"],
                "tokens_after_prune": tok_stats.get("tokens_after_hf") if tok_stats.get("tokens_after_hf") is not None else tok_stats["tokens_after"],
                "tokens_in_proxy_words": tok_stats.get("tokens_in"),
                "tokens_after_prune_proxy_words": tok_stats.get("tokens_after"),
                "tokens_out": tokens_out_llm,
                "tokens_task": tok_stats.get("tokens_task"),
                "tokens_state": tok_stats.get("tokens_state"),
                "tokens_safety": tok_stats.get("tokens_safety"),
                "tokens_coord": tok_stats.get("tokens_coord"),
                "tokens_history": tok_stats.get("tokens_history"),
                "context_length_before_chars": tok_stats.get("context_length_before_chars"),
                "context_length_after_chars": tok_stats.get("context_length_after_chars"),
                "target_prunable_keep": tok_stats.get("target_prunable_keep"),
                "target_prunable_budget": tok_stats.get("target_prunable_budget"),
                "retrieved_tokens": tok_stats.get("retrieved_tokens"),
                "kept_tokens": tok_stats.get("kept_tokens"),
                "retrieval_note": tok_stats.get("retrieval_note"),
                "lat_total_ms": float(lat_total_ms),
                "lat_prune_ms": float(lat_prune_ms),
                "lat_retrieval_ms": float(lat_retrieval_ms) if lat_retrieval_ms is not None else None,
                "lat_summary_ms": float(lat_summary_ms) if lat_summary_ms is not None else None,
                "lat_serialize_ms": float(lat_serialize_ms),
                "lat_llm_ms": float(lat_llm_ms) if lat_llm_ms is not None else None,
                "plan_hash": ph,
                "plan_churn_score": float(churn_score),
                "deadlock_flag": bool(deadlock),
                "wait_time_ms": float(self.wait_time_ms),
                "error": self.error,
                **hazards,
            }
        )

    def step(self, action: Any) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
        if self.hard_step_limit is not None and self.t >= int(self.hard_step_limit):
            raise RuntimeError(f"Step limit exceeded: t={self.t} >= hard_step_limit={self.hard_step_limit}")

        # Replan decision happens on the current state before taking the next action.
        self._maybe_replan()

        t_step0 = time.time()
        obs, reward, terminated, truncated, info = self.env.step(action)
        step_ms = (time.time() - t_step0) * 1000.0

        # Update wait proxy: all agents idle (small action norm).
        action_norms = _action_norms(action)
        wait_eps = float(self.base_cfg.get("wait_action_l2_epsilon", 0.01))
        waiting_agents = [k for k, v in action_norms.items() if float(v) < wait_eps]
        if action_norms and len(waiting_agents) == len(action_norms):
            self.wait_time_ms += float(step_ms)

        # Update stall/deadlock proxy from obs delta.
        od = _obs_delta(self.obs, obs)
        stall_obs_delta_epsilon = float(self.base_cfg.get("stall_obs_delta_epsilon", 1e-6))
        stalled = (od is not None) and (float(od) < stall_obs_delta_epsilon)
        self.stall_steps = self.stall_steps + 1 if stalled else 0

        self.episode_reward_sum += float(reward) if isinstance(reward, (int, float)) else 0.0

        success = _coerce_bool(info.get("success")) if isinstance(info, dict) else None

        self.history.append(
            {
                "t": int(self.t),
                "reward": float(reward) if isinstance(reward, (int, float)) else reward,
                "success": success,
                "obs_delta": float(od) if od is not None else None,
                "action_norms": action_norms,
            }
        )
        if self.max_history > 0 and len(self.history) > int(self.max_history) * 2:
            # Keep the list bounded; serialize_context will subselect last N anyway.
            self.history = self.history[-int(self.max_history) * 2 :]

        self.obs = obs
        self.info = dict(info) if isinstance(info, dict) else {"info": info}
        self.t += 1
        return obs, reward, terminated, truncated, self.info

    def finalize_episode(self) -> None:
        success_env = _coerce_bool(self.info.get("success"))
        success_final = bool(success_env) and (self.error is None)
        ep_wall_ms = (time.time() - float(self.episode_t0)) * 1000.0
        replan_interval_steps = int(self.variant.get("replan_interval_steps", self.base_cfg.get("replan_interval_steps", 10)))
        trigger_cooldown_steps = int(self.variant.get("trigger_cooldown_steps", self.base_cfg.get("trigger_cooldown_steps", 0)))

        # Optional: append a simple episodic memory record for RAG baselines (per-variant, per-env).
        rag_enabled = bool(self.variant.get("rag_enabled", self.base_cfg.get("rag_enabled", False)))
        rag_source = str(self.variant.get("rag_source", self.base_cfg.get("rag_source", "static"))).strip().lower()
        use_memory = rag_enabled and rag_source in ("memory", "memory_only", "memory_then_static", "episodic")
        if use_memory:
            try:
                memory_root = self.base_cfg.get("rag_memory_root")
                if not memory_root:
                    memory_root = os.path.join(str(self.ctx.run_dir), "rag_memory")
                variant_mem_dir = os.path.join(str(memory_root), str(self.variant.get("name", "unknown")))
                os.makedirs(variant_mem_dir, exist_ok=True)
                memory_jsonl = os.path.join(variant_mem_dir, f"{self.env_id}.jsonl")
                rec = {
                    "env_id": self.env_id,
                    "variant": self.variant.get("name"),
                    "episode_id": self.episode_id,
                    "success": bool(success_final),
                    "deadlock_flag": bool(self.deadlock_flag),
                    "replan_cycles": int(self.replan_cycles),
                    "error": self.error,
                    "text": (
                        f"EP_SUMMARY env_id={self.env_id} episode_id={self.episode_id} "
                        f"success={int(bool(success_final))} deadlock={int(bool(self.deadlock_flag))} "
                        f"replans={int(self.replan_cycles)} error={str(self.error)}"
                    ),
                }
                with open(memory_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
            except Exception:
                pass

        # Optional demo video flush (when env is wrapped by RecordEpisodeMA).
        if bool(self.base_cfg.get("record_video", False)) and hasattr(self.env, "flush_video"):
            try:
                video_name = f"{self.variant['name']}__seed{self.seed}"
                self.env.flush_video(name=video_name)
                demo_root = self.base_cfg.get("demo_output_dir")
                if demo_root is None:
                    demo_root = str(_PROJ_ROOT / "artifacts" / "demos" / "robofactory" / self.ctx.run_id)
                self.ctx.log_phase(
                    phase="demo_video",
                    variant=self.variant["name"],
                    episode_id=self.episode_id,
                    demo_dir=str(demo_root),
                    video_name=str(video_name),
                )
            except Exception:
                pass

        self.ctx.append_episode(
            {
                "domain": "robofactory",
                "task": self.base_cfg.get("task", self.env_id),
                "variant": self.variant["name"],
                "episode_id": self.episode_id,
                "success": float(bool(success_final)),
                "spl": 0.0,
                "step_count": float(self.t),
                "replan_cycles": float(self.replan_cycles),
                "effective_replans_per_episode": float(self.replan_cycles),
                "suppressed_triggers": float(self.suppressed_triggers),
                "replan_interval_steps": float(replan_interval_steps),
                "trigger_cooldown_steps": float(trigger_cooldown_steps),
                "episode_wall_time_ms": float(ep_wall_ms),
                "episode_virtual_time_ms": float(self.virtual_time_ms),
                "effective_replans_per_min": float(self.replan_cycles) / (float(ep_wall_ms) / 60000.0)
                if ep_wall_ms > 0
                else None,
                "deadlock_flag": float(bool(self.deadlock_flag)),
                "wait_time_ms": float(self.wait_time_ms),
                "instruction_style": self.instruction_style,
                "ambiguity_type": self.ambiguity_type,
                "clarification_budget_turns": float(self.clarification_budget_turns),
                "clarification_tokens": float(self.clarification_tokens),
                "clarification_lat_ms": float(self.clarification_lat_ms),
                "error": self.error,
            }
        )


def run_variant(ctx: RunContext, env: Any, variant: Dict[str, Any], base_cfg: Dict[str, Any]) -> None:
    rf_cfg = dict(base_cfg.get("robofactory", {}) or {})

    episodes = int(variant.get("episodes", base_cfg.get("episodes", 10)))
    max_steps = int(variant.get("max_steps", base_cfg.get("max_steps", 100)))

    env_id = str(rf_cfg.get("env_id", ""))
    oracle_task_text = str(base_cfg.get("task_text", base_cfg.get("task", env_id or "robofactory_task")))

    instruction_style = str(variant.get("instruction_style", base_cfg.get("instruction_style", "oracle"))).strip().lower()
    if instruction_style not in ("oracle", "coarsened"):
        instruction_style = "oracle"
    ambiguity_type = str(variant.get("ambiguity_type", base_cfg.get("ambiguity_type", "goal"))).strip().lower()
    if ambiguity_type not in ("goal", "process", "success"):
        ambiguity_type = "goal"
    clarification_budget_turns = int(
        variant.get(
            "clarification_budget_turns",
            variant.get(
                "clarification_turns",
                base_cfg.get("clarification_budget_turns", base_cfg.get("clarification_turns", 0)),
            ),
        )
    )
    clarification_budget_turns = max(0, clarification_budget_turns)
    clarification_overhead_ms = float(
        variant.get("clarification_overhead_ms", base_cfg.get("clarification_overhead_ms", 0.0))
    )
    clarification_ms_per_token = float(
        variant.get("clarification_ms_per_token", base_cfg.get("clarification_ms_per_token", 0.0))
    )

    clarification_tokens = 0
    clarification_lat_ms = 0.0
    clarification_transcript: List[Dict[str, Any]] = []
    task_text = oracle_task_text
    if instruction_style == "coarsened":
        clarif = build_text_instruction(
            oracle_instruction=oracle_task_text,
            style="coarsened",
            clarification_budget_turns=int(clarification_budget_turns),
            ambiguity_type=ambiguity_type,  # type: ignore[arg-type]
            token_counter=approx_tokens,
        )
        clarification_tokens = int(clarif.clarification_tokens)
        clarification_transcript = clarif.transcript_dicts()
        if clarif.clarification_transcript:
            clarification_lat_ms = float(
                float(clarification_overhead_ms) + float(clarification_ms_per_token) * float(clarification_tokens)
            )
        # If no clarification is allowed, the planner sees only the coarsened instruction.
        # Otherwise, assume clarification reveals details so the planner can use the oracle instruction.
        task_text = str(clarif.instruction) if clarification_budget_turns <= 0 else oracle_task_text

    slo_ms = int(base_cfg.get("slo_ms", 250))
    hparams_dict: Dict[str, Any] = dict(base_cfg.get("brace_hparams", {}) or {})
    hparams_dict.update(dict(variant.get("brace_hparams", {}) or {}))
    hparams = BraceHyperparams(slo_ms=slo_ms, **hparams_dict) if hparams_dict else BraceHyperparams(slo_ms=slo_ms)
    controller = BraceController(hparams)

    policy = str(base_cfg.get("policy", "random")).strip().lower()
    mp_solve = _load_mp_solve(env_id) if policy == "motion_planning" else None

    random_seed = int(base_cfg.get("random_seed", 0))

    record_video = bool(base_cfg.get("record_video", False))
    env_for_variant = env
    created_env = False
    if record_video:
        # For demo recording, isolate recorder state by using a fresh env per variant.
        demo_root = base_cfg.get("demo_output_dir")
        if demo_root is None:
            demo_root = str(_PROJ_ROOT / "artifacts" / "demos" / "robofactory" / ctx.run_id)
        env_for_variant = _maybe_wrap_recording(_load_env(rf_cfg), output_dir=str(demo_root), base_cfg=base_cfg)
        created_env = True

    try:
        for ep_i in range(episodes):
            seed = random_seed + ep_i
            wrapper = ReplanLoggingWrapper(
                env=env_for_variant,
                ctx=ctx,
                base_cfg=base_cfg,
                variant=variant,
                controller=controller,
                env_id=env_id,
                task_text=task_text,
                instruction_style=instruction_style,
                ambiguity_type=ambiguity_type,
                clarification_budget_turns=int(clarification_budget_turns),
                clarification_tokens=int(clarification_tokens),
                clarification_lat_ms=float(clarification_lat_ms),
                clarification_transcript=clarification_transcript,
            )
            wrapper.set_episode(ep_i, seed)

            if policy == "motion_planning":
                if mp_solve is None:
                    raise RuntimeError(
                        f"policy=motion_planning requested but no MP solve() is available for env_id={env_id}"
                    )
                try:
                    suppress = bool(base_cfg.get("suppress_policy_stdout", True))
                    if suppress:
                        with open(os.devnull, "w", encoding="utf-8") as devnull:
                            with contextlib.redirect_stdout(devnull):
                                mp_solve(wrapper, seed=seed, debug=False, vis=False)
                    else:
                        mp_solve(wrapper, seed=seed, debug=False, vis=False)
                except Exception as e:
                    wrapper.error = f"{type(e).__name__}: {e}"
                finally:
                    # If the solve() didn't call reset due to early failure, ensure we still emit an episode row.
                    if not wrapper.episode_id:
                        wrapper.reset(seed=seed)
                    wrapper.finalize_episode()
            else:
                wrapper.reset(seed=seed)
                try:
                    if policy.startswith("openmarl_"):
                        from experiments.robofactory.openmarl_executor import rollout_openmarl_policy

                        rollout_openmarl_policy(
                            wrapper,
                            ctx=ctx,
                            base_cfg=base_cfg,
                            variant=variant,
                            policy=policy,
                        )
                    else:
                        for _ in range(max_steps):
                            action = wrapper.action_space.sample()
                            obs, reward, terminated, truncated, info = wrapper.step(action)
                            if _coerce_bool(terminated) or _coerce_bool(truncated) or bool(_coerce_bool(info.get("success"))):
                                break
                except Exception as e:
                    wrapper.error = f"{type(e).__name__}: {e}"
                wrapper.finalize_episode()
    finally:
        if created_env:
            try:
                env_for_variant.close()
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robofactory/rf_table_lift_barrier_smoke.json")
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--run_name", default="robofactory_domainB_smoke")
    args = ap.parse_args()

    base_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    # Set CUDA_VISIBLE_DEVICES early to avoid initializing all GPUs via torch warnings.
    cuda_cfg = base_cfg.get("cuda_visible_devices")
    if cuda_cfg is not None:
        try:
            from experiments.common.gpu import resolve_cuda_visible_devices

            max_mem_used_mb = int(base_cfg.get("gpu_auto_max_mem_used_mb", 1024))
            max_util_pct = int(base_cfg.get("gpu_auto_max_util_pct", 10))
            resolved = resolve_cuda_visible_devices(cuda_cfg, max_mem_used_mb=max_mem_used_mb, max_util_pct=max_util_pct)
        except Exception:
            resolved = str(cuda_cfg)

        if resolved:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(resolved)
            base_cfg["cuda_visible_devices_resolved"] = str(resolved)

    # Optional: pin HuggingFace cache to the project/shared run dir to avoid downloads.
    # This matters for WS10 (OpenVLA/Pi0), where base models are large.
    hf_home_cfg = base_cfg.get("hf_home")
    if hf_home_cfg is not None:
        hf_home = _resolve_path(str(hf_home_cfg))
    else:
        rf_cfg_probe = dict(base_cfg.get("robofactory", {}) or {})
        run_dir_probe = rf_cfg_probe.get("run_dir")
        hf_home = None
        if run_dir_probe is not None:
            run_dir_abs = _resolve_path(str(run_dir_probe))
            policy_probe = str(base_cfg.get("policy", "")).strip().lower()
            # Pi0 often uses its own cache dir to avoid conflicting dependencies.
            preferred = "hf_cache_pi0" if policy_probe == "openmarl_pi0" else "hf_cache"
            candidates = [preferred, "hf_cache"]
            for sub in candidates:
                cand = Path(run_dir_abs) / sub
                if cand.exists():
                    hf_home = str(cand.resolve())
                    break

    if hf_home:
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HF_HUB_CACHE", str(Path(hf_home) / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "transformers"))

    ctx = RunContext.create(args.runs_root, args.run_name, base_cfg)
    env = _load_env(dict(base_cfg.get("robofactory", {}) or {}))

    for variant in _expand_variants(base_cfg):
        run_variant(ctx, env, variant, base_cfg)

    try:
        env.close()
    except Exception:
        pass

    if bool(base_cfg.get("record_video", False)):
        demo_root = base_cfg.get("demo_output_dir")
        if demo_root is None:
            demo_root = str(_PROJ_ROOT / "artifacts" / "demos" / "robofactory" / ctx.run_id)
        demo_dir = Path(str(demo_root))
        demo_dir.mkdir(parents=True, exist_ok=True)
        mp4s = sorted(demo_dir.glob("*.mp4"))

        screenshots: Dict[str, List[str]] = {}
        if bool(base_cfg.get("extract_screenshots", True)) and mp4s:
            try:
                from experiments.common.video_screenshots import extract_keyframes

                max_videos = int(base_cfg.get("screenshot_max_videos", 16))
                num_frames = int(base_cfg.get("screenshot_num_frames", 3))
                screenshots_root = demo_dir / "screenshots"
                for p in mp4s[: max(0, max_videos)]:
                    out_dir = screenshots_root / p.stem
                    try:
                        shots = extract_keyframes(video_path=p, output_dir=out_dir, num_frames=num_frames)
                        if shots:
                            screenshots[p.name] = [str(s.relative_to(demo_dir)) for s in shots]
                    except Exception:
                        # Never fail the run due to screenshot extraction issues.
                        continue
            except Exception:
                pass

        index_lines: List[str] = []
        index_lines.append(f"# RoboFactory demo index: `{ctx.run_id}`\n\n")
        index_lines.append(f"- Run dir: `{ctx.run_dir}`\n")
        if mp4s:
            index_lines.append("- Videos:\n")
            for p in mp4s:
                index_lines.append(f"  - `{p.name}`\n")
                if screenshots.get(p.name):
                    index_lines.append("    - Screenshots:\n")
                    for rel in screenshots[p.name]:
                        index_lines.append(f"      - `{rel}`\n")
        else:
            index_lines.append("- Videos: (none found; check recorder settings)\n")
        (demo_dir / "INDEX.md").write_text("".join(index_lines), encoding="utf-8")
        try:
            ctx.log_phase(phase="demo_index", demo_dir=str(demo_dir), index_path=str(demo_dir / "INDEX.md"))
        except Exception:
            pass

    ctx.write_summary(
        {
            "note": "robofactory runner (tokens are word-count proxy; latency can be simulated via token->ms model; optional per-episode time budget)",
            "record_video": bool(base_cfg.get("record_video", False)),
        }
    )
    print(ctx.run_dir)


if __name__ == "__main__":
    main()
