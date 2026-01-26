from __future__ import annotations

import importlib.util
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class OpenMARLConfig:
    policy: str
    device: str
    checkpoint_tag: str
    topp_dt: float
    topp_subsample: int
    no_topp: bool
    max_env_steps: int


_OPENVLA_CACHE: Dict[Tuple[str, int, str, str], List[Any]] = {}
_PI0_CACHE: Dict[Tuple[str, int, str, str], List[Any]] = {}
_DP_CACHE: Dict[Tuple[str, int, str, str, int, int], List[Any]] = {}


def _as_numpy(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import numpy as np

        return np.asarray(x)
    except Exception:
        return x


def _resolve_openmarl_cfg(base_cfg: Dict[str, Any], policy: str) -> OpenMARLConfig:
    checkpoint_tag = str(base_cfg.get("openmarl_checkpoint_tag", "best")).strip() or "best"
    device = str(base_cfg.get("openmarl_device", "cuda:0")).strip() or "cuda:0"
    topp_dt = float(base_cfg.get("openmarl_topp_dt", 0.05))
    topp_subsample = int(base_cfg.get("openmarl_topp_subsample", 10))
    topp_subsample = max(1, topp_subsample)
    no_topp = bool(base_cfg.get("openmarl_no_topp", False))

    # Use hard_step_limit if present; otherwise fall back to max_steps (env-steps).
    max_env_steps = int(base_cfg.get("hard_step_limit", 0) or 0)
    if max_env_steps <= 0:
        max_env_steps = int(base_cfg.get("max_steps", 0) or 0)
    if max_env_steps <= 0:
        max_env_steps = 30000

    return OpenMARLConfig(
        policy=str(policy),
        device=device,
        checkpoint_tag=checkpoint_tag,
        topp_dt=topp_dt,
        topp_subsample=topp_subsample,
        no_topp=no_topp,
        max_env_steps=max_env_steps,
    )


def _detect_multi_agent(env: Any) -> Tuple[bool, int]:
    env_unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    is_multi_agent = hasattr(getattr(env_unwrapped, "agent", None), "agents")
    if is_multi_agent:
        agent_num = len(env_unwrapped.agent.agents)
    else:
        agent_num = 1
    return bool(is_multi_agent), int(agent_num)


def _build_base_pose(env: Any, *, is_multi_agent: bool) -> List[Any]:
    env_unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    if is_multi_agent:
        agents_list = env_unwrapped.agent.agents
    else:
        agents_list = [env_unwrapped.agent]
    return [agent.robot.pose for agent in agents_list]


def _get_qpos(obs: Dict[str, Any], *, agent_id: int, is_multi_agent: bool) -> Any:
    if is_multi_agent:
        key = f"panda-{agent_id}"
        qpos = obs["agent"][key]["qpos"]
    else:
        qpos = obs["agent"]["qpos"]
    qpos = _as_numpy(qpos)
    try:
        qpos = qpos.squeeze(0)
    except Exception:
        pass
    return qpos


def _get_head_cam_hwc(obs: Dict[str, Any], *, agent_id: int) -> Any:
    cam_key = f"head_camera_agent{agent_id}"
    rgb = obs["sensor_data"][cam_key]["rgb"]
    rgb = _as_numpy(rgb)
    try:
        rgb = rgb.squeeze(0)
    except Exception:
        pass
    # Many RoboFactory configs produce CHW; normalize to HWC for OpenVLA.
    try:
        if len(rgb.shape) == 3 and int(rgb.shape[0]) == 3:
            import numpy as np

            rgb = np.transpose(rgb, (1, 2, 0))
    except Exception:
        pass
    return rgb


def _subsample_indices(n: int, stride: int) -> List[int]:
    if n <= 0:
        return [0]
    stride = max(1, int(stride))
    idxs = list(range(0, int(n), int(stride)))
    if idxs and idxs[-1] != int(n) - 1:
        idxs.append(int(n) - 1)
    if not idxs:
        idxs = [int(n) - 1]
    return idxs


def _topp_path_to_actions(
    *,
    planner: Any,
    agent_id: int,
    start_qpos_7: Any,
    target_qpos_7: Any,
    gripper_cmd: float,
    dt: float,
    subsample: int,
) -> List[Any]:
    import numpy as np

    path = np.vstack((np.asarray(start_qpos_7, dtype=float), np.asarray(target_qpos_7, dtype=float)))
    try:
        _times, position, _right_vel, _acc, _duration = planner.planner[agent_id].TOPP(path, float(dt), verbose=False)
        position = np.asarray(position, dtype=float)
        idxs = _subsample_indices(int(position.shape[0]), int(subsample))
        return [np.hstack([position[i], float(gripper_cmd)]) for i in idxs]
    except Exception:
        # Fallback: single-step jump (still keeps pipeline running).
        return [np.hstack([np.asarray(start_qpos_7, dtype=float), float(gripper_cmd)])]


def _phase_policy_call(ctx, *, phase: str, episode_id: str, variant: str, t: int, model: str, lat_ms: float, **extra):
    ctx.log_phase(
        phase=str(phase),
        episode_id=str(episode_id),
        variant=str(variant),
        t=int(t),
        vlm_model=str(model),
        lat_vlm_ms=float(lat_ms),
        lat_total_ms=float(lat_ms),
        **extra,
    )


def _load_openvla_policy(*, env_id: str, agent_id: int, device: str, checkpoint_tag: str):
    from robofactory.policy.OpenVLA.openvla_policy.policy.openvla_policy import OpenVLAPolicy

    run_dir = os.environ.get("ROBOFACTORY_RUN_DIR", "").strip()
    if not run_dir:
        raise RuntimeError("ROBOFACTORY_RUN_DIR must be set for OpenMARL checkpoints.")
    ckpt_root = Path(run_dir) / "checkpoints" / "openvla" / f"{env_id}_Agent{agent_id}"
    ckpt_dir = ckpt_root / checkpoint_tag
    if not ckpt_dir.exists():
        # Fallbacks commonly present in OpenMARL__run.
        for cand in (ckpt_root / "best", ckpt_root / "final", ckpt_root / "epoch_1"):
            if cand.exists():
                ckpt_dir = cand
                break
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"OpenVLA checkpoint dir not found: {ckpt_dir}")
    return OpenVLAPolicy(checkpoint_path=str(ckpt_dir), device=str(device))


def _openvla_model_input(obs: Dict[str, Any], *, agent_id: int, agent_pos_8: Any) -> Dict[str, Any]:
    return {"image": _get_head_cam_hwc(obs, agent_id=agent_id), "agent_pos": _as_numpy(agent_pos_8)}


def rollout_openmarl_openvla(
    wrapper: Any,
    *,
    ctx: Any,
    base_cfg: Dict[str, Any],
    variant: Dict[str, Any],
) -> None:
    cfg = _resolve_openmarl_cfg(base_cfg, "openmarl_openvla")
    env_id = str(getattr(wrapper, "env_id", base_cfg.get("robofactory", {}).get("env_id", "")))
    is_multi_agent, agent_num = _detect_multi_agent(wrapper)

    from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver
    from robofactory.policy.shared.task_instructions import get_task_instruction

    base_pose = _build_base_pose(wrapper, is_multi_agent=is_multi_agent)
    planner = PandaArmMotionPlanningSolver(
        wrapper,
        debug=False,
        vis=False,
        base_pose=base_pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        is_multi_agent=is_multi_agent,
    )
    planner.gripper_state = [1.0] * int(agent_num)

    # Load per-agent policies (cached to avoid re-loading huge base weights per episode/variant).
    cache_key = (env_id, int(agent_num), str(cfg.device), str(cfg.checkpoint_tag))
    policies = _OPENVLA_CACHE.get(cache_key)
    if policies is None:
        policies = [
            _load_openvla_policy(env_id=env_id, agent_id=i, device=cfg.device, checkpoint_tag=cfg.checkpoint_tag)
            for i in range(agent_num)
        ]
        _OPENVLA_CACHE[cache_key] = policies
    instruction = get_task_instruction(env_id, policy_type="openvla")

    steps = 0
    while steps < int(cfg.max_env_steps):
        # Predict target action per agent (one inference per agent).
        obs_now = wrapper.get_obs() if hasattr(wrapper, "get_obs") else getattr(wrapper, "obs", None)
        if obs_now is None:
            obs_now = wrapper.obs
        action_targets: List[Any] = []
        for agent_id in range(agent_num):
            qpos = _get_qpos(obs_now, agent_id=agent_id, is_multi_agent=is_multi_agent)
            qpos_7 = _as_numpy(qpos)[:-2]
            import numpy as np

            agent_pos_8 = np.append(qpos_7, float(planner.gripper_state[agent_id]))
            model_in = _openvla_model_input(obs_now, agent_id=agent_id, agent_pos_8=agent_pos_8)

            t0 = time.time()
            act = policies[agent_id].predict(model_in, instruction)
            lat_ms = (time.time() - t0) * 1000.0
            _phase_policy_call(
                ctx,
                phase="vla_policy_call",
                episode_id=str(wrapper.episode_id),
                variant=str(variant.get("name")),
                t=int(getattr(wrapper, "t", 0)),
                model=f"openvla_agent{agent_id}",
                lat_ms=float(lat_ms),
            )

            action_targets.append(_as_numpy(act).reshape(-1))

        # Execute (with optional TOPP subsampling).
        for agent_id in range(agent_num):
            target = action_targets[agent_id]
            planner.gripper_state[agent_id] = float(target[-1])

        # Build per-agent action trajectories.
        per_agent_actions: List[List[Any]] = []
        for agent_id in range(agent_num):
            obs_now = wrapper.get_obs() if hasattr(wrapper, "get_obs") else getattr(wrapper, "obs", None)
            qpos = _get_qpos(obs_now, agent_id=agent_id, is_multi_agent=is_multi_agent)
            qpos_7 = _as_numpy(qpos)[:-2]
            target = action_targets[agent_id]
            tgt_qpos_7 = _as_numpy(target)[:-1]
            gr = float(target[-1])
            if cfg.no_topp:
                import numpy as np

                per_agent_actions.append([np.hstack([tgt_qpos_7, gr])])
            else:
                per_agent_actions.append(
                    _topp_path_to_actions(
                        planner=planner,
                        agent_id=agent_id,
                        start_qpos_7=qpos_7,
                        target_qpos_7=tgt_qpos_7,
                        gripper_cmd=gr,
                        dt=cfg.topp_dt,
                        subsample=cfg.topp_subsample,
                    )
                )

        # Execute synchronized steps (pad shorter sequences by repeating last action).
        max_len = max(len(seq) for seq in per_agent_actions) if per_agent_actions else 0
        for k in range(max_len):
            if steps >= int(cfg.max_env_steps):
                break
            if is_multi_agent:
                act_dict: Dict[str, Any] = {}
                for agent_id in range(agent_num):
                    seq = per_agent_actions[agent_id]
                    a = seq[min(k, len(seq) - 1)]
                    act_dict[f"panda-{agent_id}"] = a
                _obs, _rew, terminated, truncated, info = wrapper.step(act_dict)
            else:
                a0 = per_agent_actions[0][min(k, len(per_agent_actions[0]) - 1)]
                _obs, _rew, terminated, truncated, info = wrapper.step(a0)

            steps += 1
            success = bool(info.get("success", False)) if isinstance(info, dict) else False
            if success or bool(terminated) or bool(truncated):
                return


def rollout_openmarl_pi0(
    wrapper: Any,
    *,
    ctx: Any,
    base_cfg: Dict[str, Any],
    variant: Dict[str, Any],
) -> None:
    cfg = _resolve_openmarl_cfg(base_cfg, "openmarl_pi0")
    env_id = str(getattr(wrapper, "env_id", base_cfg.get("robofactory", {}).get("env_id", "")))
    is_multi_agent, agent_num = _detect_multi_agent(wrapper)

    from robofactory.policy.Pi0.pi0_policy.policy.pi0_policy import Pi0Policy
    from robofactory.policy.shared.task_instructions import get_task_instruction

    run_dir = os.environ.get("ROBOFACTORY_RUN_DIR", "").strip()
    if not run_dir:
        raise RuntimeError("ROBOFACTORY_RUN_DIR must be set for OpenMARL checkpoints.")

    # Load per-agent Pi0 policies (action chunking is handled by the policy itself: it predicts a horizon).
    cache_key = (env_id, int(agent_num), str(cfg.device), str(cfg.checkpoint_tag))
    policies = _PI0_CACHE.get(cache_key)
    if policies is None:
        import robofactory  # type: ignore

        rf_root = Path(robofactory.__file__).resolve().parent
        pi0_cfg = rf_root / "policy" / "Pi0" / "pi0_policy" / "config" / "robot_pi0.yaml"
        if not pi0_cfg.exists():
            # Fallback: some packaging layouts may differ.
            pi0_cfg = rf_root / "policy" / "Pi0" / "pi0_policy" / "config" / "robot_pi0.yml"
        if not pi0_cfg.exists():
            raise FileNotFoundError(f"Pi0 config YAML not found under robofactory package: {pi0_cfg}")

        loaded: List[Pi0Policy] = []
        for agent_id in range(agent_num):
            ckpt_root = Path(run_dir) / "checkpoints" / "pi0" / f"{env_id}_Agent{agent_id}"
            ckpt_dir = ckpt_root / cfg.checkpoint_tag
            if not ckpt_dir.exists():
                for cand in (ckpt_root / "best", ckpt_root / "latest", ckpt_root / "epoch_5000"):
                    if cand.exists():
                        ckpt_dir = cand
                        break
            if not ckpt_dir.exists():
                raise FileNotFoundError(f"Pi0 checkpoint dir not found: {ckpt_dir}")
            loaded.append(
                Pi0Policy.from_checkpoint(
                    checkpoint_path=str(ckpt_dir),
                    config_path=str(pi0_cfg),
                    device=str(cfg.device),
                    task_name=env_id,
                )
            )
        policies = loaded
        _PI0_CACHE[cache_key] = policies

    instruction = get_task_instruction(env_id, policy_type="pi0")
    for p in policies:
        try:
            p.set_instruction(str(instruction))
        except Exception:
            pass

    def model_input(obs: Dict[str, Any], *, agent_id: int, agent_pos_8: Any) -> Dict[str, Any]:
        import numpy as np

        sensor_data = obs["sensor_data"]
        images = []
        head_key = f"head_camera_agent{agent_id}"
        if head_key not in sensor_data:
            raise KeyError(f"Missing camera {head_key} in obs['sensor_data']")
        img0 = _as_numpy(sensor_data[head_key]["rgb"]).squeeze(0)
        if len(img0.shape) == 3 and int(img0.shape[0]) == 3:
            img0 = np.transpose(img0, (1, 2, 0))
        images.append(img0)

        global_key = "head_camera_global"
        if global_key in sensor_data:
            img1 = _as_numpy(sensor_data[global_key]["rgb"]).squeeze(0)
            if len(img1.shape) == 3 and int(img1.shape[0]) == 3:
                img1 = np.transpose(img1, (1, 2, 0))
        else:
            img1 = images[0].copy()
        images.append(img1)

        wrist_key = f"wrist_camera_agent{agent_id}"
        if wrist_key in sensor_data:
            img2 = _as_numpy(sensor_data[wrist_key]["rgb"]).squeeze(0)
            if len(img2.shape) == 3 and int(img2.shape[0]) == 3:
                img2 = np.transpose(img2, (1, 2, 0))
        else:
            img2 = images[0].copy()
        images.append(img2)

        return {"images": np.asarray(images), "state": _as_numpy(agent_pos_8)}

    # Chunking policy: take the first action each cycle; repeat a small number of env steps.
    action_repeat = int(base_cfg.get("openmarl_action_repeat", 2))
    action_repeat = max(1, action_repeat)
    gripper_state = [1.0] * int(agent_num)

    steps = 0
    while steps < int(cfg.max_env_steps):
        obs_now = wrapper.get_obs() if hasattr(wrapper, "get_obs") else getattr(wrapper, "obs", None)
        if obs_now is None:
            obs_now = wrapper.obs

        per_agent_action: List[Any] = []
        for agent_id in range(agent_num):
            qpos = _get_qpos(obs_now, agent_id=agent_id, is_multi_agent=is_multi_agent)
            import numpy as np

            qpos_7 = _as_numpy(qpos)[:-2]
            agent_pos_8 = np.append(qpos_7, float(gripper_state[agent_id]))
            inp = model_input(obs_now, agent_id=agent_id, agent_pos_8=agent_pos_8)

            t0 = time.time()
            seq = policies[agent_id].predict_action(inp)
            lat_ms = (time.time() - t0) * 1000.0
            _phase_policy_call(
                ctx,
                phase="vla_policy_call",
                episode_id=str(wrapper.episode_id),
                variant=str(variant.get("name")),
                t=int(getattr(wrapper, "t", 0)),
                model=f"pi0_agent{agent_id}",
                lat_ms=float(lat_ms),
            )

            seq = _as_numpy(seq)
            if seq.ndim == 2 and seq.shape[0] > 0:
                per_agent_action.append(seq[0])
            else:
                per_agent_action.append(seq.reshape(-1))

        for _ in range(action_repeat):
            if steps >= int(cfg.max_env_steps):
                break
            if is_multi_agent:
                act_dict = {f"panda-{i}": per_agent_action[i] for i in range(agent_num)}
                _obs, _rew, terminated, truncated, info = wrapper.step(act_dict)
                for i in range(agent_num):
                    try:
                        gripper_state[i] = float(_as_numpy(per_agent_action[i]).reshape(-1)[-1])
                    except Exception:
                        pass
            else:
                _obs, _rew, terminated, truncated, info = wrapper.step(per_agent_action[0])
                try:
                    gripper_state[0] = float(_as_numpy(per_agent_action[0]).reshape(-1)[-1])
                except Exception:
                    pass
            steps += 1
            success = bool(info.get("success", False)) if isinstance(info, dict) else False
            if success or bool(terminated) or bool(truncated):
                return


def rollout_openmarl_dp(
    wrapper: Any,
    *,
    ctx: Any,
    base_cfg: Dict[str, Any],
    variant: Dict[str, Any],
) -> None:
    """Diffusion-Policy rollout via dynamic import of OpenMARL eval module (directory contains a hyphen)."""
    cfg = _resolve_openmarl_cfg(base_cfg, "openmarl_dp")
    env_id = str(getattr(wrapper, "env_id", base_cfg.get("robofactory", {}).get("env_id", "")))
    is_multi_agent, agent_num = _detect_multi_agent(wrapper)

    # Locate eval_multi_dp.py inside OpenMARL robofactory package.
    import robofactory  # type: ignore

    rf_root = Path(robofactory.__file__).resolve().parent
    eval_path = rf_root / "policy" / "Diffusion-Policy" / "eval_multi_dp.py"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing OpenMARL DP evaluator at: {eval_path}")

    spec = importlib.util.spec_from_file_location("_openmarl_eval_multi_dp", str(eval_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for: {eval_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[misc]

    DP = getattr(mod, "DP")

    def dp_model_input(obs: Dict[str, Any], agent_pos_8: Any, agent_id: int) -> Dict[str, Any]:
        import numpy as np

        cam_key = f"head_camera_agent{agent_id}"
        rgb = _as_numpy(obs["sensor_data"][cam_key]["rgb"])
        try:
            rgb = rgb.squeeze(0)
        except Exception:
            pass

        # Robustly handle either HWC or CHW.
        if isinstance(rgb, np.ndarray) and rgb.ndim == 3 and int(rgb.shape[0]) == 3:
            chw = rgb.astype(np.float32)
        else:
            if isinstance(rgb, np.ndarray) and rgb.ndim == 3 and int(rgb.shape[-1]) == 3:
                chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
            else:
                chw = np.asarray(rgb, dtype=np.float32)

        if float(chw.max()) > 1.0:
            chw = chw / 255.0
        return {"head_cam": chw, "agent_pos": _as_numpy(agent_pos_8)}

    # DP checkpoints: default to best if checkpoint_num<=0.
    checkpoint_num = int(base_cfg.get("openmarl_dp_checkpoint_num", -1))
    data_num = int(base_cfg.get("openmarl_dp_data_num", 50))

    from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

    base_pose = _build_base_pose(wrapper, is_multi_agent=is_multi_agent)
    planner = PandaArmMotionPlanningSolver(
        wrapper,
        debug=False,
        vis=False,
        base_pose=base_pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        is_multi_agent=is_multi_agent,
    )
    planner.gripper_state = [1.0] * int(agent_num)

    cache_key = (env_id, int(agent_num), str(cfg.device), str(cfg.checkpoint_tag), int(checkpoint_num), int(data_num))
    models = _DP_CACHE.get(cache_key)
    if models is None:
        models = [DP(env_id, checkpoint_num=checkpoint_num, data_num=data_num, id=i, device=cfg.device) for i in range(agent_num)]
        _DP_CACHE[cache_key] = models

    steps = 0
    while steps < int(cfg.max_env_steps):
        obs_now = wrapper.get_obs() if hasattr(wrapper, "get_obs") else getattr(wrapper, "obs", None)
        if obs_now is None:
            obs_now = wrapper.obs

        action_lists: List[Any] = []
        for agent_id in range(agent_num):
            qpos = _get_qpos(obs_now, agent_id=agent_id, is_multi_agent=is_multi_agent)
            import numpy as np

            qpos_7 = _as_numpy(qpos)[:-2]
            agent_pos_8 = np.append(qpos_7, float(planner.gripper_state[agent_id]))
            model_in = dp_model_input(obs_now, agent_pos_8, agent_id)
            models[agent_id].update_obs(model_in)

            t0 = time.time()
            action_list = models[agent_id].get_action()
            lat_ms = (time.time() - t0) * 1000.0
            _phase_policy_call(
                ctx,
                phase="vla_policy_call",
                episode_id=str(wrapper.episode_id),
                variant=str(variant.get("name")),
                t=int(getattr(wrapper, "t", 0)),
                model=f"dp_agent{agent_id}",
                lat_ms=float(lat_ms),
            )
            action_lists.append(action_list)

        # DP typically returns 6 actions per policy call.
        horizon = min(len(al) for al in action_lists) if action_lists else 0
        horizon = min(int(horizon), 6) if horizon > 0 else 0
        if horizon <= 0:
            return

        for i in range(horizon):
            # Build per-agent TOPP actions for this substep.
            per_agent_actions: List[List[Any]] = []
            for agent_id in range(agent_num):
                now_action = _as_numpy(action_lists[agent_id][i]).reshape(-1)
                planner.gripper_state[agent_id] = float(now_action[-1])

                obs_now2 = wrapper.get_obs() if hasattr(wrapper, "get_obs") else getattr(wrapper, "obs", None)
                qpos2 = _get_qpos(obs_now2, agent_id=agent_id, is_multi_agent=is_multi_agent)
                qpos_7 = _as_numpy(qpos2)[:-2]
                tgt_qpos_7 = now_action[:-1]
                gr = float(now_action[-1])

                if cfg.no_topp:
                    import numpy as np

                    per_agent_actions.append([np.hstack([tgt_qpos_7, gr])])
                else:
                    per_agent_actions.append(
                        _topp_path_to_actions(
                            planner=planner,
                            agent_id=agent_id,
                            start_qpos_7=qpos_7,
                            target_qpos_7=tgt_qpos_7,
                            gripper_cmd=gr,
                            dt=cfg.topp_dt,
                            subsample=cfg.topp_subsample,
                        )
                    )

            max_len = max(len(seq) for seq in per_agent_actions)
            for k in range(max_len):
                if steps >= int(cfg.max_env_steps):
                    return
                if is_multi_agent:
                    act_dict = {f"panda-{aid}": per_agent_actions[aid][min(k, len(per_agent_actions[aid]) - 1)] for aid in range(agent_num)}
                    _obs, _rew, terminated, truncated, info = wrapper.step(act_dict)
                else:
                    a0 = per_agent_actions[0][min(k, len(per_agent_actions[0]) - 1)]
                    _obs, _rew, terminated, truncated, info = wrapper.step(a0)
                steps += 1
                success = bool(info.get("success", False)) if isinstance(info, dict) else False
                if success or bool(terminated) or bool(truncated):
                    return


def rollout_openmarl_policy(
    wrapper: Any,
    *,
    ctx: Any,
    base_cfg: Dict[str, Any],
    variant: Dict[str, Any],
    policy: str,
) -> None:
    policy = str(policy).strip().lower()
    if policy == "openmarl_openvla":
        rollout_openmarl_openvla(wrapper, ctx=ctx, base_cfg=base_cfg, variant=variant)
        return
    if policy == "openmarl_pi0":
        rollout_openmarl_pi0(wrapper, ctx=ctx, base_cfg=base_cfg, variant=variant)
        return
    if policy == "openmarl_dp":
        rollout_openmarl_dp(wrapper, ctx=ctx, base_cfg=base_cfg, variant=variant)
        return
    raise ValueError(f"Unknown OpenMARL policy: {policy}")
