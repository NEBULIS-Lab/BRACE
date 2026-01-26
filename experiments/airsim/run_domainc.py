from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import random
import re
import signal
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgpackrpc

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from brace.controller import BraceController, BraceHyperparams, BraceState
from experiments.common import replan_schedule as rs
from experiments.common.context_compress.baselines import extra_overhead_ms, normalize_method
from experiments.common.logging import RunContext

from experiments.airsim.airsim_bridge.import_airsim import import_airsim
from experiments.airsim.airsim_bridge.overlay import OverlayLine, decode_png, draw_overlay_panel, encode_png
from experiments.airsim.airsim_bridge.paths import settings_json_path
from experiments.airsim.airsim_bridge.video import (
    build_grid_video,
    build_side_by_side_video,
    build_video_from_frames,
    probe_video,
)


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _try_run(cmd: List[str], *, timeout_s: float = 5.0) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=float(timeout_s))
    except Exception:
        return None
    return out.strip()


def _select_idle_gpu(*, gpu_override: Optional[int] = None) -> int:
    if gpu_override is not None:
        return int(gpu_override)

    if shutil.which("nvidia-smi") is None:
        return 0

    gpu_info = _try_run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=5.0,
    )
    if not gpu_info:
        # Some clusters have a broken / hanging `nvidia-smi` even though GPUs exist.
        # Fall back to "highest index GPU" to avoid defaulting to GPU0 (often the busiest).
        devs = sorted(Path("/dev").glob("nvidia[0-9]*"))
        idxs: List[int] = []
        for p in devs:
            try:
                idxs.append(int(p.name.replace("nvidia", "")))
            except Exception:
                continue
        return int(max(idxs)) if idxs else 0

    busy_uuids_raw = _try_run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
        timeout_s=5.0,
    )
    busy_uuids = {l.strip() for l in (busy_uuids_raw or "").splitlines() if l.strip()}

    idle = []
    idle_not_busy = []
    for line in gpu_info.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            uuid = str(parts[1])
            util = int(parts[2])
            mem = int(parts[3])
        except Exception:
            continue
        if util >= 10:
            continue
        if mem >= 500:
            continue
        idle.append(idx)
        if uuid not in busy_uuids:
            idle_not_busy.append(idx)

    if idle_not_busy:
        return int(max(idle_not_busy))
    if idle:
        return int(max(idle))
    return 0


def _lsof_listen_pids(port: int) -> List[int]:
    if shutil.which("lsof") is None:
        return []
    out = _try_run(["lsof", "-nP", f"-iTCP:{int(port)}", "-sTCP:LISTEN", "-t"], timeout_s=3.0)
    if not out:
        return []
    pids = []
    for tok in out.split():
        try:
            pids.append(int(tok))
        except Exception:
            continue
    return pids


def _pid_cmdline(pid: int) -> str:
    try:
        cmdline = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        if cmdline:
            return cmdline.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except Exception:
        pass
    out = _try_run(["ps", "-p", str(int(pid)), "-o", "args="], timeout_s=2.0)
    return out or ""


def _kill_pid(pid: int, *, grace_s: float = 2.0) -> None:
    try:
        os.kill(int(pid), signal.SIGTERM)
    except Exception:
        return

    deadline = time.time() + float(grace_s)
    while time.time() < deadline:
        if not Path(f"/proc/{int(pid)}").exists():
            return
        time.sleep(0.1)
    try:
        os.kill(int(pid), signal.SIGKILL)
    except Exception:
        return


def _ensure_rpc_free(*, port: int, allowed_cmdline_substrings: List[str], ctx: RunContext) -> None:
    pids = _lsof_listen_pids(int(port))
    if not pids:
        return

    ctx.append_event({"phase": "rpc_busy", "port": int(port), "pids": [int(p) for p in pids]})
    killed = []
    refused: List[Dict[str, Any]] = []
    for pid in pids:
        cmd = _pid_cmdline(pid)
        allowed = any(sub and sub in cmd for sub in allowed_cmdline_substrings)
        if not allowed:
            refused.append({"pid": int(pid), "cmd": cmd})
            continue
        _kill_pid(pid)
        killed.append({"pid": int(pid), "cmd": cmd})

    if refused:
        raise RuntimeError(
            "AirSim RPC port is in use by a non-Airsim/unknown process; refusing to kill it. "
            f"port={int(port)} refused={refused}"
        )

    ctx.append_event({"phase": "rpc_killed", "port": int(port), "killed": killed})

    for _ in range(60):
        if not _lsof_listen_pids(int(port)):
            return
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for port {int(port)} to become free after killing env process(es).")


def _canonical_env_name(name: str) -> str:
    n = str(name or "").strip().lower()
    n = n.replace("-", "").replace("_", "")
    if "zhang" in n or "jiajie" in n:
        return "zhangjiajie"
    if n in {"airsimnh", "nh"}:
        return "airsimnh"
    if n in {"blocks"}:
        return "blocks"
    if n in {"abandonedpark", "abandoned"}:
        return "abandonedpark"
    if n in {"landscapemountains", "landscape", "mountains"}:
        return "landscapemountains"
    return n


def _resolve_env_bin(*, env_name: str, envs_root: Path, env_bin_override: Optional[str]) -> Path:
    if env_bin_override:
        return Path(str(env_bin_override)).expanduser().resolve()

    env_name = _canonical_env_name(env_name)
    if env_name == "zhangjiajie":
        raise ValueError("ZhangJiajie is explicitly disabled for this demo run.")

    mapping = {
        "airsimnh": envs_root / "AirSimNH" / "AirSimNH" / "LinuxNoEditor" / "AirSimNH.sh",
        "blocks": envs_root / "Blocks" / "LinuxBlocks1.8.1" / "LinuxNoEditor" / "Blocks.sh",
        "abandonedpark": envs_root / "AbandonedPark" / "LinuxNoEditor" / "AbandonedPark.sh",
        "landscapemountains": envs_root / "LandscapeMountains" / "LinuxNoEditor" / "LandscapeMountains.sh",
    }
    if env_name not in mapping:
        raise ValueError(
            f"Unknown UE env '{env_name}' (supported: airsimnh|blocks|abandonedpark|landscapemountains; zhangjiajie disabled)."
        )
    return mapping[env_name].resolve()


@dataclass
class UEEnvHandle:
    env_name: str
    env_bin: Path
    settings_path: Path
    log_path: Path
    gpu: int
    proc: Any

    def stop(self, *, ctx: Optional[RunContext] = None) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is not None:
                return
        except Exception:
            pass
        try:
            os.killpg(int(self.proc.pid), signal.SIGTERM)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass
        deadline = time.time() + 8.0
        while time.time() < deadline:
            try:
                if self.proc.poll() is not None:
                    if ctx is not None:
                        ctx.append_event({"phase": "ue_stopped", "env": self.env_name, "gpu": int(self.gpu)})
                    return
            except Exception:
                break
            time.sleep(0.2)
        try:
            os.killpg(int(self.proc.pid), signal.SIGKILL)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        if ctx is not None:
            ctx.append_event({"phase": "ue_stopped_force", "env": self.env_name, "gpu": int(self.gpu)})


def _launch_ue_env(
    *,
    env_name: str,
    env_bin: Path,
    settings_path: Path,
    log_path: Path,
    ip: str,
    port: int,
    timeout_s: int,
    gpu: int,
    res_x: int,
    res_y: int,
    allowed_cmdline_substrings: List[str],
    ctx: RunContext,
) -> UEEnvHandle:
    if not env_bin.exists():
        raise FileNotFoundError(f"UE env binary not found: {env_bin}")
    if not os.access(str(env_bin), os.X_OK):
        raise PermissionError(f"UE env binary is not executable: {env_bin}")

    _ensure_rpc_free(port=int(port), allowed_cmdline_substrings=allowed_cmdline_substrings, ctx=ctx)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(env_bin),
        f"-settings={str(settings_path)}",
        "-windowed",
        f"-ResX={int(res_x)}",
        f"-ResY={int(res_y)}",
        "-NoSound",
        "-RenderOffScreen",
        f"-graphicsadapter={int(gpu)}",
    ]
    ctx.append_event(
        {
            "phase": "ue_launch",
            "env": str(env_name),
            "env_bin": str(env_bin),
            "settings_path": str(settings_path),
            "log_path": str(log_path),
            "gpu": int(gpu),
            "res": [int(res_x), int(res_y)],
            "cmd": cmd,
        }
    )

    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, start_new_session=True)

    handle = UEEnvHandle(
        env_name=str(env_name),
        env_bin=env_bin,
        settings_path=settings_path,
        log_path=log_path,
        gpu=int(gpu),
        proc=proc,
    )

    # Wait for RPC to become ready (TCP + msgpack handshake).
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            with socket.create_connection((str(ip), int(port)), timeout=1.0):
                ctx.append_event({"phase": "rpc_tcp_ready", "ip": str(ip), "port": int(port)})
                break
        except OSError:
            time.sleep(0.5)

    # Some environments take longer to fully initialize AirSim even after TCP is listening.
    time.sleep(1.5)
    return handle


def _wait_for_multirotor_client(
    airsim,
    *,
    ip: str,
    port: int,
    rpc_wait_s: int,
    connect_timeout_s: int,
    ctx: RunContext,
):
    last_exc: Optional[Exception] = None
    deadline = time.time() + float(rpc_wait_s)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            client = airsim.MultirotorClient(ip=str(ip), port=int(port), timeout_value=int(connect_timeout_s))
            client.confirmConnection()
            ctx.append_event({"phase": "rpc_ready", "ip": str(ip), "port": int(port), "attempt": int(attempt)})
            return client
        except Exception as e:
            last_exc = e
            time.sleep(1.0)
    raise RuntimeError(
        f"Timed out waiting for AirSim RPC. ip={ip} port={int(port)} wait_s={int(rpc_wait_s)} last={type(last_exc).__name__ if last_exc else None}: {last_exc}"
    )

@dataclass(frozen=True)
class ConflictZone:
    zone_id: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains_xy(self, x: float, y: float) -> bool:
        return (self.x_min <= float(x) <= self.x_max) and (self.y_min <= float(y) <= self.y_max)

    def dist_to_xy(self, x: float, y: float) -> float:
        x = float(x)
        y = float(y)
        dx = 0.0
        if x < self.x_min:
            dx = self.x_min - x
        elif x > self.x_max:
            dx = x - self.x_max
        dy = 0.0
        if y < self.y_min:
            dy = self.y_min - y
        elif y > self.y_max:
            dy = y - self.y_max
        return math.hypot(dx, dy)


@dataclass
class ZoneLock:
    holder: Optional[str] = None
    ttl_remaining: int = 0
    holder_outside_steps: int = 0


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        expanded = _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), expanded)
        return expanded
    return value


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

    out: List[Dict[str, Any]] = []
    for brace_enabled in brace_opts:
        for pruning_enabled in prune_opts:
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
                        f"__B{token_budget}"
                    )
                    out.append(
                        {
                            "name": name,
                            "brace_enabled": bool(brace_enabled),
                            "pruning_enabled": bool(pruning_enabled),
                            "keep_ratio": float(keep_ratio),
                            "token_budget": int(token_budget),
                        }
                    )
    return out


def _apply_context_compress(
    *,
    tokens_in: int,
    tokens_protected: int,
    method: str,
    keep_ratio: float,
    token_budget: Optional[int],
) -> int:
    tokens_in = max(0, int(tokens_in))
    tokens_protected = max(0, int(tokens_protected))
    method = str(method or "none").strip().lower()

    if method == "none":
        return tokens_in

    if method == "erecap":
        keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
        prunable = max(0, tokens_in - tokens_protected)
        kept_prunable = max(1, int(round(prunable * keep_ratio))) if prunable > 0 else 0
        tokens_after = tokens_protected + kept_prunable
        if token_budget is not None and token_budget > 0:
            tokens_after = min(tokens_after, int(token_budget))
        return max(0, int(tokens_after))

    if token_budget is not None and token_budget > 0:
        return max(0, int(min(tokens_in, int(token_budget))))
    return tokens_in


def _hash_json(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _stable_int_hash(text: str, *, mod: int = 1000) -> int:
    h = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    return int(x % int(mod))


def _dist_xyz(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _dist_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _write_settings(template_path: Path, *, ctx: RunContext) -> Path:
    dst = settings_json_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        prev = Path(ctx.run_dir) / "airsim_settings_prev.json"
        try:
            shutil.copyfile(dst, prev)
        except Exception:
            pass
    dst.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
    snap = Path(ctx.run_dir) / "airsim_settings.json"
    try:
        shutil.copyfile(dst, snap)
    except Exception:
        pass
    return dst


def _write_settings_payload(payload: Dict[str, Any], *, ctx: RunContext) -> Path:
    dst = settings_json_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        prev = Path(ctx.run_dir) / "airsim_settings_prev.json"
        try:
            shutil.copyfile(dst, prev)
        except Exception:
            pass
    dst.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    snap = Path(ctx.run_dir) / "airsim_settings.json"
    try:
        shutil.copyfile(dst, snap)
    except Exception:
        pass
    return dst


def _build_multidrone_settings(
    *,
    num_drones: int,
    api_server_port: int,
    fpv_wh: Tuple[int, int] = (640, 360),
    global_wh: Tuple[int, int] = (1280, 720),
    chase_wh: Tuple[int, int] = (1280, 720),
    fpv_fov_deg: float = 80.0,
    global_fov_deg: float = 60.0,
    chase_fov_deg: float = 75.0,
    cam_global_vehicle: str = "CamGlobal",
    cam_chase_vehicle: str = "CamChase",
) -> Dict[str, Any]:
    num_drones = max(1, int(num_drones))
    vehicles: Dict[str, Any] = {}

    def drone_cameras():
        # Provide 4 per-drone cameras so we can always populate 4 "representative" views
        # even when K < 4 (K=1 uses 4 headings; K=2 uses 2 drones × 2 headings).
        cams: Dict[str, Any] = {}
        for idx, yaw in enumerate([0.0, 90.0, 180.0, 270.0]):
            cams[str(idx)] = {
                # Place camera forward & slightly up to reduce body/prop occlusion.
                "X": 2.0,
                "Y": 0.0,
                "Z": -0.5,
                "Pitch": 0.0,
                "Roll": 0.0,
                "Yaw": float(yaw),
                "CaptureSettings": [
                    {
                        "ImageType": 0,
                        "Width": int(fpv_wh[0]),
                        "Height": int(fpv_wh[1]),
                        "FOV_Degrees": float(fpv_fov_deg),
                        "Compress": True,
                    }
                ],
            }
        return cams

    for i in range(1, num_drones + 1):
        name = f"Drone{i}"
        vehicles[name] = {
            "VehicleType": "SimpleFlight",
            "AutoCreate": True,
            "Cameras": drone_cameras(),
        }

    # Global camera rig: one camera, pose controlled by the runner.
    vehicles[str(cam_global_vehicle)] = {
        # Use a normal multirotor for compatibility with these prebuilt UE environments.
        "VehicleType": "SimpleFlight",
        "AutoCreate": True,
        "Cameras": {
            "0": {
                # Put camera far enough forward to avoid the rig drone's prop arms occluding the view.
                "X": 6.0,
                "Y": 0.0,
                "Z": 0.0,
                "Pitch": 0.0,
                "Roll": 0.0,
                "Yaw": 0.0,
                "CaptureSettings": [
                    {
                        "ImageType": 0,
                        "Width": int(global_wh[0]),
                        "Height": int(global_wh[1]),
                        "FOV_Degrees": float(global_fov_deg),
                        "Compress": True,
                    }
                ],
            },
        },
    }
    # Close chase rig: one camera, pose controlled by the runner.
    vehicles[str(cam_chase_vehicle)] = {
        "VehicleType": "SimpleFlight",
        "AutoCreate": True,
        "Cameras": {
            "0": {
                "X": 6.0,
                "Y": 0.0,
                "Z": 0.0,
                "Pitch": 0.0,
                "Roll": 0.0,
                "Yaw": 0.0,
                "CaptureSettings": [
                    {
                        "ImageType": 0,
                        "Width": int(chase_wh[0]),
                        "Height": int(chase_wh[1]),
                        "FOV_Degrees": float(chase_fov_deg),
                        "Compress": True,
                    }
                ],
            }
        },
    }

    return {
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "ViewMode": "NoDisplay",
        "ClockSpeed": 1.0,
        "LocalHostIp": "127.0.0.1",
        "ApiServerPort": int(api_server_port),
        "RpcEnabled": True,
        "Wind": {"X": 0.0, "Y": 0.0, "Z": 0.0},
        "Vehicles": vehicles,
        "Recording": {"Enabled": False},
    }


def _sim_get_images_with_retry(client, requests, *, vehicle_name: str, retries: int = 5, base_sleep_s: float = 0.2):
    for attempt in range(int(retries)):
        try:
            return client.simGetImages(requests, vehicle_name=vehicle_name)
        except msgpackrpc.error.TimeoutError:
            if attempt >= int(retries) - 1:
                return None
            time.sleep(float(base_sleep_s) * float(attempt + 1))
        except Exception:
            if attempt >= int(retries) - 1:
                return None
            time.sleep(float(base_sleep_s) * float(attempt + 1))


def _capture_rgb_png(
    airsim,
    client,
    *,
    vehicle_name: str,
    camera: str,
    retries: int = 5,
    base_sleep_s: float = 0.2,
):
    reqs = [airsim.ImageRequest(str(camera), airsim.ImageType.Scene, pixels_as_float=False, compress=True)]
    resp = _sim_get_images_with_retry(
        client,
        reqs,
        vehicle_name=str(vehicle_name),
        retries=int(retries),
        base_sleep_s=float(base_sleep_s),
    )
    if not resp or len(resp) != 1:
        return None
    return resp[0].image_data_uint8


def _overlay_png(png_bytes: bytes, lines: List[OverlayLine]) -> bytes:
    if not png_bytes:
        return png_bytes
    img = decode_png(png_bytes)
    if img is None:
        return png_bytes
    draw_overlay_panel(img, lines)
    # Re-encode via OpenCV.
    import cv2  # local import to keep module import light

    ok, out = cv2.imencode(".png", img)
    if not ok:
        return png_bytes
    return bytes(out.tobytes())


def _render_minimap(
    *,
    positions: Dict[str, Tuple[float, float, float]],
    zones: List[ConflictZone],
    size_wh: Tuple[int, int] = (260, 260),
):
    import cv2
    import numpy as np

    w, h = int(size_wh[0]), int(size_wh[1])
    img = np.full((h, w, 3), 32, dtype=np.uint8)
    pts = [(float(p[0]), float(p[1])) for p in positions.values()] if positions else [(0.0, 0.0)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = 5.0
    x0 -= pad
    x1 += pad
    y0 -= pad
    y1 += pad
    span_x = max(1e-3, x1 - x0)
    span_y = max(1e-3, y1 - y0)
    scale = 0.88 * min(w / span_x, h / span_y)

    def to_px(x: float, y: float) -> Tuple[int, int]:
        px = int(round((x - x0) * scale + 0.06 * w))
        py = int(round((y1 - y) * scale + 0.06 * h))
        return px, py

    # Draw zones.
    for z in zones:
        p1 = to_px(z.x_min, z.y_min)
        p2 = to_px(z.x_max, z.y_max)
        cv2.rectangle(img, p1, p2, (0, 140, 255), 1)

    # Draw agents.
    for name, (x, y, _z) in positions.items():
        px, py = to_px(float(x), float(y))
        cv2.circle(img, (px, py), 3, (255, 255, 255), -1)
        cv2.putText(img, str(name), (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)

    cv2.putText(img, "minimap (x,y)", (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return img


def _mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(sum(float(v) for v in vals)) / float(len(vals))


def _percentile(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(float(v) for v in vals)
    if len(xs) == 1:
        return float(xs[0])
    qq = min(1.0, max(0.0, float(q)))
    idx = qq * float(len(xs) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(xs[lo])
    w = float(idx - lo)
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def _compose_demo_frame(
    *,
    global_png: bytes,
    rep_fpv_pngs: List[Optional[bytes]],
    chase_png: Optional[bytes],
    minimap_img,
    overlay_lines: List[OverlayLine],
):
    import cv2
    import numpy as np

    global_img = decode_png(global_png) if global_png else None
    if global_img is None:
        return None

    # Global view base: 1280x720.
    global_img = cv2.resize(global_img, (1280, 720), interpolation=cv2.INTER_AREA)

    # Paste minimap.
    try:
        mh, mw = minimap_img.shape[:2]
        x0, y0 = 18, 80
        global_img[y0 : y0 + mh, x0 : x0 + mw] = minimap_img
    except Exception:
        pass

    # Text overlay.
    draw_overlay_panel(global_img, overlay_lines)

    # Right panel: 640x720: top 2x2 reps (640x360), bottom chase (640x360).
    panel = np.full((720, 640, 3), 18, dtype=np.uint8)

    def tile_or_blank(png: Optional[bytes], label: str, size: Tuple[int, int]):
        tw, th = int(size[0]), int(size[1])
        if not png:
            img = np.full((th, tw, 3), 25, dtype=np.uint8)
            cv2.putText(img, label, (10, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
            return img
        im = decode_png(png)
        if im is None:
            im = np.full((th, tw, 3), 25, dtype=np.uint8)
            cv2.putText(im, label, (10, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
            return im
        return cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)

    # Representative FPVs.
    reps = (rep_fpv_pngs + [None, None, None, None])[:4]
    t00 = tile_or_blank(reps[0], "rep1", (320, 180))
    t01 = tile_or_blank(reps[1], "rep2", (320, 180))
    t10 = tile_or_blank(reps[2], "rep3", (320, 180))
    t11 = tile_or_blank(reps[3], "rep4", (320, 180))
    panel[0:180, 0:320] = t00
    panel[0:180, 320:640] = t01
    panel[180:360, 0:320] = t10
    panel[180:360, 320:640] = t11
    cv2.putText(panel, "Representative FPVs", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)

    # Chase.
    chase_tile = tile_or_blank(chase_png, "chase", (640, 360))
    panel[360:720, 0:640] = chase_tile
    cv2.putText(panel, "Close chase", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)

    frame = cv2.hconcat([global_img, panel])
    ok, out = cv2.imencode(".png", frame)
    if not ok:
        return None
    return bytes(out.tobytes())


def _wirebox_segments(
    airsim,
    *,
    x: float,
    y: float,
    z: float,
    half_xy_m: float,
    half_z_m: float,
):
    """Return a Vector3r list suitable for `simPlotLineList` (pairs of points = segments)."""
    hx = float(half_xy_m)
    hy = float(half_xy_m)
    hz = float(half_z_m)

    # 8 corners.
    c000 = airsim.Vector3r(float(x - hx), float(y - hy), float(z - hz))
    c001 = airsim.Vector3r(float(x - hx), float(y - hy), float(z + hz))
    c010 = airsim.Vector3r(float(x - hx), float(y + hy), float(z - hz))
    c011 = airsim.Vector3r(float(x - hx), float(y + hy), float(z + hz))
    c100 = airsim.Vector3r(float(x + hx), float(y - hy), float(z - hz))
    c101 = airsim.Vector3r(float(x + hx), float(y - hy), float(z + hz))
    c110 = airsim.Vector3r(float(x + hx), float(y + hy), float(z - hz))
    c111 = airsim.Vector3r(float(x + hx), float(y + hy), float(z + hz))

    edges = [
        # bottom rectangle
        (c000, c010),
        (c010, c110),
        (c110, c100),
        (c100, c000),
        # top rectangle
        (c001, c011),
        (c011, c111),
        (c111, c101),
        (c101, c001),
        # verticals
        (c000, c001),
        (c010, c011),
        (c110, c111),
        (c100, c101),
    ]

    pts = []
    for a, b in edges:
        pts.append(a)
        pts.append(b)
    return pts


def _connect_multirotor(airsim, *, ip: str, port: int, timeout_s: int):
    client = airsim.MultirotorClient(ip=ip, port=int(port), timeout_value=int(timeout_s))
    client.confirmConnection()
    return client


def _arm_takeoff(client, *, vehicle_name: str) -> None:
    client.enableApiControl(True, vehicle_name=vehicle_name)
    client.armDisarm(True, vehicle_name=vehicle_name)
    try:
        client.takeoffAsync(vehicle_name=vehicle_name).join()
    except Exception:
        pass


def _arm_only(client, *, vehicle_name: str) -> None:
    client.enableApiControl(True, vehicle_name=vehicle_name)
    client.armDisarm(True, vehicle_name=vehicle_name)


def _reset_to_start_and_takeoff(
    client,
    airsim,
    *,
    vehicle_name: str,
    x: float,
    y: float,
    z_ned: float,
    yaw_deg: float,
) -> None:
    # Put on ground, then take off to the target altitude for stability across envs.
    # (We ignore collisions for the first few steps to avoid spurious t=0 collision flags.)
    _set_pose(client, airsim, vehicle_name=vehicle_name, x=x, y=y, z=0.0, yaw_deg=yaw_deg)
    time.sleep(0.05)
    try:
        client.takeoffAsync(vehicle_name=vehicle_name).join()
    except Exception:
        pass
    try:
        client.moveToZAsync(float(z_ned), 2.0, vehicle_name=vehicle_name).join()
    except Exception:
        _move_step(client, vehicle_name=vehicle_name, vx=0.0, vy=0.0, z_ned=float(z_ned), dt_s=1.0)


def _set_pose(client, airsim, *, vehicle_name: str, x: float, y: float, z: float, yaw_deg: float = 0.0) -> None:
    pose = client.simGetVehiclePose(vehicle_name=vehicle_name)
    pose.position.x_val = float(x)
    pose.position.y_val = float(y)
    pose.position.z_val = float(z)
    pose.orientation = airsim.to_quaternion(0.0, 0.0, math.radians(float(yaw_deg)))
    client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)


def _set_pose_rpy(
    client,
    airsim,
    *,
    vehicle_name: str,
    x: float,
    y: float,
    z: float,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> None:
    pose = client.simGetVehiclePose(vehicle_name=vehicle_name)
    pose.position.x_val = float(x)
    pose.position.y_val = float(y)
    pose.position.z_val = float(z)
    pose.orientation = airsim.to_quaternion(
        math.radians(float(pitch_deg)),
        math.radians(float(roll_deg)),
        math.radians(float(yaw_deg)),
    )
    client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)


def _get_pos(client, *, vehicle_name: str) -> Tuple[float, float, float]:
    st = client.getMultirotorState(vehicle_name=vehicle_name)
    pos = st.kinematics_estimated.position
    return float(pos.x_val), float(pos.y_val), float(pos.z_val)


def _get_collision(client, *, vehicle_name: str) -> Tuple[bool, float]:
    try:
        c = client.simGetCollisionInfo(vehicle_name=vehicle_name)
        collided = bool(getattr(c, "has_collided", False))
        pen = float(getattr(c, "penetration_depth", 0.0) or 0.0)
        return collided, max(0.0, pen)
    except Exception:
        return False, 0.0


def _move_step(
    client,
    *,
    vehicle_name: str,
    vx: float,
    vy: float,
    z_ned: float,
    dt_s: float,
) -> None:
    try:
        client.moveByVelocityZAsync(float(vx), float(vy), float(z_ned), float(dt_s), vehicle_name=vehicle_name).join()
    except Exception:
        try:
            client.moveByVelocityAsync(float(vx), float(vy), 0.0, float(dt_s), vehicle_name=vehicle_name).join()
        except Exception:
            pass


def _move_step_async(
    client,
    *,
    vehicle_name: str,
    vx: float,
    vy: float,
    z_ned: float,
    dt_s: float,
):
    try:
        return client.moveByVelocityZAsync(float(vx), float(vy), float(z_ned), float(dt_s), vehicle_name=vehicle_name)
    except Exception:
        try:
            return client.moveByVelocityAsync(float(vx), float(vy), 0.0, float(dt_s), vehicle_name=vehicle_name)
        except Exception:
            return None


def _choose_lock_holder(requesters: List[str]) -> Optional[str]:
    if not requesters:
        return None
    # Deterministic: prefer smaller numeric suffix if present, else lexicographic.
    def key(name: str) -> Tuple[int, str]:
        digits = "".join(ch for ch in name if ch.isdigit())
        return (int(digits) if digits else 10**9, name)

    return sorted(requesters, key=key)[0]


def _build_multidrone_episode(
    *,
    client,
    vehicles: List[str],
    seed: int,
    z_ned: float,
    z_ned_by_vehicle: Optional[Dict[str, float]] = None,
    center_xy: Optional[Tuple[float, float]] = None,
    start_radius_m: float,
    zone_halfwidth_m: float,
    longseq_phases: int,
    scenario: str,
) -> Tuple[
    Dict[str, Tuple[float, float, float]],
    List[Dict[str, Tuple[float, float, float]]],
    List[ConflictZone],
]:
    """Return (starts, goal_phases, zones)."""
    scenario = str(scenario or "intersection").strip().lower()
    if scenario not in {"intersection", "longseq"}:
        scenario = "intersection"

    rng = random.Random(int(seed))
    if isinstance(center_xy, (tuple, list)) and len(center_xy) >= 2:
        cx, cy = float(center_xy[0]), float(center_xy[1])
    else:
        base_x, base_y, _base_z = _get_pos(client, vehicle_name=vehicles[0])
        cx, cy = float(base_x), float(base_y)

    k = max(1, len(vehicles))
    r = max(10.0, float(start_radius_m))
    jitter_r = float(rng.uniform(-1.0, 1.0))
    r_eff = r + jitter_r

    starts: Dict[str, Tuple[float, float, float]] = {}
    base_goals: Dict[str, Tuple[float, float, float]] = {}
    for i, v in enumerate(vehicles):
        ang = 2.0 * math.pi * float(i) / float(k)
        sx = cx + r_eff * math.cos(ang)
        sy = cy + r_eff * math.sin(ang)
        gx = cx - r_eff * math.cos(ang)
        gy = cy - r_eff * math.sin(ang)
        z_v = float(z_ned_by_vehicle.get(v, z_ned)) if isinstance(z_ned_by_vehicle, dict) else float(z_ned)
        starts[v] = (float(sx), float(sy), float(z_v))
        base_goals[v] = (float(gx), float(gy), float(z_v))

    # Single shared conflict zone at the center (forces multi-agent coordination).
    half = max(4.0, float(zone_halfwidth_m))
    zone = ConflictZone(zone_id="zone0", x_min=cx - half, x_max=cx + half, y_min=cy - half, y_max=cy + half)
    zones = [zone]

    # Goal phases:
    if scenario == "intersection":
        phases = [base_goals]
    else:
        phases_n = max(2, int(longseq_phases))
        vs = list(vehicles)
        phases = []
        for p in range(phases_n):
            # Deterministic cyclic goal reassignment (forces replanning on goal updates).
            phase_goals = {vs[i]: base_goals[vs[(i + p) % len(vs)]] for i in range(len(vs))}
            phases.append(phase_goals)
    return starts, phases, zones


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a Domain C (AirSim) config JSON.")
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--run_name", default="airsim_domainc")
    ap.add_argument(
        "--ue_env",
        default=None,
        help="UE env name: airsimnh|blocks|abandonedpark|landscapemountains (zhangjiajie disabled).",
    )
    ap.add_argument("--no_auto_launch", action="store_true", help="Connect to an already-running UE env (do not launch).")
    ap.add_argument("--gpu", type=int, default=None, help="Override GPU index for UE launch (default: pick idle GPU).")
    ap.add_argument("--z_up_m", type=float, default=None, help="Override altitude above ground (meters).")
    ap.add_argument("--res", nargs=2, type=int, metavar=("RESX", "RESY"), default=None, help="UE render resolution.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _PROJ_ROOT / cfg_path
    base_cfg = _load_json(cfg_path)
    base_cfg.setdefault("domain", "airsim")
    base_cfg.setdefault("task", base_cfg.get("task", "domainc_multidrone_conflict"))

    if args.ue_env is not None:
        base_cfg["ue_env_name"] = str(args.ue_env)
    if bool(args.no_auto_launch):
        base_cfg["auto_launch_env"] = False
    if args.gpu is not None:
        base_cfg["ue_gpu"] = int(args.gpu)
    if args.z_up_m is not None:
        base_cfg["z_up_m"] = float(args.z_up_m)
    if args.res is not None and len(args.res) == 2:
        base_cfg["ue_res_x"] = int(args.res[0])
        base_cfg["ue_res_y"] = int(args.res[1])

    base_cfg.setdefault("auto_launch_env", True)
    base_cfg.setdefault("ue_env_name", "airsimnh")
    base_cfg.setdefault("ue_envs_root", "${BRACE_AIRSIM_ENVS_ROOT}")
    base_cfg.setdefault("ue_res_x", 1280)
    base_cfg.setdefault("ue_res_y", 720)
    base_cfg.setdefault("ue_rpc_wait_s", 120)
    base_cfg.setdefault("cam_global_vehicle", "CamGlobal")
    base_cfg.setdefault("cam_chase_vehicle", "CamChase")
    base_cfg.setdefault("scenario", "intersection")
    base_cfg.setdefault("start_radius_m", 45.0)
    base_cfg.setdefault("zone_halfwidth_m", 10.0)
    base_cfg.setdefault("longseq_phases", 3)

    base_cfg = _expand_env_vars(base_cfg)

    ctx = RunContext.create(runs_root=str(_PROJ_ROOT / args.runs_root), run_name=str(args.run_name), config=base_cfg)
    ctx.append_event({"phase": "config_loaded", "config_path": str(cfg_path)})

    ip = str(base_cfg.get("airsim_ip", "127.0.0.1"))
    port = int(base_cfg.get("airsim_port", 41451))
    timeout_s = int(base_cfg.get("airsim_timeout_s", 30))

    env_name = _canonical_env_name(str(base_cfg.get("ue_env_name", "airsimnh")))
    if env_name == "zhangjiajie":
        print("ERROR: ZhangJiajie is disabled for this demo run. Use a different --ue_env.")
        ctx.write_summary({"status": "error", "error": "zhangjiajie_disabled"})
        return 2

    env_handle: Optional[UEEnvHandle] = None
    auto_launch_env = bool(base_cfg.get("auto_launch_env", True))
    if auto_launch_env:
        envs_root_raw = str(base_cfg.get("ue_envs_root", "")).strip()
        if not envs_root_raw or "$" in envs_root_raw:
            msg = (
                "Missing AirSim UE envs root. Set BRACE_AIRSIM_ENVS_ROOT or set ue_envs_root in the config JSON "
                "(e.g., configs/airsim/*.json)."
            )
            print(f"ERROR: {msg}")
            ctx.write_summary({"status": "error", "error": "missing_ue_envs_root", "hint": msg})
            return 2
        envs_root = Path(envs_root_raw).expanduser().resolve()
        if not envs_root.exists():
            msg = f"AirSim UE envs root does not exist: {envs_root}"
            print(f"ERROR: {msg}")
            ctx.write_summary({"status": "error", "error": "ue_envs_root_not_found", "ue_envs_root": str(envs_root)})
            return 2
        env_bin = _resolve_env_bin(env_name=env_name, envs_root=envs_root, env_bin_override=base_cfg.get("ue_env_bin"))
        gpu = _select_idle_gpu(gpu_override=base_cfg.get("ue_gpu"))

        runtime_dir = Path(ctx.run_dir) / "_runtime_settings"
        runtime_settings = runtime_dir / "settings.json"
        os.environ["AIRSIM_SETTINGS_PATH"] = str(runtime_settings)

        # Settings: either write a provided template, or generate one for multi-drone demos.
        settings_mode = str(base_cfg.get("settings_mode", "") or "").strip().lower()
        num_drones_cfg = base_cfg.get("num_drones")
        if isinstance(num_drones_cfg, (int, float)) and int(num_drones_cfg) > 0:
            settings_mode = settings_mode or "generated_multidrone"

        if settings_mode == "generated_multidrone":
            num_drones = int(num_drones_cfg) if isinstance(num_drones_cfg, (int, float)) else int(len(base_cfg.get("vehicles") or []) or 2)
            payload = _build_multidrone_settings(
                num_drones=int(num_drones),
                api_server_port=int(port),
                fpv_wh=tuple(base_cfg.get("fpv_wh", [640, 360])),
                global_wh=tuple(base_cfg.get("global_wh", [1280, 720])),
                chase_wh=tuple(base_cfg.get("chase_wh", [1280, 720])),
                fpv_fov_deg=float(base_cfg.get("fpv_fov_deg", 80.0)),
                global_fov_deg=float(base_cfg.get("global_fov_deg", 60.0)),
                chase_fov_deg=float(base_cfg.get("chase_fov_deg", 75.0)),
                cam_global_vehicle=str(base_cfg.get("cam_global_vehicle", "CamGlobal")),
                cam_chase_vehicle=str(base_cfg.get("cam_chase_vehicle", "CamChase")),
            )
            written = _write_settings_payload(payload, ctx=ctx)
            ctx.append_event({"phase": "settings_written", "path": str(written), "template": None, "mode": "generated_multidrone"})
        else:
            settings_template = base_cfg.get("settings_template")
            if settings_template:
                st_path = Path(str(settings_template))
                if not st_path.is_absolute():
                    st_path = _PROJ_ROOT / st_path
                if st_path.exists():
                    written = _write_settings(st_path, ctx=ctx)
                    ctx.append_event({"phase": "settings_written", "path": str(written), "template": str(st_path), "mode": "template"})
                else:
                    ctx.append_event({"phase": "settings_missing", "template": str(st_path)})

        if not runtime_settings.exists():
            err = f"Auto-launch requires a valid AirSim settings file; missing: {runtime_settings}"
            print(f"ERROR: {err}")
            ctx.write_summary({"status": "error", "error": err})
            return 2

        log_path = runtime_dir / f"ue_{env_name}_gpu{int(gpu)}_{_utc_now_compact()}.log"
        allowed_cmdline_substrings = [str(envs_root)]
        env_handle = _launch_ue_env(
            env_name=env_name,
            env_bin=env_bin,
            settings_path=runtime_settings,
            log_path=log_path,
            ip=ip,
            port=port,
            timeout_s=int(base_cfg.get("ue_rpc_wait_s", 120)),
            gpu=int(gpu),
            res_x=int(base_cfg.get("ue_res_x", 1280)),
            res_y=int(base_cfg.get("ue_res_y", 720)),
            allowed_cmdline_substrings=allowed_cmdline_substrings,
            ctx=ctx,
        )
        atexit.register(env_handle.stop, ctx=ctx)

    airsim = import_airsim()

    try:
        if auto_launch_env:
            client = _wait_for_multirotor_client(
                airsim,
                ip=ip,
                port=port,
                rpc_wait_s=int(base_cfg.get("ue_rpc_wait_s", 120)),
                connect_timeout_s=2,
                ctx=ctx,
            )
        else:
            client = _connect_multirotor(airsim, ip=ip, port=port, timeout_s=timeout_s)
    except Exception as e:
        if env_handle is not None:
            try:
                env_handle.stop(ctx=ctx)
            except Exception:
                pass
        print("")
        print("ERROR: Failed to connect to AirSim RPC server.")
        print(f"- ip={ip} port={port} timeout_s={timeout_s}")
        if auto_launch_env:
            print(f"- UE env: {env_name} (auto-launched)")
        else:
            print(f"- UE env: {env_name} (auto_launch_env=false; expected already running)")
        if env_handle is not None:
            print(f"- UE log: {env_handle.log_path}")
        print("")
        print(f"Exception: {type(e).__name__}: {e}")
        ctx.write_summary({"status": "error", "error": f"{type(e).__name__}: {e}"})
        return 2

    # Multi-agent size: support K ∈ {1,2,4,8} by config `num_drones` or explicit `vehicles`.
    num_drones_cfg2 = base_cfg.get("num_drones")
    if isinstance(num_drones_cfg2, (int, float)) and int(num_drones_cfg2) > 0:
        vehicles = [f"Drone{i}" for i in range(1, int(num_drones_cfg2) + 1)]
    else:
        vehicles = list(base_cfg.get("vehicles") or ["Drone1", "Drone2"])
        vehicles = [str(v) for v in vehicles if v]
    cam_global_vehicle = str(base_cfg.get("cam_global_vehicle", "CamGlobal"))
    cam_chase_vehicle = str(base_cfg.get("cam_chase_vehicle", "CamChase"))

    # Best-effort: arm all vehicles once at run start (takeoff handled per-episode).
    for v in vehicles:
        _arm_only(client, vehicle_name=v)
    try:
        _arm_only(client, vehicle_name=cam_global_vehicle)
        _arm_only(client, vehicle_name=cam_chase_vehicle)
    except Exception:
        pass

    variants = _expand_variants(base_cfg)
    base_seed = int(base_cfg.get("seed", 0))

    # Fix the episode center per run so all variants share the same starts/goals (fair comparisons).
    episode_center_xy = None
    cfg_center = base_cfg.get("episode_center_xy")
    if isinstance(cfg_center, (list, tuple)) and len(cfg_center) >= 2:
        try:
            episode_center_xy = (float(cfg_center[0]), float(cfg_center[1]))
        except Exception:
            episode_center_xy = None
    if episode_center_xy is None:
        try:
            _cx, _cy, _cz = _get_pos(client, vehicle_name=vehicles[0])
            episode_center_xy = (float(_cx), float(_cy))
        except Exception:
            episode_center_xy = None

    dt_s = float(base_cfg.get("dt_s", 0.08))
    speed_m_s = float(base_cfg.get("speed_m_s", 3.0))
    z_up_m = float(base_cfg.get("z_up_m", 30.0))
    z_ned = -float(z_up_m)
    alt_stagger_m = float(base_cfg.get("alt_stagger_m", 0.0))
    z_ned_by_vehicle: Dict[str, float] = {}
    for i, v in enumerate(vehicles):
        # NED: negative is up. Staggering makes physical collisions far less likely while keeping agents visible.
        z_ned_by_vehicle[v] = float(z_ned) - float(alt_stagger_m) * float(i)

    episodes_default = int(base_cfg.get("episodes", 3))
    max_steps_default = int(base_cfg.get("max_steps", 400))
    goal_radius_m = float(base_cfg.get("goal_radius_m", 2.0))
    goal_slowdown_dist_m = float(base_cfg.get("goal_slowdown_dist_m", max(6.0, 3.0 * float(goal_radius_m))))

    start_radius_m = float(base_cfg.get("start_radius_m", 45.0))
    zone_halfwidth_m = float(base_cfg.get("zone_halfwidth_m", 10.0))
    longseq_phases = int(base_cfg.get("longseq_phases", 3))
    scenario = str(base_cfg.get("scenario", "intersection"))

    # Trigger params.
    near_miss_dist_m = float(base_cfg.get("near_miss_dist_m", 3.0))
    near_miss_metric = str(base_cfg.get("near_miss_metric", "xyz")).strip().lower()
    progress_epsilon = float(base_cfg.get("progress_epsilon", 0.1))
    deadlock_window_steps = int(base_cfg.get("deadlock_window_steps", 120))
    deadlock_escape_steps = int(base_cfg.get("deadlock_escape_steps", 40))
    lock_ttl_steps = int(base_cfg.get("lock_ttl_steps", 200))
    lock_release_grace_steps = int(base_cfg.get("lock_release_grace_steps", 8))
    lock_approach_dist_m = float(base_cfg.get("lock_approach_dist_m", 2.5))
    lock_wait_mode = str(base_cfg.get("lock_wait_mode", "stop")).strip().lower()
    lock_wait_speed_factor = float(base_cfg.get("lock_wait_speed_factor", 0.25))
    lock_enabled = bool(base_cfg.get("lock_enabled", True))
    ignore_collision_steps = int(base_cfg.get("ignore_collision_steps", 12))
    failure_trigger_min_t = int(base_cfg.get("failure_trigger_min_t", 10))

    # Simple local collision-avoidance term (keeps demos collision-free without hiding interaction).
    repulse_radius_m = float(base_cfg.get("repulse_radius_m", 0.0))
    repulse_gain = float(base_cfg.get("repulse_gain", 0.0))

    goal_update_steps_cfg = base_cfg.get("goal_update_steps")
    goal_update_steps: Optional[List[int]] = None
    if isinstance(goal_update_steps_cfg, list) and goal_update_steps_cfg:
        goal_update_steps = [int(x) for x in goal_update_steps_cfg if isinstance(x, (int, float))]

    # Proxy latency model (used for stub planner; real planners can override).
    ms_per_token = float(base_cfg.get("ms_per_token", 2.0))
    overhead_ms = float(base_cfg.get("planner_overhead_ms", 40.0))
    summary_overhead_ms = float(base_cfg.get("summary_overhead_ms", 15.0))

    # Token model (proxy).
    tokens_task = int(base_cfg.get("tokens_task", 120))
    tokens_state_per_agent = int(base_cfg.get("tokens_state_per_agent", 60))
    tokens_safety_per_agent = int(base_cfg.get("tokens_safety_per_agent", 40))
    tokens_coord_per_agent = int(base_cfg.get("tokens_coord_per_agent", 50))
    tokens_history_base = int(base_cfg.get("tokens_history_base", 200))
    tokens_history_per_replan = int(base_cfg.get("tokens_history_per_replan", 80))

    demo_enabled = bool(base_cfg.get("demo_enabled", True))
    demo_fps = int(base_cfg.get("demo_fps", 15))
    demo_capture_every = int(base_cfg.get("demo_capture_every", 2))
    demo_warmup_steps = int(base_cfg.get("demo_warmup_steps", 12))
    demo_image_sleep_s = float(base_cfg.get("demo_image_sleep_s", 0.02))
    demo_image_retries = int(base_cfg.get("demo_image_retries", 8))
    demo_image_base_sleep_s = float(base_cfg.get("demo_image_base_sleep_s", 0.15))
    demo_plot_agents = bool(base_cfg.get("demo_plot_agents", True))
    demo_plot_size = float(base_cfg.get("demo_plot_size", 18.0))
    demo_plot_style = str(base_cfg.get("demo_plot_style", "wirebox")).strip().lower()
    demo_plot_color = base_cfg.get("demo_plot_color_rgba", [0.0, 1.0, 1.0, 1.0])
    demo_plot_thickness = float(base_cfg.get("demo_plot_thickness", 6.0))
    demo_plot_half_xy_m = float(base_cfg.get("demo_plot_box_half_xy_m", 1.6))
    demo_plot_half_z_m = float(base_cfg.get("demo_plot_box_half_z_m", 0.8))
    demo_plot_z_offset_m = float(base_cfg.get("demo_plot_z_offset_m", -0.2))
    demo_emit_panels = bool(base_cfg.get("demo_emit_panels", False))
    demo_make_comparisons = bool(base_cfg.get("demo_make_comparisons", True))
    demo_make_grid_2x2 = bool(base_cfg.get("demo_make_grid_2x2", True))

    demo_index: Dict[str, Dict[str, Path]] = {}

    variant_order = [str(v.get("name", "unknown")) for v in variants]

    for variant in variants:
        vname = str(variant.get("name", "unknown"))
        episodes = int(variant.get("episodes", episodes_default))
        max_steps = int(variant.get("max_steps", max_steps_default))

        replan_interval_steps = int(
            variant.get(
                "replan_interval_steps",
                variant.get("replan_interval", base_cfg.get("replan_interval_steps", base_cfg.get("replan_interval", 10))),
            )
        )
        trigger_cooldown_steps = int(variant.get("trigger_cooldown_steps", base_cfg.get("trigger_cooldown_steps", 0)))

        brace_enabled = bool(variant.get("brace_enabled", False))
        pruning_enabled_cfg = bool(variant.get("pruning_enabled", False))
        keep_ratio = float(variant.get("keep_ratio", base_cfg.get("keep_ratio", 1.0)))
        token_budget = int(variant.get("token_budget", base_cfg.get("token_budget", 0)))
        token_budget = token_budget if token_budget > 0 else None

        slo_ms = variant.get("slo_ms", base_cfg.get("slo_ms"))
        slo_ms = int(slo_ms) if isinstance(slo_ms, (int, float)) and int(slo_ms) > 0 else None

        context_method = normalize_method(
            variant.get("context_compress_method", variant.get("context_strategy")),
            pruning_enabled=pruning_enabled_cfg,
        )
        pruning_enabled_event = context_method in ("erecap", "random", "recency")
        summary_compress_enabled_event = context_method == "structured_summary"

        hparams_dict: Dict[str, Any] = dict(base_cfg.get("brace_hparams", {}) or {})
        hparams_dict.update(dict(variant.get("brace_hparams", {}) or {}))
        if slo_ms is not None and "slo_ms" not in hparams_dict:
            hparams_dict["slo_ms"] = int(slo_ms)
        if "slo_ms" not in hparams_dict:
            hparams_dict["slo_ms"] = int(base_cfg.get("slo_ms", 2500))
        controller = BraceController(BraceHyperparams(**hparams_dict))

        ctx.append_event(
            {
                "phase": "variant_start",
                "variant": vname,
                "episodes": episodes,
                "max_steps": max_steps,
                "vehicles": vehicles,
                "replan_interval_steps": int(replan_interval_steps),
                "trigger_cooldown_steps": int(trigger_cooldown_steps),
                "brace_enabled": bool(brace_enabled),
                "pruning_enabled": bool(pruning_enabled_event),
                "summary_compress_enabled": bool(summary_compress_enabled_event),
                "context_compress_method": context_method,
                "keep_ratio": float(keep_ratio),
                "token_budget": token_budget,
                "slo_ms": slo_ms,
                "z_ned": float(z_ned),
            }
        )

        record_demo = bool(demo_enabled and bool(variant.get("demo", True)))

        for ep_i in range(episodes):
            episode_id = f"airsim_ep{ep_i:04d}"

            # Re-seed per episode so demo is stable, and shared across variants (so comparisons are aligned).
            ep_seed = base_seed + int(ep_i)
            starts, goal_phases, zones = _build_multidrone_episode(
                client=client,
                vehicles=vehicles,
                seed=ep_seed,
                z_ned=z_ned,
                z_ned_by_vehicle=z_ned_by_vehicle,
                center_xy=episode_center_xy,
                start_radius_m=start_radius_m,
                zone_halfwidth_m=zone_halfwidth_m,
                longseq_phases=longseq_phases,
                scenario=scenario,
            )
            goal_phase_idx = 0
            goals = dict(goal_phases[0]) if goal_phases else {}

            # For long sequences, update goals multiple times (if not explicitly provided).
            if goal_update_steps is None and str(scenario).strip().lower() == "longseq" and len(goal_phases) > 1:
                # Avoid updates too early; spread across episode.
                goal_update_steps = [max_steps // 3, (2 * max_steps) // 3]

            # Reset vehicles to deterministic starts (ground) then takeoff to target altitude.
            for idx, v in enumerate(vehicles):
                if v not in starts:
                    continue
                x, y, z = starts[v]
                _reset_to_start_and_takeoff(client, airsim, vehicle_name=v, x=x, y=y, z_ned=z, yaw_deg=(360.0 * idx / max(1, len(vehicles))))
            # Reset camera rigs to a safe high-altitude pose so they don't physically collide across variants.
            try:
                # Use the episode center (derived from Drone1's start) as reference.
                cx0, cy0, cz0 = starts.get(vehicles[0], (0.0, 0.0, float(z_ned)))
                safe_z = float(min(z_ned_by_vehicle.values()) if z_ned_by_vehicle else z_ned) - 200.0
                _set_pose(client, airsim, vehicle_name=cam_global_vehicle, x=float(cx0), y=float(cy0), z=float(safe_z), yaw_deg=0.0)
                _set_pose(client, airsim, vehicle_name=cam_chase_vehicle, x=float(cx0), y=float(cy0), z=float(safe_z), yaw_deg=0.0)
            except Exception:
                pass
            time.sleep(0.2)
            episode_starts = dict(starts)

            zone_locks: Dict[str, ZoneLock] = {z.zone_id: ZoneLock() for z in zones}
            wait_steps: Dict[str, int] = {v: 0 for v in vehicles}
            wait_time_ms: Dict[str, float] = {v: 0.0 for v in vehicles}

            brace_state = BraceState()
            deadlock_escape_remaining = 0
            replan_count = 0
            planner_calls = 0
            suppressed_triggers = 0
            last_replan_step = -10**9
            last_plan_latency_ms: Optional[float] = None
            last_plan_hash: Optional[str] = None
            last_plan: Optional[Dict[str, Any]] = None
            last_plan_changed_flag = False
            dist_at_last_replan: Optional[float] = None
            lat_total_samples: List[float] = []
            tokens_in_samples: List[float] = []
            tokens_after_samples: List[float] = []
            slo_violation_count = 0

            near_miss_total = 0
            collisions_total = 0
            max_pen_total = 0.0
            reached_steps = 0

            demo_dir = Path(_PROJ_ROOT) / "artifacts" / "demos" / "airsim" / ctx.run_id
            demo_frames_root = Path(ctx.run_dir) / "demo_frames" / f"{episode_id}" / f"{vname}"
            frames_panel = demo_frames_root / "frames_panel"
            screenshots_dir = demo_dir / "screenshots"
            if record_demo:
                frames_panel.mkdir(parents=True, exist_ok=True)
                screenshots_dir.mkdir(parents=True, exist_ok=True)

            demo_panel_path = demo_dir / f"{vname}__{episode_id}__panel.mp4"
            if record_demo:
                ctx.append_event(
                    {
                        "phase": "demo_start",
                        "variant": vname,
                        "episode_id": episode_id,
                        "path": str(demo_panel_path),
                        "fps": int(demo_fps),
                        "frame_wh": None,
                    }
                )

            if record_demo:
                demo_index.setdefault(episode_id, {})[vname] = frames_panel

            ep_t0 = time.time()
            step_count = 0
            reached_success = False
            for t in range(max_steps):
                step_count = t + 1

                # Goal update (optional, deterministic).
                goal_update_flag = False
                if goal_update_steps is not None and int(t) in set(int(x) for x in goal_update_steps):
                    if goal_phase_idx + 1 < len(goal_phases):
                        goal_phase_idx += 1
                        goals = dict(goal_phases[goal_phase_idx])
                        goal_update_flag = True

                # Gather positions, distances, safety signals.
                positions: Dict[str, Tuple[float, float, float]] = {}
                dists: Dict[str, float] = {}
                any_collision = False
                for v in vehicles:
                    pos = _get_pos(client, vehicle_name=v)
                    positions[v] = pos
                    g = goals.get(v)
                    if g is not None:
                        dists[v] = _dist_xyz(pos, g)
                    collided, pen = _get_collision(client, vehicle_name=v)
                    if t < int(ignore_collision_steps):
                        collided = False
                        pen = 0.0
                    if collided:
                        any_collision = True
                        collisions_total += 1
                    max_pen_total = max(max_pen_total, float(pen))

                # Min inter-agent distance (near-miss).
                min_dist_xyz = float("inf")
                min_dist_xy = float("inf")
                if len(vehicles) >= 2:
                    for i in range(len(vehicles)):
                        for j in range(i + 1, len(vehicles)):
                            a = positions.get(vehicles[i])
                            b = positions.get(vehicles[j])
                            if a is None or b is None:
                                continue
                            min_dist_xyz = min(min_dist_xyz, _dist_xyz(a, b))
                            min_dist_xy = min(min_dist_xy, _dist_xy((a[0], a[1]), (b[0], b[1])))
                if str(near_miss_metric) == "xy":
                    min_dist = min_dist_xy
                else:
                    min_dist = min_dist_xyz
                unsafe_near_miss = bool(min_dist != float("inf") and min_dist < float(near_miss_dist_m))
                if unsafe_near_miss:
                    near_miss_total += 1

                # Success condition (robust to brief oscillations near goal).
                all_reached = True
                for v in vehicles:
                    if v not in dists:
                        continue
                    if float(dists[v]) > float(goal_radius_m):
                        all_reached = False
                        break
                if str(scenario) == "longseq":
                    # Only allow terminating success on the final phase, so the demo
                    # reliably contains goal updates / long-horizon replanning.
                    if int(goal_phase_idx) != int(len(goal_phases) - 1):
                        all_reached = False
                if all_reached:
                    reached_steps += 1
                else:
                    reached_steps = 0

                if reached_steps >= int(base_cfg.get("goal_hold_steps", 8)) and not any_collision:
                    reached_success = True
                    break

                # Lock acquisition / release bookkeeping (based on current positions).
                locks_held: Dict[str, str] = {}
                locks_waiting: Dict[str, List[str]] = {}
                if lock_enabled:
                    for z in zones:
                        st = zone_locks[z.zone_id]

                        # Expire TTL.
                        if st.holder is not None and st.ttl_remaining > 0:
                            st.ttl_remaining -= 1
                            if st.ttl_remaining <= 0:
                                st.holder = None
                                st.holder_outside_steps = 0

                        # Holder release when outside for grace window.
                        if st.holder is not None:
                            hx, hy, _hz = positions.get(st.holder, (0.0, 0.0, 0.0))
                            if z.contains_xy(hx, hy):
                                st.holder_outside_steps = 0
                            else:
                                st.holder_outside_steps += 1
                                if st.holder_outside_steps >= int(lock_release_grace_steps):
                                    st.holder = None
                                    st.holder_outside_steps = 0

                        # Determine requesters: agents close to zone.
                        requesters: List[str] = []
                        for v in vehicles:
                            x, y, _zz = positions[v]
                            if z.contains_xy(x, y) or z.dist_to_xy(x, y) <= float(lock_approach_dist_m):
                                requesters.append(v)

                        # Acquire if free.
                        if st.holder is None:
                            chosen = _choose_lock_holder(requesters)
                            if chosen is not None:
                                st.holder = chosen
                                st.ttl_remaining = int(lock_ttl_steps)
                                st.holder_outside_steps = 0

                        if st.holder is not None:
                            locks_held[z.zone_id] = st.holder
                            waiting = [v for v in requesters if v != st.holder]
                            if waiting:
                                locks_waiting[z.zone_id] = waiting

                # Deadlock escape: temporarily ignore locks so agents can resolve jams (keeps motion visible in demos).
                waiting_agents: List[str] = []
                if not lock_enabled:
                    waiting_agents = []
                elif deadlock_escape_remaining > 0:
                    deadlock_escape_remaining -= 1
                    for z in zones:
                        st = zone_locks[z.zone_id]
                        st.holder = None
                        st.ttl_remaining = 0
                        st.holder_outside_steps = 0
                else:
                    for z in zones:
                        holder = zone_locks[z.zone_id].holder
                        for v in vehicles:
                            x, y, _zz = positions[v]
                            wants = z.contains_xy(x, y) or z.dist_to_xy(x, y) <= float(lock_approach_dist_m)
                            if wants and holder is not None and v != holder:
                                waiting_agents.append(v)

                for v in vehicles:
                    if v in waiting_agents:
                        wait_steps[v] += 1
                        wait_time_ms[v] += float(dt_s) * 1000.0
                    else:
                        wait_steps[v] = 0

                deadlock_trigger = bool(lock_enabled) and any(int(wait_steps[v]) >= int(deadlock_window_steps) for v in vehicles)
                if lock_enabled and deadlock_trigger and int(deadlock_escape_steps) > 0 and deadlock_escape_remaining <= 0:
                    deadlock_escape_remaining = int(deadlock_escape_steps)
                    waiting_agents = []
                    for z in zones:
                        st = zone_locks[z.zone_id]
                        st.holder = None
                        st.ttl_remaining = 0
                        st.holder_outside_steps = 0

                # Progress tracking (mean distance).
                mean_dist = float(sum(dists.values()) / max(1, len(dists))) if dists else float("nan")
                if dist_at_last_replan is None and mean_dist == mean_dist:
                    dist_at_last_replan = mean_dist
                progress_since_last = None
                if mean_dist == mean_dist and dist_at_last_replan is not None:
                    progress_since_last = float(dist_at_last_replan) - float(mean_dist)
                failure_trigger = bool(
                    int(t) >= int(failure_trigger_min_t)
                    and progress_since_last is not None
                    and float(progress_since_last) < float(progress_epsilon)
                )

                periodic_trigger = rs.periodic_trigger(
                    t=t, interval_steps=int(replan_interval_steps), last_replan_step=int(last_replan_step)
                )
                allow_trigger = rs.allow_trigger(
                    t=t, last_replan_step=int(last_replan_step), trigger_cooldown_steps=int(trigger_cooldown_steps)
                )

                extra_types: List[str] = []
                if goal_update_flag:
                    extra_types.append("goal_update")
                if unsafe_near_miss:
                    extra_types.append("near_miss")
                if any_collision:
                    extra_types.append("collision")
                if waiting_agents:
                    extra_types.append("lock_wait")

                any_trigger = bool(periodic_trigger or failure_trigger or deadlock_trigger or unsafe_near_miss or goal_update_flag)
                if any_trigger and not allow_trigger:
                    suppressed_triggers += 1

                do_replan = bool(any_trigger and allow_trigger)
                if do_replan:
                    replan_count += 1
                    last_replan_step = t

                    # BRACE decision.
                    telemetry = {
                        "clarification_budget_turns": int(base_cfg.get("clarification_budget_turns", 0)),
                        "churn": bool(last_plan_changed_flag),
                        "progress": progress_since_last,
                        "lat_total_ms": last_plan_latency_ms,
                    }
                    trigger_dict = rs.build_trigger_dict(
                        periodic=bool(periodic_trigger),
                        failure=bool(failure_trigger),
                        deadlock=bool(deadlock_trigger),
                        unsafe=bool(unsafe_near_miss or any_collision),
                        extra_types=extra_types,
                    )

                    if brace_enabled:
                        decision, brace_state = controller.step(
                            state=brace_state,
                            trigger=trigger_dict,
                            telemetry=telemetry,
                            remaining_budget=token_budget,
                            num_agents=len(vehicles),
                        )
                        mode = decision.mode
                        eff_budget = decision.token_budget
                        brace_reason = str(decision.reason)
                        hazards = {
                            "hazard_slo": bool(getattr(decision, "hazard_slo", False)),
                            "hazard_churn": bool(getattr(decision, "hazard_churn", False)),
                            "hazard_deadlock": bool(getattr(decision, "hazard_deadlock", False)),
                            "hazard_unsafe": bool(getattr(decision, "hazard_unsafe", False)),
                            "cooldown_active": bool(getattr(decision, "cooldown_active", False)),
                        }
                        rollback_flag = bool(decision.rollback_flag)
                        min_commit_window = int(decision.min_commit_window)
                    else:
                        mode = "partial_replan"
                        eff_budget = token_budget
                        brace_reason = "nobrace"
                        hazards = {
                            "hazard_slo": False,
                            "hazard_churn": False,
                            "hazard_deadlock": bool(deadlock_trigger),
                            "hazard_unsafe": bool(unsafe_near_miss or any_collision),
                            "cooldown_active": False,
                        }
                        rollback_flag = False
                        min_commit_window = 0

                    planner_called = mode in ("full_replan", "partial_replan")
                    if planner_called:
                        planner_calls += 1
                    budget_log = int(eff_budget) if eff_budget is not None else 0

                    # Proxy tokens/latency for the planner call.
                    tokens_state = int(tokens_state_per_agent) * len(vehicles)
                    tokens_safety = int(tokens_safety_per_agent) * len(vehicles)
                    tokens_coord = int(tokens_coord_per_agent) * len(vehicles)
                    tokens_protected = int(tokens_task) + tokens_state + tokens_safety + tokens_coord
                    tokens_history = int(tokens_history_base) + int(tokens_history_per_replan) * int(max(0, replan_count - 1))
                    tokens_in = int(tokens_protected) + int(tokens_history)

                    if planner_called:
                        tokens_after = _apply_context_compress(
                            tokens_in=tokens_in,
                            tokens_protected=tokens_protected,
                            method=context_method,
                            keep_ratio=keep_ratio,
                            token_budget=eff_budget,
                        )
                        lat_total_ms = float(overhead_ms) + float(ms_per_token) * float(tokens_after) + float(
                            extra_overhead_ms(context_method, summary_overhead_ms=summary_overhead_ms)
                        )
                        last_plan_latency_ms = lat_total_ms
                        lat_total_samples.append(float(lat_total_ms))
                        tokens_in_samples.append(float(tokens_in))
                        tokens_after_samples.append(float(tokens_after))
                        if slo_ms is not None and float(lat_total_ms) > float(slo_ms):
                            slo_violation_count += 1
                    else:
                        tokens_in = 0
                        tokens_task = 0
                        tokens_state = 0
                        tokens_safety = 0
                        tokens_coord = 0
                        tokens_history = 0
                        tokens_after = 0
                        lat_total_ms = 0.0

                    # Planner (stub): joint macro-actions for all agents.
                    if planner_called or last_plan is None:
                        # --- Coordination (lock holder) update ---
                        # To make BRACE vs baseline *visibly* different, we update the lock holder only when the
                        # planner is called. BRACE can then stabilize via `reuse_subplan/defer_replan` modes.
                        if lock_enabled:
                            lock_policy = str(base_cfg.get("lock_policy", "closest")).strip().lower()
                            locks_held = {}
                            locks_waiting = {}
                            for z in zones:
                                st = zone_locks[z.zone_id]
                                requesters2: List[str] = []
                                for vv in vehicles:
                                    x2, y2, _zz2 = positions[vv]
                                    if z.contains_xy(x2, y2) or z.dist_to_xy(x2, y2) <= float(lock_approach_dist_m):
                                        requesters2.append(vv)
                                if not requesters2:
                                    continue
                                if lock_policy == "closest":
                                    chosen = min(
                                        requesters2,
                                        key=lambda name: z.dist_to_xy(positions[name][0], positions[name][1]),
                                    )
                                else:
                                    chosen = _choose_lock_holder(requesters2)
                                if chosen is not None:
                                    st.holder = chosen
                                    st.ttl_remaining = int(lock_ttl_steps)
                                    st.holder_outside_steps = 0
                                    locks_held[z.zone_id] = chosen
                                    waiting2 = [vv for vv in requesters2 if vv != chosen]
                                    if waiting2:
                                        locks_waiting[z.zone_id] = waiting2

                        # Refresh waiting_agents based on the new holder decision (affects motion immediately).
                        if lock_enabled:
                            waiting_agents = []
                            for z in zones:
                                holder = zone_locks[z.zone_id].holder
                                for vv in vehicles:
                                    x2, y2, _zz2 = positions[vv]
                                    wants2 = z.contains_xy(x2, y2) or z.dist_to_xy(x2, y2) <= float(lock_approach_dist_m)
                                    if wants2 and holder is not None and vv != holder:
                                        waiting_agents.append(vv)
                        else:
                            waiting_agents = []

                        actions: Dict[str, Any] = {}
                        for v in vehicles:
                            g = goals.get(v)
                            if v in waiting_agents and lock_wait_mode == "stop":
                                actions[v] = {"type": "wait"}
                            elif v in waiting_agents and lock_wait_mode == "slow":
                                actions[v] = {
                                    "type": "goto",
                                    "target": list(g) if g is not None else None,
                                    "speed": float(speed_m_s) * max(0.0, float(lock_wait_speed_factor)),
                                    "wait": True,
                                }
                            else:
                                actions[v] = {"type": "goto", "target": list(g) if g is not None else None, "speed": float(speed_m_s)}
                        plan = {
                            "episode_id": episode_id,
                            "t": t,
                            "actions": actions,
                            "locks_held": locks_held,
                            "locks_waiting": locks_waiting,
                            "goal_update": goal_update_flag,
                        }
                        plan_hash = _hash_json(plan)
                        plan_changed = bool(last_plan_hash is not None and plan_hash != last_plan_hash)
                        plan_churn_score = 1.0 if plan_changed else 0.0
                        last_plan = plan
                        last_plan_hash = plan_hash
                        last_plan_changed_flag = bool(plan_changed)
                        dist_at_last_replan = mean_dist if mean_dist == mean_dist else dist_at_last_replan
                    else:
                        plan = last_plan
                        plan_hash = last_plan_hash or _hash_json(plan)
                        plan_changed = False
                        plan_churn_score = 0.0
                        last_plan_changed_flag = False

                    # Compute SLO violation derived fields.
                    slo_violation = bool(slo_ms is not None and float(lat_total_ms) > float(slo_ms))
                    slo_over_ms = max(0.0, float(lat_total_ms) - float(slo_ms)) if slo_ms is not None else 0.0

                    # Primary trigger label for audit table.
                    if unsafe_near_miss or any_collision:
                        replan_trigger_type = "unsafe"
                    elif deadlock_trigger:
                        replan_trigger_type = "deadlock"
                    elif failure_trigger:
                        replan_trigger_type = "failure"
                    elif goal_update_flag:
                        replan_trigger_type = "goal_update"
                    elif periodic_trigger:
                        replan_trigger_type = "periodic"
                    else:
                        replan_trigger_type = "unknown"

                    ctx.append_event(
                        {
                            "domain": "airsim",
                            "task": base_cfg.get("task"),
                            "variant": vname,
                            "episode_id": episode_id,
                            "t": t,
                            "brace_enabled": bool(brace_enabled),
                            "pruning_enabled": bool(pruning_enabled_event),
                            "summary_compress_enabled": bool(summary_compress_enabled_event),
                            "rag_enabled": False,
                            "mode": mode,
                            "token_budget": budget_log,
                            "slo_ms": slo_ms,
                            "tokens_in": int(tokens_in),
                            "tokens_after_prune": int(tokens_after),
                            "tokens_task": int(tokens_task),
                            "tokens_state": int(tokens_state),
                            "tokens_safety": int(tokens_safety),
                            "tokens_coord": int(tokens_coord),
                            "tokens_history": int(tokens_history),
                            "lat_total_ms": float(lat_total_ms),
                            "slo_violation": bool(slo_violation) if slo_ms is not None else None,
                            "slo_over_ms": float(slo_over_ms) if slo_ms is not None else None,
                            "plan_hash": str(plan_hash),
                            "plan_churn_score": float(plan_churn_score),
                            "brace_reason": brace_reason,
                            "hazards": hazards,
                            "cooldown_active": bool(hazards.get("cooldown_active", False)) if isinstance(hazards, dict) else None,
                            "rollback_flag": bool(rollback_flag),
                            "min_commit_window": int(min_commit_window),
                            "replan_trigger_type": str(replan_trigger_type),
                            "replan_interval_steps": int(replan_interval_steps),
                            "trigger_cooldown_steps": int(trigger_cooldown_steps),
                            "trigger": trigger_dict,
                            "min_interagent_dist_m": float(min_dist) if min_dist != float("inf") else None,
                            "near_miss_count": int(near_miss_total),
                            "collision": bool(any_collision),
                            "collision_count": int(collisions_total),
                            "max_penetration_depth": float(max_pen_total),
                            "locks_held": locks_held,
                            "locks_waiting": locks_waiting,
                            "deadlock_flag": bool(deadlock_trigger) if deadlock_trigger else None,
                            "wait_time_ms": float(sum(wait_time_ms.values())),
                            "goal_update_flag": bool(goal_update_flag),
                        }
                    )

                # Execute the current plan (best-effort).
                if last_plan is None:
                    last_plan = {"actions": {v: {"type": "goto", "target": list(goals.get(v) or []), "speed": float(speed_m_s)} for v in vehicles}}
                    last_plan_hash = _hash_json(last_plan)

                move_futs = []
                for v in vehicles:
                    act = (last_plan.get("actions") or {}).get(v) if isinstance(last_plan, dict) else None
                    if not isinstance(act, dict):
                        act = {}

                    is_waiting = v in waiting_agents
                    tgt = act.get("target")
                    if not isinstance(tgt, (list, tuple)) or len(tgt) < 2:
                        tgt = goals.get(v)
                    if not isinstance(tgt, (list, tuple)) or len(tgt) < 2:
                        fut = _move_step_async(
                            client,
                            vehicle_name=v,
                            vx=0.0,
                            vy=0.0,
                            z_ned=float(z_ned_by_vehicle.get(v, z_ned)),
                            dt_s=dt_s,
                        )
                        if fut is not None:
                            move_futs.append(fut)
                        continue

                    sp = float(act.get("speed", speed_m_s))
                    if is_waiting:
                        if lock_wait_mode == "stop":
                            sp = 0.0
                        elif lock_wait_mode == "slow":
                            sp = float(sp) * max(0.0, float(lock_wait_speed_factor))

                    x, y, _z = positions[v]
                    tx, ty = float(tgt[0]), float(tgt[1])
                    dx = tx - float(x)
                    dy = ty - float(y)
                    n = math.hypot(dx, dy)
                    if n <= 1e-6 or sp <= 1e-6:
                        fut = _move_step_async(
                            client,
                            vehicle_name=v,
                            vx=0.0,
                            vy=0.0,
                            z_ned=float(z_ned_by_vehicle.get(v, z_ned)),
                            dt_s=dt_s,
                        )
                        if fut is not None:
                            move_futs.append(fut)
                        continue

                    # Slow down near goal to avoid oscillation/overshoot.
                    sp_eff = float(sp)
                    if float(n) <= float(goal_radius_m):
                        sp_eff = 0.0
                    elif float(n) < float(goal_slowdown_dist_m):
                        sp_eff = float(sp_eff) * max(0.15, float(n) / float(goal_slowdown_dist_m))

                    vx = float(sp_eff) * dx / n
                    vy = float(sp_eff) * dy / n

                    if float(repulse_radius_m) > 1e-6 and float(repulse_gain) > 1e-6 and len(vehicles) >= 2:
                        rx = 0.0
                        ry = 0.0
                        for u in vehicles:
                            if u == v:
                                continue
                            pu = positions.get(u)
                            if pu is None:
                                continue
                            ux, uy, _uz = pu
                            ddx = float(x) - float(ux)
                            ddy = float(y) - float(uy)
                            dxy = math.hypot(ddx, ddy)
                            if dxy <= 1e-6 or dxy >= float(repulse_radius_m):
                                continue
                            # Push away in XY.
                            strength = float(repulse_gain) * (float(repulse_radius_m) - float(dxy)) / float(repulse_radius_m)
                            rx += strength * (ddx / dxy)
                            ry += strength * (ddy / dxy)
                        vx += rx
                        vy += ry
                        # Clamp to requested speed.
                        vmag = math.hypot(vx, vy)
                        if vmag > float(sp) and vmag > 1e-6:
                            vx = float(sp) * vx / vmag
                            vy = float(sp) * vy / vmag

                    fut = _move_step_async(
                        client,
                        vehicle_name=v,
                        vx=vx,
                        vy=vy,
                        z_ned=float(z_ned_by_vehicle.get(v, z_ned)),
                        dt_s=dt_s,
                    )
                    if fut is not None:
                        move_futs.append(fut)

                # Demo capture: cinematic global + representative FPVs + close chase (must show multi-agent).
                if record_demo and int(t) >= int(demo_warmup_steps) and (t % max(1, int(demo_capture_every)) == 0):
                    # Update camera rig pose to follow the group (and provide a close chase angle).
                    try:
                        cx = sum(positions[v][0] for v in vehicles) / float(len(vehicles))
                        cy = sum(positions[v][1] for v in vehicles) / float(len(vehicles))
                        # Extent controls camera distance.
                        ext = 0.0
                        for v in vehicles:
                            dx = float(positions[v][0]) - float(cx)
                            dy = float(positions[v][1]) - float(cy)
                            ext = max(ext, math.hypot(dx, dy))
                        dist = max(float(base_cfg.get("cam_global_min_dist_m", 22.0)), float(ext) * float(base_cfg.get("cam_global_dist_factor", 1.8)))
                        height = max(float(base_cfg.get("cam_global_min_height_m", 28.0)), float(ext) * float(base_cfg.get("cam_global_height_factor", 1.4)))
                        azim_deg = float(base_cfg.get("cam_global_azim_deg", 45.0))
                        pitch_deg = float(base_cfg.get("cam_global_pitch_deg", -35.0))
                        dx = -dist * math.cos(math.radians(azim_deg))
                        dy = -dist * math.sin(math.radians(azim_deg))
                        x_cam = float(cx) + float(dx)
                        y_cam = float(cy) + float(dy)
                        z_cam = float(z_ned) - float(height)
                        yaw = math.degrees(math.atan2(float(cy) - y_cam, float(cx) - x_cam))
                        _set_pose_rpy(
                            client,
                            airsim,
                            vehicle_name=cam_global_vehicle,
                            x=x_cam,
                            y=y_cam,
                            z=z_cam,
                            roll_deg=0.0,
                            pitch_deg=pitch_deg,
                            yaw_deg=yaw,
                        )

                        # Close chase: behind Drone1, looking at the group center.
                        main_v = vehicles[0]
                        mx, my, _mz = positions.get(main_v, (cx, cy, z_ned))
                        chase_dist = max(float(base_cfg.get("cam_chase_min_dist_m", 14.0)), float(ext) * float(base_cfg.get("cam_chase_dist_factor", 0.8)))
                        chase_h = max(float(base_cfg.get("cam_chase_min_height_m", 8.0)), float(ext) * float(base_cfg.get("cam_chase_height_factor", 0.5)))
                        vx = float(cx) - float(mx)
                        vy = float(cy) - float(my)
                        vn = max(1e-6, math.hypot(vx, vy))
                        bx = float(mx) - chase_dist * (vx / vn)
                        by = float(my) - chase_dist * (vy / vn)
                        bz = float(z_ned) - float(chase_h)
                        yaw2 = math.degrees(math.atan2(float(cy) - by, float(cx) - bx))
                        pitch2 = float(base_cfg.get("cam_chase_pitch_deg", -20.0))
                        _set_pose_rpy(
                            client,
                            airsim,
                            vehicle_name=cam_chase_vehicle,
                            x=bx,
                            y=by,
                            z=bz,
                            roll_deg=0.0,
                            pitch_deg=pitch2,
                            yaw_deg=yaw2,
                        )
                    except Exception:
                        pass

                    # Make agents clearly visible in the cinematic view (world-space markers).
                    if demo_plot_agents:
                        try:
                            if demo_plot_style == "points":
                                pts = [
                                    airsim.Vector3r(float(positions[v][0]), float(positions[v][1]), float(positions[v][2]))
                                    for v in vehicles
                                ]
                                client.simPlotPoints(pts, color_rgba=list(demo_plot_color), size=float(demo_plot_size), duration=0.25)
                            else:
                                segs = []
                                for v in vehicles:
                                    x0, y0, z0 = positions[v]
                                    zc = float(z0) + float(demo_plot_z_offset_m)
                                    segs.extend(
                                        _wirebox_segments(
                                            airsim,
                                            x=float(x0),
                                            y=float(y0),
                                            z=float(zc),
                                            half_xy_m=float(demo_plot_half_xy_m),
                                            half_z_m=float(demo_plot_half_z_m),
                                        )
                                    )
                                # Outline-only: pairs of points form segments.
                                client.simPlotLineList(
                                    segs,
                                    color_rgba=list(demo_plot_color),
                                    thickness=float(demo_plot_thickness),
                                    duration=0.25,
                                )
                            lbl_pos = [
                                airsim.Vector3r(float(positions[v][0]), float(positions[v][1]), float(positions[v][2]) - 2.0)
                                for v in vehicles
                            ]
                            client.simPlotStrings([str(v) for v in vehicles], lbl_pos, scale=10, color_rgba=[1.0, 1.0, 0.2, 1.0], duration=0.25)
                        except Exception:
                            pass

                    global_png = _capture_rgb_png(
                        airsim,
                        client,
                        vehicle_name=cam_global_vehicle,
                        camera="0",
                        retries=demo_image_retries,
                        base_sleep_s=demo_image_base_sleep_s,
                    )
                    if float(demo_image_sleep_s) > 1e-6:
                        time.sleep(float(demo_image_sleep_s))
                    chase_png = _capture_rgb_png(
                        airsim,
                        client,
                        vehicle_name=cam_chase_vehicle,
                        camera="0",
                        retries=demo_image_retries,
                        base_sleep_s=demo_image_base_sleep_s,
                    )
                    if float(demo_image_sleep_s) > 1e-6:
                        time.sleep(float(demo_image_sleep_s))
                    # Pick 4 representative views (always fill the 2x2 panel):
                    # - K>=4: 4 different agents, forward camera (0)
                    # - K=3: 3 agents forward + Drone1 side (1)
                    # - K=2: Drone1 forward + Drone2 forward + Drone1 side + Drone2 side
                    # - K=1: Drone1 cams 0/1/2/3
                    rep_sources: List[Tuple[str, str]] = []
                    if len(vehicles) >= 4:
                        stride = max(1, int(math.ceil(len(vehicles) / 4.0)))
                        idxs = [0, stride, 2 * stride, 3 * stride]
                        for ii in idxs:
                            if ii >= len(vehicles):
                                continue
                            rep_sources.append((vehicles[ii], "0"))
                    elif len(vehicles) == 3:
                        rep_sources = [(vehicles[0], "0"), (vehicles[1], "0"), (vehicles[2], "0"), (vehicles[0], "1")]
                    elif len(vehicles) == 2:
                        rep_sources = [(vehicles[0], "0"), (vehicles[1], "0"), (vehicles[0], "1"), (vehicles[1], "1")]
                    elif len(vehicles) == 1:
                        rep_sources = [(vehicles[0], "0"), (vehicles[0], "1"), (vehicles[0], "2"), (vehicles[0], "3")]
                    rep_pngs = []
                    for v, cam in rep_sources:
                        rep_pngs.append(
                            _capture_rgb_png(
                                airsim,
                                client,
                                vehicle_name=v,
                                camera=str(cam),
                                retries=demo_image_retries,
                                base_sleep_s=demo_image_base_sleep_s,
                            )
                        )
                        if float(demo_image_sleep_s) > 1e-6:
                            time.sleep(float(demo_image_sleep_s))

                    if global_png and chase_png:
                        overlay_lines: List[OverlayLine] = []
                        overlay_lines.append(OverlayLine(f"DomainC AirSim | {env_name} | K={len(vehicles)} | {scenario}", (255, 255, 255)))
                        trigger_hint = "-"
                        if unsafe_near_miss or any_collision:
                            trigger_hint = "unsafe"
                        elif deadlock_trigger:
                            trigger_hint = "deadlock"
                        elif failure_trigger:
                            trigger_hint = "failure"
                        elif goal_update_flag:
                            trigger_hint = "goal_update"
                        elif periodic_trigger and do_replan:
                            trigger_hint = "periodic"
                        overlay_lines.append(OverlayLine(f"{vname} | {episode_id} | t={t} | trigger={trigger_hint if do_replan else '-'}", (230, 230, 230)))
                        overlay_lines.append(OverlayLine(f"mode={mode} planner_called={planner_called}", (200, 200, 200)))
                        if mean_dist == mean_dist:
                            overlay_lines.append(OverlayLine(f"mean_dist_to_goal={mean_dist:.1f}m", (200, 200, 200)))
                        overlay_lines.append(
                            OverlayLine(
                                (
                                    f"min_dist({near_miss_metric})={min_dist:.2f}m "
                                    f"near_miss={unsafe_near_miss} collision={any_collision}"
                                ),
                                (0, 255, 255) if not any_collision else (0, 0, 255),
                            )
                        )
                        overlay_lines.append(OverlayLine(f"locks={sum(1 for z in zone_locks.values() if z.holder)} waiting={len(waiting_agents)}", (200, 200, 200)))

                        minimap = _render_minimap(positions=positions, zones=zones, size_wh=(260, 260))
                        frame_png = _compose_demo_frame(
                            global_png=global_png,
                            rep_fpv_pngs=rep_pngs,
                            chase_png=chase_png,
                            minimap_img=minimap,
                            overlay_lines=overlay_lines,
                        )
                        if not frame_png:
                            continue
                        idx = t // max(1, int(demo_capture_every))
                        frame_path = frames_panel / f"frame_{idx:06d}.png"
                        frame_path.write_bytes(frame_png)

                        # Deterministic screenshots (guarantee >= 6 per demo episode when capture succeeds).
                        if idx < 6 or do_replan or unsafe_near_miss or goal_update_flag:
                            shot = screenshots_dir / f"{vname}__{episode_id}__t{t:04d}__panel.png"
                            try:
                                encode_png(shot, decode_png(frame_png))
                            except Exception:
                                pass

                # Wait for this step's motion commands after any expensive demo capture.
                for fut in move_futs:
                    try:
                        fut.join()
                    except Exception:
                        pass

            ep_wall_ms = (time.time() - ep_t0) * 1000.0
            success = bool(reached_success and int(collisions_total) <= 0)
            end_positions = {v: positions.get(v) for v in vehicles}
            end_mean_dist = float(sum(dists.values()) / max(1, len(dists))) if dists else None

            ctx.append_episode(
                {
                    "domain": "airsim",
                    "task": base_cfg.get("task"),
                    "variant": vname,
                    "episode_id": episode_id,
                    "scene": env_name,
                    "scenario": str(scenario),
                    "K": int(len(vehicles)),
                    "success": 1.0 if success else 0.0,
                    "spl": None,
                    "step_count": int(step_count),
                    "replan_cycles": int(replan_count),
                    "planner_calls": int(planner_calls),
                    "effective_replans_per_episode": float(replan_count),
                    "replan_interval_steps": float(replan_interval_steps),
                    "trigger_cooldown_steps": float(trigger_cooldown_steps),
                    "suppressed_triggers": float(suppressed_triggers),
                    "episode_wall_time_ms": float(ep_wall_ms),
                    "effective_replans_per_min": float(replan_count) / (float(ep_wall_ms) / 60000.0) if ep_wall_ms > 0 else None,
                    "lat_p50_ms": _percentile(lat_total_samples, 0.50),
                    "lat_p95_ms": _percentile(lat_total_samples, 0.95),
                    "lat_p99_ms": _percentile(lat_total_samples, 0.99),
                    "slo_violation_rate": (float(slo_violation_count) / float(len(lat_total_samples))) if (slo_ms is not None and lat_total_samples) else None,
                    "tokens_in_mean": _mean(tokens_in_samples),
                    "tokens_in_p95": _percentile(tokens_in_samples, 0.95),
                    "tokens_in_p99": _percentile(tokens_in_samples, 0.99),
                    "tokens_after_prune_mean": _mean(tokens_after_samples),
                    "tokens_after_prune_p95": _percentile(tokens_after_samples, 0.95),
                    "tokens_after_prune_p99": _percentile(tokens_after_samples, 0.99),
                    "deadlock_flag": True if any(int(wait_steps[v]) >= int(deadlock_window_steps) for v in vehicles) else None,
                    "wait_time_ms": float(sum(wait_time_ms.values())),
                    "collisions_total": int(collisions_total),
                    "near_misses_total": int(near_miss_total),
                    "max_penetration_depth": float(max_pen_total),
                    "dist_to_goal_end_mean": end_mean_dist,
                    "episode_starts": episode_starts,
                    "episode_ends": end_positions,
                }
            )

            if record_demo and demo_emit_panels:
                demo_dir.mkdir(parents=True, exist_ok=True)
                try:
                    frames_written, w = build_video_from_frames(frames_dir=frames_panel, output_path=demo_panel_path, fps=float(demo_fps))
                    info = probe_video(demo_panel_path)
                    ctx.append_event(
                        {
                            "phase": "demo_end",
                            "variant": vname,
                            "episode_id": episode_id,
                            "path": str(demo_panel_path),
                            "fps": int(demo_fps),
                            "frame_wh": None,
                            "frames_written": int(frames_written),
                            "video_width_px": int(w),
                            "probe": info,
                        }
                    )
                except Exception as e:
                    ctx.append_event(
                        {
                            "phase": "demo_end",
                            "variant": vname,
                            "episode_id": episode_id,
                            "path": str(demo_panel_path),
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )

    # Build comparison/grid videos after all variants finished (per episode).
    if demo_enabled and demo_index:
        demo_root = Path(_PROJ_ROOT) / "artifacts" / "demos" / "airsim" / ctx.run_id
        demo_root.mkdir(parents=True, exist_ok=True)
        outputs: List[Dict[str, Any]] = []

        for episode_id, var_map in demo_index.items():
            existing = {k: v for k, v in var_map.items() if v.exists()}
            if not existing:
                continue

            # Side-by-side comparison: pick the first two variants in declared order.
            if demo_make_comparisons:
                ordered = [v for v in variant_order if v in existing]
                if len(ordered) >= 2:
                    left, right = ordered[0], ordered[1]
                    out_path = demo_root / f"compare__{left}__vs__{right}__{episode_id}.mp4"
                    try:
                        frames_written, w = build_side_by_side_video(
                            frames_left_dir=existing[left],
                            frames_right_dir=existing[right],
                            output_path=out_path,
                            fps=float(demo_fps),
                        )
                        info = probe_video(out_path)
                        outputs.append(
                            {
                                "type": "compare",
                                "episode_id": episode_id,
                                "left": left,
                                "right": right,
                                "path": str(out_path),
                                "frames_written": int(frames_written),
                                "video_width_px": int(w),
                                "probe": info,
                            }
                        )
                        ctx.append_event(
                            {
                                "phase": "demo_compare_end",
                                "episode_id": episode_id,
                                "left": left,
                                "right": right,
                                "path": str(out_path),
                                "frames_written": int(frames_written),
                                "probe": info,
                            }
                        )
                    except Exception as e:
                        ctx.append_event(
                            {
                                "phase": "demo_compare_end",
                                "episode_id": episode_id,
                                "path": str(out_path),
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )

            # 2x2 grid (strong ablation view): if at least 4 variants are present.
            if demo_make_grid_2x2:
                ordered4 = [v for v in variant_order if v in existing][:4]
                if len(ordered4) == 4:
                    out_path = demo_root / f"grid2x2__{episode_id}.mp4"
                    try:
                        frames_written, w = build_grid_video(
                            frames_dirs=[existing[v] for v in ordered4],
                            grid_wh=(2, 2),
                            output_path=out_path,
                            fps=float(demo_fps),
                        )
                        info = probe_video(out_path)
                        outputs.append(
                            {
                                "type": "grid2x2",
                                "episode_id": episode_id,
                                "variants": ordered4,
                                "path": str(out_path),
                                "frames_written": int(frames_written),
                                "video_width_px": int(w),
                                "probe": info,
                            }
                        )
                        ctx.append_event(
                            {
                                "phase": "demo_grid_end",
                                "episode_id": episode_id,
                                "variants": ordered4,
                                "path": str(out_path),
                                "frames_written": int(frames_written),
                                "probe": info,
                            }
                        )
                    except Exception as e:
                        ctx.append_event(
                            {
                                "phase": "demo_grid_end",
                                "episode_id": episode_id,
                                "path": str(out_path),
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )

        # Demo self-check manifest (double-check helper).
        manifest_path = demo_root / "demo_manifest.json"
        manifest = {
            "run_id": ctx.run_id,
            "env": env_name,
            "scenario": scenario,
            "num_agents": int(len(vehicles)),
            "demo_fps": int(demo_fps),
            "capture_every": int(demo_capture_every),
            "outputs": outputs,
            "screenshots_dir": str(demo_root / "screenshots"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        ctx.append_event({"phase": "demo_manifest_written", "path": str(manifest_path), "num_outputs": len(outputs)})

    if env_handle is not None:
        try:
            env_handle.stop(ctx=ctx)
        except Exception:
            pass

    ctx.write_summary({"status": "ok"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
