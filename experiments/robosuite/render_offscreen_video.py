#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _pick_image_from_obs(obs: dict) -> np.ndarray | None:
    for key in sorted(obs.keys()):
        value = obs[key]
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3 and value.dtype == np.uint8:
            return value
    return None


def _write_gif(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v3 as iio

    duration_ms = int(round(1000.0 / float(fps)))
    iio.imwrite(path, frames, duration=duration_ms, loop=0)


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    try:
        import mediapy as media  # type: ignore

        media.write_video(path, frames, fps=fps)
        return
    except Exception:
        pass

    import imageio

    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a short RoboSuite offscreen rollout to GIF/MP4.")
    parser.add_argument("--env", default="Lift", help="robosuite env_name, e.g. Lift, Door, TwoArmPegInHole.")
    parser.add_argument("--robots", nargs="+", default=["Panda"], help="Robot list, e.g. Panda or Panda Panda.")
    parser.add_argument("--camera", default="agentview", help="Camera name.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output path (.gif or .mp4).")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    import robosuite as suite

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    make_kwargs = dict(
        env_name=str(args.env),
        robots=[str(robot) for robot in args.robots],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=str(args.camera),
        camera_heights=int(args.height),
        camera_widths=int(args.width),
        horizon=int(args.steps),
        render_gpu_device_id=int(args.render_gpu_device_id),
    )

    env = None
    try:
        try:
            env = suite.make(**make_kwargs)
        except TypeError:
            make_kwargs.pop("render_gpu_device_id", None)
            env = suite.make(**make_kwargs)

        obs = env.reset()
        low, high = env.action_spec
        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)

        frames: list[np.ndarray] = []
        for _ in range(int(args.steps)):
            action = np.random.uniform(low=low, high=high).astype(np.float32)
            obs, *_ = env.step(action)
            image = _pick_image_from_obs(obs)
            if image is None:
                try:
                    image = env.sim.render(camera_name=str(args.camera), width=int(args.width), height=int(args.height))
                except Exception:
                    image = None
            if image is None:
                raise RuntimeError(f"Could not find an RGB image in observations. Keys: {sorted(list(obs.keys()))[:30]}")
            frames.append(image)

        if out_path.suffix.lower() == ".gif":
            _write_gif(out_path, frames, fps=int(args.fps))
        else:
            _write_mp4(out_path, frames, fps=int(args.fps))
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
