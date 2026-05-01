#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np


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
    parser = argparse.ArgumentParser(description="Playback a RoboSuite DataCollectionWrapper episode to MP4.")
    parser.add_argument("--ep-dir", required=True, help="Path to an ep_* directory containing model.xml + state_*.npz.")
    parser.add_argument("--env", default="Lift", help="Bootstrap env name for suite.make.")
    parser.add_argument("--robots", nargs="+", default=["Panda"], help="Bootstrap robots for suite.make.")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out", required=True, help="Output .mp4 path.")
    parser.add_argument("--max-frames", type=int, default=600)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    import robosuite as suite

    episode_dir = Path(args.ep_dir)
    xml_path = episode_dir / "model.xml"
    if not episode_dir.exists() or not xml_path.exists():
        raise FileNotFoundError(f"Missing episode directory or model.xml: {episode_dir}")

    env = suite.make(
        env_name=str(args.env),
        robots=[str(robot) for robot in args.robots],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=False,
        camera_names=str(args.camera),
        camera_heights=int(args.height),
        camera_widths=int(args.width),
        ignore_done=True,
    )

    try:
        env.reset_from_xml_string(xml_path.read_text())
        env.sim.reset()

        state_paths = sorted(glob(str(episode_dir / "state_*.npz")))
        if not state_paths:
            raise FileNotFoundError(f"No state_*.npz found in {episode_dir}")

        frames: list[np.ndarray] = []
        total = 0
        for state_path in state_paths:
            states = np.load(state_path, allow_pickle=True)["states"]
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                image = env.sim.render(camera_name=str(args.camera), width=int(args.width), height=int(args.height))
                frames.append(image)
                total += 1
                if total >= int(args.max_frames):
                    break
            if total >= int(args.max_frames):
                break

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_mp4(out_path, frames, fps=int(args.fps))

        manifest = {
            "ep_dir": str(episode_dir),
            "env_bootstrap": str(args.env),
            "robots_bootstrap": [str(robot) for robot in args.robots],
            "camera": str(args.camera),
            "width": int(args.width),
            "height": int(args.height),
            "fps": int(args.fps),
            "frames_written": int(len(frames)),
        }
        out_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
        print(out_path)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
