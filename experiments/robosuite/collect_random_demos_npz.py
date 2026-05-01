#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect random RoboSuite demonstrations using DataCollectionWrapper.")
    parser.add_argument("--env", default="Lift", help="robosuite env_name, e.g. Lift, Door, TwoArmPegInHole.")
    parser.add_argument("--robots", nargs="+", default=["Panda"], help="Robot list.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--collect-freq", type=int, default=1)
    parser.add_argument("--flush-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="artifacts/robosuite_demos", help="Output directory for ep_* folders.")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    import robosuite as suite
    from robosuite.wrappers.data_collection_wrapper import DataCollectionWrapper

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__{args.env}__{'-'.join(args.robots)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = suite.make(
        env_name=str(args.env),
        robots=[str(robot) for robot in args.robots],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        horizon=int(args.steps),
        ignore_done=True,
        seed=int(args.seed),
    )
    env = DataCollectionWrapper(
        env,
        directory=str(run_dir),
        collect_freq=int(args.collect_freq),
        flush_freq=int(args.flush_freq),
    )

    rng = np.random.default_rng(int(args.seed))
    try:
        for _ in range(int(args.episodes)):
            env.reset()
            low, high = env.action_spec
            low = np.asarray(low, dtype=np.float32)
            high = np.asarray(high, dtype=np.float32)
            for _ in range(int(args.steps)):
                action = rng.uniform(low=low, high=high).astype(np.float32)
                env.step(action)
            print(env.ep_directory)
    finally:
        env.close()

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
