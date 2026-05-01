#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


def _resolve_libero_root(cli_root: str | None) -> Path | None:
    candidates = [cli_root, os.environ.get("BRACE_LIBERO_ROOT"), os.environ.get("LIBERO_ROOT")]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser().resolve()
        if (root / "src" / "LIBERO").exists():
            return root
        if (root / "libero").exists() or (root / "benchmark_scripts").exists():
            return root
    return None


def _prepare_imports(libero_root: Path | None) -> None:
    if libero_root is None:
        return
    src_root = libero_root / "src" / "LIBERO"
    if src_root.exists():
        sys.path.insert(0, str(src_root))
    else:
        sys.path.insert(0, str(libero_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Render one LIBERO task into a side-by-side still image.")
    parser.add_argument("--benchmark-name", required=True, help="Benchmark suite name, e.g. libero_10 or libero_spatial.")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--bddl-file", required=True)
    parser.add_argument("--demo-file", required=True)
    parser.add_argument("--libero-root", default=None, help="Optional LIBERO workspace root.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    args = parser.parse_args()

    _prepare_imports(_resolve_libero_root(args.libero_root))

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_instance = benchmark.get_benchmark_dict()[args.benchmark_name]()
    task = benchmark_instance.get_task(int(args.task_id))
    init_states = benchmark_instance.get_task_init_states(int(args.task_id))

    env = OffScreenRenderEnv(
        bddl_file_name=str(args.bddl_file),
        camera_heights=128,
        camera_widths=128,
    )

    try:
        env.reset()
        obs = env.set_init_state(init_states[0])
        for _ in range(5):
            obs, _, _, _ = env.step([0.0] * 7)
        images = [obs["agentview_image"]]

        with h5py.File(str(args.demo_file), "r") as demo:
            states = demo["data/demo_0/states"][()]
            obs = env.set_init_state(states[-1])

        images.append(obs["agentview_image"])
        panel = np.concatenate(images, axis=1)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), panel[::-1, :, ::-1])
        print(out_path)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
