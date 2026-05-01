#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


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
    parser = argparse.ArgumentParser(description="List LIBERO suites and optionally verify benchmark assets.")
    parser.add_argument("--libero-root", default=None, help="Optional LIBERO workspace root.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["libero_object", "libero_goal", "libero_spatial", "libero_10", "libero_90"],
        help="Benchmark names to inspect.",
    )
    parser.add_argument("--verify-files", action="store_true", help="Check that referenced BDDL, init-state, and demo files exist.")
    args = parser.parse_args()

    _prepare_imports(_resolve_libero_root(args.libero_root))

    from libero.libero import benchmark, get_libero_path

    bddl_root = Path(get_libero_path("bddl_files"))
    init_root = Path(get_libero_path("init_states"))
    datasets_root = Path(get_libero_path("datasets"))

    for benchmark_name in args.benchmarks:
        benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
        print(f"{benchmark_instance.name}: {benchmark_instance.get_num_tasks()} tasks")
        for task_id, task_name in enumerate(benchmark_instance.get_task_names()):
            task = benchmark_instance.get_task(task_id)
            line = f"  [{task_id}] {task_name}"
            if args.verify_files:
                bddl_file = bddl_root / task.problem_folder / task.bddl_file
                init_states_file = init_root / task.problem_folder / task.init_states_file
                demo_file = datasets_root / benchmark_instance.get_task_demonstration(task_id)
                status = all(path.exists() for path in (bddl_file, init_states_file, demo_file))
                line += f" :: files_ok={status}"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
