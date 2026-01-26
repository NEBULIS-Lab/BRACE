"""Stub runner to validate the new run/logging plumbing.

This does NOT execute Habitat or E-RECAP yet. It only:
- creates a run directory under runs/
- emits a few fake events/episodes

Next step: replace stubs with calls into habitat-setup and/or E-RECAP evaluation code.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from experiments.common.logging import RunContext


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--run_name", default="habitat_stub")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    cfg = {
        "domain": "habitat",
        "runner": "stub",
        "episodes": args.episodes,
        "slo_ms": 1000,
        "token_budget": 2048,
        "clarification_budget_turns": 0,
    }
    ctx = RunContext.create(args.runs_root, args.run_name, cfg)

    for ep in range(args.episodes):
        episode_id = f"ep{ep:04d}"
        for t in range(5):
            # Optional "phase" events to validate overhead accounting (VLM/VLA tracks, baselines, etc.).
            ctx.log_phase(
                phase="vlm_summarize",
                domain="habitat",
                variant="stub",
                episode_id=episode_id,
                t=t,
                lat_total_ms=20 + 2 * t,
                lat_vlm_ms=20 + 2 * t,
                vlm_model="stub_vlm",
                vlm_tokens_in=300 + 10 * t,
                vlm_tokens_out=50,
            )
            ctx.log_phase(
                phase="context_compress",
                domain="habitat",
                variant="stub",
                episode_id=episode_id,
                t=t,
                lat_total_ms=5 + t,
                lat_prune_ms=5 + t,
                tokens_in=1000 + 50 * t,
                tokens_after_prune=900 + 40 * t,
            )
            ctx.log_phase(
                phase="planner_call",
                domain="habitat",
                variant="stub",
                episode_id=episode_id,
                t=t,
                lat_total_ms=80 + 5 * t,
                lat_prefill_ms=50 + 3 * t,
                lat_decode_ms=30 + 2 * t,
            )
            ctx.log_phase(
                phase="vla_policy_call",
                domain="habitat",
                variant="stub",
                episode_id=episode_id,
                t=t,
                lat_total_ms=12 + t,
                lat_vlm_ms=12 + t,
                vlm_model="stub_vla",
                vlm_tokens_in=20 + t,
                vlm_tokens_out=4,
            )
            ctx.append_event(
                {
                    "domain": "habitat",
                    "variant": "stub",
                    "episode_id": episode_id,
                    "t": t,
                    "brace_enabled": False,
                    "pruning_enabled": False,
                    "mode": "partial_replan",
                    "tokens_in": 1000 + 50 * t,
                    "tokens_after_prune": 1000 + 50 * t,
                    "lat_total_ms": 100 + 10 * t,
                }
            )
        ctx.append_episode(
            {
                "domain": "habitat",
                "variant": "stub",
                "episode_id": episode_id,
                "success": float(random.random() > 0.3),
                "spl": random.random(),
                "step_count": 5,
                "replans": 5,
            }
        )

    ctx.write_summary({"episodes": args.episodes, "note": "stub run only"})
    print(ctx.run_dir)


if __name__ == "__main__":
    main()
