from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Tuple

_PROJ_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from experiments.common.clarification import build_pointnav_instruction


def _load_habitat_wrapper():
    proj_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(proj_root / "habitat-setup" / "src"))
    from habitat_multi_agent_wrapper import HabitatMultiAgentWrapper

    return HabitatMultiAgentWrapper


def _parse_int_list(csv: str) -> List[int]:
    out: List[int] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_str_list(csv: str) -> List[str]:
    out: List[str] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(part)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--styles", default="oracle,coarsened")
    ap.add_argument("--clarification_turns", default="0,1,2")
    ap.add_argument("--ambiguity_types", default="goal,process,success")
    ap.add_argument("--success_distance_m", type=float, default=0.2)
    ap.add_argument("--max_episode_steps", type=int, default=200)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_path = Path(args.out) if args.out else (_PROJ_ROOT / "data" / "habitat_instructions" / f"pointnav_manifest_{ts}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _parse_str_list(args.styles)
    clarification_turns = _parse_int_list(args.clarification_turns)
    ambiguity_types = _parse_str_list(args.ambiguity_types)

    HabitatMultiAgentWrapper = _load_habitat_wrapper()
    env = HabitatMultiAgentWrapper(
        config_path=None,
        scene_id=None,
        max_episode_steps=int(args.max_episode_steps),
    )

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for _ in range(int(args.episodes)):
                st = env.reset()
                ep = st["episode_info"]
                episode_id = str(ep["episode_id"])
                start_pos = tuple(ep["start_position"])
                goal_pos = tuple(ep["goal_position"])

                for style in styles:
                    for ambiguity_type in ambiguity_types:
                        for k in clarification_turns:
                            res = build_pointnav_instruction(
                                start_pos=start_pos,
                                goal_pos=goal_pos,
                                style=style,  # type: ignore[arg-type]
                                clarification_budget_turns=int(k),
                                ambiguity_type=ambiguity_type,  # type: ignore[arg-type]
                                success_distance_m=float(args.success_distance_m),
                            )
                            rec = {
                                "domain": "habitat_pointnav",
                                "episode_id": episode_id,
                                "start_pos": start_pos,
                                "goal_pos": goal_pos,
                                "instruction_style": style,
                                "ambiguity_type": ambiguity_type,
                                "success_distance_m": float(args.success_distance_m),
                                "clarification_budget_turns": int(k),
                                "instruction": res.instruction,
                                "clarification_transcript": [t.to_dict() for t in res.clarification_transcript],
                                "clarification_tokens": int(res.clarification_tokens),
                            }
                            f.write(json.dumps(rec, sort_keys=True) + "\n")
    finally:
        try:
            env.close()
        except Exception:
            pass

    print(out_path)


if __name__ == "__main__":
    main()
