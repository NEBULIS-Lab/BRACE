from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List

_PROJ_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from experiments.common.clarification import build_text_instruction
from experiments.robofactory.serialize import approx_tokens


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
    ap.add_argument("--config", default=None, help="RoboFactory JSON config to read oracle task text from")
    ap.add_argument("--task_text", default=None, help="Oracle task text (overrides --config if provided)")
    ap.add_argument("--out", default=None, help="Output JSONL manifest path (default: data/robofactory_instructions/*)")
    ap.add_argument("--styles", default="oracle,coarsened")
    ap.add_argument("--clarification_turns", default="0,1,2")
    ap.add_argument("--ambiguity_types", default="goal,process,success")
    args = ap.parse_args()

    base_cfg = {}
    if args.config:
        cfg_path = Path(args.config)
        base_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    oracle_task_text = args.task_text or base_cfg.get("task_text") or base_cfg.get("task") or ""
    oracle_task_text = str(oracle_task_text).strip()
    if not oracle_task_text:
        raise ValueError("Missing oracle task text: provide --task_text or a --config with `task_text`.")

    task_name = str(base_cfg.get("task", "robofactory_task")).strip()
    env_id = str((base_cfg.get("robofactory") or {}).get("env_id", "")).strip()

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    if args.out:
        out_path = Path(args.out)
    else:
        stem = f"{task_name}__manifest_{ts}".replace("/", "_").replace(" ", "_")
        out_path = _PROJ_ROOT / "data" / "robofactory_instructions" / f"{stem}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _parse_str_list(args.styles)
    clarification_turns = _parse_int_list(args.clarification_turns)
    ambiguity_types = _parse_str_list(args.ambiguity_types)

    with open(out_path, "w", encoding="utf-8") as f:
        for style in styles:
            for ambiguity_type in ambiguity_types:
                for k in clarification_turns:
                    res = build_text_instruction(
                        oracle_instruction=oracle_task_text,
                        style=style,  # type: ignore[arg-type]
                        clarification_budget_turns=int(k),
                        ambiguity_type=ambiguity_type,  # type: ignore[arg-type]
                        token_counter=approx_tokens,
                    )
                    rec = {
                        "domain": "robofactory",
                        "task": task_name,
                        "env_id": env_id,
                        "instruction_style": style,
                        "ambiguity_type": ambiguity_type,
                        "clarification_budget_turns": int(k),
                        "instruction": res.instruction,
                        "clarification_transcript": [t.to_dict() for t in res.clarification_transcript],
                        "clarification_tokens": int(res.clarification_tokens),
                        "oracle_instruction": oracle_task_text,
                    }
                    f.write(json.dumps(rec, sort_keys=True) + "\n")

    print(out_path)


if __name__ == "__main__":
    main()

