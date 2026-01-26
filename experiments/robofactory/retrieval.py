from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple


# Small, deterministic “memory” snippets to make RAG auditable in RoboFactory.
# This is intentionally lightweight: it provides a stable retrieval payload that
# (i) increases context length and (ii) can be pruned under budget, while
# remaining easy to reproduce across environments.
_CORPUS = {
    "PassShoe-rf": [
        "If handover fails, re-align grippers and retry with smaller approach steps.",
        "Avoid simultaneous grasp: assign one arm as holder and the other as placer.",
        "If shoe slips, switch to a top-down grasp and increase lift height before transfer.",
        "Keep the handover region clear; pause the receiver until the holder is stable.",
    ],
    "LiftBarrier-rf": [
        "Synchronize lift timing: wait until both grippers have firm contact before lifting.",
        "If barrier twists, lower slightly, re-center grippers, and lift again.",
        "Avoid deadlock: designate a leader arm to initiate motion and the other to follow.",
        "Use smaller joint deltas near the target height to reduce oscillation.",
    ],
    "CameraAlignment-rf": [
        "If alignment oscillates, reduce step size and enforce a short cooldown before replanning.",
        "When close to target, prefer rotation-only corrections over translation.",
        "If the camera frame jumps, reset the target pose and re-approach slowly.",
        "Avoid competing motions: let one arm stabilize while the other adjusts.",
    ],
    "TakePhoto-rf": [
        "Before triggering, ensure the camera is stable for a short dwell window.",
        "If the camera is occluded, re-position the supporting arm first.",
        "Prefer small final adjustments to avoid overshooting alignment.",
        "If trigger fails, reset to the last stable pose and retry.",
    ],
}


def _load_memory_texts(memory_jsonl: Optional[str], *, max_items: int) -> List[str]:
    if not memory_jsonl:
        return []
    p = Path(str(memory_jsonl))
    if not p.exists():
        return []
    out: List[str] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("summary") or obj.get("memory") or ""
                    text = str(text).strip()
                    if text:
                        out.append(text)
                except Exception:
                    out.append(str(line))
    except Exception:
        return []

    if max_items > 0 and len(out) > int(max_items):
        out = out[-int(max_items) :]
    return out


def retrieve_snippets(
    *,
    env_id: str,
    query: str,
    k: int,
    rng: random.Random,
    memory_jsonl: Optional[str] = None,
    max_memory_items: int = 200,
    fallback_static: bool = True,
) -> Tuple[List[str], str]:
    """Return up to k snippets and a short retrieval note (auditable).

    Retrieval sources:
    - Optional episodic memory (JSONL), written by the runner (per-variant) into the run dir.
    - Static in-code corpus as a deterministic fallback.
    """
    env_id = str(env_id)
    query = str(query or "").strip().lower()
    k = max(0, int(k))
    if k <= 0:
        return [], "k=0"

    memory_pool = _load_memory_texts(memory_jsonl, max_items=int(max_memory_items))
    static_pool = list(_CORPUS.get(env_id, [])) if bool(fallback_static) else []

    pool = list(memory_pool) + list(static_pool)
    if not pool:
        return [], f"empty_pool env_id={env_id}"

    # Basic lexical “relevance”: score snippets by token overlap with the query.
    q_words = set([w for w in query.replace(".", " ").replace(",", " ").split() if w])
    scored = []
    for s in pool:
        s_words = set([w for w in s.lower().replace(".", " ").replace(",", " ").split() if w])
        scored.append((len(q_words & s_words), s))

    # Stable shuffle among equal-score candidates.
    rng.shuffle(scored)
    scored.sort(key=lambda x: x[0], reverse=True)

    picked = [s for _, s in scored[: min(k, len(scored))]]
    note = (
        f"env_id={env_id} k={k} picked={len(picked)} "
        f"mem={len(memory_pool)} static={len(static_pool)}"
    )
    return picked, note
