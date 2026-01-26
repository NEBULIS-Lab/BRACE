from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

TokenSelectStrategy = Literal[
    "erecap",  # reserved for learned importance (not shipped here)
    "recency_layerwise",
    "random_layerwise",
]


@dataclass(frozen=True)
class ErecapSelection:
    """Layerwise token selection result.

    Notes:
      - This is a *token selection* approximation used for BRACE replanning prompts.
      - In the original E-RECAP stack, pruning happens progressively in KV-cache layers.
    """

    kept_indices: List[int]
    protected_head: int
    protected_tail: int
    target_kept: int
    strategy: str


def head_tail_mask(n: int, *, head: int, tail: int) -> List[bool]:
    n = max(0, int(n))
    head = max(0, int(head))
    tail = max(0, int(tail))
    mask = [False] * n
    for i in range(min(head, n)):
        mask[i] = True
    for i in range(max(0, n - tail), n):
        mask[i] = True
    return mask


def layerwise_target_lengths(
    seq_len: int, *, prune_layers: Sequence[int] = (4, 7, 10, 13, 16, 19, 22, 25), keep_ratio: float = 0.7
) -> List[int]:
    """Compute the post-pruning token count after each pruning layer.

    This is a pure arithmetic helper used for paper-style token-count matching.
    """

    n = max(0, int(seq_len))
    r = float(keep_ratio)
    r = max(0.0, min(1.0, r))
    out: List[int] = []
    for _ in prune_layers:
        n = max(1, int(round(n * r))) if n > 0 else 0
        out.append(n)
    return out


def _sample_without_protected(
    *,
    candidate_indices: Sequence[int],
    k: int,
    rng: random.Random,
) -> List[int]:
    if k <= 0 or not candidate_indices:
        return []
    if k >= len(candidate_indices):
        return list(candidate_indices)
    return list(rng.sample(list(candidate_indices), k=k))


def select_indices(
    tokens: Sequence[str],
    *,
    target_keep: int,
    protected_head: int = 0,
    protected_tail: int = 0,
    strategy: TokenSelectStrategy = "recency_layerwise",
    rng: Optional[random.Random] = None,
    importance_scores: Optional[Sequence[float]] = None,
) -> ErecapSelection:
    """Select token indices under a target budget with head/tail protection.

    Strategy:
      - recency_layerwise: keep most recent tokens (excluding protected regions).
      - random_layerwise: uniform sample (excluding protected regions).
      - erecap: placeholder for learned importance; if `importance_scores` is provided,
        selects top-k by score (excluding protected regions), otherwise falls back to recency.
    """

    n = len(tokens)
    target_keep = max(0, int(target_keep))
    protected_head = max(0, int(protected_head))
    protected_tail = max(0, int(protected_tail))
    protected_head = min(protected_head, n)
    protected_tail = min(protected_tail, max(0, n - protected_head))

    protected_mask = head_tail_mask(n, head=protected_head, tail=protected_tail)
    protected = [i for i, m in enumerate(protected_mask) if m]
    unprotected = [i for i, m in enumerate(protected_mask) if not m]

    # Always keep protected tokens.
    remaining = max(0, target_keep - len(protected))

    kept_extra: List[int]
    strat = str(strategy)
    if strat == "recency_layerwise":
        kept_extra = unprotected[-remaining:] if remaining > 0 else []
    elif strat == "random_layerwise":
        rng = rng or random.Random(0)
        kept_extra = sorted(_sample_without_protected(candidate_indices=unprotected, k=remaining, rng=rng))
    elif strat == "erecap":
        if importance_scores is not None and len(importance_scores) == n and remaining > 0:
            scored: List[Tuple[float, int]] = [
                (float(importance_scores[i]), i) for i in unprotected if 0 <= i < len(importance_scores)
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            kept_extra = sorted([i for _, i in scored[:remaining]])
        else:
            kept_extra = unprotected[-remaining:] if remaining > 0 else []
    else:
        raise ValueError(f"Unknown token selection strategy: {strategy}")

    kept = sorted(set(protected + kept_extra))
    # If protected regions exceed the budget, keep only protected (still deterministic).
    if target_keep > 0 and len(kept) > target_keep:
        kept = kept[:target_keep]

    return ErecapSelection(
        kept_indices=kept,
        protected_head=protected_head,
        protected_tail=protected_tail,
        target_kept=target_keep,
        strategy=strategy,
    )

