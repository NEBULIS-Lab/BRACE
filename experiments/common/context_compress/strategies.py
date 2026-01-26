from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SelectionResult:
    kept_indices: List[int]
    summary_tokens: int
    meta: Dict[str, Any]


def budget_target_history_keep(*, tokens_protected: int, tokens_in: int, token_budget: Optional[int]) -> int:
    tokens_protected = max(0, int(tokens_protected))
    tokens_in = max(0, int(tokens_in))
    if token_budget is None or int(token_budget) <= 0:
        return max(0, tokens_in - tokens_protected)

    token_budget_int = int(token_budget)
    if tokens_in < token_budget_int:
        token_budget_int = tokens_in
    if token_budget_int < tokens_protected:
        raise ValueError(
            f"token_budget ({token_budget_int}) < tokens_protected ({tokens_protected}); "
            "cannot satisfy protected blocks."
        )
    return max(0, token_budget_int - tokens_protected)


def select_history_tokens(
    *,
    strategy: str,
    history_len: int,
    target_keep: int,
    rng: random.Random,
    summary_head_tokens: int = 32,
    summary_tail_tokens: int = 64,
) -> SelectionResult:
    strategy = str(strategy or "none").lower()
    history_len = max(0, int(history_len))
    target_keep = max(0, int(target_keep))

    if target_keep >= history_len:
        return SelectionResult(
            kept_indices=list(range(history_len)),
            summary_tokens=0,
            meta={"strategy": strategy, "history_len": history_len, "target_keep": target_keep},
        )

    if strategy in ("none", "identity"):
        return SelectionResult(
            kept_indices=list(range(history_len)),
            summary_tokens=0,
            meta={"strategy": strategy, "history_len": history_len, "target_keep": target_keep},
        )

    if strategy in ("recency", "recent", "tail", "erecap", "erecap_like"):
        start = max(0, history_len - target_keep)
        kept = list(range(start, history_len))
        return SelectionResult(
            kept_indices=kept,
            summary_tokens=0,
            meta={"strategy": strategy, "history_len": history_len, "target_keep": target_keep, "start": start},
        )

    if strategy in ("random", "rand"):
        kept = sorted(rng.sample(range(history_len), k=target_keep))
        return SelectionResult(
            kept_indices=kept,
            summary_tokens=0,
            meta={"strategy": strategy, "history_len": history_len, "target_keep": target_keep},
        )

    if strategy in ("structured_summary", "summary", "head_tail_summary"):
        head = max(0, int(summary_head_tokens))
        tail = max(0, int(summary_tail_tokens))

        head = min(head, target_keep, history_len)
        remaining = target_keep - head

        # Avoid overlap between head and tail regions.
        max_tail_no_overlap = max(0, history_len - head)
        tail = min(tail, remaining, max_tail_no_overlap)

        summary_tokens = max(0, target_keep - head - tail)
        kept: List[int] = []
        if head > 0:
            kept.extend(range(0, head))
        if tail > 0:
            kept.extend(range(history_len - tail, history_len))

        return SelectionResult(
            kept_indices=kept,
            summary_tokens=summary_tokens,
            meta={
                "strategy": strategy,
                "history_len": history_len,
                "target_keep": target_keep,
                "head_kept": head,
                "tail_kept": tail,
                "summary_tokens": summary_tokens,
            },
        )

    raise ValueError(f"Unknown context compression strategy: {strategy}")
