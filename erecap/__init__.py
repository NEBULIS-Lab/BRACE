"""E-RECAP: Embodied REplanning with Cost-Aware Pruning (lightweight OSS subset).

This repository uses E-RECAP as an *optional efficiency module* on the replanning path.
For open-sourcing BRACE, we ship a small, dependency-free subset focused on:
  - head/tail protected token selection
  - token-count-matched heuristic baselines (recency/random) in an E-RECAP-style interface

The full learned KV-cache pruning stack and checkpoints are intentionally not included.
"""

from .layerwise import (
    ErecapSelection,
    TokenSelectStrategy,
    head_tail_mask,
    layerwise_target_lengths,
    select_indices,
)

__all__ = [
    "ErecapSelection",
    "TokenSelectStrategy",
    "head_tail_mask",
    "layerwise_target_lengths",
    "select_indices",
]

