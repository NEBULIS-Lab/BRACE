from __future__ import annotations

from .strategies import SelectionResult, budget_target_history_keep, select_history_tokens

__all__ = [
    "SelectionResult",
    "budget_target_history_keep",
    "select_history_tokens",
    "normalize_method",
    "quality_multiplier",
    "extra_overhead_ms",
]


def normalize_method(method: str | None, *, pruning_enabled: bool) -> str:
    """Normalize context compression method names used across proxy/real runners.

    Returns one of: {"none","erecap","random","recency","structured_summary"}.
    """

    if not pruning_enabled:
        return "none"

    if method is None:
        return "erecap"

    m = str(method).strip().lower()
    if m in ("", "none", "identity", "no", "off"):
        return "none"
    if m in ("erecap", "prune", "pruning", "cost_aware_prune"):
        return "erecap"
    if m in ("random", "rand"):
        return "random"
    if m in ("recency", "recent", "tail"):
        return "recency"
    if m in ("structured_summary", "summary", "head_tail_summary"):
        return "structured_summary"
    return m


def quality_multiplier(method: str) -> float:
    """Synthetic quality multiplier for proxy-only experiments."""

    m = str(method or "none").strip().lower()
    if m in ("none", "erecap"):
        return 1.0
    if m == "recency":
        return 0.92
    if m == "structured_summary":
        return 0.90
    if m == "random":
        return 0.85
    return 1.0


def extra_overhead_ms(method: str, *, summary_overhead_ms: float = 0.0) -> float:
    """Synthetic overhead (ms) for context compression."""

    m = str(method or "none").strip().lower()
    if m == "structured_summary":
        return float(summary_overhead_ms)
    return 0.0
