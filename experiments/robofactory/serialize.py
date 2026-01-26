from __future__ import annotations

import hashlib
import random
from typing import Any, Callable, Dict, List, Optional, Tuple


def approx_tokens(text: str) -> int:
    # Deterministic, dependency-free token proxy (word count).
    return max(1, len(str(text).split()))


def plan_hash(text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return h[:12]


def _to_float_list(x: Any) -> List[float]:
    if x is None:
        return []
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return [float(v) for v in x.detach().cpu().float().reshape(-1).tolist()]
    except Exception:
        pass
    try:
        import numpy as np

        arr = np.asarray(x, dtype=float).reshape(-1)
        return [float(v) for v in arr.tolist()]
    except Exception:
        return []


def _summary(x: Any, *, prefix: str) -> str:
    vals = _to_float_list(x)
    if not vals:
        return f"{prefix}_n=0"
    n = len(vals)
    mn = min(vals)
    mx = max(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
    std = var**0.5
    l2 = (sum(v * v for v in vals) ** 0.5) if vals else 0.0
    return f"{prefix}_n={n} {prefix}_mean={mean:.4f} {prefix}_std={std:.4f} {prefix}_min={mn:.4f} {prefix}_max={mx:.4f} {prefix}_l2={l2:.4f}"


def serialize_context(
    *,
    task: str,
    env_id: str,
    obs: Any,
    info: Dict[str, Any],
    history: List[Dict[str, Any]],
    retrieval_lines: Optional[List[str]] = None,
    protected_blocks: Tuple[str, ...],
    token_budget: Optional[int],
    context_compress_method: str,
    pruning_enabled: bool,
    keep_ratio: float,
    token_counter: Optional[Callable[[str], int]] = None,
    rng: Optional[random.Random] = None,
    summary_head_tokens: int = 32,
    summary_tail_tokens: int = 64,
    max_history: int = 20,
) -> Tuple[str, Dict[str, Any], str]:
    """Serialize RoboFactory state into typed blocks.

    Typed blocks:
      A: Task
      B: State (obs + success/elapsed)
      C: Safety (placeholder; domain-specific signals can be added later)
      D: Coordination (agents + waiting/locks placeholder)
      E: History (last N steps)
    """

    task = str(task)
    env_id = str(env_id)
    context_compress_method = str(context_compress_method or "none").lower()
    keep_ratio = max(0.0, min(1.0, float(keep_ratio)))

    elapsed_steps = info.get("elapsed_steps")
    success = info.get("success")
    success_bool = None
    try:
        import torch

        if isinstance(success, torch.Tensor):
            success_bool = bool(success.item())
    except Exception:
        pass
    if success_bool is None and isinstance(success, (bool, int, float)):
        success_bool = bool(success)

    blocks: Dict[str, str] = {}
    blocks["A"] = f"[A] TASK\n{task}\n"
    blocks["B"] = (
        "[B] STATE\n"
        f"env_id={env_id}\n"
        f"elapsed_steps={elapsed_steps}\n"
        f"success={success_bool}\n"
        f"{_summary(obs, prefix='obs')}\n"
    )
    blocks["C"] = "[C] SAFETY\nunsafe=false\n"
    blocks["D"] = "[D] COORDINATION\nlocks=none\nwaiting=false\n"

    recent = history[-max_history:] if max_history > 0 else []
    hist_lines: List[str] = []
    for h in recent:
        hist_lines.append(
            "t={t} rew={rew} succ={succ} obs_delta={obs_delta} act_norms={act_norms}".format(
                t=h.get("t"),
                rew=h.get("reward"),
                succ=h.get("success"),
                obs_delta=h.get("obs_delta"),
                act_norms=h.get("action_norms"),
            )
        )

    retrieval_lines = list(retrieval_lines or [])
    retrieval_lines = [f"RAG: {str(x).strip()}" for x in retrieval_lines if str(x).strip()]
    if retrieval_lines:
        # Tail placement so recency heuristics preferentially keep retrieved snippets.
        hist_lines = list(hist_lines) + ["[R] RETRIEVAL"] + retrieval_lines
    blocks["E"] = "[E] HISTORY\n" + "\n".join(hist_lines) + ("\n" if hist_lines else "none\n")

    # Respect protected blocks from BRACE controller.
    protected_blocks = tuple(str(b) for b in protected_blocks)
    protected_text = "".join(blocks[b] for b in ["A", "B", "C", "D"] if b in blocks and b in protected_blocks)
    prunable_text = "".join(blocks[b] for b in ["A", "B", "C", "D", "E"] if b in blocks and b not in protected_blocks)
    if "E" in blocks and "E" not in protected_blocks:
        prunable_text = blocks["E"]

    tokens_task = approx_tokens(blocks["A"])
    tokens_state = approx_tokens(blocks["B"])
    tokens_safety = approx_tokens(blocks["C"])
    tokens_coord = approx_tokens(blocks["D"])
    tokens_history = approx_tokens(blocks["E"])

    tokens_protected = approx_tokens(protected_text) if protected_text else 0
    tokens_prunable = approx_tokens(prunable_text) if prunable_text else 0
    tokens_in = tokens_protected + tokens_prunable

    ctx_before = protected_text + prunable_text
    ctx_after = ctx_before
    tokens_after = tokens_in
    pruned_prunable = prunable_text

    target_prunable_keep = None
    target_prunable_budget = None

    if pruning_enabled and prunable_text:
        target_prunable_keep = max(1, int(tokens_prunable * keep_ratio)) if tokens_prunable > 0 else 0
        if token_budget is not None and token_budget > 0:
            target_prunable_budget = max(0, int(token_budget) - int(tokens_protected))
        target_prunable = target_prunable_keep
        if target_prunable_budget is not None:
            target_prunable = min(target_prunable, target_prunable_budget)

        # Approximate pruning on the history region with strategy-dependent selection.
        # (History dominates the prunable region for RoboFactory.)
        header = "[E] HISTORY"
        content_lines = hist_lines[:] if hist_lines else ["none"]

        target = max(0, int(target_prunable))
        header_tokens = approx_tokens(header)
        remaining = max(0, target - header_tokens)

        def keep_suffix_lines(lines: List[str], budget_tokens: int) -> List[str]:
            kept_rev: List[str] = []
            used = 0
            for ln in reversed(lines):
                t = approx_tokens(ln)
                if kept_rev and used + t > budget_tokens:
                    break
                if used + t > budget_tokens and not kept_rev:
                    # Always keep at least one line.
                    kept_rev.append(ln)
                    break
                kept_rev.append(ln)
                used += t
            return list(reversed(kept_rev))

        def keep_prefix_lines(lines: List[str], budget_tokens: int) -> List[str]:
            kept: List[str] = []
            used = 0
            for ln in lines:
                t = approx_tokens(ln)
                if kept and used + t > budget_tokens:
                    break
                if used + t > budget_tokens and not kept:
                    kept.append(ln)
                    break
                kept.append(ln)
                used += t
            return kept

        if context_compress_method in ("recency", "recent", "tail"):
            kept_lines = keep_suffix_lines(content_lines, remaining)
        elif context_compress_method in ("random", "rand"):
            if rng is None:
                rng = random.Random(0)
            idxs = list(range(len(content_lines)))
            rng.shuffle(idxs)
            selected: List[int] = []
            used = 0
            for i in idxs:
                t = approx_tokens(content_lines[i])
                if selected and used + t > remaining:
                    continue
                if used + t > remaining and not selected:
                    selected.append(i)
                    break
                selected.append(i)
                used += t
            kept_lines = [content_lines[i] for i in sorted(set(selected))]
        elif context_compress_method in ("structured_summary", "summary", "head_tail_summary"):
            head_budget = max(0, min(int(summary_head_tokens), remaining))
            tail_budget = max(0, min(int(summary_tail_tokens), remaining))

            head_lines = keep_prefix_lines(content_lines, head_budget)
            remaining_after_head = max(0, remaining - sum(approx_tokens(x) for x in head_lines))

            tail_lines = keep_suffix_lines(content_lines[len(head_lines) :], min(tail_budget, remaining_after_head))
            remaining_after_tail = max(0, remaining_after_head - sum(approx_tokens(x) for x in tail_lines))

            # Fill the middle with placeholder "summary tokens" (proxy for expensive summarization).
            fill_n = max(0, int(remaining_after_tail))
            summary_line = ""
            if fill_n > 0:
                summary_line = ("s " * fill_n).strip()
            kept_lines = head_lines + ([summary_line] if summary_line else []) + tail_lines
        else:
            # E-RECAP (and default pruning) uses a recency-biased selection in this proxy.
            kept_lines = keep_suffix_lines(content_lines, remaining)

        # Deterministically fill to the exact budget (word-count proxy) when possible.
        # This avoids “free” under-budget baselines on binding replans.
        if context_compress_method not in ("structured_summary", "summary", "head_tail_summary"):
            used_tokens = sum(approx_tokens(x) for x in kept_lines if x)
            fill_n = max(0, int(remaining) - int(used_tokens))
            if fill_n > 0:
                kept_lines = list(kept_lines) + [("f " * fill_n).strip()]

        pruned_prunable = header + "\n" + "\n".join([ln for ln in kept_lines if ln != ""]) + "\n"
        tokens_after = tokens_protected + (approx_tokens(pruned_prunable) if pruned_prunable else 0)
        ctx_after = protected_text + pruned_prunable

        # Optional: enforce token_budget using a real token counter (e.g., HF tokenizer via HTTP service).
        if token_budget is not None and token_budget > 0 and token_counter is not None:
            try:
                while True:
                    total = int(token_counter(ctx_after))
                    if total <= int(token_budget):
                        break
                    # Shrink strategy: recency-biased removal on history lines (keep at least one line).
                    lines = [ln for ln in kept_lines if ln]
                    if not lines or (len(lines) == 1 and lines[0] not in ("none", "")):
                        break
                    if context_compress_method in ("structured_summary", "summary", "head_tail_summary"):
                        # First, truncate the summary filler line if present.
                        shrunk = False
                        for i, ln in enumerate(lines):
                            if ln.strip().startswith("s "):
                                parts = ln.split()
                                if len(parts) > 1:
                                    lines[i] = " ".join(parts[:-1])
                                    shrunk = True
                                else:
                                    lines[i] = ""
                                    shrunk = True
                                break
                        if not shrunk:
                            # Fall back to dropping the oldest non-empty line.
                            for i, ln in enumerate(lines):
                                if ln and ln != "none":
                                    lines[i] = ""
                                    break
                    else:
                        # Drop the oldest non-empty line (keeps recency).
                        for i, ln in enumerate(lines):
                            if ln and ln != "none":
                                lines[i] = ""
                                break
                    kept_lines = [ln for ln in lines if ln]
                    if not kept_lines:
                        kept_lines = ["none"]
                    pruned_prunable = header + "\n" + "\n".join([ln for ln in kept_lines if ln != ""]) + "\n"
                    ctx_after = protected_text + pruned_prunable
            except Exception:
                pass

    retrieved_tokens = sum(approx_tokens(x) for x in retrieval_lines) if retrieval_lines else 0
    kept_tokens = 0
    if retrieval_lines:
        kept_tokens = sum(approx_tokens(x) for x in ctx_after.splitlines() if str(x).startswith("RAG:"))

    stats: Dict[str, Any] = {
        "tokens_in": int(tokens_in),
        "tokens_after": int(tokens_after),
        "tokens_task": int(tokens_task),
        "tokens_state": int(tokens_state),
        "tokens_safety": int(tokens_safety),
        "tokens_coord": int(tokens_coord),
        "tokens_history": int(tokens_history),
        "retrieved_tokens": int(retrieved_tokens),
        "kept_tokens": int(kept_tokens),
        "tokens_protected": int(tokens_protected),
        "tokens_prunable": int(tokens_prunable),
        "target_prunable_keep": int(target_prunable_keep) if target_prunable_keep is not None else None,
        "target_prunable_budget": int(target_prunable_budget) if target_prunable_budget is not None else None,
        "context_length_before_chars": int(len(ctx_before)),
        "context_length_after_chars": int(len(ctx_after)),
        "protected_blocks": protected_blocks,
        "token_budget": int(token_budget) if token_budget is not None else None,
        "context_compress_method": context_compress_method,
    }
    return ctx_after, stats, ctx_before
