from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

Mode = Literal["full_replan", "partial_replan", "reuse_subplan", "defer_replan"]


@dataclass
class BraceHyperparams:
    slo_ms: int
    recovery_horizon: int = 5
    progress_epsilon: float = 0.02
    deadlock_window: int = 50
    cooldown_steps: int = 0
    min_commit_window: int = 0
    churn_threshold: float = 0.35
    churn_ema_alpha: float = 0.2
    slo_guard_ratio: float = 0.95
    partial_budget_ratio: float = 1.0
    max_consecutive_defers: int = 0


@dataclass
class BraceState:
    cooldown_timer: int = 0
    commit_timer: int = 0
    consecutive_defers: int = 0
    no_progress_steps: int = 0
    churn_ema: float = 0.0
    last_mode: Optional[Mode] = None
    last_plan_hash: Optional[str] = None
    commitments: Optional[Dict[str, Any]] = None
    locks: Optional[Dict[str, Any]] = None


@dataclass
class BraceDecision:
    mode: Mode
    token_budget: Optional[int]
    time_budget_ms: Optional[int]
    clarification_budget_turns: int
    protected_blocks: Tuple[str, ...]
    reason: str = ""
    hazard_slo: bool = False
    hazard_churn: bool = False
    hazard_deadlock: bool = False
    hazard_unsafe: bool = False
    cooldown_active: bool = False
    rollback_flag: bool = False
    min_commit_window: int = 0


class BraceController:
    """Minimal BRACE controller skeleton.

    This is intentionally conservative: it defines the interface + deterministic rules,
    but does not yet depend on any domain-specific signals.
    """

    def __init__(self, hparams: BraceHyperparams):
        self.hparams = hparams

    def step(
        self,
        *,
        state: BraceState,
        trigger: Dict[str, Any],
        telemetry: Dict[str, Any],
        remaining_budget: Optional[int] = None,
        num_agents: int = 1,
    ) -> Tuple[BraceDecision, BraceState]:
        del num_agents  # placeholder for future multi-agent policies

        # --- Hazards (inputs are deliberately permissive for proxy runners) ---
        hazard_unsafe = bool(trigger.get("unsafe", False))
        hazard_deadlock = bool(trigger.get("deadlock", False))

        progress = telemetry.get("progress")
        no_progress_steps = int(state.no_progress_steps)
        if isinstance(progress, (int, float)):
            if float(progress) < float(self.hparams.progress_epsilon):
                no_progress_steps += 1
            else:
                no_progress_steps = 0
        hazard_deadlock = hazard_deadlock or (
            self.hparams.deadlock_window > 0 and no_progress_steps >= self.hparams.deadlock_window
        )

        lat_ms = telemetry.get("lat_total_ms")
        hazard_slo = False
        if isinstance(lat_ms, (int, float)) and self.hparams.slo_ms > 0:
            hazard_slo = float(lat_ms) > float(self.hparams.slo_ms) * float(self.hparams.slo_guard_ratio)

        cooldown_active = state.cooldown_timer > 0

        # Churn is a replanning-time stability proxy (e.g., "replan again soon with
        # little progress"). Proxy runners should set `telemetry['churn']=True` when
        # they detect this; we track it as an EMA for debounce.
        hazard_churn = bool(telemetry.get("churn", False)) or (
            float(state.churn_ema) > float(self.hparams.churn_threshold)
        )

        # --- Mode selection ---
        reason = "default"
        if hazard_unsafe or hazard_deadlock:
            mode: Mode = "full_replan"
            reason = "unsafe" if hazard_unsafe else "deadlock"
        elif cooldown_active:
            mode = "defer_replan"
            reason = "cooldown"
        elif hazard_churn:
            mode = "defer_replan"
            reason = "churn"
        elif state.commit_timer > 0:
            mode = "reuse_subplan"
            reason = "commit_window"
        elif hazard_slo:
            mode = "partial_replan"
            reason = "slo"
        else:
            mode = "partial_replan"

        # Avoid indefinite deferral.
        if (
            mode == "defer_replan"
            and self.hparams.max_consecutive_defers > 0
            and state.consecutive_defers >= self.hparams.max_consecutive_defers
            and not hazard_slo
        ):
            mode = "partial_replan"
            reason = f"{reason}+defer_guard"

        protected_blocks = ("A", "B", "C", "D")

        # --- Budgeting ---
        time_budget_ms: Optional[int] = None
        if self.hparams.slo_ms > 0:
            guard_ratio = float(self.hparams.slo_guard_ratio)
            guard_ratio = max(0.0, min(1.0, guard_ratio))
            time_budget_ms = max(0, int(round(float(self.hparams.slo_ms) * guard_ratio)))
            if mode in ("reuse_subplan", "defer_replan"):
                time_budget_ms = 0

        token_budget: Optional[int]
        if remaining_budget is None:
            token_budget = None
        else:
            remaining_budget = max(0, int(remaining_budget))
            if mode == "full_replan":
                token_budget = remaining_budget
            elif mode == "partial_replan":
                token_budget = max(1, int(round(remaining_budget * float(self.hparams.partial_budget_ratio))))
            else:
                token_budget = 0

        clarification_budget_turns = int(telemetry.get("clarification_budget_turns", 0))
        if mode in ("reuse_subplan", "defer_replan") or hazard_slo:
            clarification_budget_turns = 0

        rollback_flag = bool(hazard_churn or hazard_slo)

        decision = BraceDecision(
            mode=mode,
            token_budget=token_budget,
            time_budget_ms=time_budget_ms,
            clarification_budget_turns=clarification_budget_turns,
            protected_blocks=protected_blocks,
            reason=str(reason),
            hazard_slo=hazard_slo,
            hazard_churn=hazard_churn,
            hazard_deadlock=hazard_deadlock,
            hazard_unsafe=hazard_unsafe,
            cooldown_active=cooldown_active,
            rollback_flag=rollback_flag,
            min_commit_window=int(self.hparams.min_commit_window),
        )

        # --- State update ---
        new_cooldown = max(0, int(state.cooldown_timer) - 1)
        new_commit = max(0, int(state.commit_timer) - 1)
        new_consecutive_defers = int(state.consecutive_defers)

        churn_event = 1.0 if bool(telemetry.get("churn", False)) else 0.0
        alpha = float(self.hparams.churn_ema_alpha)
        alpha = min(1.0, max(0.0, alpha))
        new_churn_ema = (1.0 - alpha) * float(state.churn_ema) + alpha * churn_event

        if hazard_churn and self.hparams.cooldown_steps > 0:
            new_cooldown = int(self.hparams.cooldown_steps)

        if mode in ("full_replan", "partial_replan") and self.hparams.min_commit_window > 0:
            new_commit = int(self.hparams.min_commit_window)
            new_consecutive_defers = 0
        elif mode == "defer_replan":
            new_consecutive_defers += 1
        else:
            new_consecutive_defers = 0

        new_state = BraceState(
            cooldown_timer=new_cooldown,
            commit_timer=new_commit,
            consecutive_defers=new_consecutive_defers,
            no_progress_steps=no_progress_steps,
            churn_ema=new_churn_ema,
            last_mode=mode,
            last_plan_hash=state.last_plan_hash,
            commitments=state.commitments,
            locks=state.locks,
        )

        return decision, new_state
