from __future__ import annotations

import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from brace.controller import BraceController, BraceHyperparams, BraceState


def main() -> None:
    hp = BraceHyperparams(
        slo_ms=100,
        cooldown_steps=3,
        min_commit_window=2,
        deadlock_window=3,
        progress_epsilon=0.01,
        churn_threshold=0.5,
        churn_ema_alpha=1.0,  # make churn_ema update immediate for testing
        partial_budget_ratio=0.5,
        max_consecutive_defers=2,
    )
    ctl = BraceController(hp)

    # Unsafe always forces full replanning.
    st = BraceState()
    dec, st = ctl.step(state=st, trigger={"unsafe": True}, telemetry={}, remaining_budget=1000)
    assert dec.mode == "full_replan"
    assert dec.token_budget == 1000
    assert dec.time_budget_ms == int(round(hp.slo_ms * hp.slo_guard_ratio))
    assert dec.reason == "unsafe"
    assert st.commit_timer == hp.min_commit_window

    # Commit window should avoid plan changes.
    dec2, st2 = ctl.step(state=st, trigger={}, telemetry={}, remaining_budget=1000)
    assert dec2.mode == "reuse_subplan"
    assert dec2.token_budget == 0
    assert dec2.time_budget_ms == 0
    assert dec2.reason == "commit_window"
    assert st2.commit_timer == hp.min_commit_window - 1

    # Churn hazard triggers cooldown + deferral.
    churn_state = BraceState(churn_ema=1.0)
    dec3, st3 = ctl.step(state=churn_state, trigger={}, telemetry={}, remaining_budget=1000)
    assert dec3.mode == "defer_replan"
    assert dec3.time_budget_ms == 0
    assert dec3.reason == "churn"
    assert st3.cooldown_timer == hp.cooldown_steps

    # Cooldown continues to defer unless forced by hazards.
    dec4, st4 = ctl.step(state=st3, trigger={}, telemetry={}, remaining_budget=1000)
    assert dec4.mode == "defer_replan"
    assert dec4.time_budget_ms == 0
    assert dec4.reason == "cooldown"
    assert st4.cooldown_timer == hp.cooldown_steps - 1

    print("brace.selfcheck_controller: OK")


if __name__ == "__main__":
    main()
