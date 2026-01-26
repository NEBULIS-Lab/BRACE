from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ReplanSchedule:
    """Minimal schedule/cooldown parameters for replanning triggers."""

    replan_interval_steps: int
    trigger_cooldown_steps: int = 0


def periodic_trigger(*, t: int, interval_steps: int, last_replan_step: int) -> bool:
    if interval_steps <= 0:
        return False
    return (t % interval_steps == 0) and (t != last_replan_step)


def allow_trigger(*, t: int, last_replan_step: int, trigger_cooldown_steps: int) -> bool:
    if last_replan_step < 0:
        return True
    if trigger_cooldown_steps <= 0:
        return True
    return (t - last_replan_step) >= trigger_cooldown_steps


def trigger_type_primary(*, periodic: bool, failure: bool, deadlock: bool) -> str:
    if deadlock:
        return "deadlock"
    if failure:
        return "failure"
    if periodic:
        return "periodic"
    return "unknown"


def trigger_types_list(*, periodic: bool, failure: bool, deadlock: bool) -> List[str]:
    out: List[str] = []
    if periodic:
        out.append("periodic")
    if failure:
        out.append("failure")
    if deadlock:
        out.append("deadlock")
    return out


def build_trigger_dict(
    *,
    periodic: bool,
    failure: bool,
    deadlock: bool,
    unsafe: bool = False,
    extra_types: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    types: List[str] = []
    if extra_types:
        types.extend([str(x) for x in extra_types])
    types.extend([x for x in [("failure" if failure else None), ("deadlock" if deadlock else None)] if x])
    return {"unsafe": bool(unsafe), "deadlock": bool(deadlock), "periodic": bool(periodic), "types": types}

