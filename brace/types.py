from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

Mode = Literal["full_replan", "partial_replan", "reuse_subplan", "defer_replan"]
BlockId = Literal["A", "B", "C", "D", "E", "F"]


@dataclass(frozen=True)
class BraceTrigger:
    unsafe: bool = False
    deadlock: bool = False
    periodic: bool = False
    types: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BraceTrigger":
        return cls(
            unsafe=bool(d.get("unsafe", False)),
            deadlock=bool(d.get("deadlock", False)),
            periodic=bool(d.get("periodic", False)),
            types=tuple(d.get("types", ())) if d.get("types") is not None else (),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unsafe": self.unsafe,
            "deadlock": self.deadlock,
            "periodic": self.periodic,
            "types": list(self.types),
        }


@dataclass(frozen=True)
class BraceTelemetry:
    progress: Optional[float] = None
    lat_total_ms: Optional[float] = None
    churn: bool = False
    clarification_budget_turns: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BraceTelemetry":
        progress = d.get("progress")
        lat_ms = d.get("lat_total_ms")
        return cls(
            progress=float(progress) if isinstance(progress, (int, float)) else None,
            lat_total_ms=float(lat_ms) if isinstance(lat_ms, (int, float)) else None,
            churn=bool(d.get("churn", False)),
            clarification_budget_turns=int(d.get("clarification_budget_turns", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "churn": self.churn,
            "clarification_budget_turns": self.clarification_budget_turns,
        }
        if self.progress is not None:
            out["progress"] = float(self.progress)
        if self.lat_total_ms is not None:
            out["lat_total_ms"] = float(self.lat_total_ms)
        return out

