from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ACTIONS: List[str] = ["move_forward", "turn_left", "turn_right", "stop"]


def action_vocab() -> List[str]:
    return list(_ACTIONS)


def action_to_index(action: str) -> int:
    try:
        return _ACTIONS.index(str(action))
    except ValueError:
        return 0


def index_to_action(index: int) -> str:
    index = int(index)
    if index < 0 or index >= len(_ACTIONS):
        return _ACTIONS[0]
    return _ACTIONS[index]


def _extract_pointgoal(agent_state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    pointgoal = agent_state.get("pointgoal")
    if pointgoal is None:
        return None
    try:
        if len(pointgoal) >= 2:
            return float(pointgoal[0]), float(pointgoal[1])
    except Exception:
        return None
    return None


@dataclass
class PointGoalMLPMeta:
    input_dim: int = 2
    hidden_dim: int = 64
    distance_clip_m: float = 10.0
    angle_clip_rad: float = 3.141592653589793
    stop_distance_m: float = 0.2
    action_vocab: Tuple[str, ...] = tuple(_ACTIONS)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PointGoalMLPMeta":
        out = PointGoalMLPMeta()
        for k in ("input_dim", "hidden_dim", "distance_clip_m", "angle_clip_rad", "stop_distance_m"):
            if k in d:
                try:
                    setattr(out, k, type(getattr(out, k))(d[k]))
                except Exception:
                    pass
        if "action_vocab" in d and isinstance(d["action_vocab"], (list, tuple)):
            try:
                out.action_vocab = tuple(str(x) for x in d["action_vocab"])
            except Exception:
                pass
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": int(self.input_dim),
            "hidden_dim": int(self.hidden_dim),
            "distance_clip_m": float(self.distance_clip_m),
            "angle_clip_rad": float(self.angle_clip_rad),
            "stop_distance_m": float(self.stop_distance_m),
            "action_vocab": list(self.action_vocab),
        }


def build_pointgoal_mlp(*, hidden_dim: int = 64):
    import torch.nn as nn  # type: ignore

    hidden_dim = int(hidden_dim)
    return nn.Sequential(
        nn.Linear(2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, len(_ACTIONS)),
    )


def load_pointgoal_mlp(ckpt_path: str | Path, *, map_location: str = "cpu"):
    import torch  # type: ignore

    ckpt_path = Path(ckpt_path)
    obj = torch.load(str(ckpt_path), map_location=map_location)
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise ValueError(f"Invalid pointgoal-MLP checkpoint format: {ckpt_path}")
    meta = PointGoalMLPMeta.from_dict(obj.get("meta", {}) or {})
    model = build_pointgoal_mlp(hidden_dim=int(meta.hidden_dim))
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, meta


class PointGoalMLPExecutor:
    def __init__(self, model: Any, *, meta: Optional[PointGoalMLPMeta] = None, device: str = "cpu") -> None:
        self.model = model
        self.meta = meta or PointGoalMLPMeta()
        self.device = str(device)
        try:
            self.model.to(self.device)
        except Exception:
            self.device = "cpu"
            try:
                self.model.to(self.device)
            except Exception:
                pass

    def act(self, agent_state: Dict[str, Any]) -> Dict[str, str]:
        import torch  # type: ignore

        pg = _extract_pointgoal(agent_state)
        if pg is None:
            return {"action": "move_forward"}

        dist, ang = pg
        dist = max(0.0, min(float(dist), float(self.meta.distance_clip_m)))
        ang_clip = float(self.meta.angle_clip_rad)
        if ang_clip > 0:
            ang = max(-ang_clip, min(float(ang), ang_clip))

        x = torch.tensor([[dist, ang]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x)
            cls = int(torch.argmax(logits, dim=-1).item())
        action = index_to_action(cls)

        try:
            if action == "stop" and dist > float(self.meta.stop_distance_m):
                action = "move_forward"
        except Exception:
            pass
        return {"action": str(action)}


def save_pointgoal_mlp_ckpt(
    out_path: str | Path,
    *,
    state_dict: Dict[str, Any],
    meta: PointGoalMLPMeta,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    import torch  # type: ignore

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"state_dict": state_dict, "meta": meta.to_dict()}
    if extra:
        payload["extra"] = json.loads(json.dumps(extra))
    torch.save(payload, str(out_path))

