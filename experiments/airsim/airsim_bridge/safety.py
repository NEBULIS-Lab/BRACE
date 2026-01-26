from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class SafetyScore:
    collisions: int
    max_penetration_depth: float


def _penetration_depth(collision) -> float:
    try:
        return float(getattr(collision, "penetration_depth", 0.0))
    except Exception:
        return 0.0


def score_pose_path(
    *,
    client,
    airsim,
    vehicle_name: str,
    pose_points: Iterable[Tuple[float, float, float, float, float, float, float]],
    sample_every: int = 10,
    clear_z_ned: float = -80.0,
    settle_s: float = 0.02,
    penetration_threshold: float = 0.05,
) -> SafetyScore:
    """Score a candidate path by teleport-sampling and checking collision penetration.

    pose_points: iterable of (x, y, z, pitch, roll, yaw, ignore_collision_flag_as_float)
    """

    pose = client.simGetVehiclePose(vehicle_name=vehicle_name)
    clear_pose = airsim.Pose(
        airsim.Vector3r(float(pose.position.x_val), float(pose.position.y_val), float(clear_z_ned)),
        airsim.to_quaternion(0.0, 0.0, 0.0),
    )

    collisions = 0
    max_pen = 0.0

    for idx, (x, y, z, pitch, roll, yaw, _ign) in enumerate(pose_points):
        if idx % int(sample_every) != 0 and idx != 0:
            continue
        pose.position.x_val = float(x)
        pose.position.y_val = float(y)
        pose.position.z_val = float(z)
        pose.orientation = airsim.to_quaternion(float(pitch), float(roll), float(yaw))
        client.simSetVehiclePose(pose, False, vehicle_name=vehicle_name)
        time.sleep(float(settle_s))
        c = client.simGetCollisionInfo(vehicle_name=vehicle_name)
        pen = _penetration_depth(c)
        max_pen = max(max_pen, pen)
        if pen >= float(penetration_threshold) or bool(getattr(c, "has_collided", False)):
            collisions += 1
        # attempt to clear collision state between samples
        client.simSetVehiclePose(clear_pose, True, vehicle_name=vehicle_name)
        time.sleep(float(settle_s))

    return SafetyScore(collisions=collisions, max_penetration_depth=float(max_pen))


def pick_best(
    *,
    candidates: List,
    scorer: Callable[[object], SafetyScore],
) -> Tuple[object, SafetyScore]:
    best = candidates[0]
    best_score = scorer(best)
    for c in candidates[1:]:
        s = scorer(c)
        if (s.collisions, s.max_penetration_depth) < (best_score.collisions, best_score.max_penetration_depth):
            best, best_score = c, s
    return best, best_score

