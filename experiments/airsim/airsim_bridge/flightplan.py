from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class FlightPlan:
    start_xy_ned: Tuple[float, float]
    target_xy_ned: Tuple[float, float]
    z_up_m: float
    bend_m: float
    orbit_radius_m: float
    orbit_turns: float
    curve_sign: float


def _normalize(vx: float, vy: float) -> Tuple[float, float]:
    n = math.hypot(vx, vy)
    if n <= 1e-9:
        return 1.0, 0.0
    return vx / n, vy / n


def choose_flightplan(
    *,
    run_id: str,
    spawn_xy_ned: Tuple[float, float],
    prefer_offroad: bool = True,
    scene_profile: str = "airsimnh",
    seed_salt: str = "",
) -> FlightPlan:
    """Pick a demo-friendly start/target near 'side areas' (e.g., houses) without user input.

    We don't have semantic map access here, so we rely on curated offsets + small jitter.
    """

    rng = random.Random(f"{run_id}:{seed_salt}:{scene_profile}")
    x0, y0 = spawn_xy_ned

    # Curated offsets (no semantic map access): we bias target_y away from 0 to avoid "always on the road".
    # Scene profiles tune ranges for demo stability and visual quality.
    profile = scene_profile.lower()
    if profile in {"abandonedpark", "park"}:
        target_dy_choices = [12.0, 18.0, 24.0, 28.0]
        z_range = (55.0, 75.0)
        bend_range = (6.0, 14.0)
        orbit_r_range = (6.0, 9.0)
        orbit_turns_range = (0.45, 0.70)
        target_dx_range = (55.0, 105.0)
        start_dx_range = (-8.0, 12.0)
        start_dy_range = (-10.0, 10.0)
    elif profile in {"landscapemountains", "mountains", "mountain"}:
        target_dy_choices = [18.0, 26.0, 34.0, 42.0]
        z_range = (85.0, 125.0)
        bend_range = (10.0, 20.0)
        orbit_r_range = (10.0, 16.0)
        orbit_turns_range = (0.40, 0.70)
        target_dx_range = (90.0, 150.0)
        start_dx_range = (-12.0, 18.0)
        start_dy_range = (-18.0, 18.0)
    # NOTE: CityEnviron was intentionally removed from the demo pack due to extraction issues.
    else:
        target_dy_choices = [18.0, 24.0, 30.0, 36.0]
        z_range = (30.0, 48.0)
        bend_range = (10.0, 22.0)
        orbit_r_range = (6.0, 10.0)
        orbit_turns_range = (0.45, 0.75)
        target_dx_range = (55.0, 110.0)
        start_dx_range = (-10.0, 15.0)
        start_dy_range = (-12.0, 12.0)
    if not prefer_offroad:
        target_dy_choices = [0.0, 10.0, -10.0]

    sign = rng.choice([-1.0, 1.0])
    target_dx = rng.uniform(*target_dx_range)
    target_dy = sign * rng.choice(target_dy_choices) + rng.uniform(-4.0, 4.0)

    # Start point: small offset so different demos don't overlap.
    start_dx = rng.uniform(*start_dx_range)
    start_dy = rng.uniform(*start_dy_range)

    z_up = rng.uniform(*z_range)
    bend = rng.uniform(*bend_range)
    orbit_r = rng.uniform(*orbit_r_range)
    orbit_turns = rng.uniform(*orbit_turns_range)

    return FlightPlan(
        start_xy_ned=(x0 + start_dx, y0 + start_dy),
        target_xy_ned=(x0 + target_dx, y0 + target_dy),
        z_up_m=z_up,
        bend_m=bend,
        orbit_radius_m=orbit_r,
        orbit_turns=orbit_turns,
        curve_sign=float(sign),
    )


def candidate_flightplans(
    *,
    run_id: str,
    spawn_xy_ned: Tuple[float, float],
    n: int = 10,
    prefer_offroad: bool = True,
    scene_profile: str = "airsimnh",
) -> List[FlightPlan]:
    return [
        choose_flightplan(
            run_id=run_id,
            spawn_xy_ned=spawn_xy_ned,
            prefer_offroad=prefer_offroad,
            scene_profile=scene_profile,
            seed_salt=str(i),
        )
        for i in range(int(n))
    ]


def sample_sine_path_xy(
    *,
    start_xy_ned: Tuple[float, float],
    target_xy_ned: Tuple[float, float],
    steps: int,
    lateral_amp_m: float,
    cycles: float,
) -> List[Tuple[float, float, float]]:
    """Return list of (x, y, yaw_rad) along a smooth, non-straight trajectory.

    - Base path is a line from start -> target.
    - Add a lateral oscillation that is 0 at endpoints (via sin(pi*s) envelope).
    - Yaw is computed from finite differences.
    """

    if steps <= 1:
        x, y = start_xy_ned
        return [(float(x), float(y), 0.0)]

    sx, sy = start_xy_ned
    tx, ty = target_xy_ned
    dx, dy = (tx - sx), (ty - sy)
    ux, uy = _normalize(dx, dy)
    nx, ny = (-uy, ux)  # left normal

    pts_xy: List[Tuple[float, float]] = []
    for i in range(steps):
        s = float(i) / float(steps - 1)
        bx = sx + dx * s
        by = sy + dy * s

        env = math.sin(math.pi * s)
        lat = float(lateral_amp_m) * math.sin(2.0 * math.pi * float(cycles) * s) * env
        x = bx + nx * lat
        y = by + ny * lat
        pts_xy.append((x, y))

    out: List[Tuple[float, float, float]] = []
    for i, (x, y) in enumerate(pts_xy):
        if i == 0:
            x2, y2 = pts_xy[1]
        else:
            x2, y2 = pts_xy[i] if i == len(pts_xy) - 1 else pts_xy[i + 1]
        vx, vy = (x2 - x), (y2 - y)
        yaw = math.atan2(vy, vx)
        out.append((float(x), float(y), float(yaw)))
    return out


def _quadratic_bezier(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], t: float) -> Tuple[float, float]:
    u = 1.0 - float(t)
    x = u * u * float(p0[0]) + 2.0 * u * float(t) * float(p1[0]) + float(t) * float(t) * float(p2[0])
    y = u * u * float(p0[1]) + 2.0 * u * float(t) * float(p1[1]) + float(t) * float(t) * float(p2[1])
    return float(x), float(y)


def _polyline_length(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        total += math.hypot(float(x2) - float(x1), float(y2) - float(y1))
    return float(total)


def _resample_polyline(points: List[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
    if n <= 1:
        return [points[0]]
    if len(points) < 2:
        return [points[0] for _ in range(int(n))]

    total = _polyline_length(points)
    if total <= 1e-9:
        return [points[0] for _ in range(int(n))]

    step = total / float(n - 1)
    out: List[Tuple[float, float]] = [points[0]]

    seg_i = 0
    seg_x1, seg_y1 = points[0]
    seg_x2, seg_y2 = points[1]
    seg_len = math.hypot(seg_x2 - seg_x1, seg_y2 - seg_y1)
    dist_into = 0.0

    for k in range(1, n - 1):
        target_dist = float(k) * float(step)
        while float(dist_into) + float(seg_len) < target_dist and seg_i < len(points) - 2:
            dist_into += seg_len
            seg_i += 1
            seg_x1, seg_y1 = points[seg_i]
            seg_x2, seg_y2 = points[seg_i + 1]
            seg_len = math.hypot(seg_x2 - seg_x1, seg_y2 - seg_y1)
        if seg_len <= 1e-9:
            out.append((float(seg_x1), float(seg_y1)))
            continue
        frac = (target_dist - float(dist_into)) / float(seg_len)
        x = float(seg_x1) + (float(seg_x2) - float(seg_x1)) * float(frac)
        y = float(seg_y1) + (float(seg_y2) - float(seg_y1)) * float(frac)
        out.append((float(x), float(y)))

    out.append(points[-1])
    return out


def sample_showcase_path_xy(
    *,
    run_id: str,
    plan: FlightPlan,
    steps: int,
    speed_m_s: float,
    dt_s: float,
    face_target: bool = True,
    orbit_samples: int = 240,
    bezier_samples: int = 220,
    final_samples: int = 120,
) -> Tuple[Tuple[float, float], List[Tuple[float, float, float]]]:
    """Generate a target-driven, cinematic path: curved approach -> partial orbit around target.

    Returns:
      (target_xy_ned_scaled, [(x, y, yaw_rad), ...]) where the output is resampled so that the
      total arc length approximately matches speed_m_s * dt_s * (steps-1).
    """

    if steps <= 1:
        x, y = plan.start_xy_ned
        return (float(plan.target_xy_ned[0]), float(plan.target_xy_ned[1])), [(float(x), float(y), 0.0)]

    sx, sy = map(float, plan.start_xy_ned)
    tx0, ty0 = map(float, plan.target_xy_ned)

    # Build an initial curve towards a point on a circle around target, then orbit around the target.
    vx, vy = (tx0 - sx), (ty0 - sy)
    ux, uy = _normalize(vx, vy)
    r0 = float(plan.orbit_radius_m)
    # Approach ends on the near side of the orbit circle (towards the start).
    ex0 = tx0 - ux * r0
    ey0 = ty0 - uy * r0

    # One smooth, purposeful bend (single curve).
    nx, ny = (-uy, ux)
    midx = 0.5 * (sx + ex0)
    midy = 0.5 * (sy + ey0)
    c1 = (midx + float(plan.curve_sign) * float(plan.bend_m) * nx, midy + float(plan.curve_sign) * float(plan.bend_m) * ny)

    curve: List[Tuple[float, float]] = []
    for i in range(int(bezier_samples)):
        t = float(i) / float(max(1, int(bezier_samples) - 1))
        curve.append(_quadratic_bezier((sx, sy), c1, (ex0, ey0), t))

    # Orbit arc around target.
    theta0 = math.atan2(ey0 - ty0, ex0 - tx0)
    theta1 = theta0 + float(plan.curve_sign) * 2.0 * math.pi * float(plan.orbit_turns)
    orbit: List[Tuple[float, float]] = []
    for i in range(int(orbit_samples)):
        t = float(i) / float(max(1, int(orbit_samples) - 1))
        th = theta0 + (theta1 - theta0) * t
        orbit.append((tx0 + r0 * math.cos(th), ty0 + r0 * math.sin(th)))

    # Final approach to the target marker (ends near/at the marker so "arrival" is obvious in overlays).
    lx0, ly0 = orbit[-1] if orbit else curve[-1]
    final_leg: List[Tuple[float, float]] = []
    for i in range(int(final_samples)):
        t = float(i) / float(max(1, int(final_samples) - 1))
        final_leg.append((float(lx0) + (tx0 - float(lx0)) * t, float(ly0) + (ty0 - float(ly0)) * t))

    poly = curve + orbit[1:] + final_leg[1:]

    # Scale geometry so total length matches desired duration and speed (helps auxline complete within video).
    desired_len = max(1.0, float(speed_m_s) * float(dt_s) * float(steps - 1))
    cur_len = _polyline_length(poly)
    scale = 1.0 if cur_len <= 1e-6 else float(desired_len) / float(cur_len)
    # Limit scale to avoid extreme teleports; better to be slightly off-speed than to explode the map.
    scale = max(0.55, min(1.60, float(scale)))

    def _scale_pt(p: Tuple[float, float]) -> Tuple[float, float]:
        return (sx + (float(p[0]) - sx) * float(scale), sy + (float(p[1]) - sy) * float(scale))

    poly_s = [_scale_pt(p) for p in poly]
    tx, ty = _scale_pt((tx0, ty0))

    pts = _resample_polyline(poly_s, int(steps))

    out: List[Tuple[float, float, float]] = []
    for i, (x, y) in enumerate(pts):
        if face_target:
            yaw = math.atan2(float(ty) - float(y), float(tx) - float(x))
        else:
            if i == 0:
                x2, y2 = pts[1]
            else:
                x2, y2 = pts[i] if i == len(pts) - 1 else pts[i + 1]
            yaw = math.atan2(float(y2) - float(y), float(x2) - float(x))
        out.append((float(x), float(y), float(yaw)))
    return (float(tx), float(ty)), out
