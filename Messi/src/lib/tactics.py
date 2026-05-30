"""Shared tactical helper facts derived from world poses.

Pure Python — no ROS imports. All functions take Point2D arguments and return
plain values so they can be unit-tested without a running ROS graph.
"""

import math
from typing import List, Optional

from .geometry import Point2D


def distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def distance_self_to_ball(self_pos: Point2D, ball: Point2D) -> float:
    return distance(self_pos, ball)


def distance_teammate_to_ball(teammate: Point2D, ball: Point2D) -> float:
    return distance(teammate, ball)


def nearest_opponent_to_ball(
    ball: Point2D,
    opponent_1: Optional[Point2D],
    opponent_2: Optional[Point2D],
) -> Optional[Point2D]:
    """Return the opponent closest to the ball, or None if neither is available."""
    candidates = [p for p in (opponent_1, opponent_2) if p is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda p: distance(p, ball))


def ball_velocity_magnitude(vx: float, vy: float) -> float:
    return math.hypot(vx, vy)


def ball_heading_to_goal(
    ball: Point2D,
    ball_vx: float,
    ball_vy: float,
    goal: Point2D,
) -> bool:
    """True if the ball velocity vector points within 90 degrees toward the goal."""
    dx = goal.x - ball.x
    dy = goal.y - ball.y
    dot = ball_vx * dx + ball_vy * dy
    return dot > 0.0


def shot_lane_clear(
    self_pos: Point2D,
    attack_goal: Point2D,
    opponents: List[Optional[Point2D]],
    lane_half_width: float = 0.3,
) -> bool:
    """True if no opponent is within lane_half_width of the self->goal line segment."""
    dx = attack_goal.x - self_pos.x
    dy = attack_goal.y - self_pos.y
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return True
    ux, uy = dx / length, dy / length

    for opp in opponents:
        if opp is None:
            continue
        # Project opponent onto the shot line
        ox = opp.x - self_pos.x
        oy = opp.y - self_pos.y
        proj = ox * ux + oy * uy
        if proj < 0.0 or proj > length:
            continue
        # Perpendicular distance from opponent to line
        perp = abs(ox * uy - oy * ux)
        if perp < lane_half_width:
            return False
    return True


def keeper_zone_contains_ball(
    ball: Point2D,
    defense_goal: Point2D,
    goal_width: float,
    keeper_depth: float,
) -> bool:
    """True if the ball is inside the rectangular keeper zone in front of the goal."""
    half_w = goal_width * 0.5
    inward_sign = -1.0 if defense_goal.y > 0.0 else 1.0
    y_near = defense_goal.y
    y_far = defense_goal.y + inward_sign * keeper_depth
    y_min = min(y_near, y_far)
    y_max = max(y_near, y_far)
    return (
        abs(ball.x - defense_goal.x) <= half_w
        and y_min <= ball.y <= y_max
    )


def teammate_has_better_angle(
    self_pos: Point2D,
    teammate: Optional[Point2D],
    ball: Point2D,
    attack_goal: Point2D,
) -> bool:
    """True if the teammate is closer to the ball and has a better angle to the goal."""
    if teammate is None:
        return False
    self_dist = distance(self_pos, ball)
    mate_dist = distance(teammate, ball)
    if mate_dist >= self_dist:
        return False
    # Teammate is closer; check if their angle to goal is better (smaller)
    def _angle_to_goal(pos: Point2D) -> float:
        dx = attack_goal.x - pos.x
        dy = attack_goal.y - pos.y
        db = math.hypot(ball.x - pos.x, ball.y - pos.y)
        if db < 1e-6:
            return 0.0
        dg = math.hypot(dx, dy)
        if dg < 1e-6:
            return 0.0
        # Angle between pos->ball and pos->goal
        bx, by = ball.x - pos.x, ball.y - pos.y
        gx, gy = dx, dy
        cos_a = (bx * gx + by * gy) / (db * dg)
        return math.acos(max(-1.0, min(1.0, cos_a)))

    return _angle_to_goal(teammate) < _angle_to_goal(self_pos)
