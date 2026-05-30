"""Geometry helpers for soccer coordinate frames.

Motion capture provides raw room/world poses. Soccer logic consumes the field
frame: origin at the midpoint between the two goals, ``+y`` from ``goal_left``
to ``goal_right``, and ``+x`` perpendicular to keep a right-handed planar
frame. Helpers here keep mocap-to-field and field-to-body math in one place so
striker, goalkeeper, locator and tests use the same convention.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Point2D:
    """Small immutable 2D point/vector used by non-message algorithms."""

    x: float
    y: float


@dataclass(frozen=True)
class FieldTransform:
    """Rigid transform from motion-capture coordinates into field coordinates."""

    origin: Point2D
    origin_z: float
    x_axis: Point2D
    y_axis: Point2D
    length: float
    width: float

    def point_to_field(self, point):
        """Transform an object with ``x`` and ``y`` attributes into field xy."""

        dx = float(point.x) - self.origin.x
        dy = float(point.y) - self.origin.y
        return Point2D(
            dx * self.x_axis.x + dy * self.x_axis.y,
            dx * self.y_axis.x + dy * self.y_axis.y,
        )

    def position_to_field(self, position):
        """Transform an object with ``x``, ``y`` and optional ``z`` attributes."""

        point = self.point_to_field(position)
        z = float(getattr(position, "z", 0.0)) - self.origin_z
        return point.x, point.y, z

    def vector_to_field(self, dx, dy):
        """Rotate a mocap-frame vector into the field frame."""

        dx = float(dx)
        dy = float(dy)
        return Point2D(
            dx * self.x_axis.x + dy * self.x_axis.y,
            dx * self.y_axis.x + dy * self.y_axis.y,
        )

    def yaw_to_field(self, yaw):
        """Rotate a mocap-frame planar yaw into the field frame."""

        heading = self.vector_to_field(math.cos(yaw), math.sin(yaw))
        return normalize_angle(math.atan2(heading.y, heading.x))


def clamp(value, lower, upper):
    """Clamp ``value`` into the closed interval ``[lower, upper]``."""

    return max(lower, min(upper, value))


def normalize_angle(angle):
    """Normalize an angle to ``[-pi, pi]`` for yaw-error control."""

    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def build_field_transform(goal_left, goal_right, width_to_length_ratio=0.55):
    """Build the canonical field frame from two goal-center mocap positions.

    ``goal_left`` and ``goal_right`` are objects with ``x`` and ``y`` attributes
    and optional ``z``. The returned transform uses the midpoint as origin,
    goal distance as field length, and ``length * width_to_length_ratio`` as
    field width.
    """

    left_x = float(goal_left.x)
    left_y = float(goal_left.y)
    right_x = float(goal_right.x)
    right_y = float(goal_right.y)
    dy_x = right_x - left_x
    dy_y = right_y - left_y
    length = math.hypot(dy_x, dy_y)
    if length < 1e-6:
        raise ValueError("goal_left and goal_right are too close to define a field frame")

    y_axis = Point2D(dy_x / length, dy_y / length)
    x_axis = Point2D(y_axis.y, -y_axis.x)
    origin = Point2D((left_x + right_x) * 0.5, (left_y + right_y) * 0.5)
    left_z = float(getattr(goal_left, "z", 0.0))
    right_z = float(getattr(goal_right, "z", 0.0))
    ratio = max(float(width_to_length_ratio), 1e-6)
    return FieldTransform(
        origin=origin,
        origin_z=(left_z + right_z) * 0.5,
        x_axis=x_axis,
        y_axis=y_axis,
        length=length,
        width=length * ratio,
    )


def yaw_from_quaternion(q):
    """Extract planar yaw from a ROS quaternion."""

    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def quaternion_from_yaw(yaw):
    """Return ``(x, y, z, w)`` quaternion components for a planar yaw."""

    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def field_to_body(dx, dy, yaw):
    """Rotate a field-frame vector into the robot body frame."""

    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    left = dx * math.cos(yaw + math.pi * 0.5) + dy * math.sin(yaw + math.pi * 0.5)
    return Point2D(forward, left)


def body_to_field(forward, left, yaw):
    """Rotate a body-frame command/vector back into the field frame."""

    dx = forward * math.cos(yaw) + left * math.cos(yaw + math.pi * 0.5)
    dy = forward * math.sin(yaw) + left * math.sin(yaw + math.pi * 0.5)
    return Point2D(dx, dy)


def distance_xy(a, b):
    """Euclidean distance between objects with ``x`` and ``y`` attributes."""

    return math.hypot(a.x - b.x, a.y - b.y)


def yaw_to_point(origin, target):
    """Yaw in the field frame from ``origin`` to ``target``."""

    return math.atan2(target.y - origin.y, target.x - origin.x)


def safe_unit(dx, dy, fallback_x=1.0, fallback_y=0.0):
    """Return a unit vector, or a configured fallback for near-zero vectors."""

    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return Point2D(fallback_x, fallback_y)
    return Point2D(dx / norm, dy / norm)
