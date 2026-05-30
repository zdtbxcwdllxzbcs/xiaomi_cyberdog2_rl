"""Pure control helpers for the two-state rule goalkeeper."""

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional, Tuple

from .geometry import Point2D, clamp


LATERAL_DEFEND = "lateral_defend"
ACTIVE_INTERCEPT = "active_intercept"

_MIN_PREDICTION_SPEED = 0.06


@dataclass(frozen=True)
class GoalkeeperControlConfig:
    goal_width: float
    guard_depth: float
    guard_margin: float = 0.12
    guard_width: float = 2.8
    field_length: float = 10.0
    field_width: float = 5.55
    field_margin: float = 0.40
    activity_depth: float = 3.8
    activity_width: float = 5.0
    ball_control_distance: float = 0.45
    lost_ball_y_margin: float = 0.12
    clear_push_distance: float = 1.6
    clear_inward_pull: float = 0.9
    clear_backoff_distance: float = 0.28
    sweep_speed: float = 0.48
    sweep_max_speed: float = 0.62
    sweep_side_offset: float = 0.28
    sweep_side_gain: float = 1.1
    sweep_x_gain: float = 0.65
    sweep_x_max_speed: float = 0.32
    sweep_breakaway_distance: float = 0.85
    guard_max_speed: float = 0.55
    approach_max_speed: float = 0.70
    recover_max_speed: float = 0.75
    max_crab_speed: float = 0.65
    guard_deadband: float = 0.18
    approach_deadband: float = 0.12
    recover_deadband: float = 0.28
    slow_radius: float = 0.90
    yaw_translation_gate: float = 0.45
    recover_ball_avoid_radius: float = 1.15
    recover_robot_avoid_radius: float = 0.75
    recover_avoid_lateral: float = 1.45
    lateral_gain: float = 1.25
    drive_gain: float = 1.35
    recover_gain: float = 1.15
    latency_bias: float = 0.04
    prediction_horizon: float = 0.22
    prediction_max_distance: float = 0.30
    # Legacy fields kept for one compatibility window.
    penalty_depth: float = 0.0
    penalty_width: float = 0.0
    keeper_zone_depth: float = 0.0
    keeper_zone_width: float = 0.0
    active_zone_depth: float = 0.0
    active_zone_width: float = 0.0
    ball_prediction_delay: float = 0.0
    ball_prediction_lead: float = 0.0
    ball_prediction_max_time: float = 0.0
    lateral_prediction_horizon: float = 3.0


@dataclass(frozen=True)
class ShotMetrics:
    speed: float
    moving_toward_goal: bool
    ttc_goal: float
    ttc_guard: float
    x_at_goal: float
    x_at_guard: float


@dataclass(frozen=True)
class PredictionInfo:
    predicted: Point2D
    velocity: Point2D
    velocity_source: str
    sample_age: float
    horizon: float
    displacement: float


@dataclass(frozen=True)
class GoalkeeperPlan:
    state: str
    target: Point2D
    yaw: float
    metrics: ShotMetrics
    reason: str
    phase: str
    gain: float
    has_possession: bool
    ball_behind: bool
    prediction: PredictionInfo


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float
    wz: float


@dataclass(frozen=True)
class BallSample:
    point: Point2D
    source_time: Optional[float]
    received_time: float


class BallTracker:
    """Tracks ball pose history and predicts only enough to compensate sample age."""

    def __init__(
        self,
        history_size: int = 6,
        min_speed: float = _MIN_PREDICTION_SPEED,
        twist_sync_tolerance: float = 0.12,
        max_valid_source_age: float = 2.0,
    ):
        self.samples: Deque[BallSample] = deque(maxlen=max(int(history_size), 2))
        self.min_speed = max(float(min_speed), 0.0)
        self.twist_sync_tolerance = max(float(twist_sync_tolerance), 0.0)
        self.max_valid_source_age = max(float(max_valid_source_age), 0.1)

    def record_pose(self, point: Point2D, source_stamp=None, received_time: float = 0.0) -> None:
        self.samples.append(
            BallSample(
                point=point,
                source_time=stamp_to_seconds(source_stamp),
                received_time=float(received_time),
            )
        )

    def predict(
        self,
        now: float,
        config: GoalkeeperControlConfig,
        twist: Optional[Point2D] = None,
        twist_stamp=None,
    ) -> PredictionInfo:
        if not self.samples:
            zero = Point2D(0.0, 0.0)
            return PredictionInfo(zero, zero, "none", 0.0, 0.0, 0.0)

        latest = self.samples[-1]
        sample_age = self._sample_age(latest, now)
        velocity, velocity_source = self._velocity(twist, twist_stamp)
        if math.hypot(velocity.x, velocity.y) < self.min_speed:
            velocity = Point2D(0.0, 0.0)
            velocity_source = "none" if velocity_source == "none" else f"{velocity_source}_slow"

        return predict_ball_with_info(
            latest.point,
            velocity,
            config,
            sample_age=sample_age,
            velocity_source=velocity_source,
        )

    def _sample_age(self, sample: BallSample, now: float) -> float:
        source_age = _valid_age(sample.source_time, now, self.max_valid_source_age)
        if source_age is not None:
            return source_age
        received_age = max(float(now) - sample.received_time, 0.0)
        return min(received_age, self.max_valid_source_age)

    def _velocity(self, twist: Optional[Point2D], twist_stamp=None) -> Tuple[Point2D, str]:
        if twist is not None and self._twist_matches_latest(twist_stamp):
            return twist, "twist"

        history_velocity = self._history_velocity()
        if history_velocity is not None:
            return history_velocity, "history"
        return Point2D(0.0, 0.0), "none"

    def _twist_matches_latest(self, twist_stamp) -> bool:
        if not self.samples:
            return False
        latest_time = self.samples[-1].source_time
        twist_time = stamp_to_seconds(twist_stamp)
        if latest_time is None or twist_time is None:
            return False
        return abs(twist_time - latest_time) <= self.twist_sync_tolerance

    def _history_velocity(self) -> Optional[Point2D]:
        if len(self.samples) < 2:
            return None

        samples = list(self.samples)
        newest = samples[-1]
        previous = None
        previous_index = None
        for index in range(len(samples) - 2, -1, -1):
            sample = samples[index]
            dt = _sample_dt(sample, newest)
            if dt is not None and dt > 1e-4:
                previous = sample
                previous_index = index
                break
        if previous is None or previous_index is None:
            return None

        dt = _sample_dt(previous, newest)
        if dt is None or dt <= 1e-4:
            return None

        velocity = Point2D(
            (newest.point.x - previous.point.x) / dt,
            (newest.point.y - previous.point.y) / dt,
        )

        if previous_index > 0:
            before_previous = samples[previous_index - 1]
            earlier_dt = _sample_dt(before_previous, previous)
            if earlier_dt is not None and earlier_dt > 1e-4:
                earlier = Point2D(
                    (previous.point.x - before_previous.point.x) / earlier_dt,
                    (previous.point.y - before_previous.point.y) / earlier_dt,
                )
                if velocity.x * earlier.x + velocity.y * earlier.y < 0.0:
                    return None

        return velocity


class VelocitySmoother:
    """Jerk-limited command filter for body-frame velocity commands."""

    def __init__(
        self,
        max_linear_accel: float,
        max_linear_jerk: float,
        max_angular_accel: float,
        max_angular_jerk: float,
    ):
        self.max_linear_accel = abs(float(max_linear_accel))
        self.max_linear_jerk = abs(float(max_linear_jerk))
        self.max_angular_accel = abs(float(max_angular_accel))
        self.max_angular_jerk = abs(float(max_angular_jerk))
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.awz = 0.0

    def reset(self) -> None:
        self.vx = self.vy = self.wz = 0.0
        self.ax = self.ay = self.awz = 0.0

    def step(self, target: VelocityCommand, dt: float) -> VelocityCommand:
        dt = max(float(dt), 1e-4)
        self.vx, self.vy, self.ax, self.ay = _step_planar(
            self.vx,
            self.vy,
            self.ax,
            self.ay,
            target.vx,
            target.vy,
            self.max_linear_accel,
            self.max_linear_jerk,
            dt,
        )
        self.wz, self.awz = _step_axis(
            self.wz, self.awz, target.wz, self.max_angular_accel, self.max_angular_jerk, dt
        )
        return VelocityCommand(self.vx, self.vy, self.wz)


def stamp_to_seconds(stamp) -> Optional[float]:
    if stamp is None:
        return None
    stamp_seconds = float(getattr(stamp, "sec", 0.0)) + float(getattr(stamp, "nanosec", 0.0)) * 1e-9
    return stamp_seconds if stamp_seconds > 0.0 else None


def _valid_age(source_time: Optional[float], now: float, max_age: float) -> Optional[float]:
    if source_time is None:
        return None
    age = float(now) - source_time
    if -0.05 <= age <= max_age:
        return max(age, 0.0)
    return None


def _sample_dt(oldest: BallSample, newest: BallSample) -> Optional[float]:
    if oldest.source_time is not None and newest.source_time is not None:
        dt = newest.source_time - oldest.source_time
        if 1e-4 < dt <= 1.0:
            return dt
    dt = newest.received_time - oldest.received_time
    if 1e-4 < dt <= 1.0:
        return dt
    return None


def _step_axis(
    value: float,
    accel: float,
    target: float,
    max_accel: float,
    max_jerk: float,
    dt: float,
) -> Tuple[float, float]:
    if max_accel <= 0.0:
        return target, 0.0

    desired_accel = clamp((target - value) / dt, -max_accel, max_accel)
    if max_jerk > 0.0:
        accel = accel + clamp(desired_accel - accel, -max_jerk * dt, max_jerk * dt)
    else:
        accel = desired_accel
    accel = clamp(accel, -max_accel, max_accel)

    return value + accel * dt, accel


def _step_planar(
    vx: float,
    vy: float,
    ax: float,
    ay: float,
    target_vx: float,
    target_vy: float,
    max_accel: float,
    max_jerk: float,
    dt: float,
) -> Tuple[float, float, float, float]:
    if max_accel <= 0.0:
        return target_vx, target_vy, 0.0, 0.0

    desired_ax = (target_vx - vx) / dt
    desired_ay = (target_vy - vy) / dt
    desired_ax, desired_ay = _limit_vector(desired_ax, desired_ay, max_accel)

    if max_jerk > 0.0:
        delta_ax = desired_ax - ax
        delta_ay = desired_ay - ay
        delta_ax, delta_ay = _limit_vector(delta_ax, delta_ay, max_jerk * dt)
        ax += delta_ax
        ay += delta_ay
    else:
        ax, ay = desired_ax, desired_ay
    return vx + ax * dt, vy + ay * dt, ax, ay


def _limit_vector(x: float, y: float, limit: float) -> Tuple[float, float]:
    magnitude = math.hypot(x, y)
    if magnitude <= limit or magnitude <= 1e-9:
        return x, y
    scale = limit / magnitude
    return x * scale, y * scale


def damped_speed(distance: float, gain: float, max_speed: float, deadband: float) -> float:
    """Distance-proportional speed with a hard deadband for sim-to-real damping."""
    distance = max(float(distance), 0.0)
    deadband = max(float(deadband), 0.0)
    if distance <= deadband:
        return 0.0
    return min(max(float(max_speed), 0.0), max(float(gain), 0.0) * (distance - deadband))


def lateral_push_command(
    ball_body: Point2D,
    defense_goal: Point2D,
    config: GoalkeeperControlConfig,
) -> VelocityCommand:
    """Closed-loop side push: keep the ball on the pushing side while moving upfield."""
    sign = away_sign(defense_goal)
    desired_side_offset = sign * max(float(config.sweep_side_offset), 0.0)
    side_error = ball_body.y - desired_side_offset
    vx = clamp(
        float(config.sweep_x_gain) * ball_body.x,
        -max(float(config.sweep_x_max_speed), 0.0),
        max(float(config.sweep_x_max_speed), 0.0),
    )
    vy = sign * max(float(config.sweep_speed), 0.0) + float(config.sweep_side_gain) * side_error
    vy = clamp(vy, -max(float(config.sweep_max_speed), 0.0), max(float(config.sweep_max_speed), 0.0))
    return VelocityCommand(vx, vy, 0.0)


def segment_distance_to_point(start: Point2D, end: Point2D, point: Point2D) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return _distance(start, point)
    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq
    t = clamp(t, 0.0, 1.0)
    closest = Point2D(start.x + dx * t, start.y + dy * t)
    return _distance(closest, point)


def segment_intersects_circle(start: Point2D, end: Point2D, center: Point2D, radius: float) -> bool:
    return segment_distance_to_point(start, end, center) <= max(float(radius), 0.0)


def recover_avoidance_target(
    self_point: Point2D,
    guard_target: Point2D,
    ball: Point2D,
    defense_goal: Point2D,
    config: GoalkeeperControlConfig,
    robot_obstacles: Iterable[Point2D] = (),
) -> Tuple[Point2D, bool]:
    """Return a temporary waypoint when the direct recovery segment would sweep the ball."""
    del defense_goal
    obstacles = [(ball, max(config.recover_ball_avoid_radius, 0.0))]
    robot_radius = max(config.recover_robot_avoid_radius, 0.0)
    obstacles.extend((obstacle, robot_radius) for obstacle in robot_obstacles)

    for obstacle, radius in obstacles:
        if radius <= 0.0:
            continue
        if segment_intersects_circle(self_point, guard_target, obstacle, radius):
            return _avoidance_waypoint(self_point, guard_target, obstacle, radius, config), True
    return guard_target, False


def _avoidance_waypoint(
    self_point: Point2D,
    guard_target: Point2D,
    obstacle: Point2D,
    radius: float,
    config: GoalkeeperControlConfig,
) -> Point2D:
    side = 1.0 if self_point.x >= obstacle.x else -1.0
    if abs(self_point.x - obstacle.x) < 1e-3:
        side = 1.0 if guard_target.x >= obstacle.x else -1.0
    lateral_clearance = max(
        radius + max(float(config.recover_deadband), 0.0) + 0.12,
        float(config.recover_avoid_lateral),
    )
    if abs(self_point.x - obstacle.x) < max(radius - 0.02, 0.0):
        return _clamp_to_field(Point2D(obstacle.x + side * lateral_clearance, self_point.y), config)
    return _clamp_to_field(Point2D(self_point.x, guard_target.y), config)


def away_sign(defense_goal: Point2D) -> float:
    return 1.0 if defense_goal.y <= 0.0 else -1.0


def guard_line_y(defense_goal: Point2D, guard_depth: float) -> float:
    return defense_goal.y + away_sign(defense_goal) * float(guard_depth)


def lateral_stance_yaw() -> float:
    return 0.0


def shot_metrics(
    ball: Point2D,
    velocity: Optional[Point2D],
    defense_goal: Point2D,
    guard_y: float,
) -> ShotMetrics:
    if velocity is None:
        velocity = Point2D(0.0, 0.0)
    speed = math.hypot(velocity.x, velocity.y)
    moving_toward_goal = (defense_goal.y - ball.y) * velocity.y > 0.0
    ttc_goal = _time_to_y(ball, velocity, defense_goal.y)
    ttc_guard = _time_to_y(ball, velocity, guard_y)
    x_at_goal = ball.x + velocity.x * ttc_goal if math.isfinite(ttc_goal) else ball.x
    x_at_guard = ball.x + velocity.x * ttc_guard if math.isfinite(ttc_guard) else ball.x
    return ShotMetrics(speed, moving_toward_goal, ttc_goal, ttc_guard, x_at_goal, x_at_guard)


def _time_to_y(ball: Point2D, velocity: Point2D, target_y: float) -> float:
    if abs(velocity.y) <= 1e-5:
        return math.inf
    ttc = (target_y - ball.y) / velocity.y
    return ttc if ttc >= 0.0 else math.inf


def goalkeeper_plan(
    ball: Point2D,
    ball_velocity: Optional[Point2D],
    defense_goal: Point2D,
    config: GoalkeeperControlConfig,
    self_point: Optional[Point2D] = None,
    prediction: Optional[PredictionInfo] = None,
    recovering: bool = False,
) -> GoalkeeperPlan:
    prediction = prediction or predict_ball_with_info(ball, ball_velocity, config)
    guard_y = guard_line_y(defense_goal, config.guard_depth)
    metrics = shot_metrics(prediction.predicted, prediction.velocity, defense_goal, guard_y)
    ball_behind = bool(self_point and _ball_behind_self(ball, self_point, defense_goal, config))
    has_possession = bool(
        self_point
        and not ball_behind
        and _distance(self_point, ball) <= max(config.ball_control_distance, 0.0)
    )

    if recovering or ball_behind:
        return _plan(
            LATERAL_DEFEND,
            "recover_guard",
            _guard_home(defense_goal, config),
            metrics,
            "ball_behind_recover" if ball_behind else "recovering_to_guard",
            config.recover_gain,
            has_possession,
            ball_behind,
            prediction,
        )

    if has_possession:
        return _plan(
            ACTIVE_INTERCEPT,
            "dribble_clear",
            _clear_push_target(ball, defense_goal, config),
            metrics,
            "has_ball_drive_clear",
            config.drive_gain,
            has_possession,
            ball_behind,
            prediction,
        )

    if _inside_activity_zone(ball, defense_goal, config):
        distance = _distance(self_point, ball) if self_point is not None else math.inf
        if distance <= max(config.ball_control_distance, 0.0):
            phase = "push_clear"
            target = _clear_push_target(ball, defense_goal, config)
        else:
            phase = "approach_clear"
            target = _clear_approach_target(ball, defense_goal, config)
        return _plan(
            ACTIVE_INTERCEPT,
            phase,
            target,
            metrics,
            "inside_activity_zone",
            config.drive_gain,
            has_possession,
            ball_behind,
            prediction,
        )

    return _plan(
        LATERAL_DEFEND,
        "guard",
        _lateral_guard_target(prediction.predicted, defense_goal, metrics, config),
        metrics,
        "outside_activity_zone",
        config.lateral_gain,
        has_possession,
        ball_behind,
        prediction,
    )


def _plan(
    state: str,
    phase: str,
    target: Point2D,
    metrics: ShotMetrics,
    reason: str,
    gain: float,
    has_possession: bool,
    ball_behind: bool,
    prediction: PredictionInfo,
) -> GoalkeeperPlan:
    return GoalkeeperPlan(
        state=state,
        target=target,
        yaw=lateral_stance_yaw(),
        metrics=metrics,
        reason=reason,
        phase=phase,
        gain=float(gain),
        has_possession=has_possession,
        ball_behind=ball_behind,
        prediction=prediction,
    )


def predict_ball(
    ball: Point2D,
    velocity: Optional[Point2D],
    config: GoalkeeperControlConfig,
    sample_age: float = 0.0,
) -> Point2D:
    return predict_ball_with_info(ball, velocity, config, sample_age=sample_age).predicted


def predict_ball_with_info(
    ball: Point2D,
    velocity: Optional[Point2D],
    config: GoalkeeperControlConfig,
    sample_age: float = 0.0,
    velocity_source: str = "direct",
) -> PredictionInfo:
    velocity = velocity or Point2D(0.0, 0.0)
    speed = math.hypot(velocity.x, velocity.y)
    if speed < _MIN_PREDICTION_SPEED:
        source = "none" if velocity_source in ("direct", "none") else velocity_source
        return PredictionInfo(ball, Point2D(0.0, 0.0), source, max(sample_age, 0.0), 0.0, 0.0)

    horizon = clamp(
        max(float(sample_age), 0.0) + max(float(config.latency_bias), 0.0),
        0.0,
        max(float(config.prediction_horizon), 0.0),
    )
    dx = velocity.x * horizon
    dy = velocity.y * horizon
    displacement = math.hypot(dx, dy)
    max_distance = max(float(config.prediction_max_distance), 0.0)
    if max_distance > 0.0 and displacement > max_distance:
        scale = max_distance / displacement
        dx *= scale
        dy *= scale
        displacement = max_distance

    return PredictionInfo(
        predicted=Point2D(ball.x + dx, ball.y + dy),
        velocity=velocity,
        velocity_source=velocity_source,
        sample_age=max(sample_age, 0.0),
        horizon=horizon,
        displacement=displacement,
    )


def _activity_depth(config: GoalkeeperControlConfig) -> float:
    if config.activity_depth > 0.0:
        return config.activity_depth
    if config.keeper_zone_depth > 0.0:
        return config.keeper_zone_depth
    if config.active_zone_depth > 0.0:
        return config.active_zone_depth
    return max(config.field_length * 0.35, config.guard_depth)


def _activity_width(config: GoalkeeperControlConfig) -> float:
    if config.activity_width > 0.0:
        return config.activity_width
    if config.keeper_zone_width > 0.0:
        return config.keeper_zone_width
    if config.active_zone_width > 0.0:
        return config.active_zone_width
    return config.field_width


def _guard_width(config: GoalkeeperControlConfig) -> float:
    if config.guard_width > 0.0:
        return config.guard_width
    if config.penalty_width > 0.0:
        return max(config.penalty_width - config.guard_margin * 2.0, config.goal_width)
    return max(config.goal_width, 0.1)


def _goal_depth(ball: Point2D, defense_goal: Point2D) -> float:
    return abs(ball.y - defense_goal.y)


def _inside_width(ball: Point2D, defense_goal: Point2D, width: float) -> bool:
    return abs(ball.x - defense_goal.x) <= max(float(width), 0.0) * 0.5


def _inside_activity_zone(
    ball: Point2D,
    defense_goal: Point2D,
    config: GoalkeeperControlConfig,
) -> bool:
    return _goal_depth(ball, defense_goal) <= _activity_depth(config) and _inside_width(
        ball, defense_goal, _activity_width(config)
    )


def _lateral_guard_target(
    ball: Point2D,
    defense_goal: Point2D,
    metrics: ShotMetrics,
    config: GoalkeeperControlConfig,
) -> Point2D:
    half_width = max(_guard_width(config) * 0.5, config.goal_width * 0.5)
    use_guard_intersection = (
        metrics.moving_toward_goal
        and math.isfinite(metrics.ttc_guard)
        and metrics.ttc_guard <= config.lateral_prediction_horizon
    )
    target_x = metrics.x_at_guard if use_guard_intersection else defense_goal.x
    return Point2D(
        clamp(target_x, defense_goal.x - half_width, defense_goal.x + half_width),
        guard_line_y(defense_goal, config.guard_depth),
    )


def _guard_home(defense_goal: Point2D, config: GoalkeeperControlConfig) -> Point2D:
    return Point2D(defense_goal.x, guard_line_y(defense_goal, config.guard_depth))


def _clear_direction(ball: Point2D, defense_goal: Point2D, config: GoalkeeperControlConfig) -> Point2D:
    sign = away_sign(defense_goal)
    clear_dx = clamp(defense_goal.x - ball.x, -config.clear_inward_pull, config.clear_inward_pull)
    clear_dy = sign * max(config.clear_push_distance, 1e-6)
    length = max(math.hypot(clear_dx, clear_dy), 1e-6)
    return Point2D(clear_dx / length, clear_dy / length)


def _clear_approach_target(ball: Point2D, defense_goal: Point2D, config: GoalkeeperControlConfig) -> Point2D:
    return _clamp_to_field(ball, config)


def _clear_push_target(ball: Point2D, defense_goal: Point2D, config: GoalkeeperControlConfig) -> Point2D:
    clear_unit = _clear_direction(ball, defense_goal, config)
    return _clamp_to_field(
        Point2D(
            ball.x + clear_unit.x * config.clear_push_distance,
            ball.y + clear_unit.y * config.clear_push_distance,
        ),
        config,
    )


def _clamp_to_field(point: Point2D, config: GoalkeeperControlConfig) -> Point2D:
    margin = max(float(config.field_margin), 0.05)
    x_limit = max(config.field_width * 0.5 - margin, 0.0)
    y_limit = max(config.field_length * 0.5 - margin, 0.0)
    return Point2D(
        clamp(point.x, -x_limit, x_limit),
        clamp(point.y, -y_limit, y_limit),
    )


def _ball_behind_self(
    ball: Point2D,
    self_point: Point2D,
    defense_goal: Point2D,
    config: GoalkeeperControlConfig,
) -> bool:
    return (ball.y - self_point.y) * away_sign(defense_goal) < -max(config.lost_ball_y_margin, 0.0)


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)
