"""Two-state goalkeeper role controller.

The controller keeps the dog sideways in both defense states. State changes only
move the target point between the guard line and the predicted ball position.
"""

import json
import math
from typing import Optional, Tuple

from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile
from std_msgs.msg import String

from .lib.geometry import Point2D, clamp, field_to_body, normalize_angle, yaw_from_quaternion
from .lib.goalkeeper_control import (
    ACTIVE_INTERCEPT,
    LATERAL_DEFEND,
    BallTracker,
    GoalkeeperControlConfig,
    VelocityCommand,
    VelocitySmoother,
    damped_speed,
    lateral_push_command,
    goalkeeper_plan,
    lateral_stance_yaw,
    recover_avoidance_target,
)
from .lib.topics import global_soccer_topic, soccer_topic

_ACTIVE_GAME_STATES = {"PLAYING", "KICKOFF_TEAM_A", "KICKOFF_TEAM_B"}


class GoalkeeperNode(Node):
    """Rule goalkeeper with exactly two behavioral states."""

    LATERAL_DEFEND = LATERAL_DEFEND
    ACTIVE_INTERCEPT = ACTIVE_INTERCEPT
    STATES = (LATERAL_DEFEND, ACTIVE_INTERCEPT)

    def __init__(self, config, move_commander, name="goalkeeper"):
        super().__init__(name)
        self.config = config
        self.move = move_commander
        self.low_latency_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        field_config = config.get("field", {})
        goalkeeper_config = config.get("goalkeeper", {})
        move_config = config.get("move", {})
        path_config = config.get("path", {})

        self.field_length = float(field_config.get("length", 10.0))
        self.field_width = float(field_config.get("width", 5.55))
        self.goal_width = float(field_config.get("goal_width", 1.5))
        self.field_margin = _cfg_float(
            goalkeeper_config,
            "field_margin",
            default=float(field_config.get("boundary_margin", 0.15)) + 0.25,
        )
        legacy_position_gain = float(goalkeeper_config.get("position_gain", 0.9))
        legacy_clear_gain = float(goalkeeper_config.get("clear_gain", 1.35))
        self.guard_depth = _cfg_float(goalkeeper_config, "guard_depth", default=0.30)
        self.guard_width = _cfg_float(
            goalkeeper_config,
            "guard_width",
            "penalty_width",
            "keeper_zone_width",
            default=max(self.goal_width + 1.3, 2.8),
        )
        self.guard_margin = _cfg_float(goalkeeper_config, "guard_margin", default=0.12)
        self.lateral_gain = _cfg_float(
            goalkeeper_config,
            "lateral_gain",
            "position_gain",
            default=legacy_position_gain,
        )
        self.drive_gain = _cfg_float(
            goalkeeper_config,
            "drive_gain",
            "clear_gain",
            default=legacy_clear_gain,
        )
        self.recover_gain = _cfg_float(
            goalkeeper_config,
            "recover_gain",
            "position_gain",
            default=legacy_position_gain,
        )
        self.yaw_gain = _cfg_float(goalkeeper_config, "yaw_gain", default=1.2)
        self.ball_control_distance = _cfg_float(
            goalkeeper_config,
            "ball_control_distance",
            "possession_distance",
            "clear_contact_distance",
            default=0.45,
        )
        self.lost_ball_y_margin = _cfg_float(goalkeeper_config, "lost_ball_y_margin", default=0.12)
        self.recover_guard_tolerance = _cfg_float(goalkeeper_config, "recover_guard_tolerance", default=0.25)
        self.recover_guard_x_tolerance = _cfg_float(
            goalkeeper_config,
            "recover_guard_x_tolerance",
            default=max(self.recover_guard_tolerance, self.guard_width * 0.22),
        )
        self.recover_guard_y_tolerance = _cfg_float(
            goalkeeper_config,
            "recover_guard_y_tolerance",
            default=max(self.recover_guard_tolerance, 0.35),
        )
        self.pose_timeout = _cfg_float(goalkeeper_config, "pose_timeout", default=0.35)
        self.ball_timeout = _cfg_float(goalkeeper_config, "ball_timeout", default=0.45)
        self.guard_max_speed = _cfg_float(goalkeeper_config, "guard_max_speed", default=0.55)
        self.approach_max_speed = _cfg_float(goalkeeper_config, "approach_max_speed", default=0.70)
        self.recover_max_speed = _cfg_float(goalkeeper_config, "recover_max_speed", default=0.75)
        self.sweep_max_speed = _cfg_float(goalkeeper_config, "sweep_max_speed", default=0.62)
        self.max_crab_speed = _cfg_float(goalkeeper_config, "max_crab_speed", default=0.65)
        self.guard_deadband = _cfg_float(goalkeeper_config, "guard_deadband", default=0.18)
        self.approach_deadband = _cfg_float(goalkeeper_config, "approach_deadband", default=0.12)
        self.recover_deadband = _cfg_float(goalkeeper_config, "recover_deadband", default=0.28)
        self.slow_radius = _cfg_float(goalkeeper_config, "slow_radius", default=0.90)
        self.yaw_translation_gate = _cfg_float(goalkeeper_config, "yaw_translation_gate", default=0.45)
        self.recover_ball_avoid_radius = _cfg_float(
            goalkeeper_config,
            "recover_ball_avoid_radius",
            default=1.15,
        )
        self.recover_robot_avoid_radius = _cfg_float(
            goalkeeper_config,
            "recover_robot_avoid_radius",
            default=0.75,
        )
        self.recover_avoid_lateral = _cfg_float(
            goalkeeper_config,
            "recover_avoid_lateral",
            default=1.45,
        )
        self.goal_tolerance = float(path_config.get("goal_tolerance", 0.18))
        self.walk_motion_id = int(move_config.get("motion_id", 303))

        self.control_rate_hz = float(goalkeeper_config.get("control_rate_hz", 10.0))
        self.control_dt = 1.0 / max(self.control_rate_hz, 1.0)
        self.debug_log_hz = float(goalkeeper_config.get("debug_log_hz", 0.0))
        self._debug_log_period = 1.0 / self.debug_log_hz if self.debug_log_hz > 0.0 else 0.0
        self.max_linear_accel = _cfg_float(
            goalkeeper_config,
            "command_accel_limit",
            "max_linear_accel",
            default=1.2,
        )
        self.max_linear_jerk = _cfg_float(
            goalkeeper_config,
            "command_jerk_limit",
            "max_linear_jerk",
            default=4.0,
        )
        self.max_angular_accel = float(goalkeeper_config.get("max_angular_accel", 1.8))
        self.max_angular_jerk = float(goalkeeper_config.get("max_angular_jerk", 5.0))
        self._smoother = VelocitySmoother(
            self.max_linear_accel,
            self.max_linear_jerk,
            self.max_angular_accel,
            self.max_angular_jerk,
        )

        self._control_config = GoalkeeperControlConfig(
            goal_width=self.goal_width,
            guard_depth=self.guard_depth,
            guard_margin=self.guard_margin,
            guard_width=self.guard_width,
            field_length=self.field_length,
            field_width=self.field_width,
            field_margin=self.field_margin,
            activity_depth=_cfg_float(
                goalkeeper_config,
                "activity_depth",
                "keeper_zone_depth",
                "active_zone_depth",
                default=3.8,
            ),
            activity_width=_cfg_float(
                goalkeeper_config,
                "activity_width",
                "keeper_zone_width",
                "active_zone_width",
                default=self.field_width,
            ),
            ball_control_distance=self.ball_control_distance,
            lost_ball_y_margin=self.lost_ball_y_margin,
            clear_push_distance=_cfg_float(
                goalkeeper_config,
                "clear_push_distance",
                "clear_forward_distance",
                default=1.6,
            ),
            clear_inward_pull=_cfg_float(
                goalkeeper_config,
                "clear_inward_pull",
                "clear_center_pull",
                default=0.9,
            ),
            clear_backoff_distance=_cfg_float(goalkeeper_config, "clear_backoff_distance", default=0.28),
            sweep_speed=_cfg_float(goalkeeper_config, "sweep_speed", default=0.48),
            sweep_max_speed=self.sweep_max_speed,
            sweep_side_offset=_cfg_float(goalkeeper_config, "sweep_side_offset", default=0.28),
            sweep_side_gain=_cfg_float(goalkeeper_config, "sweep_side_gain", default=1.1),
            sweep_x_gain=_cfg_float(goalkeeper_config, "sweep_x_gain", default=0.65),
            sweep_x_max_speed=_cfg_float(goalkeeper_config, "sweep_x_max_speed", default=0.32),
            sweep_breakaway_distance=_cfg_float(
                goalkeeper_config,
                "sweep_breakaway_distance",
                default=0.85,
            ),
            guard_max_speed=self.guard_max_speed,
            approach_max_speed=self.approach_max_speed,
            recover_max_speed=self.recover_max_speed,
            max_crab_speed=self.max_crab_speed,
            guard_deadband=self.guard_deadband,
            approach_deadband=self.approach_deadband,
            recover_deadband=self.recover_deadband,
            slow_radius=self.slow_radius,
            yaw_translation_gate=self.yaw_translation_gate,
            recover_ball_avoid_radius=self.recover_ball_avoid_radius,
            recover_robot_avoid_radius=self.recover_robot_avoid_radius,
            recover_avoid_lateral=self.recover_avoid_lateral,
            lateral_gain=self.lateral_gain,
            drive_gain=self.drive_gain,
            recover_gain=self.recover_gain,
            latency_bias=_cfg_float(
                goalkeeper_config,
                "latency_bias",
                "ball_prediction_delay",
                default=0.04,
            ),
            prediction_horizon=_cfg_float(
                goalkeeper_config,
                "prediction_horizon",
                "ball_prediction_max_time",
                default=0.22,
            ),
            prediction_max_distance=_cfg_float(goalkeeper_config, "prediction_max_distance", default=0.30),
            penalty_depth=float(goalkeeper_config.get("penalty_depth", 1.35)),
            penalty_width=float(goalkeeper_config.get("penalty_width", self.guard_width)),
            keeper_zone_depth=float(goalkeeper_config.get("keeper_zone_depth", 0.0)),
            keeper_zone_width=float(goalkeeper_config.get("keeper_zone_width", 0.0)),
            active_zone_depth=float(goalkeeper_config.get("active_zone_depth", self.field_length * 0.25)),
            active_zone_width=float(goalkeeper_config.get("active_zone_width", self.field_width)),
            lateral_prediction_horizon=float(goalkeeper_config.get("lateral_prediction_horizon", 3.0)),
        )
        self._ball_tracker = BallTracker()

        sim_config = config.get("sim", {})
        self._team = sim_config.get("team", "")
        self._robot_id = sim_config.get("robot_id", "")

        self.self_pose: Optional[PoseStamped] = None
        self.ball_pose: Optional[PoseStamped] = None
        self.ball_twist: Optional[TwistStamped] = None
        self.defense_goal_pose: Optional[PoseStamped] = None
        self._self_received_time = 0.0
        self._ball_received_time = 0.0
        self._defense_goal_received_time = 0.0
        self._obstacle_received_times = {}
        self._obstacle_poses = {}
        self.state = self.LATERAL_DEFEND
        self._game_state = "INIT"
        self._last_plan = None
        self._last_target = None
        self._last_avoidance_active = False
        self._last_command = VelocityCommand(0.0, 0.0, 0.0)
        self._last_debug_log_time = 0.0
        self._last_stale_warning_time = 0.0
        self._recovering_to_guard = False

        self._role_state_pub = self.create_publisher(
            String,
            soccer_topic(config, "goalkeeper", "state"),
            10,
        )
        if self._team and self._robot_id:
            self._intent_pub = self.create_publisher(
                String, global_soccer_topic("team", self._team, self._robot_id, "intent"), 10
            )
        else:
            self._intent_pub = None

        self.create_subscription(
            String,
            global_soccer_topic("game_state"),
            self._game_state_callback,
            self.low_latency_qos,
        )
        self.create_subscription(
            PoseStamped,
            soccer_topic(config, "world", "self"),
            self._self_callback,
            self.low_latency_qos,
        )
        self.create_subscription(
            PoseStamped,
            soccer_topic(config, "world", "ball"),
            self._ball_callback,
            self.low_latency_qos,
        )
        self.create_subscription(
            TwistStamped,
            soccer_topic(config, "world", "ball_twist"),
            self._ball_twist_callback,
            self.low_latency_qos,
        )
        self.create_subscription(
            PoseStamped,
            soccer_topic(config, "world", "defense_goal"),
            self._defense_goal_callback,
            self.low_latency_qos,
        )
        for obstacle_name in ("teammate", "opponent_1", "opponent_2"):
            self.create_subscription(
                PoseStamped,
                soccer_topic(config, "world", obstacle_name),
                self._make_obstacle_callback(obstacle_name),
                self.low_latency_qos,
            )
        self.timer = self.create_timer(self.control_dt, self._timer_callback)

    def _self_callback(self, msg):
        self.self_pose = msg
        self._self_received_time = self._now_seconds()

    def _ball_callback(self, msg):
        self.ball_pose = msg
        self._ball_received_time = self._now_seconds()
        self._ball_tracker.record_pose(
            self._point_from_pose(msg),
            msg.header.stamp,
            self._ball_received_time,
        )

    def _ball_twist_callback(self, msg):
        self.ball_twist = msg

    def _defense_goal_callback(self, msg):
        self.defense_goal_pose = msg
        self._defense_goal_received_time = self._now_seconds()

    def _make_obstacle_callback(self, name):
        def _callback(msg):
            self._obstacle_poses[name] = msg
            self._obstacle_received_times[name] = self._now_seconds()

        return _callback

    def _game_state_callback(self, msg):
        self._game_state = msg.data

    def _timer_callback(self):
        if self._game_state not in _ACTIVE_GAME_STATES:
            self._set_state(self.LATERAL_DEFEND)
            self._hard_stop()
            self._publish_role_state()
            return

        if self.self_pose is None or self.defense_goal_pose is None:
            self._set_state(self.LATERAL_DEFEND)
            self._hard_stop()
            self._publish_role_state()
            return

        if self.ball_pose is None:
            self._set_state(self.LATERAL_DEFEND)
            self._last_target = self._fallback_guard_target()
            self._drive_to_pose(
                self._last_target,
                self.recover_gain,
                max_speed=self.recover_max_speed,
                deadband=self.recover_deadband,
            )
            self._publish_role_state()
            return

        self_age = self._pose_age(self.self_pose, self._self_received_time)
        if self_age > self.pose_timeout:
            self._set_state(self.LATERAL_DEFEND)
            self._recovering_to_guard = False
            self._hard_stop()
            self._warn_stale_pose("self", self_age)
            self._publish_role_state()
            return

        ball_age = self._pose_age(self.ball_pose, self._ball_received_time)
        if ball_age > self.ball_timeout:
            self._set_state(self.LATERAL_DEFEND)
            self._recovering_to_guard = False
            self._last_target = self._fallback_guard_target()
            self._drive_to_pose(
                self._last_target,
                self.recover_gain,
                max_speed=self.recover_max_speed,
                deadband=self.recover_deadband,
            )
            self._warn_stale_pose("ball", ball_age)
            self._publish_role_state()
            return

        self_point = self._point_from_pose(self.self_pose)
        guard_target = self._fallback_guard_target()
        if self._recovering_to_guard and self._near_recover_guard(self_point, guard_target):
            self._recovering_to_guard = False

        prediction = self._ball_tracker.predict(
            self._now_seconds(),
            self._control_config,
            twist=self._ball_velocity(),
            twist_stamp=self._ball_twist_stamp(),
        )
        plan = goalkeeper_plan(
            ball=self._point_from_pose(self.ball_pose),
            ball_velocity=self._ball_velocity(),
            defense_goal=self._point_from_pose(self.defense_goal_pose),
            config=self._control_config,
            self_point=self_point,
            prediction=prediction,
            recovering=self._recovering_to_guard,
        )
        if plan.phase == "recover_guard":
            self._recovering_to_guard = True
        self._last_plan = plan
        self._set_state(plan.state)
        target = plan.target
        self._last_avoidance_active = False
        if plan.phase == "recover_guard":
            target, self._last_avoidance_active = self._recover_target_with_avoidance(
                self_point,
                plan.target,
            )
        self._last_target = target
        self._execute_plan(plan, target)
        self._maybe_log_debug(plan)
        self._publish_role_state()

    def _execute_plan(self, plan, target: Point2D):
        if plan.phase in ("push_clear", "dribble_clear"):
            self._drive_lateral_sweep(plan)
            return

        if plan.phase == "recover_guard":
            self._drive_to_pose(
                target,
                self.recover_gain,
                max_speed=self.recover_max_speed,
                deadband=self.recover_deadband,
            )
            return

        if plan.phase == "approach_clear":
            self._drive_to_pose(
                target,
                self.drive_gain,
                max_speed=self.approach_max_speed,
                deadband=self.approach_deadband,
            )
            return

        self._drive_to_pose(
            target,
            self.lateral_gain,
            max_speed=self.guard_max_speed,
            deadband=self.guard_deadband,
        )

    def _drive_lateral_sweep(self, plan):
        del plan
        if self.self_pose is None or self.ball_pose is None or self.defense_goal_pose is None:
            self._hard_stop()
            return

        self_point = self._point_from_pose(self.self_pose)
        ball = self._point_from_pose(self.ball_pose)
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        ball_body = field_to_body(ball.x - self_point.x, ball.y - self_point.y, yaw)
        ball_distance = math.hypot(ball_body.x, ball_body.y)
        if ball_distance > max(self._control_config.sweep_breakaway_distance, self.ball_control_distance):
            self._drive_to_pose(
                ball,
                self.drive_gain,
                max_speed=self.approach_max_speed,
                deadband=self.approach_deadband,
            )
            return

        command = lateral_push_command(
            ball_body,
            self._point_from_pose(self.defense_goal_pose),
            self._control_config,
        )
        self._apply_body_command(
            command.vx,
            command.vy,
            max_speed=self.sweep_max_speed,
            max_crab_speed=self.max_crab_speed,
        )

    def _drive_to_pose(
        self,
        target: Point2D,
        gain: Optional[float] = None,
        max_speed: Optional[float] = None,
        deadband: Optional[float] = None,
    ):
        self_point = self._point_from_pose(self.self_pose)
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        dx = target.x - self_point.x
        dy = target.y - self_point.y
        distance = math.hypot(dx, dy)
        vector = field_to_body(dx, dy, yaw)

        used_deadband = self.goal_tolerance if deadband is None else max(float(deadband), 0.0)
        used_gain = self.lateral_gain if gain is None else float(gain)
        used_max_speed = self.guard_max_speed if max_speed is None else max(float(max_speed), 0.0)
        if distance <= used_deadband:
            desired_vx = desired_vy = 0.0
        else:
            speed = damped_speed(distance, used_gain, used_max_speed, used_deadband)
            if self.slow_radius > 0.0 and distance < self.slow_radius:
                speed *= clamp((distance - used_deadband) / max(self.slow_radius - used_deadband, 1e-6), 0.0, 1.0)
            desired_vx = speed * vector.x / max(distance, 1e-6)
            desired_vy = speed * vector.y / max(distance, 1e-6)

        self._apply_body_command(
            desired_vx,
            desired_vy,
            max_speed=used_max_speed,
            max_crab_speed=self.max_crab_speed,
        )

    def _apply_body_command(
        self,
        desired_vx: float,
        desired_vy: float,
        max_speed: Optional[float] = None,
        max_crab_speed: Optional[float] = None,
    ):
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        yaw_error = normalize_angle(lateral_stance_yaw() - yaw)
        translation_scale = self._yaw_translation_scale(abs(yaw_error))
        desired_vx *= translation_scale
        desired_vy *= translation_scale
        desired_wz = self.yaw_gain * yaw_error
        desired_vx, desired_vy, desired_wz = self._limit_target_command(
            desired_vx,
            desired_vy,
            desired_wz,
            max_speed=max_speed,
            max_crab_speed=max_crab_speed,
        )
        if abs(desired_vx) <= 1e-6 and abs(desired_vy) <= 1e-6 and abs(desired_wz) <= 1e-3:
            self._smoother.reset()
        command = self._smoother.step(VelocityCommand(desired_vx, desired_vy, desired_wz), self.control_dt)

        self.move.change_motion_id(self.walk_motion_id)
        self.move.change_speed(command.vx, command.vy, command.wz)
        self._last_command = VelocityCommand(self.move.speed_x, self.move.speed_y, self.move.speed_yaw)

    def _yaw_translation_scale(self, abs_yaw_error: float) -> float:
        gate = max(float(self.yaw_translation_gate), 0.05)
        if abs_yaw_error <= gate:
            return 1.0
        hard_gate = gate * 2.0
        if abs_yaw_error >= hard_gate:
            return 0.0
        return clamp((hard_gate - abs_yaw_error) / gate, 0.0, 1.0)

    def _limit_target_command(
        self,
        vx: float,
        vy: float,
        wz: float,
        max_speed: Optional[float] = None,
        max_crab_speed: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        max_x = abs(float(getattr(self.move, "max_speed_x", 1.0)))
        max_y = abs(float(getattr(self.move, "max_speed_y", 1.0)))
        if max_crab_speed is not None:
            max_y = min(max_y, max(abs(float(max_crab_speed)), 0.0))
        max_wz = abs(float(getattr(self.move, "max_speed_yaw", 1.5)))
        if max_speed is not None:
            vx, vy = self._limit_planar_speed(vx, vy, max(float(max_speed), 0.0))
        scale = 1.0
        if vx != 0.0:
            scale = min(scale, max_x / abs(vx) if max_x > 0.0 else 0.0)
        if vy != 0.0:
            scale = min(scale, max_y / abs(vy) if max_y > 0.0 else 0.0)
        return vx * scale, vy * scale, clamp(wz, -max_wz, max_wz)

    @staticmethod
    def _limit_planar_speed(vx: float, vy: float, max_speed: float) -> Tuple[float, float]:
        magnitude = math.hypot(vx, vy)
        if max_speed <= 0.0 or magnitude <= max_speed or magnitude <= 1e-9:
            return vx, vy
        scale = max_speed / magnitude
        return vx * scale, vy * scale

    def _hard_stop(self):
        self._smoother.reset()
        self._last_plan = None
        self._last_target = None
        self._last_avoidance_active = False
        self.move.stop(use_stop_motion=True)

    def _fallback_guard_target(self) -> Point2D:
        defense_goal = self._point_from_pose(self.defense_goal_pose)
        inward_sign = 1.0 if defense_goal.y <= 0.0 else -1.0
        return Point2D(defense_goal.x, defense_goal.y + inward_sign * self.guard_depth)

    def _recover_target_with_avoidance(self, self_point: Point2D, guard_target: Point2D) -> Tuple[Point2D, bool]:
        if self.ball_pose is None or self.defense_goal_pose is None:
            return guard_target, False
        return recover_avoidance_target(
            self_point,
            guard_target,
            self._point_from_pose(self.ball_pose),
            self._point_from_pose(self.defense_goal_pose),
            self._control_config,
            robot_obstacles=self._fresh_robot_obstacles(),
        )

    def _fresh_robot_obstacles(self):
        obstacles = []
        for name, pose in self._obstacle_poses.items():
            received_time = self._obstacle_received_times.get(name, 0.0)
            if self._pose_age(pose, received_time) <= max(self.pose_timeout * 2.0, 0.5):
                obstacles.append(self._point_from_pose(pose))
        return obstacles

    def _ball_velocity(self) -> Point2D:
        if self.ball_twist is None:
            return Point2D(0.0, 0.0)
        return Point2D(
            self.ball_twist.twist.linear.x,
            self.ball_twist.twist.linear.y,
        )

    def _ball_twist_stamp(self):
        if self.ball_twist is None:
            return None
        return self.ball_twist.header.stamp

    def _set_state(self, state: str):
        if state != self.state:
            self.get_logger().info(f"goalkeeper state: {state}")
            self.state = state

    def _maybe_log_debug(self, plan) -> None:
        if self.debug_log_hz <= 0.0:
            return
        if self.self_pose is None or self.ball_pose is None or self.defense_goal_pose is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_debug_log_time < self._debug_log_period:
            return
        self._last_debug_log_time = now

        self_point = self._point_from_pose(self.self_pose)
        ball = self._point_from_pose(self.ball_pose)
        ball_velocity = self._ball_velocity()
        prediction = plan.prediction
        predicted_ball = prediction.predicted
        self_age = self._pose_age(self.self_pose, self._self_received_time)
        ball_age = self._pose_age(self.ball_pose, self._ball_received_time)
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        target = self._last_target if self._last_target is not None else plan.target
        target_dx = target.x - self_point.x
        target_dy = target.y - self_point.y
        self.get_logger().info(
            "goalkeeper debug: "
            f"state={plan.state} "
            f"phase={plan.phase} "
            f"reason={plan.reason} "
            f"avoidance={int(self._last_avoidance_active)} "
            f"self=({self_point.x:.2f},{self_point.y:.2f}) "
            f"yaw={yaw:.2f} "
            f"self_age_ms={self_age * 1000.0:.0f} "
            f"ball=({ball.x:.2f},{ball.y:.2f}) "
            f"ball_age_ms={ball_age * 1000.0:.0f} "
            f"vel=({ball_velocity.x:.2f},{ball_velocity.y:.2f}) "
            f"velocity_source={prediction.velocity_source} "
            f"pred=({predicted_ball.x:.2f},{predicted_ball.y:.2f}) "
            f"sample_age_ms={prediction.sample_age * 1000.0:.0f} "
            f"used_horizon_ms={prediction.horizon * 1000.0:.0f} "
            f"predicted_disp={prediction.displacement:.2f} "
            f"target=({target.x:.2f},{target.y:.2f}) "
            f"error=({target_dx:.2f},{target_dy:.2f}) "
            f"cmd=({self._last_command.vx:.2f},{self._last_command.vy:.2f},{self._last_command.wz:.2f}) "
            f"ttc_guard={plan.metrics.ttc_guard:.2f} "
            f"ttc_goal={plan.metrics.ttc_goal:.2f}"
        )

    def _publish_role_state(self):
        msg = String()
        msg.data = self.state
        self._role_state_pub.publish(msg)
        if self._intent_pub is None:
            return

        ball = self._point_from_pose(self.ball_pose) if self.ball_pose is not None else None
        self_point = self._point_from_pose(self.self_pose) if self.self_pose is not None else None
        has_ball = (
            self._last_plan.has_possession
            if self._last_plan is not None
            else (
                ball is not None
                and self_point is not None
                and math.hypot(ball.x - self_point.x, ball.y - self_point.y) <= self.ball_control_distance
            )
        )
        payload = {
            "stamp": self.get_clock().now().nanoseconds * 1e-9,
            "team": self._team,
            "robot": self._robot_id,
            "role": "goalkeeper",
            "state": self.state,
            "target": ({"x": ball.x, "y": ball.y} if ball else None),
            "has_ball": has_ball,
            "priority": 80 if self.state == self.ACTIVE_INTERCEPT else 60,
            "controller_mode": "two_state_rule",
        }
        if self._last_plan is not None:
            payload.update(
                {
                    "reason": self._last_plan.reason,
                    "phase": self._last_plan.phase,
                    "ttc_goal": self._last_plan.metrics.ttc_goal,
                    "ttc_guard": self._last_plan.metrics.ttc_guard,
                    "sample_age": self._last_plan.prediction.sample_age,
                    "prediction_horizon": self._last_plan.prediction.horizon,
                    "velocity_source": self._last_plan.prediction.velocity_source,
                    "avoidance_active": self._last_avoidance_active,
                }
            )
        intent_msg = String()
        intent_msg.data = json.dumps(payload)
        self._intent_pub.publish(intent_msg)

    @staticmethod
    def _point_from_pose(msg: PoseStamped) -> Point2D:
        return Point2D(msg.pose.position.x, msg.pose.position.y)

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _pose_age(self, msg: PoseStamped, received_time: float) -> float:
        now = self._now_seconds()
        stamp_age = _stamp_age_seconds(msg.header.stamp, now)
        if stamp_age is not None:
            return stamp_age
        return max(now - received_time, 0.0)

    def _near_recover_guard(self, self_point: Point2D, guard_target: Point2D) -> bool:
        return (
            abs(self_point.x - guard_target.x) <= self.recover_guard_x_tolerance
            and abs(self_point.y - guard_target.y) <= self.recover_guard_y_tolerance
        )

    def _warn_stale_pose(self, name: str, age: float) -> None:
        now = self._now_seconds()
        if now - self._last_stale_warning_time < 1.0:
            return
        self._last_stale_warning_time = now
        self.get_logger().warning(f"goalkeeper stale {name} pose: age_ms={age * 1000.0:.0f}")


def _cfg_float(config: dict, key: str, *legacy_keys: str, default: float) -> float:
    for candidate in (key, *legacy_keys):
        if candidate in config:
            return float(config[candidate])
    return float(default)


def _stamp_age_seconds(stamp, now: float) -> Optional[float]:
    stamp_seconds = float(getattr(stamp, "sec", 0.0)) + float(getattr(stamp, "nanosec", 0.0)) * 1e-9
    if stamp_seconds <= 0.0:
        return None
    age = float(now) - stamp_seconds
    if -0.05 <= age <= 10.0:
        return max(age, 0.0)
    return None
