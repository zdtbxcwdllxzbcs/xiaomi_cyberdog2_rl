"""Striker role controller.

The striker no longer subscribes to VRPN directly. It consumes normalized
``/soccer/world/*`` topics from ``LocatorNode``, publishes an approach target
for the A* planner, and directly commands ``MoveCommander`` only while it is
close enough to dribble or shoot.
"""

import json
import math
from typing import Optional

from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from std_msgs.msg import String

from .lib.geometry import (
    Point2D,
    clamp,
    field_to_body,
    normalize_angle,
    quaternion_from_yaw,
    safe_unit,
    yaw_from_quaternion,
    yaw_to_point,
)
from .lib.topics import global_soccer_topic, soccer_topic

# Game states that allow motion
_ACTIVE_GAME_STATES = {"PLAYING", "KICKOFF_TEAM_A", "KICKOFF_TEAM_B"}


class StrikerNode(Node):
    """High-level attacking state machine."""

    WAITING = "waiting"
    RECOVER = "recover"
    APPROACH = "approach"
    DRIBBLE = "dribble"

    def __init__(self, config, move_commander, path_follower=None, name="striker"):
        super().__init__(name)
        self.config = config
        self.move = move_commander
        self.path_follower = path_follower
        field_config = config.get("field", {})
        path_config = config.get("path", {})
        striker_config = config.get("striker", {})
        move_config = config.get("move", {})

        self.frame_id = config.get("frames", {}).get("field", "field_red")
        self.field_length = float(field_config.get("length", 10.0))
        self.field_width = float(field_config.get("width", 5.5))
        self.goal_width = float(field_config.get("goal_width", 1.5))
        self.boundary_margin = float(field_config.get("boundary_margin", 0.15))
        self.out_of_bounds_margin = float(striker_config.get("out_of_bounds_margin", 0.15))
        self.approach_distance = float(path_config.get("approach_distance", 0.55))
        self.approach_target_tolerance = float(
            path_config.get("approach_target_tolerance", path_config.get("goal_tolerance", 0.18))
        )
        self.possession_distance = float(striker_config.get("possession_distance", 0.45))
        self.dribble_speed = float(striker_config.get("dribble_speed", 0.65))
        self.orbit_radius = float(
            striker_config.get(
                "orbit_radius",
                min(self.approach_distance, self.possession_distance * 0.85),
            )
        )
        self.orbit_tolerance = float(
            striker_config.get("orbit_tolerance", self.approach_target_tolerance)
        )
        self.approach_handoff_distance = float(
            striker_config.get(
                "approach_handoff_distance",
                max(self.orbit_radius + 0.35, self.possession_distance + 0.25),
            )
        )
        self.align_yaw_tolerance = float(striker_config.get("align_yaw_tolerance", 0.35))
        self.kick_duration = float(striker_config.get("kick_duration", 0.45))
        self.kick_speed = float(striker_config.get("kick_speed", min(self.dribble_speed * 1.35, 1.0)))
        self.kick_min_speed = float(striker_config.get("kick_min_speed", min(self.dribble_speed, self.kick_speed)))
        self.kick_lateral_tolerance = float(
            striker_config.get("kick_lateral_tolerance", max(self.orbit_tolerance * 1.35, 0.22))
        )
        self.dribble_release_distance = float(
            striker_config.get("dribble_release_distance", self.possession_distance + 0.18)
        )
        self.dribble_lateral_release = float(
            striker_config.get(
                "dribble_lateral_release",
                max(self.kick_lateral_tolerance, self.orbit_tolerance * 1.8, 0.30),
            )
        )
        self.dribble_rear_release = float(striker_config.get("dribble_rear_release", 0.12))
        self.approach_gain = float(striker_config.get("approach_gain", 0.8))
        self.lateral_gain = float(striker_config.get("lateral_gain", 0.8))
        self.kick_lateral_gain = float(striker_config.get("kick_lateral_gain", self.lateral_gain))
        self.yaw_gain = float(striker_config.get("yaw_gain", 1.2))
        self.wall_escape_margin = float(striker_config.get("wall_escape_margin", 0.55))
        self.wall_escape_inward = float(striker_config.get("wall_escape_inward", 1.0))
        self.endline_escape_backoff = float(striker_config.get("endline_escape_backoff", 1.0))
        self.debug_log_interval = float(striker_config.get("debug_log_interval", 1.0))
        self.walk_motion_id = int(move_config.get("motion_id", 303))
        self.attack_motion_id = int(striker_config.get("attack_motion_id", self.walk_motion_id))
        role_state_rate = float(striker_config.get("role_state_publish_rate_hz", 5.0))
        self.role_state_publish_interval = 1.0 / max(role_state_rate, 0.1)
        self.target_publish_min_delta = float(
            striker_config.get("target_publish_min_delta", 0.04)
        )
        self.target_publish_min_yaw_delta = float(
            striker_config.get("target_publish_min_yaw_delta", 0.08)
        )
        self.target_publish_max_interval = float(
            striker_config.get("target_publish_max_interval", 0.25)
        )

        sim_config = config.get("sim", {})
        self._team = sim_config.get("team", "")
        self._robot_id = sim_config.get("robot_id", "")

        self.self_pose: Optional[PoseStamped] = None
        self.ball_pose: Optional[PoseStamped] = None
        self.ball_twist: Optional[TwistStamped] = None
        self.attack_goal_pose: Optional[PoseStamped] = None
        self.state = self.WAITING
        self._game_state = "INIT"
        self._dribble_phase = "align"
        self._kick_start_time: Optional[float] = None
        self._last_debug_log_time = 0.0
        self._last_role_state_publish_time = 0.0
        self._last_published_role_state = None
        self._last_target_publish_time = 0.0
        self._last_target_point = None
        self._last_target_yaw = None

        self.target_pub = self.create_publisher(
            PoseStamped,
            soccer_topic(config, "striker", "target"),
            10,
        )
        self._role_state_pub = self.create_publisher(
            String,
            soccer_topic(config, "striker", "state"),
            10,
        )
        if self._team and self._robot_id:
            self._intent_pub = self.create_publisher(
                String, global_soccer_topic("team", self._team, self._robot_id, "intent"), 10
            )
        else:
            self._intent_pub = None

        self._game_state_sub = self.create_subscription(
            String, global_soccer_topic("game_state"), self._game_state_callback, 10
        )
        self.self_sub = self.create_subscription(
            PoseStamped,
            soccer_topic(config, "world", "self"),
            self._self_callback,
            10,
        )
        self.ball_sub = self.create_subscription(
            PoseStamped,
            soccer_topic(config, "world", "ball"),
            self._ball_callback,
            10,
        )
        self.ball_twist_sub = self.create_subscription(
            TwistStamped,
            soccer_topic(config, "world", "ball_twist"),
            self._ball_twist_callback,
            10,
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            soccer_topic(config, "world", "attack_goal"),
            self._attack_goal_callback,
            10,
        )
        rate = float(striker_config.get("control_rate_hz", 10.0))
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._timer_callback)

    def _self_callback(self, msg):
        self.self_pose = msg

    def _ball_callback(self, msg):
        self.ball_pose = msg

    def _ball_twist_callback(self, msg):
        self.ball_twist = msg

    def _attack_goal_callback(self, msg):
        self.attack_goal_pose = msg

    def _game_state_callback(self, msg):
        self._game_state = msg.data

    def _timer_callback(self):
        if self._game_state not in _ACTIVE_GAME_STATES:
            self._set_state(self.WAITING)
            self._enable_path_follower(False)
            self.move.stop(use_stop_motion=True)
            self._publish_role_state()
            return

        if self.self_pose is None or self.ball_pose is None or self.attack_goal_pose is None:
            self._set_state(self.WAITING)
            self._enable_path_follower(False)
            self.move.stop(use_stop_motion=True)
            self._publish_role_state()
            return

        self_point = self._point_from_pose(self.self_pose)
        ball = self._point_from_pose(self.ball_pose)
        attack_goal = self._point_from_pose(self.attack_goal_pose)
        kick_target = self._kick_target(ball, attack_goal)

        if self._outside_soft_field(self_point):
            self._set_state(self.RECOVER)
            self._enable_path_follower(False)
            self._drive_to_point(self._field_center(), self.walk_motion_id)
            self._publish_role_state()
            return

        approach_target = self._approach_target(ball, kick_target)
        if not self._ready_to_dribble(self_point, ball, approach_target):
            self._set_state(self.APPROACH)
            self.move.change_motion_id(self.walk_motion_id)
            if self._use_direct_approach(self_point, ball, approach_target):
                self._enable_path_follower(False)
                self._drive_to_point(approach_target, self.walk_motion_id, face_target=ball)
                self._log_debug(
                    "striker approach direct: "
                    f"ball={self._distance(self_point, ball):.2f} "
                    f"target={self._distance(self_point, approach_target):.2f}"
                )
            else:
                self._publish_approach_target(approach_target, ball)
                self._enable_path_follower(True)
            self._publish_role_state()
            return

        self._set_state(self.DRIBBLE)
        self._enable_path_follower(False)
        self._dribble_toward_goal(self_point, ball, attack_goal)
        self._publish_role_state()

    def _approach_target(self, ball: Point2D, attack_goal: Point2D) -> Point2D:
        # Stand behind the ball relative to the selected kick target so the
        # final approach naturally lines up for dribbling or shooting.
        attack_direction = safe_unit(
            attack_goal.x - ball.x,
            attack_goal.y - ball.y,
            fallback_x=0.0,
            fallback_y=1.0,
        )
        return self._clamp_to_field(Point2D(
            ball.x - attack_direction.x * self.orbit_radius,
            ball.y - attack_direction.y * self.orbit_radius,
        ))

    def _kick_target(self, ball: Point2D, attack_goal: Point2D) -> Point2D:
        """Pick a temporary target that keeps wall balls moving back infield."""
        margin = max(float(getattr(self, "wall_escape_margin", 0.55)), 0.0)
        if margin <= 0.0:
            return attack_goal

        x_limit = self.field_width * 0.5 - self.boundary_margin
        y_limit = self.field_length * 0.5 - self.boundary_margin
        near_side_wall = abs(ball.x) >= max(x_limit - margin, 0.0)
        near_endline = abs(ball.y) >= max(y_limit - margin, 0.0)
        if not near_side_wall and not near_endline:
            return attack_goal

        goal_mouth_half_width = max(float(getattr(self, "goal_width", 1.5)) * 0.5, 0.4)
        central_attack_goal = (
            near_endline
            and attack_goal.y != 0.0
            and ball.y * attack_goal.y > 0.0
            and abs(ball.x - attack_goal.x) <= goal_mouth_half_width
        )
        if central_attack_goal and not near_side_wall:
            return attack_goal

        safe_x_limit = max(self.field_width * 0.5 - self.boundary_margin - margin, 0.0)
        safe_y_limit = max(self.field_length * 0.5 - self.boundary_margin - margin, 0.0)
        target_x = clamp(attack_goal.x, -safe_x_limit, safe_x_limit)
        target_y = clamp(attack_goal.y, -safe_y_limit, safe_y_limit)

        if near_side_wall:
            inward = max(float(getattr(self, "wall_escape_inward", 1.0)), 0.0)
            target_x = clamp(
                ball.x - math.copysign(inward, ball.x),
                -safe_x_limit,
                safe_x_limit,
            )

        if near_endline and (near_side_wall or not central_attack_goal):
            backoff = max(float(getattr(self, "endline_escape_backoff", 1.0)), 0.0)
            target_y = clamp(
                ball.y - math.copysign(backoff, ball.y),
                -safe_y_limit,
                safe_y_limit,
            )

        return Point2D(target_x, target_y)

    def _publish_approach_target(self, target: Point2D, ball: Point2D):
        yaw = self._approach_yaw(target, ball)
        now = self.get_clock().now().nanoseconds * 1e-9
        if not self._should_publish_target(target, yaw, now):
            return
        self.target_pub.publish(self._pose_message(target, yaw))
        self._last_target_publish_time = now
        self._last_target_point = target
        self._last_target_yaw = yaw

    def _should_publish_target(self, target: Point2D, yaw: float, now: float) -> bool:
        if self._last_target_point is None or self._last_target_yaw is None:
            return True
        if now - self._last_target_publish_time >= self.target_publish_max_interval:
            return True
        if self._distance(target, self._last_target_point) >= self.target_publish_min_delta:
            return True
        if abs(normalize_angle(yaw - self._last_target_yaw)) >= self.target_publish_min_yaw_delta:
            return True
        return False

    def _ready_to_dribble(self, self_point: Point2D, ball: Point2D, approach_target: Point2D) -> bool:
        del approach_target
        if (
            getattr(self, "state", None) == StrikerNode.DRIBBLE
            and StrikerNode._ball_recoverable(self, self_point, ball)
        ):
            return True
        return StrikerNode._ball_controlled(self, self_point, ball)

    def _use_direct_approach(
        self,
        self_point: Point2D,
        ball: Point2D,
        approach_target: Point2D,
    ) -> bool:
        return (
            self._distance(self_point, approach_target) <= self.approach_handoff_distance
            or self._distance(self_point, ball) <= self.possession_distance + 0.25
        )

    def _dribble_toward_goal(self, self_point: Point2D, ball: Point2D, attack_goal: Point2D):
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        ball_controlled = self._ball_controlled(self_point, ball)
        ball_recoverable = self._ball_recoverable(self_point, ball)
        if not ball_recoverable:
            self._dribble_phase = "align"
            self._kick_start_time = None
            self.move.change_speed(0.0, 0.0, 0.0)
            self._log_debug("striker dribble abort: ball lost")
            return

        kick_target = self._kick_target(ball, attack_goal)
        kick_direction = safe_unit(
            kick_target.x - ball.x,
            kick_target.y - ball.y,
            fallback_x=math.cos(yaw),
            fallback_y=math.sin(yaw),
        )
        orbit_point = self._approach_target(ball, kick_target)
        ball_distance = StrikerNode._distance(self_point, ball)
        orbit_error = self._distance(self_point, orbit_point)
        ball_body = field_to_body(ball.x - self_point.x, ball.y - self_point.y, yaw)
        ball_velocity = self._ball_velocity()
        ball_velocity_along = ball_velocity.x * kick_direction.x + ball_velocity.y * kick_direction.y
        yaw_to_ball_error = normalize_angle(yaw_to_point(self_point, ball) - yaw)
        kick_yaw = math.atan2(kick_direction.y, kick_direction.x)
        kick_yaw_error = normalize_angle(kick_yaw - yaw)

        on_orbit = orbit_error <= self.orbit_tolerance
        ball_in_front = ball_body.x > 0.0 and abs(ball_body.y) <= self.possession_distance

        self.move.change_motion_id(self.attack_motion_id)

        kick_released = False
        if self._dribble_phase == "kick":
            now = self.get_clock().now().nanoseconds * 1e-9
            start_time = self._kick_start_time if self._kick_start_time is not None else now
            elapsed = now - start_time
            ball_stable = (
                elapsed < self.kick_duration
                and ball_distance <= self.possession_distance
                and ball_body.x >= 0.0
                and abs(ball_body.y) <= self.kick_lateral_tolerance
                and ball_velocity_along >= -0.08
            )
            if ball_stable:
                self._kick_in_field_direction(kick_direction, yaw)
                self._log_debug(
                    "striker dribble kick: "
                    f"ball={ball_distance:.2f} side={ball_body.y:.2f} "
                    f"vel={ball_velocity_along:.2f}"
                )
                return
            self._dribble_phase = "align"
            self._kick_start_time = None
            kick_released = True

        if (
            not kick_released
            and ball_controlled
            and (on_orbit or ball_distance <= self.possession_distance)
            and ball_in_front
        ):
            self._dribble_phase = "kick"
            self._kick_start_time = self.get_clock().now().nanoseconds * 1e-9
            self._kick_in_field_direction(kick_direction, yaw)
            self._log_debug(
                "striker dribble trigger: "
                f"ball={ball_distance:.2f} orbit={orbit_error:.2f} yaw={kick_yaw_error:.2f}"
            )
            return

        direction = safe_unit(
            orbit_point.x - self_point.x,
            orbit_point.y - self_point.y,
            fallback_x=math.cos(yaw),
            fallback_y=math.sin(yaw),
        )
        body_direction = field_to_body(direction.x, direction.y, yaw)
        align_speed = min(max(orbit_error * self.approach_gain, 0.12), self.dribble_speed)
        self.move.change_speed(
            align_speed * body_direction.x,
            align_speed * self.lateral_gain * body_direction.y,
            self.yaw_gain * yaw_to_ball_error,
        )
        self._log_debug(
            "striker dribble align: "
            f"ball={ball_distance:.2f} orbit={orbit_error:.2f} yaw_ball={yaw_to_ball_error:.2f}"
        )

    def _kick_in_field_direction(self, kick_direction: Point2D, yaw: float):
        kick_body = field_to_body(kick_direction.x, kick_direction.y, yaw)
        self.move.change_speed(
            self.kick_speed * kick_body.x,
            self.kick_speed * kick_body.y,
            0.0,
        )

    def _drive_to_point(self, target: Point2D, motion_id: int, face_target: Optional[Point2D] = None):
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        self_point = self._point_from_pose(self.self_pose)
        vector = field_to_body(target.x - self_point.x, target.y - self_point.y, yaw)
        yaw_target = face_target if face_target is not None else target
        yaw_error = normalize_angle(yaw_to_point(self_point, yaw_target) - yaw)
        self.move.change_motion_id(motion_id)
        self.move.change_speed(
            self.approach_gain * vector.x,
            self.lateral_gain * vector.y,
            self.yaw_gain * yaw_error,
        )

    def _enable_path_follower(self, enabled: bool):
        if self.path_follower is not None:
            self.path_follower.set_enabled(enabled)

    def _set_state(self, state: str):
        if state != self.state:
            self.get_logger().info(f"striker state: {state}")
            if state != self.DRIBBLE:
                self._dribble_phase = "align"
                self._kick_start_time = None
            self.state = state

    def _log_debug(self, message: str):
        if self.debug_log_interval <= 0.0:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_debug_log_time >= self.debug_log_interval:
            self.get_logger().info(message)
            self._last_debug_log_time = now

    def _publish_role_state(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if (
            self.state == self._last_published_role_state
            and now - self._last_role_state_publish_time < self.role_state_publish_interval
        ):
            return

        msg = String()
        msg.data = self.state
        self._role_state_pub.publish(msg)
        if self._intent_pub is not None:
            ball = self._point_from_pose(self.ball_pose) if self.ball_pose else None
            ball_controlled = (
                ball is not None
                and self.self_pose is not None
                and self._ball_controlled(self._point_from_pose(self.self_pose), ball)
            )
            payload = {
                "stamp": now,
                "team": self._team,
                "robot": self._robot_id,
                "role": "striker",
                "state": self.state,
                "target": (
                    {
                        "x": self.ball_pose.pose.position.x,
                        "y": self.ball_pose.pose.position.y,
                    }
                    if ball
                    else None
                ),
                "has_ball": ball_controlled,
                "priority": 60 if ball_controlled else 40,
            }
            intent_msg = String()
            intent_msg.data = json.dumps(payload)
            self._intent_pub.publish(intent_msg)
        self._last_role_state_publish_time = now
        self._last_published_role_state = self.state

    def _outside_soft_field(self, point: Point2D) -> bool:
        x_limit = self.field_width * 0.5 - self.out_of_bounds_margin
        y_limit = self.field_length * 0.5 - self.out_of_bounds_margin
        return abs(point.x) > x_limit or abs(point.y) > y_limit

    def _clamp_to_field(self, point: Point2D) -> Point2D:
        x_limit = self.field_width * 0.5 - self.boundary_margin
        y_limit = self.field_length * 0.5 - self.boundary_margin
        return Point2D(
            clamp(point.x, -x_limit, x_limit),
            clamp(point.y, -y_limit, y_limit),
        )

    def _field_center(self) -> Point2D:
        return Point2D(0.0, 0.0)

    @staticmethod
    def _approach_yaw(target: Point2D, ball: Point2D) -> float:
        return yaw_to_point(target, ball)

    @staticmethod
    def _distance(a: Point2D, b: Point2D) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _ball_velocity(self) -> Point2D:
        if self.ball_twist is None:
            return Point2D(0.0, 0.0)
        return Point2D(
            self.ball_twist.twist.linear.x,
            self.ball_twist.twist.linear.y,
        )

    def _ball_controlled(self, self_point: Point2D, ball: Point2D) -> bool:
        if self.self_pose is None:
            return False
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        ball_body = field_to_body(ball.x - self_point.x, ball.y - self_point.y, yaw)
        ball_distance = StrikerNode._distance(self_point, ball)
        lateral_limit = max(self.orbit_tolerance, 0.12)
        return (
            ball_distance <= self.possession_distance
            and ball_body.x >= -0.03
            and abs(ball_body.y) <= lateral_limit
        )

    def _ball_recoverable(self, self_point: Point2D, ball: Point2D) -> bool:
        if self.self_pose is None:
            return False
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        ball_body = field_to_body(ball.x - self_point.x, ball.y - self_point.y, yaw)
        ball_distance = StrikerNode._distance(self_point, ball)
        return (
            ball_distance <= self.dribble_release_distance
            and ball_body.x >= -self.dribble_rear_release
            and abs(ball_body.y) <= self.dribble_lateral_release
        )

    def _pose_message(self, point: Point2D, yaw: float) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = point.x
        msg.pose.position.y = point.y
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    @staticmethod
    def _point_from_pose(msg: PoseStamped) -> Point2D:
        return Point2D(msg.pose.position.x, msg.pose.position.y)
