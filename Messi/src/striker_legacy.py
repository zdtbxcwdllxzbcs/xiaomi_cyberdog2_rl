"""Striker (legacy, no A*).

States
------
WAITING  – poses not yet received or game inactive.
RECOVER  – robot outside soft field boundary; drive back to centre.
APPROACH – close to ball, arc-navigate behind it to the kick-behind position
           while avoiding opponents with a simple potential-field side-step.
DRIBBLE  – possession:
  ALIGN  – orbit behind the ball relative to the goal, avoiding the goalkeeper
           with a lateral dodge if she blocks the direct line.
  KICK   – drive forward for a fixed duration.
"""

import json
import math
from typing import Dict, Optional, Tuple

from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from std_msgs.msg import String

from .lib.geometry import (
    Point2D,
    clamp,
    field_to_body,
    normalize_angle,
    safe_unit,
    yaw_from_quaternion,
    yaw_to_point,
)
from .lib.topics import global_soccer_topic, soccer_topic

_ACTIVE_GAME_STATES = {"PLAYING", "KICKOFF_TEAM_A", "KICKOFF_TEAM_B"}

# ── tuning ────────────────────────────────────────────────────────────────────
# Minimum clearance (m) between the desired path and an obstacle before the
# potential-field repulsion kicks in.
_OBSTACLE_INFLUENCE   = 0.7
# Max lateral deflection added by repulsion (m/step, not really m/s – it is
# treated as a body-frame y velocity bias).
_OBSTACLE_REPULSE_MAX = 0.6

# How far in front of the goalkeeper we consider her to be "blocking" the shot.
_GK_BLOCK_CORRIDOR    = 0.4   # half-width of the imaginary corridor to goal
_GK_DODGE_OFFSET      = 0.55  # lateral offset we aim for when dodging


class StrikerLegacyNode(Node):
    """Geometry-only attacking state machine (no A* planner)."""

    WAITING  = "waiting"
    RECOVER  = "recover"
    APPROACH = "approach"
    DRIBBLE  = "dribble"

    _ALIGN = "align"
    _KICK  = "kick"

    def __init__(self, config, move_commander, name="striker_legacy"):
        super().__init__(name)
        self.config = config
        self.move   = move_commander

        field_cfg   = config.get("field",   {})
        striker_cfg = config.get("striker", {})
        move_cfg    = config.get("move",    {})
        sim_cfg     = config.get("sim",     {})

        self.frame_id            = config.get("frames", {}).get("field", "field_red")
        self.field_length        = float(field_cfg.get("length",           10.0))
        self.field_width         = float(field_cfg.get("width",             5.5))
        self.goal_width          = float(field_cfg.get("goal_width",        1.5))
        self.boundary_margin     = float(field_cfg.get("boundary_margin",   0.15))
        self.out_of_bounds_margin = float(striker_cfg.get("out_of_bounds_margin", 0.15))

        self.possession_distance = float(
            striker_cfg.get("legacy_possession_distance", striker_cfg.get("possession_distance", 0.45))
        )
        self.orbit_radius        = float(striker_cfg.get("orbit_radius",         0.35))
        self.orbit_tol           = float(
            striker_cfg.get("legacy_orbit_tolerance", striker_cfg.get("orbit_tolerance", 0.12))
        )
        self.dribble_behind_min  = float(
            striker_cfg.get("legacy_dribble_behind_min", max(self.orbit_radius * 0.45, 0.16))
        )
        self._align_yaw_tol      = float(
            striker_cfg.get("legacy_align_yaw_tolerance", striker_cfg.get("align_yaw_tolerance", 0.24))
        )

        self.dribble_speed         = float(striker_cfg.get("legacy_dribble_speed", striker_cfg.get("dribble_speed", 0.65)))
        self._kick_duration        = float(
            striker_cfg.get("legacy_kick_duration", striker_cfg.get("kick_duration", 0.4))
        )
        self._kick_speed           = float(
            striker_cfg.get(
                "legacy_kick_speed",
                striker_cfg.get("kick_speed", min(self.dribble_speed * 1.35, 1.0)),
            )
        )
        self._kick_min_speed       = float(
            striker_cfg.get("legacy_kick_min_speed", min(self.dribble_speed, self._kick_speed))
        )
        self.kick_contact_distance = float(
            striker_cfg.get(
                "legacy_kick_contact_distance",
                min(self.possession_distance, max(self.orbit_radius, 0.38)),
            )
        )
        self.approach_gain         = float(striker_cfg.get("approach_gain",        0.8))
        self.lateral_gain          = float(striker_cfg.get("lateral_gain",         0.8))
        self.kick_lateral_gain     = float(striker_cfg.get("kick_lateral_gain",    self.lateral_gain))
        self.yaw_gain              = float(striker_cfg.get("yaw_gain",             1.2))
        self.kick_lateral_tolerance = float(
            striker_cfg.get("kick_lateral_tolerance", max(self.orbit_tol * 2.0, 0.22))
        )
        self.dribble_release_distance = float(
            striker_cfg.get(
                "legacy_dribble_release_distance",
                striker_cfg.get("dribble_release_distance", self.possession_distance + 0.18),
            )
        )
        self.dribble_lateral_release = float(
            striker_cfg.get(
                "legacy_dribble_lateral_release",
                striker_cfg.get("dribble_lateral_release", max(self.kick_lateral_tolerance, 0.30)),
            )
        )
        self.dribble_rear_release = float(
            striker_cfg.get("legacy_dribble_rear_release", striker_cfg.get("dribble_rear_release", 0.12))
        )
        self.wall_escape_margin = float(striker_cfg.get("wall_escape_margin", 0.55))
        self.wall_escape_inward = float(striker_cfg.get("wall_escape_inward", 1.0))
        self.endline_escape_backoff = float(striker_cfg.get("endline_escape_backoff", 1.0))
        self.latency_compensation = float(
            striker_cfg.get("legacy_latency_compensation", striker_cfg.get("latency_compensation", 0.12))
        )
        self.max_prediction_time = float(striker_cfg.get("legacy_max_prediction_time", 0.30))
        self.obstacle_influence = float(
            striker_cfg.get("legacy_obstacle_influence", striker_cfg.get("obstacle_influence", _OBSTACLE_INFLUENCE))
        )
        self.obstacle_corridor = float(striker_cfg.get("legacy_obstacle_corridor", _GK_BLOCK_CORRIDOR + 0.08))
        self.obstacle_repulse_max = float(
            striker_cfg.get(
                "legacy_obstacle_repulse_max",
                striker_cfg.get("obstacle_repulse_max", _OBSTACLE_REPULSE_MAX),
            )
        )
        self.obstacle_slowdown_gain = float(striker_cfg.get("legacy_obstacle_slowdown_gain", 0.45))
        self.obstacle_slowdown_min = float(striker_cfg.get("legacy_obstacle_slowdown_min", 0.45))
        self.dodge_hold_time = float(striker_cfg.get("legacy_dodge_hold_time", 0.45))
        self.gk_block_corridor = float(striker_cfg.get("legacy_gk_block_corridor", _GK_BLOCK_CORRIDOR))
        self.gk_dodge_offset = float(striker_cfg.get("legacy_gk_dodge_offset", _GK_DODGE_OFFSET))
        self.ball_path_keepout = float(
            striker_cfg.get("legacy_ball_path_keepout", max(self.possession_distance * 0.9, 0.40))
        )
        self.ball_path_clearance = float(
            striker_cfg.get("legacy_ball_path_clearance", self.ball_path_keepout + 0.16)
        )
        self.dribble_orbit_side_offset = float(
            striker_cfg.get(
                "legacy_dribble_orbit_side_offset",
                max(self.ball_path_clearance, self.kick_lateral_tolerance * 1.4),
            )
        )
        self.dribble_handoff_lateral = float(
            striker_cfg.get(
                "legacy_dribble_handoff_lateral",
                max(self.kick_lateral_tolerance * 1.4, self.orbit_tol * 1.8, 0.30),
            )
        )
        self.dribble_gate_behind_offset = float(
            striker_cfg.get(
                "legacy_dribble_gate_behind_offset",
                max(self.orbit_radius * 0.75, self.dribble_behind_min + 0.10),
            )
        )
        self.ball_avoid_hold_time = float(striker_cfg.get("legacy_ball_avoid_hold_time", 0.40))
        self.recover_infield_margin = float(striker_cfg.get("legacy_recover_infield_margin", 0.35))
        self.debug_log_interval = float(
            striker_cfg.get("legacy_debug_log_interval", striker_cfg.get("debug_log_interval", 1.0))
        )

        self.walk_motion_id   = int(move_cfg.get("motion_id",      303))
        self.attack_motion_id = int(striker_cfg.get("attack_motion_id", self.walk_motion_id))

        self._team     = sim_cfg.get("team",     "")
        self._robot_id = sim_cfg.get("robot_id", "")

        # Runtime state
        self.self_pose:        Optional[PoseStamped] = None
        self.ball_pose:        Optional[PoseStamped] = None
        self.ball_twist:       Optional[TwistStamped] = None
        self.attack_goal_pose: Optional[PoseStamped] = None
        self._opponent_poses:  Dict[str, PoseStamped] = {}

        self.state           = self.WAITING
        self._dribble_sub    = self._ALIGN
        self._kick_start_time: Optional[float] = None
        self._dodge_side = 0.0
        self._dodge_until = 0.0
        self._ball_avoid_side = 0.0
        self._ball_avoid_until = 0.0
        self._last_debug_log_time = 0.0

        # Publishers
        self._role_state_pub = self.create_publisher(
            String, soccer_topic(config, "striker", "state"), 10
        )
        if self._team and self._robot_id:
            self._intent_pub = self.create_publisher(
                String, global_soccer_topic("team", self._team, self._robot_id, "intent"), 10
            )
        else:
            self._intent_pub = None

        # Subscriptions
        self.create_subscription(
            String, global_soccer_topic("game_state"), self._game_state_cb, 10
        )
        self.create_subscription(
            PoseStamped, soccer_topic(config, "world", "self"), self._self_cb, 10
        )
        self.create_subscription(
            PoseStamped, soccer_topic(config, "world", "ball"), self._ball_cb, 10
        )
        self.create_subscription(
            TwistStamped, soccer_topic(config, "world", "ball_twist"), self._ball_twist_cb, 10
        )
        self.create_subscription(
            PoseStamped,
            soccer_topic(config, "world", "attack_goal"),
            self._attack_goal_cb,
            10,
        )
        for opp in ("opponent_1", "opponent_2"):
            self.create_subscription(
                PoseStamped,
                soccer_topic(config, "world", opp),
                self._make_opp_cb(opp),
                10,
            )

        rate = float(striker_cfg.get("control_rate_hz", 10.0))
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._tick)

        self._game_state = "INIT"

    # ── callbacks ──────────────────────────────────────────────────────────────

    def _self_cb(self, msg):        self.self_pose = msg
    def _ball_cb(self, msg):        self.ball_pose = msg
    def _ball_twist_cb(self, msg):  self.ball_twist = msg
    def _attack_goal_cb(self, msg): self.attack_goal_pose = msg
    def _game_state_cb(self, msg):  self._game_state = msg.data

    def _make_opp_cb(self, name):
        def cb(msg): self._opponent_poses[name] = msg
        return cb

    # ── main control loop ──────────────────────────────────────────────────────

    def _tick(self):
        if self._game_state not in _ACTIVE_GAME_STATES:
            self._set_state(self.WAITING)
            self.move.stop(use_stop_motion=True)
            self._publish_state()
            return

        if self.self_pose is None or self.ball_pose is None or self.attack_goal_pose is None:
            self._set_state(self.WAITING)
            self.move.stop(use_stop_motion=True)
            self._publish_state()
            return

        me   = _pose_to_point(self.self_pose)
        ball = self._predicted_ball(_pose_to_point(self.ball_pose))
        goal = _pose_to_point(self.attack_goal_pose)

        if self._outside_soft_field(me):
            self._set_state(self.RECOVER)
            self._recover_step(me, ball)
            self._publish_state()
            return

        if not self._ready_to_dribble(me, ball, goal):
            self._set_state(self.APPROACH)
            self._approach_step(me, ball, goal)
        else:
            self._set_state(self.DRIBBLE)
            self._dribble_step(me, ball, goal)

        self._publish_state()

    # ── APPROACH ───────────────────────────────────────────────────────────────

    def _approach_step(self, me: Point2D, ball: Point2D, goal: Point2D):
        """Drive toward the behind-ball orbit point with potential-field avoidance."""
        self.move.change_motion_id(self.walk_motion_id)
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)

        kick_target = self._pick_kick_target(ball, goal)
        kick_dir    = safe_unit(kick_target.x - ball.x, kick_target.y - ball.y)
        behind_depth, line_lateral = self._behind_ball_metrics(me, ball, kick_dir)
        ball_dist = math.hypot(ball.x - me.x, ball.y - me.y)
        orbit_point = self._clamp_to_field(Point2D(
            ball.x - kick_dir.x * self.orbit_radius,
            ball.y - kick_dir.y * self.orbit_radius,
        ))

        if ball_dist <= self.dribble_release_distance and behind_depth < self.dribble_behind_min:
            drive_target = self._dribble_orbit_gate(ball, kick_dir, line_lateral)
        elif ball_dist <= self.dribble_release_distance:
            drive_target = orbit_point
        else:
            drive_target = self._ball_safe_target(me, orbit_point, ball)
        speed_scale, repulse_y = self._obstacle_avoidance(me, drive_target, yaw)

        to_orbit   = safe_unit(drive_target.x - me.x, drive_target.y - me.y)
        body       = field_to_body(to_orbit.x, to_orbit.y, yaw)
        dist       = math.hypot(drive_target.x - me.x, drive_target.y - me.y)
        speed      = min(max(dist * self.approach_gain, 0.10), self.dribble_speed) * speed_scale

        desired_yaw = yaw_to_point(me, ball)
        yaw_error   = normalize_angle(desired_yaw - yaw)

        self.move.change_speed(
            speed * body.x,
            clamp(speed * self.lateral_gain * body.y + repulse_y,
                  -self.dribble_speed, self.dribble_speed),
            self.yaw_gain * yaw_error,
        )
        self._log_debug(
            "legacy approach: "
            f"ball={ball_dist:.2f} behind={behind_depth:.2f} "
            f"target=({drive_target.x:+.2f},{drive_target.y:+.2f})"
        )

    # ── DRIBBLE ────────────────────────────────────────────────────────────────

    def _dribble_step(self, me: Point2D, ball: Point2D, goal: Point2D):
        self.move.change_motion_id(self.attack_motion_id)
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)

        # Choose kick target: dodge the goalkeeper when she blocks the direct line
        kick_target = self._pick_kick_target(ball, goal)
        kick_dir    = safe_unit(kick_target.x - ball.x, kick_target.y - ball.y)
        behind_depth, line_lateral = self._behind_ball_metrics(me, ball, kick_dir)
        well_behind = self._well_behind_ball(behind_depth, line_lateral)

        if not self._ball_recoverable(me, ball):
            self._dribble_sub = self._ALIGN
            self._kick_start_time = None
            self.move.change_speed(0.0, 0.0, 0.0)
            return

        orbit_point = self._clamp_to_field(Point2D(
            ball.x - kick_dir.x * self.orbit_radius,
            ball.y - kick_dir.y * self.orbit_radius,
        ))

        desired_yaw = yaw_to_point(me, ball)
        yaw_error   = normalize_angle(desired_yaw - yaw)

        kick_yaw       = math.atan2(kick_dir.y, kick_dir.x)
        kick_yaw_error = normalize_angle(kick_yaw - yaw)

        orbit_err = math.hypot(orbit_point.x - me.x, orbit_point.y - me.y)
        ball_dist = math.hypot(ball.x - me.x, ball.y - me.y)
        ball_body = field_to_body(ball.x - me.x, ball.y - me.y, yaw)
        on_orbit  = orbit_err <= self.orbit_tol or (well_behind and ball_dist <= self.possession_distance)
        ball_in_front = (
            well_behind
            and ball_body.x >= -0.03
            and abs(ball_body.y) <= self.kick_lateral_tolerance
        )
        close_enough_to_kick = ball_dist <= self.kick_contact_distance
        position_ready_to_kick = on_orbit and ball_in_front and close_enough_to_kick

        if self._dribble_sub == self._ALIGN:
            close_bad_angle = ball_dist <= self.dribble_release_distance and not well_behind
            needs_behind_gate = close_bad_angle and behind_depth < self.dribble_behind_min
            if needs_behind_gate:
                drive_target = self._dribble_orbit_gate(ball, kick_dir, line_lateral)
            elif close_bad_angle:
                drive_target = orbit_point
            else:
                drive_target = self._ball_safe_target(me, orbit_point, ball)
            speed_scale, repulse_y = self._obstacle_avoidance(me, drive_target, yaw)
            to_orbit = safe_unit(
                drive_target.x - me.x, drive_target.y - me.y,
                fallback_x=math.cos(yaw), fallback_y=math.sin(yaw),
            )
            body  = field_to_body(to_orbit.x, to_orbit.y, yaw)
            drive_err = math.hypot(drive_target.x - me.x, drive_target.y - me.y)
            speed = min(max(drive_err * self.approach_gain, 0.08), self.dribble_speed * 0.7)
            speed *= speed_scale
            if ball_dist <= self.dribble_release_distance and ball_body.x >= -self.dribble_rear_release:
                align_yaw_control = normalize_angle(0.65 * kick_yaw_error + 0.35 * yaw_error)
            else:
                align_yaw_control = yaw_error
            forward_cmd = speed * body.x
            if needs_behind_gate:
                forward_cmd = clamp(forward_cmd, -self.dribble_speed * 0.30, 0.0)
            self.move.change_speed(
                forward_cmd,
                clamp(
                    speed * self.lateral_gain * body.y + repulse_y,
                    -self.dribble_speed,
                    self.dribble_speed,
                ),
                self.yaw_gain * align_yaw_control,
            )
            self._log_debug(
                "legacy dribble align: "
                f"ball={ball_dist:.2f} side={ball_body.y:.2f} "
                f"orbit={orbit_err:.2f} yaw_goal={kick_yaw_error:.2f}"
            )
            if position_ready_to_kick:
                self._kick_start_time = self.get_clock().now().nanoseconds * 1e-9
                self._dribble_sub     = self._KICK
                self.get_logger().info("dribble: kick triggered")
                self._kick_in_field_direction(kick_dir, yaw)

        else:  # _KICK
            now     = self.get_clock().now().nanoseconds * 1e-9
            start   = self._kick_start_time if self._kick_start_time is not None else now
            elapsed = now - start
            stable = elapsed < self._kick_duration and ball_in_front and self._ball_recoverable(me, ball)
            if stable:
                self._kick_in_field_direction(kick_dir, yaw)
            else:
                self._dribble_sub     = self._ALIGN
                self._kick_start_time = None
                self.move.change_speed(0.0, 0.0, 0.0)

    def _kick_in_field_direction(self, kick_dir: Point2D, yaw: float):
        kick_body = field_to_body(kick_dir.x, kick_dir.y, yaw)
        self.move.change_speed(
            self._kick_speed * kick_body.x,
            self._kick_speed * kick_body.y,
            0.0,
        )

    # ── goalkeeper dodge ───────────────────────────────────────────────────────

    def _pick_kick_target(self, ball: Point2D, goal: Point2D) -> Point2D:
        """Return a cheap tactical target for wall escape and blocked shots."""
        target = self._wall_escape_target(ball, goal)
        blocker = self._nearest_blocker_on_lane(ball, target, self.gk_block_corridor)
        if blocker is None:
            return target

        to_target = safe_unit(target.x - ball.x, target.y - ball.y)
        to_blocker = Point2D(blocker.x - ball.x, blocker.y - ball.y)
        lateral = to_blocker.x * (-to_target.y) + to_blocker.y * to_target.x
        dodge_sign = -1.0 if lateral >= 0.0 else 1.0
        perp = Point2D(-to_target.y, to_target.x)
        return self._clamp_to_field(Point2D(
            clamp(
                target.x + dodge_sign * self.gk_dodge_offset * perp.x,
                -self.field_width * 0.5 + self.boundary_margin,
                self.field_width * 0.5 - self.boundary_margin,
            ),
            target.y + dodge_sign * self.gk_dodge_offset * perp.y,
        ))

    def _wall_escape_target(self, ball: Point2D, goal: Point2D) -> Point2D:
        margin = max(self.wall_escape_margin, 0.0)
        if margin <= 0.0:
            return goal

        x_limit = self.field_width * 0.5 - self.boundary_margin
        y_limit = self.field_length * 0.5 - self.boundary_margin
        near_side_wall = abs(ball.x) >= max(x_limit - margin, 0.0)
        near_endline = abs(ball.y) >= max(y_limit - margin, 0.0)
        if not near_side_wall and not near_endline:
            return goal

        goal_mouth_half_width = max(self.goal_width * 0.5, 0.4)
        central_attack_goal = (
            near_endline
            and goal.y != 0.0
            and ball.y * goal.y > 0.0
            and abs(ball.x - goal.x) <= goal_mouth_half_width
        )
        if central_attack_goal and not near_side_wall:
            return goal

        safe_x_limit = max(self.field_width * 0.5 - self.boundary_margin - margin, 0.0)
        safe_y_limit = max(self.field_length * 0.5 - self.boundary_margin - margin, 0.0)
        target_x = clamp(goal.x, -safe_x_limit, safe_x_limit)
        target_y = clamp(goal.y, -safe_y_limit, safe_y_limit)

        if near_side_wall:
            target_x = clamp(
                ball.x - math.copysign(max(self.wall_escape_inward, 0.0), ball.x),
                -safe_x_limit,
                safe_x_limit,
            )
        if near_endline and (near_side_wall or not central_attack_goal):
            target_y = clamp(
                ball.y - math.copysign(max(self.endline_escape_backoff, 0.0), ball.y),
                -safe_y_limit,
                safe_y_limit,
            )
        return Point2D(target_x, target_y)

    def _nearest_opponent_to_goal(self, goal: Point2D) -> Optional[Point2D]:
        best, best_dist = None, float("inf")
        for pose_msg in self._opponent_poses.values():
            p    = _pose_to_point(pose_msg)
            dist = math.hypot(p.x - goal.x, p.y - goal.y)
            if dist < best_dist:
                best, best_dist = p, dist
        return best

    # ── potential-field obstacle avoidance (approach phase) ───────────────────

    def _obstacle_avoidance(self, me: Point2D, target: Point2D, yaw: float) -> Tuple[float, float]:
        """Return ``(speed_scale, body_lateral_bias)`` for the nearest blocker."""
        blocker = self._nearest_blocker_on_lane(
            me,
            target,
            self.obstacle_corridor,
            max_along=self.obstacle_influence,
        )
        now = self._now_seconds()
        if blocker is None:
            if now >= self._dodge_until:
                self._dodge_side = 0.0
            return 1.0, 0.0

        path_dir = safe_unit(target.x - me.x, target.y - me.y)
        perp = Point2D(-path_dir.y, path_dir.x)
        dx = blocker.x - me.x
        dy = blocker.y - me.y
        along = dx * path_dir.x + dy * path_dir.y
        lateral = dx * perp.x + dy * perp.y

        if abs(lateral) > 0.05:
            dodge_side = -1.0 if lateral > 0.0 else 1.0
        elif self._dodge_side != 0.0 and now < self._dodge_until:
            dodge_side = self._dodge_side
        else:
            left_x = me.x + perp.x
            right_x = me.x - perp.x
            dodge_side = 1.0 if abs(left_x) < abs(right_x) else -1.0

        self._dodge_side = dodge_side
        self._dodge_until = now + self.dodge_hold_time

        lateral_strength = 1.0 - clamp(abs(lateral) / max(self.obstacle_corridor, 1e-6), 0.0, 1.0)
        along_strength = 1.0 - clamp(along / max(self.obstacle_influence, 1e-6), 0.0, 1.0)
        strength = max(lateral_strength, along_strength * 0.65, 0.25)

        dodge_field = Point2D(dodge_side * perp.x, dodge_side * perp.y)
        dodge_body = field_to_body(dodge_field.x, dodge_field.y, yaw)
        lateral_bias = clamp(
            self.obstacle_repulse_max * strength * dodge_body.y,
            -self.obstacle_repulse_max,
            self.obstacle_repulse_max,
        )
        speed_scale = clamp(
            1.0 - self.obstacle_slowdown_gain * strength,
            self.obstacle_slowdown_min,
            1.0,
        )
        return speed_scale, lateral_bias

    def _nearest_blocker_on_lane(
        self,
        start: Point2D,
        target: Point2D,
        corridor: float,
        max_along: Optional[float] = None,
    ) -> Optional[Point2D]:
        path_dx = target.x - start.x
        path_dy = target.y - start.y
        path_len = math.hypot(path_dx, path_dy)
        if path_len < 1e-6:
            return None

        path_dir = Point2D(path_dx / path_len, path_dy / path_len)
        perp = Point2D(-path_dir.y, path_dir.x)
        best = None
        best_score = float("inf")
        influence = path_len + 0.20 if max_along is None else min(max_along, path_len + 0.20)
        for pose_msg in self._opponent_poses.values():
            obs = _pose_to_point(pose_msg)
            dx = obs.x - start.x
            dy = obs.y - start.y
            along = dx * path_dir.x + dy * path_dir.y
            if along <= 0.0 or along > influence:
                continue
            lateral = abs(dx * perp.x + dy * perp.y)
            if lateral > corridor:
                continue
            score = lateral + 0.20 * along
            if score < best_score:
                best = obs
                best_score = score
        return best

    # ── utilities ──────────────────────────────────────────────────────────────

    def _behind_ball_metrics(self, me: Point2D, ball: Point2D, kick_dir: Point2D) -> Tuple[float, float]:
        rel = Point2D(me.x - ball.x, me.y - ball.y)
        behind_dir = Point2D(-kick_dir.x, -kick_dir.y)
        perp = Point2D(-kick_dir.y, kick_dir.x)
        behind_depth = rel.x * behind_dir.x + rel.y * behind_dir.y
        lateral = rel.x * perp.x + rel.y * perp.y
        return behind_depth, lateral

    def _well_behind_ball(self, behind_depth: float, lateral: float) -> bool:
        return (
            behind_depth >= self.dribble_behind_min
            and abs(lateral) <= max(self.kick_lateral_tolerance, self.orbit_tol)
        )

    def _dribble_orbit_gate(self, ball: Point2D, kick_dir: Point2D, lateral: float) -> Point2D:
        perp = Point2D(-kick_dir.y, kick_dir.x)
        side_offset = max(self.dribble_orbit_side_offset, self.ball_path_clearance)
        behind_offset = max(self.dribble_gate_behind_offset, self.dribble_behind_min)
        left = self._clamp_to_field(Point2D(
            ball.x - kick_dir.x * behind_offset + perp.x * side_offset,
            ball.y - kick_dir.y * behind_offset + perp.y * side_offset,
        ))
        right = self._clamp_to_field(Point2D(
            ball.x - kick_dir.x * behind_offset - perp.x * side_offset,
            ball.y - kick_dir.y * behind_offset - perp.y * side_offset,
        ))

        if abs(lateral) > 0.05:
            target = left if lateral > 0.0 else right
            other = right if lateral > 0.0 else left
            if abs(target.x) > abs(other.x) + 0.20:
                target = other
        else:
            target = left if abs(left.x) <= abs(right.x) else right
        return target

    def _ball_safe_target(self, me: Point2D, target: Point2D, ball: Point2D) -> Point2D:
        """Divert around the ball when repositioning would otherwise clip it."""
        path_dx = target.x - me.x
        path_dy = target.y - me.y
        path_len = math.hypot(path_dx, path_dy)
        if path_len < 1e-6:
            return target

        path_dir = Point2D(path_dx / path_len, path_dy / path_len)
        rel_ball = Point2D(ball.x - me.x, ball.y - me.y)
        along = rel_ball.x * path_dir.x + rel_ball.y * path_dir.y
        if along <= 0.0 or along >= path_len:
            if self._now_seconds() >= self._ball_avoid_until:
                self._ball_avoid_side = 0.0
            return target

        perp = Point2D(-path_dir.y, path_dir.x)
        lateral = rel_ball.x * perp.x + rel_ball.y * perp.y
        if abs(lateral) >= self.ball_path_keepout:
            if self._now_seconds() >= self._ball_avoid_until:
                self._ball_avoid_side = 0.0
            return target

        now = self._now_seconds()
        if self._ball_avoid_side != 0.0 and now < self._ball_avoid_until:
            side = self._ball_avoid_side
        else:
            left = self._clamp_to_field(Point2D(
                ball.x + perp.x * self.ball_path_clearance,
                ball.y + perp.y * self.ball_path_clearance,
            ))
            right = self._clamp_to_field(Point2D(
                ball.x - perp.x * self.ball_path_clearance,
                ball.y - perp.y * self.ball_path_clearance,
            ))
            side = 1.0 if abs(left.x) <= abs(right.x) else -1.0

        self._ball_avoid_side = side
        self._ball_avoid_until = now + self.ball_avoid_hold_time
        return self._clamp_to_field(Point2D(
            ball.x + side * perp.x * self.ball_path_clearance,
            ball.y + side * perp.y * self.ball_path_clearance,
        ))

    def _recover_step(self, me: Point2D, ball: Point2D):
        x_limit = self.field_width * 0.5 - self.boundary_margin - self.recover_infield_margin
        y_limit = self.field_length * 0.5 - self.boundary_margin - self.recover_infield_margin
        target = Point2D(
            clamp(me.x, -max(x_limit, 0.0), max(x_limit, 0.0)),
            clamp(me.y, -max(y_limit, 0.0), max(y_limit, 0.0)),
        )
        target = self._ball_safe_target(me, target, ball)
        self._drive_to(me, target, self.walk_motion_id)

    def _predicted_ball(self, ball: Point2D) -> Point2D:
        velocity = self._ball_velocity()
        if velocity is None:
            return ball

        age = self._message_age(self.ball_pose)
        horizon = max(self.latency_compensation, 0.0)
        if age is not None:
            horizon += age
        horizon = clamp(horizon, 0.0, max(self.max_prediction_time, 0.0))
        return self._clamp_to_field(Point2D(
            ball.x + velocity.x * horizon,
            ball.y + velocity.y * horizon,
        ))

    def _ball_velocity(self) -> Optional[Point2D]:
        if self.ball_twist is None:
            return None
        return Point2D(
            self.ball_twist.twist.linear.x,
            self.ball_twist.twist.linear.y,
        )

    def _message_age(self, msg) -> Optional[float]:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            return None
        stamp_seconds = float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9
        if stamp_seconds <= 0.0:
            return None
        return max(self._now_seconds() - stamp_seconds, 0.0)

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _ready_to_dribble(self, me: Point2D, ball: Point2D, goal: Point2D) -> bool:
        kick_target = self._pick_kick_target(ball, goal)
        kick_dir = safe_unit(kick_target.x - ball.x, kick_target.y - ball.y)
        behind_depth, line_lateral = self._behind_ball_metrics(me, ball, kick_dir)
        well_behind = self._well_behind_ball(behind_depth, line_lateral)
        ball_controlled = self._ball_controlled(me, ball)
        ball_recoverable = self._ball_recoverable(me, ball)
        if self.state == self.DRIBBLE and ball_recoverable:
            return ball_controlled or well_behind or abs(line_lateral) <= self.dribble_handoff_lateral
        return ball_controlled and well_behind

    def _ball_controlled(self, me: Point2D, ball: Point2D) -> bool:
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        dx = ball.x - me.x
        dy = ball.y - me.y
        body = field_to_body(dx, dy, yaw)
        lateral_limit = max(self.orbit_tol * 1.6, 0.16)
        return (
            dx * dx + dy * dy <= self.possession_distance * self.possession_distance
            and body.x >= -0.03
            and abs(body.y) <= lateral_limit
        )

    def _ball_recoverable(self, me: Point2D, ball: Point2D) -> bool:
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        dx = ball.x - me.x
        dy = ball.y - me.y
        body = field_to_body(dx, dy, yaw)
        return (
            dx * dx + dy * dy <= self.dribble_release_distance * self.dribble_release_distance
            and body.x >= -self.dribble_rear_release
            and abs(body.y) <= self.dribble_lateral_release
        )

    def _drive_to(self, me: Point2D, target: Point2D, motion_id: int):
        yaw       = yaw_from_quaternion(self.self_pose.pose.orientation)
        vec       = field_to_body(target.x - me.x, target.y - me.y, yaw)
        yaw_error = normalize_angle(yaw_to_point(me, target) - yaw)
        self.move.change_motion_id(motion_id)
        self.move.change_speed(
            self.approach_gain * vec.x,
            self.lateral_gain  * vec.y,
            self.yaw_gain      * yaw_error,
        )

    def _outside_soft_field(self, point: Point2D) -> bool:
        x_lim = self.field_width  * 0.5 - self.out_of_bounds_margin
        y_lim = self.field_length * 0.5 - self.out_of_bounds_margin
        return abs(point.x) > x_lim or abs(point.y) > y_lim

    def _clamp_to_field(self, point: Point2D) -> Point2D:
        x_lim = self.field_width * 0.5 - self.boundary_margin
        y_lim = self.field_length * 0.5 - self.boundary_margin
        return Point2D(
            clamp(point.x, -x_lim, x_lim),
            clamp(point.y, -y_lim, y_lim),
        )

    def _set_state(self, state: str):
        if state != self.state:
            self.get_logger().info(f"striker_legacy state: {state}")
            if state != self.DRIBBLE:
                self._dribble_sub     = self._ALIGN
                self._kick_start_time = None
            self.state = state

    def _log_debug(self, message: str):
        if self.debug_log_interval <= 0.0:
            return
        now = self._now_seconds()
        if now - self._last_debug_log_time >= self.debug_log_interval:
            self.get_logger().info(message)
            self._last_debug_log_time = now

    def _publish_state(self):
        msg      = String()
        msg.data = self.state
        self._role_state_pub.publish(msg)
        if self._intent_pub is None:
            return

        ball = _pose_to_point(self.ball_pose) if self.ball_pose is not None else None
        me = _pose_to_point(self.self_pose) if self.self_pose is not None else None
        has_ball = ball is not None and me is not None and self._ball_controlled(me, ball)
        payload = {
            "stamp": self._now_seconds(),
            "team": self._team,
            "robot": self._robot_id,
            "role": "striker",
            "state": self.state,
            "target": ({"x": ball.x, "y": ball.y} if ball is not None else None),
            "has_ball": has_ball,
            "priority": 60 if has_ball else 40,
            "controller": "legacy",
        }
        intent_msg = String()
        intent_msg.data = json.dumps(payload)
        self._intent_pub.publish(intent_msg)


# ── module-level helpers ───────────────────────────────────────────────────────

def _pose_to_point(msg: PoseStamped) -> Point2D:
    return Point2D(msg.pose.position.x, msg.pose.position.y)
