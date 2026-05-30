"""Waypoint follower for striker paths.

This node converts field-frame waypoints from ``PathPlannerNode`` into body
frame velocity commands. It intentionally talks to ``MoveCommander`` by direct
Python method calls because all nodes run in one process from ``main.py`` and
we do not want to introduce a custom velocity-command message package.
"""

import math
from typing import List, Optional

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from .geometry import Point2D, field_to_body, normalize_angle, yaw_from_quaternion, yaw_to_point
from .path_planner import PATH_MESSAGE_TYPE, path_points_from_message
from .topics import soccer_topic


class PathFollowerNode(Node):
    """Tracks a waypoint path and drives the shared ``MoveCommander``."""

    def __init__(self, config, move_commander, name="path_follower"):
        super().__init__(name)
        self.config = config
        self.move = move_commander
        path_config = config.get("path", {})
        striker_config = config.get("striker", {})
        move_config = config.get("move", {})

        self.enabled = False
        self.self_pose: Optional[PoseStamped] = None
        self.target_pose: Optional[PoseStamped] = None
        self.path: List[Point2D] = []
        self.waypoint_index = 0
        self.walk_motion_id = int(move_config.get("motion_id", 303))
        self.waypoint_tolerance = float(path_config.get("waypoint_tolerance", 0.18))
        self.goal_tolerance = float(path_config.get("goal_tolerance", 0.18))
        self.lookahead_points = max(int(path_config.get("lookahead_points", 2)), 1)
        self.position_gain = float(striker_config.get("approach_gain", 0.8))
        self.lateral_gain = float(striker_config.get("lateral_gain", self.position_gain))
        self.yaw_gain = float(striker_config.get("yaw_gain", 1.2))

        self.path_sub = self.create_subscription(
            PATH_MESSAGE_TYPE,
            soccer_topic(config, "striker", "path"),
            self._path_callback,
            10,
        )
        self.target_sub = self.create_subscription(
            PoseStamped,
            soccer_topic(config, "striker", "target"),
            self._target_callback,
            10,
        )
        self.self_sub = self.create_subscription(
            PoseStamped,
            soccer_topic(config, "world", "self"),
            self._self_callback,
            10,
        )
        rate = float(striker_config.get("control_rate_hz", 10.0))
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._timer_callback)

    def set_enabled(self, enabled: bool):
        """Enable or pause path following without destroying the current path."""

        enabled = bool(enabled)
        if self.enabled == enabled:
            return
        self.enabled = enabled
        if not enabled:
            self.move.change_speed(0.0, 0.0, 0.0)

    def reset_path(self):
        """Forget the current path so the next planner message starts fresh."""

        self.path = []
        self.waypoint_index = 0

    def _path_callback(self, msg):
        points = path_points_from_message(msg)
        if points != self.path:
            self.path = points
            self.waypoint_index = 0

    def _self_callback(self, msg):
        self.self_pose = msg

    def _target_callback(self, msg):
        self.target_pose = msg

    def _timer_callback(self):
        if not self.enabled:
            return
        if self.self_pose is None or not self.path:
            self.move.stop(use_stop_motion=False)
            return

        self.move.change_motion_id(self.walk_motion_id)
        self._advance_waypoint_index()
        if self._at_final_goal():
            if self.target_pose is None:
                self.move.change_speed(0.0, 0.0, 0.0)
                return
            yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
            desired_yaw = yaw_from_quaternion(self.target_pose.pose.orientation)
            yaw_error = normalize_angle(desired_yaw - yaw)
            if abs(yaw_error) <= 0.12:
                self.move.change_speed(0.0, 0.0, 0.0)
            else:
                self.move.change_speed(0.0, 0.0, self.yaw_gain * yaw_error)
            return

        target = self._lookahead_target()
        current = Point2D(
            self.self_pose.pose.position.x,
            self.self_pose.pose.position.y,
        )
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        vector = field_to_body(target.x - current.x, target.y - current.y, yaw)
        desired_yaw = (
            yaw_from_quaternion(self.target_pose.pose.orientation)
            if self.target_pose is not None
            else yaw_to_point(current, self.path[-1])
        )
        yaw_error = normalize_angle(desired_yaw - yaw)

        # MoveCommander owns final speed limits; this node only shapes the
        # body-frame command so the striker converges to the waypoint path.
        self.move.change_speed(
            self.position_gain * vector.x,
            self.lateral_gain * vector.y,
            self.yaw_gain * yaw_error,
        )

    def _advance_waypoint_index(self):
        current = Point2D(
            self.self_pose.pose.position.x,
            self.self_pose.pose.position.y,
        )
        while self.waypoint_index < len(self.path) - 1:
            waypoint = self.path[self.waypoint_index]
            if math.hypot(waypoint.x - current.x, waypoint.y - current.y) > self.waypoint_tolerance:
                break
            self.waypoint_index += 1

    def _at_final_goal(self) -> bool:
        current = Point2D(
            self.self_pose.pose.position.x,
            self.self_pose.pose.position.y,
        )
        goal = self.path[-1]
        return math.hypot(goal.x - current.x, goal.y - current.y) <= self.goal_tolerance

    def _lookahead_target(self) -> Point2D:
        index = min(self.waypoint_index + self.lookahead_points - 1, len(self.path) - 1)
        return self.path[index]
