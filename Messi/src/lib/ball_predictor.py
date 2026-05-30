"""Ball-path predictor for the goalkeeper.

The predictor listens to normalized soccer topics rather than raw VRPN names.
It projects the ball trajectory onto the moving ``defense_goal`` rigid body's
goal line and publishes a clamped guard target inside the goal mouth.
"""

import math
from collections import deque
from typing import Deque, Optional, Tuple

from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node

from .geometry import Point2D, clamp, quaternion_from_yaw, yaw_to_point
from .topics import soccer_topic


def stamp_to_seconds(stamp) -> float:
    """Convert a ROS builtin_interfaces/Time stamp to seconds."""

    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def predict_guard_target(
    ball: Point2D,
    velocity: Optional[Point2D],
    defense_goal: Point2D,
    goal_width: float,
    guard_depth: float,
    guard_margin: float,
    min_ball_speed: float,
) -> Point2D:
    """Predict where the keeper should stand on the defense line."""

    half_width = max(float(goal_width) * 0.5 - float(guard_margin), 0.05)
    x_min = defense_goal.x - half_width
    x_max = defense_goal.x + half_width

    # Move from the goal rigid body toward midfield so the keeper stands just
    # in front of the line rather than directly on the posts.
    inward_sign = -1.0 if defense_goal.y > 0.0 else 1.0
    guard_y = defense_goal.y + inward_sign * float(guard_depth)
    target_x = ball.x if velocity is None else defense_goal.x

    if velocity is not None:
        speed = math.hypot(velocity.x, velocity.y)
        moving_toward_goal = (defense_goal.y - ball.y) * velocity.y > 0.0
        if speed < min_ball_speed:
            # Slow balls do not have a stable trajectory; matching lateral ball
            # position is more useful than overfitting a noisy velocity sample.
            target_x = ball.x
        elif moving_toward_goal and abs(velocity.y) > 1e-4:
            time_to_goal_line = (defense_goal.y - ball.y) / velocity.y
            if time_to_goal_line >= 0.0:
                target_x = ball.x + velocity.x * time_to_goal_line

    return Point2D(clamp(target_x, x_min, x_max), guard_y)


class BallPredictorNode(Node):
    """Publishes ``/soccer/goalkeeper/guard_target`` from ball and goal state."""

    def __init__(self, config, name="ball_predictor"):
        super().__init__(name)
        self.config = config
        field_config = config.get("field", {})
        goalkeeper_config = config.get("goalkeeper", {})
        self.frame_id = config.get("frames", {}).get("field", "field_red")
        self.goal_width = float(field_config.get("goal_width", 1.5))
        self.guard_depth = float(goalkeeper_config.get("guard_depth", 0.45))
        self.guard_margin = float(goalkeeper_config.get("guard_margin", 0.12))
        self.min_ball_speed = float(goalkeeper_config.get("min_ball_speed", 0.08))

        self.ball_pose: Optional[PoseStamped] = None
        self.ball_twist: Optional[TwistStamped] = None
        self.defense_goal_pose: Optional[PoseStamped] = None
        self.ball_history: Deque[Tuple[float, Point2D]] = deque(maxlen=6)

        self.guard_pub = self.create_publisher(
            PoseStamped,
            soccer_topic(config, "goalkeeper", "guard_target"),
            10,
        )
        self.ball_sub = self.create_subscription(
            PoseStamped,
            soccer_topic(config, "world", "ball"),
            self._ball_pose_callback,
            10,
        )
        self.twist_sub = self.create_subscription(
            TwistStamped,
            soccer_topic(config, "world", "ball_twist"),
            self._ball_twist_callback,
            10,
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            soccer_topic(config, "world", "defense_goal"),
            self._defense_goal_callback,
            10,
        )
        rate = float(goalkeeper_config.get("control_rate_hz", 10.0))
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._timer_callback)

    def _ball_pose_callback(self, msg):
        self.ball_pose = msg
        timestamp = stamp_to_seconds(msg.header.stamp)
        if timestamp <= 0.0:
            timestamp = self.get_clock().now().nanoseconds * 1e-9
        self.ball_history.append(
            (
                timestamp,
                Point2D(msg.pose.position.x, msg.pose.position.y),
            )
        )

    def _ball_twist_callback(self, msg):
        self.ball_twist = msg

    def _defense_goal_callback(self, msg):
        self.defense_goal_pose = msg

    def _timer_callback(self):
        if self.ball_pose is None or self.defense_goal_pose is None:
            return

        ball = Point2D(
            self.ball_pose.pose.position.x,
            self.ball_pose.pose.position.y,
        )
        defense_goal = Point2D(
            self.defense_goal_pose.pose.position.x,
            self.defense_goal_pose.pose.position.y,
        )
        target = predict_guard_target(
            ball=ball,
            velocity=self._ball_velocity(),
            defense_goal=defense_goal,
            goal_width=self.goal_width,
            guard_depth=self.guard_depth,
            guard_margin=self.guard_margin,
            min_ball_speed=self.min_ball_speed,
        )
        self.guard_pub.publish(self._target_message(target, ball))

    def _ball_velocity(self) -> Optional[Point2D]:
        if self.ball_twist is not None:
            return Point2D(
                self.ball_twist.twist.linear.x,
                self.ball_twist.twist.linear.y,
            )
        if len(self.ball_history) < 2:
            return None

        oldest_time, oldest = self.ball_history[0]
        newest_time, newest = self.ball_history[-1]
        dt = newest_time - oldest_time
        if dt <= 1e-4:
            return None
        return Point2D((newest.x - oldest.x) / dt, (newest.y - oldest.y) / dt)

    def _target_message(self, target: Point2D, ball: Point2D) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = target.x
        msg.pose.position.y = target.y
        yaw = yaw_to_point(target, ball)
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg
