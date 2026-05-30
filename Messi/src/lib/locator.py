"""VRPN locator and relative-state publisher.

This node is the boundary between motion-capture poses and soccer logic. It
subscribes to every configured rigid body, builds the field frame dynamically
from the two goal centers, republishes field-frame poses under
``/soccer/world/*``, and also publishes each object in the robot's
``base_link`` frame under ``/soccer/relative/*``.
"""

from dataclasses import dataclass

from geometry_msgs.msg import AccelStamped, PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile

from .geometry import (
    build_field_transform,
    field_to_body,
    normalize_angle,
    quaternion_from_yaw,
    yaw_from_quaternion,
)
from .topics import soccer_topic


@dataclass
class ObjectState:
    """Latest VRPN samples for one semantic object."""

    pose: PoseStamped = None
    twist: TwistStamped = None
    accel: AccelStamped = None


class LocatorNode(Node):
    """Subscribes to configured VRPN rigids and publishes normalized state topics."""

    def __init__(self, config, name="locator"):
        super().__init__(name)
        self.config = config
        self.frames = config.get("frames", {})
        self.field_frame = self.frames.get("field", "field_red")
        self.base_frame = self.frames.get("base", "base_link")
        self.rigids = dict(config.get("rigids", {}))
        field_config = config.get("field", {})
        configured_length = float(field_config.get("length", 10.0))
        configured_width = float(field_config.get("width", 5.5))
        if configured_length > 1e-6:
            self.width_to_length_ratio = configured_width / configured_length
        else:
            self.width_to_length_ratio = 0.55
        self.goal_left_object, self.goal_right_object = self._resolve_field_goal_objects()
        self.field_transform = None
        self._last_field_goal_positions = None
        self._logged_missing_field_goals = False
        self._logged_invalid_field_goals = False
        locator_config = config.get("locator", {})
        self.static_field_transform = bool(locator_config.get("static_field_transform", False))
        self.immediate_publish = bool(locator_config.get("immediate_publish", True))
        self.low_latency_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        self.required_objects = set(locator_config.get("required_objects", []))
        self.states = {object_name: ObjectState() for object_name in self.rigids}
        self.world_pose_publishers = {}
        self.relative_pose_publishers = {}
        self.world_ball_twist_pub = self.create_publisher(
            TwistStamped, soccer_topic(config, "world", "ball_twist"), self.low_latency_qos
        )
        self.relative_ball_twist_pub = self.create_publisher(
            TwistStamped, soccer_topic(config, "relative", "ball_twist"), self.low_latency_qos
        )
        # Keep explicit references to VRPN subscriptions without touching
        # rclpy.Node's own internal subscription bookkeeping.
        self.vrpn_subscriptions = []

        for object_name, rigid_name in self.rigids.items():
            self.world_pose_publishers[object_name] = self.create_publisher(
                PoseStamped, soccer_topic(config, "world", object_name), self.low_latency_qos
            )
            self.relative_pose_publishers[object_name] = self.create_publisher(
                PoseStamped, soccer_topic(config, "relative", object_name), self.low_latency_qos
            )
            self.vrpn_subscriptions.extend(
                [
                    self.create_subscription(
                        PoseStamped,
                        f"/vrpn/{rigid_name}/pose",
                        self._make_pose_callback(object_name),
                        self.low_latency_qos,
                    ),
                    self.create_subscription(
                        TwistStamped,
                        f"/vrpn/{rigid_name}/twist",
                        self._make_twist_callback(object_name),
                        self.low_latency_qos,
                    ),
                    self.create_subscription(
                        AccelStamped,
                        f"/vrpn/{rigid_name}/accel",
                        self._make_accel_callback(object_name),
                        self.low_latency_qos,
                    ),
                ]
            )
            self.get_logger().info(f"Subscribing to /vrpn/{rigid_name}/pose as {object_name}")

        rate = float(config.get("locator", {}).get("publish_rate_hz", 20.0))
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.publish_state)

    def _make_pose_callback(self, object_name):
        def callback(msg):
            self.states[object_name].pose = msg
            if self.immediate_publish:
                self._publish_immediate_pose(object_name)

        return callback

    def _make_twist_callback(self, object_name):
        def callback(msg):
            self.states[object_name].twist = msg
            if self.immediate_publish and object_name == "ball":
                self._publish_ball_twist()

        return callback

    def _make_accel_callback(self, object_name):
        def callback(msg):
            self.states[object_name].accel = msg

        return callback

    def is_ready(self):
        return self._update_field_transform() and all(
            self.states[name].pose is not None for name in self.required_objects
        )

    def get_pose(self, object_name):
        state = self.states.get(object_name)
        return state.pose if state else None

    def get_twist(self, object_name):
        state = self.states.get(object_name)
        return state.twist if state else None

    def publish_state(self):
        if not self._update_field_transform():
            return

        self_pose = self.get_pose("self")
        if not self_pose:
            # Until our own pose arrives, relative coordinates are undefined.
            self._publish_world_only()
            return

        world_self_pose = self._world_pose(self_pose)
        self_yaw = yaw_from_quaternion(world_self_pose.pose.orientation)
        for object_name, state in self.states.items():
            if state.pose is None:
                continue
            world_pose = self._world_pose(state.pose)
            self.world_pose_publishers[object_name].publish(world_pose)
            self.relative_pose_publishers[object_name].publish(
                self._relative_pose(world_pose, world_self_pose, self_yaw)
            )

        ball_state = self.states.get("ball")
        if ball_state and ball_state.twist is not None:
            self._publish_ball_twist(world_self_pose, self_yaw)

    def _publish_world_only(self):
        if self.field_transform is None:
            return
        for object_name, state in self.states.items():
            if state.pose is not None:
                self.world_pose_publishers[object_name].publish(self._world_pose(state.pose))

    def _publish_immediate_pose(self, object_name):
        if not self._update_field_transform():
            return
        state = self.states.get(object_name)
        if state is None or state.pose is None:
            return

        world_pose = self._world_pose(state.pose)
        self.world_pose_publishers[object_name].publish(world_pose)

        self_pose = self.get_pose("self")
        if self_pose is None:
            return

        world_self_pose = world_pose if object_name == "self" else self._world_pose(self_pose)
        self_yaw = yaw_from_quaternion(world_self_pose.pose.orientation)
        if object_name == "self":
            for other_name, other_state in self.states.items():
                if other_state.pose is None:
                    continue
                other_world_pose = world_self_pose if other_name == "self" else self._world_pose(other_state.pose)
                self.relative_pose_publishers[other_name].publish(
                    self._relative_pose(other_world_pose, world_self_pose, self_yaw)
                )
            self._publish_ball_twist(world_self_pose, self_yaw)
            return

        self.relative_pose_publishers[object_name].publish(
            self._relative_pose(world_pose, world_self_pose, self_yaw)
        )

    def _publish_ball_twist(self, world_self_pose=None, self_yaw=None):
        if not self._update_field_transform():
            return
        ball_state = self.states.get("ball")
        if ball_state is None or ball_state.twist is None:
            return

        world_twist = self._world_twist(ball_state.twist)
        self.world_ball_twist_pub.publish(world_twist)

        if self_yaw is None:
            self_pose = self.get_pose("self")
            if self_pose is None:
                return
            world_self_pose = self._world_pose(self_pose)
            self_yaw = yaw_from_quaternion(world_self_pose.pose.orientation)
        self.relative_ball_twist_pub.publish(self._relative_twist(world_twist, self_yaw))

    def _world_pose(self, source):
        msg = PoseStamped()
        msg.header.stamp = source.header.stamp
        msg.header.frame_id = self.field_frame
        x, y, z = self.field_transform.position_to_field(source.pose.position)
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        yaw = self.field_transform.yaw_to_field(yaw_from_quaternion(source.pose.orientation))
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def _world_twist(self, source):
        msg = TwistStamped()
        msg.header.stamp = source.header.stamp
        msg.header.frame_id = self.field_frame
        linear = self.field_transform.vector_to_field(
            source.twist.linear.x,
            source.twist.linear.y,
        )
        msg.twist.linear.x = linear.x
        msg.twist.linear.y = linear.y
        msg.twist.linear.z = source.twist.linear.z
        msg.twist.angular.x = source.twist.angular.x
        msg.twist.angular.y = source.twist.angular.y
        msg.twist.angular.z = source.twist.angular.z
        return msg

    def _relative_pose(self, source, self_pose, self_yaw):
        msg = PoseStamped()
        msg.header.stamp = source.header.stamp
        msg.header.frame_id = self.base_frame
        dx = source.pose.position.x - self_pose.pose.position.x
        dy = source.pose.position.y - self_pose.pose.position.y
        local = field_to_body(dx, dy, self_yaw)
        msg.pose.position.x = local.x
        msg.pose.position.y = local.y
        msg.pose.position.z = source.pose.position.z - self_pose.pose.position.z
        object_yaw = yaw_from_quaternion(source.pose.orientation)
        _, _, qz, qw = quaternion_from_yaw(normalize_angle(object_yaw - self_yaw))
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def _relative_twist(self, source, self_yaw):
        msg = TwistStamped()
        msg.header.stamp = source.header.stamp
        msg.header.frame_id = self.base_frame
        local_linear = field_to_body(
            source.twist.linear.x,
            source.twist.linear.y,
            self_yaw,
        )
        msg.twist.linear.x = local_linear.x
        msg.twist.linear.y = local_linear.y
        msg.twist.linear.z = source.twist.linear.z
        msg.twist.angular.x = source.twist.angular.x
        msg.twist.angular.y = source.twist.angular.y
        msg.twist.angular.z = source.twist.angular.z
        return msg

    def _update_field_transform(self):
        if self.static_field_transform and self.field_transform is not None:
            return True

        if self.goal_left_object is None or self.goal_right_object is None:
            if not self._logged_missing_field_goals:
                self.get_logger().warning(
                    "Cannot build field frame: configure or infer goal_left and goal_right rigids"
                )
                self._logged_missing_field_goals = True
            return False

        left_pose = self.get_pose(self.goal_left_object)
        right_pose = self.get_pose(self.goal_right_object)
        if left_pose is None or right_pose is None:
            return False

        positions = self._field_goal_positions(left_pose, right_pose)
        if self.field_transform is not None and positions == self._last_field_goal_positions:
            return True

        try:
            self.field_transform = build_field_transform(
                left_pose.pose.position,
                right_pose.pose.position,
                self.width_to_length_ratio,
            )
        except ValueError as exc:
            if not self._logged_invalid_field_goals:
                self.get_logger().warning(str(exc))
                self._logged_invalid_field_goals = True
            return False

        self._logged_invalid_field_goals = False
        self._last_field_goal_positions = positions
        return True

    @staticmethod
    def _field_goal_positions(left_pose, right_pose):
        left = left_pose.pose.position
        right = right_pose.pose.position
        return (
            left.x,
            left.y,
            getattr(left, "z", 0.0),
            right.x,
            right.y,
            getattr(right, "z", 0.0),
        )

    def _resolve_field_goal_objects(self):
        locator_config = self.config.get("locator", {})
        configured = locator_config.get("field_goals", {})
        left = self._configured_goal_object(
            configured.get("left")
            or locator_config.get("goal_left")
            or locator_config.get("goal_left_object")
        )
        right = self._configured_goal_object(
            configured.get("right")
            or locator_config.get("goal_right")
            or locator_config.get("goal_right_object")
        )
        if left and right:
            return left, right

        by_object = {object_name.lower(): object_name for object_name in self.rigids}
        left = left or by_object.get("goal_left")
        right = right or by_object.get("goal_right")
        if left and right:
            return left, right

        by_rigid = {}
        for object_name, rigid_name in self.rigids.items():
            by_rigid.setdefault(self._rigid_leaf(rigid_name), object_name)
        left = left or by_rigid.get("goal_left") or by_rigid.get("goal_a")
        right = right or by_rigid.get("goal_right") or by_rigid.get("goal_b")
        return left, right

    def _configured_goal_object(self, value):
        if value is None:
            return None
        value = str(value).strip()
        if value in self.rigids:
            return value
        leaf = self._rigid_leaf(value)
        for object_name, rigid_name in self.rigids.items():
            if self._rigid_leaf(rigid_name) == leaf:
                return object_name
        return None

    @staticmethod
    def _rigid_leaf(value):
        parts = str(value).strip().strip("/").lower().split("/")
        if len(parts) >= 2 and parts[-1] in {"pose", "twist", "accel"}:
            return parts[-2]
        return parts[-1] if parts else ""
