"""Motion command bridge for the Cyberdog chassis.

The real robot consumes ``protocol/msg/MotionServoCmd``. Some simulation or
CI environments only provide core ROS2 messages, so this module falls back to
``geometry_msgs/Twist`` when the Cyberdog protocol package is unavailable.
"""

from geometry_msgs.msg import Twist
from rclpy.node import Node

try:
    from protocol.msg import MotionServoCmd
except ImportError:
    MotionServoCmd = None


def _clamp(value, limit):
    limit = abs(float(limit))
    return max(-limit, min(limit, float(value)))


def _scale_planar_velocity(speed_x, speed_y, limit_x, limit_y):
    speed_x = float(speed_x)
    speed_y = float(speed_y)
    limit_x = abs(float(limit_x))
    limit_y = abs(float(limit_y))
    scale = 1.0

    if speed_x != 0.0:
        scale = min(scale, limit_x / abs(speed_x) if limit_x > 0.0 else 0.0)
    if speed_y != 0.0:
        scale = min(scale, limit_y / abs(speed_y) if limit_y > 0.0 else 0.0)

    return speed_x * scale, speed_y * scale


class MoveCommander(Node):
    """Owns the outbound chassis command topic and applies configured limits."""

    def __init__(self, name="move_commander", config=None, dog_namespace=None):
        super().__init__(name)
        config = config or {}
        move_config = config.get("move", {})
        self.dog_name = dog_namespace or config.get("dog_namespace", "Hephaestus_1")
        self.speed_x = 0.0
        self.speed_y = 0.0
        self.speed_yaw = 0.0
        self.walk_motion_id = int(move_config.get("motion_id", 303))
        self.motion_id = self.walk_motion_id
        self.stop_motion_id = int(move_config.get("stop_motion_id", 101))
        self.max_speed_x = float(move_config.get("max_speed_x", 1.0))
        self.max_speed_y = float(move_config.get("max_speed_y", 1.0))
        self.max_speed_yaw = float(move_config.get("max_speed_yaw", 1.5))
        self.command_sign_x = float(move_config.get("command_sign_x", 1.0))
        self.command_sign_y = float(move_config.get("command_sign_y", 1.0))
        self.command_sign_yaw = float(move_config.get("command_sign_yaw", 1.0))
        self.step_height = float(move_config.get("step_height", 0.05))
        self.publish_on_change = bool(move_config.get("publish_on_change", False))
        self.command_epsilon = float(move_config.get("command_epsilon", 1e-4))
        self._motion_changed = False
        rate = float(move_config.get("publish_rate_hz", 10.0))
        topic = f"/{self.dog_name}/motion_servo_cmd"
        if MotionServoCmd is None:
            self.command_message = "twist"
            self.pub = self.create_publisher(Twist, topic, 10)
            self.get_logger().warning(
                "protocol.msg.MotionServoCmd is unavailable; publishing Twist for tests"
            )
        else:
            self.command_message = "motion_servo"
            self.pub = self.create_publisher(MotionServoCmd, topic, 10)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.timer_callback)

    def change_speed(self, speed_x, speed_y, speed_yaw):
        """Update body-frame speed command after applying configured limits."""

        next_speed_x, next_speed_y = _scale_planar_velocity(
            speed_x,
            speed_y,
            self.max_speed_x,
            self.max_speed_y,
        )
        next_speed_yaw = _clamp(speed_yaw, self.max_speed_yaw)
        current_speed_x = getattr(self, "speed_x", 0.0)
        current_speed_y = getattr(self, "speed_y", 0.0)
        current_speed_yaw = getattr(self, "speed_yaw", 0.0)
        changed = (
            abs(next_speed_x - current_speed_x) > getattr(self, "command_epsilon", 1e-4)
            or abs(next_speed_y - current_speed_y) > getattr(self, "command_epsilon", 1e-4)
            or abs(next_speed_yaw - current_speed_yaw) > getattr(self, "command_epsilon", 1e-4)
            or getattr(self, "_motion_changed", False)
        )
        self.speed_x = next_speed_x
        self.speed_y = next_speed_y
        self.speed_yaw = next_speed_yaw
        if changed and getattr(self, "publish_on_change", False):
            self.timer_callback()

    def change_motion_id(self, motion_id):
        """Switch the Cyberdog gait/motion mode used by MotionServoCmd."""

        motion_id = int(motion_id)
        if motion_id != self.motion_id:
            self.motion_id = motion_id
            self._motion_changed = True

    def stop(self, use_stop_motion=True):
        """Request zero velocity and optionally switch to the configured stop ID."""

        if use_stop_motion:
            self.change_motion_id(self.stop_motion_id)
        self.change_speed(0.0, 0.0, 0.0)

    def timer_callback(self):
        """Publish the latest command at a stable rate."""

        vx = self.command_sign_x * self.speed_x
        vy = self.command_sign_y * self.speed_y
        wz = self.command_sign_yaw * self.speed_yaw

        if self.command_message == "twist":
            msg = Twist()
            msg.linear.x = vx
            msg.linear.y = vy
            msg.angular.z = wz
        else:
            msg = MotionServoCmd()
            msg.motion_id = self.motion_id
            msg.cmd_type = 1
            msg.value = 2
            msg.vel_des = [vx, vy, wz]
            msg.step_height = [self.step_height, self.step_height]
        self.pub.publish(msg)
        self._motion_changed = False


class basic_move(MoveCommander):
    def __init__(self, name="basic_move", config=None, dog_namespace=None):
        super().__init__(name=name, config=config, dog_namespace=dog_namespace)


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = MoveCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
