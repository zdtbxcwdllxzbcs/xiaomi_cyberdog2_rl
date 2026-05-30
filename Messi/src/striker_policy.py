"""Model-backed striker controller.

This node replaces the scripted striker with a PPO policy trained in
``gym/soccer_env.py``. It keeps the ROS-facing contract small: subscribe to the
normalized world topics, build the same observation vector used during
training, run ``model.predict()``, then command body-frame velocity through
``MoveCommander``.
"""

import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from std_msgs.msg import String

from .lib.geometry import (
    Point2D,
    clamp,
    field_to_body,
    normalize_angle,
    yaw_from_quaternion,
    yaw_to_point,
)
from .lib.topics import global_soccer_topic, soccer_topic


REPO_ROOT = Path(__file__).resolve().parents[1]
_ACTIVE_GAME_STATES = {"PLAYING", "KICKOFF_TEAM_A", "KICKOFF_TEAM_B"}

TRAINING_MAX_LINEAR_SPD = 1.2
TRAINING_MAX_ANGULAR_SPD = 3.0
BALL_VEL_OBS_LIMIT = 5.0

DEFAULT_MODEL_PATHS = (
    "checkpoints/striker_rl/phase3/best/best_model.zip",
    "checkpoints/striker_rl/phase3/striker_final.zip",
    "checkpoints/striker_rl/phase2/best/best_model.zip",
    "checkpoints/striker_rl/phase2/striker_final.zip",
    "checkpoints/striker_rl/phase1/best/best_model.zip",
    "checkpoints/striker_rl/phase1/striker_final.zip",
)


class StrikerPolicyNode(Node):
    """High-level striker node driven by a trained SB3 model."""

    WAITING = "waiting"
    RECOVER = "recover"
    POLICY = "policy"

    def __init__(self, config, move_commander, name="striker_policy"):
        super().__init__(name)
        self.config = config
        self.move = move_commander

        field_config = config.get("field", {})
        striker_config = config.get("striker", {})
        policy_config = config.get("striker_policy", {})
        move_config = config.get("move", {})
        sim_config = config.get("sim", {})

        self.field_length = float(field_config.get("length", 10.0))
        self.field_width = float(field_config.get("width", 5.55))
        self.half_length = self.field_length * 0.5
        self.half_width = self.field_width * 0.5
        self.out_of_bounds_margin = float(striker_config.get("out_of_bounds_margin", 0.15))

        self.max_linear_speed = float(
            policy_config.get("max_linear_speed", TRAINING_MAX_LINEAR_SPD)
        )
        self.max_angular_speed = float(
            policy_config.get("max_angular_speed", TRAINING_MAX_ANGULAR_SPD)
        )
        self.recover_gain = float(policy_config.get("recover_gain", 0.8))
        self.recover_yaw_gain = float(policy_config.get("recover_yaw_gain", 1.2))
        self.possession_distance = float(striker_config.get("possession_distance", 0.45))
        default_motion_id = striker_config.get(
            "attack_motion_id",
            move_config.get("motion_id", 303),
        )
        self.motion_id = int(policy_config.get("motion_id", default_motion_id))
        self.deterministic = bool(policy_config.get("deterministic", True))

        self._team = sim_config.get("team", "")
        self._robot_id = sim_config.get("robot_id", "")
        self._game_state = "INIT"
        self.state = self.WAITING

        self.self_pose: Optional[PoseStamped] = None
        self.ball_pose: Optional[PoseStamped] = None
        self.ball_twist: Optional[TwistStamped] = None
        self.attack_goal_pose: Optional[PoseStamped] = None
        self._opponent_poses = {}

        self._model_path = self._resolve_model_path(policy_config)
        self._device = self._resolve_device(str(policy_config.get("device", "auto")))
        self._model = self._load_model(self._model_path, self._device)

        self._role_state_pub = self.create_publisher(
            String, soccer_topic(config, "striker", "state"), 10
        )
        if self._team and self._robot_id:
            self._intent_pub = self.create_publisher(
                String,
                global_soccer_topic("team", self._team, self._robot_id, "intent"),
                10,
            )
        else:
            self._intent_pub = None

        self.create_subscription(
            String, global_soccer_topic("game_state"), self._game_state_callback, 10
        )
        self.create_subscription(
            PoseStamped, soccer_topic(config, "world", "self"), self._self_callback, 10
        )
        self.create_subscription(
            PoseStamped, soccer_topic(config, "world", "ball"), self._ball_callback, 10
        )
        self.create_subscription(
            TwistStamped,
            soccer_topic(config, "world", "ball_twist"),
            self._ball_twist_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            soccer_topic(config, "world", "attack_goal"),
            self._attack_goal_callback,
            10,
        )
        for opponent in ("opponent_1", "opponent_2"):
            self.create_subscription(
                PoseStamped,
                soccer_topic(config, "world", opponent),
                self._make_opponent_callback(opponent),
                10,
            )

        rate = float(
            policy_config.get("control_rate_hz", striker_config.get("control_rate_hz", 10.0))
        )
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._timer_callback)
        self.get_logger().info(
            f"striker policy loaded: model={self._model_path} device={self._device}"
        )

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

    def _make_opponent_callback(self, name):
        def callback(msg):
            self._opponent_poses[name] = msg

        return callback

    def _timer_callback(self):
        if self._game_state not in _ACTIVE_GAME_STATES:
            self._set_state(self.WAITING)
            self.move.stop(use_stop_motion=True)
            self._publish_role_state()
            return

        if (
            self.self_pose is None
            or self.ball_pose is None
            or self.attack_goal_pose is None
        ):
            self._set_state(self.WAITING)
            self.move.stop(use_stop_motion=True)
            self._publish_role_state()
            return

        self_point = self._point_from_pose(self.self_pose)
        if self._outside_soft_field(self_point):
            self._set_state(self.RECOVER)
            self._recover_to_center(self_point)
            self._publish_role_state()
            return

        self._set_state(self.POLICY)
        obs = self._observation()
        action, _ = self._model.predict(obs, deterministic=self.deterministic)
        self._apply_action(action)
        self._publish_role_state()

    def _observation(self) -> np.ndarray:
        self_point = self._point_from_pose(self.self_pose)
        ball = self._point_from_pose(self.ball_pose)
        attack_goal = self._point_from_pose(self.attack_goal_pose)
        attack_sign = self._attack_sign(attack_goal)

        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        canonical_yaw = attack_sign * yaw
        ball_vx = 0.0
        ball_vy = 0.0
        if self.ball_twist is not None:
            ball_vx = self.ball_twist.twist.linear.x
            ball_vy = self.ball_twist.twist.linear.y

        keeper = self._select_goalkeeper_pose(attack_goal)
        keeper_x = 0.0
        keeper_y = 0.0
        if keeper is not None:
            keeper_point = self._point_from_pose(keeper)
            keeper_x = keeper_point.x
            keeper_y = attack_sign * keeper_point.y

        return np.array(
            [
                self._clip(self_point.x / self.half_width),
                self._clip(attack_sign * self_point.y / self.half_length),
                math.sin(canonical_yaw),
                math.cos(canonical_yaw),
                self._clip(ball.x / self.half_width),
                self._clip(attack_sign * ball.y / self.half_length),
                self._clip(ball_vx / TRAINING_MAX_LINEAR_SPD, BALL_VEL_OBS_LIMIT),
                self._clip(attack_sign * ball_vy / TRAINING_MAX_LINEAR_SPD, BALL_VEL_OBS_LIMIT),
                self._clip(attack_goal.x / self.half_width),
                self._clip(attack_sign * attack_goal.y / self.half_length),
                self._clip(keeper_x / self.half_width),
                self._clip(keeper_y / self.half_length),
            ],
            dtype=np.float32,
        )

    def _apply_action(self, action):
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape[0] != 3:
            self.get_logger().warning(f"policy returned invalid action shape: {values.shape}")
            self.move.stop(use_stop_motion=False)
            return

        values = np.clip(values, -1.0, 1.0)
        attack_sign = self._attack_sign(self._point_from_pose(self.attack_goal_pose))
        vx = float(values[0] * self.max_linear_speed)
        vy = float(values[1] * self.max_linear_speed)
        wz = float(values[2] * self.max_angular_speed)

        if attack_sign < 0.0:
            vy = -vy
            wz = -wz

        self.move.change_motion_id(self.motion_id)
        self.move.change_speed(vx, vy, wz)

    def _recover_to_center(self, self_point: Point2D):
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        target = Point2D(0.0, 0.0)
        vector = field_to_body(target.x - self_point.x, target.y - self_point.y, yaw)
        yaw_error = normalize_angle(yaw_to_point(self_point, target) - yaw)
        self.move.change_speed(
            self.recover_gain * vector.x,
            self.recover_gain * vector.y,
            self.recover_yaw_gain * yaw_error,
        )

    def _select_goalkeeper_pose(self, attack_goal: Point2D):
        if not self._opponent_poses:
            return None
        return min(
            self._opponent_poses.values(),
            key=lambda pose: math.hypot(
                pose.pose.position.x - attack_goal.x,
                pose.pose.position.y - attack_goal.y,
            ),
        )

    def _outside_soft_field(self, point: Point2D) -> bool:
        x_limit = self.field_width * 0.5 - self.out_of_bounds_margin
        y_limit = self.field_length * 0.5 - self.out_of_bounds_margin
        return abs(point.x) > x_limit or abs(point.y) > y_limit

    def _set_state(self, state: str):
        if state != self.state:
            self.get_logger().info(f"striker policy state: {state}")
            self.state = state

    def _publish_role_state(self):
        msg = String()
        msg.data = self.state
        self._role_state_pub.publish(msg)
        if self._intent_pub is None:
            return

        ball = self._point_from_pose(self.ball_pose) if self.ball_pose else None
        self_point = self._point_from_pose(self.self_pose) if self.self_pose else None
        has_ball = (
            ball is not None
            and self_point is not None
            and math.hypot(ball.x - self_point.x, ball.y - self_point.y) <= self.possession_distance
        )
        payload = {
            "stamp": self.get_clock().now().nanoseconds * 1e-9,
            "team": self._team,
            "robot": self._robot_id,
            "role": "striker",
            "state": self.state,
            "target": ({"x": ball.x, "y": ball.y} if ball else None),
            "has_ball": has_ball,
            "priority": 70 if self.state == self.POLICY else 40,
            "policy_model": str(self._model_path),
        }
        intent_msg = String()
        intent_msg.data = json.dumps(payload)
        self._intent_pub.publish(intent_msg)

    def _resolve_model_path(self, policy_config: dict) -> Path:
        candidates = []
        configured = policy_config.get("model_path")
        if configured:
            candidates.append(str(configured))
        fallback_paths = policy_config.get("fallback_model_paths", [])
        if isinstance(fallback_paths, str):
            candidates.append(fallback_paths)
        else:
            candidates.extend(fallback_paths)
        candidates.extend(DEFAULT_MODEL_PATHS)

        checked = []
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = REPO_ROOT / path
            checked.append(path)
            if path.exists():
                return path
        checked_text = "\n  ".join(str(path) for path in checked)
        raise FileNotFoundError(f"No striker policy checkpoint found. Checked:\n  {checked_text}")

    def _load_model(self, model_path: Path, device: str):
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self._install_numpy_core_aliases()
        try:
            from stable_baselines3 import PPO

            return PPO.load(
                str(model_path),
                device=device,
                custom_objects=self._model_custom_objects(),
            )
        except ModuleNotFoundError:
            pass

        # Fallback for environments where SB3/PyTorch cannot be installed
        # (e.g. glibc 2.27 + Python 3.6 on the robot). Requires onnxruntime
        # and a sidecar .onnx file exported by scripts/export_policy_onnx.py.
        onnx_path = model_path.with_suffix(".onnx")
        if not onnx_path.exists():
            raise FileNotFoundError(
                "stable-baselines3 is not installed and no ONNX sidecar found at "
                f"{onnx_path}. Generate it with: python scripts/export_policy_onnx.py"
            )
        try:
            from .lib.policy_onnx import OnnxPolicy
        except ImportError:
            from lib.policy_onnx import OnnxPolicy

        self.get_logger().info(f"SB3 unavailable, loading ONNX policy: {onnx_path}")
        return OnnxPolicy(str(onnx_path))

    def _model_custom_objects(self):
        from gymnasium import spaces

        return {
            "_last_obs": None,
            "_last_episode_starts": None,
            "ep_info_buffer": deque(maxlen=100),
            "ep_success_buffer": deque(maxlen=100),
            "observation_space": spaces.Box(
                low=np.array(
                    [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -BALL_VEL_OBS_LIMIT, -BALL_VEL_OBS_LIMIT, -1.0, -1.0, -1.0, -1.0],
                    dtype=np.float32,
                ),
                high=np.array(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, BALL_VEL_OBS_LIMIT, BALL_VEL_OBS_LIMIT, 1.0, 1.0, 1.0, 1.0],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            ),
            "action_space": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        }

    @staticmethod
    def _install_numpy_core_aliases():
        """Let NumPy 1.x load checkpoints saved under NumPy 2.x."""

        import sys
        import numpy as np

        if hasattr(np, "_core"):
            return
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
        sys.modules.setdefault("numpy._core.numeric", numpy_numeric)

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        try:
            import torch
        except ModuleNotFoundError:
            return "cpu"
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _attack_sign(attack_goal: Point2D) -> float:
        return 1.0 if attack_goal.y >= 0.0 else -1.0

    @staticmethod
    def _point_from_pose(msg: PoseStamped) -> Point2D:
        return Point2D(msg.pose.position.x, msg.pose.position.y)

    @staticmethod
    def _clip(value: float, limit: float = 1.0) -> float:
        return float(clamp(value, -limit, limit))
