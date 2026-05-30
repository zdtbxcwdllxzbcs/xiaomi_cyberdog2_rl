"""Model-backed goalkeeper controller.

This node mirrors ``gym.goalkeeper_env.GoalkeeperEnv`` at runtime: subscribe to
normalized soccer topics, rebuild the 12-D goalkeeper observation, run a PPO
checkpoint when configured, and publish body-frame velocity through
``MoveCommander``. The default hybrid mode keeps the keeper inside its own
keeper zone and uses local rules for guard/intercept/clear safety.
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

from .lib.ball_predictor import predict_guard_target
from .lib.geometry import Point2D, clamp, field_to_body, normalize_angle, yaw_from_quaternion, yaw_to_point
from .lib.topics import global_soccer_topic, soccer_topic


REPO_ROOT = Path(__file__).resolve().parents[1]
_ACTIVE_GAME_STATES = {"PLAYING", "KICKOFF_TEAM_A", "KICKOFF_TEAM_B"}

TRAINING_MAX_LINEAR_SPD = 1.2
TRAINING_MAX_ANGULAR_SPD = 3.0
BALL_VEL_OBS_LIMIT = 5.0

DEFAULT_MODEL_PATHS = (
    "checkpoints/goalkeeper_rl/best/best_model.zip",
    "checkpoints/goalkeeper_rl/goalkeeper_final.zip",
)


class GoalkeeperPolicyNode(Node):
    """High-level goalkeeper node driven by a trained SB3 PPO model."""

    WAITING = "waiting"
    RECOVER = "recover"
    GUARD = "guard"
    INTERCEPT = "intercept"
    CLEAR = "clear"
    COUNTER = "striker"
    POLICY = "policy"

    def __init__(self, config, move_commander, name="goalkeeper_policy"):
        super().__init__(name)
        self.config = config
        self.move = move_commander

        field_config = config.get("field", {})
        goalkeeper_config = config.get("goalkeeper", {})
        controller_config = config.get("goalkeeper_controller", {})
        policy_config = config.get("goalkeeper_policy", {})
        move_config = config.get("move", {})
        sim_config = config.get("sim", {})

        self.field_length = float(field_config.get("length", 10.0))
        self.field_width = float(field_config.get("width", 5.55))
        self.half_length = self.field_length * 0.5
        self.half_width = self.field_width * 0.5
        self.goal_width = float(field_config.get("goal_width", 1.0))
        self.guard_depth = float(goalkeeper_config.get("guard_depth", 0.45))
        self.guard_margin = float(goalkeeper_config.get("guard_margin", 0.12))
        self.min_ball_speed = float(goalkeeper_config.get("min_ball_speed", 0.08))
        self.clear_gain = float(goalkeeper_config.get("clear_gain", 1.35))
        self.clear_distance = float(goalkeeper_config.get("clear_distance", 0.72))
        self.clear_contact_distance = float(goalkeeper_config.get("clear_contact_distance", 0.38))
        self.clear_forward_distance = float(goalkeeper_config.get("clear_forward_distance", 1.6))
        self.clear_center_pull = float(goalkeeper_config.get("clear_center_pull", 0.9))
        self.clear_backoff_distance = float(goalkeeper_config.get("clear_backoff_distance", 0.28))
        self.possession_distance = float(goalkeeper_config.get("possession_distance", 0.45))
        self.counter_distance = float(goalkeeper_config.get("counter_distance", self.clear_distance))
        self.counter_chase_distance = float(goalkeeper_config.get("counter_chase_distance", 1.3))
        self.counter_hold = float(goalkeeper_config.get("counter_hold", 1.8))
        self.keeper_zone_depth = float(goalkeeper_config.get("keeper_zone_depth", 2.4))
        self.keeper_zone_width = float(goalkeeper_config.get("keeper_zone_width", self.goal_width + 0.9))
        self.distant_ball_zone_depth = float(goalkeeper_config.get("distant_ball_zone_depth", self.half_length))
        self.distant_ball_zone_width = float(goalkeeper_config.get("distant_ball_zone_width", self.field_width))
        self.counter_zone_depth = float(goalkeeper_config.get("counter_zone_depth", self.keeper_zone_depth + 0.8))
        self.counter_zone_width = float(goalkeeper_config.get("counter_zone_width", self.keeper_zone_width + 0.7))
        configured_mode = controller_config.get("mode")
        if configured_mode is None:
            self.controller_mode = "hybrid" if policy_config.get("hybrid_clear", True) else "policy"
        else:
            self.controller_mode = str(configured_mode).strip().lower()
        if self.controller_mode not in ("hybrid", "policy"):
            raise ValueError("GoalkeeperPolicyNode only supports hybrid or policy controller modes")

        self.recover_gain = float(controller_config.get("recover_gain", policy_config.get("recover_gain", 0.8)))
        self.recover_yaw_gain = float(
            controller_config.get("recover_yaw_gain", policy_config.get("recover_yaw_gain", 1.2))
        )
        self.max_linear_speed = float(policy_config.get("max_linear_speed", TRAINING_MAX_LINEAR_SPD))
        self.max_angular_speed = float(
            policy_config.get("max_angular_speed", TRAINING_MAX_ANGULAR_SPD)
        )
        self.motion_id = int(
            controller_config.get("motion_id", policy_config.get("motion_id", move_config.get("motion_id", 303)))
        )
        self.deterministic = bool(policy_config.get("deterministic", True))
        self.hybrid_clear = self.controller_mode == "hybrid"
        self.safety_recover = bool(
            controller_config.get("safety_recover", policy_config.get("safety_recover", True))
        )
        self.max_goal_depth = float(
            controller_config.get(
                "max_goal_depth",
                policy_config.get("max_goal_depth", self.keeper_zone_depth + 0.35),
            )
        )
        self.recover_release_depth = float(
            controller_config.get(
                "recover_release_depth",
                policy_config.get("recover_release_depth", self.keeper_zone_depth),
            )
        )
        self.opponent_striker_key = str(policy_config.get("opponent_striker", "opponent_1"))

        self._team = sim_config.get("team", "")
        self._robot_id = sim_config.get("robot_id", "")
        self._game_state = "INIT"
        self.state = self.WAITING
        self._recovering = False
        self._counter_until = 0.0

        self.self_pose: Optional[PoseStamped] = None
        self.ball_pose: Optional[PoseStamped] = None
        self.ball_twist: Optional[TwistStamped] = None
        self.defense_goal_pose: Optional[PoseStamped] = None
        self._opponent_poses = {}

        self._model_path = None
        self._device = None
        self._model = None
        if not self.hybrid_clear:
            self._model_path = self._resolve_model_path(policy_config)
            self._device = self._resolve_device(str(policy_config.get("device", "auto")))
            self._model = self._load_model(self._model_path, self._device)

        self._role_state_pub = self.create_publisher(
            String, soccer_topic(config, "goalkeeper", "state"), 10
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
            soccer_topic(config, "world", "defense_goal"),
            self._defense_goal_callback,
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
            controller_config.get(
                "control_rate_hz",
                policy_config.get("control_rate_hz", goalkeeper_config.get("control_rate_hz", 10.0)),
            )
        )
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._timer_callback)
        if self.hybrid_clear:
            self.get_logger().info("goalkeeper controller mode=hybrid: rule guard/intercept/clear")
        else:
            self.get_logger().info(
                f"goalkeeper controller mode=policy: model={self._model_path} device={self._device}"
            )

    def _self_callback(self, msg):
        self.self_pose = msg

    def _ball_callback(self, msg):
        self.ball_pose = msg

    def _ball_twist_callback(self, msg):
        self.ball_twist = msg

    def _defense_goal_callback(self, msg):
        self.defense_goal_pose = msg

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

        if self.self_pose is None or self.ball_pose is None or self.defense_goal_pose is None:
            self._set_state(self.WAITING)
            self.move.stop(use_stop_motion=True)
            self._publish_role_state()
            return

        self_point = self._point_from_pose(self.self_pose)
        if self.safety_recover:
            if self._outside_soft_field(self_point):
                self._recovering = True
            elif self._recovering and self._inside_recover_release(self_point):
                self._recovering = False
            if self._recovering:
                self._set_state(self.RECOVER)
                self._recover_to_home(self_point)
                self._publish_role_state()
                return

        ball = self._point_from_pose(self.ball_pose)
        if self.hybrid_clear and self._ball_in_keeper_zone(ball):
            if self._should_clear(self_point):
                self._counter_until = self.get_clock().now().nanoseconds * 1e-9 + self.counter_hold
                self._set_state(self.CLEAR)
                self._clear_ball()
            else:
                self._set_state(self.INTERCEPT)
                self._active_defense_to_ball(ball)
            self._publish_role_state()
            return

        if self.hybrid_clear and self._should_counter(self_point, ball):
            self._set_state(self.COUNTER)
            self._clear_ball()
            self._publish_role_state()
            return

        if self.hybrid_clear:
            self._set_state(self.GUARD)
            self._guard_with_policy()
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
        defense_goal = self._point_from_pose(self.defense_goal_pose)
        defense_sign = self._defense_sign(defense_goal)
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        canonical_yaw = defense_sign * yaw
        ball_velocity = self._ball_velocity()
        target = predict_guard_target(
            ball=ball,
            velocity=ball_velocity,
            defense_goal=defense_goal,
            goal_width=self.goal_width,
            guard_depth=self.guard_depth,
            guard_margin=self.guard_margin,
            min_ball_speed=self.min_ball_speed,
        )
        opponent = self._select_opponent_striker()

        return np.array(
            [
                self._clip(self_point.x / self.half_width),
                self._clip(defense_sign * self_point.y / self.half_length),
                math.sin(canonical_yaw),
                math.cos(canonical_yaw),
                self._clip(ball.x / self.half_width),
                self._clip(defense_sign * ball.y / self.half_length),
                self._clip(ball_velocity.x / TRAINING_MAX_LINEAR_SPD, BALL_VEL_OBS_LIMIT),
                self._clip(
                    defense_sign * ball_velocity.y / TRAINING_MAX_LINEAR_SPD,
                    BALL_VEL_OBS_LIMIT,
                ),
                self._clip(target.x / self.half_width),
                self._clip(defense_sign * target.y / self.half_length),
                self._clip(opponent.x / self.half_width),
                self._clip(defense_sign * opponent.y / self.half_length),
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
        defense_sign = self._defense_sign(self._point_from_pose(self.defense_goal_pose))
        vx = float(values[0] * self.max_linear_speed)
        vy = float(defense_sign * values[1] * self.max_linear_speed)
        wz = float(defense_sign * values[2] * self.max_angular_speed)

        self.move.change_motion_id(self.motion_id)
        self.move.change_speed(vx, vy, wz)

    def _should_clear(self, self_point: Point2D) -> bool:
        ball = self._point_from_pose(self.ball_pose)
        if not self._ball_in_keeper_zone(ball):
            return False
        return math.hypot(ball.x - self_point.x, ball.y - self_point.y) <= self.clear_distance

    def _should_counter(self, self_point: Point2D, ball: Point2D) -> bool:
        if not self._ball_in_counter_zone(ball):
            return False
        distance = math.hypot(ball.x - self_point.x, ball.y - self_point.y)
        now = self.get_clock().now().nanoseconds * 1e-9
        if distance <= self.counter_distance:
            self._counter_until = max(self._counter_until, now + self.counter_hold)
            return True
        return now <= self._counter_until and distance <= self.counter_chase_distance

    def _clear_ball(self):
        ball = self._point_from_pose(self.ball_pose)
        self_point = self._point_from_pose(self.self_pose)
        away_sign = self._away_sign()
        defense_goal = self._point_from_pose(self.defense_goal_pose)
        clear_dx = clamp(defense_goal.x - ball.x, -self.clear_center_pull, self.clear_center_pull)
        clear_dy = away_sign * self.clear_forward_distance
        clear_len = max(math.hypot(clear_dx, clear_dy), 1e-6)
        clear_unit = Point2D(clear_dx / clear_len, clear_dy / clear_len)
        if math.hypot(ball.x - self_point.x, ball.y - self_point.y) > self.clear_contact_distance:
            target = Point2D(
                ball.x - clear_unit.x * self.clear_backoff_distance,
                ball.y - clear_unit.y * self.clear_backoff_distance,
            )
            self._drive_to_target(target, ball, self.clear_gain)
            return

        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        clear_body = field_to_body(clear_unit.x, clear_unit.y, yaw)
        speed = self.clear_gain * self.clear_forward_distance
        self.move.change_motion_id(self.motion_id)
        self.move.change_speed(speed * clear_body.x, speed * clear_body.y, 0.0)

    def _active_defense_to_ball(self, ball: Point2D):
        target = self._active_defense_target(ball)
        self._drive_to_target(target, ball, self.recover_gain)

    def _guard_with_policy(self):
        defense_goal = self._point_from_pose(self.defense_goal_pose)
        target = Point2D(defense_goal.x, defense_goal.y + self._away_sign() * self.guard_depth)
        self._drive_to_target(target, self._point_from_pose(self.ball_pose), self.recover_gain)

    def _active_defense_target(self, ball: Point2D) -> Point2D:
        away_sign = self._away_sign()
        defense_goal = self._point_from_pose(self.defense_goal_pose)
        half_width = self.keeper_zone_width * 0.5
        y_low = min(defense_goal.y + away_sign * self.guard_depth, defense_goal.y + away_sign * self.keeper_zone_depth)
        y_high = max(defense_goal.y + away_sign * self.guard_depth, defense_goal.y + away_sign * self.keeper_zone_depth)
        return Point2D(
            clamp(ball.x, defense_goal.x - half_width, defense_goal.x + half_width),
            clamp(ball.y - away_sign * 0.22, y_low, y_high),
        )

    def _recover_to_home(self, self_point: Point2D):
        defense_goal = self._point_from_pose(self.defense_goal_pose)
        target = Point2D(defense_goal.x, defense_goal.y + self._away_sign() * self.guard_depth)
        self._drive_to_target(target, self._point_from_pose(self.ball_pose), self.recover_gain)

    def _drive_to_target(self, target: Point2D, face_point: Point2D, gain: float):
        self_point = self._point_from_pose(self.self_pose)
        yaw = yaw_from_quaternion(self.self_pose.pose.orientation)
        vector = field_to_body(target.x - self_point.x, target.y - self_point.y, yaw)
        yaw_error = normalize_angle(yaw_to_point(self_point, face_point) - yaw)
        self.move.change_motion_id(self.motion_id)
        self.move.change_speed(
            gain * vector.x,
            gain * vector.y,
            self.recover_yaw_gain * yaw_error,
        )

    def _ball_velocity(self) -> Point2D:
        if self.ball_twist is None:
            return Point2D(0.0, 0.0)
        return Point2D(
            self.ball_twist.twist.linear.x,
            self.ball_twist.twist.linear.y,
        )

    def _select_opponent_striker(self) -> Point2D:
        pose = self._opponent_poses.get(self.opponent_striker_key)
        if pose is not None:
            return self._point_from_pose(pose)
        if not self._opponent_poses:
            return Point2D(0.0, 0.0)
        ball = self._point_from_pose(self.ball_pose)
        closest = min(
            self._opponent_poses.values(),
            key=lambda msg: math.hypot(msg.pose.position.x - ball.x, msg.pose.position.y - ball.y),
        )
        return self._point_from_pose(closest)

    def _outside_soft_field(self, point: Point2D) -> bool:
        x_limit = self.field_width * 0.5
        y_limit = self.field_length * 0.5
        if abs(point.x) > x_limit or abs(point.y) > y_limit:
            return True
        defense_goal = self._point_from_pose(self.defense_goal_pose)
        depth, width = self._activity_limits()
        return (
            abs(point.y - defense_goal.y) > depth
            or abs(point.x - defense_goal.x) > width * 0.5
        )

    def _inside_recover_release(self, point: Point2D) -> bool:
        defense_goal = self._point_from_pose(self.defense_goal_pose)
        depth, width = self._activity_limits()
        release_depth = depth if self._ball_in_opponent_half() else min(self.recover_release_depth, depth)
        return (
            abs(point.y - defense_goal.y) <= release_depth
            and abs(point.x - defense_goal.x) <= width * 0.5
        )

    def _activity_limits(self):
        if self._ball_in_opponent_half():
            return self.distant_ball_zone_depth, self.distant_ball_zone_width
        return self.max_goal_depth, self.keeper_zone_width

    def _ball_in_opponent_half(self) -> bool:
        if self.ball_pose is None or self.defense_goal_pose is None:
            return False
        ball = self._point_from_pose(self.ball_pose)
        return self._away_sign() * ball.y > 0.0

    def _ball_in_keeper_zone(self, ball: Point2D) -> bool:
        defense_goal = self._point_from_pose(self.defense_goal_pose)
        half_width = self.keeper_zone_width * 0.5
        goal_to_ball = abs(ball.y - defense_goal.y)
        inside_width = abs(ball.x - defense_goal.x) <= half_width
        return inside_width and goal_to_ball <= self.keeper_zone_depth

    def _ball_in_counter_zone(self, ball: Point2D) -> bool:
        defense_goal = self._point_from_pose(self.defense_goal_pose)
        half_width = self.counter_zone_width * 0.5
        goal_to_ball = abs(ball.y - defense_goal.y)
        inside_width = abs(ball.x - defense_goal.x) <= half_width
        return inside_width and goal_to_ball <= self.counter_zone_depth

    def _set_state(self, state: str):
        if state != self.state:
            self.get_logger().info(f"goalkeeper defense state: {state}")
            self.state = state

    def _publish_role_state(self):
        msg = String()
        msg.data = self.state
        self._role_state_pub.publish(msg)
        if self._intent_pub is None:
            return

        ball = self._point_from_pose(self.ball_pose) if self.ball_pose is not None else None
        self_point = self._point_from_pose(self.self_pose) if self.self_pose is not None else None
        has_ball = (
            ball is not None
            and self_point is not None
            and math.hypot(ball.x - self_point.x, ball.y - self_point.y) <= self.possession_distance
        )
        priority = (
            95
            if self.state == self.COUNTER
            else 90
            if self.state == self.CLEAR
            else 80
            if self.state == self.INTERCEPT
            else 60
        )
        attacking_role = self.state == self.COUNTER or (self.state == self.CLEAR and has_ball)
        payload = {
            "stamp": self.get_clock().now().nanoseconds * 1e-9,
            "team": self._team,
            "robot": self._robot_id,
            "role": "striker" if attacking_role else "goalkeeper",
            "state": self.state,
            "target": ({"x": ball.x, "y": ball.y} if ball else None),
            "has_ball": has_ball,
            "priority": priority,
            "controller_mode": self.controller_mode,
            "policy_model": str(self._model_path) if self._model_path is not None else None,
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
        raise FileNotFoundError(f"No goalkeeper policy checkpoint found. Checked:\n  {checked_text}")

    def _load_model(self, model_path: Path, device: str):
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self._install_numpy_core_aliases()
        try:
            from stable_baselines3 import PPO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "goalkeeper_policy requires stable-baselines3. Install RL deps first."
            ) from exc
        return PPO.load(
            str(model_path),
            device=device,
            custom_objects=self._model_custom_objects(),
        )

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
    def _defense_sign(defense_goal: Point2D) -> float:
        return 1.0 if defense_goal.y <= 0.0 else -1.0

    def _away_sign(self) -> float:
        return self._defense_sign(self._point_from_pose(self.defense_goal_pose))

    @staticmethod
    def _point_from_pose(msg: PoseStamped) -> Point2D:
        return Point2D(msg.pose.position.x, msg.pose.position.y)

    @staticmethod
    def _clip(value: float, limit: float = 1.0) -> float:
        return float(clamp(value, -limit, limit))
