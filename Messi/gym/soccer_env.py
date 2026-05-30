"""Gymnasium environment for striker training.

Phase 1 trains the striker on an empty field.
Phase 2 adds the scripted goalkeeper.
Phase 3 adds a scripted opponent striker and goalkeeper for adversarial play.
Phase 4 keeps Phase 3's 2v2 setup and adds runtime latency randomization.
"""

from __future__ import annotations

import math
import sys
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "SoccerEnv requires gymnasium. Install it with: "
        "uv sync"
    ) from exc

try:
    import yaml
except Exception:  # pragma: no cover - optional fallback
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lib.geometry import Point2D, clamp, field_to_body, normalize_angle, safe_unit, yaw_to_point


FIELD_CONFIG_PATH = REPO_ROOT / "sim" / "config" / "field.yaml"

DEFAULT_FIELD = {
    "length": 8.8,
    "width": 5.55,
    "goal_width": 1.0,
    "boundary_margin": 0.15,
}
DEFAULT_PHYSICS = {
    "ball_radius": 0.125,
    "robot_length": 0.43,
    "robot_body_width": 0.20,
    "robot_leg_width": 0.32,
    "robot_radius": 0.27,
    "ball_friction": 1.5,
    "wall_restitution": 0.6,
    "ball_restitution": 0.7,
    "robot_ball_restitution": 0.55,
    "robot_ball_tangent_gain": 0.12,
    "physics_hz": 50,
}

MAX_EPISODE_STEPS = 500
MAX_LINEAR_SPD = 1.2
MAX_ANGULAR_SPD = 3.0
BALL_VEL_OBS_LIMIT = 5.0
DEFAULT_POLICY_HZ_RANGE = (8.0, 12.0)
DEFAULT_ACTION_LATENCY_RANGE = (0.08, 0.18)
DEFAULT_OBSERVATION_LATENCY_RANGE = (0.0, 0.10)
BALL_WALL_PENALTY = 4.0
SELF_WALL_PENALTY = 0.8
TEAMMATE_COLLISION_PENALTY = 0.5
OPPONENT_COLLISION_PENALTY = 0.5
APPROACH_TARGET_DISTANCE = 0.48
APPROACH_TARGET_REWARD = 0.8
PHASE1_SHOOTING_RESET_PROB = 0.75
SHOOTING_RESET_BALL_X = (-0.38, 0.38)
SHOOTING_RESET_BALL_Y = (-1.6, 2.4)
SHOOTING_RESET_DEPTH = (0.55, 0.85)
SHOOTING_RESET_LATERAL = (-0.14, 0.14)
GOAL_LANE_CLEARANCE = 0.05
GOAL_LANE_REWARD = 1.2
OFF_TARGET_PROGRESS_PENALTY = 2.0
WALL_PROXIMITY_MARGIN = 0.45
WALL_PROXIMITY_PENALTY = 0.15
WRONG_SIDE_RESET_PROB = 0.35
WRONG_SIDE_RESET_FORWARD_OFFSET = (0.55, 1.10)
WRONG_SIDE_RESET_LATERAL_OFFSET = (-0.45, 0.45)
WRONG_SIDE_CLEARANCE = 0.14
WRONG_SIDE_NEAR_DISTANCE = 1.25
WRONG_SIDE_LATERAL_WIDTH = 0.65
WRONG_SIDE_PROGRESS_REWARD = 0.6
WRONG_SIDE_STEP_PENALTY = 0.03
WRONG_SIDE_BACKDRIVE_PENALTY = 1.2
KEEPER_GOAL_WIDTH = 1.0
KEEPER_GUARD_DEPTH = 0.45
KEEPER_GUARD_MARGIN = 0.12
KEEPER_INTERCEPT_DISTANCE = 0.65
KEEPER_POSITION_GAIN = 0.9
KEEPER_YAW_GAIN = 1.2
KEEPER_GOAL_TOLERANCE = 0.18
KEEPER_MIN_BALL_SPEED = 0.08

DR_RANGES = {
    "ball_friction": (1.1, 1.9),
    "wall_restitution": (0.45, 0.75),
    "ball_restitution": (0.55, 0.85),
    "robot_ball_restitution": (0.45, 0.70),
    "robot_ball_tangent_gain": (0.05, 0.18),
    "ball_radius": (0.115, 0.135),
    "robot_length": (0.40, 0.46),
    "robot_body_width": (0.18, 0.22),
    "robot_leg_width": (0.29, 0.35),
    "robot_radius": (0.26, 0.29),
    "goalkeeper_position_gain": (0.75, 1.05),
    "goalkeeper_yaw_gain": (1.0, 1.5),
    "goalkeeper_guard_depth": (0.35, 0.55),
    "goalkeeper_guard_margin": (0.08, 0.16),
    "opponent_goalkeeper_position_gain": (0.75, 1.05),
    "opponent_goalkeeper_yaw_gain": (1.0, 1.5),
    "opponent_goalkeeper_guard_depth": (0.35, 0.55),
    "opponent_goalkeeper_guard_margin": (0.08, 0.16),
    "opponent_striker_speed": (0.55, 0.9),
    "opponent_striker_approach_gain": (0.7, 1.1),
    "opponent_striker_lateral_gain": (0.7, 1.1),
    "opponent_striker_yaw_gain": (0.9, 1.5),
    "opponent_possession_distance": (0.35, 0.55),
}


def _robot_circumradius(length: float, width: float) -> float:
    return math.hypot(float(length) * 0.5, float(width) * 0.5)


def _load_field_config() -> Tuple[dict, dict]:
    if yaml is None or not FIELD_CONFIG_PATH.exists():
        return dict(DEFAULT_FIELD), dict(DEFAULT_PHYSICS)
    with FIELD_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    field = dict(DEFAULT_FIELD)
    field.update({k: float(v) for k, v in (data.get("field", {}) or {}).items() if k in field})
    physics = dict(DEFAULT_PHYSICS)
    physics.update(
        {k: float(v) for k, v in (data.get("physics", {}) or {}).items() if k in physics}
    )
    return field, physics


FIELD, PHYSICS = _load_field_config()
FIELD_LENGTH = float(FIELD["length"])
FIELD_WIDTH = float(FIELD["width"])
GOAL_WIDTH = float(FIELD["goal_width"])
GOAL_A_Y = -FIELD_LENGTH * 0.5
GOAL_B_Y = FIELD_LENGTH * 0.5
BALL_RADIUS = float(PHYSICS["ball_radius"])
ROBOT_LENGTH = float(PHYSICS["robot_length"])
ROBOT_BODY_WIDTH = float(PHYSICS["robot_body_width"])
ROBOT_LEG_WIDTH = float(PHYSICS["robot_leg_width"])
ROBOT_RADIUS = max(
    float(PHYSICS["robot_radius"]),
    _robot_circumradius(ROBOT_LENGTH, ROBOT_LEG_WIDTH),
)
BALL_FRICTION = float(PHYSICS["ball_friction"])
WALL_RESTITUTION = float(PHYSICS["wall_restitution"])
BALL_RESTITUTION = float(PHYSICS["ball_restitution"])
PHYSICS_DT = 1.0 / float(PHYSICS["physics_hz"])


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    cmd_vx: float = 0.0
    cmd_vy: float = 0.0
    cmd_wz: float = 0.0
    vx_world: float = 0.0
    vy_world: float = 0.0


@dataclass
class BallState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


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

    inward_sign = -1.0 if defense_goal.y > 0.0 else 1.0
    guard_y = defense_goal.y + inward_sign * float(guard_depth)
    target_x = ball.x if velocity is None else defense_goal.x

    if velocity is not None:
        speed = math.hypot(velocity.x, velocity.y)
        moving_toward_goal = (defense_goal.y - ball.y) * velocity.y > 0.0
        if speed < float(min_ball_speed):
            target_x = ball.x
        elif moving_toward_goal and abs(velocity.y) > 1e-4:
            time_to_goal_line = (defense_goal.y - ball.y) / velocity.y
            if time_to_goal_line >= 0.0:
                target_x = ball.x + velocity.x * time_to_goal_line

    return Point2D(clamp(target_x, x_min, x_max), guard_y)


class SoccerEnv(gym.Env):
    """Headless striker environment with staged scripted opponents."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": int(round(1.0 / PHYSICS_DT))}
    TERMINAL_EVENTS = {
        "GOAL",
        "OWN_GOAL",
        "OUT_OF_BOUNDS",
        "BALL_WALL",
        "SELF_WALL",
        "TEAMMATE_COLLISION",
    }

    def __init__(
        self,
        phase: int = 1,
        render_mode: Optional[str] = None,
        domain_randomization: bool = True,
        wrong_side_reset_prob: float = WRONG_SIDE_RESET_PROB,
        phase1_shooting_reset_prob: Optional[float] = None,
        latency_randomization: Optional[bool] = None,
        policy_hz_range: tuple[float, float] = DEFAULT_POLICY_HZ_RANGE,
        action_latency_range: tuple[float, float] = DEFAULT_ACTION_LATENCY_RANGE,
        observation_latency_range: tuple[float, float] = DEFAULT_OBSERVATION_LATENCY_RANGE,
    ):
        super().__init__()
        if phase not in (1, 2, 3, 4):
            raise ValueError("phase must be 1, 2, 3 or 4")
        self.phase = int(phase)
        self.goalkeeper_active = self.phase >= 2
        self.teammate_goalkeeper_active = self.phase >= 3
        self.opponent_striker_active = self.phase >= 3
        self.domain_randomization = bool(domain_randomization)
        self.wrong_side_reset_prob = float(clamp(wrong_side_reset_prob, 0.0, 1.0))
        if phase1_shooting_reset_prob is None:
            phase1_shooting_reset_prob = PHASE1_SHOOTING_RESET_PROB if self.phase == 1 else 0.0
        self.phase1_shooting_reset_prob = float(clamp(phase1_shooting_reset_prob, 0.0, 1.0))
        self.latency_randomization = (
            self.phase >= 4
            if latency_randomization is None
            else bool(latency_randomization)
        )
        self.policy_hz_range = self._ordered_range(policy_hz_range, min_value=1.0)
        self.action_latency_range = self._ordered_range(action_latency_range, min_value=0.0)
        self.observation_latency_range = self._ordered_range(
            observation_latency_range,
            min_value=0.0,
        )
        self.render_mode = render_mode

        self.half_width = FIELD_WIDTH * 0.5
        self.half_length = FIELD_LENGTH * 0.5
        self.goal_a = Point2D(0.0, GOAL_A_Y)
        self.goal_b = Point2D(0.0, GOAL_B_Y)

        self.max_episode_steps = MAX_EPISODE_STEPS
        self.step_count = 0
        self.last_event = None
        self._step_events: list[str] = []
        self._opponent_obs_role = "goalkeeper"
        self._reset_scenario = "random"
        self.policy_interval_steps = 1
        self.policy_dt = PHYSICS_DT
        self.action_latency_seconds = 0.0
        self.observation_latency_seconds = 0.0
        self.action_latency_steps = 0
        self.observation_latency_steps = 0
        self._held_action = np.zeros(3, dtype=np.float32)
        self._action_delay_buffer: list[np.ndarray] = []
        self._observation_buffer: list[np.ndarray] = []

        self._set_default_domain()
        self.striker = RobotState(x=0.0, y=-4.0, yaw=math.pi * 0.5)
        self.goalkeeper = RobotState(x=0.0, y=self.goal_b.y - 0.5, yaw=-math.pi * 0.5)
        self.teammate_goalkeeper = RobotState(x=0.0, y=self.goal_a.y + 0.5, yaw=math.pi * 0.5)
        self.opponent_striker = RobotState(x=0.0, y=2.5, yaw=-math.pi * 0.5)
        self.ball = BallState()

        obs_low = np.array(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -BALL_VEL_OBS_LIMIT, -BALL_VEL_OBS_LIMIT, -1.0, -1.0, -1.0, -1.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, BALL_VEL_OBS_LIMIT, BALL_VEL_OBS_LIMIT, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.last_event = None
        self._step_events = []
        self._set_default_domain()
        if self.domain_randomization:
            self._randomize_domain()
        self._sample_latency_domain()
        self._opponent_obs_role = self._sample_opponent_obs_role()

        robots: list[RobotState] = []
        paired_ball = None
        if self._use_wrong_side_reset():
            self.striker, paired_ball = self._sample_wrong_side_pair(existing=robots)
            self._reset_scenario = "wrong_side"
        elif self._use_phase1_shooting_reset():
            self.striker, paired_ball = self._sample_phase1_shooting_pair(existing=robots)
            self._reset_scenario = "shooting"
        else:
            self.striker = self._sample_striker(existing=robots)
            self._reset_scenario = "random"
        robots.append(self.striker)

        if self.phase >= 2:
            self.goalkeeper = self._sample_goalkeeper(
                defense_goal=self.goal_b,
                guard_depth=self.opponent_goalkeeper_guard_depth,
                yaw=-math.pi * 0.5,
                existing=robots,
            )
            robots.append(self.goalkeeper)
        else:
            self.goalkeeper = RobotState()

        if self.phase >= 3:
            self.teammate_goalkeeper = self._sample_goalkeeper(
                defense_goal=self.goal_a,
                guard_depth=self.goalkeeper_guard_depth,
                yaw=math.pi * 0.5,
                existing=robots,
            )
            robots.append(self.teammate_goalkeeper)
            self.opponent_striker = self._sample_opponent_striker(existing=robots)
            robots.append(self.opponent_striker)
        else:
            self.teammate_goalkeeper = RobotState()
            self.opponent_striker = RobotState()

        if paired_ball is not None and self._ball_clear(paired_ball, robots):
            self.ball = paired_ball
        else:
            self.ball = self._sample_ball(robots)
            if paired_ball is not None:
                self._reset_scenario = "random"
        self._zero_commands()
        self._reset_latency_buffers()
        obs = self._delayed_observation()
        info = self._info(event=None, reward=0.0, terminated=False, truncated=False)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 3:
            raise ValueError("action must have shape (3,)")
        action = np.clip(action, -1.0, 1.0)

        applied_action = self._delayed_action(action)
        policy_events: list[str] = []
        event = None
        reward = 0.0
        physics_steps = 0

        for _ in range(self.policy_interval_steps):
            self._step_events = []
            prev_striker = copy(self.striker)
            prev_ball = copy(self.ball)

            self._apply_striker_action(applied_action)
            self._update_scripted_robots()

            self._step_robots()
            ball_event = self._step_ball()
            self._resolve_robot_robot()
            self._clamp_robots()

            tick_event = self._select_event(ball_event)
            reward += self._compute_reward(prev_striker, prev_ball, tick_event, self._step_events)
            for step_event in self._step_events:
                if step_event not in policy_events:
                    policy_events.append(step_event)
            physics_steps += 1
            if tick_event is not None:
                event = tick_event
            if tick_event in self.TERMINAL_EVENTS:
                break

        self._step_events = policy_events
        self.step_count += 1
        terminated = event in self.TERMINAL_EVENTS
        truncated = not terminated and self.step_count >= self.max_episode_steps
        self._record_observation(self._observe())
        obs = self._delayed_observation()

        self.last_event = event
        info = self._info(event=event, reward=reward, terminated=terminated, truncated=truncated)
        info["physics_steps"] = physics_steps
        return obs, reward, terminated, truncated, info

    def render(self):
        text = self._render_text()
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self):
        return None

    @staticmethod
    def _ordered_range(values: tuple[float, float], min_value: float) -> tuple[float, float]:
        if len(values) != 2:
            raise ValueError("range values must contain exactly two numbers")
        lower = max(float(values[0]), min_value)
        upper = max(float(values[1]), min_value)
        if lower > upper:
            lower, upper = upper, lower
        return lower, upper

    def _sample_range(self, values: tuple[float, float]) -> float:
        lower, upper = values
        if abs(upper - lower) <= 1e-9:
            return float(lower)
        return float(self.np_random.uniform(lower, upper))

    def _seconds_to_policy_steps(self, seconds: float) -> int:
        if seconds <= 0.0 or self.policy_dt <= 0.0:
            return 0
        return max(0, int(math.floor(seconds / self.policy_dt + 0.5)))

    def _sample_latency_domain(self) -> None:
        if not self.latency_randomization:
            self.policy_interval_steps = 1
            self.policy_dt = PHYSICS_DT
            self.action_latency_seconds = 0.0
            self.observation_latency_seconds = 0.0
            self.action_latency_steps = 0
            self.observation_latency_steps = 0
            self.max_episode_steps = MAX_EPISODE_STEPS
            return

        physics_hz = 1.0 / PHYSICS_DT
        policy_hz = self._sample_range(self.policy_hz_range)
        self.policy_interval_steps = max(1, int(round(physics_hz / max(policy_hz, 1e-6))))
        self.policy_dt = self.policy_interval_steps * PHYSICS_DT
        self.action_latency_seconds = self._sample_range(self.action_latency_range)
        self.observation_latency_seconds = self._sample_range(self.observation_latency_range)
        self.action_latency_steps = self._seconds_to_policy_steps(self.action_latency_seconds)
        self.observation_latency_steps = self._seconds_to_policy_steps(
            self.observation_latency_seconds
        )
        self.max_episode_steps = max(1, int(round(MAX_EPISODE_STEPS / self.policy_interval_steps)))

    def _reset_latency_buffers(self) -> None:
        zero_action = np.zeros(3, dtype=np.float32)
        self._held_action = zero_action.copy()
        self._action_delay_buffer = [
            zero_action.copy() for _ in range(self.action_latency_steps)
        ]
        obs = self._observe()
        self._observation_buffer = [
            obs.copy() for _ in range(self.observation_latency_steps + 1)
        ]

    def _delayed_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        if self.action_latency_steps <= 0:
            self._held_action = action
            return self._held_action

        self._action_delay_buffer.append(action)
        self._held_action = self._action_delay_buffer.pop(0)
        return self._held_action

    def _record_observation(self, obs: np.ndarray) -> None:
        self._observation_buffer.append(np.asarray(obs, dtype=np.float32).copy())
        max_len = self.observation_latency_steps + 1
        if len(self._observation_buffer) > max_len:
            self._observation_buffer = self._observation_buffer[-max_len:]

    def _delayed_observation(self) -> np.ndarray:
        if not self._observation_buffer:
            self._reset_latency_buffers()
        return self._observation_buffer[0].copy()

    def _set_default_domain(self) -> None:
        self.ball_radius = BALL_RADIUS
        self.robot_length = ROBOT_LENGTH
        self.robot_body_width = ROBOT_BODY_WIDTH
        self.robot_leg_width = ROBOT_LEG_WIDTH
        self.robot_radius = ROBOT_RADIUS
        self.ball_friction = BALL_FRICTION
        self.wall_restitution = WALL_RESTITUTION
        self.ball_restitution = BALL_RESTITUTION
        self.robot_ball_restitution = float(PHYSICS.get("robot_ball_restitution", 0.55))
        self.robot_ball_tangent_gain = float(PHYSICS.get("robot_ball_tangent_gain", 0.12))

        self.goalkeeper_position_gain = KEEPER_POSITION_GAIN
        self.goalkeeper_yaw_gain = KEEPER_YAW_GAIN
        self.goalkeeper_guard_depth = KEEPER_GUARD_DEPTH
        self.goalkeeper_guard_margin = KEEPER_GUARD_MARGIN
        self.opponent_goalkeeper_position_gain = KEEPER_POSITION_GAIN
        self.opponent_goalkeeper_yaw_gain = KEEPER_YAW_GAIN
        self.opponent_goalkeeper_guard_depth = KEEPER_GUARD_DEPTH
        self.opponent_goalkeeper_guard_margin = KEEPER_GUARD_MARGIN

        self.opponent_striker_speed = 0.75
        self.opponent_striker_approach_gain = 0.9
        self.opponent_striker_lateral_gain = 0.9
        self.opponent_striker_yaw_gain = 1.2
        self.opponent_possession_distance = 0.45

    def _randomize_domain(self) -> None:
        for name, bounds in DR_RANGES.items():
            setattr(self, name, float(self.np_random.uniform(bounds[0], bounds[1])))
        self._sync_robot_radius_to_footprint()

    def _sync_robot_radius_to_footprint(self) -> None:
        self.robot_radius = max(
            float(self.robot_radius),
            _robot_circumradius(self.robot_length, self.robot_leg_width),
        )

    def _sample_opponent_obs_role(self) -> str:
        if self.phase < 3:
            return "goalkeeper"
        return "striker" if float(self.np_random.random()) < 0.5 else "goalkeeper"

    def _use_wrong_side_reset(self) -> bool:
        return float(self.np_random.random()) < self.wrong_side_reset_prob

    def _use_phase1_shooting_reset(self) -> bool:
        if self.phase != 1:
            return False
        return float(self.np_random.random()) < self.phase1_shooting_reset_prob

    def _sample_striker(self, existing: list[RobotState]) -> RobotState:
        return self._sample_robot(
            x_range=(-2.2, 2.2),
            y_range=(-4.3, 0.5),
            yaw_range=(-math.pi, math.pi),
            existing=existing,
        )

    def _sample_opponent_striker(self, existing: list[RobotState]) -> RobotState:
        return self._sample_robot(
            x_range=(-2.2, 2.2),
            y_range=(-0.5, 4.3),
            yaw_range=(-math.pi, math.pi),
            existing=existing,
        )

    def _sample_wrong_side_pair(
        self,
        existing: list[RobotState],
    ) -> tuple[RobotState, Optional[BallState]]:
        ball_x_min = max(-2.0, -self.half_width + self.ball_radius + 0.2)
        ball_x_max = min(2.0, self.half_width - self.ball_radius - 0.2)
        ball_y_min = max(-2.6, -self.half_length + self.ball_radius + 0.8)
        ball_y_max = min(1.2, self.half_length - self.ball_radius - 1.4)

        for _ in range(128):
            ball = BallState(
                x=float(self.np_random.uniform(ball_x_min, ball_x_max)),
                y=float(self.np_random.uniform(ball_y_min, ball_y_max)),
                vx=float(self.np_random.uniform(-0.04, 0.04)),
                vy=float(self.np_random.uniform(-0.04, 0.04)),
            )
            attack_dir = self._attack_direction(ball)
            side_dir = Point2D(-attack_dir.y, attack_dir.x)
            forward_offset = float(
                self.np_random.uniform(
                    WRONG_SIDE_RESET_FORWARD_OFFSET[0],
                    WRONG_SIDE_RESET_FORWARD_OFFSET[1],
                )
            )
            lateral_offset = float(
                self.np_random.uniform(
                    WRONG_SIDE_RESET_LATERAL_OFFSET[0],
                    WRONG_SIDE_RESET_LATERAL_OFFSET[1],
                )
            )
            robot = RobotState(
                x=ball.x + attack_dir.x * forward_offset + side_dir.x * lateral_offset,
                y=ball.y + attack_dir.y * forward_offset + side_dir.y * lateral_offset,
                yaw=math.atan2(attack_dir.y, attack_dir.x)
                + float(self.np_random.uniform(-0.45, 0.45)),
            )
            robot.yaw = math.atan2(math.sin(robot.yaw), math.cos(robot.yaw))
            if not self._robot_inside_field(robot):
                continue
            if not self._robot_clear(robot, existing):
                continue
            if self._ball_clear(ball, existing + [robot]):
                return robot, ball

        return self._sample_striker(existing=existing), None

    def _sample_phase1_shooting_pair(
        self,
        existing: list[RobotState],
    ) -> tuple[RobotState, Optional[BallState]]:
        ball_x_min = max(SHOOTING_RESET_BALL_X[0], -GOAL_WIDTH * 0.5 + self.ball_radius)
        ball_x_max = min(SHOOTING_RESET_BALL_X[1], GOAL_WIDTH * 0.5 - self.ball_radius)
        ball_y_min = max(SHOOTING_RESET_BALL_Y[0], -self.half_length + self.ball_radius + 0.7)
        ball_y_max = min(SHOOTING_RESET_BALL_Y[1], self.half_length - self.ball_radius - 1.0)

        for _ in range(128):
            ball = BallState(
                x=float(self.np_random.uniform(ball_x_min, ball_x_max)),
                y=float(self.np_random.uniform(ball_y_min, ball_y_max)),
                vx=float(self.np_random.uniform(-0.03, 0.03)),
                vy=float(self.np_random.uniform(-0.03, 0.03)),
            )
            attack_dir = self._attack_direction(ball)
            side_dir = Point2D(-attack_dir.y, attack_dir.x)
            depth = float(self.np_random.uniform(SHOOTING_RESET_DEPTH[0], SHOOTING_RESET_DEPTH[1]))
            lateral = float(self.np_random.uniform(SHOOTING_RESET_LATERAL[0], SHOOTING_RESET_LATERAL[1]))
            robot = RobotState(
                x=ball.x - attack_dir.x * depth + side_dir.x * lateral,
                y=ball.y - attack_dir.y * depth + side_dir.y * lateral,
                yaw=math.atan2(attack_dir.y, attack_dir.x)
                + float(self.np_random.uniform(-0.20, 0.20)),
            )
            robot.yaw = math.atan2(math.sin(robot.yaw), math.cos(robot.yaw))
            if not self._robot_inside_field(robot):
                continue
            if not self._robot_clear(robot, existing):
                continue
            if self._ball_clear(ball, existing + [robot]):
                return robot, ball

        return self._sample_striker(existing=existing), None

    def _sample_goalkeeper(
        self,
        defense_goal: Point2D,
        guard_depth: float,
        yaw: float,
        existing: list[RobotState],
    ) -> RobotState:
        inward_sign = -1.0 if defense_goal.y > 0.0 else 1.0
        guard_y = defense_goal.y + inward_sign * guard_depth
        return self._sample_robot(
            x_range=(-GOAL_WIDTH * 0.5 + 0.1, GOAL_WIDTH * 0.5 - 0.1),
            y_range=(guard_y - 0.18, guard_y + 0.18),
            yaw_range=(yaw - 0.15, yaw + 0.15),
            existing=existing,
        )

    def _sample_robot(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        yaw_range: tuple[float, float],
        existing: list[RobotState],
    ) -> RobotState:
        x_min = max(x_range[0], -self.half_width + self.robot_radius + 0.05)
        x_max = min(x_range[1], self.half_width - self.robot_radius - 0.05)
        y_min = max(y_range[0], -self.half_length + self.robot_radius + 0.05)
        y_max = min(y_range[1], self.half_length - self.robot_radius - 0.05)
        fallback = RobotState(
            x=(x_min + x_max) * 0.5,
            y=(y_min + y_max) * 0.5,
            yaw=(yaw_range[0] + yaw_range[1]) * 0.5,
        )
        for _ in range(128):
            robot = RobotState(
                x=float(self.np_random.uniform(x_min, x_max)),
                y=float(self.np_random.uniform(y_min, y_max)),
                yaw=float(self.np_random.uniform(yaw_range[0], yaw_range[1])),
            )
            if self._robot_clear(robot, existing):
                return robot
        return fallback

    def _sample_ball(self, robots: list[RobotState]) -> BallState:
        x_min = -self.half_width + self.ball_radius + 0.1
        x_max = self.half_width - self.ball_radius - 0.1
        y_min = -self.half_length + self.ball_radius + 0.6
        y_max = self.half_length - self.ball_radius - 0.6
        for _ in range(128):
            ball = BallState(
                x=float(self.np_random.uniform(max(-2.2, x_min), min(2.2, x_max))),
                y=float(self.np_random.uniform(max(-2.4, y_min), min(2.4, y_max))),
                vx=float(self.np_random.uniform(-0.08, 0.08)),
                vy=float(self.np_random.uniform(-0.08, 0.08)),
            )
            if self._ball_clear(ball, robots):
                return ball
        return BallState(x=0.0, y=0.0)

    def _robot_clear(self, candidate: RobotState, existing: list[RobotState]) -> bool:
        min_dist = 2.0 * self.robot_radius + 0.12
        return all(
            math.hypot(candidate.x - robot.x, candidate.y - robot.y) >= min_dist
            for robot in existing
        )

    def _ball_clear(self, ball: BallState, robots: list[RobotState]) -> bool:
        min_dist = self.robot_radius + self.ball_radius + 0.12
        return all(math.hypot(ball.x - robot.x, ball.y - robot.y) >= min_dist for robot in robots)

    def _robot_inside_field(self, robot: RobotState) -> bool:
        return (
            -self.half_width + self.robot_radius + 0.05
            <= robot.x
            <= self.half_width - self.robot_radius - 0.05
            and -self.half_length + self.robot_radius + 0.05
            <= robot.y
            <= self.half_length - self.robot_radius - 0.05
        )

    def _zero_commands(self) -> None:
        for robot in (self.striker, self.goalkeeper, self.teammate_goalkeeper, self.opponent_striker):
            robot.cmd_vx = 0.0
            robot.cmd_vy = 0.0
            robot.cmd_wz = 0.0
            robot.vx_world = 0.0
            robot.vy_world = 0.0

    def _apply_striker_action(self, action: np.ndarray) -> None:
        self.striker.cmd_vx = float(action[0] * MAX_LINEAR_SPD)
        self.striker.cmd_vy = float(action[1] * MAX_LINEAR_SPD)
        self.striker.cmd_wz = float(action[2] * MAX_ANGULAR_SPD)

    def _update_scripted_robots(self) -> None:
        self._idle_inactive_robots()
        if self.phase >= 2:
            self._update_goalkeeper(
                keeper=self.goalkeeper,
                defense_goal=self.goal_b,
                position_gain=self.opponent_goalkeeper_position_gain,
                yaw_gain=self.opponent_goalkeeper_yaw_gain,
                guard_depth=self.opponent_goalkeeper_guard_depth,
                guard_margin=self.opponent_goalkeeper_guard_margin,
            )
        if self.phase >= 3:
            self._update_goalkeeper(
                keeper=self.teammate_goalkeeper,
                defense_goal=self.goal_a,
                position_gain=self.goalkeeper_position_gain,
                yaw_gain=self.goalkeeper_yaw_gain,
                guard_depth=self.goalkeeper_guard_depth,
                guard_margin=self.goalkeeper_guard_margin,
            )
            self._update_opponent_striker()

    def _idle_inactive_robots(self) -> None:
        inactive = []
        if self.phase < 2:
            inactive.append(self.goalkeeper)
        if self.phase < 3:
            inactive.extend([self.teammate_goalkeeper, self.opponent_striker])
        for robot in inactive:
            robot.cmd_vx = 0.0
            robot.cmd_vy = 0.0
            robot.cmd_wz = 0.0

    def _update_goalkeeper(
        self,
        keeper: RobotState,
        defense_goal: Point2D,
        position_gain: float,
        yaw_gain: float,
        guard_depth: float,
        guard_margin: float,
    ) -> None:
        ball_point = Point2D(self.ball.x, self.ball.y)
        velocity = Point2D(self.ball.vx, self.ball.vy)
        target = predict_guard_target(
            ball=ball_point,
            velocity=velocity,
            defense_goal=defense_goal,
            goal_width=GOAL_WIDTH,
            guard_depth=guard_depth,
            guard_margin=guard_margin,
            min_ball_speed=KEEPER_MIN_BALL_SPEED,
        )
        close_to_goal_line = abs(ball_point.y - defense_goal.y) <= KEEPER_INTERCEPT_DISTANCE
        inside_goal_mouth = abs(ball_point.x - defense_goal.x) <= GOAL_WIDTH * 0.5
        if close_to_goal_line and inside_goal_mouth:
            target = Point2D(
                self._clamp_keeper_x(ball_point.x, defense_goal, guard_margin),
                ball_point.y,
            )

        dx = target.x - keeper.x
        dy = target.y - keeper.y
        yaw_error = normalize_angle(yaw_to_point(Point2D(keeper.x, keeper.y), ball_point) - keeper.yaw)
        vector = field_to_body(dx, dy, keeper.yaw)
        distance = math.hypot(dx, dy)

        if distance <= KEEPER_GOAL_TOLERANCE:
            cmd_vx = 0.0
            cmd_vy = 0.0
        else:
            cmd_vx = position_gain * vector.x
            cmd_vy = position_gain * vector.y
        keeper.cmd_vx = float(clamp(cmd_vx, -MAX_LINEAR_SPD, MAX_LINEAR_SPD))
        keeper.cmd_vy = float(clamp(cmd_vy, -MAX_LINEAR_SPD, MAX_LINEAR_SPD))
        keeper.cmd_wz = float(clamp(yaw_gain * yaw_error, -MAX_ANGULAR_SPD, MAX_ANGULAR_SPD))

    def _update_opponent_striker(self) -> None:
        robot = self.opponent_striker
        ball = Point2D(self.ball.x, self.ball.y)
        robot_point = Point2D(robot.x, robot.y)
        ball_distance = math.hypot(ball.x - robot.x, ball.y - robot.y)
        if ball_distance <= self.opponent_possession_distance:
            target = self.goal_a
            face_target = self.goal_a
        else:
            target = ball
            face_target = ball

        vector = field_to_body(target.x - robot.x, target.y - robot.y, robot.yaw)
        yaw_error = normalize_angle(yaw_to_point(robot_point, face_target) - robot.yaw)
        robot.cmd_vx = float(
            clamp(
                self.opponent_striker_approach_gain * vector.x,
                -self.opponent_striker_speed,
                self.opponent_striker_speed,
            )
        )
        robot.cmd_vy = float(
            clamp(
                self.opponent_striker_lateral_gain * vector.y,
                -self.opponent_striker_speed,
                self.opponent_striker_speed,
            )
        )
        robot.cmd_wz = float(
            clamp(
                self.opponent_striker_yaw_gain * yaw_error,
                -MAX_ANGULAR_SPD,
                MAX_ANGULAR_SPD,
            )
        )

    def _step_robots(self) -> None:
        for robot in self._active_robots():
            c = math.cos(robot.yaw)
            s = math.sin(robot.yaw)
            robot.vx_world = robot.cmd_vx * c - robot.cmd_vy * s
            robot.vy_world = robot.cmd_vx * s + robot.cmd_vy * c
            robot.x += robot.vx_world * PHYSICS_DT
            robot.y += robot.vy_world * PHYSICS_DT
            robot.yaw += robot.cmd_wz * PHYSICS_DT
            robot.yaw = math.atan2(math.sin(robot.yaw), math.cos(robot.yaw))

    def _step_ball(self) -> Optional[str]:
        ball = self.ball
        speed = math.hypot(ball.vx, ball.vy)
        if speed > 1e-4:
            new_speed = max(0.0, speed - self.ball_friction * PHYSICS_DT)
            scale = new_speed / speed
            ball.vx *= scale
            ball.vy *= scale
        else:
            ball.vx = 0.0
            ball.vy = 0.0

        for robot in self._active_robots():
            self._robot_ball_impulse(robot, ball)

        ball.x += ball.vx * PHYSICS_DT
        ball.y += ball.vy * PHYSICS_DT

        in_goal_mouth = abs(ball.x) <= GOAL_WIDTH * 0.5 + self.ball_radius
        if in_goal_mouth:
            if ball.y <= GOAL_A_Y:
                return "OWN_GOAL"
            if ball.y >= GOAL_B_Y:
                return "GOAL"

        half_w = FIELD_WIDTH * 0.5
        half_l = FIELD_LENGTH * 0.5
        touched_wall = False
        if ball.x - self.ball_radius < -half_w:
            ball.x = -half_w + self.ball_radius
            ball.vx = abs(ball.vx) * self.wall_restitution
            touched_wall = True
        elif ball.x + self.ball_radius > half_w:
            ball.x = half_w - self.ball_radius
            ball.vx = -abs(ball.vx) * self.wall_restitution
            touched_wall = True

        if not in_goal_mouth:
            if ball.y - self.ball_radius < -half_l:
                ball.y = -half_l + self.ball_radius
                ball.vy = abs(ball.vy) * self.wall_restitution
                touched_wall = True
            elif ball.y + self.ball_radius > half_l:
                ball.y = half_l - self.ball_radius
                ball.vy = -abs(ball.vy) * self.wall_restitution
                touched_wall = True

        if touched_wall:
            self._add_event("BALL_WALL")
            return "BALL_WALL"
        if abs(ball.x) > half_w + self.ball_radius or abs(ball.y) > half_l + self.ball_radius:
            return "OUT_OF_BOUNDS"
        return None

    def _robot_ball_impulse(self, robot: RobotState, ball: BallState) -> None:
        dx = ball.x - robot.x
        dy = ball.y - robot.y
        c = math.cos(robot.yaw)
        s = math.sin(robot.yaw)
        local_x = dx * c + dy * s
        local_y = -dx * s + dy * c

        half_l = max(self.robot_length * 0.5, 1e-6)
        half_w = max(self.robot_leg_width * 0.5, 1e-6)
        closest_x = clamp(local_x, -half_l, half_l)
        closest_y = clamp(local_y, -half_w, half_w)
        sep_x = local_x - closest_x
        sep_y = local_y - closest_y
        sep_dist = math.hypot(sep_x, sep_y)

        if sep_dist >= self.ball_radius:
            return

        if sep_dist > 1e-6:
            local_nx = sep_x / sep_dist
            local_ny = sep_y / sep_dist
            overlap = self.ball_radius - sep_dist
        else:
            # If the ball center is already inside the footprint, push through
            # the nearest face to avoid sticky low-speed dribble contacts.
            exit_x = half_l - abs(local_x)
            exit_y = half_w - abs(local_y)
            if exit_x <= exit_y:
                local_nx = 1.0 if local_x >= 0.0 else -1.0
                local_ny = 0.0
                overlap = self.ball_radius + exit_x
            else:
                local_nx = 0.0
                local_ny = 1.0 if local_y >= 0.0 else -1.0
                overlap = self.ball_radius + exit_y

        nx = local_nx * c - local_ny * s
        ny = local_nx * s + local_ny * c

        ball.x += nx * overlap
        ball.y += ny * overlap
        rel_vn = (ball.vx - robot.vx_world) * nx + (ball.vy - robot.vy_world) * ny
        if rel_vn < 0.0:
            impulse = -(1.0 + self.robot_ball_restitution) * rel_vn
            ball.vx += impulse * nx
            ball.vy += impulse * ny
        if self.robot_ball_tangent_gain > 0.0:
            tx, ty = -ny, nx
            rel_vt = (robot.vx_world - ball.vx) * tx + (robot.vy_world - ball.vy) * ty
            ball.vx += self.robot_ball_tangent_gain * rel_vt * tx
            ball.vy += self.robot_ball_tangent_gain * rel_vt * ty

    def _resolve_robot_robot(self) -> None:
        robots = list(self._active_robots())
        if len(robots) < 2:
            return
        min_dist = 2.0 * self.robot_radius
        for i, first in enumerate(robots[:-1]):
            for second in robots[i + 1 :]:
                dx = second.x - first.x
                dy = second.y - first.y
                dist = math.hypot(dx, dy)
                if dist >= min_dist or dist <= 1e-6:
                    continue
                nx, ny = dx / dist, dy / dist
                push = (min_dist - dist) * 0.5
                first.x -= nx * push
                first.y -= ny * push
                second.x += nx * push
                second.y += ny * push
                collision_event = self._striker_collision_event(first, second)
                if collision_event is not None:
                    self._add_event(collision_event)

    def _clamp_robots(self) -> None:
        half_w = FIELD_WIDTH * 0.5 - self.robot_radius
        half_l = FIELD_LENGTH * 0.5 - self.robot_radius
        for robot in self._active_robots():
            touched = robot.x < -half_w or robot.x > half_w or robot.y < -half_l or robot.y > half_l
            robot.x = float(clamp(robot.x, -half_w, half_w))
            robot.y = float(clamp(robot.y, -half_l, half_l))
            if touched and robot is self.striker:
                self._add_event("SELF_WALL")

    def _active_robots(self):
        robots = [self.striker]
        if self.phase >= 2:
            robots.append(self.goalkeeper)
        if self.phase >= 3:
            robots.extend([self.teammate_goalkeeper, self.opponent_striker])
        return tuple(robots)

    def _observe(self) -> np.ndarray:
        opponent = self._observed_opponent()
        obs = np.array(
            [
                self._clip(self.striker.x / self.half_width),
                self._clip(self.striker.y / self.half_length),
                math.sin(self.striker.yaw),
                math.cos(self.striker.yaw),
                self._clip(self.ball.x / self.half_width),
                self._clip(self.ball.y / self.half_length),
                self._clip(self.ball.vx / MAX_LINEAR_SPD, limit=BALL_VEL_OBS_LIMIT),
                self._clip(self.ball.vy / MAX_LINEAR_SPD, limit=BALL_VEL_OBS_LIMIT),
                0.0,
                1.0,
                self._clip(opponent.x / self.half_width),
                self._clip(opponent.y / self.half_length),
            ],
            dtype=np.float32,
        )
        return obs

    def _observed_opponent(self) -> RobotState:
        if self.phase == 2:
            return self.goalkeeper
        if self.phase >= 3:
            if self._opponent_obs_role == "striker":
                return self.opponent_striker
            return self.goalkeeper
        return RobotState()

    def _compute_reward(
        self,
        prev_striker: RobotState,
        prev_ball: BallState,
        event,
        events: list[str],
    ) -> float:
        prev_goal_dist = self._goal_distance(prev_ball)
        curr_goal_dist = self._goal_distance(self.ball)
        prev_striker_ball = self._robot_ball_distance(prev_striker, prev_ball)
        curr_striker_ball = self._robot_ball_distance(self.striker, self.ball)
        prev_approach_dist = self._approach_target_distance(prev_striker, prev_ball)
        curr_approach_dist = self._approach_target_distance(self.striker, self.ball)
        prev_wrong_side = self._wrong_side_error(prev_striker, prev_ball)
        curr_wrong_side = self._wrong_side_error(self.striker, self.ball)
        ball_progress = prev_goal_dist - curr_goal_dist
        prev_lane_error = self._goal_lane_error(prev_ball)
        curr_lane_error = self._goal_lane_error(self.ball)
        lane_scale = self._goal_lane_progress_scale(self.ball)

        reward = 3.0 * ball_progress * lane_scale
        reward += 1.0 * (prev_striker_ball - curr_striker_ball)
        reward += APPROACH_TARGET_REWARD * (prev_approach_dist - curr_approach_dist)
        reward += GOAL_LANE_REWARD * (prev_lane_error - curr_lane_error)
        if ball_progress > 0.0 and curr_lane_error > 0.0:
            reward -= OFF_TARGET_PROGRESS_PENALTY * ball_progress * curr_lane_error
        reward += WRONG_SIDE_PROGRESS_REWARD * (prev_wrong_side - curr_wrong_side)
        reward -= WRONG_SIDE_STEP_PENALTY * curr_wrong_side
        if ball_progress < 0.0 and prev_wrong_side > 0.05:
            reward += WRONG_SIDE_BACKDRIVE_PENALTY * ball_progress
        reward -= WALL_PROXIMITY_PENALTY * self._wall_proximity_error(self.striker)
        reward -= 0.01

        if event == "GOAL":
            reward += 10.0
        elif event == "OWN_GOAL":
            reward -= 2.0
        elif event == "OUT_OF_BOUNDS":
            reward -= 1.0
        if "BALL_WALL" in events:
            reward -= BALL_WALL_PENALTY
        if "SELF_WALL" in events:
            reward -= SELF_WALL_PENALTY
        if "TEAMMATE_COLLISION" in events:
            reward -= TEAMMATE_COLLISION_PENALTY
        # if "OPPONENT_COLLISION" in events:
        #     reward -= OPPONENT_COLLISION_PENALTY
        return float(reward)

    def _select_event(self, ball_event: Optional[str]):
        if ball_event is not None:
            return ball_event
        for event in ("SELF_WALL", "TEAMMATE_COLLISION", "OPPONENT_COLLISION", "BALL_WALL"):
            if event in self._step_events:
                return event
        return None

    def _add_event(self, event: str) -> None:
        if event not in self._step_events:
            self._step_events.append(event)

    def _striker_collision_event(self, first: RobotState, second: RobotState) -> Optional[str]:
        if first is self.striker:
            other = second
        elif second is self.striker:
            other = first
        else:
            return None

        if self.phase >= 3 and other is self.teammate_goalkeeper:
            return "TEAMMATE_COLLISION"
        if self.phase >= 2 and other is self.goalkeeper:
            return "OPPONENT_COLLISION"
        if self.phase >= 3 and other is self.opponent_striker:
            return "OPPONENT_COLLISION"
        return None

    def _goal_distance(self, ball: BallState) -> float:
        return math.hypot(ball.x - self.goal_b.x, ball.y - self.goal_b.y)

    def _goal_lane_error(self, ball: BallState) -> float:
        target_half_width = max(GOAL_WIDTH * 0.5 - GOAL_LANE_CLEARANCE, 0.05)
        return max(abs(ball.x - self.goal_b.x) - target_half_width, 0.0)

    def _goal_lane_progress_scale(self, ball: BallState) -> float:
        usable_width = max(self.half_width - GOAL_WIDTH * 0.5, 1e-6)
        return float(clamp(1.0 - self._goal_lane_error(ball) / usable_width, 0.0, 1.0))

    def _attack_direction(self, ball: BallState) -> Point2D:
        return safe_unit(
            self.goal_b.x - ball.x,
            self.goal_b.y - ball.y,
            fallback_x=0.0,
            fallback_y=1.0,
        )

    def _approach_target(self, ball: BallState) -> Point2D:
        attack_dir = self._attack_direction(ball)
        return self._clamp_point_to_field(
            Point2D(
                ball.x - attack_dir.x * APPROACH_TARGET_DISTANCE,
                ball.y - attack_dir.y * APPROACH_TARGET_DISTANCE,
            )
        )

    def _approach_target_distance(self, robot: RobotState, ball: BallState) -> float:
        target = self._approach_target(ball)
        return math.hypot(robot.x - target.x, robot.y - target.y)

    def _behind_ball_metrics(self, robot: RobotState, ball: BallState) -> tuple[float, float]:
        attack_dir = self._attack_direction(ball)
        rel_x = ball.x - robot.x
        rel_y = ball.y - robot.y
        behind_depth = rel_x * attack_dir.x + rel_y * attack_dir.y
        lateral = rel_x * -attack_dir.y + rel_y * attack_dir.x
        return behind_depth, lateral

    def _wrong_side_error(self, robot: RobotState, ball: BallState) -> float:
        behind_depth, lateral = self._behind_ball_metrics(robot, ball)
        depth_error = max(WRONG_SIDE_CLEARANCE - behind_depth, 0.0)
        if depth_error <= 0.0:
            return 0.0
        distance = self._robot_ball_distance(robot, ball)
        near_scale = max(WRONG_SIDE_NEAR_DISTANCE - distance, 0.0) / WRONG_SIDE_NEAR_DISTANCE
        if near_scale <= 0.0:
            return 0.0
        line_scale = max(WRONG_SIDE_LATERAL_WIDTH - abs(lateral), 0.0) / WRONG_SIDE_LATERAL_WIDTH
        return depth_error * near_scale * (0.35 + 0.65 * line_scale)

    @staticmethod
    def _robot_ball_distance(robot: RobotState, ball: BallState) -> float:
        return math.hypot(robot.x - ball.x, robot.y - ball.y)

    def _wall_proximity_error(self, robot: RobotState) -> float:
        x_margin = self.half_width - self.robot_radius - abs(robot.x)
        y_margin = self.half_length - self.robot_radius - abs(robot.y)
        margin = min(x_margin, y_margin)
        return max(WALL_PROXIMITY_MARGIN - margin, 0.0)

    def _clip(self, value: float, limit: float = 1.0) -> float:
        return float(clamp(value, -limit, limit))

    def _clamp_keeper_x(self, x_value: float, defense_goal: Point2D, guard_margin: float) -> float:
        half_width = max(GOAL_WIDTH * 0.5 - guard_margin, 0.05)
        return float(clamp(x_value, defense_goal.x - half_width, defense_goal.x + half_width))

    def _clamp_point_to_field(self, point: Point2D) -> Point2D:
        return Point2D(
            clamp(point.x, -self.half_width + self.robot_radius, self.half_width - self.robot_radius),
            clamp(point.y, -self.half_length + self.robot_radius, self.half_length - self.robot_radius),
        )

    def _domain_info(self) -> dict:
        return {
            "ball_radius": self.ball_radius,
            "robot_length": self.robot_length,
            "robot_body_width": self.robot_body_width,
            "robot_leg_width": self.robot_leg_width,
            "robot_radius": self.robot_radius,
            "ball_friction": self.ball_friction,
            "wall_restitution": self.wall_restitution,
            "robot_ball_restitution": self.robot_ball_restitution,
            "robot_ball_tangent_gain": self.robot_ball_tangent_gain,
            "goalkeeper_position_gain": self.goalkeeper_position_gain,
            "opponent_goalkeeper_position_gain": self.opponent_goalkeeper_position_gain,
            "opponent_striker_speed": self.opponent_striker_speed,
            "policy_hz": 1.0 / self.policy_dt,
            "policy_interval_steps": self.policy_interval_steps,
            "action_latency_seconds": self.action_latency_seconds,
            "action_latency_steps": self.action_latency_steps,
            "observation_latency_seconds": self.observation_latency_seconds,
            "observation_latency_steps": self.observation_latency_steps,
        }

    def _info(self, event, reward: float, terminated: bool, truncated: bool) -> dict:
        return {
            "phase": self.phase,
            "step": self.step_count,
            "event": event,
            "events": tuple(self._step_events),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "observed_opponent_role": self._opponent_obs_role,
            "reset_scenario": self._reset_scenario,
            "domain": self._domain_info(),
            "ball": (self.ball.x, self.ball.y, self.ball.vx, self.ball.vy),
            "striker": (self.striker.x, self.striker.y, self.striker.yaw),
            "goalkeeper": (self.goalkeeper.x, self.goalkeeper.y, self.goalkeeper.yaw),
            "teammate_goalkeeper": (
                self.teammate_goalkeeper.x,
                self.teammate_goalkeeper.y,
                self.teammate_goalkeeper.yaw,
            ),
            "opponent_striker": (
                self.opponent_striker.x,
                self.opponent_striker.y,
                self.opponent_striker.yaw,
            ),
        }

    def _render_text(self) -> str:
        return (
            f"phase={self.phase} step={self.step_count} event={self.last_event} "
            f"striker=({self.striker.x:.2f},{self.striker.y:.2f}) "
            f"ball=({self.ball.x:.2f},{self.ball.y:.2f}) "
            f"opp_keeper=({self.goalkeeper.x:.2f},{self.goalkeeper.y:.2f}) "
            f"mate_keeper=({self.teammate_goalkeeper.x:.2f},{self.teammate_goalkeeper.y:.2f}) "
            f"opp_striker=({self.opponent_striker.x:.2f},{self.opponent_striker.y:.2f})"
        )


__all__ = ["SoccerEnv", "predict_guard_target"]
