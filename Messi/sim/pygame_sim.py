"""Pygame-only 2v2 CyberDog soccer simulation.

Replaces Gazebo + sim_locator + game_manager + motion_bridge + tactical_console.
Runs 2D physics, publishes VRPN pose topics, subscribes to motion_servo_cmd Twist
messages from strategy nodes, and renders a live tactical board.

Run standalone (READY state, no strategy nodes):
  python3 sim/pygame_sim.py

Run via launch file:
  ros2 launch launch/sim_2v2_pygame.launch.py

Keyboard controls:
  S - start / resume
  P - pause
  R - reset
  V - toggle velocity arrows
  D - toggle collision radius circles
  Q - quit
"""

import copy
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

try:
    import pygame
except ImportError:
    print("pygame not installed: pip install pygame", file=sys.stderr)
    sys.exit(1)

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.cyberdog_2v2 import ROBOT_SPECS  # noqa: E402
from src.lib.path_planner import PATH_MESSAGE_TYPE, path_points_from_message  # noqa: E402

# ---------------------------------------------------------------------------
# Field / physics constants
# ---------------------------------------------------------------------------
FIELD_CONFIG_PATH = REPO_ROOT / "sim" / "config" / "field.yaml"

DEFAULT_KICKOFF_POSES = {
    "KICKOFF_A": {
        "team_a_1": (0.0, -0.5, math.pi / 2),
        "team_a_2": (0.0, -4.5, math.pi / 2),
        "team_b_1": (0.0, 2.5, -math.pi / 2),
        "team_b_2": (0.0, 4.5, -math.pi / 2),
        "ball": (0.0, 0.0, 0.0),
    },
    "KICKOFF_B": {
        "team_a_1": (0.0, -2.5, math.pi / 2),
        "team_a_2": (0.0, -4.5, math.pi / 2),
        "team_b_1": (0.0, 0.5, -math.pi / 2),
        "team_b_2": (0.0, 4.5, -math.pi / 2),
        "ball": (0.0, 0.0, 0.0),
    },
}

DEFAULT_FIELD_MARKS = {
    "kickoff": [(0.0, -0.5), (0.0, 0.0), (0.0, 0.5)],
    "robot_starts": [
        (0.0, -2.5),
        (0.0, 2.5),
        (-2.275, -4.5),
        (2.275, -4.5),
        (-2.275, 4.5),
        (2.275, 4.5),
    ],
}


def _load_field_config() -> dict:
    if not FIELD_CONFIG_PATH.exists():
        return {}
    with FIELD_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _pose_tuple(value, fallback):
    if not isinstance(value, dict):
        return fallback
    return (
        float(value.get("x", fallback[0])),
        float(value.get("y", fallback[1])),
        float(value.get("yaw", fallback[2])),
    )


def _point_tuple(value):
    return (float(value.get("x", 0.0)), float(value.get("y", 0.0)))


def _load_kickoff_poses(config: dict) -> dict:
    configured = config.get("kickoff_poses", {})
    poses = copy.deepcopy(DEFAULT_KICKOFF_POSES)
    for kickoff_key, defaults in DEFAULT_KICKOFF_POSES.items():
        section = configured.get(kickoff_key, {})
        for rigid_name, fallback in defaults.items():
            poses[kickoff_key][rigid_name] = _pose_tuple(section.get(rigid_name), fallback)
    return poses


def _load_field_marks(config: dict) -> dict:
    configured = config.get("field_marks", {})
    marks = copy.deepcopy(DEFAULT_FIELD_MARKS)
    for group in ("kickoff", "robot_starts"):
        values = configured.get(group)
        if isinstance(values, list):
            marks[group] = [_point_tuple(value) for value in values if isinstance(value, dict)]
    return marks


_FIELD_CONFIG = _load_field_config()
_FIELD = _FIELD_CONFIG.get("field", {})
_PHYSICS = _FIELD_CONFIG.get("physics", {})
_DISPLAY = _FIELD_CONFIG.get("display", {})

FIELD_LENGTH = float(_FIELD.get("length", 10.0))   # y-axis (goal-to-goal)
FIELD_WIDTH = float(_FIELD.get("width", 5.55))     # x-axis (sideline-to-sideline)
GOAL_WIDTH = float(_FIELD.get("goal_width", 1.0))
GOAL_DEPTH = float(_FIELD.get("goal_depth", 0.35))
CENTER_CIRCLE_RADIUS = float(_FIELD.get("center_circle_radius", 1.0))
GOAL_A_Y = -FIELD_LENGTH * 0.5       # team_a defends
GOAL_B_Y = FIELD_LENGTH * 0.5        # team_b defends
BALL_RADIUS = float(_PHYSICS.get("ball_radius", 0.125))
ROBOT_LENGTH = float(_PHYSICS.get("robot_length", 0.43))
ROBOT_BODY_WIDTH = float(_PHYSICS.get("robot_body_width", 0.20))
ROBOT_LEG_WIDTH = float(_PHYSICS.get("robot_leg_width", max(ROBOT_BODY_WIDTH, 0.32)))
ROBOT_RADIUS = float(
    _PHYSICS.get("robot_radius", math.hypot(ROBOT_LENGTH * 0.5, ROBOT_LEG_WIDTH * 0.5))
)
BALL_FRICTION = float(_PHYSICS.get("ball_friction", 1.5))   # m/s^2 deceleration
WALL_RESTITUTION = float(_PHYSICS.get("wall_restitution", 0.6))
BALL_RESTITUTION = float(_PHYSICS.get("ball_restitution", 0.7))
ROBOT_BALL_RESTITUTION = float(_PHYSICS.get("robot_ball_restitution", BALL_RESTITUTION))
ROBOT_BALL_TANGENT_GAIN = float(_PHYSICS.get("robot_ball_tangent_gain", 0.0))
PHYSICS_HZ = int(_PHYSICS.get("physics_hz", 50))
PHYSICS_DT = 1.0 / PHYSICS_HZ

KICKOFF_POSES = _load_kickoff_poses(_FIELD_CONFIG)
FIELD_MARKS = _load_field_marks(_FIELD_CONFIG)

ACTIVE_STATES = {"KICKOFF_TEAM_A", "KICKOFF_TEAM_B", "PLAYING"}

# Display constants
SCREEN_W = int(_DISPLAY.get("screen_width", 1000))
SCREEN_H = int(_DISPLAY.get("screen_height", 650))
FIELD_MARGIN = int(_DISPLAY.get("field_margin", 60))
SIDEBAR_W = int(_DISPLAY.get("sidebar_width", 200))
FIELD_LEFT = FIELD_MARGIN
FIELD_TOP = FIELD_MARGIN

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
YELLOW = (255, 220, 0)
RED = (220, 50, 50)
BLUE = (50, 100, 220)
ORANGE = (255, 140, 0)
GRAY = (160, 160, 160)
LIGHT_GRAY = (220, 220, 220)
CYAN = (0, 200, 200)

# Modern dark-theme palette
BG_COLOR       = (15,  20,  30)
FIELD_COLOR    = (28,  90,  40)
FIELD_STRIPE   = (32, 100,  45)
LINE_COLOR     = (230, 240, 230)
GOAL_COLOR     = (255, 210,  50)
BALL_COLOR     = (255, 220,  30)
BALL_SHADOW    = (180, 140,   0)
TEAM_A_COLOR   = (220,  60,  60)
TEAM_B_COLOR   = ( 60, 120, 220)
TEAM_A_LIGHT   = (255, 130, 130)
TEAM_B_LIGHT   = (130, 180, 255)
ACCENT_CYAN    = ( 40, 210, 210)
ACCENT_ORANGE  = (255, 140,  30)
SIDEBAR_BG     = ( 18,  24,  38)
SIDEBAR_BORDER = ( 40,  55,  80)
TEXT_PRIMARY   = (230, 235, 245)
TEXT_SECONDARY = (140, 155, 175)
TEXT_DIM       = ( 80,  95, 115)

TEAM_COLORS = {"team_a": TEAM_A_COLOR, "team_b": TEAM_B_COLOR}


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

@dataclass
class RobotState:
    x: float
    y: float
    yaw: float
    cmd_vx: float = 0.0
    cmd_vy: float = 0.0
    cmd_wz: float = 0.0
    # world-frame velocity (computed each step, used for rendering)
    vx_world: float = 0.0
    vy_world: float = 0.0


@dataclass
class BallState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _robot_local_to_field(robot: RobotState, lx: float, ly: float) -> tuple[float, float]:
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    return robot.x + lx * c - ly * s, robot.y + lx * s + ly * c


class PhysicsWorld:
    def __init__(self):
        self.robots: dict[str, RobotState] = {}
        self.ball = BallState()
        for spec in ROBOT_SPECS:
            self.robots[spec.rigid_name] = RobotState(spec.x, spec.y, spec.yaw)
        self.reset("KICKOFF_A")

    def reset(self, kickoff_key: str) -> None:
        poses = KICKOFF_POSES.get(kickoff_key, KICKOFF_POSES["KICKOFF_A"])
        for rigid_name, robot in self.robots.items():
            if rigid_name in poses:
                x, y, yaw = poses[rigid_name]
                robot.x, robot.y, robot.yaw = x, y, yaw
                robot.cmd_vx = robot.cmd_vy = robot.cmd_wz = 0.0
                robot.vx_world = robot.vy_world = 0.0
        bx, by, _ = poses.get("ball", (0.0, 0.0, 0.0))
        self.ball = BallState(x=bx, y=by)

    def freeze_robots(self) -> None:
        for robot in self.robots.values():
            robot.cmd_vx = robot.cmd_vy = robot.cmd_wz = 0.0

    def set_robot_cmd(self, rigid_name: str, vx: float, vy: float, wz: float) -> None:
        if rigid_name in self.robots:
            r = self.robots[rigid_name]
            r.cmd_vx, r.cmd_vy, r.cmd_wz = vx, vy, wz

    def step(self) -> list:
        dt = PHYSICS_DT
        self._step_robots(dt)
        events = self._step_ball(dt)
        self._resolve_robot_robot()
        self._clamp_robots()
        return events

    def _step_robots(self, dt: float) -> None:
        for robot in self.robots.values():
            c, s = math.cos(robot.yaw), math.sin(robot.yaw)
            vx_w = robot.cmd_vx * c - robot.cmd_vy * s
            vy_w = robot.cmd_vx * s + robot.cmd_vy * c
            robot.vx_world = vx_w
            robot.vy_world = vy_w
            robot.x += vx_w * dt
            robot.y += vy_w * dt
            robot.yaw += robot.cmd_wz * dt
            robot.yaw = math.atan2(math.sin(robot.yaw), math.cos(robot.yaw))

    def _step_ball(self, dt: float) -> list:
        b = self.ball
        # Friction
        speed = math.hypot(b.vx, b.vy)
        if speed > 1e-4:
            decel = BALL_FRICTION * dt
            new_speed = max(0.0, speed - decel)
            b.vx *= new_speed / speed
            b.vy *= new_speed / speed
        else:
            b.vx = b.vy = 0.0

        # Robot-ball collisions (before position update)
        for robot in self.robots.values():
            self._robot_ball_impulse(robot, b)

        b.x += b.vx * dt
        b.y += b.vy * dt

        # Goal detection (before wall bounce)
        in_goal_mouth = abs(b.x) <= GOAL_WIDTH * 0.5 + BALL_RADIUS
        if in_goal_mouth:
            if b.y <= GOAL_A_Y:
                return ["GOAL_B"]
            if b.y >= GOAL_B_Y:
                return ["GOAL_A"]

        # Wall bounce (side walls)
        half_w = FIELD_WIDTH * 0.5
        if b.x - BALL_RADIUS < -half_w:
            b.x = -half_w + BALL_RADIUS
            b.vx = abs(b.vx) * WALL_RESTITUTION
        elif b.x + BALL_RADIUS > half_w:
            b.x = half_w - BALL_RADIUS
            b.vx = -abs(b.vx) * WALL_RESTITUTION

        # End-wall bounce (only outside goal mouth)
        half_l = FIELD_LENGTH * 0.5
        if not in_goal_mouth:
            if b.y - BALL_RADIUS < -half_l:
                b.y = -half_l + BALL_RADIUS
                b.vy = abs(b.vy) * WALL_RESTITUTION
            elif b.y + BALL_RADIUS > half_l:
                b.y = half_l - BALL_RADIUS
                b.vy = -abs(b.vy) * WALL_RESTITUTION

        # Out-of-bounds safety net
        if abs(b.x) > half_w + BALL_RADIUS or abs(b.y) > half_l + BALL_RADIUS:
            return ["OUT_OF_BOUNDS"]

        return []

    def _robot_ball_impulse(self, robot: RobotState, b: BallState) -> None:
        dx = b.x - robot.x
        dy = b.y - robot.y
        c, s = math.cos(robot.yaw), math.sin(robot.yaw)
        local_x = dx * c + dy * s
        local_y = -dx * s + dy * c

        half_l = max(ROBOT_LENGTH * 0.5, 1e-6)
        half_w = max(ROBOT_LEG_WIDTH * 0.5, 1e-6)
        closest_x = _clamp(local_x, -half_l, half_l)
        closest_y = _clamp(local_y, -half_w, half_w)
        sep_x = local_x - closest_x
        sep_y = local_y - closest_y
        sep_dist = math.hypot(sep_x, sep_y)

        if sep_dist >= BALL_RADIUS:
            return

        if sep_dist > 1e-6:
            local_nx = sep_x / sep_dist
            local_ny = sep_y / sep_dist
            overlap = BALL_RADIUS - sep_dist
        else:
            # Ball center is inside the contact footprint; push it through the
            # nearest face so close, slow dribbles do not get stuck.
            exit_x = half_l - abs(local_x)
            exit_y = half_w - abs(local_y)
            if exit_x <= exit_y:
                local_nx = 1.0 if local_x >= 0.0 else -1.0
                local_ny = 0.0
                overlap = BALL_RADIUS + exit_x
            else:
                local_nx = 0.0
                local_ny = 1.0 if local_y >= 0.0 else -1.0
                overlap = BALL_RADIUS + exit_y

        nx = local_nx * c - local_ny * s
        ny = local_nx * s + local_ny * c

        b.x += nx * overlap
        b.y += ny * overlap

        # Impulse along normal
        rel_vn = (b.vx - robot.vx_world) * nx + (b.vy - robot.vy_world) * ny
        if rel_vn < 0:
            impulse = -(1.0 + ROBOT_BALL_RESTITUTION) * rel_vn
            b.vx += impulse * nx
            b.vy += impulse * ny
        if ROBOT_BALL_TANGENT_GAIN > 0.0:
            tx, ty = -ny, nx
            rel_vt = (robot.vx_world - b.vx) * tx + (robot.vy_world - b.vy) * ty
            b.vx += ROBOT_BALL_TANGENT_GAIN * rel_vt * tx
            b.vy += ROBOT_BALL_TANGENT_GAIN * rel_vt * ty

    def _resolve_robot_robot(self) -> None:
        robots = list(self.robots.values())
        min_dist = 2.0 * ROBOT_RADIUS
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                r1, r2 = robots[i], robots[j]
                dx = r2.x - r1.x
                dy = r2.y - r1.y
                dist = math.hypot(dx, dy)
                if dist < min_dist and dist > 1e-6:
                    nx, ny = dx / dist, dy / dist
                    push = (min_dist - dist) * 0.5
                    r1.x -= nx * push
                    r1.y -= ny * push
                    r2.x += nx * push
                    r2.y += ny * push

    def _clamp_robots(self) -> None:
        half_w = FIELD_WIDTH * 0.5 - ROBOT_RADIUS
        half_l = FIELD_LENGTH * 0.5 - ROBOT_RADIUS
        for robot in self.robots.values():
            robot.x = max(-half_w, min(half_w, robot.x))
            robot.y = max(-half_l, min(half_l, robot.y))


# ---------------------------------------------------------------------------
# Game state machine
# ---------------------------------------------------------------------------

class GameStateMachine:
    INIT = "INIT"
    READY = "READY"
    KICKOFF_TEAM_A = "KICKOFF_TEAM_A"
    KICKOFF_TEAM_B = "KICKOFF_TEAM_B"
    PLAYING = "PLAYING"
    GOAL_TEAM_A = "GOAL_TEAM_A"
    GOAL_TEAM_B = "GOAL_TEAM_B"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    PAUSED = "PAUSED"
    RESETTING = "RESETTING"

    def __init__(self, kickoff_team: str = "team_a"):
        self.state = self.INIT
        self.score = {"team_a": 0, "team_b": 0}
        self._kickoff_team = kickoff_team
        self._entry_time = time.monotonic()
        self._play_start_time: Optional[float] = None
        self._play_elapsed: float = 0.0

    @property
    def match_time(self) -> float:
        if self._play_start_time is None:
            return self._play_elapsed
        if self.state == self.PLAYING:
            return self._play_elapsed + (time.monotonic() - self._play_start_time)
        return self._play_elapsed

    def tick(self, physics_events: list, now: float) -> list:
        actions = []
        elapsed = now - self._entry_time

        if self.state == self.INIT:
            self._transition(self.READY, now)

        elif self.state == self.READY:
            pass

        elif self.state in (self.KICKOFF_TEAM_A, self.KICKOFF_TEAM_B):
            if elapsed > 3.0:
                self._transition(self.PLAYING, now)
                actions.append("UNFREEZE")

        elif self.state == self.PLAYING:
            for ev in physics_events:
                if ev == "GOAL_A":
                    self.score["team_a"] += 1
                    self._kickoff_team = "team_b"
                    self._transition(self.GOAL_TEAM_A, now)
                    actions.append("FREEZE")
                    break
                elif ev == "GOAL_B":
                    self.score["team_b"] += 1
                    self._kickoff_team = "team_a"
                    self._transition(self.GOAL_TEAM_B, now)
                    actions.append("FREEZE")
                    break
                elif ev == "OUT_OF_BOUNDS":
                    self._transition(self.OUT_OF_BOUNDS, now)
                    actions.append("FREEZE")
                    break

        elif self.state in (self.GOAL_TEAM_A, self.GOAL_TEAM_B):
            if elapsed > 3.0:
                self._transition(self.RESETTING, now)
                kickoff_key = "KICKOFF_A" if self._kickoff_team == "team_a" else "KICKOFF_B"
                actions.append(f"RESET_{kickoff_key}")

        elif self.state == self.OUT_OF_BOUNDS:
            if elapsed > 2.0:
                self._transition(self.RESETTING, now)
                kickoff_key = "KICKOFF_A" if self._kickoff_team == "team_a" else "KICKOFF_B"
                actions.append(f"RESET_{kickoff_key}")

        elif self.state == self.RESETTING:
            if elapsed > 2.0:
                kickoff_state = self.KICKOFF_TEAM_A if self._kickoff_team == "team_a" else self.KICKOFF_TEAM_B
                self._transition(kickoff_state, now)

        elif self.state == self.PAUSED:
            pass

        return actions

    def start(self) -> bool:
        if self.state in (self.INIT, self.READY, self.PAUSED):
            kickoff_state = self.KICKOFF_TEAM_A if self._kickoff_team == "team_a" else self.KICKOFF_TEAM_B
            self._transition(kickoff_state, time.monotonic())
            kickoff_key = "KICKOFF_A" if self._kickoff_team == "team_a" else "KICKOFF_B"
            return True, kickoff_key
        return False, None

    def pause(self) -> bool:
        if self.state == self.PLAYING:
            if self._play_start_time is not None:
                self._play_elapsed += time.monotonic() - self._play_start_time
                self._play_start_time = None
            self._transition(self.PAUSED, time.monotonic())
            return True
        return False

    def reset(self) -> str:
        self._play_elapsed = 0.0
        self._play_start_time = None
        self._transition(self.RESETTING, time.monotonic())
        return "KICKOFF_A" if self._kickoff_team == "team_a" else "KICKOFF_B"

    def _transition(self, new_state: str, now: float) -> None:
        if new_state != self.state:
            self.state = new_state
            self._entry_time = now
            if new_state == self.PLAYING and self._play_start_time is None:
                self._play_start_time = now
            elif new_state != self.PLAYING:
                if self._play_start_time is not None:
                    self._play_elapsed += now - self._play_start_time
                    self._play_start_time = None


# ---------------------------------------------------------------------------
# Shared state (main thread ↔ ROS thread)
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        # Written by main thread, read by ROS thread
        self.robots: dict[str, RobotState] = {}
        self.ball = BallState()
        self.game_state = "INIT"
        self.score = {"team_a": 0, "team_b": 0}
        # Written by ROS thread, read by main thread
        self.robot_cmds: dict[str, tuple] = {}
        # Written by ROS thread for rendering
        self.intents: dict[str, dict] = {}
        # Planned paths for rendering: {name: [(x, y), ...]}
        self.paths: dict[str, list] = {}

    def update_from_physics(self, physics: PhysicsWorld, game_sm: GameStateMachine) -> None:
        self.robots = {k: copy.copy(v) for k, v in physics.robots.items()}
        self.ball = copy.copy(physics.ball)
        self.game_state = game_sm.state
        self.score = dict(game_sm.score)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "robots": {k: copy.copy(v) for k, v in self.robots.items()},
                "ball": copy.copy(self.ball),
                "game_state": self.game_state,
                "score": dict(self.score),
                "intents": dict(self.intents),
                "paths": {k: list(v) for k, v in self.paths.items()},
            }


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

def _quaternion_from_yaw(yaw: float):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class SimNode(Node):
    def __init__(self, shared: SharedState, game_sm: GameStateMachine, physics: PhysicsWorld):
        super().__init__("pygame_sim")
        self._shared = shared
        self._game_sm = game_sm
        self._physics = physics
        self._last_game_state: Optional[str] = None
        self._last_score: Optional[tuple[int, int]] = None
        self._last_snapshot_log_time = 0.0

        self.declare_parameter("kickoff_team", "team_a")
        kickoff_team = self.get_parameter("kickoff_team").value
        game_sm._kickoff_team = kickoff_team

        # Subscribers: motion commands from strategy nodes
        from geometry_msgs.msg import Twist
        for spec in ROBOT_SPECS:
            topic = f"/{spec.dog_namespace}/motion_servo_cmd"
            self.create_subscription(
                Twist, topic, self._make_cmd_cb(spec.rigid_name), 10
            )

        # Publishers
        self._pose_pubs: dict[str, object] = {}
        self._twist_pubs: dict[str, object] = {}
        for spec in ROBOT_SPECS:
            self._pose_pubs[spec.rigid_name] = self.create_publisher(
                PoseStamped, f"/vrpn/{spec.rigid_name}/pose", 10
            )
            self._twist_pubs[spec.rigid_name] = self.create_publisher(
                TwistStamped, f"/vrpn/{spec.rigid_name}/twist", 10
            )
        self._pose_pubs["soccer_ball"] = self.create_publisher(
            PoseStamped, "/vrpn/soccer_ball/pose", 10
        )
        self._twist_pubs["soccer_ball"] = self.create_publisher(
            TwistStamped, "/vrpn/soccer_ball/twist", 10
        )
        self._pose_pubs["goal_a"] = self.create_publisher(
            PoseStamped, "/vrpn/goal_a/pose", 10
        )
        self._pose_pubs["goal_b"] = self.create_publisher(
            PoseStamped, "/vrpn/goal_b/pose", 10
        )
        self._state_pub = self.create_publisher(String, "/soccer/game_state", 10)
        self._score_pub = self.create_publisher(String, "/soccer/score", 10)

        # Intent subscribers for rendering
        from std_msgs.msg import String as Str
        import json
        self._json = json
        with self._shared.lock:
            for spec in ROBOT_SPECS:
                self._shared.intents.setdefault(spec.rigid_name, _default_intent(spec.rigid_name))
        for spec in ROBOT_SPECS:
            team = spec.team
            robot = f"robot_{spec.index}"
            key = spec.rigid_name
            self.create_subscription(
                Str,
                f"/soccer/team/{team}/{robot}/intent",
                self._make_intent_cb(key),
                10,
            )
            topic_prefix = f"/soccer/team/{team}/{robot}"
            self.create_subscription(
                Str,
                f"{topic_prefix}/striker/state",
                self._make_role_state_cb(key, "striker"),
                10,
            )
            self.create_subscription(
                Str,
                f"{topic_prefix}/goalkeeper/state",
                self._make_role_state_cb(key, "goalkeeper"),
                10,
            )

        # Path subscribers for rendering
        self.create_subscription(
            PATH_MESSAGE_TYPE, "/soccer/striker/path",
            self._make_path_cb("approach"), 10,
        )
        for spec in ROBOT_SPECS:
            self.create_subscription(
                PATH_MESSAGE_TYPE,
                f"/soccer/team/{spec.team}/{spec.dog_namespace.split('/')[-1]}/striker/path",
                self._make_path_cb(f"{spec.rigid_name}_approach"),
                10,
            )

        # Services
        self.create_service(Trigger, "/soccer/start", self._start_cb)
        self.create_service(Trigger, "/soccer/pause", self._pause_cb)
        self.create_service(Trigger, "/soccer/reset", self._reset_cb)

        # 20 Hz publish timer
        self.create_timer(0.05, self._publish)

        self.get_logger().info("pygame_sim node started")

    def _make_cmd_cb(self, rigid_name: str):
        def cb(msg):
            with self._shared.lock:
                self._shared.robot_cmds[rigid_name] = (
                    msg.linear.x, msg.linear.y, msg.angular.z
                )
        return cb

    def _make_intent_cb(self, key: str):
        def cb(msg):
            try:
                data = self._json.loads(msg.data)
                with self._shared.lock:
                    current = _default_intent(key)
                    current.update(self._shared.intents.get(key, {}))
                    current.update(data)
                    self._shared.intents[key] = current
            except Exception:
                pass
        return cb

    def _make_role_state_cb(self, key: str, role: str):
        def cb(msg):
            with self._shared.lock:
                current = _default_intent(key)
                current.update(self._shared.intents.get(key, {}))
                current.setdefault("role", role)
                current["state"] = msg.data
                self._shared.intents[key] = current
        return cb

    def _make_path_cb(self, name: str):
        def cb(msg):
            pts = [(p.x, p.y) for p in path_points_from_message(msg)]
            with self._shared.lock:
                self._shared.paths[name] = pts
        return cb

    def _start_cb(self, request, response):
        ok, kickoff_key = self._game_sm.start()
        if ok:
            self._physics.reset(kickoff_key)
            response.success = True
            response.message = f"Started: {self._game_sm.state}"
        else:
            response.success = False
            response.message = f"Cannot start from {self._game_sm.state}"
        return response

    def _pause_cb(self, request, response):
        ok = self._game_sm.pause()
        response.success = ok
        response.message = "Paused" if ok else f"Cannot pause from {self._game_sm.state}"
        return response

    def _reset_cb(self, request, response):
        kickoff_key = self._game_sm.reset()
        self._physics.reset(kickoff_key)
        response.success = True
        response.message = "Resetting"
        return response

    def _publish(self):
        snap = self._shared.snapshot()
        now = self.get_clock().now().to_msg()
        self._log_match_snapshot(snap)

        for spec in ROBOT_SPECS:
            robot = snap["robots"].get(spec.rigid_name)
            if robot is None:
                continue
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = "field_red"
            ps.pose.position.x = robot.x
            ps.pose.position.y = robot.y
            ps.pose.position.z = 0.0
            ps.pose.orientation = _quaternion_from_yaw(robot.yaw)
            self._pose_pubs[spec.rigid_name].publish(ps)

            ts = TwistStamped()
            ts.header.stamp = now
            ts.header.frame_id = "field_red"
            ts.twist.linear.x = robot.vx_world
            ts.twist.linear.y = robot.vy_world
            self._twist_pubs[spec.rigid_name].publish(ts)

        ball = snap["ball"]
        bps = PoseStamped()
        bps.header.stamp = now
        bps.header.frame_id = "field_red"
        bps.pose.position.x = ball.x
        bps.pose.position.y = ball.y
        bps.pose.position.z = 0.0
        bps.pose.orientation = _quaternion_from_yaw(0.0)
        self._pose_pubs["soccer_ball"].publish(bps)

        bts = TwistStamped()
        bts.header.stamp = now
        bts.header.frame_id = "field_red"
        bts.twist.linear.x = ball.vx
        bts.twist.linear.y = ball.vy
        self._twist_pubs["soccer_ball"].publish(bts)

        for name, (gx, gy) in (("goal_a", (0.0, GOAL_A_Y)), ("goal_b", (0.0, GOAL_B_Y))):
            gps = PoseStamped()
            gps.header.stamp = now
            gps.header.frame_id = "field_red"
            gps.pose.position.x = gx
            gps.pose.position.y = gy
            gps.pose.position.z = 0.0
            gps.pose.orientation = _quaternion_from_yaw(0.0)
            self._pose_pubs[name].publish(gps)

        state_msg = String()
        state_msg.data = snap["game_state"]
        self._state_pub.publish(state_msg)

        score_msg = String()
        score_msg.data = f"{snap['score']['team_a']}:{snap['score']['team_b']}"
        self._score_pub.publish(score_msg)

    def _log_match_snapshot(self, snap: dict) -> None:
        game_state = snap["game_state"]
        score = (snap["score"]["team_a"], snap["score"]["team_b"])
        if game_state != self._last_game_state:
            self.get_logger().info(
                f"pygame_sim state: {self._last_game_state or 'None'} -> {game_state}"
            )
            self._last_game_state = game_state
        if score != self._last_score:
            self.get_logger().info(f"pygame_sim score: team_a={score[0]} team_b={score[1]}")
            self._last_score = score
        now = time.monotonic()
        if game_state == "PLAYING" and now - self._last_snapshot_log_time >= 1.0:
            ball = snap["ball"]
            robot_bits = []
            for rigid_name in ("team_a_1", "team_a_2", "team_b_1", "team_b_2"):
                robot = snap["robots"].get(rigid_name)
                if robot is None:
                    continue
                intent = snap["intents"].get(rigid_name, {})
                robot_bits.append(
                    f"{ROBOT_LABELS.get(rigid_name, '?')}="
                    f"({robot.x:+.2f},{robot.y:+.2f}) "
                    f"{_intent_role_short(rigid_name, intent)}-{_intent_state(intent)[:3].upper()}"
                )
            self.get_logger().info(
                "pygame_sim snapshot: "
                f"ball=({ball.x:+.2f},{ball.y:+.2f}) "
                f"v=({ball.vx:+.2f},{ball.vy:+.2f}) "
                + " ".join(robot_bits)
            )
            self._last_snapshot_log_time = now


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

ROBOT_LABELS = {
    "team_a_1": "A1", "team_a_2": "A2",
    "team_b_1": "B1", "team_b_2": "B2",
}


def _load_robot_role_hints() -> dict[str, str]:
    roles: dict[str, str] = {}
    for spec in ROBOT_SPECS:
        role = ""
        config_path = REPO_ROOT / "config" / spec.config_file
        try:
            with config_path.open("r", encoding="utf-8") as stream:
                role = str((yaml.safe_load(stream) or {}).get("role", "")).strip()
        except OSError:
            role = ""
        roles[spec.rigid_name] = role or "?"
    return roles


ROBOT_ROLE_HINTS = _load_robot_role_hints()


def _default_intent(rigid_name: str) -> dict:
    configured_role = ROBOT_ROLE_HINTS.get(rigid_name, "?")
    intent = {"state": "?"}
    if configured_role == "legacy_striker":
        intent.update({"role": "striker", "controller": "legacy"})
    elif configured_role == "policy_striker":
        intent.update({"role": "striker", "controller": "policy"})
    elif configured_role and configured_role != "?":
        intent["role"] = configured_role
    return intent


def _intent_role_name(rigid_name: str, intent: dict) -> str:
    configured_role = ROBOT_ROLE_HINTS.get(rigid_name, "")
    controller = str(intent.get("controller") or "").strip().lower()
    role = str(intent.get("role") or "").strip().lower()

    if role == "goalkeeper" or configured_role == "goalkeeper":
        return "goalkeeper"
    if role == "legacy_striker" or controller == "legacy" or configured_role == "legacy_striker":
        return "legacy striker"
    if (
        role == "policy_striker"
        or controller == "policy"
        or "policy_model" in intent
        or configured_role == "policy_striker"
    ):
        return "policy striker"
    if role == "striker" or configured_role == "striker":
        return "striker"
    return role or configured_role or "?"


def _intent_role_short(rigid_name: str, intent: dict) -> str:
    role_name = _intent_role_name(rigid_name, intent)
    if role_name == "legacy striker":
        return "L"
    if role_name == "policy striker":
        return "P"
    if role_name == "goalkeeper":
        return "G"
    if role_name == "striker":
        return "S"
    return "?"


def _intent_state(intent: dict) -> str:
    state = str(intent.get("state") or "?").strip()
    return state or "?"


def _robot_overlay_label(rigid_name: str, intent: dict) -> str:
    label = ROBOT_LABELS.get(rigid_name, "?")
    state = _intent_state(intent)[:3].upper()
    return f"{label}:{_intent_role_short(rigid_name, intent)}-{state}"


def _robot_sidebar_line(rigid_name: str, intent: dict) -> str:
    label = ROBOT_LABELS.get(rigid_name, "?")
    return f"  {label}: {_intent_role_name(rigid_name, intent)} {_intent_state(intent)}"


def field_to_screen(fx: float, fy: float, field_w: int, field_h: int) -> tuple:
    # Field frame: +y points to the right goal, +x points down the pitch sketch.
    px = int(FIELD_LEFT + (fy / FIELD_LENGTH + 0.5) * field_w)
    py = int(FIELD_TOP + (fx / FIELD_WIDTH + 0.5) * field_h)
    return px, py


def screen_to_field(px: int, py: int, field_w: int, field_h: int) -> tuple:
    fy = (px - FIELD_LEFT) / field_w * FIELD_LENGTH - FIELD_LENGTH * 0.5
    fx = (py - FIELD_TOP) / field_h * FIELD_WIDTH - FIELD_WIDTH * 0.5
    return fx, fy


def field_delta_to_screen(dx: float, dy: float, field_w: int, field_h: int) -> tuple:
    return int(dy / FIELD_LENGTH * field_w), int(dx / FIELD_WIDTH * field_h)


def _draw_rounded_rect(
    surf: pygame.Surface,
    color: tuple,
    rect: pygame.Rect,
    radius: int = 8,
    border: int = 0,
    border_color: Optional[tuple] = None,
) -> None:
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)


def _draw_pill(
    surf: pygame.Surface,
    color: tuple,
    cx: int, cy: int,
    w: int, h: int,
    text: str,
    font: pygame.font.Font,
    text_color: tuple = TEXT_PRIMARY,
) -> None:
    r = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
    _draw_rounded_rect(surf, color, r, radius=h // 2)
    lbl = font.render(text, True, text_color)
    surf.blit(lbl, lbl.get_rect(center=(cx, cy)))


class Renderer:
    def __init__(self, screen: pygame.Surface, field_w: int, field_h: int):
        self._screen = screen
        self._fw = field_w
        self._fh = field_h
        self._ball_trace: deque = deque(maxlen=80)
        self._font_xs     = pygame.font.SysFont("segoeui,helvetica,sans", 10)
        self._font_sm     = pygame.font.SysFont("segoeui,helvetica,sans", 12)
        self._font_md     = pygame.font.SysFont("segoeui,helvetica,sans", 14, bold=True)
        self._font_lg     = pygame.font.SysFont("segoeui,helvetica,sans", 18, bold=True)
        self._font_xl     = pygame.font.SysFont("segoeui,helvetica,sans", 26, bold=True)
        self._font_score  = pygame.font.SysFont("segoeui,helvetica,sans", 36, bold=True)
        self.show_vel   = False
        self.show_debug = False
        self._field_surf = self._make_field_surface()

    def _make_field_surface(self) -> pygame.Surface:
        surf = pygame.Surface((self._fw, self._fh))
        surf.fill(FIELD_COLOR)
        n_stripes = 10
        stripe_w = self._fw // n_stripes
        for i in range(n_stripes):
            if i % 2 == 0:
                pygame.draw.rect(surf, FIELD_STRIPE, (i * stripe_w, 0, stripe_w, self._fh))
        return surf

    def draw(self, snap: dict, match_time: float, drag_info: Optional[dict] = None) -> None:
        self._screen.fill(BG_COLOR)
        self._screen.blit(self._field_surf, (FIELD_LEFT, FIELD_TOP))
        self._draw_field_lines()
        self._draw_goals()
        self._draw_paths(snap.get("paths", {}))
        self._draw_ball_trace(snap["ball"])
        self._draw_robots(snap["robots"], snap["intents"])
        self._draw_ball(snap["ball"])
        if drag_info and drag_info.get("active"):
            self._draw_drag_highlight(drag_info)
        self._draw_goal_flash(snap["game_state"])
        self._draw_banner(snap["game_state"])
        self._draw_hud(snap, match_time)
        self._draw_sidebar(snap, match_time)

    def _draw_field_lines(self) -> None:
        fw, fh = self._fw, self._fh
        s = self._screen
        # Boundary
        pygame.draw.rect(s, LINE_COLOR, (FIELD_LEFT, FIELD_TOP, fw, fh), 2)
        # Halfway line
        cx = FIELD_LEFT + fw // 2
        cy = FIELD_TOP + fh // 2
        pygame.draw.line(s, LINE_COLOR, (cx, FIELD_TOP), (cx, FIELD_TOP + fh), 1)
        # Centre circle
        meter_px = min(fw / FIELD_LENGTH, fh / FIELD_WIDTH)
        r_px = int(meter_px * CENTER_CIRCLE_RADIUS)
        pygame.draw.circle(s, LINE_COLOR, (cx, cy), r_px, 1)
        pygame.draw.circle(s, LINE_COLOR, (cx, cy), 3)
        # Penalty boxes
        pen_depth = 1.5
        pen_width = 2.5
        for gy_field in (GOAL_A_Y, GOAL_B_Y):
            sign = 1 if gy_field > 0 else -1
            tl = field_to_screen(-pen_width / 2, gy_field - sign * pen_depth, fw, fh)
            br = field_to_screen( pen_width / 2, gy_field, fw, fh)
            rx = min(tl[0], br[0])
            ry = min(tl[1], br[1])
            rw = abs(br[0] - tl[0])
            rh = abs(br[1] - tl[1])
            pygame.draw.rect(s, LINE_COLOR, (rx, ry, rw, rh), 1)
        self._draw_field_marks()

    def _draw_goals(self) -> None:
        fw, fh = self._fw, self._fh
        s = self._screen
        gw_px = int(fh * GOAL_WIDTH / FIELD_WIDTH)
        gy = FIELD_TOP + (fh - gw_px) // 2
        depth = max(10, int(fw * GOAL_DEPTH / FIELD_LENGTH))
        ga_rect = pygame.Rect(FIELD_LEFT - depth, gy, depth, gw_px)
        pygame.draw.rect(s, (40, 40, 40), ga_rect)
        pygame.draw.rect(s, GOAL_COLOR, ga_rect, 3)
        gb_rect = pygame.Rect(FIELD_LEFT + fw, gy, depth, gw_px)
        pygame.draw.rect(s, (40, 40, 40), gb_rect)
        pygame.draw.rect(s, GOAL_COLOR, gb_rect, 3)

    def _draw_field_marks(self) -> None:
        for x, y in FIELD_MARKS.get("kickoff", []):
            self._draw_cross(x, y, LINE_COLOR, size=8, width=2)
        for x, y in FIELD_MARKS.get("robot_starts", []):
            self._draw_cross(x, y, FIELD_STRIPE, size=9, width=3)

    def _draw_cross(self, fx: float, fy: float, color: tuple, size: int, width: int) -> None:
        sx, sy = field_to_screen(fx, fy, self._fw, self._fh)
        pygame.draw.line(self._screen, color, (sx - size, sy), (sx + size, sy), width)
        pygame.draw.line(self._screen, color, (sx, sy - size), (sx, sy + size), width)

    def _draw_paths(self, paths: dict) -> None:
        fw, fh = self._fw, self._fh
        for name, pts in paths.items():
            if len(pts) < 2:
                continue
            if name == "dribble" or name.endswith("_dribble"):
                color, width = ACCENT_ORANGE, 2
            elif name == "approach":
                color, width = ACCENT_CYAN, 1
            elif name.endswith("_approach"):
                color = TEAM_A_LIGHT if name.startswith("team_a") else (
                    TEAM_B_LIGHT if name.startswith("team_b") else ACCENT_CYAN)
                width = 1
            else:
                color, width = TEXT_DIM, 1
            screen_pts = [field_to_screen(x, y, fw, fh) for x, y in pts]
            pygame.draw.lines(self._screen, color, False, screen_pts, width)
            for sp in screen_pts:
                pygame.draw.circle(self._screen, color, sp, 3)

    def _draw_ball_trace(self, ball: BallState) -> None:
        self._ball_trace.append((ball.x, ball.y))
        n = len(self._ball_trace)
        for i, tp in enumerate(self._ball_trace):
            alpha = int(200 * i / max(n, 1))
            px, py = field_to_screen(tp[0], tp[1], self._fw, self._fh)
            r = max(1, int(3 * i / max(n, 1)))
            pygame.draw.circle(self._screen, (alpha, alpha, 0), (px, py), r)

    def _draw_ball(self, ball: BallState) -> None:
        fw, fh = self._fw, self._fh
        meter_px = min(fw / FIELD_LENGTH, fh / FIELD_WIDTH)
        ball_r_px = max(5, int(meter_px * BALL_RADIUS))
        bx, by = field_to_screen(ball.x, ball.y, fw, fh)
        # Shadow
        pygame.draw.circle(self._screen, (0, 0, 0), (bx + 3, by + 3), ball_r_px)
        # Ball
        pygame.draw.circle(self._screen, BALL_COLOR, (bx, by), ball_r_px)
        pygame.draw.circle(self._screen, BALL_SHADOW, (bx, by), ball_r_px, 2)
        # Shine
        shine_r = max(2, ball_r_px // 3)
        pygame.draw.circle(self._screen, (255, 255, 200),
                           (bx - ball_r_px // 3, by - ball_r_px // 3), shine_r)
        if self.show_vel:
            dx, dy = field_delta_to_screen(ball.vx * 0.4, ball.vy * 0.4, fw, fh)
            if abs(ball.vx) > 0.05 or abs(ball.vy) > 0.05:
                pygame.draw.line(self._screen, ACCENT_CYAN, (bx, by), (bx + dx, by + dy), 2)

    def _draw_robots(self, robots: dict, intents: dict) -> None:
        fw, fh = self._fw, self._fh
        meter_px = min(fw / FIELD_LENGTH, fh / FIELD_WIDTH)
        for rigid_name, robot in robots.items():
            team = "team_a" if rigid_name.startswith("team_a") else "team_b"
            color = TEAM_A_COLOR if team == "team_a" else TEAM_B_COLOR
            light = TEAM_A_LIGHT if team == "team_a" else TEAM_B_LIGHT
            sx, sy = field_to_screen(robot.x, robot.y, fw, fh)

            def footprint_points(length: float, width: float) -> list[tuple[int, int]]:
                hl, hw = length * 0.5, width * 0.5
                corners = ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
                return [field_to_screen(*_robot_local_to_field(robot, lx, ly), fw, fh) for lx, ly in corners]

            contact_pts = footprint_points(ROBOT_LENGTH, ROBOT_LEG_WIDTH)
            body_pts    = footprint_points(ROBOT_LENGTH, ROBOT_BODY_WIDTH)

            # Shadow
            shadow_pts = [(x + 3, y + 3) for x, y in body_pts]
            pygame.draw.polygon(self._screen, (0, 0, 0), shadow_pts)
            # Body
            pygame.draw.polygon(self._screen, color, body_pts)
            pygame.draw.polygon(self._screen, light, contact_pts, 1)
            pygame.draw.polygon(self._screen, WHITE, body_pts, 2)

            # Direction arrow
            dx, dy = field_delta_to_screen(
                math.cos(robot.yaw) * ROBOT_LENGTH * 0.7,
                math.sin(robot.yaw) * ROBOT_LENGTH * 0.7,
                fw, fh,
            )
            ex, ey = sx + dx, sy + dy
            pygame.draw.line(self._screen, WHITE, (sx, sy), (ex, ey), 2)
            pygame.draw.circle(self._screen, WHITE, (ex, ey), 3)

            # Label badge
            intent = intents.get(rigid_name, {})
            label = _robot_overlay_label(rigid_name, intent)
            label_offset = max(14, int(meter_px * ROBOT_LEG_WIDTH * 0.5)) + 4
            _draw_pill(self._screen, color, sx, sy + label_offset,
                       max(36, len(label) * 7 + 8), 16, label, self._font_xs, WHITE)

            # Velocity arrow
            if self.show_vel:
                dx, dy = field_delta_to_screen(robot.vx_world * 0.4, robot.vy_world * 0.4, fw, fh)
                if abs(robot.vx_world) > 0.05 or abs(robot.vy_world) > 0.05:
                    pygame.draw.line(self._screen, ACCENT_CYAN, (sx, sy), (sx + dx, sy + dy), 2)
            # Debug footprint
            if self.show_debug:
                pygame.draw.polygon(self._screen, ACCENT_ORANGE, contact_pts, 2)
                r_px = int(meter_px * ROBOT_RADIUS)
                pygame.draw.circle(self._screen, ACCENT_ORANGE, (sx, sy), r_px, 1)

    def _draw_drag_highlight(self, drag_info: dict) -> None:
        fw, fh = self._fw, self._fh
        meter_px = min(fw / FIELD_LENGTH, fh / FIELD_WIDTH)
        kind = drag_info.get("kind")
        if kind == "ball":
            ball_r_px = max(5, int(meter_px * BALL_RADIUS))
            bx, by = field_to_screen(drag_info["x"], drag_info["y"], fw, fh)
            surf = pygame.Surface((ball_r_px * 4, ball_r_px * 4), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255, 255, 100, 120),
                               (ball_r_px * 2, ball_r_px * 2), ball_r_px + 4)
            self._screen.blit(surf, (bx - ball_r_px * 2, by - ball_r_px * 2))
        elif kind == "robot":
            rx, ry = field_to_screen(drag_info["x"], drag_info["y"], fw, fh)
            r_px = int(meter_px * ROBOT_RADIUS)
            surf = pygame.Surface(((r_px + 6) * 2, (r_px + 6) * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255, 255, 100, 100),
                               (r_px + 6, r_px + 6), r_px + 6)
            self._screen.blit(surf, (rx - r_px - 6, ry - r_px - 6))

    def _draw_goal_flash(self, game_state: str) -> None:
        fw, fh = self._fw, self._fh
        gw_px = int(fh * GOAL_WIDTH / FIELD_WIDTH)
        gy = FIELD_TOP + (fh - gw_px) // 2
        depth = max(10, int(fw * GOAL_DEPTH / FIELD_LENGTH))
        if game_state == "GOAL_TEAM_A":
            pygame.draw.rect(self._screen, TEAM_B_COLOR, (FIELD_LEFT + fw, gy, depth, gw_px))
        elif game_state == "GOAL_TEAM_B":
            pygame.draw.rect(self._screen, TEAM_A_COLOR, (FIELD_LEFT - depth, gy, depth, gw_px))

    def _draw_banner(self, game_state: str) -> None:
        banners = {
            "KICKOFF_TEAM_A": ("KICKOFF  —  TEAM A", TEAM_A_COLOR),
            "KICKOFF_TEAM_B": ("KICKOFF  —  TEAM B", TEAM_B_COLOR),
            "GOAL_TEAM_A":    ("GOAL!  TEAM A",       TEAM_A_COLOR),
            "GOAL_TEAM_B":    ("GOAL!  TEAM B",       TEAM_B_COLOR),
            "OUT_OF_BOUNDS":  ("OUT OF BOUNDS",        ACCENT_ORANGE),
            "RESETTING":      ("RESETTING...",         TEXT_SECONDARY),
            "PAUSED":         ("PAUSED",               TEXT_SECONDARY),
            "READY":          ("READY  —  Press  S",  ACCENT_CYAN),
        }
        entry = banners.get(game_state)
        if entry is None:
            return
        text, color = entry
        surf = self._font_xl.render(text, True, color)
        cx = FIELD_LEFT + self._fw // 2
        cy = FIELD_TOP + self._fh // 2
        rect = surf.get_rect(center=(cx, cy))
        bg = pygame.Surface((rect.width + 32, rect.height + 16), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 170))
        pygame.draw.rect(bg, color, bg.get_rect(), 2, border_radius=8)
        self._screen.blit(bg, (rect.x - 16, rect.y - 8))
        self._screen.blit(surf, rect)

    def _draw_hud(self, snap: dict, match_time: float) -> None:
        sc = snap["score"]
        mins = int(match_time) // 60
        secs = int(match_time) % 60
        bar_w, bar_h = 320, 44
        bar_x = FIELD_LEFT + self._fw // 2 - bar_w // 2
        bar_y = max(4, FIELD_TOP - bar_h - 8)
        bar_surf = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
        bar_surf.fill((10, 15, 25, 200))
        pygame.draw.rect(bar_surf, SIDEBAR_BORDER, bar_surf.get_rect(), 1, border_radius=10)
        self._screen.blit(bar_surf, (bar_x, bar_y))
        a_txt = self._font_score.render(str(sc["team_a"]), True, TEAM_A_LIGHT)
        self._screen.blit(a_txt, a_txt.get_rect(midright=(bar_x + bar_w // 2 - 30,
                                                            bar_y + bar_h // 2)))
        sep = self._font_md.render(":", True, TEXT_SECONDARY)
        self._screen.blit(sep, sep.get_rect(center=(bar_x + bar_w // 2, bar_y + bar_h // 2)))
        b_txt = self._font_score.render(str(sc["team_b"]), True, TEAM_B_LIGHT)
        self._screen.blit(b_txt, b_txt.get_rect(midleft=(bar_x + bar_w // 2 + 30,
                                                           bar_y + bar_h // 2)))
        t_txt = self._font_sm.render(f"{mins:02d}:{secs:02d}", True, TEXT_SECONDARY)
        self._screen.blit(t_txt, t_txt.get_rect(midtop=(bar_x + bar_w // 2, bar_y + 2)))

    def _draw_sidebar(self, snap: dict, match_time: float) -> None:
        screen_w = self._screen.get_width()
        screen_h = self._screen.get_height()
        sb_x = screen_w - SIDEBAR_W

        # Background panel
        sb_surf = pygame.Surface((SIDEBAR_W, screen_h), pygame.SRCALPHA)
        sb_surf.fill((18, 24, 38, 230))
        pygame.draw.line(sb_surf, SIDEBAR_BORDER, (0, 0), (0, screen_h), 1)
        self._screen.blit(sb_surf, (sb_x, 0))

        x0 = sb_x + 12
        y = 16

        # Title
        title = self._font_lg.render("CyberDog Soccer Sim", True, TEXT_PRIMARY)
        self._screen.blit(title, (x0, y)); y += 28
        pygame.draw.line(self._screen, SIDEBAR_BORDER, (sb_x + 8, y), (screen_w - 8, y), 1)
        y += 10

        # Game state pill
        gs = snap["game_state"]
        gs_color = (ACCENT_CYAN if gs == "PLAYING"
                    else TEAM_A_COLOR if "TEAM_A" in gs
                    else TEAM_B_COLOR if "TEAM_B" in gs
                    else TEXT_DIM)
        _draw_pill(self._screen, gs_color, x0 + (SIDEBAR_W - 24) // 2, y + 10,
                   SIDEBAR_W - 28, 22, gs, self._font_xs, TEXT_PRIMARY)
        y += 32

        # Match time
        mins = int(match_time) // 60
        secs = int(match_time) % 60
        t_lbl = self._font_sm.render(f"Time  {mins:02d}:{secs:02d}", True, TEXT_SECONDARY)
        self._screen.blit(t_lbl, (x0, y)); y += 22

        pygame.draw.line(self._screen, SIDEBAR_BORDER, (sb_x + 8, y), (screen_w - 8, y), 1)
        y += 10

        # Robots
        hdr = self._font_xs.render("ROBOTS", True, TEXT_DIM)
        self._screen.blit(hdr, (x0, y)); y += 16
        for rigid_name in ("team_a_1", "team_a_2", "team_b_1", "team_b_2"):
            intent = snap["intents"].get(rigid_name, {})
            color = TEAM_A_LIGHT if rigid_name.startswith("team_a") else TEAM_B_LIGHT
            lbl = self._font_sm.render(_robot_sidebar_line(rigid_name, intent), True, color)
            self._screen.blit(lbl, (x0, y)); y += 16

        y += 8
        pygame.draw.line(self._screen, SIDEBAR_BORDER, (sb_x + 8, y), (screen_w - 8, y), 1)
        y += 10

        # Controls
        hdr2 = self._font_xs.render("CONTROLS", True, TEXT_DIM)
        self._screen.blit(hdr2, (x0, y)); y += 16
        controls = [
            ("[S]  Start / Resume",   TEXT_SECONDARY),
            ("[P]  Pause",            TEXT_SECONDARY),
            ("[R]  Reset",            TEXT_SECONDARY),
            ("[V]  Vel arrows",       ACCENT_CYAN if self.show_vel   else TEXT_DIM),
            ("[D]  Debug footprint",  ACCENT_CYAN if self.show_debug else TEXT_DIM),
            ("[Drag]  Move ball/robot", GOAL_COLOR),
            ("[Q]  Quit",             TEXT_DIM),
        ]
        for text, color in controls:
            lbl = self._font_xs.render(text, True, color)
            self._screen.blit(lbl, (x0, y)); y += 15


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(args=None):
    global FIELD_LEFT, FIELD_TOP

    rclpy.init(args=args)

    physics = PhysicsWorld()
    game_sm = GameStateMachine()
    shared = SharedState()
    node = SimNode(shared, game_sm, physics)

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("CyberDog Soccer - Pygame Sim")
    clock = pygame.time.Clock()
    screenshot_dir = os.environ.get("CYBERDOG_SIM_SCREENSHOT_DIR", "").strip()
    screenshot_interval = float(os.environ.get("CYBERDOG_SIM_SCREENSHOT_INTERVAL", "2.0"))
    last_screenshot_time = 0.0
    if screenshot_dir:
        Path(screenshot_dir).mkdir(parents=True, exist_ok=True)

    available_w = SCREEN_W - 2 * FIELD_MARGIN - SIDEBAR_W
    available_h = SCREEN_H - 2 * FIELD_MARGIN
    field_aspect = FIELD_LENGTH / FIELD_WIDTH
    field_w = available_w
    field_h = int(round(field_w / field_aspect))
    if field_h > available_h:
        field_h = available_h
        field_w = int(round(field_h * field_aspect))
    FIELD_LEFT = FIELD_MARGIN + max((available_w - field_w) // 2, 0)
    FIELD_TOP = FIELD_MARGIN + max((available_h - field_h) // 2, 0)
    renderer = Renderer(screen, field_w, field_h)

    physics_accumulator = 0.0
    running = True
    # Drag state: tracks whether we're dragging the ball or a robot
    _drag_active = False
    _drag_kind = ""       # "ball" or "robot"
    _drag_rigid = ""      # rigid_name when dragging a robot
    _drag_x = 0.0
    _drag_y = 0.0

    def _hit_ball(mx: int, my: int) -> bool:
        bsx, bsy = field_to_screen(physics.ball.x, physics.ball.y, field_w, field_h)
        ball_r_px = max(5, int(min(field_w / FIELD_LENGTH, field_h / FIELD_WIDTH) * BALL_RADIUS))
        return math.hypot(mx - bsx, my - bsy) < ball_r_px + 6

    def _hit_robot(mx: int, my: int) -> Optional[str]:
        r_px = max(8, int(min(field_w / FIELD_LENGTH, field_h / FIELD_WIDTH) * ROBOT_RADIUS))
        for spec in ROBOT_SPECS:
            robot = physics.robots.get(spec.rigid_name)
            if robot is None:
                continue
            rsx, rsy = field_to_screen(robot.x, robot.y, field_w, field_h)
            if math.hypot(mx - rsx, my - rsy) < r_px + 4:
                return spec.rigid_name
        return None

    while running and rclpy.ok():
        frame_dt = clock.tick(30) / 1000.0
        frame_dt = min(frame_dt, 0.1)  # cap to avoid spiral of death
        physics_accumulator += frame_dt

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_s:
                    ok, kickoff_key = game_sm.start()
                    if ok:
                        physics.reset(kickoff_key)
                elif event.key == pygame.K_p:
                    game_sm.pause()
                elif event.key == pygame.K_r:
                    kickoff_key = game_sm.reset()
                    physics.reset(kickoff_key)
                elif event.key == pygame.K_v:
                    renderer.show_vel = not renderer.show_vel
                elif event.key == pygame.K_d:
                    renderer.show_debug = not renderer.show_debug
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if _hit_ball(mx, my):
                    _drag_active = True
                    _drag_kind = "ball"
                    _drag_rigid = ""
                    _drag_x, _drag_y = physics.ball.x, physics.ball.y
                else:
                    hit = _hit_robot(mx, my)
                    if hit is not None:
                        _drag_active = True
                        _drag_kind = "robot"
                        _drag_rigid = hit
                        r = physics.robots[hit]
                        _drag_x, _drag_y = r.x, r.y
            elif event.type == pygame.MOUSEMOTION and _drag_active:
                mx, my = event.pos
                fx, fy = screen_to_field(mx, my, field_w, field_h)
                fx = max(-FIELD_WIDTH * 0.5, min(FIELD_WIDTH * 0.5, fx))
                fy = max(-FIELD_LENGTH * 0.5, min(FIELD_LENGTH * 0.5, fy))
                _drag_x, _drag_y = fx, fy
                if _drag_kind == "ball":
                    physics.ball.x = fx
                    physics.ball.y = fy
                    physics.ball.vx = 0.0
                    physics.ball.vy = 0.0
                elif _drag_kind == "robot" and _drag_rigid in physics.robots:
                    r = physics.robots[_drag_rigid]
                    r.x, r.y = fx, fy
                    r.cmd_vx = r.cmd_vy = r.cmd_wz = 0.0
                    r.vx_world = r.vy_world = 0.0
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                _drag_active = False

        # Physics steps
        while physics_accumulator >= PHYSICS_DT:
            # Apply incoming commands
            with shared.lock:
                cmds = dict(shared.robot_cmds)

            if game_sm.state in ACTIVE_STATES:
                for rigid_name, (vx, vy, wz) in cmds.items():
                    physics.set_robot_cmd(rigid_name, vx, vy, wz)
            else:
                physics.freeze_robots()

            events = physics.step()
            now = time.monotonic()
            actions = game_sm.tick(events, now)

            with shared.lock:
                shared.update_from_physics(physics, game_sm)

            for action in actions:
                if action.startswith("RESET_"):
                    physics.reset(action[len("RESET_"):])
                    physics.freeze_robots()

            physics_accumulator -= PHYSICS_DT

        # Render
        snap = shared.snapshot()
        drag_info = {
            "active": _drag_active,
            "kind": _drag_kind,
            "rigid_name": _drag_rigid,
            "x": _drag_x,
            "y": _drag_y,
        } if _drag_active else None
        renderer.draw(snap, game_sm.match_time, drag_info)
        pygame.display.flip()
        if screenshot_dir:
            now = time.monotonic()
            if now - last_screenshot_time >= max(screenshot_interval, 0.1):
                pygame.image.save(screen, str(Path(screenshot_dir) / "latest.png"))
                last_screenshot_time = now

    pygame.quit()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
