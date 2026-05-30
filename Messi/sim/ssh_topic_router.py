"""SSH router for streaming remote ROS2 soccer topics as JSON snapshots.

The local tactical board uses this module to avoid joining the remote DDS
domain directly. It starts a tiny Python/rclpy program on the robot over SSH,
subscribes to selected pose/twist topics there, and reads newline-delimited
JSON snapshots from stdout.
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


DEFAULT_REMOTE_SETUPS = (
    "/opt/ros2/cyberdog/setup.bash",
    "/opt/ros2/galactic/setup.bash",
)

DEFAULT_SLOT_RIGIDS = {
    "team_a_1": "self",
    "team_a_2": "teammate",
    "team_b_1": "opponent_1",
    "team_b_2": "opponent_2",
    "ball": "ball",
}


REMOTE_ROUTER_SCRIPT = r'''
import argparse
import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Path as NavPath
from protocol.msg import MotionServoCmd
from rclpy.node import Node
from std_msgs.msg import String


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class TopicRouter(Node):
    def __init__(self, specs, hz, game_state_topic, score_topic,
                 path_topics=None, state_topics=None, cmd_topics=None):
        super().__init__("ssh_topic_router")
        self._specs = specs
        self._states = {}
        self._game_state = "REMOTE"
        self._score = {"team_a": 0, "team_b": 0}
        self._last_error = ""
        self._paths = {}
        self._role_states = {}
        self._cmd_vels = {}

        for spec in specs:
            slot = spec["slot"]
            self._states[slot] = {
                "kind": spec.get("kind", "robot"),
                "pose_topic": spec["pose_topic"],
                "twist_topic": spec.get("twist_topic"),
                "pose": None,
                "twist": None,
                "pose_time": None,
                "twist_time": None,
            }
            self.create_subscription(
                PoseStamped,
                spec["pose_topic"],
                self._make_pose_callback(slot),
                10,
            )
            twist_topic = spec.get("twist_topic")
            if twist_topic:
                self.create_subscription(
                    TwistStamped,
                    twist_topic,
                    self._make_twist_callback(slot),
                    10,
                )

        if game_state_topic:
            self.create_subscription(String, game_state_topic, self._game_state_callback, 10)
        if score_topic:
            self.create_subscription(String, score_topic, self._score_callback, 10)

        for name, topic in (path_topics or {}).items():
            self.create_subscription(NavPath, topic, self._make_path_callback(name), 10)
        for name, topic in (state_topics or {}).items():
            self.create_subscription(String, topic, self._make_state_callback(name), 10)
        for name, topic in (cmd_topics or {}).items():
            self.create_subscription(MotionServoCmd, topic, self._make_cmd_callback(name), 10)

        self.create_timer(1.0 / max(float(hz), 1.0), self._publish_snapshot)

    def _make_pose_callback(self, slot):
        def callback(msg):
            self._states[slot]["pose"] = msg
            self._states[slot]["pose_time"] = time.monotonic()

        return callback

    def _make_twist_callback(self, slot):
        def callback(msg):
            self._states[slot]["twist"] = msg
            self._states[slot]["twist_time"] = time.monotonic()

        return callback

    def _make_path_callback(self, name):
        def callback(msg):
            self._paths[name] = [
                {"x": float(p.pose.position.x), "y": float(p.pose.position.y)}
                for p in msg.poses
            ]
        return callback

    def _make_state_callback(self, name):
        def callback(msg):
            self._role_states[name] = msg.data
        return callback

    def _make_cmd_callback(self, name):
        def callback(msg):
            vel = list(msg.vel_des) if hasattr(msg, "vel_des") else [0.0, 0.0, 0.0]
            self._cmd_vels[name] = {
                "vx": float(vel[0]) if len(vel) > 0 else 0.0,
                "vy": float(vel[1]) if len(vel) > 1 else 0.0,
                "wz": float(vel[2]) if len(vel) > 2 else 0.0,
            }
        return callback

    def _game_state_callback(self, msg):
        self._game_state = msg.data or "REMOTE"

    def _score_callback(self, msg):
        text = msg.data.strip()
        if ":" not in text:
            return
        left, right = text.split(":", 1)
        try:
            self._score = {"team_a": int(left), "team_b": int(right)}
        except ValueError:
            pass

    def _publish_snapshot(self):
        now = time.monotonic()
        robots = {}
        ball = None
        status = {}

        for slot, state in self._states.items():
            pose = state["pose"]
            twist = state["twist"]
            pose_age = None if state["pose_time"] is None else now - state["pose_time"]
            twist_age = None if state["twist_time"] is None else now - state["twist_time"]
            status[slot] = {
                "pose_topic": state["pose_topic"],
                "twist_topic": state["twist_topic"],
                "pose_seen": pose is not None,
                "twist_seen": twist is not None,
                "pose_age": pose_age,
                "twist_age": twist_age,
            }
            if pose is None:
                continue

            item = {
                "x": float(pose.pose.position.x),
                "y": float(pose.pose.position.y),
                "yaw": float(yaw_from_quaternion(pose.pose.orientation)),
            }
            if twist is not None:
                item["vx"] = float(twist.twist.linear.x)
                item["vy"] = float(twist.twist.linear.y)
                item["wz"] = float(twist.twist.angular.z)

            if state["kind"] == "ball":
                ball = item
            else:
                robots[slot] = item

        payload = {
            "type": "snapshot",
            "stamp": time.time(),
            "game_state": self._game_state,
            "score": self._score,
            "robots": robots,
            "ball": ball,
            "status": status,
            "paths": dict(self._paths),
            "role_states": dict(self._role_states),
            "cmd_vels": dict(self._cmd_vels),
        }
        sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--hz", type=float, default=20.0)
    args = parser.parse_args()
    config = json.loads(args.config_json)

    rclpy.init(args=None)
    node = TopicRouter(
        specs=config["topics"],
        hz=args.hz,
        game_state_topic=config.get("game_state_topic"),
        score_topic=config.get("score_topic"),
        path_topics=config.get("path_topics"),
        state_topics=config.get("state_topics"),
        cmd_topics=config.get("cmd_topics"),
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
'''


@dataclass(frozen=True)
class TopicSpec:
    """One remote ROS2 topic pair routed to one local board slot."""

    slot: str
    pose_topic: str
    twist_topic: str | None = None
    kind: str = "robot"

    def to_json(self) -> dict:
        return {
            "slot": self.slot,
            "pose_topic": self.pose_topic,
            "twist_topic": self.twist_topic,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class RemoteRouterConfig:
    """Configuration needed to start the remote SSH router."""

    host: str = "10.0.0.54"
    user: str = "mi"
    password: str | None = "123"
    remote_workspace: str = "~/cyberdog_soccer"
    remote_setups: tuple[str, ...] = DEFAULT_REMOTE_SETUPS
    strict_host_key_checking: bool = False
    hz: float = 20.0
    game_state_topic: str | None = None
    score_topic: str | None = None
    topics: tuple[TopicSpec, ...] = field(default_factory=tuple)
    path_topics: dict[str, str] = field(default_factory=dict)
    state_topics: dict[str, str] = field(default_factory=dict)
    cmd_topics: dict[str, str] = field(default_factory=dict)

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


def load_default_slot_mapping(config_path: str | os.PathLike) -> dict[str, str]:
    """Load renderer-slot to remote-rigid defaults from a soccer config file."""

    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    rigids = config.get("rigids", {})
    mapping = {}
    for slot, semantic_name in DEFAULT_SLOT_RIGIDS.items():
        rigid = rigids.get(semantic_name)
        if rigid:
            mapping[slot] = str(rigid)
    return mapping


def apply_mapping_overrides(mapping: dict[str, str], overrides: Iterable[str]) -> dict[str, str]:
    """Apply CLI overrides such as ``team_a_1=SoTaGo1`` or ``ball=/vrpn/ball/pose``."""

    updated = dict(mapping)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected SLOT=RIGID_OR_TOPIC, got {override!r}")
        slot, rigid = override.split("=", 1)
        slot = slot.strip()
        rigid = rigid.strip()
        if not slot or not rigid:
            raise ValueError(f"Expected SLOT=RIGID_OR_TOPIC, got {override!r}")
        updated[slot] = rigid
    return updated


def topic_specs_from_mapping(mapping: dict[str, str]) -> tuple[TopicSpec, ...]:
    """Convert slot-to-rigid mapping into typed pose/twist subscriptions."""

    specs = []
    for slot, rigid_or_topic in mapping.items():
        pose_topic, twist_topic = _pose_and_twist_topics(rigid_or_topic)
        specs.append(
            TopicSpec(
                slot=slot,
                pose_topic=pose_topic,
                twist_topic=twist_topic,
                kind="ball" if slot == "ball" else "robot",
            )
        )
    return tuple(specs)


def _pose_and_twist_topics(rigid_or_topic: str) -> tuple[str, str | None]:
    value = rigid_or_topic.strip()
    if value.startswith("/"):
        pose_topic = value
        if pose_topic.endswith("/pose"):
            return pose_topic, pose_topic[: -len("/pose")] + "/twist"
        return pose_topic, None
    rigid = value.strip("/")
    return f"/vrpn/{rigid}/pose", f"/vrpn/{rigid}/twist"


def build_remote_payload(
    topics: Iterable[TopicSpec],
    game_state_topic: str | None = None,
    score_topic: str | None = None,
    path_topics: dict[str, str] | None = None,
    state_topics: dict[str, str] | None = None,
    cmd_topics: dict[str, str] | None = None,
) -> str:
    payload = {
        "topics": [topic.to_json() for topic in topics],
        "game_state_topic": game_state_topic,
        "score_topic": score_topic,
        "path_topics": path_topics or {},
        "state_topics": state_topics or {},
        "cmd_topics": cmd_topics or {},
    }
    return json.dumps(payload, separators=(",", ":"))


def build_remote_shell_command(config: RemoteRouterConfig) -> str:
    """Build the remote shell command that starts Python and reads script stdin."""

    setup_commands = [
        f"if [ -f {_remote_path_expr(path)} ]; then source {_remote_path_expr(path)} >/dev/null 2>&1; fi"
        for path in config.remote_setups
    ]
    payload = build_remote_payload(
        config.topics,
        config.game_state_topic,
        config.score_topic,
        config.path_topics,
        config.state_topics,
        config.cmd_topics,
    )
    python_args = " ".join(
        shlex.quote(arg)
        for arg in (
            "--config-json",
            payload,
            "--hz",
            str(config.hz),
        )
    )
    commands = [
        "set -e",
        *setup_commands,
        f"cd {_remote_path_expr(config.remote_workspace)}",
        f"exec python3 -u - {python_args}",
    ]
    return "; ".join(commands)


def build_ssh_command(config: RemoteRouterConfig) -> list[str]:
    """Build the local ssh command used to launch the remote router."""

    command = []
    if config.password:
        sshpass = shutil.which("sshpass")
        if not sshpass:
            raise RuntimeError("sshpass is required for password SSH but was not found")
        command.extend([sshpass, "-p", config.password])
    command.append("ssh")
    if not config.strict_host_key_checking:
        command.extend(
            [
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
            ]
        )
    command.extend(
        [
            config.destination,
            "bash -lc " + shlex.quote(build_remote_shell_command(config)),
        ]
    )
    return command


def _remote_path_expr(path: str) -> str:
    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        rest = path[2:]
        return "$HOME/" + shlex.quote(rest)
    return shlex.quote(path)


class SshRosTopicRouter:
    """Manage the remote SSH process and expose decoded JSON snapshots."""

    def __init__(self, config: RemoteRouterConfig):
        if not config.topics:
            raise ValueError("At least one TopicSpec is required")
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._snapshots: queue.Queue[dict] = queue.Queue()
        self._stderr: queue.Queue[str] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._started_at = 0.0

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def returncode(self) -> int | None:
        return None if self._proc is None else self._proc.poll()

    def start(self) -> None:
        if self.running:
            return
        self.stop()
        self._started_at = time.monotonic()
        self._proc = subprocess.Popen(
            build_ssh_command(self.config),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdin is not None
        self._proc.stdin.write(REMOTE_ROUTER_SCRIPT)
        self._proc.stdin.close()
        self._threads = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2.0)
        self._proc = None

    def latest_snapshot(self) -> dict | None:
        latest = None
        while True:
            try:
                latest = self._snapshots.get_nowait()
            except queue.Empty:
                return latest

    def latest_stderr(self, limit: int = 6) -> list[str]:
        lines = []
        while True:
            try:
                lines.append(self._stderr.get_nowait())
            except queue.Empty:
                break
        return lines[-limit:]

    def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self._stderr.put(f"non-json stdout: {line[:160]}")
                continue
            self._snapshots.put(payload)

    def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            line = line.strip()
            if line:
                self._stderr.put(line)

