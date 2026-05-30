"""VRPN-side world-state normalization for Cyberdog soccer deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import math
import time

from cyberdog2_rl_lab.tasks.soccer.field_config import load_field_config


@dataclass(frozen=True)
class RigidBodyState:
    """Pose and planar velocity of a tracked rigid body in the world frame."""

    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wz: float = 0.0
    timestamp: float = 0.0

    @classmethod
    def from_pose(
        cls,
        position_xyz: tuple[float, float, float],
        quaternion_xyzw: tuple[float, float, float, float],
        previous: "RigidBodyState | None" = None,
        timestamp: float | None = None,
    ) -> "RigidBodyState":
        """Build a state from pose data and estimate planar velocities."""

        ts = time.time() if timestamp is None else float(timestamp)
        yaw = yaw_from_quaternion_xyzw(quaternion_xyzw)
        x, y, z = position_xyz
        if previous is None or previous.timestamp <= 0.0 or ts <= previous.timestamp:
            return cls(x=x, y=y, z=z, yaw=yaw, timestamp=ts)
        dt = max(ts - previous.timestamp, 1.0e-6)
        yaw_delta = wrap_to_pi(yaw - previous.yaw)
        return cls(
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            vx=(x - previous.x) / dt,
            vy=(y - previous.y) / dt,
            vz=(z - previous.z) / dt,
            wz=yaw_delta / dt,
            timestamp=ts,
        )


@dataclass(frozen=True)
class SoccerWorldState:
    """World state consumed by the deployment-time soccer policy runner."""

    blue_attacker: RigidBodyState
    blue_goalkeeper: RigidBodyState
    red_attacker: RigidBodyState
    red_goalkeeper: RigidBodyState
    ball: RigidBodyState
    blue_goal: RigidBodyState
    red_goal: RigidBodyState
    phase_play: float = 1.0
    timestamp: float = 0.0

    def as_dict(self) -> dict[str, RigidBodyState]:
        return {
            "blue_attacker": self.blue_attacker,
            "blue_goalkeeper": self.blue_goalkeeper,
            "red_attacker": self.red_attacker,
            "red_goalkeeper": self.red_goalkeeper,
            "ball": self.ball,
            "blue_goal": self.blue_goal,
            "red_goal": self.red_goal,
        }


@dataclass(frozen=True)
class VrpnStateReceiverConfig:
    """Mapping from VRPN rigid-body names to soccer semantic names."""

    rigid_body_names: Mapping[str, str]
    default_goals: Mapping[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "blue_goal": load_field_config().blue_goal,
            "red_goal": load_field_config().red_goal,
        }
    )


class VrpnStateReceiver:
    """Normalize snapshots from VRPN or any equivalent mocap source.

    The class intentionally accepts plain mappings so it can be used with a real
    VRPN client, recorded data, or a local adapter without changing the policy
    runner.
    """

    def __init__(self, cfg: VrpnStateReceiverConfig):
        self.cfg = cfg
        self._previous_states: dict[str, RigidBodyState] = {}

    def update_from_snapshot(
        self,
        snapshot: Mapping[str, tuple[tuple[float, float, float], tuple[float, float, float, float]] | RigidBodyState],
        timestamp: float | None = None,
    ) -> SoccerWorldState:
        """Convert a raw snapshot into a typed soccer world state."""

        ts = time.time() if timestamp is None else float(timestamp)
        semantic_states: dict[str, RigidBodyState] = {}
        for rigid_name, semantic_name in self.cfg.rigid_body_names.items():
            if rigid_name not in snapshot:
                continue
            sample = snapshot[rigid_name]
            if isinstance(sample, RigidBodyState):
                state = sample
            else:
                position, quaternion = sample
                previous = self._previous_states.get(semantic_name)
                state = RigidBodyState.from_pose(position, quaternion, previous=previous, timestamp=ts)
            semantic_states[semantic_name] = state
            self._previous_states[semantic_name] = state

        for goal_name, goal_pos in self.cfg.default_goals.items():
            semantic_states.setdefault(goal_name, RigidBodyState(x=goal_pos[0], y=goal_pos[1], z=goal_pos[2], timestamp=ts))

        required = [
            "blue_attacker",
            "blue_goalkeeper",
            "red_attacker",
            "red_goalkeeper",
            "ball",
            "blue_goal",
            "red_goal",
        ]
        missing = [name for name in required if name not in semantic_states]
        if missing:
            raise KeyError(f"Missing VRPN rigid bodies for soccer world state: {missing}")

        return SoccerWorldState(
            blue_attacker=semantic_states["blue_attacker"],
            blue_goalkeeper=semantic_states["blue_goalkeeper"],
            red_attacker=semantic_states["red_attacker"],
            red_goalkeeper=semantic_states["red_goalkeeper"],
            ball=semantic_states["ball"],
            blue_goal=semantic_states["blue_goal"],
            red_goal=semantic_states["red_goal"],
            timestamp=ts,
        )


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion_xyzw(quaternion_xyzw: tuple[float, float, float, float]) -> float:
    x, y, z, w = quaternion_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
