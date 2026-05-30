"""Map policy velocity actions into official Cyberdog locomotion commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .command_types import (
    DEFAULT_SOCCER_SERVO_ID,
    DEVELOPER_SERVO_MOTIONS,
    CmdSource,
    DeveloperServoMotion,
    MotionControlCommand,
    MotionMode,
    VelocityBounds,
)


@dataclass(frozen=True)
class CommandMapperConfig:
    """Velocity command limits and official command defaults."""

    developer_motion_id: int = DEFAULT_SOCCER_SERVO_ID
    max_lin_acc: float = 2.5
    max_lat_acc: float = 2.0
    max_yaw_acc: float = 6.0
    step_height: tuple[float, float] = (0.05, 0.05)
    duration: int = 200
    contact: int = 0x0F
    cmd_source: int = int(CmdSource.CYBERDOG2_LCM)
    use_speed_based_gait: bool = True
    profile_table: dict[int, DeveloperServoMotion] = field(default_factory=lambda: DEVELOPER_SERVO_MOTIONS)

    @property
    def default_motion(self) -> DeveloperServoMotion:
        return self.profile_table[self.developer_motion_id]

    @property
    def bounds(self) -> VelocityBounds:
        return self.default_motion.bounds

    def to_dict(self) -> dict[str, Any]:
        return {
            "developer_motion_id": self.developer_motion_id,
            "max_lin_acc": self.max_lin_acc,
            "max_lat_acc": self.max_lat_acc,
            "max_yaw_acc": self.max_yaw_acc,
            "step_height": list(self.step_height),
            "duration": self.duration,
            "contact": self.contact,
            "cmd_source": self.cmd_source,
            "use_speed_based_gait": self.use_speed_based_gait,
            "profiles": {key: value.to_dict() for key, value in self.profile_table.items()},
        }


class VelocityCommandMapper:
    """Scalar mapper used by runtime bridges and tests."""

    def __init__(self, cfg: CommandMapperConfig | None = None):
        self.cfg = cfg or CommandMapperConfig()
        self._prev_vel = [0.0, 0.0, 0.0]
        self.last_debug = self._empty_debug()

    def reset(self) -> None:
        self._prev_vel = [0.0, 0.0, 0.0]
        self.last_debug = self._empty_debug()

    def map_velocity(self, velocity: tuple[float, float, float], dt: float) -> MotionControlCommand:
        """Clip and rate-limit a body-frame velocity command."""

        raw = [float(value) for value in velocity]
        bounds = self.cfg.bounds
        clipped = [
            min(max(raw[0], bounds.lin_vel_x[0]), bounds.lin_vel_x[1]),
            min(max(raw[1], bounds.lin_vel_y[0]), bounds.lin_vel_y[1]),
            min(max(raw[2], bounds.ang_vel_z[0]), bounds.ang_vel_z[1]),
        ]
        previous = list(self._prev_vel)
        max_delta = [self.cfg.max_lin_acc * dt, self.cfg.max_lat_acc * dt, self.cfg.max_yaw_acc * dt]
        smoothed = []
        for idx, value in enumerate(clipped):
            delta = min(max(value - previous[idx], -max_delta[idx]), max_delta[idx])
            smoothed.append(previous[idx] + delta)
        self._prev_vel = smoothed
        profile = self.select_profile(tuple(smoothed))
        self.last_debug = {
            "raw_velocity": raw,
            "clipped_velocity": clipped,
            "previous_velocity": previous,
            "smoothed_velocity": [float(value) for value in smoothed],
            "max_delta": [float(value) for value in max_delta],
            "step_dt": float(dt),
            "developer_motion_id": profile.developer_motion_id,
            "profile": profile.name,
            "mode": profile.mode,
            "gait_id": profile.gait_id,
        }
        return MotionControlCommand(
            mode=profile.mode,
            gait_id=profile.gait_id,
            contact=self.cfg.contact,
            cmd_source=self.cfg.cmd_source,
            vel_des=tuple(float(value) for value in smoothed),
            step_height=self.cfg.step_height,
            duration=self.cfg.duration,
            motion_id=profile.refs_motion_id,
            metadata={"developer_motion_id": profile.developer_motion_id, "profile": profile.name},
        )

    def map_normalized_action(self, action: tuple[float, float, float], dt: float) -> MotionControlCommand:
        """Map normalized ``[-1, 1]`` action to a command."""

        scale = self.cfg.bounds.as_positive_scale()
        return self.map_velocity(
            (
                min(max(action[0], -1.0), 1.0) * scale[0],
                min(max(action[1], -1.0), 1.0) * scale[1],
                min(max(action[2], -1.0), 1.0) * scale[2],
            ),
            dt,
        )

    def _empty_debug(self) -> dict[str, Any]:
        return {
            "raw_velocity": [0.0, 0.0, 0.0],
            "clipped_velocity": [0.0, 0.0, 0.0],
            "previous_velocity": [0.0, 0.0, 0.0],
            "smoothed_velocity": [0.0, 0.0, 0.0],
            "max_delta": [0.0, 0.0, 0.0],
            "step_dt": 0.0,
            "developer_motion_id": self.cfg.default_motion.developer_motion_id,
            "profile": self.cfg.default_motion.name,
            "mode": self.cfg.default_motion.mode,
            "gait_id": self.cfg.default_motion.gait_id,
        }

    def select_profile(self, velocity: tuple[float, float, float]) -> DeveloperServoMotion:
        """Select a developer-guide servo profile from command magnitude."""

        if not self.cfg.use_speed_based_gait:
            return self.cfg.default_motion
        vx, vy, wz = abs(velocity[0]), abs(velocity[1]), abs(velocity[2])
        if vx <= 0.65 and vy <= 0.30 and wz <= 1.25:
            return self.cfg.profile_table[303]
        if vx <= 1.00 and vy <= 0.30 and wz <= 1.25:
            return self.cfg.profile_table[308]
        return self.cfg.profile_table[305]


class TensorCommandMapper:
    """Vectorized Torch mapper for IsaacLab environments."""

    def __init__(self, num_envs: int, device: str, cfg: CommandMapperConfig | None = None):
        import torch

        self.cfg = cfg or CommandMapperConfig()
        self.num_envs = num_envs
        self.device = device
        self._torch = torch
        self.previous_velocity = torch.zeros(num_envs, 3, device=device)
        self.last_velocity = torch.zeros(num_envs, 3, device=device)
        self.last_mode = torch.full((num_envs,), int(MotionMode.LOCOMOTION), dtype=torch.int16, device=device)
        self.last_gait_id = torch.full((num_envs,), self.cfg.default_motion.gait_id, dtype=torch.int16, device=device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.previous_velocity.zero_()
            self.last_velocity.zero_()
        else:
            self.previous_velocity[env_ids] = 0.0
            self.last_velocity[env_ids] = 0.0

    def map_normalized_action(self, actions, dt: float):
        torch = self._torch
        action = actions[:, :3].clamp(-1.0, 1.0)
        sx, sy, sw = self.cfg.bounds.as_positive_scale()
        scale = torch.tensor([sx, sy, sw], device=self.device, dtype=action.dtype).unsqueeze(0)
        return self.map_velocity(action * scale, dt)

    def map_velocity(self, velocity, dt: float):
        torch = self._torch
        bounds = self.cfg.bounds
        low = torch.tensor(
            [bounds.lin_vel_x[0], bounds.lin_vel_y[0], bounds.ang_vel_z[0]],
            device=self.device,
            dtype=velocity.dtype,
        ).unsqueeze(0)
        high = torch.tensor(
            [bounds.lin_vel_x[1], bounds.lin_vel_y[1], bounds.ang_vel_z[1]],
            device=self.device,
            dtype=velocity.dtype,
        ).unsqueeze(0)
        clipped = torch.maximum(torch.minimum(velocity[:, :3], high), low)
        max_delta = torch.tensor(
            [self.cfg.max_lin_acc * dt, self.cfg.max_lat_acc * dt, self.cfg.max_yaw_acc * dt],
            device=self.device,
            dtype=velocity.dtype,
        ).unsqueeze(0)
        delta = (clipped - self.previous_velocity).clamp(-max_delta, max_delta)
        self.last_velocity = self.previous_velocity + delta
        self.previous_velocity = self.last_velocity.clone()
        self._update_modes()
        return self.last_velocity

    def _update_modes(self) -> None:
        torch = self._torch
        speed = torch.abs(self.last_velocity)
        slow = (speed[:, 0] <= 0.65) & (speed[:, 1] <= 0.30) & (speed[:, 2] <= 1.25)
        medium = (speed[:, 0] <= 1.00) & (speed[:, 1] <= 0.30) & (speed[:, 2] <= 1.25)
        fast_gait = self.cfg.profile_table[305].gait_id
        medium_gait = self.cfg.profile_table[308].gait_id
        slow_gait = self.cfg.profile_table[303].gait_id
        gait = torch.full_like(self.last_gait_id, fast_gait)
        gait = torch.where(medium, torch.full_like(gait, medium_gait), gait)
        gait = torch.where(slow, torch.full_like(gait, slow_gait), gait)
        if not self.cfg.use_speed_based_gait:
            gait = torch.full_like(gait, self.cfg.default_motion.gait_id)
        self.last_mode.fill_(self.cfg.default_motion.mode)
        self.last_gait_id = gait

    def export_schema(self) -> dict[str, Any]:
        data = self.cfg.to_dict()
        data["command_fields"] = [
            "mode",
            "gait_id",
            "contact",
            "vel_des.x",
            "vel_des.y",
            "vel_des.yaw",
            "step_height.front",
            "step_height.back",
            "duration",
        ]
        return data
