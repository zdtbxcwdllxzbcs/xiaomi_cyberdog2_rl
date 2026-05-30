"""Lightweight serializers for official command bridge integration.

This module deliberately does not import generated LCM classes. The Windows
IsaacLab training loop can use these plain dictionaries, while a later Linux
runtime bridge can adapt the same schema into generated ``robot_control_cmd``
or ROS message objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .command_types import MotionControlCommand


@dataclass
class RobotControlCmdPacket:
    """Serializable packet matching ``robot_control_cmd_lcmt`` fields."""

    mode: int
    gait_id: int
    contact: int
    life_count: int
    vel_des: list[float]
    rpy_des: list[float]
    pos_des: list[float]
    acc_des: list[float]
    ctrl_point: list[float]
    foot_pose: list[float]
    step_height: list[float]
    value: int
    duration: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RobotControlCmdSerializer:
    """Create monotonically counted command packets."""

    def __init__(self):
        self.life_count = 0

    def reset(self) -> None:
        self.life_count = 0

    def pack(self, command: MotionControlCommand) -> RobotControlCmdPacket:
        self.life_count = (self.life_count + 1) % 128
        data = command.to_robot_control_lcm_dict(life_count=self.life_count)
        return RobotControlCmdPacket(**data)
