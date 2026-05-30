"""Publisher-side adapter for official Cyberdog locomotion commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np

from cyberdog2_rl_lab.official import CommandMapperConfig, VelocityCommandMapper
from cyberdog2_rl_lab.official.lcm_bridge import RobotControlCmdSerializer


@dataclass(frozen=True)
class OfficialPublisherConfig:
    """Continuous command publication config."""

    publish_rate_hz: float = 50.0
    command_mapper: CommandMapperConfig = field(default_factory=CommandMapperConfig)
    channel_name: str = "robot_control_cmd"


class OfficialCommandPublisher:
    """Convert velocity commands into official packets and forward them to a sink."""

    def __init__(self, cfg: OfficialPublisherConfig, sink: Callable[[dict[str, Any]], None] | None = None):
        self.cfg = cfg
        self.mapper = VelocityCommandMapper(cfg.command_mapper)
        self.serializer = RobotControlCmdSerializer()
        self.sink = sink

    @property
    def step_dt(self) -> float:
        return 1.0 / self.cfg.publish_rate_hz

    def reset(self) -> None:
        self.mapper.reset()
        self.serializer.reset()

    def publish_velocity(self, velocity_command: np.ndarray | tuple[float, float, float]) -> dict[str, Any]:
        """Map a velocity command and optionally emit it through the configured sink."""

        vx, vy, wz = (float(value) for value in velocity_command)
        command = self.mapper.map_velocity((vx, vy, wz), self.step_dt)
        packet = self.serializer.pack(command).to_dict()
        packet["channel_name"] = self.cfg.channel_name
        packet["metadata"] = {**command.metadata, **self.mapper.last_debug}
        if self.sink is not None:
            self.sink(packet)
        return packet
