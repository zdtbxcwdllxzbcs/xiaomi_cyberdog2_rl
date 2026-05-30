"""Deployment-time observation builder and policy runner for soccer.

Aligned with Flamez 12-D observation schema (StrikerPolicyNode._observation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cyberdog2_rl_lab.tasks.soccer.field_config import load_field_config
from cyberdog2_rl_lab.tasks.soccer.observation_schema import (
    BALL_VEL_OBS_LIMIT,
    MAX_ANGULAR_SPD,
    MAX_LINEAR_SPD,
    OPPONENTS,
    ROLE,
    TEAM_SIGN,
    TEAMMATE,
)

from .vrpn_state_receiver import RigidBodyState, SoccerWorldState

AGENT_ORDER = ["blue_attacker", "blue_goalkeeper", "red_attacker", "red_goalkeeper"]


@dataclass(frozen=True)
class SoccerPolicyRunnerConfig:
    model_path: str
    backend: str = "onnx"
    field_config_path: str | None = None
    action_scale: tuple[float, float, float] = (1.2, 1.2, 3.0)
    agent_order: tuple[str, ...] = tuple(AGENT_ORDER)


class SoccerObservationBuilder:
    """12-D Flamez-aligned observation builder for single-agent deployment."""

    def __init__(
        self,
        field_config_path: str | None = None,
        action_scale: tuple[float, float, float] = (1.2, 1.2, 3.0),
    ):
        field_cfg = load_field_config(field_config_path)
        self.half_width = 0.5 * field_cfg.x_width
        self.half_length = 0.5 * field_cfg.y_length
        self.action_scale = np.asarray(action_scale, dtype=np.float32)
        self.ball_vel_clip = BALL_VEL_OBS_LIMIT
        self.max_linear = MAX_LINEAR_SPD

    def build_agent_observation(self, world: SoccerWorldState, agent: str) -> np.ndarray:
        states = world.as_dict()
        self_state = states[agent]
        opponent_states = [states[name] for name in OPPONENTS[agent]]
        sign = np.float32(TEAM_SIGN[agent])
        attack_goal = world.red_goal if sign > 0.0 else world.blue_goal

        canonical_yaw = sign * self_state.yaw
        ball_y_canon = sign * world.ball.y
        ball_vy_canon = sign * world.ball.vy
        goal_y_canon = sign * attack_goal.y

        closest_opp = min(opponent_states, key=lambda s: np.hypot(s.x - attack_goal.x, s.y - attack_goal.y))
        opp_y_canon = sign * closest_opp.y

        obs = np.array([
            self_state.x / self.half_width,
            sign * self_state.y / self.half_length,
            np.sin(canonical_yaw),
            np.cos(canonical_yaw),
            world.ball.x / self.half_width,
            ball_y_canon / self.half_length,
            np.clip(world.ball.vx / self.max_linear, -self.ball_vel_clip, self.ball_vel_clip),
            np.clip(ball_vy_canon / self.max_linear, -self.ball_vel_clip, self.ball_vel_clip),
            attack_goal.x / self.half_width,
            goal_y_canon / self.half_length,
            closest_opp.x / self.half_width,
            opp_y_canon / self.half_length,
        ], dtype=np.float32)
        return obs

    def build_joint_observation(self, world: SoccerWorldState,
                                 agent_order: tuple[str, ...] = tuple(AGENT_ORDER)) -> np.ndarray:
        return np.concatenate([self.build_agent_observation(world, agent) for agent in agent_order], axis=0)


class SoccerPolicyRunner:
    """Run exported ONNX or TorchScript soccer policies on VRPN world states."""

    def __init__(self, cfg: SoccerPolicyRunnerConfig):
        self.cfg = cfg
        self.observation_builder = SoccerObservationBuilder(cfg.field_config_path, cfg.action_scale)
        self._session = None
        self._model = None
        self._load_backend()

    def _load_backend(self) -> None:
        model_path = Path(self.cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if self.cfg.backend == "onnx":
            import onnxruntime as ort
            self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        elif self.cfg.backend == "torchscript":
            import torch
            self._model = torch.jit.load(str(model_path), map_location="cpu")
            self._model.eval()
        else:
            raise ValueError(f"Unsupported policy backend: {self.cfg.backend}")

    def infer(self, world: SoccerWorldState) -> dict[str, np.ndarray]:
        num_agents = len(self.cfg.agent_order)
        if num_agents == 1:
            obs = self.observation_builder.build_agent_observation(world, self.cfg.agent_order[0])
        else:
            obs = self.observation_builder.build_joint_observation(world, self.cfg.agent_order)

        if self.cfg.backend == "onnx":
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: obs[None, :]})[0]
            action = np.asarray(output, dtype=np.float32).reshape(-1)
        else:
            import torch
            with torch.inference_mode():
                tensor = torch.from_numpy(obs[None, :])
                output = self._model(tensor)
            action = np.asarray(output.detach().cpu().numpy(), dtype=np.float32).reshape(-1)

        commands = {}
        scale = np.asarray(self.cfg.action_scale, dtype=np.float32)
        for index, agent in enumerate(self.cfg.agent_order):
            raw = action[index * 3 : (index + 1) * 3]
            commands[agent] = np.clip(raw, -1.0, 1.0) * scale
        return commands
