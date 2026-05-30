"""Deployment-time policy runner for official Cyberdog2 locomotion commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cyberdog2_rl_lab.official.deploy_schema import OFFICIAL_LOCOMOTION_OBSERVATION_FIELDS
from cyberdog2_rl_lab.runtime.official_cmd_publisher import OfficialCommandPublisher, OfficialPublisherConfig
from cyberdog2_rl_lab.utils.deploy_validation import (
    load_deploy_cfg,
    official_command_mapper_config_from_deploy_cfg,
    validate_official_locomotion_deploy_cfg,
)


@dataclass(frozen=True)
class OfficialLocomotionRuntimeState:
    """Body-frame state consumed by the official locomotion policy."""

    root_lin_vel_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    root_ang_vel_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    projected_gravity_b: tuple[float, float, float] = (0.0, 0.0, -1.0)
    target_velocity_b: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class OfficialLocomotionPolicyRunnerConfig:
    """Configuration for exported official locomotion policy inference."""

    model_path: str
    deploy_cfg_path: str
    backend: str = "onnx"
    publish_rate_hz: float | None = None
    channel_name: str = "robot_control_cmd"
    disable_speed_based_gait: bool = False
    sink: Any = None


class OfficialLocomotionObservationBuilder:
    """Construct official locomotion observations using the exported schema."""

    def __init__(self, deploy_cfg: dict[str, Any]):
        validate_official_locomotion_deploy_cfg(deploy_cfg)
        self.fields = tuple(deploy_cfg["observations"]["policy"])
        self.last_command = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        self.last_command.fill(0.0)
        self.last_action.fill(0.0)

    def build_observation(self, state: OfficialLocomotionRuntimeState | dict[str, Any]) -> np.ndarray:
        state = self._coerce_state(state)
        values = {
            "root_lin_vel_b.x": state.root_lin_vel_b[0],
            "root_lin_vel_b.y": state.root_lin_vel_b[1],
            "root_lin_vel_b.z": state.root_lin_vel_b[2],
            "root_ang_vel_b.x": state.root_ang_vel_b[0],
            "root_ang_vel_b.y": state.root_ang_vel_b[1],
            "root_ang_vel_b.z": state.root_ang_vel_b[2],
            "projected_gravity_b.x": state.projected_gravity_b[0],
            "projected_gravity_b.y": state.projected_gravity_b[1],
            "projected_gravity_b.z": state.projected_gravity_b[2],
            "target_vx": state.target_velocity_b[0],
            "target_vy": state.target_velocity_b[1],
            "target_wz": state.target_velocity_b[2],
            "cmd_vx": self.last_command[0],
            "cmd_vy": self.last_command[1],
            "cmd_wz": self.last_command[2],
            "action_vx": self.last_action[0],
            "action_vy": self.last_action[1],
            "action_wz": self.last_action[2],
        }
        obs = np.asarray([values[field] for field in self.fields], dtype=np.float32)
        if obs.shape[0] != len(OFFICIAL_LOCOMOTION_OBSERVATION_FIELDS):
            raise RuntimeError(f"Official locomotion observation size mismatch: got {obs.shape[0]}, expected 18")
        return obs

    def update_last(self, command: np.ndarray, action: np.ndarray) -> None:
        self.last_command = np.asarray(command, dtype=np.float32).reshape(3)
        self.last_action = np.asarray(action, dtype=np.float32).reshape(3)

    def _coerce_state(self, state: OfficialLocomotionRuntimeState | dict[str, Any]) -> OfficialLocomotionRuntimeState:
        if isinstance(state, OfficialLocomotionRuntimeState):
            return state
        return OfficialLocomotionRuntimeState(
            root_lin_vel_b=tuple(float(value) for value in state.get("root_lin_vel_b", (0.0, 0.0, 0.0))),
            root_ang_vel_b=tuple(float(value) for value in state.get("root_ang_vel_b", (0.0, 0.0, 0.0))),
            projected_gravity_b=tuple(float(value) for value in state.get("projected_gravity_b", (0.0, 0.0, -1.0))),
            target_velocity_b=tuple(float(value) for value in state.get("target_velocity_b", (0.0, 0.0, 0.0))),
        )


class OfficialLocomotionPolicyRunner:
    """Run exported official locomotion policies and emit official command packets."""

    def __init__(self, cfg: OfficialLocomotionPolicyRunnerConfig):
        self.cfg = cfg
        self.deploy_cfg = load_deploy_cfg(cfg.deploy_cfg_path)
        validate_official_locomotion_deploy_cfg(self.deploy_cfg)
        mapper_cfg = official_command_mapper_config_from_deploy_cfg(self.deploy_cfg)
        if cfg.disable_speed_based_gait:
            mapper_cfg = mapper_cfg.__class__(
                developer_motion_id=mapper_cfg.developer_motion_id,
                max_lin_acc=mapper_cfg.max_lin_acc,
                max_lat_acc=mapper_cfg.max_lat_acc,
                max_yaw_acc=mapper_cfg.max_yaw_acc,
                step_height=mapper_cfg.step_height,
                duration=mapper_cfg.duration,
                contact=mapper_cfg.contact,
                cmd_source=mapper_cfg.cmd_source,
                use_speed_based_gait=False,
                profile_table=mapper_cfg.profile_table,
            )
        publish_rate_hz = cfg.publish_rate_hz or 1.0 / float(self.deploy_cfg["step_dt"])
        publisher_cfg = OfficialPublisherConfig(
            publish_rate_hz=publish_rate_hz,
            command_mapper=mapper_cfg,
            channel_name=cfg.channel_name,
        )
        self.observation_builder = OfficialLocomotionObservationBuilder(self.deploy_cfg)
        self.publisher = OfficialCommandPublisher(publisher_cfg, sink=cfg.sink)
        self._session = None
        self._model = None
        self._load_backend()

    def reset(self) -> None:
        self.observation_builder.reset()
        self.publisher.reset()

    def infer(self, state: OfficialLocomotionRuntimeState | dict[str, Any]) -> dict[str, Any]:
        obs = self.observation_builder.build_observation(state)
        raw_action = self._run_policy(obs)
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if raw_action.shape[0] != 3:
            raise RuntimeError(f"Official locomotion policy must output 3 actions, got {raw_action.shape[0]}")
        normalized_action = np.clip(raw_action, -1.0, 1.0)
        scale = np.asarray(self.publisher.mapper.cfg.bounds.as_positive_scale(), dtype=np.float32)
        velocity_command = normalized_action * scale
        packet = self.publisher.publish_velocity(velocity_command)
        self.observation_builder.update_last(
            np.asarray(packet["metadata"]["smoothed_velocity"], dtype=np.float32),
            normalized_action,
        )
        packet["metadata"]["raw_policy_action"] = raw_action.tolist()
        packet["metadata"]["normalized_policy_action"] = normalized_action.tolist()
        packet["metadata"]["observation_fields"] = list(self.observation_builder.fields)
        return packet

    def export_runtime_schema(self) -> dict[str, Any]:
        return {
            "policy_layout": "single_agent_official_locomotion",
            "observation_fields": list(self.observation_builder.fields),
            "action_fields": ["vx", "vy", "wz"],
            "action_scale": list(self.publisher.mapper.cfg.bounds.as_positive_scale()),
            "step_dt": self.publisher.step_dt,
            "channel_name": self.publisher.cfg.channel_name,
        }

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

    def _run_policy(self, obs: np.ndarray) -> np.ndarray:
        if self.cfg.backend == "onnx":
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: obs[None, :]})[0]
            return np.asarray(output, dtype=np.float32).reshape(-1)

        import torch

        with torch.inference_mode():
            tensor = torch.from_numpy(obs[None, :])
            output = self._model(tensor)
        return np.asarray(output.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
