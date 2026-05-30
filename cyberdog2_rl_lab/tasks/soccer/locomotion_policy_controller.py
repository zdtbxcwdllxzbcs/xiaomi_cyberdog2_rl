"""Frozen locomotion policy bridge for the soccer task."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import torch
import yaml

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
else:
    Articulation = Any


LOW_LEVEL_OBSERVATION_TERMS = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)


class LocomotionPolicyController:
    """Run a trained Cyberdog2 velocity policy and apply joint targets."""

    def __init__(
        self,
        robots: Mapping[str, Articulation],
        num_envs: int,
        device: str,
        policy_path: str | None = None,
        deploy_cfg_path: str | None = None,
    ) -> None:
        self.robots = dict(robots)
        self.num_envs = num_envs
        self.device = device
        self.policy_path, self.deploy_cfg_path = resolve_locomotion_artifacts(policy_path, deploy_cfg_path)
        with open(self.deploy_cfg_path, encoding="utf-8") as file:
            self.deploy_cfg = yaml.safe_load(file)

        self.policy = torch.jit.load(str(self.policy_path), map_location=device)
        self.policy.eval()

        first_robot = next(iter(self.robots.values()))
        self.action_dim = int(first_robot.data.default_joint_pos.shape[1])
        self.last_actions = {
            agent: torch.zeros(num_envs, self.action_dim, device=device) for agent in self.robots
        }
        self._validate_deploy_cfg()
        self.action_scale = self._action_tensor("scale", self.action_dim)
        self.action_offset = self._action_tensor("offset", self.action_dim)
        self.action_clip = self._action_clip(self.action_dim)

    def reset(self, env_ids: torch.Tensor) -> None:
        for action in self.last_actions.values():
            action[env_ids] = 0.0

    def compute_joint_targets(self, velocity_commands: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the low-level policy for every robot and return joint targets."""

        joint_targets = {}
        with torch.inference_mode():
            for agent, robot in self.robots.items():
                obs = self.build_observation(agent, robot, velocity_commands[agent])
                action = self.policy(obs)
                if isinstance(action, tuple):
                    action = action[0]
                action = action.reshape(self.num_envs, -1)
                if action.shape[1] != self.action_dim:
                    raise RuntimeError(
                        f"Low-level policy for {agent} returned {action.shape[1]} actions, expected {self.action_dim}"
                    )
                action = torch.clamp(action, min=self.action_clip[:, 0], max=self.action_clip[:, 1])
                target = action * self.action_scale + self._action_offset_for_robot(robot)
                self.last_actions[agent] = action.detach().clone()
                joint_targets[agent] = target
        return joint_targets

    def apply_joint_targets(self, joint_targets: Mapping[str, torch.Tensor]) -> None:
        """Write previously computed joint targets to all robots."""

        for agent, target in joint_targets.items():
            self.robots[agent].set_joint_position_target(target)

    def apply(self, velocity_commands: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the low-level policy for every robot and write joint targets."""

        joint_targets = self.compute_joint_targets(velocity_commands)
        self.apply_joint_targets(joint_targets)
        return joint_targets

    def build_observation(self, agent: str, robot: Articulation, velocity_command: torch.Tensor) -> torch.Tensor:
        """Build the low-level observation in the exported locomotion order."""

        terms = {
            "base_lin_vel": robot.data.root_lin_vel_b,
            "base_ang_vel": robot.data.root_ang_vel_b,
            "projected_gravity": robot.data.projected_gravity_b,
            "velocity_commands": velocity_command,
            "joint_pos_rel": robot.data.joint_pos - robot.data.default_joint_pos,
            "joint_vel_rel": robot.data.joint_vel,
            "last_action": self.last_actions[agent],
        }
        obs_parts = []
        for name in LOW_LEVEL_OBSERVATION_TERMS:
            value = terms[name]
            scale = self._observation_scale(name, value.shape[1])
            clip_min, clip_max = self._observation_clip(name)
            obs_parts.append(torch.clamp(value * scale, min=clip_min, max=clip_max))
        return torch.cat(obs_parts, dim=1)

    def export_schema(self) -> dict:
        return {
            "policy_path": str(self.policy_path),
            "deploy_cfg_path": str(self.deploy_cfg_path),
            "observation_terms": list(LOW_LEVEL_OBSERVATION_TERMS),
            "action_dim": self.action_dim,
            "step_dt": self.deploy_cfg.get("step_dt"),
        }

    def _validate_deploy_cfg(self) -> None:
        observations = self.deploy_cfg.get("observations", {})
        missing = [name for name in LOW_LEVEL_OBSERVATION_TERMS if name not in observations]
        if missing:
            raise KeyError(f"Locomotion deploy cfg is missing observation terms: {missing}")
        actions = self.deploy_cfg.get("actions", {})
        if "JointPositionAction" not in actions:
            raise KeyError("Locomotion deploy cfg is missing actions.JointPositionAction")

    def _observation_scale(self, name: str, width: int) -> torch.Tensor:
        scale = self.deploy_cfg["observations"][name].get("scale", 1.0)
        if isinstance(scale, (float, int)):
            values = [float(scale)] * width
        else:
            values = [float(item) for item in scale]
        if len(values) != width:
            raise ValueError(f"Observation scale for {name!r} has {len(values)} values, expected {width}")
        return torch.tensor(values, device=self.device).reshape(1, width)

    def _observation_clip(self, name: str) -> tuple[float, float]:
        clip = self.deploy_cfg["observations"][name].get("clip")
        if clip is None:
            return -float("inf"), float("inf")
        return float(clip[0]), float(clip[1])

    def _action_tensor(self, key: str, width: int) -> torch.Tensor:
        values = self.deploy_cfg["actions"]["JointPositionAction"][key]
        if isinstance(values, (float, int)):
            values = [float(values)] * width
        else:
            values = [float(item) for item in values]
        if len(values) != width:
            raise ValueError(f"Action {key!r} has {len(values)} values, expected {width}")
        return torch.tensor(values, device=self.device).reshape(1, width)

    def _action_clip(self, width: int) -> torch.Tensor:
        values = self.deploy_cfg["actions"]["JointPositionAction"].get("clip")
        if values is None:
            return torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(width, 1)
        if len(values) != width:
            raise ValueError(f"Action clip has {len(values)} rows, expected {width}")
        return torch.tensor(values, device=self.device)

    def _action_offset_for_robot(self, robot: Articulation) -> torch.Tensor:
        if robot.data.default_joint_pos.shape[1] == self.action_offset.shape[1]:
            return robot.data.default_joint_pos
        return self.action_offset


def resolve_locomotion_artifacts(
    policy_path: str | None = None, deploy_cfg_path: str | None = None
) -> tuple[Path, Path]:
    """Resolve explicit or latest exported locomotion policy artifacts."""

    policy = _resolve_path(policy_path) if policy_path else None
    deploy = _resolve_path(deploy_cfg_path) if deploy_cfg_path else None
    if policy is None:
        policy = _discover_latest_policy()
    if deploy is None:
        deploy = policy.parent.parent / "params" / "deploy.yaml"
    if not policy.exists():
        raise FileNotFoundError(f"Locomotion policy not found: {policy}")
    if not deploy.exists():
        raise FileNotFoundError(f"Locomotion deploy cfg not found: {deploy}")
    return policy, deploy


def _resolve_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return _repo_root() / candidate


def _discover_latest_policy() -> Path:
    root = _repo_root() / "logs" / "rsl_rl" / "cyberdog2_velocity_flat"
    policies = sorted(root.glob("*/exported/policy.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not policies:
        raise FileNotFoundError(
            "No exported Cyberdog2 locomotion policy.pt found under logs/rsl_rl/cyberdog2_velocity_flat"
        )
    return policies[0]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]
