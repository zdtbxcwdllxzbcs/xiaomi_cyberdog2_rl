"""Deployment schema helpers that do not depend on IsaacLab."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from cyberdog2_rl_lab.official.command_mapper import CommandMapperConfig
from cyberdog2_rl_lab.official.command_types import DeveloperServoMotion, VelocityBounds
from cyberdog2_rl_lab.official.deploy_schema import (
    OFFICIAL_LOCOMOTION_ACTION_FIELDS,
    OFFICIAL_LOCOMOTION_OBSERVATION_FIELDS,
)


DEPLOY_SCHEMA_VERSION = 1


class DeployConfigError(ValueError):
    """Raised when deployment artifacts do not match the expected runtime schema."""


def load_deploy_cfg(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise DeployConfigError(f"Deploy config must be a mapping: {path}")
    return cfg


def with_deployment_identity(
    cfg: dict[str, Any],
    *,
    task_id: str | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    cfg = dict(cfg)
    identity = {
        "schema_version": DEPLOY_SCHEMA_VERSION,
        "task_id": task_id,
        "task": cfg.get("task"),
        "workflow": cfg.get("workflow"),
        "step_dt": cfg.get("step_dt"),
        "observation_dim": infer_policy_observation_dim(cfg),
        "action_dim": infer_action_dim(cfg),
        "normalized_action_scale": infer_normalized_action_scale(cfg),
    }
    if checkpoint_path:
        identity["checkpoint_path"] = str(Path(checkpoint_path))
    cfg["deployment"] = identity
    return cfg


def infer_policy_observation_dim(cfg: dict[str, Any]) -> int | None:
    policy_obs = cfg.get("observations", {}).get("policy")
    if isinstance(policy_obs, list):
        policy_runtime = cfg.get("policy_runtime", {})
        if isinstance(policy_runtime, dict) and policy_runtime.get("layout") == "joint_multi_agent_flattened":
            agent_order = policy_runtime.get("observation_agent_order") or cfg.get("possible_agents") or []
            return len(policy_obs) * len(agent_order)
        return len(policy_obs)
    if isinstance(policy_obs, dict):
        total = 0
        for term_cfg in policy_obs.values():
            if not isinstance(term_cfg, dict):
                continue
            scale = term_cfg.get("scale")
            dim = len(scale) if isinstance(scale, list) else term_cfg.get("dim")
            if dim is None:
                return None
            history = term_cfg.get("history_length") or 1
            total += int(dim) * int(history)
        return total
    return None


def infer_action_dim(cfg: dict[str, Any]) -> int | None:
    actions = cfg.get("actions")
    if not isinstance(actions, dict):
        return None
    if "normalized_velocity" in actions and isinstance(actions["normalized_velocity"], list):
        return len(actions["normalized_velocity"])
    if "flatdim" in actions:
        return int(actions["flatdim"])
    total = 0
    for action_cfg in actions.values():
        if isinstance(action_cfg, list):
            total += len(action_cfg)
        elif isinstance(action_cfg, dict):
            scale = action_cfg.get("scale")
            if isinstance(scale, list):
                total += len(scale)
            elif action_cfg.get("dim") is not None:
                total += int(action_cfg["dim"])
            else:
                return None
        else:
            return None
    return total


def infer_normalized_action_scale(cfg: dict[str, Any]) -> list[float] | None:
    mapper = cfg.get("command_mapper")
    if not isinstance(mapper, dict):
        return None
    developer_motion_id = mapper.get("developer_motion_id")
    profiles = mapper.get("profiles")
    if developer_motion_id is None or not isinstance(profiles, dict):
        return None
    profile = profiles.get(developer_motion_id) or profiles.get(str(developer_motion_id))
    if not isinstance(profile, dict):
        return None
    bounds = profile.get("bounds")
    if not isinstance(bounds, dict):
        return None
    try:
        return [
            _positive_scale(bounds["lin_vel_x"]),
            _positive_scale(bounds["lin_vel_y"]),
            _positive_scale(bounds["ang_vel_z"]),
        ]
    except (KeyError, TypeError, ValueError):
        return None


def validate_official_locomotion_deploy_cfg(cfg: dict[str, Any]) -> None:
    if cfg.get("task") != "official_locomotion":
        raise DeployConfigError(
            "Official locomotion bridge requires deploy.yaml with task: official_locomotion. "
            f"Got task: {cfg.get('task')!r}. Did you export Cyberdog2-Velocity-Flat by mistake?"
        )

    observations = cfg.get("observations", {}).get("policy")
    if observations != OFFICIAL_LOCOMOTION_OBSERVATION_FIELDS:
        raise DeployConfigError(
            "Official locomotion policy observation schema mismatch: expected "
            f"{len(OFFICIAL_LOCOMOTION_OBSERVATION_FIELDS)} fields, got {observations!r}."
        )

    actions = cfg.get("actions", {}).get("normalized_velocity")
    if actions != OFFICIAL_LOCOMOTION_ACTION_FIELDS:
        raise DeployConfigError(
            f"Official locomotion action schema mismatch: expected {OFFICIAL_LOCOMOTION_ACTION_FIELDS}, got {actions!r}."
        )

    step_dt = cfg.get("step_dt")
    if not isinstance(step_dt, (int, float)) or step_dt <= 0.0:
        raise DeployConfigError(f"Official locomotion deploy.yaml must contain positive step_dt, got {step_dt!r}.")

    deployment = cfg.get("deployment", {})
    if isinstance(deployment, dict):
        obs_dim = deployment.get("observation_dim")
        action_dim = deployment.get("action_dim")
        if obs_dim is not None and int(obs_dim) != len(OFFICIAL_LOCOMOTION_OBSERVATION_FIELDS):
            raise DeployConfigError(f"Deployment observation_dim mismatch: expected 18, got {obs_dim!r}.")
        if action_dim is not None and int(action_dim) != len(OFFICIAL_LOCOMOTION_ACTION_FIELDS):
            raise DeployConfigError(f"Deployment action_dim mismatch: expected 3, got {action_dim!r}.")

    if not isinstance(cfg.get("command_mapper"), dict):
        raise DeployConfigError("Official locomotion deploy.yaml is missing command_mapper.")


def official_command_mapper_config_from_deploy_cfg(cfg: dict[str, Any]) -> CommandMapperConfig:
    validate_official_locomotion_deploy_cfg(cfg)
    data = cfg["command_mapper"]
    profiles = {
        int(key): _developer_servo_motion_from_dict(value)
        for key, value in data.get("profiles", {}).items()
        if isinstance(value, dict)
    }
    return CommandMapperConfig(
        developer_motion_id=int(data.get("developer_motion_id", 305)),
        max_lin_acc=float(data.get("max_lin_acc", 2.5)),
        max_lat_acc=float(data.get("max_lat_acc", 2.0)),
        max_yaw_acc=float(data.get("max_yaw_acc", 6.0)),
        step_height=tuple(float(value) for value in data.get("step_height", (0.05, 0.05))),
        duration=int(data.get("duration", 200)),
        contact=int(data.get("contact", 0x0F)),
        cmd_source=int(data.get("cmd_source", 10)),
        use_speed_based_gait=bool(data.get("use_speed_based_gait", True)),
        profile_table=profiles or CommandMapperConfig().profile_table,
    )


def _developer_servo_motion_from_dict(data: dict[str, Any]) -> DeveloperServoMotion:
    bounds = data["bounds"]
    return DeveloperServoMotion(
        developer_motion_id=int(data["developer_motion_id"]),
        name=str(data["name"]),
        mode=int(data["mode"]),
        gait_id=int(data["gait_id"]),
        bounds=VelocityBounds(
            tuple(float(value) for value in bounds["lin_vel_x"]),
            tuple(float(value) for value in bounds["lin_vel_y"]),
            tuple(float(value) for value in bounds["ang_vel_z"]),
        ),
        refs_motion_id=data.get("refs_motion_id"),
        confidence=str(data.get("confidence", "inferred")),
        note=str(data.get("note", "")),
    )


def _positive_scale(bounds: list[float] | tuple[float, float]) -> float:
    low, high = bounds
    return float(max(abs(low), abs(high)))
