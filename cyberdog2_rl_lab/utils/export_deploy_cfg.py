import os

import numpy as np
import yaml
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def _format_value(value):
    if isinstance(value, float):
        return float(f"{value:.3g}")
    if isinstance(value, list):
        return [_format_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _format_value(item) for key, item in value.items()}
    return value


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir: str):
    """Export a small deployment-oriented description of the trained task."""

    asset: Articulation = env.scene["robot"]
    joint_sdk_names = env.cfg.scene.robot.joint_sdk_names
    joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)

    cfg = {}
    cfg["joint_ids_map"] = joint_ids_map
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation

    stiffness = np.zeros(len(joint_sdk_names))
    stiffness[joint_ids_map] = asset.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
    cfg["stiffness"] = stiffness.tolist()

    damping = np.zeros(len(joint_sdk_names))
    damping[joint_ids_map] = asset.data.default_joint_damping[0].detach().cpu().numpy().tolist()
    cfg["damping"] = damping.tolist()
    cfg["default_joint_pos"] = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()

    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):
        cfg["commands"]["base_velocity"] = {}
        command_cfg = env.cfg.commands.base_velocity
        ranges = command_cfg.limit_ranges.to_dict() if hasattr(command_cfg, "limit_ranges") else command_cfg.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            if item_name in ranges:
                ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()
        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()
        if action_name in ["JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]
        term_cfg = term_cfg.to_dict()
        for key in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            if key in term_cfg:
                del term_cfg[key]
        cfg["actions"][action_name] = term_cfg
        cfg["actions"][action_name]["joint_ids"] = None if action_term._joint_ids == slice(None) else action_term._joint_ids

    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    obs_terms = zip(obs_names, obs_cfgs)
    cfg["observations"] = {}
    for obs_name, obs_cfg in obs_terms:
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        if term_cfg.scale is not None:
            if hasattr(term_cfg.scale, "detach"):
                scale = term_cfg.scale.detach().cpu().numpy().tolist()
                term_cfg.scale = [scale for _ in range(obs_dims[1])] if isinstance(scale, float) else scale
            else:
                term_cfg.scale = [term_cfg.scale for _ in range(obs_dims[1])] if isinstance(term_cfg.scale, float) else term_cfg.scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1
        term_cfg = term_cfg.to_dict()
        for key in ["func", "modifiers", "noise", "flatten_history_dim"]:
            if key in term_cfg:
                del term_cfg[key]
        cfg["observations"][obs_name] = term_cfg

    filename = os.path.join(log_dir, "params", "deploy.yaml")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = _format_value(cfg)
    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(cfg, file, default_flow_style=None, sort_keys=False)
