from __future__ import annotations

import torch

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def energy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize absolute joint power."""

    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def joint_position_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize deviation from default posture, more strongly when idling."""

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def feet_height_body(env, command_name: str, asset_cfg: SceneEntityCfg, target_height: float, tanh_mult: float) -> torch.Tensor:
    """Measure swing feet height in the body frame."""

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    foot_pos_b = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    foot_vel_b = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        foot_pos_b[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, foot_pos[:, i, :])
        foot_vel_b[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, foot_vel[:, i, :])
    foot_z_error = torch.square(foot_pos_b[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_xy_tanh = torch.tanh(tanh_mult * torch.norm(foot_vel_b[:, :, :2], dim=2))
    reward = torch.sum(foot_z_error * foot_xy_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize large lateral contact forces at the feet."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    return torch.any(forces_xy > 4 * forces_z, dim=1).float()


def air_time_variance_penalty(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize large variance between feet contact/air durations."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor.track_air_time before using air_time_variance_penalty.")
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def trot_gait_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 5.0,
) -> torch.Tensor:
    """Reward diagonal trot: FL+RR swing together, FR+RL swing together, alternating.

    Trotting means the diagonal leg pairs (FL+RR, FR+RL) should be synchronized
    and the two pairs should alternate: when one pair is on ground, the other is in air.
    This also penalizes pacing (FL+FR together) and bounding (all together).
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]

    # Determine contact state (bool) for each foot
    in_contact = forces_z > contact_threshold  # (N, 4)

    # Diagonal pair averages: 0=both in air, 0.5=one on ground, 1=both on ground
    pair_diag1 = (in_contact[:, 0].float() + in_contact[:, 3].float()) / 2.0   # FL + RR
    pair_diag2 = (in_contact[:, 1].float() + in_contact[:, 2].float()) / 2.0   # FR + RL

    # Trotting: diagonal pairs alternate → |pair1 - pair2| close to 1.0
    alternation = torch.abs(pair_diag1 - pair_diag2)

    # Pace penalty: front pair (FL+FR) in same state while diagonal pair is mixed
    pace_front = torch.abs(in_contact[:, 0].float() - in_contact[:, 1].float())  # 0 if FL==FR
    pace_rear = torch.abs(in_contact[:, 2].float() - in_contact[:, 3].float())   # 0 if RR==RL
    pace_penalty = 1.0 - 0.5 * (pace_front + pace_rear)  # 0 if both pairs alternate, 1 if both pace

    return alternation - 0.3 * pace_penalty
