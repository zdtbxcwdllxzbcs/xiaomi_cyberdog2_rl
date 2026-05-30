"""Direct IsaacLab environment for official Cyberdog2 velocity commands."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.terrains import TerrainImporter

from cyberdog2_rl_lab.official import CommandMapperConfig, TensorCommandMapper

from .official_locomotion_env_cfg import OfficialLocomotionEnvCfg


class OfficialLocomotionEnv(DirectRLEnv):
    """Smoke-test task for official high-level locomotion command semantics."""

    cfg: OfficialLocomotionEnvCfg

    def __init__(self, cfg: OfficialLocomotionEnvCfg, render_mode: str | None = None, **kwargs):
        self._actions = None
        super().__init__(cfg, render_mode, **kwargs)

        mapper_cfg = CommandMapperConfig(
            developer_motion_id=self.cfg.developer_motion_id,
            max_lin_acc=self.cfg.max_lin_acc,
            max_lat_acc=self.cfg.max_lat_acc,
            max_yaw_acc=self.cfg.max_yaw_acc,
        )
        self.command_mapper = TensorCommandMapper(self.num_envs, self.device, mapper_cfg)
        self._actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_actions = torch.zeros_like(self._actions)
        self._target_velocity_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._command_timer = torch.zeros(self.num_envs, device=self.device)
        self._command_rate = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["track_lin_vel", "track_ang_vel", "upright", "action_rate", "command_rate"]
        }

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain: TerrainImporter = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._last_actions[:] = self._actions
        self._actions = actions[:, :3].clone()
        prev_command = self.command_mapper.last_velocity.clone()
        self.command_mapper.map_normalized_action(self._actions, self.step_dt)
        self._command_rate = torch.linalg.norm(self.command_mapper.last_velocity - prev_command, dim=1)

        self._command_timer += self.step_dt
        resample_ids = torch.nonzero(self._command_timer >= self.cfg.command_resampling_time_s, as_tuple=False).squeeze(-1)
        if len(resample_ids) > 0:
            self._sample_target_velocity(resample_ids)

    def _apply_action(self) -> None:
        cmd_b = self.command_mapper.last_velocity
        yaw = yaw_from_quat(self.robot.data.root_quat_w)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        root_vel[:, 0] = cos_yaw * cmd_b[:, 0] - sin_yaw * cmd_b[:, 1]
        root_vel[:, 1] = sin_yaw * cmd_b[:, 0] + cos_yaw * cmd_b[:, 1]
        root_vel[:, 5] = cmd_b[:, 2]
        self.robot.write_root_velocity_to_sim(root_vel)
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = torch.cat(
            (
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self._target_velocity_b,
                self.command_mapper.last_velocity,
                self._actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_error = torch.sum(torch.square(self.robot.data.root_lin_vel_b[:, :2] - self._target_velocity_b[:, :2]), dim=1)
        ang_error = torch.square(self.robot.data.root_ang_vel_b[:, 2] - self._target_velocity_b[:, 2])
        upright = torch.clamp(-self.robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
        action_rate = torch.sum(torch.square(self._actions - self._last_actions), dim=1)
        command_rate = self._command_rate
        rewards = {
            "track_lin_vel": self.cfg.rew_track_lin_vel * torch.exp(-lin_error / 0.25) * self.step_dt,
            "track_ang_vel": self.cfg.rew_track_ang_vel * torch.exp(-ang_error / 0.25) * self.step_dt,
            "upright": self.cfg.rew_upright * upright * self.step_dt,
            "action_rate": self.cfg.rew_action_rate * action_rate * self.step_dt,
            "command_rate": self.cfg.rew_command_rate * command_rate * self.step_dt,
        }
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fallen = (self.robot.data.root_pos_w[:, 2] < 0.16) | (self.robot.data.projected_gravity_b[:, 2] > -0.25)
        return fallen, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if self._actions is not None:
            self._log_reset(env_ids)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self.command_mapper.reset(env_ids)
        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._command_rate[env_ids] = 0.0
        self._command_timer[env_ids] = self.cfg.command_resampling_time_s
        self._sample_target_velocity(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(-3.14159, 3.14159)
        default_root_state[:, 3:7] = quat_from_yaw(yaw)
        default_root_state[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos[env_ids], env_ids=env_ids)

    def _sample_target_velocity(self, env_ids: torch.Tensor) -> None:
        self._target_velocity_b[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.target_lin_vel_x[0], self.cfg.target_lin_vel_x[1]
        )
        self._target_velocity_b[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.target_lin_vel_y[0], self.cfg.target_lin_vel_y[1]
        )
        self._target_velocity_b[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.target_ang_vel_z[0], self.cfg.target_ang_vel_z[1]
        )
        self._command_timer[env_ids] = 0.0

    def _log_reset(self, env_ids: torch.Tensor) -> None:
        extras = {}
        for key, value in self._episode_sums.items():
            extras[f"Episode_Reward/{key}"] = torch.mean(value[env_ids]).item() / self.max_episode_length_s
            value[env_ids] = 0.0
        extras["Metrics/command_clip_speed"] = torch.linalg.norm(self.command_mapper.last_velocity[env_ids], dim=1).mean().item()
        self.extras["log"] = extras


def yaw_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    quat = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
    half = 0.5 * yaw
    quat[:, 0] = torch.cos(half)
    quat[:, 3] = torch.sin(half)
    return quat
