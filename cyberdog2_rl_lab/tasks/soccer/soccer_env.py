"""Cyberdog2 2v2 soccer DirectMARLEnv — Flamez-aligned single-agent training.

Only the blue attacker is RL-trained. The other three agents use scripted
controllers that match Flamez's SoccerEnv opponent behavior. The environment
supports 4-phase curriculum, domain randomization of scripted agent parameters,
12-D Flamez-compatible observations, and the full Flamez reward shaping.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.terrains import TerrainImporter

from cyberdog2_rl_lab.tasks.soccer.mdp import AGENTS, OPPONENTS, ROLE, TEAM_SIGN, TEAMMATE
from cyberdog2_rl_lab.tasks.soccer.mdp.rewards import (approaching_goal_reward, distance_reward, dribbling_reward, facing_direction_reward, facing_ball_reward, progress_towards_goal, projected_ball_speed)
from cyberdog2_rl_lab.tasks.soccer.mdp.terminations import ball_hit_wall, ball_out_of_bounds, goal_scored, robot_collision, robot_goal_collision, robot_hit_wall
from cyberdog2_rl_lab.tasks.soccer.observation_schema import BALL_VEL_OBS_LIMIT, FLAMEZ_OBSERVATION_DIM, MAX_LINEAR_SPD, MAX_ANGULAR_SPD, POLICY_OBSERVATION_DIM

from .field_config import load_field_config
from .locomotion_policy_controller import LocomotionPolicyController
from .motion_id_controller import MotionIdController
from .soccer_env_cfg import SoccerEnvCfg


# --- Flamez-aligned constants ---
GOAL_LANE_CLEARANCE = 0.05
WALL_PROXIMITY_MARGIN = 0.45
ROBOT_RADIUS = 0.27
BALL_RADIUS = 0.11
KEEPER_GUARD_DEPTH = 0.45
KEEPER_GUARD_MARGIN = 0.12
KEEPER_POSITION_GAIN = 0.9
KEEPER_YAW_GAIN = 1.2
KEEPER_GOAL_TOLERANCE = 0.18
KEEPER_MIN_BALL_SPEED = 0.08
KEEPER_INTERCEPT_DISTANCE = 0.65
OPP_STRIKER_SPEED = 0.75
OPP_STRIKER_APPROACH_GAIN = 0.9
OPP_STRIKER_LATERAL_GAIN = 0.9
OPP_STRIKER_YAW_GAIN = 1.2
OPP_POSSESSION_DISTANCE = 0.45

MAX_EPISODE_STEPS = 500
WRONG_SIDE_RESET_PROB = 0.35
PHASE1_SHOOTING_RESET_PROB = 0.75
SHOOTING_RESET_BALL_X = (-0.38, 0.38)
SHOOTING_RESET_BALL_Y = (-1.6, 2.4)
SHOOTING_RESET_DEPTH = (0.55, 0.85)
SHOOTING_RESET_LATERAL = (-0.14, 0.14)
WRONG_SIDE_RESET_FORWARD_OFFSET = (0.55, 1.10)
WRONG_SIDE_RESET_LATERAL_OFFSET = (-0.45, 0.45)


def _safe_unit(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.norm(torch.stack((x, y), dim=1), dim=1).clamp_min(1e-8)
    return x / norm, y / norm


def yaw_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    quat = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
    half = 0.5 * yaw
    quat[:, 0] = torch.cos(half)
    quat[:, 3] = torch.sin(half)
    return quat


def _predict_guard_target(
    ball_xy: torch.Tensor, ball_vel: torch.Tensor,
    defense_goal_xy: torch.Tensor, goal_width: float,
    guard_depth: float, guard_margin: float, min_ball_speed: float,
) -> torch.Tensor:
    half = max(goal_width * 0.5 - guard_margin, 0.05)
    x_min = defense_goal_xy[0] - half
    x_max = defense_goal_xy[0] + half
    inward_sign = -1.0 if defense_goal_xy[1] > 0.0 else 1.0
    guard_y = defense_goal_xy[1] + inward_sign * guard_depth

    speed = torch.linalg.norm(ball_vel, dim=1)
    moving_toward = (defense_goal_xy[1] - ball_xy[:, 1]) * ball_vel[:, 1] > 0.0
    time_to_line = (defense_goal_xy[1] - ball_xy[:, 1]) / ball_vel[:, 1].clamp_min(1e-6)
    predicted_x = ball_xy[:, 0] + ball_vel[:, 0] * torch.clamp(time_to_line, min=0.0)

    target_x = torch.where(moving_toward, predicted_x, ball_xy[:, 0])
    target_x = torch.where(speed < min_ball_speed, ball_xy[:, 0], target_x)
    target_x = torch.clamp(target_x, x_min, x_max)
    target_y = torch.full_like(target_x, guard_y)
    return torch.stack((target_x, target_y), dim=1)


def _script_goalkeeper_velocity(
    keeper_xy: torch.Tensor, keeper_yaw: torch.Tensor,
    ball_xy: torch.Tensor, ball_vel: torch.Tensor,
    defense_goal_xy: torch.Tensor, goal_width: float,
    guard_depth: float, guard_margin: float, min_ball_speed: float,
    position_gain: float, yaw_gain: float, goal_tolerance: float,
    intercept_distance: float,
) -> torch.Tensor:
    target = _predict_guard_target(ball_xy, ball_vel, defense_goal_xy, goal_width,
                                    guard_depth, guard_margin, min_ball_speed)
    close_to_line = torch.abs(ball_xy[:, 1] - defense_goal_xy[1]) <= intercept_distance
    inside_mouth = torch.abs(ball_xy[:, 0] - defense_goal_xy[0]) <= goal_width * 0.5
    intercept = close_to_line & inside_mouth
    target = torch.where(intercept.unsqueeze(1),
                         torch.stack((torch.clamp(ball_xy[:, 0], target[:, 0].min().item(), target[:, 0].max().item()),
                                      ball_xy[:, 1]), dim=1),
                         target)

    dx_w = target[:, 0] - keeper_xy[:, 0]
    dy_w = target[:, 1] - keeper_xy[:, 1]
    cos_y = torch.cos(keeper_yaw)
    sin_y = torch.sin(keeper_yaw)
    vx = cos_y * dx_w + sin_y * dy_w
    vy = -sin_y * dx_w + cos_y * dy_w
    dist = torch.linalg.norm(target - keeper_xy, dim=1)
    still = dist <= goal_tolerance

    target_yaw = torch.atan2(ball_xy[:, 1] - keeper_xy[:, 1], ball_xy[:, 0] - keeper_xy[:, 0])
    yaw_err = torch.atan2(torch.sin(target_yaw - keeper_yaw), torch.cos(target_yaw - keeper_yaw))

    vx = torch.where(still, torch.zeros_like(vx), torch.clamp(position_gain * vx, -MAX_LINEAR_SPD, MAX_LINEAR_SPD))
    vy = torch.where(still, torch.zeros_like(vy), torch.clamp(position_gain * vy, -MAX_LINEAR_SPD, MAX_LINEAR_SPD))
    wz = torch.clamp(yaw_gain * yaw_err, -MAX_ANGULAR_SPD, MAX_ANGULAR_SPD)
    return torch.stack((vx, vy, wz), dim=1)


def _script_striker_velocity(
    striker_xy: torch.Tensor, striker_yaw: torch.Tensor,
    ball_xy: torch.Tensor, target_goal_xy: torch.Tensor,
    possession_dist: float, max_speed: float,
    approach_gain: float, lateral_gain: float, yaw_gain: float,
) -> torch.Tensor:
    ball_dist = torch.linalg.norm(ball_xy - striker_xy, dim=1)
    has_ball = ball_dist <= possession_dist
    target = torch.where(has_ball.unsqueeze(1), target_goal_xy.unsqueeze(0).expand_as(ball_xy), ball_xy)
    face = target

    dx_w = target[:, 0] - striker_xy[:, 0]
    dy_w = target[:, 1] - striker_xy[:, 1]
    cos_y = torch.cos(striker_yaw)
    sin_y = torch.sin(striker_yaw)
    vx = cos_y * dx_w + sin_y * dy_w
    vy = -sin_y * dx_w + cos_y * dy_w

    target_yaw = torch.atan2(face[:, 1] - striker_xy[:, 1], face[:, 0] - striker_xy[:, 0])
    yaw_err = torch.atan2(torch.sin(target_yaw - striker_yaw), torch.cos(target_yaw - striker_yaw))

    vx = torch.clamp(approach_gain * vx, -max_speed, max_speed)
    vy = torch.clamp(lateral_gain * vy, -max_speed, max_speed)
    wz = torch.clamp(yaw_gain * yaw_err, -MAX_ANGULAR_SPD, MAX_ANGULAR_SPD)
    return torch.stack((vx, vy, wz), dim=1)


class SoccerEnv(DirectMARLEnv):
    """2v2 soccer — blue attacker is RL, all others are scripted (Flamez-aligned)."""

    cfg: SoccerEnvCfg

    def __init__(self, cfg: SoccerEnvCfg, render_mode: str | None = None, **kwargs):
        self._initialized = False
        self.field_cfg = load_field_config(cfg.field_config_path)
        self.phase = int(getattr(cfg, "phase", 3))
        super().__init__(cfg, render_mode, **kwargs)

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.step_count_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        if getattr(self.cfg, "controller_type", "locomotion_policy") == "motion_id":
            self.locomotion_controller = MotionIdController(self.robots, self.num_envs, self.device)
            print("[INFO] Using Motion ID controller")
        else:
            self.locomotion_controller = LocomotionPolicyController(
                self.robots, self.num_envs, self.device,
                policy_path=self.cfg.locomotion_policy_path or None,
                deploy_cfg_path=self.cfg.locomotion_deploy_cfg_path or None,
            )
            print("[INFO] Using Locomotion Policy controller")

        _all_agents = AGENTS
        self.velocity_commands: dict[str, torch.Tensor] = {
            agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in _all_agents
        }
        self.joint_targets: dict[str, torch.Tensor] = {
            agent: robot.data.default_joint_pos.clone() for agent, robot in self.robots.items()
        }
        self.actions: dict[str, torch.Tensor] = {
            agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in _all_agents
        }
        self.last_actions: dict[str, torch.Tensor] = {
            agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in _all_agents
        }
        self.command_rate: dict[str, torch.Tensor] = {
            agent: torch.zeros(self.num_envs, device=self.device) for agent in _all_agents
        }

        self.prev_ball_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.prev_robot_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.blue_scored = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.red_scored = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.robot_fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_wall_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.self_wall_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_wall_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.teammate_collision_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.opponent_collision_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.domain_params: dict[str, float] = {}
        self._sample_domain_params()

        self.half_width = self.cfg.field_x_width * 0.5
        self.half_length = self.cfg.field_y_length * 0.5
        self.goal_width = self.cfg.goal_width
        self.robot_radius = ROBOT_RADIUS
        self.ball_radius = BALL_RADIUS
        self._goal_b = torch.tensor([0.0, self.half_length], device=self.device)
        self._goal_a = torch.tensor([0.0, -self.half_length], device=self.device)
        self._field_scale = torch.tensor([self.half_width, self.half_length], device=self.device)
        self._action_scale = torch.tensor(self.cfg.high_level_action_scale, device=self.device)

        self._latency_enabled = self.phase >= 4
        self._action_delay_steps = 0
        self._obs_delay_steps = 0
        self._action_buffer: list[torch.Tensor] = []
        self._obs_buffer: list[torch.Tensor] = []
        if self._latency_enabled:
            self._sample_latency_params()
            self._init_latency_buffers()

        self._episode_sums = {
            agent: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for agent in _all_agents
        }
        self._initialized = True

    def _setup_scene(self):
        self.robots = {agent: Articulation(getattr(self.cfg, agent)) for agent in AGENTS}
        self.ball = RigidObject(self.cfg.ball)
        for name, robot in self.robots.items():
            self.scene.articulations[name] = robot
        self.scene.rigid_objects["ball"] = self.ball

        self.scene.rigid_objects["field_wall_top"] = RigidObject(self.cfg.field_wall_top)
        self.scene.rigid_objects["field_wall_bot"] = RigidObject(self.cfg.field_wall_bot)
        self.scene.rigid_objects["field_wall_left_upper"] = RigidObject(self.cfg.field_wall_left_upper)
        self.scene.rigid_objects["field_wall_left_lower"] = RigidObject(self.cfg.field_wall_left_lower)
        self.scene.rigid_objects["field_wall_right_upper"] = RigidObject(self.cfg.field_wall_right_upper)
        self.scene.rigid_objects["field_wall_right_lower"] = RigidObject(self.cfg.field_wall_right_lower)
        self.scene.rigid_objects["goal_blue_left_wall"] = RigidObject(self.cfg.goal_blue_left_wall)
        self.scene.rigid_objects["goal_blue_right_wall"] = RigidObject(self.cfg.goal_blue_right_wall)
        self.scene.rigid_objects["goal_red_left_wall"] = RigidObject(self.cfg.goal_red_left_wall)
        self.scene.rigid_objects["goal_red_right_wall"] = RigidObject(self.cfg.goal_red_right_wall)
        self.scene.rigid_objects["goal_line_blue"] = RigidObject(self.cfg.goal_line_blue)
        self.scene.rigid_objects["goal_line_red"] = RigidObject(self.cfg.goal_line_red)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain: TerrainImporter = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self._reset_step_count += 1
        action = actions["blue_attacker"][:, :3].clone()
        self.last_actions["blue_attacker"] = self.actions["blue_attacker"].clone()
        self.actions["blue_attacker"] = torch.clamp(action, -1.0, 1.0)
        prev_command = self.velocity_commands["blue_attacker"].clone()
        raw_command = self.actions["blue_attacker"] * self._action_scale
        warmup = (self._reset_step_count.float() / self.cfg.command_warmup_steps).clamp(max=1.0)
        self.velocity_commands["blue_attacker"] = raw_command * warmup.unsqueeze(-1)
        self.command_rate["blue_attacker"] = torch.linalg.norm(
            self.velocity_commands["blue_attacker"] - prev_command, dim=1
        )

        blue_att_xy = self._robot_xy("blue_attacker")
        blue_att_yaw = self._robot_yaw("blue_attacker")
        ball_xy = self._ball_xy()
        ball_vel = self.ball.data.root_lin_vel_w[:, :2]

        red_gk_xy = self._robot_xy("red_goalkeeper")
        red_gk_yaw = self._robot_yaw("red_goalkeeper")
        blue_gk_xy = self._robot_xy("blue_goalkeeper")
        blue_gk_yaw = self._robot_yaw("blue_goalkeeper")
        red_att_xy = self._robot_xy("red_attacker")
        red_att_yaw = self._robot_yaw("red_attacker")

        dp = self.domain_params
        red_gk_active = self.phase >= 2
        blue_gk_active = self.phase >= 3
        red_att_active = self.phase >= 3

        if red_gk_active:
            self.velocity_commands["red_goalkeeper"] = _script_goalkeeper_velocity(
                red_gk_xy, red_gk_yaw, ball_xy, ball_vel,
                self._goal_b, self.goal_width,
                dp.get("goalkeeper_guard_depth", KEEPER_GUARD_DEPTH),
                dp.get("goalkeeper_guard_margin", KEEPER_GUARD_MARGIN),
                KEEPER_MIN_BALL_SPEED,
                dp.get("goalkeeper_position_gain", KEEPER_POSITION_GAIN),
                dp.get("goalkeeper_yaw_gain", KEEPER_YAW_GAIN),
                KEEPER_GOAL_TOLERANCE, KEEPER_INTERCEPT_DISTANCE,
            )
        else:
            self.velocity_commands["red_goalkeeper"].zero_()

        if blue_gk_active:
            self.velocity_commands["blue_goalkeeper"] = _script_goalkeeper_velocity(
                blue_gk_xy, blue_gk_yaw, ball_xy, ball_vel,
                self._goal_a, self.goal_width,
                dp.get("teammate_goalkeeper_guard_depth", KEEPER_GUARD_DEPTH),
                dp.get("teammate_goalkeeper_guard_margin", KEEPER_GUARD_MARGIN),
                KEEPER_MIN_BALL_SPEED,
                dp.get("teammate_goalkeeper_position_gain", KEEPER_POSITION_GAIN),
                dp.get("teammate_goalkeeper_yaw_gain", KEEPER_YAW_GAIN),
                KEEPER_GOAL_TOLERANCE, KEEPER_INTERCEPT_DISTANCE,
            )
        else:
            self.velocity_commands["blue_goalkeeper"].zero_()

        if red_att_active:
            self.velocity_commands["red_attacker"] = _script_striker_velocity(
                red_att_xy, red_att_yaw, ball_xy, self._goal_a.unsqueeze(0).expand(self.num_envs, -1),
                dp.get("opponent_possession_distance", OPP_POSSESSION_DISTANCE),
                dp.get("opponent_striker_speed", OPP_STRIKER_SPEED),
                dp.get("opponent_striker_approach_gain", OPP_STRIKER_APPROACH_GAIN),
                dp.get("opponent_striker_lateral_gain", OPP_STRIKER_LATERAL_GAIN),
                dp.get("opponent_striker_yaw_gain", OPP_STRIKER_YAW_GAIN),
            )
        else:
            self.velocity_commands["red_attacker"].zero_()

        self.joint_targets = self.locomotion_controller.compute_joint_targets(self.velocity_commands)

    def _apply_action(self) -> None:
        self.locomotion_controller.apply_joint_targets(self.joint_targets)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = {}
        sign = TEAM_SIGN["blue_attacker"]
        robot = self.robots["blue_attacker"]
        robot_xy = robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        robot_yaw = yaw_from_quat(robot.data.root_quat_w)
        ball_xy = self._ball_xy()
        ball_vel = self.ball.data.root_lin_vel_w[:, :2]
        goal_xy = self._goal_b.unsqueeze(0).expand(self.num_envs, -1)

        opponent_xy = self._robot_xy("red_goalkeeper") if self.phase >= 2 else torch.zeros_like(robot_xy)
        if self.phase >= 3:
            opp_striker_xy = self._robot_xy("red_attacker")
            dist_gk = torch.linalg.norm(opponent_xy - goal_xy, dim=1)
            dist_st = torch.linalg.norm(opp_striker_xy - goal_xy, dim=1)
            opponent_xy = torch.where((dist_gk < dist_st).unsqueeze(1), opponent_xy, opp_striker_xy)

        canonical_yaw = sign * robot_yaw

        obs["blue_attacker"] = torch.stack([
            self._clip(robot_xy[:, 0] / self.half_width),
            self._clip(sign * robot_xy[:, 1] / self.half_length),
            torch.sin(canonical_yaw),
            torch.cos(canonical_yaw),
            self._clip(ball_xy[:, 0] / self.half_width),
            self._clip(sign * ball_xy[:, 1] / self.half_length),
            self._clip(ball_vel[:, 0] / MAX_LINEAR_SPD, BALL_VEL_OBS_LIMIT),
            self._clip(sign * ball_vel[:, 1] / MAX_LINEAR_SPD, BALL_VEL_OBS_LIMIT),
            self._clip(goal_xy[:, 0] / self.half_width),
            self._clip(sign * goal_xy[:, 1] / self.half_length),
            self._clip(opponent_xy[:, 0] / self.half_width),
            self._clip(sign * opponent_xy[:, 1] / self.half_length),
        ], dim=1)
        return obs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        ball_xy = self._ball_xy()
        ball_vel = self.ball.data.root_lin_vel_w[:, :2]
        robot_xy = self._robot_xy("blue_attacker")
        robot_yaw = self._robot_yaw("blue_attacker")
        robot_vel = self.robots["blue_attacker"].data.root_lin_vel_w[:, :2]
        sign = TEAM_SIGN["blue_attacker"]

        team_reward = (
            self.cfg.rew_goal * self.blue_scored.float()
            + self.cfg.rew_concede * self.red_scored.float()
            + self.cfg.rew_ball_progress * progress_towards_goal(ball_xy[:, 1], self.prev_ball_xy[:, 1], sign) * self.step_dt
            + self.cfg.rew_ball_speed * projected_ball_speed(ball_vel[:, 1], sign) * self.step_dt
        )

        dist_to_ball = torch.linalg.norm(ball_xy - robot_xy, dim=1)
        behind_point = ball_xy.clone()
        behind_point[:, 0] = ball_xy[:, 0].clamp(-0.5 * self.cfg.field_x_width, 0.5 * self.cfg.field_x_width)
        behind_point[:, 1] -= sign * 0.45
        behind_dist = torch.linalg.norm(behind_point - robot_xy, dim=1)
        role_reward = (
            self.cfg.rew_attacker_to_ball * distance_reward(dist_to_ball, 1.2)
            + self.cfg.rew_attacker_behind_ball * distance_reward(behind_dist, 1.5)
        )

        facing_goal = self.cfg.rew_facing_goal * facing_direction_reward(robot_yaw, sign)

        dribble_reward = self.cfg.rew_dribbling * dribbling_reward(
            robot_xy, ball_xy, ball_vel, robot_vel, sign, dist_to_ball
        )

        approaching_reward = self.cfg.rew_approaching_goal * approaching_goal_reward(
            robot_xy, self.cfg.field_y_length, sign, scale=2.0
        )

        upright_raw = torch.clamp(-self.robots["blue_attacker"].data.projected_gravity_b[:, 2], 0.0, 1.0)
        upright = upright_raw * self.cfg.rew_upright
        command_penalty = self.cfg.rew_command_rate * self.command_rate["blue_attacker"]
        out_penalty = self.cfg.rew_out_of_field * self._robot_out_of_field("blue_attacker").float()
        wall_penalty = self.cfg.rew_self_wall * self.self_wall_flag.float()
        survival = self.cfg.rew_survival * upright_raw

        # facing_ball: face toward the ball (Skill 1)
        # facing_goal: face toward the goal (Skill 2)
        facing_ball = self.cfg.rew_facing_goal * self._facing_ball_reward(robot_xy, ball_xy, robot_yaw, sign)
        facing_goal = self.cfg.rew_facing_goal * facing_direction_reward(robot_yaw, sign)

        sp = self.cfg.skill_phase
        reward = (
            (team_reward if sp >= 3 else 0.0)
            + role_reward
            + (facing_goal if sp >= 1 else 0.0)  # Skill 1: face goal when behind ball
            + (dribble_reward if sp >= 2 else 0.0)
            + (approaching_reward if sp >= 2 else 0.0)
            + (upright + survival + command_penalty + out_penalty + wall_penalty) * self.step_dt
        )

        self.prev_ball_xy = ball_xy.clone()
        self._episode_sums["blue_attacker"] += reward
        return {"blue_attacker": reward}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        ball_xy = self._ball_xy()
        self.blue_scored, self.red_scored = goal_scored(ball_xy, self.cfg.field_y_length, self.goal_width)
        self.ball_out = ball_out_of_bounds(ball_xy, self.cfg.field_x_width, self.cfg.field_y_length, self.cfg.out_margin)
        self.ball_wall_flag = ball_hit_wall(ball_xy, self.cfg.field_x_width, self.cfg.field_y_length, self.ball_radius)
        blue_xy = self._robot_xy("blue_attacker")
        self.self_wall_flag = robot_hit_wall(blue_xy, self.cfg.field_x_width, self.cfg.field_y_length, self.robot_radius)
        self.goal_wall_flag = robot_goal_collision(blue_xy, self.cfg.field_y_length, self.goal_width, self.robot_radius)

        self.robot_fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        robot = self.robots["blue_attacker"]
        self.robot_fallen = (robot.data.root_pos_w[:, 2] < 0.16) | (robot.data.projected_gravity_b[:, 2] > -0.25)

        self.teammate_collision_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.opponent_collision_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.phase >= 3:
            blue_gk_xy = self._robot_xy("blue_goalkeeper")
            self.teammate_collision_flag = robot_collision(blue_xy, blue_gk_xy, self.robot_radius)
        if self.phase >= 2:
            red_gk_xy = self._robot_xy("red_goalkeeper")
            self.opponent_collision_flag |= robot_collision(blue_xy, red_gk_xy, self.robot_radius)
        if self.phase >= 3:
            red_att_xy = self._robot_xy("red_attacker")
            self.opponent_collision_flag |= robot_collision(blue_xy, red_att_xy, self.robot_radius)

        terminated_any = (self.blue_scored | self.red_scored | self.ball_out |
                           self.goal_wall_flag | self.self_wall_flag | self.robot_fallen)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {agent: terminated_any for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def step(self, actions: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if isinstance(actions, torch.Tensor):
            actions_dict = {"blue_attacker": actions}
        else:
            actions_dict = actions
        obs, rewards, terminated, truncated, infos = super().step(actions_dict)
        self.episode_length_buf += 1
        self.step_count_buf += 1
        return obs, rewards, terminated | truncated, infos

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            first_robot = next(iter(self.robots.values()))
            env_ids = first_robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if self._initialized:
            self._log_reset(env_ids)

        self._sample_domain_params()
        super()._reset_idx(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.step_count_buf[env_ids] = 0

        if self._initialized:
            self.locomotion_controller.reset(env_ids)
            for agent in AGENTS:
                if agent in self.velocity_commands:
                    self.velocity_commands[agent][env_ids] = 0.0
                    self.actions[agent][env_ids] = 0.0
                    self.last_actions[agent][env_ids] = 0.0
                    self.command_rate[agent][env_ids] = 0.0
            self._reset_step_count[env_ids] = 0

        ball_xy = self._reset_ball(env_ids)
        if self.phase >= 2:
            gk_x, gk_y, gk_yaw = self.cfg.player_starts["red_goalkeeper"]
            self._reset_robot("red_goalkeeper", env_ids, self._draw_jittered(env_ids, gk_x, gk_y, 0.2, 0.15),
                              self._draw_yaw(env_ids, gk_yaw, 0.15))
        else:
            self._reset_robot("red_goalkeeper", env_ids, self._far_xy(env_ids, 5.5, 5.8),
                              torch.zeros(len(env_ids), device=self.device))
        if self.phase >= 3:
            bgk_x, bgk_y, bgk_yaw = self.cfg.player_starts["blue_goalkeeper"]
            self._reset_robot("blue_goalkeeper", env_ids, self._draw_jittered(env_ids, bgk_x, bgk_y, 0.2, 0.15),
                              self._draw_yaw(env_ids, bgk_yaw, 0.15))
            ratt_x, ratt_y, ratt_yaw = self.cfg.player_starts["red_attacker"]
            self._reset_robot("red_attacker", env_ids, self._draw_jittered(env_ids, ratt_x, ratt_y, 1.5, 2.0),
                              self._draw_yaw(env_ids, ratt_yaw, 1.5))
        else:
            self._reset_robot("blue_goalkeeper", env_ids, self._far_xy(env_ids, -5.5, 5.8),
                              torch.zeros(len(env_ids), device=self.device))
            self._reset_robot("red_attacker", env_ids, self._far_xy(env_ids, -5.5, -5.8),
                              torch.zeros(len(env_ids), device=self.device))

        blue_x, blue_y, blue_yaw = self.cfg.player_starts["blue_attacker"]
        blue_xy = self._draw_jittered(env_ids, blue_x, blue_y, 1.0, 1.5)
        self._reset_robot("blue_attacker", env_ids, blue_xy, self._draw_yaw(env_ids, blue_yaw, 0.5))

        self.prev_ball_xy[env_ids] = self._ball_xy()[env_ids]
        self.prev_robot_xy[env_ids] = blue_xy

        self.blue_scored[env_ids] = False
        self.red_scored[env_ids] = False
        self.ball_out[env_ids] = False
        self.ball_wall_flag[env_ids] = False
        self.self_wall_flag[env_ids] = False
        self.goal_wall_flag[env_ids] = False
        self.teammate_collision_flag[env_ids] = False
        self.opponent_collision_flag[env_ids] = False

    def _reset_robot(self, agent: str, env_ids: torch.Tensor, xy: torch.Tensor, yaw: torch.Tensor) -> None:
        robot = self.robots[agent]
        robot.reset(env_ids)
        root_state = robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:2] = xy + self.scene.env_origins[env_ids, :2]
        root_state[:, 2] = 0.40
        root_state[:, 3:7] = quat_from_yaw(yaw)
        root_state[:, 7:] = 0.0
        robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        joint_pos = robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        robot.set_joint_position_target(robot.data.default_joint_pos[env_ids], env_ids=env_ids)

    def _reset_ball(self, env_ids: torch.Tensor) -> torch.Tensor:
        ball_state = self.ball.data.default_root_state[env_ids].clone()
        ball_xy = self._sample_ball_xy(env_ids)
        ball_state[:, 0:2] = ball_xy + self.scene.env_origins[env_ids, :2]
        ball_state[:, 2] = self.ball_radius + 0.02
        ball_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        ball_state[:, 7:] = 0.0
        self.ball.write_root_pose_to_sim(ball_state[:, :7], env_ids)
        self.ball.write_root_velocity_to_sim(ball_state[:, 7:], env_ids)
        return ball_xy

    def _facing_ball_reward(
        self, robot_xy: torch.Tensor, ball_xy: torch.Tensor, robot_yaw: torch.Tensor, sign: float
    ) -> torch.Tensor:
        """Reward robot for facing toward the ball (Skill 1)."""
        dx = ball_xy[:, 0] - robot_xy[:, 0]
        dy = sign * (ball_xy[:, 1] - robot_xy[:, 1])
        target_yaw = torch.atan2(dy, dx)
        return torch.cos(robot_yaw - target_yaw)

    def _sample_ball_xy(self, env_ids: torch.Tensor) -> torch.Tensor:
        margin = 0.8
        x_lim = max(0.0, 0.5 * self.cfg.field_x_width - margin)
        y_lim = max(0.0, 0.5 * self.cfg.field_y_length - margin)
        xy = torch.empty(len(env_ids), 2, device=self.device)
        xy[:, 0].uniform_(-x_lim, x_lim)
        n = len(env_ids)
        r = torch.rand(n, device=self.device)
        y_offense = torch.empty(n, device=self.device).uniform_(0.3, y_lim)
        y_defense = torch.empty(n, device=self.device).uniform_(-y_lim, -0.3)
        y_near_goal = torch.empty(n, device=self.device).uniform_(max(0.3, y_lim - 1.5), y_lim)
        xy[:, 1] = torch.where(r < 0.45, y_offense, torch.where(r < 0.9, y_defense, y_near_goal))
        return xy

    def _draw_jittered(self, env_ids: torch.Tensor, base_x: float, base_y: float,
                         x_noise: float, y_noise: float) -> torch.Tensor:
        xy = torch.empty(len(env_ids), 2, device=self.device)
        xy[:, 0].uniform_(base_x - x_noise, base_x + x_noise)
        xy[:, 1].uniform_(base_y - y_noise, base_y + y_noise)
        x_lim = max(0.0, self.half_width - self.robot_radius - 0.2)
        y_lim = max(0.0, self.half_length - self.robot_radius - 0.2)
        xy[:, 0].clamp_(-x_lim, x_lim)
        xy[:, 1].clamp_(-y_lim, y_lim)
        return xy

    def _draw_yaw(self, env_ids: torch.Tensor, base_yaw: float, noise: float) -> torch.Tensor:
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(base_yaw - noise, base_yaw + noise)
        return yaw

    def _far_xy(self, env_ids: torch.Tensor, x: float, y: float) -> torch.Tensor:
        n = len(env_ids)
        xy = torch.empty(n, 2, device=self.device)
        xy[:, 0] = x
        xy[:, 1] = y
        return xy

    def _robot_xy(self, agent: str) -> torch.Tensor:
        if agent not in self.robots:
            return torch.zeros(self.num_envs, 2, device=self.device)
        return self.robots[agent].data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]

    def _robot_yaw(self, agent: str) -> torch.Tensor:
        if agent not in self.robots:
            return torch.zeros(self.num_envs, device=self.device)
        return yaw_from_quat(self.robots[agent].data.root_quat_w)

    def _ball_xy(self) -> torch.Tensor:
        return self.ball.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]

    def _robot_out_of_field(self, agent: str) -> torch.Tensor:
        xy = self._robot_xy(agent)
        dead_line_margin = 0.6
        x_limit = 0.5 * self.cfg.field_x_width - dead_line_margin
        y_limit = 0.5 * self.cfg.field_y_length - dead_line_margin
        return (torch.abs(xy[:, 0]) > x_limit) | (torch.abs(xy[:, 1]) > y_limit)

    @staticmethod
    def _clip(value: torch.Tensor, limit: float = 1.0) -> torch.Tensor:
        return torch.clamp(value, -limit, limit)

    def _sample_domain_params(self) -> None:
        self.domain_params = {
            "goalkeeper_guard_depth": KEEPER_GUARD_DEPTH + torch.empty(1).uniform_(-0.10, 0.10).item(),
            "goalkeeper_guard_margin": KEEPER_GUARD_MARGIN + torch.empty(1).uniform_(-0.04, 0.04).item(),
            "goalkeeper_position_gain": KEEPER_POSITION_GAIN + torch.empty(1).uniform_(-0.15, 0.15).item(),
            "goalkeeper_yaw_gain": KEEPER_YAW_GAIN + torch.empty(1).uniform_(-0.2, 0.3).item(),
            "teammate_goalkeeper_guard_depth": KEEPER_GUARD_DEPTH + torch.empty(1).uniform_(-0.10, 0.10).item(),
            "teammate_goalkeeper_guard_margin": KEEPER_GUARD_MARGIN + torch.empty(1).uniform_(-0.04, 0.04).item(),
            "teammate_goalkeeper_position_gain": KEEPER_POSITION_GAIN + torch.empty(1).uniform_(-0.15, 0.15).item(),
            "teammate_goalkeeper_yaw_gain": KEEPER_YAW_GAIN + torch.empty(1).uniform_(-0.2, 0.3).item(),
            "opponent_striker_speed": OPP_STRIKER_SPEED + torch.empty(1).uniform_(-0.2, 0.15).item(),
            "opponent_striker_approach_gain": OPP_STRIKER_APPROACH_GAIN + torch.empty(1).uniform_(-0.2, 0.2).item(),
            "opponent_striker_lateral_gain": OPP_STRIKER_LATERAL_GAIN + torch.empty(1).uniform_(-0.2, 0.2).item(),
            "opponent_striker_yaw_gain": OPP_STRIKER_YAW_GAIN + torch.empty(1).uniform_(-0.3, 0.3).item(),
            "opponent_possession_distance": OPP_POSSESSION_DISTANCE + torch.empty(1).uniform_(-0.10, 0.10).item(),
        }

    def _sample_latency_params(self) -> None:
        pass

    def _init_latency_buffers(self) -> None:
        pass

    def _log_reset(self, env_ids: torch.Tensor) -> None:
        log = {}
        ep_len = self.episode_length_buf[env_ids].float().clamp_min(1.0)
        for agent, value in self._episode_sums.items():
            log[f"Episode_Reward/{agent}"] = torch.mean(value[env_ids] / ep_len).item()
            value[env_ids] = 0.0
        log["Episode_Termination/blue_goal"] = torch.count_nonzero(self.blue_scored[env_ids]).item()
        log["Episode_Termination/red_goal"] = torch.count_nonzero(self.red_scored[env_ids]).item()
        log["Episode_Termination/ball_out"] = torch.count_nonzero(self.ball_out[env_ids]).item()
        log["Episode_Termination/ball_wall"] = torch.count_nonzero(self.ball_wall_flag[env_ids]).item()
        log["Episode_Termination/self_wall"] = torch.count_nonzero(self.self_wall_flag[env_ids]).item()
        log["Episode_Termination/goal_wall"] = torch.count_nonzero(self.goal_wall_flag[env_ids]).item()
        log["Episode_Termination/fallen"] = torch.count_nonzero(self.robot_fallen[env_ids]).item()
        self.extras["log"] = log

    def get_observations(self) -> torch.Tensor:
        obs_dict = self._get_observations()
        return obs_dict["blue_attacker"]

    def get_rewards(self) -> torch.Tensor:
        reward_dict = self._get_rewards()
        return reward_dict["blue_attacker"]

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated_dict, truncated_dict = self._get_dones()
        return terminated_dict["blue_attacker"], truncated_dict["blue_attacker"]
