from __future__ import annotations

import math
from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from cyberdog2_rl_lab.assets.robots.cyberdog2 import CYBERDOG2_CFG
from cyberdog2_rl_lab.tasks.locomotion import mdp


# ============================================================================
# Low-level policy path
# ============================================================================
# This should point to a callable policy file.
# Best case: a TorchScript export (policy.pt).
# If your file is a plain checkpoint, export the policy first.
LOW_LEVEL_POLICY_PATH = r"F:\dog\cyberdog2_rl_lab\logs\rsl_rl\cyberdog2_velocity_flat\2026-05-03_17-09-36\model_39999.pt"


# ============================================================================
# Helper functions
# ============================================================================
def _load_callable_policy(policy_path: str, device: str | torch.device):
    """Load a callable policy module.

    Tries TorchScript first, then torch.load().
    """
    try:
        return torch.jit.load(policy_path, map_location=device)
    except Exception:
        obj = torch.load(policy_path, map_location=device)

        if callable(obj):
            return obj

        if isinstance(obj, dict):
            for key in ("policy", "actor", "actor_critic", "model", "module"):
                if key in obj and callable(obj[key]):
                    return obj[key]

        raise RuntimeError(
            f"Could not load a callable policy from: {policy_path}\n"
            f"Please use a TorchScript export or a callable policy file."
        )


def _goal_delta_world(env) -> torch.Tensor:
    robot = env.scene["robot"]
    goal = env.scene["goal_marker"]
    return goal.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]


def _goal_distance(env) -> torch.Tensor:
    return torch.norm(_goal_delta_world(env), dim=1).clamp_min(1e-6)


def _goal_heading_error(env) -> torch.Tensor:
    robot = env.scene["robot"]
    delta = _goal_delta_world(env)
    target_heading = torch.atan2(delta[:, 1], delta[:, 0])
    return torch.atan2(
        torch.sin(target_heading - robot.data.heading_w),
        torch.cos(target_heading - robot.data.heading_w),
    )


def _goal_delta_body(env) -> torch.Tensor:
    robot = env.scene["robot"]
    delta_w = _goal_delta_world(env)
    heading = robot.data.heading_w

    c = torch.cos(-heading)
    s = torch.sin(-heading)

    dx = delta_w[:, 0]
    dy = delta_w[:, 1]

    x_b = c * dx - s * dy
    y_b = s * dx + c * dy
    return torch.stack([x_b, y_b], dim=-1)


def goal_navigation_observation(env) -> torch.Tensor:
    """High-level observation.

    obs = [goal_xy_body, goal_dist, goal_heading_error,
           base_lin_vel_b(2), base_ang_vel_b(3), projected_gravity_b(3), last_action(3)]
    """
    robot = env.scene["robot"]

    goal_xy_body = _goal_delta_body(env)
    goal_dist = _goal_distance(env).unsqueeze(-1)
    goal_heading_error = _goal_heading_error(env).unsqueeze(-1)

    base_lin_vel_b = robot.data.root_link_lin_vel_b[:, :2]
    base_ang_vel_b = robot.data.root_link_ang_vel_b
    projected_gravity_b = robot.data.projected_gravity_b

    if hasattr(env, "action_manager") and env.action_manager is not None:
        last_action = env.action_manager.prev_action
    else:
        last_action = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)

    return torch.cat(
        [
            goal_xy_body,        # 2
            goal_dist,           # 1
            goal_heading_error,  # 1
            base_lin_vel_b,      # 2
            base_ang_vel_b,      # 3
            projected_gravity_b, # 3
            last_action,         # 3
        ],
        dim=-1,
    )


def reward_goal_progress(env) -> torch.Tensor:
    dist = _goal_distance(env)
    if not hasattr(env, "_goal_prev_dist"):
        env._goal_prev_dist = dist.detach().clone()
    progress = (env._goal_prev_dist - dist).clamp(-1.0, 1.0)
    env._goal_prev_dist = dist.detach().clone()
    return progress


def reward_goal_reach(env, sigma: float = 1.5) -> torch.Tensor:
    dist = _goal_distance(env)
    return torch.exp(-(dist ** 2) / (sigma ** 2))


def reward_goal_facing(env, sigma: float = 0.6) -> torch.Tensor:
    err = _goal_heading_error(env)
    return torch.exp(-(err ** 2) / (sigma ** 2))


def reward_goal_velocity(env) -> torch.Tensor:
    robot = env.scene["robot"]
    delta_b = _goal_delta_body(env)
    dist = _goal_distance(env).unsqueeze(-1)
    goal_dir_b = delta_b / dist

    vel_b = robot.data.root_link_lin_vel_b[:, :2]
    proj = torch.sum(vel_b * goal_dir_b, dim=1)
    return torch.clamp(proj, 0.0, 2.0)


def goal_reached(env, threshold: float = 0.25) -> torch.Tensor:
    return _goal_distance(env) < threshold


def reset_goal_marker(env, env_ids, radius_range: tuple[float, float] = (1.5, 3.0)) -> None:
    """Reset the goal marker around each env origin."""
    goal = env.scene["goal_marker"]
    robot = env.scene["robot"]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)

    root_state = goal.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]

    radius = torch.empty((len(env_ids),), device=env.device).uniform_(radius_range[0], radius_range[1])
    theta = torch.empty((len(env_ids),), device=env.device).uniform_(-math.pi, math.pi)

    goal_xy = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)

    root_state[:, 0] = origins[:, 0] + goal_xy[:, 0]
    root_state[:, 1] = origins[:, 1] + goal_xy[:, 1]
    root_state[:, 2] = 0.08
    root_state[:, 3:7] = goal.data.default_root_state[env_ids, 3:7]
    root_state[:, 7:] = 0.0

    goal.write_root_state_to_sim(root_state, env_ids=env_ids)
    goal.reset(env_ids)

    robot_xy = robot.data.root_pos_w[env_ids, :2]
    goal_xy_world = origins[:, :2] + goal_xy
    env._goal_prev_dist = torch.norm(goal_xy_world - robot_xy, dim=1)


# ============================================================================
# Custom Action Term: high-level velocity -> frozen low-level locomotion policy
# ============================================================================
@configclass
class HighLevelVelocityActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = MISSING

    low_level_policy_path: str = MISSING
    velocity_scale: tuple[float, float, float] = (2.0, 0.9, 1.8)
    joint_action_scale: float = 0.25
    joint_vel_scale: float = 0.05


class HighLevelVelocityAction(ActionTerm):
    """Convert high-level command (vx, vy, wz) to joint targets through a frozen low-level policy."""

    def __init__(self, cfg: HighLevelVelocityActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._robot = self._env.scene[self.cfg.asset_name]
        self._low_policy = _load_callable_policy(self.cfg.low_level_policy_path, self.device)

        if hasattr(self._low_policy, "eval"):
            self._low_policy.eval()

        if hasattr(self._low_policy, "parameters"):
            try:
                for p in self._low_policy.parameters():
                    p.requires_grad = False
            except Exception:
                pass

        self._num_joints = int(self._robot.data.joint_pos.shape[1])
        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._velocity_scale = torch.tensor(self.cfg.velocity_scale, device=self.device, dtype=torch.float32)

        self._joint_targets = self._robot.data.default_joint_pos.clone()
        self._prev_low_action = torch.zeros((self.num_envs, self._num_joints), device=self.device)
        self._joint_offset = self._robot.data.default_joint_pos.clone()

        joint_limits = getattr(self._robot.data, "soft_joint_pos_limits", None)
        if joint_limits is None:
            joint_limits = self._robot.data.joint_pos_limits
        self._joint_limits = joint_limits.clone()

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._prev_low_action[env_ids] = 0.0
        self._joint_targets[env_ids] = self._joint_offset[env_ids]

    def process_actions(self, actions: torch.Tensor):
        # Normalize high-level action to [-1, 1] then map to real velocity command.
        self._raw_actions[:] = actions.to(self.device)
        cmd = torch.clamp(self._raw_actions, -1.0, 1.0) * self._velocity_scale
        self._processed_actions[:] = cmd

        robot = self._robot

        # Match the low-level locomotion policy observation layout:
        # base_ang_vel, projected_gravity, command, joint_pos_rel, joint_vel_rel, last_action
        base_ang_vel = robot.data.root_link_ang_vel_b * 0.2
        projected_gravity = robot.data.projected_gravity_b
        joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel_rel = robot.data.joint_vel * self.cfg.joint_vel_scale

        low_obs = torch.cat(
            [
                base_ang_vel,          # 3
                projected_gravity,     # 3
                cmd,                   # 3
                joint_pos_rel,         # nj
                joint_vel_rel,         # nj
                self._prev_low_action, # nj
            ],
            dim=-1,
        ).float()

        with torch.no_grad():
            low_action = self._low_policy(low_obs)

        if isinstance(low_action, dict):
            for key in ("action", "actions", "policy", "logits"):
                if key in low_action:
                    low_action = low_action[key]
                    break
        elif isinstance(low_action, (tuple, list)):
            low_action = low_action[0]

        low_action = torch.as_tensor(low_action, device=self.device, dtype=torch.float32)

        if low_action.shape[-1] != self._num_joints:
            raise RuntimeError(
                f"Low-level policy output shape mismatch: expected last dim {self._num_joints}, got {tuple(low_action.shape)}"
            )

        low_action = torch.clamp(low_action, -1.0, 1.0)

        joint_targets = self._joint_offset + self.cfg.joint_action_scale * low_action
        joint_targets = torch.clamp(
            joint_targets,
            self._joint_limits[..., 0],
            self._joint_limits[..., 1],
        )

        self._joint_targets[:] = joint_targets
        self._prev_low_action[:] = low_action

    def apply_actions(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets)
        self._robot.write_data_to_sim()


HighLevelVelocityActionCfg.class_type = HighLevelVelocityAction


# ============================================================================
# Scene
# ============================================================================
@configclass
class Cyberdog2SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=2.0,
            dynamic_friction=1.8,
            restitution=0.3,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = CYBERDOG2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Kept for future contact-based shaping / push tasks.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # Kinematic marker for a stable first-stage high-level task.
    goal_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/goal_marker",
        spawn=sim_utils.SphereCfg(
            radius=0.08,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 1.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.08),
        ),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================================
# Events
# ============================================================================
@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-0.5, 1.0),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_goal_marker = EventTerm(
        func=reset_goal_marker,
        mode="reset",
        params={
            "radius_range": (1.5, 3.0),
        },
    )


# ============================================================================
# Observations
# ============================================================================
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        goal_obs = ObsTerm(
            func=goal_navigation_observation,
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ============================================================================
# Rewards
# ============================================================================
@configclass
class RewardsCfg:
    goal_progress = RewTerm(func=reward_goal_progress, weight=4.0)
    goal_reach = RewTerm(func=reward_goal_reach, weight=2.0, params={"sigma": 1.5})
    goal_facing = RewTerm(func=reward_goal_facing, weight=1.0, params={"sigma": 0.6})
    goal_velocity = RewTerm(func=reward_goal_velocity, weight=1.5)

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)


# ============================================================================
# Terminations
# ============================================================================
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=goal_reached, params={"threshold": 0.25})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.9})


# ============================================================================
# Actions
# ============================================================================
@configclass
class ActionsCfg:
    high_level_cmd = HighLevelVelocityActionCfg(
        asset_name="robot",
        low_level_policy_path=LOW_LEVEL_POLICY_PATH,
        velocity_scale=(2.0, 0.9, 1.8),
        joint_action_scale=0.25,
        joint_vel_scale=0.05,
        clip={".*": (-1.0, 1.0)},
    )


# ============================================================================
# Env config
# ============================================================================
@configclass
class Cyberdog2HighLevelEnvCfg(ManagerBasedRLEnvCfg):
    scene: Cyberdog2SceneCfg = Cyberdog2SceneCfg(
        num_envs=2048,
        env_spacing=8.0,
    )

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15