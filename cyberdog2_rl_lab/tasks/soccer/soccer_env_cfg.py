import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from cyberdog2_rl_lab.assets.robots import CYBERDOG2_CFG
from cyberdog2_rl_lab.tasks.soccer.field_config import default_field_config_path, load_field_config
from cyberdog2_rl_lab.tasks.soccer.mdp import AGENTS, GLOBAL_STATE_DIM, POLICY_OBSERVATION_DIM


FIELD_CFG = load_field_config()
LOCOMOTION_POLICY_BASE = r"F:\dog\cyberdog2_rl_lab\logs\rsl_rl\cyberdog2_velocity_flat\2026-05-27_05-04-31"


def _quat_from_yaw_tuple(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _robot_cfg(name: str, x: float, y: float, yaw_quat: tuple[float, float, float, float]) -> ArticulationCfg:
    base_cfg = CYBERDOG2_CFG.replace(prim_path=f"/World/envs/env_.*/{name}")
    return base_cfg.replace(
        spawn=base_cfg.spawn.replace(activate_contact_sensors=False),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(x, y, 0.35),
            rot=yaw_quat,
            joint_pos=CYBERDOG2_CFG.init_state.joint_pos,
            joint_vel={".*": 0.0},
        )
    )


@configclass
class SoccerEnvCfg(DirectMARLEnvCfg):
    """2v2 soccer with high-level velocity commands and frozen low-level locomotion."""

    decimation = 4
    episode_length_s = 30.0
    possible_agents = ("blue_attacker",)
    action_spaces = {"blue_attacker": 3}
    observation_spaces = {"blue_attacker": POLICY_OBSERVATION_DIM}
    state_space = GLOBAL_STATE_DIM

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim.physics_material,
        debug_vis=False,
    )

    blue_attacker: ArticulationCfg = _robot_cfg(
        "BlueAttacker", *FIELD_CFG.player_starts["blue_attacker"][:2],
        _quat_from_yaw_tuple(FIELD_CFG.player_starts["blue_attacker"][2])
    )
    blue_goalkeeper: ArticulationCfg = _robot_cfg(
        "BlueGoalkeeper", *FIELD_CFG.player_starts["blue_goalkeeper"][:2],
        _quat_from_yaw_tuple(FIELD_CFG.player_starts["blue_goalkeeper"][2])
    )
    red_attacker: ArticulationCfg = _robot_cfg(
        "RedAttacker", *FIELD_CFG.player_starts["red_attacker"][:2],
        _quat_from_yaw_tuple(FIELD_CFG.player_starts["red_attacker"][2])
    )
    red_goalkeeper: ArticulationCfg = _robot_cfg(
        "RedGoalkeeper", *FIELD_CFG.player_starts["red_goalkeeper"][:2],
        _quat_from_yaw_tuple(FIELD_CFG.player_starts["red_goalkeeper"][2])
    )

    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.95, restitution=0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=2.5,
                angular_damping=2.5,
                max_linear_velocity=8.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=2.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.8, 0.13), rot=(1.0, 0.0, 0.0, 0.0)),  # 中线附近，略微靠近蓝方进攻球门
    )

    # -------------------------------------------------------------------------
    # 🔥 旧版 IsaacLab 正确写法！无 entities 参数，直接赋值物体
    # -------------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=12.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    
    FIELD_LENGTH = 10.0
    FIELD_WIDTH = 5.5
    WALL_THICK = 0.1
    WALL_HEIGHT = 0.8
    GOAL_WIDTH = FIELD_CFG.goal_width
    GOAL_INNER_DEPTH = 1.5

    field_wall_top: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/field_wall_top",
        spawn=sim_utils.CuboidCfg(
            size=(WALL_THICK, FIELD_LENGTH, WALL_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-FIELD_WIDTH/2, 0.0, WALL_HEIGHT/2)),
    )
    field_wall_bot: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/field_wall_bot",
        spawn=sim_utils.CuboidCfg(
            size=(WALL_THICK, FIELD_LENGTH, WALL_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(FIELD_WIDTH/2, 0.0, WALL_HEIGHT/2)),
    )
    field_wall_left_upper: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/field_wall_left_upper",
        spawn=sim_utils.CuboidCfg(
            size=(0.5 * (FIELD_WIDTH - GOAL_WIDTH), WALL_THICK, WALL_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-(FIELD_WIDTH + GOAL_WIDTH) / 4.0, -FIELD_LENGTH / 2 + WALL_THICK / 2, WALL_HEIGHT / 2)
        ),
    )
    field_wall_left_lower: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/field_wall_left_lower",
        spawn=sim_utils.CuboidCfg(
            size=(0.5 * (FIELD_WIDTH - GOAL_WIDTH), WALL_THICK, WALL_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=((FIELD_WIDTH + GOAL_WIDTH) / 4.0, -FIELD_LENGTH / 2 + WALL_THICK / 2, WALL_HEIGHT / 2)
        ),
    )
    field_wall_right_upper: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/field_wall_right_upper",
        spawn=sim_utils.CuboidCfg(
            size=(0.5 * (FIELD_WIDTH - GOAL_WIDTH), WALL_THICK, WALL_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-(FIELD_WIDTH + GOAL_WIDTH) / 4.0, FIELD_LENGTH / 2 - WALL_THICK / 2, WALL_HEIGHT / 2)
        ),
    )
    field_wall_right_lower: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/field_wall_right_lower",
        spawn=sim_utils.CuboidCfg(
            size=(0.5 * (FIELD_WIDTH - GOAL_WIDTH), WALL_THICK, WALL_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=((FIELD_WIDTH + GOAL_WIDTH) / 4.0, FIELD_LENGTH / 2 - WALL_THICK / 2, WALL_HEIGHT / 2)
        ),
    )

    goal_blue_left_wall: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/goal_blue_left_wall",
            spawn=sim_utils.CuboidCfg(
                size=(WALL_THICK, 0.5, WALL_HEIGHT),  # 👈 改：深度固定0.5m（浅球门）
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-GOAL_WIDTH/2, -FIELD_LENGTH/2 + 0.5/2, WALL_HEIGHT/2)  # 👈 改：匹配浅球门位置
            ),
        )
    goal_blue_right_wall: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/goal_blue_right_wall",
            spawn=sim_utils.CuboidCfg(
                size=(WALL_THICK, 0.5, WALL_HEIGHT),  # 👈 改：深度固定0.5m
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(GOAL_WIDTH/2, -FIELD_LENGTH/2 + 0.5/2, WALL_HEIGHT/2)  # 👈 改：匹配浅球门位置
            ),
        )

    goal_red_left_wall: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/goal_red_left_wall",
            spawn=sim_utils.CuboidCfg(
                size=(WALL_THICK, 0.5, WALL_HEIGHT),  # 👈 改：深度固定0.5m
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-GOAL_WIDTH/2, FIELD_LENGTH/2 - 0.5/2, WALL_HEIGHT/2)  # 👈 改：匹配浅球门位置
            ),
        )
    goal_red_right_wall: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/goal_red_right_wall",
            spawn=sim_utils.CuboidCfg(
                size=(WALL_THICK, 0.5, WALL_HEIGHT),  # 深度固定0.5m
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(GOAL_WIDTH/2, FIELD_LENGTH/2 - 0.5/2, WALL_HEIGHT/2)  # 匹配浅球门位置
            ),
        )

    goal_line_blue: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/goal_line_blue",
            spawn=sim_utils.CuboidCfg(
                size=(GOAL_WIDTH, 0.02, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=None,  # 👈 移除碰撞，只是视觉标记
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, -FIELD_LENGTH/2 + 0.5, 0.025))  # 球门内侧末端（深度0.5m）
        )
    goal_line_red: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/goal_line_red",
            spawn=sim_utils.CuboidCfg(
                size=(GOAL_WIDTH, 0.02, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=None,  # 👈 移除碰撞，只是视觉标记
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, FIELD_LENGTH/2 - 0.5, 0.025))  # 球门内侧末端（深度0.5m）
        )

    field_config_path = default_field_config_path()
    field_x_width = FIELD_CFG.x_width
    field_y_length = FIELD_CFG.y_length
    goal_width = FIELD_CFG.goal_width
    blue_attack_sign_y = FIELD_CFG.blue_attack_sign_y
    red_attack_sign_y = FIELD_CFG.red_attack_sign_y
    player_starts = FIELD_CFG.player_starts
    high_level_action_scale = (0.8, 0.5, 1.5)
    command_warmup_steps = 40
    locomotion_policy_path = str(LOCOMOTION_POLICY_BASE + r"\exported\policy.pt")
    locomotion_deploy_cfg_path = str(LOCOMOTION_POLICY_BASE + r"\params\deploy.yaml")
    controller_type = "locomotion_policy"

    phase = 3
    skill_phase = 3
    out_margin = 0.35
    ball_radius = 0.11

    rew_goal = 100.0
    rew_concede = -15.0
    rew_ball_progress = 200.0
    rew_ball_speed = 0.5
    rew_attacker_to_ball = 0.3
    rew_attacker_behind_ball = 0.35
    rew_goalkeeper_position = 0.25
    rew_goalkeeper_clear = 0.15
    rew_upright = 5.0
    rew_command_rate = -0.2
    rew_out_of_field = -0.15
    rew_self_wall = -5.0
    rew_facing_goal = 0.05
    rew_dribbling = 0.4
    rew_approaching_goal = 0.15
    rew_survival = 0.08


@configclass
class SoccerEnvCfg_PLAY(SoccerEnvCfg):
    """Small play config."""
    def __post_init__(self):
        self.scene.num_envs = 16


@configclass
class SoccerEnvCfg_SINGLE(SoccerEnvCfg):
    """Single-agent config exposing only the blue attacker for single-dog training."""
    possible_agents = ("blue_attacker",)
    action_spaces = {"blue_attacker": 3}
    observation_spaces = {"blue_attacker": POLICY_OBSERVATION_DIM}


@configclass
class SoccerEnvCfg_SINGLE_PLAY(SoccerEnvCfg_SINGLE):
    """Small play config for single-agent setup."""
    def __post_init__(self):
        self.scene.num_envs = 16
