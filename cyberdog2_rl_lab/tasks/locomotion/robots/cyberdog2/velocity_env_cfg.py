import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg,RigidObjectCfg,AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from cyberdog2_rl_lab.assets.robots.cyberdog2 import CYBERDOG2_CFG
from cyberdog2_rl_lab.tasks.locomotion import mdp



# =========================
# 放在类外面！！！！
# =========================
FIELD_LENGTH = 10.0
FIELD_WIDTH = 5.55
WALL_THICKNESS = 0.1
WALL_HEIGHT = 0.6


@configclass
class Cyberdog2SceneCfg(InteractiveSceneCfg):

    # =========================
    # 地面
    # =========================
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

    # =========================
    # 机器人
    # =========================
    robot: ArticulationCfg = CYBERDOG2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True
    )

    # =========================
    # 光照
    # =========================
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

 

@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
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

@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.0,   # 关键：0 = 完全不许原地摆烂
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # Motion 305: 快速行走, curriculum 从小范围开始
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(-0.5, 0.5),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.6, 1.6),
            lin_vel_y=(-0.55, 0.55),
            ang_vel_z=(-2.5, 2.5),
        ),
    )
    

@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100.0, 100.0), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100.0, 100.0), params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100.0, 100.0), noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100.0, 100.0), params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100.0, 100.0))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100.0, 100.0))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class RewardsCfg:
    # 超级加倍：跟踪速度奖励，强迫它必须跟上指令、大步走
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=15.0, params={"command_name": "base_velocity", "std": 0.4})
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_exp, weight=8.0, params={"command_name": "base_velocity", "std": 0.4})

    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.005)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-3.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.15)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-2.0e-5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)

    # 关键：降低原地站姿惩罚，允许四肢大范围迈步
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 2.0,
            "velocity_threshold": 0.2,
        },
    )

    # 🔥 提升步频核心1：降低抬脚阈值 → 脚刚离地就落地，步频暴涨
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.8,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.2,  # 避免碎步，鼓励大跨步
        },
    )
    # 🔥 提升步频核心2：加强均匀性惩罚 → 强制快速交替迈步，杜绝慢步
    air_time_variance = RewTerm(func=mdp.air_time_variance_penalty, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")})
    # 🔥 对角步态（trot）奖励：FL+RR 同步，FR+RL 同步，两对交替
    trot_gait = RewTerm(
        func=mdp.trot_gait_reward,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "contact_threshold": 5.0,
        },
    )
    
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")})
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body|head|.*_abad|.*_hip")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"), "threshold": 1.0})
    head_contact = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="head"), "threshold": 1.0})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.9})

@configclass
class CurriculumCfg:
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)

@configclass
class Cyberdog2FlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: Cyberdog2SceneCfg = Cyberdog2SceneCfg(num_envs=2048, env_spacing=12)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

@configclass
class Cyberdog2FlatEnvCfg_PLAY(Cyberdog2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
