"""DirectRLEnv config for official Cyberdog2 velocity-command smoke tests."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from cyberdog2_rl_lab.assets.robots import CYBERDOG2_CFG


@configclass
class OfficialLocomotionEnvCfg(DirectRLEnvCfg):
    """Cyberdog2 task that exposes official high-level velocity command semantics."""

    decimation = 4
    episode_length_s = 20.0
    action_space = 3
    observation_space = 18
    state_space = 0

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim.physics_material,
        debug_vis=False,
    )
    robot: ArticulationCfg = CYBERDOG2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    developer_motion_id = 305
    max_lin_acc = 2.5
    max_lat_acc = 2.0
    max_yaw_acc = 6.0
    command_resampling_time_s = 4.0

    target_lin_vel_x = (-1.2, 1.2)
    target_lin_vel_y = (-0.35, 0.35)
    target_ang_vel_z = (-1.8, 1.8)

    rew_track_lin_vel = 2.0
    rew_track_ang_vel = 0.75
    rew_upright = 0.25
    rew_action_rate = -0.05
    rew_command_rate = -0.02


@configclass
class OfficialLocomotionEnvCfg_PLAY(OfficialLocomotionEnvCfg):
    """Small deterministic play config."""

    def __post_init__(self):
        self.scene.num_envs = 32
        self.command_resampling_time_s = 6.0
        self.target_lin_vel_x = (-0.8, 0.8)
        self.target_lin_vel_y = (-0.25, 0.25)
        self.target_ang_vel_z = (-1.2, 1.2)
