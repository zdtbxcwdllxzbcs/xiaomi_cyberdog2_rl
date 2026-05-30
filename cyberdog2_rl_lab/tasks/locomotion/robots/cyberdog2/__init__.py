import gymnasium as gym

from cyberdog2_rl_lab.tasks.locomotion import agents


gym.register(
    id="Cyberdog2-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:Cyberdog2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Cyberdog2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Cyberdog2-Velocity-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:Cyberdog2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Cyberdog2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Cyberdog2-HighLevel-v0",
    entry_point="cyberdog2_rl_lab.tasks.locomotion.robots.cyberdog2.highlevel_env_cfg:Cyberdog2HighLevelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "cyberdog2_rl_lab.tasks.locomotion.robots.cyberdog2.highlevel_env_cfg:Cyberdog2HighLevelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Cyberdog2FlatPPORunnerCfg",
    },
)


