"""Cyberdog2 2v2 soccer tasks."""

import gymnasium as gym

from cyberdog2_rl_lab.tasks.soccer import agents


gym.register(
    id="Cyberdog2-Soccer-2v2-v0",
    entry_point="cyberdog2_rl_lab.tasks.soccer.soccer_env:SoccerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "cyberdog2_rl_lab.tasks.soccer.soccer_env_cfg:SoccerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SoccerPPORunnerCfg",
    },
)

gym.register(
    id="Cyberdog2-Soccer-2v2-Play-v0",
    entry_point="cyberdog2_rl_lab.tasks.soccer.soccer_env:SoccerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "cyberdog2_rl_lab.tasks.soccer.soccer_env_cfg:SoccerEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SoccerPPORunnerCfg",
    },
)

gym.register(
    id="Cyberdog2-Soccer-Single-v0",
    entry_point="cyberdog2_rl_lab.tasks.soccer.soccer_env:SoccerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "cyberdog2_rl_lab.tasks.soccer.soccer_env_cfg:SoccerEnvCfg_SINGLE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SoccerPPORunnerCfg",
    },
)

gym.register(
    id="Cyberdog2-Soccer-Single-Play-v0",
    entry_point="cyberdog2_rl_lab.tasks.soccer.soccer_env:SoccerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "cyberdog2_rl_lab.tasks.soccer.soccer_env_cfg:SoccerEnvCfg_SINGLE_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SoccerPPORunnerCfg",
    },
)
