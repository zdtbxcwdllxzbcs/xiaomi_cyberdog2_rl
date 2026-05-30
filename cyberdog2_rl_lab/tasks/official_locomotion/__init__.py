"""Official-command Cyberdog2 locomotion smoke tasks."""

import gymnasium as gym

from cyberdog2_rl_lab.tasks.official_locomotion import agents


gym.register(
    id="Cyberdog2-Official-Locomotion-v0",
    entry_point="cyberdog2_rl_lab.tasks.official_locomotion.official_locomotion_env:OfficialLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "cyberdog2_rl_lab.tasks.official_locomotion.official_locomotion_env_cfg:"
            "OfficialLocomotionEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OfficialLocomotionPPORunnerCfg",
    },
)

gym.register(
    id="Cyberdog2-Official-Locomotion-Play-v0",
    entry_point="cyberdog2_rl_lab.tasks.official_locomotion.official_locomotion_env:OfficialLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "cyberdog2_rl_lab.tasks.official_locomotion.official_locomotion_env_cfg:"
            "OfficialLocomotionEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OfficialLocomotionPPORunnerCfg",
    },
)
