from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def parse_env_cfg(
    task_name: str,
    device: str = "cuda:0",
    num_envs: int | None = None,
    use_fabric: bool | None = None,
    entry_point_key: str = "env_cfg_entry_point",
) -> ManagerBasedRLEnvCfg | DirectRLEnvCfg:
    """Load and override an environment config from the gym registry."""

    cfg = load_cfg_from_registry(task_name, entry_point_key)
    if isinstance(cfg, dict):
        raise RuntimeError(f"Configuration for task '{task_name}' is not a class.")
    cfg.sim.device = device
    if use_fabric is not None:
        cfg.sim.use_fabric = use_fabric
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    return cfg
