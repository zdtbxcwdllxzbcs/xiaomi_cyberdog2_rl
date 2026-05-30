from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command config with a curriculum expansion limit."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
