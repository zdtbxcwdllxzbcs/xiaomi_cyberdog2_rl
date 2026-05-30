"""RL helpers for striker training."""

from .soccer_env import SoccerEnv, predict_guard_target

# SB3's system-info writer optionally imports ``gym`` while saving checkpoints.
# This local package intentionally exposes a version so that probe does not
# fail when the repo package shadows the old OpenAI Gym module.
__version__ = "0.1.0"

__all__ = ["SoccerEnv", "predict_guard_target"]
