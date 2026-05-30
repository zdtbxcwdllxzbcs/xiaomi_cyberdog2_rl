"""Headless smoke test for the new official locomotion and soccer tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "source" / "cyberdog2_rl_lab"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test the direct Cyberdog2 tasks.")
parser.add_argument(
    "--which",
    type=str,
    default="both",
    choices=["official", "soccer", "both"],
    help="Select which task smoke test to run.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _prefer_d3d12_on_windows(args) -> None:
    if sys.platform != "win32":
        return
    kit_args = getattr(args, "kit_args", "") or ""
    if "/app/vulkan=" in kit_args:
        return
    args.kit_args = (kit_args + " --/app/vulkan=false").strip()


_prefer_d3d12_on_windows(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import cyberdog2_rl_lab.tasks  # noqa: F401
from cyberdog2_rl_lab.tasks.official_locomotion.official_locomotion_env_cfg import OfficialLocomotionEnvCfg_PLAY
from cyberdog2_rl_lab.tasks.soccer.soccer_env_cfg import SoccerEnvCfg_PLAY


def run_official_locomotion_smoke():
    print("starting official locomotion smoke", flush=True)
    cfg = OfficialLocomotionEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device if args_cli.device is not None else cfg.sim.device
    print(f"official device: {cfg.sim.device}", flush=True)
    env = gym.make("Cyberdog2-Official-Locomotion-Play-v0", cfg=cfg)
    print("official env created", flush=True)
    obs, _ = env.reset()
    print("official reset done", flush=True)
    action = torch.zeros(1, 3, device=env.unwrapped.device)
    obs, rew, terminated, truncated, _ = env.step(action)
    print("official:", tuple(obs["policy"].shape), tuple(rew.shape), tuple(terminated.shape), tuple(truncated.shape), flush=True)
    env.close()
    print("official env closed", flush=True)


def run_soccer_smoke():
    print("starting soccer smoke", flush=True)
    cfg = SoccerEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device if args_cli.device is not None else cfg.sim.device
    print(f"soccer device: {cfg.sim.device}", flush=True)
    env = gym.make("Cyberdog2-Soccer-2v2-Play-v0", cfg=cfg)
    print("soccer env created", flush=True)
    obs, _ = env.reset()
    print("soccer reset done", flush=True)
    actions = {agent: torch.zeros(1, 3, device=env.unwrapped.device) for agent in env.unwrapped.possible_agents}
    obs, rewards, terminated, truncated, _ = env.step(actions)
    print("soccer agents:", list(obs.keys()), flush=True)
    print("soccer obs:", {agent: tuple(value.shape) for agent, value in obs.items()}, flush=True)
    print("soccer rew:", {agent: tuple(value.shape) for agent, value in rewards.items()}, flush=True)
    print("soccer done:", {agent: tuple(value.shape) for agent, value in terminated.items()}, flush=True)
    env.close()


if __name__ == "__main__":
    if args_cli.which in ("official", "both"):
        run_official_locomotion_smoke()
    if args_cli.which in ("soccer", "both"):
        run_soccer_smoke()
    simulation_app.close()
