"""Play a trained RSL-RL Cyberdog2 policy."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import pathlib
import sys

from packaging import version as semver

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT_DIR / "source" / "cyberdog2_rl_lab"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Play a trained RSL-RL Cyberdog2 policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record a play video.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Use USD I/O instead of fabric.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of simulated environments.")
parser.add_argument("--task", type=str, default="Cyberdog2-Velocity-Flat-Play-v0", help="Gym task id.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use a published checkpoint if available.")
parser.add_argument("--real-time", action="store_true", default=False, help="Try to run in real time.")
parser.add_argument("--steps", type=int, default=None, help="Exit after a fixed number of simulation steps.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _prefer_d3d12_on_windows(args) -> None:
    """Force D3D12 on Windows unless the caller already overrode the graphics backend."""

    if sys.platform != "win32":
        return
    kit_args = getattr(args, "kit_args", "") or ""
    if "/app/vulkan=" in kit_args:
        return
    args.kit_args = (kit_args + " --/app/vulkan=false").strip()


_prefer_d3d12_on_windows(args_cli)

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path

import cyberdog2_rl_lab.tasks  # noqa: F401
from cyberdog2_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="env_cfg_entry_point",
    )
    agent_cfg: RslRlBaseRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] No published checkpoint is available for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    installed_version = version("rsl-rl-lib")
    if semver.parse(installed_version) >= semver.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if semver.parse(installed_version) >= semver.parse("4.0.0") and hasattr(policy, "reset"):
                policy.reset(dones)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        elif args_cli.steps is not None:
            timestep += 1
            if timestep >= args_cli.steps:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
