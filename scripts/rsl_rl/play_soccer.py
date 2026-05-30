"""Play a trained RSL-RL Cyberdog2 soccer policy."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import pathlib
import sys
import time

from packaging import version as semver

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT_DIR / "source" / "cyberdog2_rl_lab"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Play a trained RSL-RL Cyberdog2 soccer policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record a play video.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Use USD I/O instead of fabric.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of simulated environments.")
parser.add_argument("--task", type=str, default="Cyberdog2-Soccer-Single-Play-v0", help="Gym task id.")
parser.add_argument("--phase", type=int, default=1, help="Curriculum phase (1=empty, 2=+gk, 3=2v2, 4=+latency).")
parser.add_argument("--real-time", action="store_true", default=False, help="Try to run in real time.")
parser.add_argument("--steps", type=int, default=None, help="Exit after a fixed number of simulation steps.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


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

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import cyberdog2_rl_lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # 确定checkpoint路径
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    log_dir = os.path.dirname(resume_path)
    
    env_cfg.phase = args_cli.phase
    print(f"[INFO] Phase: {args_cli.phase}")
    
    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # 如果是多智能体环境，转换为单智能体并修复观测格式
    if isinstance(env.unwrapped, DirectMARLEnv):
        marl_env = env.unwrapped
        env = multi_agent_to_single_agent(env)
        env._get_observations = lambda: {"policy": marl_env._get_observations()["blue_attacker"]}

    # 设置视频录制
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

    # 包装环境
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 加载模型
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # 获取推理策略
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 导出模型
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
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

    # 开始播放
    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    eval_std = 0.5

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            noise = torch.randn_like(actions) * eval_std
            actions = torch.clamp(actions + noise, -1.0, 1.0)
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
