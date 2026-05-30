"""Train an RSL-RL soccer policy with 4-phase Flamez-aligned curriculum.

Phase 1: empty field (only blue attacker + ball)
Phase 2: + scripted red goalkeeper
Phase 3: + scripted blue goalkeeper + scripted red attacker (full 2v2)
Phase 4: full 2v2 with latency randomization
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT_DIR / "source" / "cyberdog2_rl_lab"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Train an RSL-RL soccer policy (Flamez-aligned 4-phase curriculum).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Recorded video interval in steps.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of simulated environments.")
parser.add_argument("--task", type=str, default="Cyberdog2-Soccer-2v2-Play-v0", help="Gym task id.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum training iterations per phase.")
parser.add_argument("--distributed", action="store_true", default=False, help="Enable distributed training.")
parser.add_argument("--curriculum", type=str, default="all",
                    choices=["all", "base", "phase1", "phase2", "phase3", "phase4"],
                    help="Which phases to train. 'base' = phase1+phase2. 'all' = phase1-4.")
parser.add_argument("--phase1_iterations", type=int, default=500, help="Iterations for phase 1.")
parser.add_argument("--phase2_iterations", type=int, default=500, help="Iterations for phase 2.")
parser.add_argument("--phase3_iterations", type=int, default=1000, help="Iterations for phase 3.")
parser.add_argument("--phase4_iterations", type=int, default=300, help="Iterations for phase 4.")
parser.add_argument("--play_interval", type=int, default=0, help="Save checkpoint and print play command every N iterations. 0 = disabled.")
parser.add_argument("--resume_path", type=str, default=None, help="Path to a checkpoint .pt file to resume training from.")
parser.add_argument("--skill_phase", type=int, default=3, choices=[1, 2, 3],
                    help="Skill curriculum phase: 1=find ball+behind, 2=+dribble, 3=+score. Default 3 = all active.")
parser.add_argument("--skill_phase_end", type=int, default=None, choices=[1, 2, 3],
                    help="End skill phase (inclusive). Default = same as --skill_phase (run only one step).")
parser.add_argument("--skill1_iterations", type=int, default=10000, help="Iterations for skill phase 1.")
parser.add_argument("--skill2_iterations", type=int, default=10000, help="Iterations for skill phase 2.")
parser.add_argument("--skill3_iterations", type=int, default=30000, help="Iterations for skill phase 3.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _prefer_d3d12_on_windows(args) -> None:
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

import importlib.metadata as metadata
import inspect
import os
import platform
import shutil
from datetime import datetime

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import cyberdog2_rl_lab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r"F:\Projects\IsaacLab\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(f"Please install RSL-RL {RSL_RL_VERSION}. Existing version is '{installed_version}'.")
    raise SystemExit(1)

PHASE_CONFIGS = {
    1: {"phase": 1, "description": "empty field"},
    2: {"phase": 2, "description": "+ scripted goalkeeper"},
    3: {"phase": 3, "description": "2v2 adversarial play"},
    4: {"phase": 4, "description": "2v2 with latency"},
}

CURRICULUM_PHASES = {
    "all": (1, 2, 3, 4),
    "base": (1, 2),
    "phase1": (1,),
    "phase2": (2,),
    "phase3": (3,),
    "phase4": (4,),
}

PHASE_ITERATIONS = {
    1: "phase1_iterations",
    2: "phase2_iterations",
    3: "phase3_iterations",
    4: "phase4_iterations",
}

SKILL_DESCRIPTIONS = {
    1: "find ball + get behind",
    2: "+ dribble + face goal",
    3: "+ score goal (all rewards)",
}

SKILL_ITERATIONS = {
    1: "skill1_iterations",
    2: "skill2_iterations",
    3: "skill3_iterations",
}


def _make_env(task: str, env_cfg, phase: int):
    env_cfg.phase = phase
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    return env


def _fix_observation(env) -> None:
    ma_env = env.unwrapped.env if hasattr(env.unwrapped, 'env') else env.unwrapped
    orig = ma_env._get_observations

    def _flat():
        obs_dict = orig()
        return {"policy": obs_dict["blue_attacker"]}

    env.unwrapped._get_observations = _flat


def _train_phase(env_cfg, agent_cfg, phase: int, resume_path: str | None, skill_phase: int = None):
    desc = PHASE_CONFIGS[phase]["description"]
    skill_tag = f"_skill{skill_phase}" if skill_phase else ""
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_phase{phase}{skill_tag}"
    log_dir = os.path.join(log_root, log_dir_name)
    sp = skill_phase if skill_phase else args_cli.skill_phase
    print(f"\n{'='*60}")
    print(f"  Phase {phase} ({desc}), Skill {sp}: {SKILL_DESCRIPTIONS[sp]}")
    print(f"  Log dir: {log_dir}")
    print(f"{'='*60}\n")

    env_cfg.phase = phase
    env_cfg.skill_phase = sp
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = agent_cfg.seed

    iterations = getattr(args_cli, SKILL_ITERATIONS[sp])
    if args_cli.max_iterations is not None:
        iterations = args_cli.max_iterations
    agent_cfg.max_iterations = iterations

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        marl_env = env.unwrapped
        env = multi_agent_to_single_agent(env)
        env._get_observations = lambda: {"policy": marl_env._get_observations()["blue_attacker"]}

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env.reset()

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    _orig_update = runner.alg.update
    _MAX_STD = 1.8
    _MIN_STD = 0.35
    def _clamped_update(*args, **kwargs):
        loss_dict = _orig_update(*args, **kwargs)
        policy = runner.alg.get_policy()
        if hasattr(policy, 'distribution') and hasattr(policy.distribution, 'std_param'):
            policy.distribution.std_param.data.clamp_(min=_MIN_STD, max=_MAX_STD)
        return loss_dict
    runner.alg.update = _clamped_update

    if resume_path:
        print(f"[INFO] Loading checkpoint from phase {phase-1}: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    play_interval = getattr(args_cli, "play_interval", 0)
    if play_interval > 0:
        _orig_learn = runner.learn

        def _learn_with_play(num_learning_iterations, init_at_random_ep_len=False):
            import time as _time
            if init_at_random_ep_len:
                env.unwrapped.episode_length_buf = torch.randint_like(
                    env.unwrapped.episode_length_buf, high=int(env.unwrapped.max_episode_length)
                )
            obs = env.get_observations().to(runner.device)
            runner.alg.train_mode()
            runner.logger.init_logging_writer()
            start_it = runner.current_learning_iteration
            total_it = start_it + num_learning_iterations
            for it in range(start_it, total_it):
                t0 = _time.time()
                with torch.inference_mode():
                    for _ in range(runner.cfg["num_steps_per_env"]):
                        actions = runner.alg.act(obs)
                        obs, rewards, dones, extras = env.step(actions.to(env.device))
                        obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                        runner.alg.process_env_step(obs, rewards, dones, extras)
                        runner.logger.process_env_step(rewards, dones, extras, None)
                    collect_time = _time.time() - t0
                    t0 = _time.time()
                    runner.alg.compute_returns(obs)
                loss_dict = runner.alg.update()
                learn_time = _time.time() - t0
                runner.current_learning_iteration = it
                runner.logger.log(
                    it=it, start_it=start_it, total_it=total_it,
                    collect_time=collect_time, learn_time=learn_time,
                    loss_dict=loss_dict, learning_rate=runner.alg.learning_rate,
                    action_std=runner.alg.get_policy().output_std,
                    rnd_weight=None,
                )
                if runner.logger.writer is not None and it % runner.cfg["save_interval"] == 0:
                    runner.save(os.path.join(runner.logger.log_dir, f"model_{it}.pt"))
                if play_interval > 0 and it > 0 and it % play_interval == 0:
                    ckpt_path = os.path.join(runner.logger.log_dir, f"model_{it}.pt")
                    runner.save(ckpt_path)
                    print(f"\n{'='*60}")
                    print(f"  [AUTO-PLAY] Iteration {it}/{total_it}")
                    print(f"  Checkpoint: {ckpt_path}")
                    print(f"  Run manually: python scripts\\rsl_rl\\play_soccer.py --task Cyberdog2-Soccer-2v2-Play-v0 --phase 1 --num_envs 4 --load_run {os.path.basename(runner.logger.log_dir)} --checkpoint model_{it}.pt")
                    print(f"{'='*60}\n")
            if runner.logger.writer is not None:
                runner.save(os.path.join(runner.logger.log_dir, f"model_{runner.current_learning_iteration}.pt"))
                runner.logger.stop_logging_writer()

        runner.learn = _learn_with_play

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()

    checkpoint_dir = os.path.join(log_dir, "model")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"phase{phase}_skill{sp}_final.pt")
    runner.save(checkpoint_path)
    print(f"[INFO] Phase {phase} Skill {sp} checkpoint saved: {checkpoint_path}")

    return checkpoint_path


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    phases = CURRICULUM_PHASES[args_cli.curriculum]
    print(f"[INFO] Curriculum: {' → '.join(f'Phase {p}' for p in phases)}")

    resume_path = getattr(args_cli, "resume_path", None)
    for phase in phases:
        start_skill = args_cli.skill_phase
        end_skill = args_cli.skill_phase_end if args_cli.skill_phase_end is not None else start_skill
        for sp in range(start_skill, end_skill + 1):
            resume_path = _train_phase(env_cfg, agent_cfg, phase, resume_path, skill_phase=sp)
            if sp < end_skill:
                print(f"\n[SKILL-CURRICULUM] Skill {sp} done, advancing to skill {sp+1}...\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
