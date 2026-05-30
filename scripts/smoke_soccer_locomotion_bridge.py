"""Smoke checks for soccer field schema and locomotion bridge configuration."""

from __future__ import annotations

import argparse
import pathlib
import sys
import argparse
from isaacsim import SimulationApp

# 创建 SimulationApp 实例，headless=True 表示无界面模式
simulation_app = SimulationApp({"headless": False})

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "source" / "cyberdog2_rl_lab"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import numpy as np
import torch

from cyberdog2_rl_lab.runtime.soccer_policy_runner import SoccerObservationBuilder
from cyberdog2_rl_lab.runtime.vrpn_state_receiver import RigidBodyState, SoccerWorldState
from cyberdog2_rl_lab.tasks.soccer.field_config import load_field_config
from cyberdog2_rl_lab.tasks.soccer.locomotion_policy_controller import resolve_locomotion_artifacts
from cyberdog2_rl_lab.tasks.soccer.mdp.terminations import ball_out_of_bounds, goal_scored
from cyberdog2_rl_lab.tasks.soccer.observation_schema import POLICY_OBSERVATION_DIM, POLICY_OBSERVATION_FIELDS


def _mock_world() -> SoccerWorldState:
    return SoccerWorldState(
        blue_attacker=RigidBodyState(x=-0.5, y=-1.0, yaw=0.2, vx=0.1, vy=0.2, wz=0.3),
        blue_goalkeeper=RigidBodyState(x=0.0, y=-4.2, yaw=1.57),
        red_attacker=RigidBodyState(x=0.6, y=1.0, yaw=-0.3, vx=-0.1),
        red_goalkeeper=RigidBodyState(x=0.0, y=4.2, yaw=-1.57),
        ball=RigidBodyState(x=0.1, y=0.2, vx=0.4, vy=-0.2),
        blue_goal=RigidBodyState(x=0.0, y=-5.0),
        red_goal=RigidBodyState(x=0.0, y=5.0),
    )


def run_smoke() -> None:
    field = load_field_config()
    assert field.x_width == 5.55
    assert field.y_length == 10.0

    ball_xy = torch.tensor([[0.0, 5.01], [0.0, -5.01], [0.7, 5.01], [3.2, 0.0], [0.0, 5.4]])
    blue_scored, red_scored = goal_scored(ball_xy, field.y_length, field.goal_width)
    out = ball_out_of_bounds(ball_xy, field.x_width, field.y_length, margin=0.35)
    assert blue_scored.tolist() == [True, False, False, False, True]
    assert red_scored.tolist() == [False, True, False, False, False]
    assert out.tolist() == [False, False, False, True, True]

    assert POLICY_OBSERVATION_DIM == len(POLICY_OBSERVATION_FIELDS)
    builder = SoccerObservationBuilder(action_scale=tuple(field.normalization["action_scale"]))
    obs = builder.build_agent_observation(_mock_world(), "blue_attacker")
    assert obs.shape == (POLICY_OBSERVATION_DIM,)
    assert obs.dtype == np.float32
    joint_obs = builder.build_joint_observation(_mock_world())
    assert joint_obs.shape == (POLICY_OBSERVATION_DIM * 4,)

    policy_path, deploy_path = resolve_locomotion_artifacts()
    assert policy_path.exists()
    assert deploy_path.exists()
    print("soccer locomotion bridge smoke checks passed", flush=True)
    print(f"policy: {policy_path}", flush=True)
    print(f"deploy: {deploy_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke check soccer locomotion bridge semantics.")
    parser.parse_args()
    run_smoke()
