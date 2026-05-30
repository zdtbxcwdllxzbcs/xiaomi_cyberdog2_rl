"""Launch kickoff team striker+goalkeeper against the other goalkeeper.

This starts the normal pygame simulator, then only three strategy processes:
the kickoff team's striker and goalkeeper, plus the other team's goalkeeper.
The non-kickoff striker remains an uncontrolled simulator body.

Usage:
  ros2 launch sim/launch/sim_attack_vs_keeper_pygame.launch.py
  ros2 launch sim/launch/sim_attack_vs_keeper_pygame.launch.py kickoff_team:=team_b
"""

import os
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration


REPO_ROOT = Path(__file__).resolve().parents[2]


STRATEGY_CONFIGS_BY_KICKOFF = {
    "team_a": (
        ("team_a_robot_1", "sim_2v2_team_a_1.yaml"),
        ("team_a_robot_2", "sim_2v2_team_a_2.yaml"),
        ("team_b_robot_2", "sim_2v2_team_b_2.yaml"),
    ),
    "team_b": (
        ("team_b_robot_1", "sim_2v2_team_b_1.yaml"),
        ("team_b_robot_2", "sim_2v2_team_b_2.yaml"),
        ("team_a_robot_2", "sim_2v2_team_a_2.yaml"),
    ),
}


def _python_env():
    pythonpath = [str(REPO_ROOT)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath.append(existing)
    return {"PYTHONPATH": os.pathsep.join(pythonpath)}


def _python_process(script: Path, *args, delay=None, name=None):
    process = ExecuteProcess(
        cmd=[sys.executable, "-u", str(script), *args],
        name=name,
        output="screen",
        additional_env=_python_env(),
    )
    if delay is None:
        return process
    return TimerAction(period=delay, actions=[process])


def _strategy_process(name, config_file):
    config = REPO_ROOT / "config" / config_file
    return ExecuteProcess(
        cmd=[sys.executable, "-u", str(REPO_ROOT / "main.py"), "--config", str(config)],
        name=f"strategy_{name}",
        output="screen",
        additional_env=_python_env(),
    )


def _strategy_timer(context):
    kickoff_team = LaunchConfiguration("kickoff_team").perform(context).strip().lower()
    strategy_configs = STRATEGY_CONFIGS_BY_KICKOFF.get(kickoff_team)
    if strategy_configs is None:
        raise ValueError("kickoff_team must be team_a or team_b")
    return [
        TimerAction(
            period=3.0,
            actions=[_strategy_process(name, config) for name, config in strategy_configs],
        )
    ]


def generate_launch_description():
    kickoff_team = LaunchConfiguration("kickoff_team")

    pygame_sim = _python_process(
        REPO_ROOT / "sim" / "pygame_sim.py",
        "--ros-args",
        "-p",
        ["kickoff_team:=", kickoff_team],
        name="pygame_sim",
    )

    strategies = OpaqueFunction(function=_strategy_timer)

    start_match = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=["ros2", "service", "call", "/soccer/start", "std_srvs/srv/Trigger", "{}"],
                name="soccer_start",
                output="screen",
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("kickoff_team", default_value="team_a"),
        pygame_sim,
        strategies,
        start_match,
    ])
