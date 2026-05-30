"""Launch file for pygame-only 2v2 CyberDog soccer simulation.

Starts pygame_sim.py (physics + VRPN + game state + rendering) and the four
strategy processes. No Gazebo, no LCM, no spawn.

Usage:
  ros2 launch launch/sim_2v2_pygame.launch.py
  ros2 launch launch/sim_2v2_pygame.launch.py kickoff_team:=team_b
"""

import os
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.cyberdog_2v2 import ROBOT_SPECS  # noqa: E402


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


def _strategy_process(spec):
    config = REPO_ROOT / "config" / spec.config_file
    return ExecuteProcess(
        cmd=[sys.executable, "-u", str(REPO_ROOT / "main.py"), "--config", str(config)],
        name=f"strategy_{spec.model_name}",
        output="screen",
        additional_env=_python_env(),
    )


def generate_launch_description():
    kickoff_team = LaunchConfiguration("kickoff_team")

    pygame_sim = _python_process(
        REPO_ROOT / "sim" / "pygame_sim.py",
        "--ros-args",
        "-p",
        ["kickoff_team:=", kickoff_team],
        name="pygame_sim",
    )

    strategies = TimerAction(
        period=3.0,
        actions=[_strategy_process(spec) for spec in ROBOT_SPECS],
    )

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
