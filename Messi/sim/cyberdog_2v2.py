"""Shared CyberDog 2v2 simulation constants."""

import math
from dataclasses import dataclass


ROBOT_Z = 0.31
BALL_Z = 0.125
SHM_PREFIX = "development-simulator"


@dataclass(frozen=True)
class RobotSpec:
    team: str
    index: int
    dog_namespace: str
    rigid_name: str
    model_name: str
    config_file: str
    x: float
    y: float
    yaw: float


ROBOT_SPECS = (
    RobotSpec(
        team="team_a",
        index=1,
        dog_namespace="team_a/robot_1",
        rigid_name="team_a_1",
        model_name="team_a_robot_1",
        config_file="sim_2v2_team_a_1.yaml",
        x=0.0,
        y=-0.5,
        yaw=math.pi / 2.0,
    ),
    RobotSpec(
        team="team_a",
        index=2,
        dog_namespace="team_a/robot_2",
        rigid_name="team_a_2",
        model_name="team_a_robot_2",
        config_file="sim_2v2_team_a_2.yaml",
        x=0.0,
        y=-4.72,
        yaw=0.0,
    ),
    RobotSpec(
        team="team_b",
        index=1,
        dog_namespace="team_b/robot_1",
        rigid_name="team_b_1",
        model_name="team_b_robot_1",
        config_file="sim_2v2_team_b_1.yaml",
        x=0.0,
        y=2.5,
        yaw=-math.pi / 2.0,
    ),
    RobotSpec(
        team="team_b",
        index=2,
        dog_namespace="team_b/robot_2",
        rigid_name="team_b_2",
        model_name="team_b_robot_2",
        config_file="sim_2v2_team_b_2.yaml",
        x=0.0,
        y=4.72,
        yaw=0.0,
    ),
)


MODEL_NAMES = tuple(spec.model_name for spec in ROBOT_SPECS)
DOG_NAMESPACE_TO_MODEL = {spec.dog_namespace: spec.model_name for spec in ROBOT_SPECS}


def shared_memory_name(model_name: str) -> str:
    return f"{SHM_PREFIX}__{model_name}"


def lcm_channel(base: str, model_name: str) -> str:
    return f"{base}__{model_name}"


def robot_control_channel(model_name: str) -> str:
    return lcm_channel("robot_control_cmd", model_name)


def exec_request_channel(model_name: str) -> str:
    return lcm_channel("exec_request", model_name)


def gamepad_channel(model_name: str) -> str:
    return lcm_channel("gamepad_lcmt", model_name)


def simulator_state_channel(model_name: str) -> str:
    return lcm_channel("simulator_state", model_name)
