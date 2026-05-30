"""Smoke checks for official locomotion deployment bridge semantics."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "source" / "cyberdog2_rl_lab"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from cyberdog2_rl_lab.official import CommandMapperConfig, VelocityCommandMapper
from cyberdog2_rl_lab.official.deploy_schema import (
    OFFICIAL_LOCOMOTION_ACTION_FIELDS,
    OFFICIAL_LOCOMOTION_OBSERVATION_FIELDS,
)
from cyberdog2_rl_lab.runtime.official_cmd_publisher import OfficialCommandPublisher, OfficialPublisherConfig
from cyberdog2_rl_lab.runtime.official_locomotion_policy_runner import (
    OfficialLocomotionObservationBuilder,
    OfficialLocomotionPolicyRunner,
    OfficialLocomotionRuntimeState,
)
from cyberdog2_rl_lab.utils.deploy_validation import (
    infer_action_dim,
    infer_normalized_action_scale,
    infer_policy_observation_dim,
    validate_official_locomotion_deploy_cfg,
    with_deployment_identity,
)


def build_official_deploy_cfg() -> dict:
    mapper = VelocityCommandMapper(CommandMapperConfig())
    cfg = {
        "step_dt": 0.02,
        "workflow": "OfficialLocomotionEnv",
        "task": "official_locomotion",
        "command_mapper": mapper.cfg.to_dict(),
        "actions": {"normalized_velocity": OFFICIAL_LOCOMOTION_ACTION_FIELDS},
        "observations": {"policy": OFFICIAL_LOCOMOTION_OBSERVATION_FIELDS},
    }
    return with_deployment_identity(cfg, task_id="Cyberdog2-Official-Locomotion-Play-v0")


def check_deploy_schema() -> None:
    cfg = build_official_deploy_cfg()
    validate_official_locomotion_deploy_cfg(cfg)
    assert infer_policy_observation_dim(cfg) == 18
    assert infer_action_dim(cfg) == 3
    assert infer_normalized_action_scale(cfg) == [1.6, 0.55, 2.5]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "deploy.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        validate_official_locomotion_deploy_cfg(loaded)


def check_mapper_and_publisher() -> None:
    mapper = VelocityCommandMapper(CommandMapperConfig())
    command = mapper.map_normalized_action((1.0, 1.0, 1.0), 0.02)
    np.testing.assert_allclose(command.vel_des, (0.05, 0.04, 0.12))
    assert mapper.last_debug["raw_velocity"] == [1.6, 0.55, 2.5]
    assert mapper.last_debug["clipped_velocity"] == [1.6, 0.55, 2.5]
    assert mapper.last_debug["max_delta"] == [0.05, 0.04, 0.12]

    mapper.reset()
    assert mapper.select_profile((0.65, 0.30, 1.25)).developer_motion_id == 303
    assert mapper.select_profile((0.66, 0.30, 1.25)).developer_motion_id == 308
    assert mapper.select_profile((1.01, 0.30, 1.25)).developer_motion_id == 305

    publisher = OfficialCommandPublisher(OfficialPublisherConfig())
    packet = publisher.publish_velocity((1.6, 0.55, 2.5))
    assert packet["channel_name"] == "robot_control_cmd"
    np.testing.assert_allclose(packet["vel_des"], [0.05, 0.04, 0.12])
    np.testing.assert_allclose(packet["metadata"]["smoothed_velocity"], [0.05, 0.04, 0.12])
    assert packet["metadata"]["step_dt"] == 0.02


def check_observation_builder() -> None:
    cfg = build_official_deploy_cfg()
    builder = OfficialLocomotionObservationBuilder(cfg)
    state = OfficialLocomotionRuntimeState(
        root_lin_vel_b=(0.1, 0.2, 0.3),
        root_ang_vel_b=(0.4, 0.5, 0.6),
        projected_gravity_b=(0.0, 0.0, -1.0),
        target_velocity_b=(0.7, 0.8, 0.9),
    )
    obs = builder.build_observation(state)
    assert obs.shape == (18,)
    np.testing.assert_allclose(obs[:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(obs[9:12], [0.7, 0.8, 0.9])
    builder.update_last(command=[0.05, 0.0, 0.0], action=[1.0, 0.0, 0.0])
    obs = builder.build_observation(state)
    np.testing.assert_allclose(obs[12:15], [0.05, 0.0, 0.0])
    np.testing.assert_allclose(obs[15:18], [1.0, 0.0, 0.0])


def check_policy_runner_infer_path() -> None:
    cfg = build_official_deploy_cfg()
    runner = OfficialLocomotionPolicyRunner.__new__(OfficialLocomotionPolicyRunner)
    runner.observation_builder = OfficialLocomotionObservationBuilder(cfg)
    runner.publisher = OfficialCommandPublisher(OfficialPublisherConfig())
    runner.cfg = None
    runner._run_policy = lambda obs: np.array([2.0, 0.0, 0.0], dtype=np.float32)
    packet = runner.infer(OfficialLocomotionRuntimeState())
    np.testing.assert_allclose(packet["vel_des"], [0.05, 0.0, 0.0])
    np.testing.assert_allclose(packet["metadata"]["normalized_policy_action"], [1.0, 0.0, 0.0])
    assert len(packet["metadata"]["observation_fields"]) == 18
    parser = argparse.ArgumentParser(description="Smoke check official locomotion bridge semantics.")
    parser.parse_args()
    check_deploy_schema()
    check_mapper_and_publisher()
    check_observation_builder()
    check_policy_runner_infer_path()
    print("official locomotion bridge smoke checks passed", flush=True)


if __name__ == "__main__":
    main()
