"""Python-side mirrors of Cyberdog locomotion command concepts.

The developer guide exposes app/service-level ``motion_id`` values such as
301/302/303/305/308. The upstream locomotion controller in ``refs`` uses a
lower-level ``mode/gait_id/motion_id`` split. This module keeps both names
visible so training, deployment, and later ROS/LCM bridges do not silently mix
the two numbering schemes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any


class MotionMode(IntEnum):
    """Subset of ``MotionMode`` from ``control_flags_release.hpp``."""

    INVALID = -1
    OFF = 0
    QP_STAND = 3
    PURE_DAMPER = 7
    LIFTED = 9
    LOCOMOTION = 11
    RECOVERY_STAND = 12
    MOTOR_CTRL = 15
    JUMP_3D = 16
    POSE_CTRL = 21
    FORCE_JUMP = 22
    MOTION = 62
    TWO_LEG_STAND = 64
    RL_RESET = 80
    RL_RAPID = 81


class GaitId(IntEnum):
    """Subset of ``GaitId`` from ``control_flags_release.hpp``."""

    STAND = 1
    PRONK = 2
    TROT_MEDIUM = 3
    PASSIVE_TROT = 4
    TROT_10V4 = 5
    WALK = 6
    BOUND = 7
    PACE = 8
    TROT_10V5 = 9
    TROT_FAST = 10
    STAND_NO_PR = 31
    SPECIAL_PRONK = 50
    SPECIAL_TROT = 51
    TROT_SWING = 55
    TROT_IN_OUT = 56
    TROT_PITCH = 57
    WALK_WAVE = 60
    USER_GAIT = 110


class CmdSource(IntEnum):
    """Command source values used by the upstream command interface."""

    GAMEPAD = 7
    RC = 8
    CYBERDOG_LCM = 9
    CYBERDOG2_LCM = 10
    MOTOR = 11


@dataclass(frozen=True)
class VelocityBounds:
    """Symmetric or asymmetric body-frame velocity limits."""

    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]

    def as_positive_scale(self) -> tuple[float, float, float]:
        """Return per-axis positive scales suitable for normalized actions."""

        return (
            max(abs(self.lin_vel_x[0]), abs(self.lin_vel_x[1])),
            max(abs(self.lin_vel_y[0]), abs(self.lin_vel_y[1])),
            max(abs(self.ang_vel_z[0]), abs(self.ang_vel_z[1])),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "lin_vel_x": list(self.lin_vel_x),
            "lin_vel_y": list(self.lin_vel_y),
            "ang_vel_z": list(self.ang_vel_z),
        }


@dataclass(frozen=True)
class DeveloperServoMotion:
    """Developer-guide servo motion entry plus inferred upstream fields."""

    developer_motion_id: int
    name: str
    mode: int
    gait_id: int
    bounds: VelocityBounds
    refs_motion_id: int | None = None
    confidence: str = "inferred"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bounds"] = self.bounds.to_dict()
        return data


@dataclass
class MotionControlCommand:
    """Python mirror of upstream ``MotionControlCommand``."""

    mode: int = int(MotionMode.OFF)
    gait_id: int = 0
    contact: int = 0x0F
    cmd_source: int = int(CmdSource.CYBERDOG2_LCM)
    vel_des: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_des: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pos_des: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ctrl_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acc_des: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    foot_pose: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    step_height: tuple[float, float] = (0.05, 0.05)
    value: int = 0
    duration: int = 0
    motion_id: int | None = None
    motion_trigger: int = 0
    motion_process_bar: int = 0
    user_gait_file: str = ""
    cmd_time_delay: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_robot_control_lcm_dict(self, life_count: int = 0) -> dict[str, Any]:
        """Return fields matching ``robot_control_cmd_lcmt``."""

        return {
            "mode": self.mode,
            "gait_id": self.gait_id,
            "contact": self.contact,
            "life_count": life_count,
            "vel_des": list(self.vel_des),
            "rpy_des": list(self.rpy_des),
            "pos_des": list(self.pos_des),
            "acc_des": list(self.acc_des),
            "ctrl_point": list(self.ctrl_point),
            "foot_pose": list(self.foot_pose),
            "step_height": list(self.step_height),
            "value": self.value,
            "duration": self.duration,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEVELOPER_SERVO_MOTIONS: dict[int, DeveloperServoMotion] = {
    303: DeveloperServoMotion(
        developer_motion_id=303,
        name="slow_walk",
        mode=int(MotionMode.LOCOMOTION),
        gait_id=int(GaitId.WALK),
        bounds=VelocityBounds((-0.65, 0.65), (-0.30, 0.30), (-1.25, 1.25)),
        note="Developer guide slow walking servo command. Internal gait_id is an explicit project mapping.",
    ),
    308: DeveloperServoMotion(
        developer_motion_id=308,
        name="medium_walk",
        mode=int(MotionMode.LOCOMOTION),
        gait_id=int(GaitId.TROT_MEDIUM),
        bounds=VelocityBounds((-1.00, 1.00), (-0.30, 0.30), (-1.25, 1.25)),
        note="Developer guide medium walking servo command. Internal gait_id is an explicit project mapping.",
    ),
    305: DeveloperServoMotion(
        developer_motion_id=305,
        name="fast_walk",
        mode=int(MotionMode.LOCOMOTION),
        gait_id=int(GaitId.TROT_FAST),
        bounds=VelocityBounds((-1.60, 1.60), (-0.55, 0.55), (-2.50, 2.50)),
        note="Developer guide fast walking servo command. Used as the default soccer velocity envelope.",
    ),
    302: DeveloperServoMotion(
        developer_motion_id=302,
        name="four_leg_pronk",
        mode=int(MotionMode.LOCOMOTION),
        gait_id=int(GaitId.PRONK),
        bounds=VelocityBounds((-0.25, 0.25), (-0.10, 0.10), (-0.50, 0.50)),
        note="Developer guide four-feet-off-ground gait. Exposed for cataloging, not used in the first RL action space.",
    ),
    301: DeveloperServoMotion(
        developer_motion_id=301,
        name="fore_hind_bound",
        mode=int(MotionMode.LOCOMOTION),
        gait_id=int(GaitId.BOUND),
        bounds=VelocityBounds((-0.40, 0.40), (-0.30, 0.30), (-0.75, 0.75)),
        note="Developer guide fore/hind alternating gait. Exposed for cataloging, not used in the first RL action space.",
    ),
}

DEFAULT_SOCCER_SERVO_ID = 305
DEFAULT_LOCOMOTION_BOUNDS = DEVELOPER_SERVO_MOTIONS[DEFAULT_SOCCER_SERVO_ID].bounds
