"""Shared deployment schema for official Cyberdog2 locomotion."""

OFFICIAL_LOCOMOTION_OBSERVATION_FIELDS = [
    "root_lin_vel_b.x",
    "root_lin_vel_b.y",
    "root_lin_vel_b.z",
    "root_ang_vel_b.x",
    "root_ang_vel_b.y",
    "root_ang_vel_b.z",
    "projected_gravity_b.x",
    "projected_gravity_b.y",
    "projected_gravity_b.z",
    "target_vx",
    "target_vy",
    "target_wz",
    "cmd_vx",
    "cmd_vy",
    "cmd_wz",
    "action_vx",
    "action_vy",
    "action_wz",
]

OFFICIAL_LOCOMOTION_ACTION_FIELDS = ["vx", "vy", "wz"]
