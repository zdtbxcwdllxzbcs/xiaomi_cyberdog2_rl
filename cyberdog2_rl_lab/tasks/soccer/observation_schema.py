"""Shared soccer observation schema — 12-D Flamez-compatible observation.

12-D = [self_x, self_y_canon, sin(yaw_canon), cos(yaw_canon),
        ball_x, ball_y_canon, ball_vx, ball_vy_canon,
        goal_x, goal_y_canon, opp_x, opp_y_canon]

All positions normalized by half_width/half_length from field.yaml.
All velocities normalized by MAX_LINEAR_SPD (1.2 m/s), clipped to ±5.0.
"canonical" means multiplied by team_sign so the observation is invariant
to which side the agent plays on.

This schema is SHARED between training (soccer_env.py) and deployment
(soccer_policy_runner.py). Both MUST produce identical observations.
"""

from __future__ import annotations

AGENTS = ["blue_attacker", "blue_goalkeeper", "red_attacker", "red_goalkeeper"]

TEAM_SIGN = {
    "blue_attacker": 1.0,
    "blue_goalkeeper": 1.0,
    "red_attacker": -1.0,
    "red_goalkeeper": -1.0,
}

ROLE = {
    "blue_attacker": "attacker",
    "blue_goalkeeper": "goalkeeper",
    "red_attacker": "attacker",
    "red_goalkeeper": "goalkeeper",
}

TEAMMATE = {
    "blue_attacker": "blue_goalkeeper",
    "blue_goalkeeper": "blue_attacker",
    "red_attacker": "red_goalkeeper",
    "red_goalkeeper": "red_attacker",
}

OPPONENTS = {
    "blue_attacker": ["red_attacker", "red_goalkeeper"],
    "blue_goalkeeper": ["red_attacker", "red_goalkeeper"],
    "red_attacker": ["blue_attacker", "blue_goalkeeper"],
    "red_goalkeeper": ["blue_attacker", "blue_goalkeeper"],
}

# --- 12-D Flamez-compatible observation (PRIMARY) ---

FLAMEZ_OBSERVATION_FIELDS = [
    "self_x",
    "self_y_canon",
    "self_yaw_sin",
    "self_yaw_cos",
    "ball_x",
    "ball_y_canon",
    "ball_vx",
    "ball_vy_canon",
    "goal_x",
    "goal_y_canon",
    "opp_x",
    "opp_y_canon",
]

FLAMEZ_OBSERVATION_DIM = len(FLAMEZ_OBSERVATION_FIELDS)

POLICY_OBSERVATION_FIELDS = FLAMEZ_OBSERVATION_FIELDS
POLICY_OBSERVATION_DIM = FLAMEZ_OBSERVATION_DIM

# --- Normalization constants (shared between training and deployment) ---

BALL_VEL_OBS_LIMIT = 5.0
MAX_LINEAR_SPD = 1.2
MAX_ANGULAR_SPD = 3.0

GLOBAL_STATE_DIM = 35

ACTION_FIELDS = ["vx", "vy", "wz"]
