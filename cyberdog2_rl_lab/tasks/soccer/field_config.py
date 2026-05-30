"""Shared soccer field configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class SoccerFieldConfig:
    """Canonical soccer field semantics used by training and runtime."""

    x_width: float
    y_length: float
    goal_width: float
    blue_attack_sign_y: float
    red_attack_sign_y: float
    blue_goal: tuple[float, float, float]
    red_goal: tuple[float, float, float]
    player_starts: dict[str, tuple[float, float, float]]
    vrpn_frame_map: dict[str, str]
    normalization: dict[str, Any]
    raw: dict[str, Any]

    @property
    def half_extents_xy(self) -> tuple[float, float]:
        return (0.5 * self.x_width, 0.5 * self.y_length)

    def goal_for_team_sign(self, sign: float) -> tuple[float, float, float]:
        return self.blue_goal if sign > 0.0 else self.red_goal

    def target_goal_for_team_sign(self, sign: float) -> tuple[float, float, float]:
        return self.red_goal if sign > 0.0 else self.blue_goal


def default_field_config_path() -> str:
    """Return the packaged default field YAML path."""

    return str(files("cyberdog2_rl_lab.tasks.soccer.config").joinpath("field.yaml"))


def load_field_config(path: str | None = None) -> SoccerFieldConfig:
    """Load the canonical soccer field configuration."""

    config_path = path or default_field_config_path()
    with open(config_path, encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return parse_field_config(data)


def parse_field_config(data: Mapping[str, Any]) -> SoccerFieldConfig:
    field = data["field"]
    goals = data["goals"]
    teams = data["teams"]
    kickoff = data.get("kickoff_spots", {})
    return SoccerFieldConfig(
        x_width=float(field["x_width"]),
        y_length=float(field["y_length"]),
        goal_width=float(goals["width"]),
        blue_attack_sign_y=float(teams["blue_attack_sign_y"]),
        red_attack_sign_y=float(teams["red_attack_sign_y"]),
        blue_goal=_tuple3(goals["blue"]["center"]),
        red_goal=_tuple3(goals["red"]["center"]),
        player_starts={key: _tuple3(value) for key, value in kickoff.get("player_starts", {}).items()},
        vrpn_frame_map=dict(data.get("vrpn_frame_map", {})),
        normalization=dict(data.get("normalization", {})),
        raw=dict(data),
    )


def _tuple3(value: Any) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"Expected a 3D tuple/list, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))
