"""Catalog utilities for upstream Cyberdog locomotion metadata."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .command_types import DEVELOPER_SERVO_MOTIONS


REPO_ROOT = Path(__file__).resolve().parents[4]
REFS_ROOT = REPO_ROOT / "refs" / "cyberdog_locomotion"
CONTROL_FLAGS_RELEASE = REFS_ROOT / "common" / "include" / "control_flags_release.hpp"
CYBERDOG2_PREINSTALLED = REFS_ROOT / "control" / "motion_list" / "cyberdog2" / "preinstalled"


@dataclass(frozen=True)
class EnumEntry:
    symbol: str
    value: int
    label: str


@dataclass
class MotionFileEntry:
    path: str
    step_count: int
    step_types: list[str] = field(default_factory=list)
    gait_ids: list[int] = field(default_factory=list)
    duration_ms: int = 0


def _extract_macro_entries(text: str, macro_name: str) -> dict[str, EnumEntry]:
    macro_start = text.find(f"#define {macro_name}")
    if macro_start < 0:
        return {}
    next_define = text.find("#define ", macro_start + len(f"#define {macro_name}"))
    block = text[macro_start : next_define if next_define > 0 else len(text)]
    pattern = re.compile(r'X\(\s*([A-Za-z0-9_]+)\s*=\s*(-?\d+)\s*,\s*"([^"]*)"\s*,\s*(-?\d+)\s*\)')
    entries = {}
    for match in pattern.finditer(block):
        symbol, enum_value, label, display_value = match.groups()
        value = int(enum_value)
        if value != int(display_value):
            label = f"{label} (display={display_value})"
        entries[symbol] = EnumEntry(symbol=symbol, value=value, label=label)
    return entries


def load_control_flag_enums(path: Path = CONTROL_FLAGS_RELEASE) -> dict[str, dict[str, EnumEntry]]:
    """Load ``MotionMode``, ``GaitId`` and ``MotionId`` entries from refs."""

    if not path.exists():
        return {"motion_modes": {}, "gait_ids": {}, "motion_ids": {}}
    text = path.read_text(encoding="utf-8")
    return {
        "motion_modes": _extract_macro_entries(text, "MOTION_MODE"),
        "gait_ids": _extract_macro_entries(text, "GAIT_ID"),
        "motion_ids": _extract_macro_entries(text, "MOTION_ID"),
    }


def load_preinstalled_motion_files(directory: Path = CYBERDOG2_PREINSTALLED) -> dict[str, MotionFileEntry]:
    """Read the Cyberdog2 preinstalled TOML motion lists into a compact catalog."""

    if not directory.exists():
        return {}
    catalog = {}
    for file_path in sorted(directory.glob("*.toml")):
        try:
            data = tomllib.loads(file_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            continue
        steps = data.get("step", [])
        if not isinstance(steps, list):
            steps = []
        step_types = []
        gait_ids = []
        duration_ms = 0
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_type = step.get("type")
            if isinstance(step_type, str):
                step_types.append(step_type)
            if "gait_id" in step:
                try:
                    gait_ids.append(int(step["gait_id"]))
                except (TypeError, ValueError):
                    pass
            if "duration" in step:
                try:
                    duration_ms += int(step["duration"])
                except (TypeError, ValueError):
                    pass
        catalog[file_path.stem] = MotionFileEntry(
            path=str(file_path),
            step_count=len(steps),
            step_types=sorted(set(step_types)),
            gait_ids=sorted(set(gait_ids)),
            duration_ms=duration_ms,
        )
    return catalog


def developer_servo_mapping_table() -> list[dict[str, Any]]:
    """Return the explicit developer-guide to refs mapping used by this project."""

    return [DEVELOPER_SERVO_MOTIONS[key].to_dict() for key in sorted(DEVELOPER_SERVO_MOTIONS)]


def build_motion_catalog() -> dict[str, Any]:
    """Build a serializable catalog for export/debugging."""

    enums = load_control_flag_enums()
    files = load_preinstalled_motion_files()
    return {
        "refs_root": str(REFS_ROOT),
        "developer_servo_motions": developer_servo_mapping_table(),
        "motion_modes": {key: entry.__dict__ for key, entry in enums["motion_modes"].items()},
        "gait_ids": {key: entry.__dict__ for key, entry in enums["gait_ids"].items()},
        "motion_ids": {key: entry.__dict__ for key, entry in enums["motion_ids"].items()},
        "preinstalled_motion_files": {key: entry.__dict__ for key, entry in files.items()},
    }
