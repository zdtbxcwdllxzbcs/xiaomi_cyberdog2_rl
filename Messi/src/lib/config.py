"""YAML configuration loading helpers."""

from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

import yaml


ConfigDict = Dict[str, Any]
PathLike = Union[str, Path]


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Return ``base`` recursively updated with ``override`` values."""

    merged = dict(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def _resolve_path(path: PathLike, *, root: Optional[Path] = None) -> Path:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute() and root is not None:
        config_path = root / config_path
    return config_path


def load_config_file(path: PathLike, *, root: Optional[Path] = None) -> ConfigDict:
    """Load a YAML config, recursively merging an optional ``base_config``.

    ``base_config`` is resolved relative to the file that declares it. Values in
    the current file override the shared base, with dictionaries merged
    recursively and lists/scalars replaced wholesale.
    """

    return _load_config_file(_resolve_path(path, root=root), visited=set())


def _load_config_file(path: Path, *, visited: Set[Path]) -> ConfigDict:
    config_path = path.resolve()
    if config_path in visited:
        chain = " -> ".join(str(item) for item in (*visited, config_path))
        raise ValueError(f"Detected config base_config cycle: {chain}")
    visited.add(config_path)

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    base_config = config.get("base_config")
    if base_config is None:
        return dict(config)
    if not isinstance(base_config, str):
        raise ValueError(f"base_config must be a string path in {config_path}")

    base_path = Path(base_config).expanduser()
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base = _load_config_file(base_path, visited=visited)
    return _deep_merge(base, config)
