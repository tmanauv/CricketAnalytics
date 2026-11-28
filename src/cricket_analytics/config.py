"""Pipeline configuration loading."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the default config, overlaid with ``path`` when given."""
    with open(DEFAULT_CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH:
        with open(path) as fh:
            config = _deep_merge(config, yaml.safe_load(fh) or {})
    return config


def resolve_path(path: str | Path, root: Path = REPO_ROOT) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else root / path
