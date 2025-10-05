"""YAML configuration loader with command-line overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def merge_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Merge flat key overrides into a nested config dictionary."""
    merged = dict(config)
    for key, value in overrides.items():
        merged[key] = value
    return merged
