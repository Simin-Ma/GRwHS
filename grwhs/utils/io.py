"""I/O utilities for experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    """Write JSON with UTF-8 encoding."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> Any:
    """Load YAML file into a Python object."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))
