
"""Sweep definitions and helpers for multi-run experiments."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence, Tuple


def deep_update(target: Dict[str, Any], source: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``source`` into ``target`` (mutates and returns target)."""

    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = deepcopy(value) if isinstance(value, (dict, list)) else value
    return target


def _normalize_overrides(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert dotted keys into nested dictionaries while preserving structure."""

    normalized: Dict[str, Any] = {}
    for raw_key, value in overrides.items():
        if isinstance(value, Mapping):
            value = _normalize_overrides(value)
        key = str(raw_key)
        if "." in key:
            cursor = normalized
            parts = key.split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
                if not isinstance(cursor, dict):
                    raise ValueError(f"Override path conflict at '{key}'.")
            last = parts[-1]
            if isinstance(value, Mapping) and isinstance(cursor.get(last), dict):
                deep_update(cursor[last], value)
            else:
                cursor[last] = value
        else:
            if isinstance(value, Mapping) and isinstance(normalized.get(key), dict):
                deep_update(normalized[key], value)
            else:
                normalized[key] = value
    return normalized


def build_override_tree(*overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Combine multiple override mappings (with dotted keys) into one nested dict."""

    tree: Dict[str, Any] = {}
    for override in overrides:
        if not override:
            continue
        normalized = _normalize_overrides(override)
        deep_update(tree, normalized)
    return tree


def iter_sweep_configs(
    base_config: Mapping[str, Any],
    variations: Iterable[Mapping[str, Any]],
    *,
    common_overrides: Mapping[str, Any] | None = None,
) -> Iterator[Tuple[str, Dict[str, Any], Mapping[str, Any]]]:
    """Yield ``(name, config, metadata)`` tuples for each sweep variation."""

    base = deepcopy(dict(base_config))
    common_tree = build_override_tree(common_overrides or {})

    for idx, variation in enumerate(variations):
        name = variation.get("name") or f"variation_{idx:03d}"
        config = deepcopy(base)
        deep_update(config, common_tree)
        var_tree = build_override_tree(variation.get("overrides", {}))
        deep_update(config, var_tree)
        metadata = variation.get("metadata", {})
        yield name, config, metadata

