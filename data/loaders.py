
"""Real dataset loading helpers for GRRHS experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import json
import numpy as np

try:  # Optional pandas for CSV/TSV loading
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except Exception:  # pragma: no cover - pandas is an optional dependency at runtime
    pd = None  # type: ignore
    _HAS_PANDAS = False

try:  # YAML support (PyYAML is already a dependency)
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
    _HAS_YAML = False

GroupMap = Dict[str, int]


@dataclass
class LoadedDataset:
    """Container for externally provided datasets."""

    X: np.ndarray
    y: Optional[np.ndarray] = None
    groups: Optional[List[List[int]]] = None
    feature_names: Optional[List[str]] = None
    beta: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _ensure_path(path_like: str | Path, base_dir: Optional[Path]) -> Path:
    path = Path(path_like)
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    return path


def _load_array(path: Path, *, key: Optional[str] = None, dtype: Optional[str] = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == '.npy':
        arr = np.load(path)
    elif suffix == '.npz':
        data = np.load(path)
        if key is None:
            if len(data.files) != 1:
                raise ValueError(
                    f"Ambiguous npz file {path}; specify key via loader config (e.g. X_key)."
                )
            key = data.files[0]
        if key not in data:
            raise KeyError(f"Key '{key}' not found in npz file {path}.")
        arr = data[key]
    elif suffix in {'.csv', '.tsv', '.txt'}:
        delimiter = '	' if suffix == '.tsv' else ','
        if _HAS_PANDAS:
            arr = pd.read_csv(path, sep=delimiter).to_numpy()
        else:  # pragma: no cover - exercised only when pandas missing
            arr = np.loadtxt(path, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported file format for array loading: {path.suffix}")

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.asarray(arr)


def _load_feature_names(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature-name file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {'.json'}:
        names = json.loads(path.read_text(encoding='utf-8'))
    elif suffix in {'.yaml', '.yml'} and _HAS_YAML:
        names = yaml.safe_load(path.read_text(encoding='utf-8'))
    else:
        names = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not isinstance(names, Iterable):  # pragma: no cover - defensive programming
        raise TypeError("Feature-name file must contain an iterable of strings.")
    return [str(name) for name in names]


def _load_group_map(path: Path) -> GroupMap:
    if not path.exists():
        raise FileNotFoundError(f"Group-map file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
    elif suffix in {'.yaml', '.yml'}:
        if not _HAS_YAML:
            raise RuntimeError("PyYAML not available but required to parse YAML group map.")
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    elif suffix in {'.csv', '.tsv'}:
        delimiter = '	' if suffix == '.tsv' else ','
        if _HAS_PANDAS:
            frame = pd.read_csv(path, sep=delimiter)
            if frame.shape[1] < 2:
                raise ValueError("Group map CSV must contain at least two columns: feature and group.")
            payload = dict(zip(frame.iloc[:, 0].astype(str), frame.iloc[:, 1].astype(int)))
        else:  # pragma: no cover
            import csv

            payload = {}
            with path.open('r', encoding='utf-8') as fh:
                reader = csv.reader(fh, delimiter=delimiter)
                for row in reader:
                    if len(row) < 2:
                        continue
                    payload[str(row[0])] = int(row[1])
    else:
        raise ValueError(f"Unsupported group-map file format: {path.suffix}")

    if not isinstance(payload, Mapping):
        raise TypeError("Group map file must decode into a mapping of feature -> group id.")
    result: GroupMap = {}
    for key, val in payload.items():
        try:
            result[str(key)] = int(val)
        except Exception as exc:  # pragma: no cover - invalid entries
            raise ValueError(f"Invalid group identifier for feature '{key}': {val}") from exc
    return result


def load_real_dataset(
    loader_cfg: Mapping[str, Any],
    *,
    base_dir: Optional[Path] = None,
) -> LoadedDataset:
    """Load an external dataset according to loader configuration.

    Supported fields in ``loader_cfg``:
        path_X (str): location of features (.npy, .npz, .csv, .tsv).
        X_key (str, optional): key inside .npz when path_X points to npz.
        path_y (str, optional): target array location. Required for supervised tasks.
        y_key (str, optional): key inside .npz when path_y points to npz.
        path_feature_names (str, optional): text/JSON/YAML file listing feature names in order.
        path_group_map (str, optional): JSON/YAML/CSV mapping feature name to group id.
        beta_path (str, optional): ground-truth coefficients (for simulations based on real design matrices).
        beta_key (str, optional): companion key when ``beta_path`` is .npz.
    """

    if not loader_cfg:
        raise ValueError("loader configuration must be provided for data.type=loader")

    root = base_dir.resolve() if base_dir is not None else None

    path_X = loader_cfg.get('path_X')
    if not path_X:
        raise ValueError("loader.path_X is required to locate feature matrix")
    X = _load_array(_ensure_path(path_X, root), key=loader_cfg.get('X_key'), dtype='float32')

    path_y = loader_cfg.get('path_y')
    if path_y:
        y = _load_array(_ensure_path(path_y, root), key=loader_cfg.get('y_key'), dtype='float32').reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"Target length {y.shape[0]} does not match number of samples {X.shape[0]}.")
    else:
        y = None

    feature_names: Optional[List[str]] = None
    if loader_cfg.get('path_feature_names'):
        feature_names = _load_feature_names(_ensure_path(loader_cfg['path_feature_names'], root))
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                "Number of feature names does not match columns in X."            )

    group_map: Optional[GroupMap] = None
    if loader_cfg.get('path_group_map'):
        group_map = _load_group_map(_ensure_path(loader_cfg['path_group_map'], root))

    groups: Optional[List[List[int]]] = None
    if loader_cfg.get('groups') is not None:
        groups = [[int(idx) for idx in group] for group in loader_cfg['groups']]
    elif group_map is not None:
        if feature_names is None:
            raise ValueError('Feature names are required when using path_group_map.')
        grouped: Dict[int, List[int]] = {}
        for idx, name in enumerate(feature_names):
            if name not in group_map:
                continue
            gid = int(group_map[name])
            grouped.setdefault(gid, []).append(idx)
        groups = [grouped[key] for key in sorted(grouped.keys())]

    beta = None
    if loader_cfg.get('beta_path'):
        beta = _load_array(_ensure_path(loader_cfg['beta_path'], root), key=loader_cfg.get('beta_key'), dtype='float32').reshape(-1)
        if beta.shape[0] != X.shape[1]:
            raise ValueError("Ground-truth beta length does not match number of features.")

    metadata: Dict[str, Any] = {
        key: value
        for key, value in loader_cfg.items()
        if key not in {"path_X", "X_key", "path_y", "y_key", "path_feature_names", "path_group_map", "beta_path", "beta_key"}
    }
    metadata.setdefault('source_path', str(_ensure_path(path_X, root)))
    if path_y:
        metadata.setdefault('target_path', str(_ensure_path(path_y, root)))

    return LoadedDataset(
        X=X,
        y=y,
        groups=groups,
        feature_names=feature_names,
        beta=beta,
        metadata=metadata,
    )
