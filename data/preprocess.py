"""Data preprocessing utilities for GRwHS datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "StandardizationConfig",
    "StandardizeResult",
    "standardize_X",
    "center_y",
    "apply_standardization",
]

_EPS = 1e-8


@dataclass(frozen=True)
class StandardizationConfig:
    """Configuration for feature/target standardization."""

    X: str = "unit_variance"
    y_center: bool = True


@dataclass
class StandardizeResult:
    """Result of applying standardization to a dataset."""

    X: np.ndarray
    y: Optional[np.ndarray]
    x_mean: Optional[np.ndarray] = None
    x_scale: Optional[np.ndarray] = None
    y_mean: Optional[float] = None
    config: StandardizationConfig = field(default_factory=StandardizationConfig)


def standardize_X(
    X: np.ndarray,
    method: str = "unit_variance",
    eps: float = _EPS,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Standardize feature matrix according to the requested method."""

    arr = np.asarray(X, dtype=float)
    if method is None or str(method).lower() == "none":
        return arr.copy(), None, None

    method_l = str(method).lower()
    mean = arr.mean(axis=0, keepdims=True) if arr.size else np.zeros((1, arr.shape[1]))
    centered = arr - mean

    if method_l == "unit_variance":
        scale = np.std(centered, axis=0, keepdims=True)
        scale = np.maximum(scale, eps)
    elif method_l == "unit_l2":
        scale = np.linalg.norm(centered, axis=0, keepdims=True)
        scale = np.maximum(scale, eps)
    else:
        raise ValueError(f"Unknown standardization method '{method}'.")

    standardized = centered / scale
    return standardized, mean.squeeze(0), scale.squeeze(0)


def center_y(y: np.ndarray) -> tuple[np.ndarray, float]:
    """Center response vector to zero mean."""

    arr = np.asarray(y, dtype=float).reshape(-1)
    mean = float(arr.mean()) if arr.size else 0.0
    return arr - mean, mean


def apply_standardization(
    X: np.ndarray,
    y: Optional[np.ndarray],
    config: StandardizationConfig | None = None,
) -> StandardizeResult:
    """Apply feature/target standardization returning transformed arrays."""

    cfg = config or StandardizationConfig()
    X_std, x_mean, x_scale = standardize_X(X, cfg.X)

    y_std: Optional[np.ndarray]
    y_mean: Optional[float]
    if y is None:
        y_std = None
        y_mean = None
    elif cfg.y_center:
        y_std, y_mean = center_y(y)
    else:
        y_std = np.asarray(y, dtype=float).reshape(-1)
        y_mean = None

    result = StandardizeResult(
        X=X_std.astype(np.float32, copy=False),
        y=None if y_std is None else y_std.astype(np.float32, copy=False),
        x_mean=None if x_mean is None else np.asarray(x_mean, dtype=np.float32),
        x_scale=None if x_scale is None else np.asarray(x_scale, dtype=np.float32),
        y_mean=y_mean,
        config=cfg,
    )
    return result
