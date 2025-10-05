"""Regression metrics leveraging sklearn implementations."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _to_vector(arr: np.ndarray) -> np.ndarray:
    """Ensure flat numpy vector input for sklearn metrics."""
    return np.asarray(arr).reshape(-1)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(mean_squared_error(_to_vector(y_true), _to_vector(y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(_to_vector(y_true), _to_vector(y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    return float(r2_score(_to_vector(y_true), _to_vector(y_pred)))
