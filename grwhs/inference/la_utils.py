"""Linear algebra helpers."""
from __future__ import annotations

import numpy as np


def stable_cholesky(matrix: np.ndarray) -> np.ndarray:
    """Perform a stable Cholesky decomposition with jitter."""
    jitter = 1e-6
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(matrix + np.eye(matrix.shape[0]) * jitter)
