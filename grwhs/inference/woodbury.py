"""Woodbury identity utilities."""
from __future__ import annotations

import numpy as np


def woodbury_inverse(a: np.ndarray, u: np.ndarray, c: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute inverse using the Woodbury identity."""
    inv_a = np.linalg.inv(a)
    middle = np.linalg.inv(c + v @ inv_a @ u)
    return inv_a - inv_a @ u @ middle @ v @ inv_a
