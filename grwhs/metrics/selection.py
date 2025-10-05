"""Feature selection metrics backed by sklearn."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import recall_score


def true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute TPR (recall) for binary indicators using sklearn."""
    y_t = np.asarray(y_true).reshape(-1)
    y_p = np.asarray(y_pred).reshape(-1)
    return float(recall_score(y_t, y_p, zero_division=0))
