"""Ridge regression baseline wrapper."""
from __future__ import annotations

from typing import Any

from sklearn.linear_model import Ridge


def run_ridge(x, y, alpha: float = 1.0) -> Any:
    """Fit a ridge regression model."""
    model = Ridge(alpha=alpha)
    model.fit(x, y)
    return model
