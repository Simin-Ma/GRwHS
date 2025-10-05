"""Lasso regression baseline wrapper."""
from __future__ import annotations

from typing import Any

from sklearn.linear_model import Lasso


def run_lasso(x, y, alpha: float = 0.1) -> Any:
    """Fit a Lasso regression model."""
    model = Lasso(alpha=alpha)
    model.fit(x, y)
    return model
