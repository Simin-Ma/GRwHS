"""Elastic net regression baseline."""
from __future__ import annotations

from typing import Any

from sklearn.linear_model import ElasticNet


def run_elastic_net(x, y, alpha: float = 0.1, l1_ratio: float = 0.5) -> Any:
    """Fit an elastic net model."""
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(x, y)
    return model
