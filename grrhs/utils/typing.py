# grrhs/utils/typing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Protocol, Tuple, Union

try:
    import numpy as np
    NDArray = np.ndarray
except Exception:  # pragma: no cover
    NDArray = Any  # type: ignore

try:
    import jax.numpy as jnp
    JaxArray = jnp.ndarray  # pyright: ignore[reportAttributeAccessIssue]
except Exception:  # pragma: no cover
    JaxArray = Any  # type: ignore

ArrayLike = Union["NDArray", "JaxArray"]

JsonDict = Dict[str, Any]
ConfigDict = Dict[str, Any]


class Regressor(Protocol):
    """Minimal learner interface compatible with experiments/runner.py."""

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "Regressor":
        ...

    def predict(self, X: ArrayLike, **kwargs) -> ArrayLike:
        ...

    def get_posterior_summaries(self) -> Mapping[str, Any]:
        ...


@dataclass
class SplitData:
    X_train: ArrayLike
    y_train: ArrayLike
    X_val: Optional[ArrayLike] = None
    y_val: Optional[ArrayLike] = None
    X_test: Optional[ArrayLike] = None
    y_test: Optional[ArrayLike] = None
    meta: Optional[Dict[str, Any]] = None
