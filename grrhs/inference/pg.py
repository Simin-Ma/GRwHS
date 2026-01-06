"""Pólya–Gamma sampling utilities."""
from __future__ import annotations

from typing import Optional, Union, Sequence

import numpy as np
from numpy.random import Generator

try:
    from polyagamma import random_polyagamma
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "The 'polyagamma' package is required for logistic GRRHS sampling. "
        "Install it via `pip install polyagamma`."
    ) from exc

ArrayLike = Union[float, np.ndarray, Sequence[float]]


def sample_pg(
    b: ArrayLike,
    c: ArrayLike,
    *,
    size: Optional[Union[int, tuple[int, ...]]] = None,
    rng: Optional[Generator] = None,
) -> np.ndarray:
    """Draw samples from the Pólya–Gamma(b, c) distribution.

    Parameters
    ----------
    b : array-like
        Shape/tilt parameter ``b`` (often integer-valued). Broadcasts with ``c``.
    c : array-like
        Tilt parameter ``c``. Broadcasts with ``b``.
    size : int or tuple, optional
        Explicit sample shape. When omitted, inferred from broadcasting ``b`` and ``c``.
    rng : numpy.random.Generator, optional
        Generator used to seed the underlying sampler for reproducibility.

    Returns
    -------
    np.ndarray
        Samples matching the broadcasted shape of ``b`` and ``c`` (or ``size`` when provided).
    """
    random_state = None
    if rng is not None:
        # polyagamma.random_polyagamma accepts either a RandomState or an int seed
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        random_state = np.random.RandomState(seed)
    samples = random_polyagamma(b, c, size=size, random_state=random_state)
    return np.asarray(samples, dtype=float)


__all__ = ["sample_pg"]
