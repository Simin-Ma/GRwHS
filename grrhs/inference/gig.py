"""Generalized inverse Gaussian sampling utilities."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.random import Generator, default_rng
from scipy.stats import geninvgauss


def sample_gig(
    lambda_param: float,
    chi: float,
    psi: float,
    *,
    size: int | tuple[int, ...] = 1,
    rng: Optional[Generator] = None,
) -> np.ndarray:
    """Draw samples from a Generalized Inverse Gaussian distribution.

    The density is proportional to
        x^{lambda_param - 1} * exp(-0.5 * (psi * x + chi / x)),  x > 0.

    This wraps :func:`scipy.stats.geninvgauss` while exposing a numpy-style RNG
    interface used throughout the code base.
    """
    if chi <= 0 or psi <= 0:
        raise ValueError("chi and psi must be positive for GIG sampling")

    if rng is None:
        rng = default_rng()

    random_state = rng
    b = math.sqrt(float(chi) * float(psi))
    scale = math.sqrt(float(chi) / float(psi))

    samples = geninvgauss.rvs(
        p=lambda_param,
        b=b,
        size=size,
        scale=scale,
        random_state=random_state,
    )
    return np.asarray(samples, dtype=float)
