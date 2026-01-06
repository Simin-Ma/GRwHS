
"""Sampling routines for GRRHS."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from numpy.random import Generator, default_rng


class Sampler(Protocol):
    """Protocol for samplers that advance a state."""

    def step(self, state, rng: Generator | None = None):  # pragma: no cover - protocol
        ...


@dataclass
class SliceSampler1D:
    """Simple 1-D slice sampler with optional stepping-out."""

    log_density: Callable[[float], float]
    width: float = 1.0
    max_steps: int = 1000
    step_out: bool = True
    rng: Generator = field(default_factory=default_rng)

    def step(self, state: float, rng: Generator | None = None) -> float:
        return slice_sample_1d(
            self.log_density,
            state,
            rng or self.rng,
            width=self.width,
            max_steps=self.max_steps,
            step_out=self.step_out,
        )


def slice_sample_1d(
    log_density: Callable[[float], float],
    x0: float,
    rng: Generator,
    *,
    width: float = 1.0,
    max_steps: int = 1000,
    step_out: bool = True,
) -> float:
    """Neal-style slice sampler for a scalar log-density."""

    if width <= 0:
        raise ValueError("Slice sampling width must be positive.")

    log_y = log_density(x0) - rng.exponential(1.0)

    u = rng.uniform(0.0, 1.0)
    L = x0 - u * width
    R = L + width

    if step_out:
        j = rng.integers(0, 10)
        k = 10 - j
        while j > 0 and log_density(L) > log_y:
            L -= width
            j -= 1
        while k > 0 and log_density(R) > log_y:
            R += width
            k -= 1

    for _ in range(max_steps):
        x1 = rng.uniform(L, R)
        if log_density(x1) >= log_y:
            return float(x1)
        if x1 < x0:
            L = x1
        else:
            R = x1

    return float(x0)

