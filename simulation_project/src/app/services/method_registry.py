from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ...utils import FitResult, SamplerConfig


@dataclass(frozen=True)
class MethodContext:
    X: np.ndarray
    y: np.ndarray
    groups: list[list[int]]
    task: str
    seed: int
    p0: int
    grrhs_p0: int
    n: int
    sampler: SamplerConfig
    grrhs_kwargs: dict
    gigg_mmle_kwargs: dict
    gigg_fixed_kwargs: dict


MethodRunner = Callable[[MethodContext], FitResult]


class MethodRegistry:
    def __init__(self) -> None:
        self._runners: dict[str, MethodRunner] = {}

    def register(self, name: str, runner: MethodRunner) -> None:
        key = str(name)
        if key in self._runners:
            raise ValueError(f"method already registered: {key}")
        self._runners[key] = runner

    def run(self, name: str, ctx: MethodContext) -> FitResult:
        key = str(name)
        if key not in self._runners:
            known = sorted(self._runners.keys())
            raise ValueError(f"Unsupported method: {key}; known={known}")
        return self._runners[key](ctx)

    def names(self) -> list[str]:
        return sorted(self._runners.keys())


def build_default_method_registry() -> MethodRegistry:
    from ...fit_classical import fit_lasso_cv, fit_ols
    from ...fit_gigg import fit_gigg_fixed, fit_gigg_mmle
    from ...fit_ghs_plus import fit_ghs_plus
    from ...fit_gr_rhs import fit_gr_rhs
    from ...fit_rhs import fit_rhs

    reg = MethodRegistry()

    reg.register(
        "GR_RHS",
        lambda c: fit_gr_rhs(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.grrhs_p0,
            sampler=c.sampler,
            **c.grrhs_kwargs,
        ),
    )
    reg.register(
        "RHS",
        lambda c: fit_rhs(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.p0,
            sampler=c.sampler,
        ),
    )
    reg.register(
        "GIGG_MMLE",
        lambda c: fit_gigg_mmle(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            sampler=c.sampler,
            p0=c.p0,
            **c.gigg_mmle_kwargs,
        ),
    )
    reg.register(
        "GIGG_b_small",
        lambda c: fit_gigg_fixed(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            sampler=c.sampler,
            p0=c.p0,
            a_val=1.0 / c.n,
            b_val=1.0 / c.n,
            method_label="GIGG_b_small",
            **c.gigg_fixed_kwargs,
        ),
    )
    reg.register(
        "GIGG_GHS",
        lambda c: fit_gigg_fixed(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            sampler=c.sampler,
            p0=c.p0,
            a_val=0.5,
            b_val=0.5,
            method_label="GIGG_GHS",
            **c.gigg_fixed_kwargs,
        ),
    )
    reg.register(
        "GIGG_b_large",
        lambda c: fit_gigg_fixed(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            sampler=c.sampler,
            p0=c.p0,
            a_val=1.0 / c.n,
            b_val=1.0,
            method_label="GIGG_b_large",
            **c.gigg_fixed_kwargs,
        ),
    )
    reg.register(
        "GHS_plus",
        lambda c: fit_ghs_plus(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.p0,
            sampler=c.sampler,
        ),
    )
    reg.register("OLS", lambda c: fit_ols(c.X, c.y, task=c.task, seed=c.seed))
    reg.register("LASSO_CV", lambda c: fit_lasso_cv(c.X, c.y, task=c.task, seed=c.seed))
    return reg
