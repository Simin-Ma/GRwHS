from __future__ import annotations

from typing import Sequence

from .utils import FitResult, SamplerConfig


def as_int_groups(groups: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(map(int, g)) for g in groups]


def scaled_iteration_budget(
    sampler: SamplerConfig,
    *,
    iter_mult: int,
    iter_floor: int,
    iter_cap: int,
) -> tuple[int, int]:
    mult = max(1, int(iter_mult))
    floor = max(10, int(iter_floor))
    cap = max(floor, int(iter_cap))
    burnin = min(max(int(sampler.warmup) * mult, floor), cap)
    draws = min(max(int(sampler.post_warmup_draws) * mult, floor), cap)
    return burnin, draws


def fit_error_result(method: str, error: str) -> FitResult:
    return FitResult(
        method=str(method),
        status="error",
        beta_mean=None,
        beta_draws=None,
        kappa_draws=None,
        group_scale_draws=None,
        runtime_seconds=float("nan"),
        rhat_max=float("nan"),
        bulk_ess_min=float("nan"),
        divergence_ratio=float("nan"),
        converged=False,
        error=str(error),
        diagnostics={},
    )
