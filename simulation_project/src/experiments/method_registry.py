from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..utils import FitResult, SamplerConfig


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
    rhs_sampler_strategy: str
    rhs_kwargs: dict
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


def _mean_within_abs_corr(X: np.ndarray, groups: list[list[int]]) -> float:
    X_arr = np.asarray(X, dtype=float)
    vals: list[float] = []
    for members in groups:
        idx = np.asarray(list(members), dtype=int).reshape(-1)
        if idx.size <= 1:
            continue
        block = X_arr[:, idx]
        corr = np.corrcoef(block, rowvar=False)
        corr = np.asarray(corr, dtype=float)
        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            continue
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        if not np.any(mask):
            continue
        vals.extend(np.abs(corr[mask]).reshape(-1).tolist())
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def build_default_method_registry() -> MethodRegistry:
    reg = MethodRegistry()

    def _clean_gigg_kwargs(raw: dict) -> dict:
        kwargs = dict(raw)
        for key in ("allow_budget_retry", "extra_retry", "retry_cap"):
            kwargs.pop(key, None)
        return kwargs

    def _fit_rhs_lowdim(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_rhs import fit_rhs

        return fit_rhs(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.p0,
            sampler=c.sampler,
            method_name=str(method_name),
        )

    def _rhs_highdim_exact_sampler(c: MethodContext) -> SamplerConfig:
        within_corr = _mean_within_abs_corr(np.asarray(c.X, dtype=float), c.groups)
        if np.isfinite(within_corr) and within_corr >= 0.75:
            warmup = max(int(c.sampler.warmup), 1100)
            draws = max(int(c.sampler.post_warmup_draws), 2400)
            adapt_delta = max(0.99, float(c.sampler.adapt_delta))
        else:
            warmup = max(int(c.sampler.warmup), 1000)
            draws = max(int(c.sampler.post_warmup_draws), 2000)
            adapt_delta = max(0.985, float(c.sampler.adapt_delta))
        return SamplerConfig(
            chains=max(4, int(c.sampler.chains)),
            warmup=int(warmup),
            post_warmup_draws=int(draws),
            adapt_delta=float(adapt_delta),
            max_treedepth=max(14, int(c.sampler.max_treedepth)),
            strict_adapt_delta=max(0.995, float(c.sampler.strict_adapt_delta)),
            strict_max_treedepth=max(15, int(c.sampler.strict_max_treedepth)),
            max_divergence_ratio=min(0.01, float(c.sampler.max_divergence_ratio)),
            rhat_threshold=float(c.sampler.rhat_threshold),
            ess_threshold=float(c.sampler.ess_threshold),
        )

    def _gigg_highdim_exact_kwargs(c: MethodContext) -> dict:
        kwargs = _clean_gigg_kwargs(c.gigg_mmle_kwargs)
        within_corr = _mean_within_abs_corr(np.asarray(c.X, dtype=float), c.groups)
        kwargs["exact_highdim_fastpath"] = True
        if np.isfinite(within_corr) and within_corr >= 0.75:
            kwargs["highdim_continuation_rounds"] = max(int(kwargs.get("highdim_continuation_rounds", 0)), 280)
            kwargs["highdim_continuation_warmup"] = max(int(kwargs.get("highdim_continuation_warmup", 0) or 0), 2)
            kwargs["highdim_continuation_draws"] = max(int(kwargs.get("highdim_continuation_draws", 0) or 0), 5)
            kwargs["highdim_stage_a_burnin"] = max(int(kwargs.get("highdim_stage_a_burnin", 0) or 0), 8)
            kwargs["highdim_stage_a_draws"] = max(int(kwargs.get("highdim_stage_a_draws", 0) or 0), 8)
        return kwargs

    def _ghs_highdim_light_sampler(c: MethodContext) -> SamplerConfig:
        return SamplerConfig(
            chains=max(4, int(c.sampler.chains)),
            warmup=max(500, int(c.sampler.warmup)),
            post_warmup_draws=max(500, int(c.sampler.post_warmup_draws)),
            adapt_delta=max(0.95, float(c.sampler.adapt_delta)),
            max_treedepth=max(12, int(c.sampler.max_treedepth)),
            strict_adapt_delta=max(0.99, float(c.sampler.strict_adapt_delta)),
            strict_max_treedepth=max(14, int(c.sampler.strict_max_treedepth)),
            max_divergence_ratio=min(0.01, float(c.sampler.max_divergence_ratio)),
            rhat_threshold=float(c.sampler.rhat_threshold),
            ess_threshold=max(400.0, float(c.sampler.ess_threshold)),
        )

    def _ghs_highdim_iter_budget(c: MethodContext) -> tuple[int, int, int]:
        within_corr = _mean_within_abs_corr(np.asarray(c.X, dtype=float), c.groups)
        if np.isfinite(within_corr) and within_corr >= 0.75:
            return 1, 500, 800
        return 1, 450, 700

    def _fit_rhs_highdim(c: MethodContext, *, method_name: str) -> FitResult:
        sampler_use = _rhs_highdim_exact_sampler(c)
        res = fit_rhs(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.p0,
            sampler=sampler_use,
            method_name=str(method_name),
            progress_bar=False,
        )
        diag = dict(res.diagnostics or {})
        diag["rhs_highdim_route"] = "stan_exact"
        diag["rhs_highdim_sampler_budget"] = {
            "chains": int(sampler_use.chains),
            "warmup": int(sampler_use.warmup),
            "post_warmup_draws": int(sampler_use.post_warmup_draws),
            "adapt_delta": float(sampler_use.adapt_delta),
            "max_treedepth": int(sampler_use.max_treedepth),
            "strict_adapt_delta": float(sampler_use.strict_adapt_delta),
            "strict_max_treedepth": int(sampler_use.strict_max_treedepth),
        }
        res.diagnostics = diag
        return res

    def _fit_rhs_highdim_gibbs(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_rhs_gibbs import fit_rhs_gibbs

        kwargs = dict(c.rhs_kwargs)
        return fit_rhs_gibbs(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.p0,
            sampler=c.sampler,
            method_name=str(method_name),
            **kwargs,
        )

    def _grrhs_kwargs_for_strategy(c: MethodContext, *, high_dim: bool) -> dict:
        base = dict(c.grrhs_kwargs)
        if high_dim:
            defaults = {
                "tau_target": "groups",
                "sampler_backend": "collapsed_profile",
                "use_local_scale": False,
                "progress_bar": False,
            }
        else:
            defaults = {
                "tau_target": "groups",
                "sampler_backend": "gibbs_staged",
                "progress_bar": False,
            }
        return {**defaults, **base}

    def _fit_grrhs_lowdim(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_gr_rhs import fit_gr_rhs

        return fit_gr_rhs(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.grrhs_p0,
            sampler=c.sampler,
            method_name=str(method_name),
            **_grrhs_kwargs_for_strategy(c, high_dim=False),
        )

    def _fit_grrhs_highdim(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_gr_rhs import fit_gr_rhs

        return fit_gr_rhs(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.grrhs_p0,
            sampler=c.sampler,
            method_name=str(method_name),
            **_grrhs_kwargs_for_strategy(c, high_dim=True),
        )

    def _fit_gigg_mmle_lowdim(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_gigg import fit_gigg_mmle

        kwargs = _clean_gigg_kwargs(c.gigg_mmle_kwargs)
        kwargs["exact_highdim_fastpath"] = False
        kwargs.setdefault("progress_bar", False)
        return fit_gigg_mmle(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            sampler=c.sampler,
            p0=c.p0,
            method_label=str(method_name),
            **kwargs,
        )

    def _fit_gigg_mmle_highdim(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_gigg import fit_gigg_mmle

        kwargs = _gigg_highdim_exact_kwargs(c)
        kwargs.setdefault("progress_bar", False)
        return fit_gigg_mmle(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            sampler=c.sampler,
            p0=c.p0,
            method_label=str(method_name),
            **kwargs,
        )

    def _fit_ghs_plus_lowdim(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_ghs_plus import fit_ghs_plus

        return fit_ghs_plus(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.p0,
            sampler=c.sampler,
            progress_bar=False,
            use_process_pool=True,
        )

    def _fit_ghs_plus_highdim(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_ghs_plus import fit_ghs_plus

        sampler_use = _ghs_highdim_light_sampler(c)
        iter_mult, iter_floor, iter_cap = _ghs_highdim_iter_budget(c)
        res = fit_ghs_plus(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.p0,
            sampler=sampler_use,
            iter_mult=iter_mult,
            iter_floor=iter_floor,
            iter_cap=iter_cap,
            progress_bar=False,
            use_process_pool=False,
        )
        diag = dict(res.diagnostics or {})
        diag["ghs_highdim_route"] = "gibbs_light_exact"
        diag["ghs_highdim_sampler_budget"] = {
            "chains": int(sampler_use.chains),
            "warmup": int(sampler_use.warmup),
            "post_warmup_draws": int(sampler_use.post_warmup_draws),
            "iter_mult": int(iter_mult),
            "iter_floor": int(iter_floor),
            "iter_cap": int(iter_cap),
            "adapt_delta": float(sampler_use.adapt_delta),
            "max_treedepth": int(sampler_use.max_treedepth),
            "strict_adapt_delta": float(sampler_use.strict_adapt_delta),
            "strict_max_treedepth": int(sampler_use.strict_max_treedepth),
            "ess_threshold": float(sampler_use.ess_threshold),
        }
        res.diagnostics = diag
        return res

    reg.register(
        "GR_RHS",
        lambda c: (
            _fit_grrhs_highdim(c, method_name="GR_RHS")
            if str(c.rhs_sampler_strategy).strip().lower() == "high_dim"
            else _fit_grrhs_lowdim(c, method_name="GR_RHS")
        ),
    )
    reg.register(
        "GR_RHS_LowDim",
        lambda c: _fit_grrhs_lowdim(c, method_name="GR_RHS_LowDim"),
    )
    reg.register(
        "GR_RHS_HighDim",
        lambda c: _fit_grrhs_highdim(c, method_name="GR_RHS_HighDim"),
    )
    reg.register(
        "RHS",
        lambda c: (
            _fit_rhs_highdim(c, method_name="RHS")
            if str(c.rhs_sampler_strategy).strip().lower() == "high_dim"
            else _fit_rhs_lowdim(c, method_name="RHS")
        ),
    )
    reg.register(
        "RHS_LowDim",
        lambda c: _fit_rhs_lowdim(c, method_name="RHS_LowDim"),
    )
    reg.register(
        "RHS_HighDim",
        lambda c: _fit_rhs_highdim(c, method_name="RHS_HighDim"),
    )
    reg.register(
        "RHS_Gibbs",
        lambda c: _fit_rhs_highdim_gibbs(c, method_name="RHS_Gibbs"),
    )
    reg.register(
        "GIGG_MMLE",
        lambda c: (
            _fit_gigg_mmle_highdim(c, method_name="GIGG_MMLE")
            if str(c.rhs_sampler_strategy).strip().lower() == "high_dim"
            else _fit_gigg_mmle_lowdim(c, method_name="GIGG_MMLE")
        ),
    )
    reg.register(
        "GIGG_MMLE_LowDim",
        lambda c: _fit_gigg_mmle_lowdim(c, method_name="GIGG_MMLE_LowDim"),
    )
    reg.register(
        "GIGG_MMLE_HighDim",
        lambda c: _fit_gigg_mmle_highdim(c, method_name="GIGG_MMLE_HighDim"),
    )
    reg.register(
        "GIGG_b_small",
        lambda c: __import__("simulation_project.src.experiments.methods.fit_gigg", fromlist=["fit_gigg_fixed"]).fit_gigg_fixed(
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
        lambda c: __import__("simulation_project.src.experiments.methods.fit_gigg", fromlist=["fit_gigg_fixed"]).fit_gigg_fixed(
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
        lambda c: __import__("simulation_project.src.experiments.methods.fit_gigg", fromlist=["fit_gigg_fixed"]).fit_gigg_fixed(
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
        lambda c: (
            _fit_ghs_plus_highdim(c, method_name="GHS_plus")
            if str(c.rhs_sampler_strategy).strip().lower() == "high_dim"
            else _fit_ghs_plus_lowdim(c, method_name="GHS_plus")
        ),
    )
    reg.register(
        "OLS",
        lambda c: __import__(
            "simulation_project.src.experiments.methods.fit_classical",
            fromlist=["fit_ols"],
        ).fit_ols(c.X, c.y, task=c.task, seed=c.seed),
    )
    reg.register(
        "LASSO_CV",
        lambda c: __import__(
            "simulation_project.src.experiments.methods.fit_classical",
            fromlist=["fit_lasso_cv"],
        ).fit_lasso_cv(c.X, c.y, task=c.task, seed=c.seed),
    )
    return reg


