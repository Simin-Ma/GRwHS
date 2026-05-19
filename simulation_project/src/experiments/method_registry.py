from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..utils import FitResult, SamplerConfig
from .runtime import _mean_within_abs_corr, highdim_iter_budget, highdim_sampler_budget


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


def _sampler_budget_dict(sampler: SamplerConfig) -> dict[str, int | float]:
    return {
        "chains": int(sampler.chains),
        "warmup": int(sampler.warmup),
        "post_warmup_draws": int(sampler.post_warmup_draws),
        "adapt_delta": float(sampler.adapt_delta),
        "max_treedepth": int(sampler.max_treedepth),
        "strict_adapt_delta": float(sampler.strict_adapt_delta),
        "strict_max_treedepth": int(sampler.strict_max_treedepth),
        "max_divergence_ratio": float(sampler.max_divergence_ratio),
        "rhat_threshold": float(sampler.rhat_threshold),
        "ess_threshold": float(sampler.ess_threshold),
    }


def _as_float_or_none(value: object) -> float | None:
    try:
        arr = np.asarray(value, dtype=float)
        if arr.size != 1:
            return None
        out = float(arr.reshape(-1)[0])
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _extract_grrhs_beta_diag(diag: dict[str, object]) -> dict[str, object]:
    adaptive = diag.get("grrhs_adaptive_beta")
    if not isinstance(adaptive, dict):
        return {}
    details = adaptive.get("details")
    if not isinstance(details, dict):
        details = {}
    out: dict[str, object] = {
        "strategy": str(adaptive.get("strategy", "adaptive_beta")),
    }
    alpha = _as_float_or_none(adaptive.get("alpha_kappa"))
    beta = _as_float_or_none(adaptive.get("beta_kappa"))
    if alpha is not None:
        out["alpha_kappa"] = float(alpha)
    if beta is not None:
        out["beta_kappa"] = float(beta)
    fallback_reason = details.get("fallback_reason")
    fallback_stage = details.get("fallback_stage")
    if fallback_reason is not None:
        out["fallback_reason"] = str(fallback_reason)
    if fallback_stage is not None:
        out["fallback_stage"] = str(fallback_stage)
    return out


def _attach_computational_protocol(
    res: FitResult,
    *,
    method_family: str,
    protocol: str,
    sampler_backend: str,
    posterior_target: str = "same_model_family",
    sampler: SamplerConfig | None = None,
    implementation: str | None = None,
    notes: list[str] | None = None,
    extra: dict[str, object] | None = None,
) -> FitResult:
    diag = dict(res.diagnostics or {})
    protocol_diag: dict[str, object] = {
        "method_family": str(method_family),
        "protocol": str(protocol),
        "sampler_backend": str(sampler_backend),
        "posterior_target": str(posterior_target),
    }
    if implementation is not None:
        protocol_diag["implementation"] = str(implementation)
    if sampler is not None:
        protocol_diag["sampler_budget"] = _sampler_budget_dict(sampler)
    if notes:
        protocol_diag["notes"] = [str(x) for x in notes]
    if extra:
        protocol_diag.update(dict(extra))
    if method_family == "GR_RHS":
        protocol_diag.update(_extract_grrhs_beta_diag(diag))
    diag["computational_protocol"] = protocol_diag
    diag["method_family"] = str(method_family)
    diag["protocol"] = str(protocol)
    diag["sampler_backend"] = str(sampler_backend)
    diag["strategy"] = str(protocol_diag.get("strategy", implementation or protocol))
    if "alpha_kappa" in protocol_diag:
        diag["alpha_kappa"] = protocol_diag["alpha_kappa"]
    if "beta_kappa" in protocol_diag:
        diag["beta_kappa"] = protocol_diag["beta_kappa"]
    if "fallback_reason" in protocol_diag:
        diag["fallback_reason"] = protocol_diag["fallback_reason"]
    if "fallback_stage" in protocol_diag:
        diag["fallback_stage"] = protocol_diag["fallback_stage"]
    diag["rhat_max"] = float(res.rhat_max) if np.isfinite(res.rhat_max) else float("nan")
    diag["bulk_ess_min"] = float(res.bulk_ess_min) if np.isfinite(res.bulk_ess_min) else float("nan")
    diag["divergence_ratio"] = float(res.divergence_ratio) if np.isfinite(res.divergence_ratio) else float("nan")
    diag["converged"] = bool(res.converged)
    if "fit_attempts" not in diag:
        diag["fit_attempts"] = 1
    res.diagnostics = diag
    return res


def build_default_method_registry() -> MethodRegistry:
    reg = MethodRegistry()

    def _clean_gigg_kwargs(raw: dict) -> dict:
        kwargs = dict(raw)
        for key in ("allow_budget_retry", "extra_retry", "retry_cap"):
            kwargs.pop(key, None)
        return kwargs

    def _fit_rhs_lowdim(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_rhs import fit_rhs

        res = fit_rhs(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.p0,
            sampler=c.sampler,
            method_name=str(method_name),
        )
        return _attach_computational_protocol(
            res,
            method_family="RHS",
            protocol="low_dim",
            sampler_backend="stan_rstanarm_hs",
            sampler=c.sampler,
            implementation="joint_stan_hmc",
            notes=["Baseline low-dimensional RHS computation uses the unified Stan/HMC implementation."],
        )

    def _rhs_highdim_exact_sampler(c: MethodContext) -> SamplerConfig:
        return highdim_sampler_budget(c.sampler, c.X, c.groups, role="rhs_exact")

    def _gigg_highdim_sampler(c: MethodContext) -> SamplerConfig:
        return highdim_sampler_budget(c.sampler, c.X, c.groups, role="gigg_mmle")

    def _ghs_highdim_light_sampler(c: MethodContext) -> SamplerConfig:
        return highdim_sampler_budget(c.sampler, c.X, c.groups, role="ghs_plus")

    def _ghs_highdim_iter_budget(c: MethodContext) -> tuple[int, int, int]:
        within_corr = _mean_within_abs_corr(np.asarray(c.X, dtype=float), c.groups)
        return highdim_iter_budget(within_corr=within_corr)

    def _fit_rhs_highdim(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_rhs import fit_rhs

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
        diag["rhs_sampler_name"] = str(method_name)
        diag["rhs_sampler_strategy"] = "high_dim"
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
        return _attach_computational_protocol(
            res,
            method_family="RHS",
            protocol="high_dim",
            sampler_backend="stan_exact",
            sampler=sampler_use,
            implementation="dimension_tuned_stan_hmc",
            notes=["High-dimensional RHS keeps the RHS model family but uses a strengthened HMC budget."],
        )

    def _fit_rhs_highdim_gibbs(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_rhs_gibbs import fit_rhs_gibbs

        kwargs = dict(c.rhs_kwargs)
        res = fit_rhs_gibbs(
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
        return _attach_computational_protocol(
            res,
            method_family="RHS",
            protocol="high_dim",
            sampler_backend="rhs_gibbs_woodbury",
            sampler=c.sampler,
            implementation="woodbury_gibbs",
            notes=["Explicit RHS_Gibbs route is a high-dimensional computation protocol for Gaussian RHS."],
        )

    def _grrhs_kwargs_for_strategy(c: MethodContext, *, high_dim: bool) -> dict:
        base = dict(c.grrhs_kwargs)
        if high_dim:
            defaults = {
                "tau_target": "groups",
                "sampler_backend": "collapsed_profile",
                "use_local_scale": False,
                "collapsed_hard_min_warmup": 150,
                "collapsed_hard_min_draws": 320,
                "progress_bar": False,
            }
        else:
            defaults = {
                "tau_target": "groups",
                "sampler_backend": "gibbs_staged",
                "progress_bar": False,
            }
        return {**defaults, **base}

    def _without_adaptive_beta_kwargs(kwargs: dict) -> dict:
        out = dict(kwargs)
        for key in (
            "adaptive_strategy",
            "calibration_warmup",
            "calibration_draws",
            "validation_fraction",
            "log_beta_min",
            "log_beta_max",
            "n_initial_points",
            "n_refine_points",
            "screening_null_quantile",
            "screening_permutations",
            "ridge_screening_scale",
            "multiplicity_correction",
            "multiplicity_level",
            "multiplicity_min_active_groups",
            "mcem_rounds",
            "mcem_step_size",
            "mcem_init_strategy",
            "mcem_calibration_chains",
            "mcem_calibration_adapt_delta",
            "mcem_calibration_max_treedepth",
            "posterior_eb_prior_center",
            "posterior_eb_prior_log_sd",
            "posterior_eb_damping",
            "min_beta_kappa",
            "max_beta_kappa",
        ):
            out.pop(key, None)
        return out

    def _fit_grrhs_lowdim(c: MethodContext, *, method_name: str) -> FitResult:
        return _fit_grrhs_adaptive(c, method_name=str(method_name))

    def _fit_grrhs_lowdim_with_beta(c: MethodContext, *, method_name: str, beta_kappa: float) -> FitResult:
        from .methods.fit_gr_rhs import fit_gr_rhs

        kwargs = _without_adaptive_beta_kwargs(_grrhs_kwargs_for_strategy(c, high_dim=False))
        kwargs["alpha_kappa"] = 0.5
        kwargs["beta_kappa"] = float(beta_kappa)
        res = fit_gr_rhs(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.grrhs_p0,
            sampler=c.sampler,
            method_name=str(method_name),
            **kwargs,
        )
        diag = dict(res.diagnostics or {})
        diag.setdefault("grrhs_sampler_name", str(method_name))
        diag.setdefault("grrhs_sampler_strategy", "low_dim")
        strat = dict(diag.get("sampling_strategy") or {})
        strat.setdefault("backend", "gibbs_staged")
        diag["sampling_strategy"] = strat
        res.diagnostics = diag
        return _attach_computational_protocol(
            res,
            method_family="GR_RHS",
            protocol="low_dim",
            sampler_backend="gibbs_staged",
            sampler=c.sampler,
            implementation="staged_gibbs",
            notes=["Low-dimensional GR-RHS beta-kappa variant uses the same staged Gibbs protocol."],
            extra={"beta_kappa": float(beta_kappa)},
        )

    def _fit_grrhs_highdim(c: MethodContext, *, method_name: str) -> FitResult:
        return _fit_grrhs_adaptive(c, method_name=str(method_name))

    def _fit_grrhs_highdim_with_beta(c: MethodContext, *, method_name: str, beta_kappa: float) -> FitResult:
        from .methods.fit_gr_rhs import fit_gr_rhs

        kwargs = _without_adaptive_beta_kwargs(_grrhs_kwargs_for_strategy(c, high_dim=True))
        kwargs["alpha_kappa"] = 0.5
        kwargs["beta_kappa"] = float(beta_kappa)
        res = fit_gr_rhs(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.grrhs_p0,
            sampler=c.sampler,
            method_name=str(method_name),
            **kwargs,
        )
        diag = dict(res.diagnostics or {})
        diag.setdefault("grrhs_sampler_name", str(method_name))
        diag.setdefault("grrhs_sampler_strategy", "high_dim")
        strat = dict(diag.get("sampling_strategy") or {})
        strat.setdefault("backend", "collapsed_profile")
        diag["sampling_strategy"] = strat
        res.diagnostics = diag
        return _attach_computational_protocol(
            res,
            method_family="GR_RHS",
            protocol="high_dim",
            sampler_backend="collapsed_profile",
            sampler=c.sampler,
            implementation="collapsed_group_hyperparameter_profile",
            notes=["High-dimensional GR-RHS beta-kappa variant uses the collapsed/profile computation protocol."],
            extra={"beta_kappa": float(beta_kappa)},
        )

    def _fit_grrhs_eb_highdim(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_gr_rhs_adaptive import fit_gr_rhs_adaptive_beta

        res = fit_gr_rhs_adaptive_beta(
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
        diag = dict(res.diagnostics or {})
        diag.setdefault("grrhs_sampler_name", str(method_name))
        diag.setdefault("grrhs_sampler_strategy", "high_dim")
        strat = dict(diag.get("sampling_strategy") or {})
        strat.setdefault("backend", "collapsed_profile")
        diag["sampling_strategy"] = strat
        res.diagnostics = diag
        return _attach_computational_protocol(
            res,
            method_family="GR_RHS",
            protocol="high_dim",
            sampler_backend="collapsed_profile",
            sampler=c.sampler,
            implementation="adaptive_beta_collapsed_profile",
            notes=["Adaptive GR-RHS EB is reported as a high-dimensional GR-RHS computation protocol."],
        )

    def _fit_grrhs_adaptive(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_gr_rhs_adaptive import fit_gr_rhs_adaptive_beta

        kwargs = _grrhs_kwargs_for_strategy(
            c,
            high_dim=(str(c.rhs_sampler_strategy).strip().lower() == "high_dim"),
        )
        high_dim = str(c.rhs_sampler_strategy).strip().lower() == "high_dim"
        kwargs["alpha_kappa"] = 0.5
        kwargs["adaptive_strategy"] = "regularized_posterior_eb"
        kwargs["beta_kappa"] = 4.0
        kwargs["posterior_eb_prior_center"] = 4.0
        kwargs["posterior_eb_prior_log_sd"] = 0.75
        kwargs["posterior_eb_damping"] = 0.5
        kwargs["max_beta_kappa"] = 12.0
        kwargs["multiplicity_correction"] = "fwer"
        kwargs["multiplicity_level"] = 0.05
        kwargs["screening_permutations"] = 120 if bool(high_dim) else 200
        res = fit_gr_rhs_adaptive_beta(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            p0=c.grrhs_p0,
            sampler=c.sampler,
            method_name=str(method_name),
            **kwargs,
        )
        protocol = "high_dim" if str(c.rhs_sampler_strategy).strip().lower() == "high_dim" else "low_dim"
        backend = "collapsed_profile" if protocol == "high_dim" else "gibbs_staged"
        diag = dict(res.diagnostics or {})
        diag.setdefault("grrhs_sampler_name", str(method_name))
        diag.setdefault("grrhs_sampler_strategy", protocol)
        strat = dict(diag.get("sampling_strategy") or {})
        strat.setdefault("backend", backend)
        diag["sampling_strategy"] = strat
        res.diagnostics = diag
        return _attach_computational_protocol(
            res,
            method_family="GR_RHS",
            protocol=protocol,
            sampler_backend=backend,
            sampler=c.sampler,
            implementation="adaptive_beta_regularized_posterior_eb",
            notes=["GR-RHS uses regularized posterior EB for common beta_kappa; only the posterior computation backend changes across regimes."],
        )

    def _fit_gigg_mmle_lowdim(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_gigg import fit_gigg_mmle

        kwargs = _clean_gigg_kwargs(c.gigg_mmle_kwargs)
        kwargs["exact_highdim_fastpath"] = False
        kwargs.setdefault("progress_bar", False)
        res = fit_gigg_mmle(
            c.X,
            c.y,
            c.groups,
            task=c.task,
            seed=c.seed,
            sampler=_gigg_highdim_sampler(c),
            p0=c.p0,
            method_label=str(method_name),
            **kwargs,
        )
        return _attach_computational_protocol(
            res,
            method_family="GIGG_MMLE",
            protocol="low_dim",
            sampler_backend="mmle_direct",
            sampler=c.sampler,
            implementation="paper_aligned_gibbs_mmle",
            notes=["Low-dimensional GIGG-MMLE uses the direct paper-aligned MMLE Gibbs route."],
        )

    def _fit_gigg_mmle_highdim(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_gigg import fit_gigg_mmle

        kwargs = _clean_gigg_kwargs(c.gigg_mmle_kwargs)
        kwargs["exact_highdim_fastpath"] = True
        kwargs.setdefault("progress_bar", False)
        res = fit_gigg_mmle(
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
        return _attach_computational_protocol(
            res,
            method_family="GIGG_MMLE",
            protocol="high_dim",
            sampler_backend="mmle_btrick",
            sampler=_gigg_highdim_sampler(c),
            implementation="single_mmle_gibbs_with_bhattacharya_beta_update",
            notes=["High-dimensional GIGG-MMLE follows the original/official MMLE Gibbs route and uses the Bhattacharya Gaussian block trick for beta updates."],
            extra={"exact_highdim_fastpath": True},
        )

    def _fit_ghs_plus_lowdim(c: MethodContext, *, method_name: str) -> FitResult:
        from .methods.fit_ghs_plus import fit_ghs_plus

        res = fit_ghs_plus(
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
        return _attach_computational_protocol(
            res,
            method_family="GHS_plus",
            protocol="low_dim",
            sampler_backend="gaussian_gibbs",
            sampler=c.sampler,
            implementation="paper_style_hbghs_gibbs",
            notes=["Low-dimensional GHS+ uses the Xu et al. HBGHS Gaussian Gibbs protocol."],
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
        return _attach_computational_protocol(
            res,
            method_family="GHS_plus",
            protocol="high_dim",
            sampler_backend="gibbs_light_exact",
            sampler=sampler_use,
            implementation="light_budget_hbghs_gibbs",
            notes=["High-dimensional GHS+ keeps the HBGHS model family but uses a light exact Gibbs budget."],
            extra={"iter_mult": int(iter_mult), "iter_floor": int(iter_floor), "iter_cap": int(iter_cap)},
        )

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
        "GR_RHS_EB",
        lambda c: _fit_grrhs_eb_highdim(c, method_name="GR_RHS_EB"),
    )
    reg.register(
        "GR_RHS_B01",
        lambda c: (
            _fit_grrhs_highdim_with_beta(c, method_name="GR_RHS_B01", beta_kappa=1.0)
            if str(c.rhs_sampler_strategy).strip().lower() == "high_dim"
            else _fit_grrhs_lowdim_with_beta(c, method_name="GR_RHS_B01", beta_kappa=1.0)
        ),
    )
    reg.register(
        "GR_RHS_B04",
        lambda c: (
            _fit_grrhs_highdim_with_beta(c, method_name="GR_RHS_B04", beta_kappa=4.0)
            if str(c.rhs_sampler_strategy).strip().lower() == "high_dim"
            else _fit_grrhs_lowdim_with_beta(c, method_name="GR_RHS_B04", beta_kappa=4.0)
        ),
    )
    reg.register(
        "GR_RHS_B08",
        lambda c: (
            _fit_grrhs_highdim_with_beta(c, method_name="GR_RHS_B08", beta_kappa=8.0)
            if str(c.rhs_sampler_strategy).strip().lower() == "high_dim"
            else _fit_grrhs_lowdim_with_beta(c, method_name="GR_RHS_B08", beta_kappa=8.0)
        ),
    )
    reg.register(
        "GR_RHS_Adaptive",
        lambda c: _fit_grrhs_adaptive(c, method_name="GR_RHS_Adaptive"),
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
        lambda c: _fit_gigg_mmle_highdim(c, method_name="GIGG_MMLE"),
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
        lambda c: __import__("simulation_second.src.bayes_kernel.experiments.methods.fit_gigg", fromlist=["fit_gigg_fixed"]).fit_gigg_fixed(
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
        lambda c: __import__("simulation_second.src.bayes_kernel.experiments.methods.fit_gigg", fromlist=["fit_gigg_fixed"]).fit_gigg_fixed(
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
        lambda c: __import__("simulation_second.src.bayes_kernel.experiments.methods.fit_gigg", fromlist=["fit_gigg_fixed"]).fit_gigg_fixed(
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
            "simulation_second.src.bayes_kernel.experiments.methods.fit_classical",
            fromlist=["fit_ols"],
        ).fit_ols(c.X, c.y, task=c.task, seed=c.seed),
    )
    reg.register(
        "LASSO_CV",
        lambda c: __import__(
            "simulation_second.src.bayes_kernel.experiments.methods.fit_classical",
            fromlist=["fit_lasso_cv"],
        ).fit_lasso_cv(c.X, c.y, task=c.task, seed=c.seed),
    )
    return reg



