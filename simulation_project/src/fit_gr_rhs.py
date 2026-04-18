from __future__ import annotations

from typing import Sequence

import numpy as np

from simulation_project.src.core.models.grrhs_nuts import GRRHS_NUTS, GRRHS_CollapsedNUTS, GRRHS_Gibbs

from .utils import FitResult, SamplerConfig, diagnostics_summary_for_method, logistic_pseudo_sigma, timed_call

BACKENDS = ("nuts", "collapsed", "gibbs")


def _build_nuts(
    *,
    task: str,
    seed: int,
    p0: int,
    alpha_kappa: float,
    beta_kappa: float,
    use_group_scale: bool,
    use_local_scale: bool,
    shared_kappa: bool,
    auto_calibrate_tau: bool,
    tau0: float | None,
    sigma_reference: float,
    sampler: SamplerConfig,
    adapt_delta: float,
    max_treedepth: int,
) -> GRRHS_NUTS:
    likelihood = "logistic" if str(task).lower() == "logistic" else "gaussian"
    return GRRHS_NUTS(
        alpha_kappa=float(alpha_kappa),
        beta_kappa=float(beta_kappa),
        eta=1.0,
        p0=int(max(p0, 1)),
        tau0=None if tau0 is None else float(tau0),
        auto_calibrate_tau=bool(auto_calibrate_tau),
        sigma_reference=float(sigma_reference),
        likelihood=likelihood,
        use_group_scale=bool(use_group_scale),
        use_local_scale=bool(use_local_scale),
        shared_kappa=bool(shared_kappa),
        num_warmup=int(sampler.warmup),
        num_samples=int(sampler.post_warmup_draws),
        num_chains=int(sampler.chains),
        target_accept_prob=float(adapt_delta),
        max_tree_depth=int(max_treedepth),
        dense_mass=False,
        chain_method="sequential",
        progress_bar=False,
        seed=int(seed),
    )


def _build_collapsed(
    *,
    task: str,
    seed: int,
    p0: int,
    alpha_kappa: float,
    beta_kappa: float,
    use_group_scale: bool,
    use_local_scale: bool,
    shared_kappa: bool,
    auto_calibrate_tau: bool,
    tau0: float | None,
    sigma_reference: float,
    sampler: SamplerConfig,
    adapt_delta: float,
    max_treedepth: int,
) -> GRRHS_CollapsedNUTS:
    if str(task).lower() == "logistic":
        raise ValueError("GRRHS_CollapsedNUTS does not support logistic likelihood")
    return GRRHS_CollapsedNUTS(
        alpha_kappa=float(alpha_kappa),
        beta_kappa=float(beta_kappa),
        eta=1.0,
        p0=int(max(p0, 1)),
        tau0=None if tau0 is None else float(tau0),
        auto_calibrate_tau=bool(auto_calibrate_tau),
        sigma_reference=float(sigma_reference),
        use_group_scale=bool(use_group_scale),
        use_local_scale=bool(use_local_scale),
        shared_kappa=bool(shared_kappa),
        num_warmup=int(sampler.warmup),
        num_samples=int(sampler.post_warmup_draws),
        num_chains=int(sampler.chains),
        target_accept_prob=float(adapt_delta),
        max_tree_depth=int(max_treedepth),
        dense_mass=False,
        chain_method="sequential",
        progress_bar=False,
        seed=int(seed),
        beta_draws_per_sample=1,
        sigma_jitter=1e-6,
    )


def _build_gibbs(
    *,
    task: str,
    seed: int,
    p0: int,
    alpha_kappa: float,
    beta_kappa: float,
    use_group_scale: bool,
    use_local_scale: bool,
    shared_kappa: bool,
    auto_calibrate_tau: bool,
    tau0: float | None,
    sigma_reference: float,
    sampler: SamplerConfig,
) -> GRRHS_Gibbs:
    if str(task).lower() == "logistic":
        raise ValueError("GRRHS_Gibbs does not support logistic likelihood")
    burnin = int(sampler.warmup)
    kept = int(sampler.post_warmup_draws)
    return GRRHS_Gibbs(
        alpha_kappa=float(alpha_kappa),
        beta_kappa=float(beta_kappa),
        eta=0.5,
        p0=int(max(p0, 1)),
        tau0=None if tau0 is None else float(tau0),
        auto_calibrate_tau=bool(auto_calibrate_tau),
        sigma_reference=float(sigma_reference),
        use_group_scale=bool(use_group_scale),
        use_local_scale=bool(use_local_scale),
        shared_kappa=bool(shared_kappa),
        iters=burnin + kept,
        burnin=burnin,
        thin=1,
        num_chains=int(sampler.chains),
        seed=int(seed),
        progress_bar=False,
    )


def _build_model(backend: str, **kwargs):
    b = str(backend).strip().lower()
    if b == "nuts":
        return _build_nuts(**kwargs)
    if b == "collapsed":
        return _build_collapsed(**kwargs)
    if b == "gibbs":
        # Gibbs has no NUTS-specific params
        gibbs_keys = {k for k in kwargs if k not in ("adapt_delta", "max_treedepth")}
        return _build_gibbs(**{k: v for k, v in kwargs.items() if k in gibbs_keys})
    raise ValueError(f"unknown sampler backend: {backend!r}; expected one of {BACKENDS}")


def fit_gr_rhs(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    p0: int,
    sampler: SamplerConfig,
    alpha_kappa: float = 0.5,
    beta_kappa: float = 1.0,
    use_group_scale: bool = True,
    use_local_scale: bool = True,
    shared_kappa: bool = False,
    auto_calibrate_tau: bool = True,
    tau0: float | None = None,
    backend: str = "nuts",
) -> FitResult:
    tracked = ["beta", "tau", "kappa", "a"]
    b = str(backend).strip().lower()

    common_kwargs = dict(
        task=task,
        seed=seed,
        p0=p0,
        alpha_kappa=alpha_kappa,
        beta_kappa=beta_kappa,
        use_group_scale=use_group_scale,
        use_local_scale=use_local_scale,
        shared_kappa=shared_kappa,
        auto_calibrate_tau=auto_calibrate_tau,
        tau0=tau0,
        sampler=sampler,
    )

    try:
        pseudo_sigma = 1.0
        if str(task).lower() == "logistic":
            pseudo_sigma = logistic_pseudo_sigma(y)

        def _make(seed_: int, adapt_delta: float, max_treedepth: int):
            kw = dict(**common_kwargs, sigma_reference=pseudo_sigma, seed=seed_)
            if b in ("nuts", "collapsed"):
                kw["adapt_delta"] = adapt_delta
                kw["max_treedepth"] = max_treedepth
            return _build_model(b, **kw)

        model = _make(seed, float(sampler.adapt_delta), int(sampler.max_treedepth))
        model, runtime = timed_call(model.fit, X, y, groups=[list(map(int, g)) for g in groups])
        beta_draws = getattr(model, "coef_samples_", None)
        beta_mean = getattr(model, "coef_mean_", None)
        tau_draws = getattr(model, "tau_samples_", None)
        kappa_draws = getattr(model, "kappa_samples_", None)
        a_draws = getattr(model, "a_samples_", None)

        rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
            model=model,
            tracked_params=tracked,
            beta_draws=beta_draws,
            config=sampler,
        )

        # Auto-retry with stricter NUTS settings (only meaningful for gradient-based samplers)
        if b in ("nuts", "collapsed") and np.isfinite(div_ratio) and div_ratio >= float(sampler.max_divergence_ratio):
            strict = _make(seed + 999, float(sampler.strict_adapt_delta), int(sampler.strict_max_treedepth))
            strict, runtime2 = timed_call(strict.fit, X, y, groups=[list(map(int, g)) for g in groups])
            beta_draws = getattr(strict, "coef_samples_", None)
            beta_mean = getattr(strict, "coef_mean_", None)
            tau_draws = getattr(strict, "tau_samples_", None)
            kappa_draws = getattr(strict, "kappa_samples_", None)
            a_draws = getattr(strict, "a_samples_", None)
            rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
                model=strict,
                tracked_params=tracked,
                beta_draws=beta_draws,
                config=sampler,
            )
            runtime += runtime2

        return FitResult(
            method="GR_RHS",
            status="ok",
            beta_mean=None if beta_mean is None else np.asarray(beta_mean, dtype=float),
            beta_draws=None if beta_draws is None else np.asarray(beta_draws, dtype=float),
            kappa_draws=None if kappa_draws is None else np.asarray(kappa_draws, dtype=float),
            group_scale_draws=None if a_draws is None else np.asarray(a_draws, dtype=float),
            tau_draws=None if tau_draws is None else np.asarray(tau_draws, dtype=float),
            runtime_seconds=float(runtime),
            rhat_max=float(rhat_max),
            bulk_ess_min=float(ess_min),
            divergence_ratio=float(div_ratio),
            converged=bool(converged),
            diagnostics=details,
        )
    except Exception as exc:
        return FitResult(
            method="GR_RHS",
            status="error",
            beta_mean=None,
            beta_draws=None,
            kappa_draws=None,
            group_scale_draws=None,
            tau_draws=None,
            runtime_seconds=float("nan"),
            rhat_max=float("nan"),
            bulk_ess_min=float("nan"),
            divergence_ratio=float("nan"),
            converged=False,
            error=f"{type(exc).__name__}: {exc}",
            diagnostics={},
        )
