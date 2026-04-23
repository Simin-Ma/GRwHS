from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from simulation_project.src.core.models.grrhs_nuts import GRRHS_NUTS, GRRHS_CollapsedNUTS, GRRHS_Gibbs

from .helpers import as_int_groups, fit_error_result
from ...utils import FitResult, SamplerConfig, diagnostics_summary_for_method, logistic_pseudo_sigma, timed_call

BACKENDS = ("nuts", "collapsed", "gibbs")


def _clone_numeric_dict(obj: dict[str, Any] | None) -> dict[str, np.ndarray] | None:
    if not isinstance(obj, dict) or not obj:
        return None
    out: dict[str, np.ndarray] = {}
    for k, v in obj.items():
        if isinstance(v, np.ndarray):
            out[str(k)] = np.asarray(v, dtype=float).copy()
            continue
        if isinstance(v, (list, tuple)):
            out[str(k)] = np.asarray(v, dtype=float).copy()
            continue
        if isinstance(v, (float, int, np.floating, np.integer)):
            out[str(k)] = np.asarray(float(v), dtype=float)
    return out or None


def _clone_chain_states(states: Any) -> list[dict[str, Any]] | None:
    if not isinstance(states, list) or not states:
        return None
    out: list[dict[str, Any]] = []
    for item in states:
        if not isinstance(item, dict):
            continue
        cur: dict[str, Any] = {}
        for key, val in item.items():
            if isinstance(val, np.ndarray):
                cur[str(key)] = np.asarray(val, dtype=float).copy()
            elif isinstance(val, (list, tuple)):
                cur[str(key)] = np.asarray(val, dtype=float).copy()
            elif isinstance(val, (float, int, np.floating, np.integer)):
                cur[str(key)] = float(val)
        if cur:
            out.append(cur)
    return out or None


def _resume_payload_for_backend(
    retry_resume_payload: dict[str, Any] | None,
    *,
    backend: str,
) -> tuple[dict[str, np.ndarray] | None, list[dict[str, Any]] | None]:
    if not isinstance(retry_resume_payload, dict):
        return None, None
    source_backend = str(retry_resume_payload.get("backend", "")).strip().lower()
    if source_backend != str(backend).strip().lower():
        return None, None
    init_params = _clone_numeric_dict(retry_resume_payload.get("init_params"))
    chain_states = _clone_chain_states(retry_resume_payload.get("chain_states"))
    return init_params, chain_states


def _extract_retry_resume_payload(*, model: Any, backend: str) -> dict[str, Any] | None:
    b = str(backend).strip().lower()
    if b in {"nuts", "collapsed"}:
        init_params = _clone_numeric_dict(getattr(model, "last_init_params_", None))
        if init_params:
            return {"backend": b, "init_params": init_params}
        return None
    if b == "gibbs":
        states = _clone_chain_states(getattr(model, "chain_last_states_", None))
        if states:
            return {"backend": b, "chain_states": states}
        return None
    return None


def _build_nuts(
    *,
    task: str,
    seed: int,
    p0: int,
    alpha_kappa: float,
    beta_kappa: float,
    use_local_scale: bool,
    shared_kappa: bool,
    auto_calibrate_tau: bool,
    tau0: float | None,
    tau_target: str,
    sigma_reference: float,
    sampler: SamplerConfig,
    adapt_delta: float,
    max_treedepth: int,
    progress_bar: bool,
    init_params: dict[str, np.ndarray] | None = None,
    resume_no_warmup: bool = False,
) -> GRRHS_NUTS:
    likelihood = "logistic" if str(task).lower() == "logistic" else "gaussian"
    return GRRHS_NUTS(
        alpha_kappa=float(alpha_kappa),
        beta_kappa=float(beta_kappa),
        eta=1.0,
        p0=int(max(p0, 1)),
        tau0=None if tau0 is None else float(tau0),
        auto_calibrate_tau=bool(auto_calibrate_tau),
        tau_target=str(tau_target),
        sigma_reference=float(sigma_reference),
        likelihood=likelihood,
        use_local_scale=bool(use_local_scale),
        shared_kappa=bool(shared_kappa),
        num_warmup=int(sampler.warmup),
        num_samples=int(sampler.post_warmup_draws),
        num_chains=int(sampler.chains),
        target_accept_prob=float(adapt_delta),
        max_tree_depth=int(max_treedepth),
        dense_mass=False,
        chain_method="sequential",
        progress_bar=bool(progress_bar),
        seed=int(seed),
        init_params=_clone_numeric_dict(init_params),
        resume_no_warmup=bool(resume_no_warmup),
    )


def _build_collapsed(
    *,
    task: str,
    seed: int,
    p0: int,
    alpha_kappa: float,
    beta_kappa: float,
    use_local_scale: bool,
    shared_kappa: bool,
    auto_calibrate_tau: bool,
    tau0: float | None,
    tau_target: str,
    sigma_reference: float,
    sampler: SamplerConfig,
    adapt_delta: float,
    max_treedepth: int,
    progress_bar: bool,
    init_params: dict[str, np.ndarray] | None = None,
    resume_no_warmup: bool = False,
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
        tau_target=str(tau_target),
        sigma_reference=float(sigma_reference),
        use_local_scale=bool(use_local_scale),
        shared_kappa=bool(shared_kappa),
        num_warmup=int(sampler.warmup),
        num_samples=int(sampler.post_warmup_draws),
        num_chains=int(sampler.chains),
        target_accept_prob=float(adapt_delta),
        max_tree_depth=int(max_treedepth),
        dense_mass=False,
        chain_method="sequential",
        progress_bar=bool(progress_bar),
        seed=int(seed),
        beta_draws_per_sample=1,
        sigma_jitter=1e-6,
        init_params=_clone_numeric_dict(init_params),
        resume_no_warmup=bool(resume_no_warmup),
    )


def _build_gibbs(
    *,
    task: str,
    seed: int,
    p0: int,
    alpha_kappa: float,
    beta_kappa: float,
    use_local_scale: bool,
    shared_kappa: bool,
    auto_calibrate_tau: bool,
    tau0: float | None,
    tau_target: str,
    sigma_reference: float,
    sampler: SamplerConfig,
    progress_bar: bool,
    initial_chain_states: list[dict[str, Any]] | None = None,
    resume_no_burnin: bool = False,
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
        tau_target=str(tau_target),
        sigma_reference=float(sigma_reference),
        use_local_scale=bool(use_local_scale),
        shared_kappa=bool(shared_kappa),
        iters=burnin + kept,
        burnin=burnin,
        thin=1,
        num_chains=int(sampler.chains),
        seed=int(seed),
        progress_bar=bool(progress_bar),
        initial_chain_states=_clone_chain_states(initial_chain_states),
        resume_no_burnin=bool(resume_no_burnin),
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
    use_local_scale: bool = True,
    shared_kappa: bool = False,
    auto_calibrate_tau: bool = True,
    tau0: float | None = None,
    tau_target: str = "coefficients",
    backend: str = "nuts",
    progress_bar: bool = True,
    retry_resume_payload: dict[str, Any] | None = None,
) -> FitResult:
    tracked = ["beta", "tau", "kappa"]
    b = str(backend).strip().lower()

    common_kwargs = dict(
        task=task,
        seed=seed,
        p0=p0,
        alpha_kappa=alpha_kappa,
        beta_kappa=beta_kappa,
        use_local_scale=use_local_scale,
        shared_kappa=shared_kappa,
        auto_calibrate_tau=auto_calibrate_tau,
        tau0=tau0,
        tau_target=tau_target,
        sampler=sampler,
        progress_bar=bool(progress_bar),
    )

    try:
        pseudo_sigma = 1.0
        if str(task).lower() == "logistic":
            pseudo_sigma = logistic_pseudo_sigma(y)

        def _make(
            seed_: int,
            adapt_delta: float,
            max_treedepth: int,
            resume_payload: dict[str, Any] | None,
        ):
            kw = dict(common_kwargs)
            kw["sigma_reference"] = pseudo_sigma
            kw["seed"] = seed_
            init_params, chain_states = _resume_payload_for_backend(resume_payload, backend=b)
            if b in ("nuts", "collapsed"):
                kw["adapt_delta"] = adapt_delta
                kw["max_treedepth"] = max_treedepth
                kw["init_params"] = init_params
                kw["resume_no_warmup"] = bool(init_params)
            elif b == "gibbs":
                kw["initial_chain_states"] = chain_states
                kw["resume_no_burnin"] = bool(chain_states)
            return _build_model(b, **kw)

        model = _make(
            seed,
            float(sampler.adapt_delta),
            int(sampler.max_treedepth),
            resume_payload=retry_resume_payload,
        )
        model, runtime = timed_call(model.fit, X, y, groups=as_int_groups(groups))
        resume_payload_out = _extract_retry_resume_payload(model=model, backend=b)
        beta_draws = getattr(model, "coef_samples_", None)
        beta_mean = getattr(model, "coef_mean_", None)
        tau_draws = getattr(model, "tau_samples_", None)
        kappa_draws = getattr(model, "kappa_samples_", None)

        rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
            model=model,
            tracked_params=tracked,
            beta_draws=beta_draws,
            config=sampler,
        )

        # Auto-retry with stricter NUTS settings (only meaningful for gradient-based samplers)
        if b in ("nuts", "collapsed") and np.isfinite(div_ratio) and div_ratio >= float(sampler.max_divergence_ratio):
            strict = _make(
                seed + 999,
                float(sampler.strict_adapt_delta),
                int(sampler.strict_max_treedepth),
                resume_payload=resume_payload_out,
            )
            strict, runtime2 = timed_call(strict.fit, X, y, groups=as_int_groups(groups))
            resume_payload_out = _extract_retry_resume_payload(model=strict, backend=b)
            beta_draws = getattr(strict, "coef_samples_", None)
            beta_mean = getattr(strict, "coef_mean_", None)
            tau_draws = getattr(strict, "tau_samples_", None)
            kappa_draws = getattr(strict, "kappa_samples_", None)
            rhat_max, ess_min, div_ratio, converged, details = diagnostics_summary_for_method(
                model=strict,
                tracked_params=tracked,
                beta_draws=beta_draws,
                config=sampler,
            )
            runtime += runtime2

        diagnostics = dict(details or {})
        if resume_payload_out is not None:
            diagnostics["retry_resume_payload"] = resume_payload_out

        return FitResult(
            method="GR_RHS",
            status="ok",
            beta_mean=None if beta_mean is None else np.asarray(beta_mean, dtype=float),
            beta_draws=None if beta_draws is None else np.asarray(beta_draws, dtype=float),
            kappa_draws=None if kappa_draws is None else np.asarray(kappa_draws, dtype=float),
            group_scale_draws=None,
            tau_draws=None if tau_draws is None else np.asarray(tau_draws, dtype=float),
            runtime_seconds=float(runtime),
            rhat_max=float(rhat_max),
            bulk_ess_min=float(ess_min),
            divergence_ratio=float(div_ratio),
            converged=bool(converged),
            diagnostics=diagnostics,
        )
    except Exception as exc:
        res = fit_error_result("GR_RHS", f"{type(exc).__name__}: {exc}")
        res.tau_draws = None
        return res

