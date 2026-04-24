from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from simulation_project.src.core.models.grrhs_nuts import GRRHS_Gibbs_Staged, GRRHS_NUTS

from .helpers import as_int_groups, fit_error_result
from ...utils import FitResult, SamplerConfig, diagnostics_summary_for_method, logistic_pseudo_sigma, timed_call


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


def _resume_init_params(
    retry_resume_payload: dict[str, Any] | None,
) -> dict[str, np.ndarray] | None:
    if not isinstance(retry_resume_payload, dict):
        return None
    return _clone_numeric_dict(retry_resume_payload.get("init_params"))


def _resume_gibbs_states(
    retry_resume_payload: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    if not isinstance(retry_resume_payload, dict):
        return None
    states = retry_resume_payload.get("chain_last_states")
    if not isinstance(states, list) or not states:
        return None
    out: list[dict[str, Any]] = []
    for item in states:
        cloned = _clone_numeric_dict(item if isinstance(item, dict) else None)
        if cloned:
            out.append(cloned)
    return out or None


def _extract_retry_resume_payload(*, model: Any) -> dict[str, Any] | None:
    init_params = _clone_numeric_dict(getattr(model, "last_init_params_", None))
    if init_params:
        return {"backend": "nuts", "init_params": init_params}
    chain_last_states = getattr(model, "chain_last_states_", None)
    if isinstance(chain_last_states, list) and chain_last_states:
        payload_states: list[dict[str, Any]] = []
        for st in chain_last_states:
            cloned = _clone_numeric_dict(st if isinstance(st, dict) else None)
            if cloned:
                payload_states.append(cloned)
        if payload_states:
            return {"backend": "gibbs_staged", "chain_last_states": payload_states}
    return None


def _filter_nuts_init_params(
    init_params: dict[str, np.ndarray] | None,
    *,
    task: str,
    use_local_scale: bool,
    shared_kappa: bool,
) -> dict[str, np.ndarray] | None:
    if not isinstance(init_params, dict) or not init_params:
        return None
    gaussian = str(task).strip().lower() != "logistic"
    allowed = {"tau"}
    if gaussian:
        allowed.add("sigma")
    if gaussian:
        allowed.add("beta_raw")
    if bool(use_local_scale):
        allowed.add("lambda")
    if bool(shared_kappa):
        allowed.add("logit_kappa_shared_raw")
    else:
        allowed.add("logit_kappa_raw")
    out = {str(k): np.asarray(v, dtype=float).copy() for k, v in init_params.items() if str(k) in allowed}
    return out or None


def _expand_manual_init_params_for_chains(
    init_params: dict[str, np.ndarray] | None,
    *,
    num_chains: int,
) -> dict[str, np.ndarray] | None:
    if not isinstance(init_params, dict) or not init_params:
        return None
    chains = max(1, int(num_chains))
    out: dict[str, np.ndarray] = {}
    for key, value in init_params.items():
        arr = np.asarray(value, dtype=float)
        if chains <= 1:
            out[str(key)] = arr.copy()
            continue
        if arr.ndim == 0:
            out[str(key)] = np.repeat(arr.reshape(1), chains, axis=0)
            continue
        out[str(key)] = np.repeat(np.expand_dims(arr, axis=0), chains, axis=0)
    return out or None


def _clip_prob(value: float, *, eps: float = 1e-4) -> float:
    return float(min(max(float(value), eps), 1.0 - eps))


def _safe_logit(value: float, *, eps: float = 1e-4) -> float:
    p = _clip_prob(value, eps=eps)
    return float(math.log(p / (1.0 - p)))


def _ridge_beta_mean(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    ridge: float,
) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    p = int(X_arr.shape[1])
    pen = max(float(ridge), 1e-6)
    lhs = X_arr.T @ X_arr + np.eye(p, dtype=float) * pen
    rhs = X_arr.T @ y_arr
    if str(task).strip().lower() == "logistic":
        y_center = y_arr - float(np.mean(y_arr))
        rhs = X_arr.T @ y_center
    try:
        beta = np.linalg.solve(lhs, rhs)
    except Exception:
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    beta = np.nan_to_num(np.asarray(beta, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return beta.reshape(-1)


def _ridge_init_params(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    p0: int,
    alpha_kappa: float,
    beta_kappa: float,
    tau_target: str,
    sigma_reference: float,
    shared_kappa: bool,
    use_local_scale: bool,
    ridge: float = 1.0,
) -> dict[str, np.ndarray]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    p = int(X_arr.shape[1])
    group_list = [np.asarray(g, dtype=int) for g in groups]
    G = len(group_list)
    beta_ridge = _ridge_beta_mean(X_arr, y_arr, task=task, ridge=ridge)
    beta_scale = float(max(np.std(beta_ridge), 0.25))
    beta_raw = np.clip(beta_ridge / beta_scale, -4.0, 4.0).astype(np.float32)

    resid = y_arr - X_arr @ beta_ridge
    sigma_guess = float(np.sqrt(max(np.mean(resid * resid), 1e-6)))
    if str(task).strip().lower() == "logistic":
        sigma_guess = float(max(sigma_reference, 1.0))

    target_dim = p if str(tau_target).strip().lower() == "coefficients" else max(G, 1)
    tau0_eff = GRRHS_NUTS.calibrate_tau0(
        p0=max(float(p0), 1.0),
        D=max(int(target_dim), 1),
        n=max(int(X_arr.shape[0]), 1),
        sigma_ref=float(max(sigma_reference, 1e-6)),
    )
    beta_norm = float(np.linalg.norm(beta_ridge))
    tau_guess = float(max(tau0_eff, beta_norm / math.sqrt(max(p, 1)), 1e-4))

    group_mass = np.zeros(G, dtype=float)
    for gid, idxs in enumerate(group_list):
        if idxs.size == 0:
            continue
        group_mass[gid] = float(np.linalg.norm(beta_ridge[idxs]))
    if np.max(group_mass) > 0.0:
        group_mass = group_mass / float(np.max(group_mass))
    prior_mean_kappa = float(alpha_kappa / max(alpha_kappa + beta_kappa, 1e-8))
    kappa_guess = np.clip(0.05 + 0.8 * group_mass, 0.02, 0.98)
    if not np.any(np.isfinite(kappa_guess)):
        kappa_guess = np.full(G, _clip_prob(prior_mean_kappa), dtype=float)
    kappa_guess = np.where(np.isfinite(kappa_guess), kappa_guess, _clip_prob(prior_mean_kappa))
    kappa_guess = 0.5 * kappa_guess + 0.5 * _clip_prob(prior_mean_kappa)
    logit_kappa = np.asarray([_safe_logit(v) for v in kappa_guess], dtype=np.float32)

    out: dict[str, np.ndarray] = {
        "sigma": np.asarray(float(max(sigma_guess, 1e-4)), dtype=np.float32),
        "tau": np.asarray(float(max(tau_guess, 1e-4)), dtype=np.float32),
    }
    if use_local_scale:
        local_scale = np.sqrt(np.clip(np.abs(beta_ridge) / max(tau_guess, 1e-4), 0.05, 5.0))
        out["lambda"] = np.asarray(local_scale, dtype=np.float32)
    if shared_kappa:
        out["logit_kappa_shared_raw"] = np.asarray(float(np.mean(logit_kappa)), dtype=np.float32)
    else:
        out["logit_kappa_raw"] = np.asarray(logit_kappa, dtype=np.float32)
    if str(task).strip().lower() != "logistic":
        out["beta_raw"] = beta_raw
    return out


def _design_hardness_profile(
    X: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    tau_target: str,
) -> dict[str, float | bool | str]:
    X_arr = np.asarray(X, dtype=float)
    p = int(X_arr.shape[1])
    group_list = [np.asarray(g, dtype=int) for g in groups]
    G = max(len(group_list), 1)
    corr = np.corrcoef(X_arr, rowvar=False) if p > 1 else np.ones((p, p), dtype=float)
    corr = np.nan_to_num(np.asarray(corr, dtype=float), nan=0.0)
    np.fill_diagonal(corr, 1.0)

    within_vals: list[float] = []
    between_vals: list[float] = []
    max_group = 0
    for gid, idxs in enumerate(group_list):
        max_group = max(max_group, int(idxs.size))
        if idxs.size > 1:
            block = np.abs(corr[np.ix_(idxs, idxs)])
            tri = block[np.triu_indices(int(idxs.size), k=1)]
            if tri.size:
                within_vals.append(float(np.mean(tri)))
        other = np.setdiff1d(np.arange(p, dtype=int), idxs, assume_unique=False)
        if idxs.size and other.size:
            between_vals.append(float(np.mean(np.abs(corr[np.ix_(idxs, other)]))))

    within_mean = float(np.mean(within_vals)) if within_vals else 0.0
    between_mean = float(np.mean(between_vals)) if between_vals else 0.0
    corr_gap = float(within_mean - between_mean)
    max_group_ratio = float(max_group / max(p, 1))
    grouped_target = str(tau_target).strip().lower() == "groups"
    gaussian = str(task).strip().lower() != "logistic"
    hard = bool(
        gaussian
        and grouped_target
        and (
            within_mean >= 0.65
            or corr_gap >= 0.35
            or max_group_ratio >= 0.35
            or (within_mean >= 0.5 and max_group_ratio >= 0.2)
        )
    )
    return {
        "within_mean_abs_corr": within_mean,
        "between_mean_abs_corr": between_mean,
        "corr_gap": corr_gap,
        "max_group_ratio": max_group_ratio,
        "gaussian": gaussian,
        "grouped_target": grouped_target,
        "hard_design": hard,
    }


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


def _build_gibbs_staged(
    *,
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
    hard_design: bool = False,
) -> GRRHS_Gibbs_Staged:
    warmup = max(40, int(sampler.warmup))
    draws = max(20, int(sampler.post_warmup_draws))
    hard_mult = 1.6 if bool(hard_design) else 1.25
    phase_a_max = int(max(140, round(warmup * 1.10 * hard_mult)))
    phase_b_max = int(max(80, round(warmup * 0.45 * hard_mult)))
    phase_a_hyper_only = int(max(40, min(220, round(phase_a_max * 0.40))))
    min_phase_a = int(max(50, min(180, round(phase_a_max * 0.50))))
    min_phase_b = int(max(30, min(120, round(phase_b_max * 0.50))))
    geometry_window = int(max(16, min(80, round(min_phase_a * 0.55))))
    transition_window = int(max(16, min(70, round(min_phase_b * 0.55))))
    total_iters = int(max(4, phase_a_max + phase_b_max + draws))
    burnin = int(max(0, min(phase_a_max + phase_b_max, total_iters - 1)))
    return GRRHS_Gibbs_Staged(
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
        iters=int(total_iters),
        burnin=int(burnin),
        thin=1,
        num_chains=int(sampler.chains),
        progress_bar=bool(progress_bar),
        seed=int(seed),
        initial_chain_states=initial_chain_states,
        resume_no_burnin=bool(resume_no_burnin),
        phase_a_max_iters=int(phase_a_max),
        phase_b_max_iters=int(phase_b_max),
        phase_a_hyper_only_iters=int(phase_a_hyper_only),
        min_phase_a_iters=int(min_phase_a),
        min_phase_b_iters=int(min_phase_b),
        geometry_window=int(geometry_window),
        transition_window=int(transition_window),
        geometry_tol=0.075 if not bool(hard_design) else 0.095,
        transition_tol=0.09 if not bool(hard_design) else 0.11,
    )


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
    progress_bar: bool = True,
    retry_resume_payload: dict[str, Any] | None = None,
    retry_attempt: int = 0,
    warm_start_strategy: str = "ridge",
    sampler_backend: str | None = None,
) -> FitResult:
    tracked = ["beta", "tau", "kappa"]
    groups_use = as_int_groups(groups)

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
        design_profile = _design_hardness_profile(
            X,
            groups_use,
            task=task,
            tau_target=tau_target,
        )
        warm_mode = str(warm_start_strategy).strip().lower()
        if warm_mode not in {"ridge", "none"}:
            warm_mode = "ridge"
        task_name = str(task).strip().lower()
        backend_name = str(sampler_backend or ("gibbs_staged" if task_name != "logistic" else "nuts")).strip().lower()
        if backend_name not in {"nuts", "gibbs_staged"}:
            backend_name = "gibbs_staged" if task_name != "logistic" else "nuts"
        if task_name == "logistic":
            backend_name = "nuts"
        ridge_init = None
        if warm_mode == "ridge":
            ridge_init = _ridge_init_params(
                X,
                y,
                groups_use,
                task=task,
                p0=p0,
                alpha_kappa=alpha_kappa,
                beta_kappa=beta_kappa,
                tau_target=tau_target,
                sigma_reference=float(pseudo_sigma),
                shared_kappa=shared_kappa,
                use_local_scale=use_local_scale,
            )

        def _make_nuts(
            seed_: int,
            adapt_delta: float,
            max_treedepth: int,
            resume_payload: dict[str, Any] | None,
        ):
            kw = dict(common_kwargs)
            kw["sigma_reference"] = pseudo_sigma
            kw["seed"] = seed_
            kw["adapt_delta"] = adapt_delta
            kw["max_treedepth"] = max_treedepth
            init_params = _resume_init_params(resume_payload)
            warm_init = init_params
            if warm_init is None:
                warm_init = _expand_manual_init_params_for_chains(
                    _clone_numeric_dict(ridge_init),
                    num_chains=int(sampler.chains),
                )
            kw["init_params"] = _filter_nuts_init_params(
                warm_init,
                task=task,
                use_local_scale=use_local_scale,
                shared_kappa=shared_kappa,
            )
            kw["resume_no_warmup"] = bool(init_params)
            return _build_nuts(**kw)

        def _make_gibbs(
            seed_: int,
            resume_payload: dict[str, Any] | None,
        ):
            init_states = _resume_gibbs_states(resume_payload)
            return _build_gibbs_staged(
                seed=seed_,
                p0=p0,
                alpha_kappa=alpha_kappa,
                beta_kappa=beta_kappa,
                use_local_scale=use_local_scale,
                shared_kappa=shared_kappa,
                auto_calibrate_tau=auto_calibrate_tau,
                tau0=tau0,
                tau_target=tau_target,
                sigma_reference=float(pseudo_sigma),
                sampler=sampler,
                progress_bar=bool(progress_bar),
                initial_chain_states=init_states,
                resume_no_burnin=bool(init_states),
                hard_design=bool(design_profile.get("hard_design", False)),
            )

        if backend_name == "gibbs_staged":
            model = _make_gibbs(seed, resume_payload=retry_resume_payload)
        else:
            model = _make_nuts(
                seed,
                float(sampler.adapt_delta),
                int(sampler.max_treedepth),
                resume_payload=retry_resume_payload,
            )
        model, runtime = timed_call(model.fit, X, y, groups=groups_use)
        resume_payload_out = _extract_retry_resume_payload(model=model)
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

        # Retry the same NUTS design with stricter adaptation when divergences are high.
        if backend_name == "nuts":
            if np.isfinite(div_ratio) and div_ratio >= float(sampler.max_divergence_ratio):
                strict = _make_nuts(
                    seed + 999,
                    float(sampler.strict_adapt_delta),
                    int(sampler.strict_max_treedepth),
                    resume_payload=resume_payload_out,
                )
                strict, runtime2 = timed_call(strict.fit, X, y, groups=groups_use)
                resume_payload_out = _extract_retry_resume_payload(model=strict)
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
        diagnostics["sampling_strategy"] = {
            "backend": str(backend_name),
            "retry_attempt": int(max(retry_attempt, 0)),
            "warm_start_strategy": str(warm_mode),
            "hard_design": bool(design_profile.get("hard_design", False)),
            "design_profile": design_profile,
        }
        sampler_diag = diagnostics.get("sampler_diagnostics")
        if backend_name == "gibbs_staged" and isinstance(sampler_diag, dict):
            diagnostics["sampling_strategy"]["staged_defaults"] = {
                "phase_a_max_iters": sampler_diag.get("phase_a_max_iters"),
                "phase_b_max_iters": sampler_diag.get("phase_b_max_iters"),
                "min_phase_a_iters": sampler_diag.get("min_phase_a_iters"),
                "min_phase_b_iters": sampler_diag.get("min_phase_b_iters"),
                "geometry_window": sampler_diag.get("geometry_window"),
                "transition_window": sampler_diag.get("transition_window"),
                "geometry_tol": sampler_diag.get("geometry_tol"),
                "transition_tol": sampler_diag.get("transition_tol"),
            }

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

