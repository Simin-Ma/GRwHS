from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from simulation_project.src.core.models.grrhs_nuts import GRRHS_CollapsedNUTS, GRRHS_Gibbs_Staged, GRRHS_NUTS

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
        diag = getattr(model, "sampler_diagnostics_", {})
        backend = "nuts"
        if isinstance(diag, dict) and str(diag.get("backend", "")).strip() == "simcore_collapsed_nuts":
            backend = "collapsed_profile"
        return {"backend": backend, "init_params": init_params}
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
    allowed = {"tau", "tau_raw"}
    if gaussian:
        allowed.add("sigma")
    if gaussian:
        allowed.add("beta_raw")
    if bool(use_local_scale):
        allowed.add("lambda")
        allowed.add("log_lambda_raw")
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


def _beta_logit_moments(alpha: float, beta: float) -> tuple[float, float]:
    a = max(float(alpha), 1e-8)
    b = max(float(beta), 1e-8)
    mean = math.log(a) - math.log(b)
    var = max(1.0 / a + 1.0 / b, 1e-8)
    return float(mean), float(math.sqrt(var))


def _collapsed_profile_init_params(
    init_params: dict[str, np.ndarray] | None,
    *,
    alpha_kappa: float,
    beta_kappa: float,
    shared_kappa: bool,
    kappa_reparameterization: str,
    tau_parameterization: str = "sigma_scaled",
    lambda_parameterization: str = "prior_log_affine",
) -> dict[str, np.ndarray] | None:
    if not isinstance(init_params, dict) or not init_params:
        return None
    out = _clone_numeric_dict(init_params) or {}
    tau_mode = str(tau_parameterization).strip().lower()
    if tau_mode == "sigma_scaled" and "tau_raw" not in out and "tau" in out and "sigma" in out:
        sigma = float(np.asarray(out["sigma"], dtype=float))
        tau = float(np.asarray(out["tau"], dtype=float))
        denom = max(abs(sigma), 1e-8)
        out["tau_raw"] = np.asarray(max(tau / denom, 1e-4), dtype=np.float32)
        out.pop("tau", None)
    lambda_mode = str(lambda_parameterization).strip().lower()
    if lambda_mode == "prior_log_affine" and "log_lambda_raw" not in out and "lambda" in out:
        lam = np.asarray(out["lambda"], dtype=float)
        lam = np.maximum(np.nan_to_num(lam, nan=1.0, posinf=1.0, neginf=1.0), 1e-6)
        out["log_lambda_raw"] = np.asarray(np.log(lam) / (math.pi / 2.0), dtype=np.float32)
        out.pop("lambda", None)
    mode = str(kappa_reparameterization).strip().lower()
    if mode != "prior_logit_affine":
        return out or None
    loc, scale = _beta_logit_moments(alpha_kappa, beta_kappa)
    if not np.isfinite(scale) or scale <= 0.0:
        return out or None
    key = "logit_kappa_shared_raw" if bool(shared_kappa) else "logit_kappa_raw"
    if key not in out:
        return out or None
    raw = np.asarray(out[key], dtype=float)
    out[key] = np.asarray((raw - float(loc)) / float(scale), dtype=np.float32)
    return out or None


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


def _collapsed_profile_ridge_init_params(
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
    kappa_reparameterization: str,
    ridge: float = 1.0,
) -> dict[str, np.ndarray]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    p = int(X_arr.shape[1])
    group_list = [np.asarray(g, dtype=int) for g in groups]
    G = max(len(group_list), 1)
    beta_ridge = _ridge_beta_mean(X_arr, y_arr, task=task, ridge=ridge)

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

    group_mass = np.zeros(G, dtype=float)
    for gid, idxs in enumerate(group_list):
        if idxs.size == 0:
            continue
        beta_g = beta_ridge[idxs]
        group_mass[gid] = float(np.linalg.norm(beta_g) / math.sqrt(max(int(idxs.size), 1)))
    total_mass = float(np.sum(group_mass))
    if total_mass > 0.0 and np.isfinite(total_mass):
        group_share = group_mass / total_mass
    else:
        group_share = np.full(G, 1.0 / max(G, 1), dtype=float)

    active_score = float(np.sum(group_share > (1.0 / max(2 * G, 1))))
    tau_guess = float(max(tau0_eff, np.linalg.norm(beta_ridge) / math.sqrt(max(p, 1)), 1e-4))
    if str(tau_target).strip().lower() == "groups":
        tau_guess = float(max(tau_guess, tau0_eff * max(1.0, math.sqrt(max(active_score, 1.0)))))

    prior_mean_kappa = float(alpha_kappa / max(alpha_kappa + beta_kappa, 1e-8))
    centered_share = group_share - float(np.mean(group_share))
    share_scale = float(max(np.std(group_share), 1e-6))
    kappa_guess = np.clip(
        prior_mean_kappa + 0.30 * centered_share / share_scale,
        0.02,
        0.98,
    )
    if not np.any(np.isfinite(kappa_guess)):
        kappa_guess = np.full(G, _clip_prob(prior_mean_kappa), dtype=float)
    kappa_guess = np.where(np.isfinite(kappa_guess), kappa_guess, _clip_prob(prior_mean_kappa))
    logit_kappa = np.asarray([_safe_logit(v) for v in kappa_guess], dtype=np.float32)

    key = "logit_kappa_shared_raw" if bool(shared_kappa) else "logit_kappa_raw"
    if str(kappa_reparameterization).strip().lower() == "prior_logit_affine":
        loc, scale = _beta_logit_moments(alpha_kappa, beta_kappa)
        scale_use = float(max(scale, 1e-6))
        logit_kappa = ((logit_kappa.astype(float) - float(loc)) / scale_use).astype(np.float32)

    out: dict[str, np.ndarray] = {
        "sigma": np.asarray(float(max(sigma_guess, 1e-4)), dtype=np.float32),
        "tau": np.asarray(float(max(tau_guess, 1e-4)), dtype=np.float32),
    }
    if use_local_scale:
        local_scale = np.sqrt(np.clip(np.abs(beta_ridge) / max(tau_guess, 1e-4), 0.05, 5.0))
        out["lambda"] = np.asarray(local_scale, dtype=np.float32)
    if shared_kappa:
        out[key] = np.asarray(float(np.mean(logit_kappa)), dtype=np.float32)
    else:
        out[key] = np.asarray(logit_kappa, dtype=np.float32)
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
    budget_scale: float | None = None,
) -> GRRHS_Gibbs_Staged:
    warmup = max(40, int(sampler.warmup))
    draws = max(20, int(sampler.post_warmup_draws))
    budget_scale_use = 1.0 if budget_scale is None else float(max(0.05, min(float(budget_scale), 4.0)))
    light_budget = bool(budget_scale is not None and budget_scale_use < 0.999)
    hard_mult = 2.2 if bool(hard_design) else 1.25
    micro_budget = bool(light_budget and budget_scale_use <= 0.15)
    if micro_budget:
        phase_a_floor = 16
        phase_b_floor = 10
        hyper_only_floor = 6
        min_phase_a_floor = 8
        min_phase_b_floor = 6
        window_floor = 6
    else:
        phase_a_floor = 48 if light_budget else 220
        phase_b_floor = 28 if light_budget else 140
        hyper_only_floor = 16 if light_budget else 80
        min_phase_a_floor = 24 if light_budget else 90
        min_phase_b_floor = 18 if light_budget else 60
        window_floor = 12 if light_budget else 24
    phase_a_max = int(max(phase_a_floor, round(warmup * 1.55 * hard_mult * budget_scale_use)))
    phase_b_max = int(max(phase_b_floor, round(warmup * 0.80 * hard_mult * budget_scale_use)))
    if micro_budget:
        phase_a_hyper_only_cap = 18
        min_phase_a_cap = 24
        min_phase_b_cap = 18
        geometry_window_cap = 18
        transition_window_cap = 16
    else:
        phase_a_hyper_only_cap = 180 if light_budget else 420
        min_phase_a_cap = 180 if light_budget else 340
        min_phase_b_cap = 120 if light_budget else 240
        geometry_window_cap = 80 if light_budget else 160
        transition_window_cap = 70 if light_budget else 140
    phase_a_hyper_only = int(max(hyper_only_floor, min(phase_a_hyper_only_cap, round(phase_a_max * 0.52))))
    min_phase_a = int(max(min_phase_a_floor, min(min_phase_a_cap, round(phase_a_max * 0.62))))
    min_phase_b = int(max(min_phase_b_floor, min(min_phase_b_cap, round(phase_b_max * 0.62))))
    geometry_window = int(max(window_floor, min(geometry_window_cap, round(min_phase_a * 0.62))))
    transition_window = int(max(window_floor, min(transition_window_cap, round(min_phase_b * 0.62))))
    total_iters = int(max(4, phase_a_max + phase_b_max + draws))
    burnin = int(max(0, min(phase_a_max + phase_b_max, total_iters - 1)))
    grouped_tau_refresh_repeats = 1
    grouped_sigma_tau_block_repeats = 2
    slice_max_steps = 200
    phase_a_late_beta_refresh = True
    phase_b_extra_beta_refresh = True
    concentrated_block_refresh = True
    structure_aware_warmup = True
    distributed_block_top_groups = 3
    concentrated_top_groups = 2
    phase_a_refresh_interval = 12
    phase_a_refresh_repeats = 1
    phase_b_initial_extra_refresh_steps = 20
    phase_b_initial_refresh_repeats = 1
    phase_a_block_refresh_interval = 10
    phase_b_block_refresh_steps = 24
    phase_b_block_refresh_repeats = 1

    if light_budget:
        slice_max_steps = 96 if not micro_budget else 56
        grouped_tau_refresh_repeats = 0 if micro_budget else 1
        grouped_sigma_tau_block_repeats = 0 if micro_budget else 1
        phase_a_refresh_interval = 18 if not micro_budget else 999999
        phase_a_refresh_repeats = 0 if micro_budget else 1
        phase_b_initial_extra_refresh_steps = 8 if not micro_budget else 0
        phase_b_initial_refresh_repeats = 0 if micro_budget else 1
        phase_a_block_refresh_interval = 18 if not micro_budget else 999999
        phase_b_block_refresh_steps = 8 if not micro_budget else 0
        phase_b_block_refresh_repeats = 0 if micro_budget else 1
        distributed_block_top_groups = 2 if not micro_budget else 1
        concentrated_top_groups = 1
        if budget_scale_use <= 0.35:
            phase_a_late_beta_refresh = False
            concentrated_block_refresh = False
        if budget_scale_use <= 0.20:
            phase_b_extra_beta_refresh = False
            structure_aware_warmup = False
        if micro_budget:
            phase_a_late_beta_refresh = False
            phase_b_extra_beta_refresh = False
            concentrated_block_refresh = False
            structure_aware_warmup = False

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
        slice_max_steps=int(slice_max_steps),
        grouped_tau_refresh_repeats=int(grouped_tau_refresh_repeats),
        grouped_sigma_tau_block_repeats=int(grouped_sigma_tau_block_repeats),
        phase_a_late_beta_refresh=bool(phase_a_late_beta_refresh),
        phase_b_extra_beta_refresh=bool(phase_b_extra_beta_refresh),
        phase_a_refresh_interval=int(phase_a_refresh_interval),
        phase_a_refresh_repeats=int(phase_a_refresh_repeats),
        phase_b_initial_extra_refresh_steps=int(phase_b_initial_extra_refresh_steps),
        phase_b_initial_refresh_repeats=int(phase_b_initial_refresh_repeats),
        concentrated_block_refresh=bool(concentrated_block_refresh),
        concentrated_top_groups=int(concentrated_top_groups),
        phase_a_block_refresh_interval=int(phase_a_block_refresh_interval),
        phase_b_block_refresh_steps=int(phase_b_block_refresh_steps),
        phase_b_block_refresh_repeats=int(phase_b_block_refresh_repeats),
        structure_aware_warmup=bool(structure_aware_warmup),
        distributed_block_top_groups=int(distributed_block_top_groups),
    )


def _build_collapsed_profile(
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
    init_params: dict[str, np.ndarray] | None = None,
    resume_no_warmup: bool = False,
    adapt_delta: float | None = None,
    max_treedepth: int | None = None,
    kappa_reparameterization: str = "prior_logit_affine",
    tau_parameterization: str = "sigma_scaled",
    lambda_parameterization: str = "prior_log_affine",
    step_size: float | None = None,
    find_heuristic_step_size: bool = False,
    dense_mass: bool = False,
) -> GRRHS_CollapsedNUTS:
    """Build GR-RHS-CaP: collapsed-and-profile GR-RHS for high-dimensional Gaussian runs."""
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
        target_accept_prob=float(sampler.adapt_delta if adapt_delta is None else adapt_delta),
        max_tree_depth=int(sampler.max_treedepth if max_treedepth is None else max_treedepth),
        dense_mass=bool(dense_mass),
        chain_method="sequential",
        progress_bar=bool(progress_bar),
        seed=int(seed),
        step_size=None if step_size is None else float(step_size),
        find_heuristic_step_size=bool(find_heuristic_step_size),
        init_params=_clone_numeric_dict(init_params),
        resume_no_warmup=bool(resume_no_warmup),
        kappa_reparameterization=str(kappa_reparameterization),
        tau_parameterization=str(tau_parameterization),
        lambda_parameterization=str(lambda_parameterization),
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
    gibbs_budget_scale: float | None = None,
    collapsed_kappa_reparameterization: str = "prior_logit_affine",
    collapsed_dense_mass: bool | None = None,
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
        backend_name = str(sampler_backend or "gibbs_staged").strip().lower()
        backend_name = {
            "cap": "collapsed_profile",
            "grrhs_cap": "collapsed_profile",
            "gr-rhs-cap": "collapsed_profile",
            "collapsed": "collapsed_profile",
            "collapsed_nuts": "collapsed_profile",
            "profile_collapsed": "collapsed_profile",
            "collapsed_profile": "collapsed_profile",
            "gibbs": "gibbs_staged",
            "staged_gibbs": "gibbs_staged",
            "gibbs_staged": "gibbs_staged",
            "nuts": "nuts",
        }.get(backend_name, "gibbs_staged")
        if task_name == "logistic" and backend_name == "gibbs_staged":
            raise NotImplementedError(
                "GR_RHS staged Gibbs is currently implemented for Gaussian likelihood only; "
                "logistic Gibbs is not yet available."
            )
        if task_name == "logistic" and backend_name == "collapsed_profile":
            raise NotImplementedError(
                "GR-RHS-CaP collapsed/profile sampling is currently implemented for Gaussian likelihood only."
            )
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
        collapsed_ridge_init = None
        if warm_mode == "ridge":
            collapsed_ridge_init = _collapsed_profile_ridge_init_params(
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
                kappa_reparameterization=str(collapsed_kappa_reparameterization),
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
                budget_scale=gibbs_budget_scale,
            )

        def _make_collapsed_profile(
            seed_: int,
            adapt_delta: float,
            max_treedepth: int,
            resume_payload: dict[str, Any] | None,
        ):
            init_params = _resume_init_params(resume_payload)
            warm_init = init_params
            if warm_init is None:
                base_init = collapsed_ridge_init if collapsed_ridge_init is not None else ridge_init
                warm_init = _expand_manual_init_params_for_chains(
                    _clone_numeric_dict(base_init),
                    num_chains=int(sampler.chains),
                )
            warm_init = _collapsed_profile_init_params(
                _filter_nuts_init_params(
                    warm_init,
                    task=task,
                    use_local_scale=use_local_scale,
                    shared_kappa=shared_kappa,
                ),
                alpha_kappa=alpha_kappa,
                beta_kappa=beta_kappa,
                shared_kappa=shared_kappa,
                kappa_reparameterization=str(collapsed_kappa_reparameterization),
                tau_parameterization="sigma_scaled",
                lambda_parameterization="prior_log_affine",
            )
            dense_mass_use = bool(collapsed_dense_mass) if collapsed_dense_mass is not None else (
                bool(design_profile.get("hard_design", False))
                and str(task).strip().lower() != "logistic"
                and str(tau_target).strip().lower() == "groups"
            )
            hard_profile = bool(design_profile.get("hard_design", False))
            within_corr = float(design_profile.get("within_mean_abs_corr", 0.0))
            corr_gap = float(design_profile.get("corr_gap", 0.0))
            step_size_use = None
            heuristic_step_size = False
            if (
                hard_profile
                and bool(use_local_scale)
                and str(task).strip().lower() != "logistic"
            ):
                heuristic_step_size = True
                if within_corr >= 0.7 or corr_gap >= 0.5:
                    step_size_use = 0.01
                elif within_corr >= 0.55 or corr_gap >= 0.35:
                    step_size_use = 0.02
            return _build_collapsed_profile(
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
                init_params=warm_init,
                resume_no_warmup=bool(init_params),
                adapt_delta=float(adapt_delta),
                max_treedepth=int(max_treedepth),
                kappa_reparameterization=str(collapsed_kappa_reparameterization),
                tau_parameterization="sigma_scaled",
                lambda_parameterization="prior_log_affine",
                step_size=step_size_use,
                find_heuristic_step_size=bool(heuristic_step_size),
                dense_mass=bool(dense_mass_use),
            )

        if backend_name == "gibbs_staged":
            model = _make_gibbs(seed, resume_payload=retry_resume_payload)
        elif backend_name == "collapsed_profile":
            model = _make_collapsed_profile(
                seed,
                float(sampler.adapt_delta),
                int(sampler.max_treedepth),
                resume_payload=retry_resume_payload,
            )
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

        # Retry HMC/NUTS designs with stricter adaptation when divergences are high.
        if backend_name in {"nuts", "collapsed_profile"}:
            if np.isfinite(div_ratio) and div_ratio >= float(sampler.max_divergence_ratio):
                if backend_name == "collapsed_profile":
                    strict = _make_collapsed_profile(
                        seed + 999,
                        float(sampler.strict_adapt_delta),
                        int(sampler.strict_max_treedepth),
                        resume_payload=resume_payload_out,
                    )
                else:
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
            "gibbs_budget_scale": None if gibbs_budget_scale is None else float(gibbs_budget_scale),
            "collapsed_kappa_reparameterization": str(collapsed_kappa_reparameterization),
            "collapsed_dense_mass": None if collapsed_dense_mass is None else bool(collapsed_dense_mass),
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

