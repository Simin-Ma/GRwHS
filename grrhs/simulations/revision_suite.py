from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from grrhs.diagnostics.convergence import summarize_convergence
from grrhs.models.baselines import (
    GroupedHorseshoePlus,
    HorseshoeRegression,
    RegularizedHorseshoeRegression,
)
from grrhs.models.grrhs_nuts import GRRHS_NUTS

try:
    from grrhs.models.gigg_regression import GIGGRegression
except Exception:  # pragma: no cover
    GIGGRegression = None  # type: ignore[assignment]

try:
    from grrhs.models.gigg_cran import GIGGRegressionCRAN
except Exception:  # pragma: no cover
    GIGGRegressionCRAN = None  # type: ignore[assignment]

_EPS = 1e-12
DEFAULT_METHODS = ("grrhs", "grrhs_flat", "rhs", "gigg_mmle", "hs")
PAPER_GAUSSIAN_METHODS = (
    "rhs",
    "grrhs_group_horseshoe",
    "grrhs_flat",
    "grrhs_recommended",
    "gigg_fixed",
    "gigg_mmle",
    "group_horseshoe_plus",
)


@dataclass(frozen=True)
class RevisionQuickProfile:
    theory_reps: int = 4
    benchmark_reps: int = 2
    heterogeneity_reps: int = 3
    inferential_reps: int = 3
    logistic_reps: int = 2
    ablation_reps: int = 2
    hyper_reps: int = 2
    nuts_warmup: int = 20
    nuts_samples: int = 20
    nuts_chains: int = 1
    rhs_warmup: int = 20
    rhs_samples: int = 20
    rhs_chains: int = 1


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _canonical_groups(group_sizes: Sequence[int]) -> List[List[int]]:
    cursor = 0
    groups: List[List[int]] = []
    for size in group_sizes:
        width = int(size)
        if width <= 0:
            raise ValueError("group sizes must be positive")
        groups.append(list(range(cursor, cursor + width)))
        cursor += width
    return groups


def _standardize(X: np.ndarray) -> np.ndarray:
    X_std = np.asarray(X, dtype=float).copy()
    X_std -= X_std.mean(axis=0, keepdims=True)
    scale = X_std.std(axis=0, ddof=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return X_std / scale


def _block_covariance(groups: Sequence[Sequence[int]], rho_within: float, rho_between: float) -> np.ndarray:
    p = int(sum(len(g) for g in groups))
    cov = np.full((p, p), float(rho_between), dtype=float)
    np.fill_diagonal(cov, 1.0)
    for members in groups:
        idx = np.asarray(members, dtype=int)
        for i in idx:
            for j in idx:
                if i != j:
                    cov[i, j] = float(rho_within)
    return cov


def _sample_design(
    n: int,
    groups: Sequence[Sequence[int]],
    rho_within: float,
    rho_between: float,
    rng: np.random.Generator,
) -> np.ndarray:
    cov = _block_covariance(groups, rho_within=rho_within, rho_between=rho_between)
    X = rng.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov, size=int(n))
    return _standardize(X)


def _sample_orthonormal_design(
    n: int,
    groups: Sequence[Sequence[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    p = int(sum(len(g) for g in groups))
    if int(n) < p:
        raise ValueError("orthonormal design requires n >= p")
    raw = rng.normal(size=(int(n), p))
    q, _ = np.linalg.qr(raw)
    return np.asarray(q[:, :p], dtype=float)


def _solve_signal_scale(X: np.ndarray, beta_shape: np.ndarray, snr: float) -> tuple[np.ndarray, float]:
    beta_shape = np.asarray(beta_shape, dtype=float)
    signal_raw = np.asarray(X @ beta_shape, dtype=float)
    signal_var = float(np.var(signal_raw))
    if signal_var <= 1e-12:
        return np.zeros_like(beta_shape), 1.0
    amp = math.sqrt(max(float(snr), 1e-8) / signal_var)
    return amp * beta_shape, 1.0


def _distributed_group_shape(groups: Sequence[Sequence[int]], gid: int, target_norm_sq: float) -> np.ndarray:
    p = int(sum(len(g) for g in groups))
    beta = np.zeros(p, dtype=float)
    members = np.asarray(groups[int(gid)], dtype=int)
    level = math.sqrt(max(float(target_norm_sq), 0.0) / max(members.size, 1))
    beta[members] = level
    return beta


def _concentrated_group_shape(groups: Sequence[Sequence[int]], gid: int, target_norm_sq: float) -> np.ndarray:
    p = int(sum(len(g) for g in groups))
    beta = np.zeros(p, dtype=float)
    members = np.asarray(groups[int(gid)], dtype=int)
    beta[members[0]] = math.sqrt(max(float(target_norm_sq), 0.0))
    return beta


def _sample_gaussian_response(X: np.ndarray, beta: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(X @ beta + float(sigma) * rng.normal(size=X.shape[0]), dtype=float)


def _sample_logistic_response(X: np.ndarray, beta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    logits = np.clip(np.asarray(X @ beta, dtype=float), -25.0, 25.0)
    prob = 1.0 / (1.0 + np.exp(-logits))
    return rng.binomial(1, prob, size=X.shape[0]).astype(float)


def _sample_profile_signal_group(
    p_g: int,
    mu: float,
    *,
    seed: int,
    sigma: float = 1.0,
    signal_type: str = "distributed",
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    beta = np.zeros(int(p_g), dtype=float)
    norm_sq = max(2.0 * float(mu), 0.0)
    signal_type_norm = str(signal_type).lower()
    if signal_type_norm == "concentrated":
        beta[0] = math.sqrt(norm_sq)
    elif signal_type_norm == "sparse":
        k_active = max(1, int(round(math.sqrt(max(int(p_g), 1)))))
        idx = rng.choice(int(p_g), size=k_active, replace=False)
        beta[idx] = math.sqrt(norm_sq / float(k_active))
    else:
        beta[:] = math.sqrt(norm_sq / max(int(p_g), 1))
    y = beta + float(sigma) * rng.normal(size=int(p_g))
    return {
        "y_group": np.asarray(y, dtype=float),
        "beta_group": beta,
        "mu": float(mu),
        "p_g": int(p_g),
        "sigma": float(sigma),
        "signal_type": signal_type_norm,
    }


def profile_prior_variance(kappa: np.ndarray | float, tau: float, sigma: float = 1.0) -> np.ndarray:
    kap = np.clip(np.asarray(kappa, dtype=float), _EPS, 1.0 - _EPS)
    tau2 = float(tau) ** 2
    sigma2 = float(sigma) ** 2
    num = sigma2 * kap * tau2
    den = sigma2 * kap + (1.0 - kap) * tau2
    return num / np.maximum(den, _EPS)


def profile_kappa_log_posterior_grid(
    y_group: np.ndarray,
    *,
    tau: float,
    sigma: float = 1.0,
    alpha_kappa: float = 1.0,
    beta_kappa: float = 1.0,
    grid_size: int = 2001,
) -> Dict[str, np.ndarray]:
    y = np.asarray(y_group, dtype=float).reshape(-1)
    p_g = int(y.size)
    grid = np.linspace(1e-5, 1.0 - 1e-5, int(grid_size))
    prior_var = profile_prior_variance(grid, tau=tau, sigma=sigma)
    marginal_var = float(sigma) ** 2 + prior_var
    sq = np.sum(y * y)
    loglik = -0.5 * p_g * np.log(2.0 * np.pi * marginal_var) - 0.5 * sq / np.maximum(marginal_var, _EPS)
    logprior = (float(alpha_kappa) - 1.0) * np.log(grid) + (float(beta_kappa) - 1.0) * np.log1p(-grid)
    logpost = loglik + logprior
    logpost -= float(np.max(logpost))
    weight = np.exp(logpost)
    weight /= np.trapezoid(weight, grid)
    return {"kappa": grid, "density": weight, "log_posterior": logpost}


def profile_kappa_posterior_summary(
    y_group: np.ndarray,
    *,
    tau: float,
    sigma: float = 1.0,
    alpha_kappa: float = 1.0,
    beta_kappa: float = 1.0,
    grid_size: int = 2001,
) -> Dict[str, float]:
    grid_obj = profile_kappa_log_posterior_grid(
        y_group,
        tau=tau,
        sigma=sigma,
        alpha_kappa=alpha_kappa,
        beta_kappa=beta_kappa,
        grid_size=grid_size,
    )
    grid = grid_obj["kappa"]
    dens = grid_obj["density"]
    mean = float(np.trapezoid(grid * dens, grid))
    cdf = np.cumsum(dens)
    cdf = cdf / max(float(cdf[-1]), _EPS)
    median = float(np.interp(0.5, cdf, grid))
    mode = float(grid[int(np.argmax(dens))])
    return {"mean": mean, "median": median, "mode": mode}


def profile_beta_posterior_mean_norm_sq(
    y_group: np.ndarray,
    *,
    tau: float,
    sigma: float = 1.0,
    alpha_kappa: float = 1.0,
    beta_kappa: float = 1.0,
    grid_size: int = 2001,
) -> float:
    y = np.asarray(y_group, dtype=float).reshape(-1)
    obj = profile_kappa_log_posterior_grid(
        y,
        tau=tau,
        sigma=sigma,
        alpha_kappa=alpha_kappa,
        beta_kappa=beta_kappa,
        grid_size=grid_size,
    )
    kappa = obj["kappa"]
    dens = obj["density"]
    prior_var = profile_prior_variance(kappa, tau=tau, sigma=sigma)
    shrink = prior_var / np.maximum(float(sigma) ** 2 + prior_var, _EPS)
    mean_shrink = float(np.trapezoid(shrink * dens, kappa))
    return float((mean_shrink**2) * np.dot(y, y))


def _theorem3p30_ratio_bounds(rho: float, alpha_kappa: float, beta_kappa: float) -> tuple[float, float]:
    rho2 = float(rho) ** 2
    alpha = max(float(alpha_kappa), 1e-8)
    beta = max(float(beta_kappa), 1e-8)
    center = rho2 / max(1.0 + rho2, 1e-8)
    loosen = math.sqrt((alpha + beta + 1.0) / max(alpha * beta, 1e-8))
    x_lower = max(center / max(loosen, 1e-8), 1e-6)
    x_upper = min(center * max(loosen, 1e-8), 10.0)
    if x_upper <= x_lower:
        x_upper = x_lower * 1.25
    return float(x_lower), float(x_upper)


def _posterior_draws(model: Any) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    mapping = {
        "beta": "coef_samples_",
        "sigma": "sigma_samples_",
        "tau": "tau_samples_",
        "lambda": "lambda_samples_",
        "phi": "phi_samples_",
        "a": "a_samples_",
        "kappa": "kappa_samples_",
        "c2": "c2_samples_",
        "intercept": "intercept_samples_",
        "c": "c_samples_",
    }
    for key, attr in mapping.items():
        value = getattr(model, attr, None)
        if value is not None:
            out[key] = np.asarray(value, dtype=float)
    return out


def _posterior_mean(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    data = np.asarray(arr, dtype=float)
    if data.ndim == 0:
        return data.reshape(1)
    if data.ndim == 1:
        return data
    if data.ndim == 2:
        return data.mean(axis=0)
    return data.reshape(-1, *data.shape[2:]).mean(axis=0)


def _flatten_draws(arr: Optional[np.ndarray], *, scalar: bool = False) -> Optional[np.ndarray]:
    if arr is None:
        return None
    data = np.asarray(arr, dtype=float)
    if scalar:
        return data.reshape(-1)
    if data.ndim == 1:
        return data.reshape(1, -1)
    if data.ndim == 2:
        return data
    return data.reshape(-1, *data.shape[2:])


def _coef_ci95(draws: Optional[np.ndarray]) -> Optional[np.ndarray]:
    flat = _flatten_draws(draws, scalar=False)
    if flat is None:
        return None
    return np.quantile(flat, [0.025, 0.975], axis=0)


def _group_norms(beta: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    beta_arr = np.asarray(beta, dtype=float)
    return np.array([float(np.sum(beta_arr[np.asarray(group, dtype=int)] ** 2)) for group in groups], dtype=float)


def _group_scores_from_model(model: Any, groups: Sequence[Sequence[int]]) -> Dict[str, np.ndarray]:
    posterior = _posterior_draws(model)
    beta_draws = _flatten_draws(posterior.get("beta"), scalar=False)
    beta_mean = _posterior_mean(posterior.get("beta"))
    if beta_mean is None:
        raise RuntimeError("fitted model does not expose coefficient draws or means")
    out: Dict[str, np.ndarray] = {"beta_norm_sq": _group_norms(beta_mean, groups)}
    kappa_mean = _posterior_mean(posterior.get("kappa"))
    if kappa_mean is not None:
        out["kappa_mean"] = np.asarray(kappa_mean, dtype=float).reshape(-1)
    if beta_draws is not None:
        out["beta_norm_draws"] = np.stack([_group_norms(draw, groups) for draw in beta_draws], axis=0)
    gamma2 = getattr(model, "gamma2_mean_", None)
    if gamma2 is None and posterior.get("gamma") is not None:
        gamma_draws = _flatten_draws(posterior.get("gamma"), scalar=False)
        if gamma_draws is not None:
            gamma2 = np.mean(np.square(gamma_draws), axis=0)
    if gamma2 is None:
        gamma2_draws = getattr(model, "gamma2_samples_", None)
        if gamma2_draws is not None:
            gamma2 = _posterior_mean(np.asarray(gamma2_draws, dtype=float))
    if gamma2 is None:
        gamma2 = getattr(model, "bg_mean_", None)
    if gamma2 is None and hasattr(model, "b_g_mean_"):
        gamma2 = getattr(model, "b_g_mean_")
    if gamma2 is not None:
        out["gamma2_mean"] = np.asarray(gamma2, dtype=float).reshape(-1)
    return out


def _ci_length_and_coverage(beta_true: np.ndarray, beta_draws: Optional[np.ndarray]) -> tuple[float, float]:
    ci = _coef_ci95(beta_draws)
    if ci is None:
        return float("nan"), float("nan")
    low = np.asarray(ci[0], dtype=float)
    high = np.asarray(ci[1], dtype=float)
    width = float(np.mean(high - low))
    covered = (np.asarray(beta_true, dtype=float) >= low) & (np.asarray(beta_true, dtype=float) <= high)
    return width, float(np.mean(covered))


def _mse_partition(beta_true: np.ndarray, beta_hat: np.ndarray) -> Dict[str, float]:
    truth = np.asarray(beta_true, dtype=float).reshape(-1)
    hat = np.asarray(beta_hat, dtype=float).reshape(-1)
    active = np.abs(truth) > 1e-10
    inactive = ~active
    return {
        "MSE_overall": float(np.mean((hat - truth) ** 2)),
        "MSE_signal": float(np.mean((hat[active] - truth[active]) ** 2)) if np.any(active) else float("nan"),
        "MSE_null": float(np.mean((hat[inactive] - truth[inactive]) ** 2)) if np.any(inactive) else float("nan"),
    }


def _convergence_gate(model: Any) -> Dict[str, Any]:
    posterior = _posterior_draws(model)
    tracked = {k: v for k, v in posterior.items() if k in {"beta", "tau", "a", "kappa", "lambda", "sigma", "c2"}}
    summary = summarize_convergence(tracked) if tracked else {}
    diagnostics = getattr(model, "sampler_diagnostics_", {}) if hasattr(model, "sampler_diagnostics_") else {}
    hmc = diagnostics.get("hmc", {}) if isinstance(diagnostics, Mapping) else {}
    max_rhat = max(
        [float(v.get("rhat_max", float("nan"))) for v in summary.values() if isinstance(v, Mapping) and "rhat_max" in v]
        or [float("nan")]
    )
    min_ess = min(
        [float(v.get("ess_min", float("nan"))) for v in summary.values() if isinstance(v, Mapping) and "ess_min" in v]
        or [float("nan")]
    )
    divergences = int(hmc.get("divergences", -1)) if isinstance(hmc, Mapping) else -1
    draws = 0
    beta_draws = posterior.get("beta")
    if beta_draws is not None:
        arr = np.asarray(beta_draws)
        draws = int(arr.shape[0] * arr.shape[1]) if arr.ndim >= 3 else int(arr.shape[0])
    div_rate = float(divergences / max(draws, 1)) if divergences >= 0 else float("nan")
    passed = bool(
        np.isfinite(max_rhat)
        and max_rhat < 1.01
        and np.isfinite(min_ess)
        and min_ess > 400.0
        and (not np.isfinite(div_rate) or div_rate < 0.001)
    )
    return {
        "passed": passed,
        "max_rhat": max_rhat,
        "min_bulk_ess": min_ess,
        "divergence_rate": div_rate,
        "summary": summary,
        "sampler_hmc": hmc,
    }


def _fit_method(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    seed: int,
    quick: bool,
    alpha_kappa: float = 0.5,
    beta_kappa: float = 1.0,
    eta: float = 0.5,
    p0: Optional[float] = None,
    tau0_override: Optional[float] = None,
    auto_calibrate_tau: bool = True,
    use_group_scale: bool = True,
    use_local_scale: bool = True,
    shared_kappa: bool = False,
) -> Any:
    name = str(method).strip().lower()
    profile = RevisionQuickProfile() if quick else RevisionQuickProfile(
        theory_reps=500,
        benchmark_reps=400,
        heterogeneity_reps=400,
        inferential_reps=500,
        logistic_reps=200,
        ablation_reps=400,
        hyper_reps=400,
        nuts_warmup=800,
        nuts_samples=800,
        nuts_chains=4,
        rhs_warmup=800,
        rhs_samples=800,
        rhs_chains=4,
    )
    likelihood = "logistic" if str(task).lower() == "logistic" else "gaussian"

    if name in {"grrhs_group_horseshoe", "grrhs_half_half"}:
        return _fit_method(
            "grrhs",
            X,
            y,
            groups,
            task=task,
            seed=seed,
            quick=quick,
            alpha_kappa=0.5,
            beta_kappa=0.5,
            eta=eta,
            p0=p0,
            tau0_override=tau0_override,
            auto_calibrate_tau=auto_calibrate_tau,
            use_group_scale=use_group_scale,
            use_local_scale=use_local_scale,
            shared_kappa=shared_kappa,
        )

    if name in {"grrhs_recommended", "grrhs_default"}:
        return _fit_method(
            "grrhs",
            X,
            y,
            groups,
            task=task,
            seed=seed,
            quick=quick,
            alpha_kappa=0.5,
            beta_kappa=1.0,
            eta=eta,
            p0=p0,
            tau0_override=tau0_override,
            auto_calibrate_tau=auto_calibrate_tau,
            use_group_scale=use_group_scale,
            use_local_scale=use_local_scale,
            shared_kappa=shared_kappa,
        )

    if name == "grrhs":
        model = GRRHS_NUTS(
            alpha_kappa=alpha_kappa,
            beta_kappa=beta_kappa,
            eta=eta,
            p0=p0,
            tau0=tau0_override,
            auto_calibrate_tau=bool(auto_calibrate_tau),
            num_warmup=profile.nuts_warmup,
            num_samples=profile.nuts_samples,
            num_chains=profile.nuts_chains,
            target_accept_prob=0.95,
            max_tree_depth=12,
            dense_mass=True,
            chain_method="sequential",
            progress_bar=False,
            seed=int(seed),
            likelihood=likelihood,
            use_group_scale=use_group_scale,
            use_local_scale=use_local_scale,
            shared_kappa=shared_kappa,
        )
        return model.fit(X, y, groups=[list(map(int, g)) for g in groups])

    if name == "grrhs_flat":
        return _fit_method(
            "grrhs",
            X,
            y,
            groups,
            task=task,
            seed=seed,
            quick=quick,
            alpha_kappa=1.0,
            beta_kappa=1.0,
            eta=eta,
            p0=p0,
            tau0_override=tau0_override,
            auto_calibrate_tau=auto_calibrate_tau,
            use_group_scale=use_group_scale,
            use_local_scale=use_local_scale,
            shared_kappa=shared_kappa,
        )

    if name == "rhs":
        model = RegularizedHorseshoeRegression(
            scale_global=float(
                GRRHS_NUTS.calibrate_tau0(
                    p0=max(p0 or 2.0, 1.0),
                    D=X.shape[1],
                    n=X.shape[0],
                    sigma_ref=2.0 if likelihood == "logistic" else 1.0,
                )
            ),
            likelihood=likelihood,
            backend="numpyro",
            num_warmup=profile.rhs_warmup,
            num_samples=profile.rhs_samples,
            num_chains=profile.rhs_chains,
            target_accept_prob=0.95,
            max_tree_depth=12,
            chain_method="sequential",
            progress_bar=False,
            seed=int(seed),
        )
        return model.fit(X, y)

    if name == "hs":
        model = HorseshoeRegression(
            scale_global=float(
                GRRHS_NUTS.calibrate_tau0(
                    p0=max(p0 or 2.0, 1.0),
                    D=X.shape[1],
                    n=X.shape[0],
                    sigma_ref=2.0 if likelihood == "logistic" else 1.0,
                )
            ),
            likelihood=likelihood,
            backend="numpyro",
            num_warmup=profile.rhs_warmup,
            num_samples=profile.rhs_samples,
            num_chains=profile.rhs_chains,
            target_accept_prob=0.95,
            max_tree_depth=12,
            chain_method="sequential",
            progress_bar=False,
            seed=int(seed),
        )
        return model.fit(X, y)

    if name == "gigg_mmle":
        if likelihood != "gaussian":
            raise NotImplementedError("GIGG-MMLE baseline is only available for gaussian outcomes in this repository.")
        if GIGGRegression is None:
            raise ImportError("GIGGRegression is not available in this environment.")
        model = GIGGRegression(
            method="mmle",
            n_burn_in=profile.rhs_warmup,
            n_samples=profile.rhs_samples,
            n_thin=1,
            seed=int(seed),
            num_chains=profile.rhs_chains,
            b_init=0.5,
            mmle_enabled=True,
            fit_intercept=True,
            store_lambda=True,
        )
        return model.fit(X, y, groups=[list(map(int, g)) for g in groups])

    if name == "gigg_fixed":
        if likelihood != "gaussian":
            raise NotImplementedError("GIGG fixed baseline is only available for gaussian outcomes in this repository.")
        if GIGGRegressionCRAN is not None:
            model = GIGGRegressionCRAN(
                method="fixed",
                n_burn_in=profile.rhs_warmup,
                n_samples=profile.rhs_samples,
                n_thin=1,
                seed=int(seed),
                num_chains=profile.rhs_chains,
                fit_intercept=True,
                store_lambda=True,
                a=np.full(len(groups), 0.5, dtype=float),
                b=np.full(len(groups), 0.5, dtype=float),
            )
            return model.fit(X, y, groups=[list(map(int, g)) for g in groups])
        if GIGGRegression is None:
            raise ImportError("Neither GIGGRegressionCRAN nor GIGGRegression is available in this environment.")
        grp_idx = np.zeros(X.shape[1], dtype=int)
        for gid, members in enumerate(groups):
            grp_idx[np.asarray(members, dtype=int)] = gid
        model = GIGGRegression(
            method="fixed",
            n_burn_in=profile.rhs_warmup,
            n_samples=profile.rhs_samples,
            n_thin=1,
            seed=int(seed),
            num_chains=profile.rhs_chains,
            a_value=0.5,
            b_init=0.5,
            mmle_enabled=False,
            fit_intercept=True,
            store_lambda=True,
            force_a_1_over_n=False,
        )
        return model.fit(X, y, grp_idx=grp_idx)

    if name in {"group_horseshoe_plus", "grouped_horseshoe_plus", "ghs_plus", "hbghs"}:
        if likelihood != "gaussian":
            raise NotImplementedError("Grouped Horseshoe+ is implemented for gaussian outcomes only.")
        model = GroupedHorseshoePlus(
            fit_intercept=True,
            tau0=float(
                GRRHS_NUTS.calibrate_tau0(
                    p0=max(p0 or 2.0, 1.0),
                    D=X.shape[1],
                    n=X.shape[0],
                    sigma_ref=1.0,
                )
            ),
            group_scale_prior=1.0,
            local_scale_prior=1.0,
            iters=profile.rhs_warmup + profile.rhs_samples,
            burnin=profile.rhs_warmup,
            thin=1,
            num_chains=profile.rhs_chains,
            seed=int(seed),
            progress_bar=False,
        )
        return model.fit(X, y, groups=[list(map(int, g)) for g in groups])

    raise ValueError(f"unknown method '{method}'")


def _fit_many_methods(
    methods: Sequence[str],
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    base_seed: int,
    quick: bool,
    p0: Optional[float] = None,
    grrhs_options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    options = dict(grrhs_options or {})
    for idx, method in enumerate(methods):
        try:
            if str(method).lower().startswith("grrhs"):
                out[str(method)] = _fit_method(
                    method,
                    X,
                    y,
                    groups,
                    task=task,
                    seed=int(base_seed + idx),
                    quick=quick,
                    p0=p0,
                    tau0_override=float(options["tau0_override"]) if "tau0_override" in options else None,
                    auto_calibrate_tau=bool(options.get("auto_calibrate_tau", True)),
                    alpha_kappa=float(options.get("alpha_kappa", 0.5)),
                    beta_kappa=float(options.get("beta_kappa", 1.0)),
                    eta=float(options.get("eta", 0.5)),
                    use_group_scale=bool(options.get("use_group_scale", True)),
                    use_local_scale=bool(options.get("use_local_scale", True)),
                    shared_kappa=bool(options.get("shared_kappa", False)),
                )
            else:
                out[str(method)] = _fit_method(
                    method,
                    X,
                    y,
                    groups,
                    task=task,
                    seed=int(base_seed + idx),
                    quick=quick,
                    p0=p0,
                )
        except Exception as exc:
            out[str(method)] = exc
    return out


def _benchmark_setting(setting: str) -> Dict[str, Any]:
    sid = str(setting).upper()
    lookup = {
        "L0": {"group_sizes": [10] * 5, "rho_within": 0.0, "signal_type": "distributed", "rho_between": 0.0, "orthonormal_design": True},
        "S1": {"group_sizes": [10] * 5, "rho_within": 0.3, "signal_type": "concentrated"},
        "S2": {"group_sizes": [10] * 5, "rho_within": 0.3, "signal_type": "distributed"},
        "S3": {"group_sizes": [10] * 5, "rho_within": 0.8, "signal_type": "concentrated"},
        "S4": {"group_sizes": [10] * 5, "rho_within": 0.8, "signal_type": "distributed"},
        "S5": {"group_sizes": [5] * 10, "rho_within": 0.3, "signal_type": "concentrated"},
        "S6": {"group_sizes": [5] * 10, "rho_within": 0.3, "signal_type": "distributed"},
        "S7": {"group_sizes": [5] * 10, "rho_within": 0.8, "signal_type": "concentrated"},
        "S8": {"group_sizes": [5] * 10, "rho_within": 0.8, "signal_type": "distributed"},
        "S9": {"group_sizes": [30, 10, 5, 3, 2], "rho_within": 0.8, "signal_type": "concentrated", "active_groups": [0]},
        "S10": {"group_sizes": [30, 10, 5, 3, 2], "rho_within": 0.8, "signal_type": "distributed", "active_groups": [0]},
        "S11": {"group_sizes": [30, 10, 5, 3, 2], "rho_within": 0.8, "signal_type": "concentrated", "active_groups": [3, 4]},
        "S12": {"group_sizes": [30, 10, 5, 3, 2], "rho_within": 0.8, "signal_type": "distributed", "active_groups": [3, 4]},
    }
    if sid not in lookup:
        raise ValueError(f"unknown benchmark setting '{setting}'")
    out = dict(lookup[sid])
    out.setdefault("active_groups", [0, 1])
    out["id"] = sid
    return out


def _make_benchmark_dataset(setting: str, *, seed: int, n: int = 500, snr: float = 0.7, rho_between: float = 0.1) -> Dict[str, Any]:
    cfg = _benchmark_setting(setting)
    rng = np.random.default_rng(seed)
    groups = _canonical_groups(cfg["group_sizes"])
    use_orth = bool(cfg.get("orthonormal_design", False))
    rho_between_eff = float(cfg.get("rho_between", rho_between))
    if use_orth:
        X = _sample_orthonormal_design(n=n, groups=groups, rng=rng)
    else:
        X = _sample_design(n=n, groups=groups, rho_within=cfg["rho_within"], rho_between=rho_between_eff, rng=rng)
    beta_shape = np.zeros(X.shape[1], dtype=float)
    for gid in cfg["active_groups"]:
        if cfg["signal_type"] == "concentrated":
            beta_shape += _concentrated_group_shape(groups, gid, target_norm_sq=1.0)
        else:
            beta_shape += _distributed_group_shape(groups, gid, target_norm_sq=1.0)
    beta, sigma = _solve_signal_scale(X, beta_shape, snr=snr)
    y = _sample_gaussian_response(X, beta, sigma, rng)
    return {"X": X, "y": y, "beta": beta, "sigma": sigma, "groups": groups, "setting": cfg, "task": "gaussian"}


def _make_heterogeneity_dataset(seed: int, *, group_width: int = 10) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = _canonical_groups([int(group_width)] * 6)
    n = max(300, int(group_width) * 6)
    X = _sample_design(n=n, groups=groups, rho_within=0.7, rho_between=0.05, rng=rng)
    mu = [0.0, 0.0, 2.0, 8.0, 25.0, 80.0]
    beta = np.zeros(X.shape[1], dtype=float)
    for gid, mu_g in enumerate(mu):
        if mu_g > 0.0:
            beta += _distributed_group_shape(groups, gid, target_norm_sq=2.0 * mu_g)
    y = _sample_gaussian_response(X, beta, 1.0, rng)
    return {
        "X": X,
        "y": y,
        "beta": beta,
        "sigma": 1.0,
        "group_width": int(group_width),
        "groups": groups,
        "mu": np.asarray(mu, dtype=float),
        "task": "gaussian",
    }


def _make_inferential_dataset(seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = _canonical_groups([15] * 5)
    X = _sample_design(n=400, groups=groups, rho_within=0.6, rho_between=0.05, rng=rng)
    strengths = [0.0, 0.0, 2.0, 8.0, 25.0]
    beta = np.zeros(X.shape[1], dtype=float)
    for gid, mu_g in enumerate(strengths):
        if mu_g > 0.0:
            beta += _distributed_group_shape(groups, gid, target_norm_sq=2.0 * mu_g)
    y = _sample_gaussian_response(X, beta, 1.0, rng)
    return {
        "X": X,
        "y": y,
        "beta": beta,
        "groups": groups,
        "strengths": np.asarray(strengths, dtype=float),
        "signal_labels": np.asarray([0, 0, 1, 1, 1], dtype=int),
        "task": "gaussian",
    }


def _make_grouped_logistic_dataset(seed: int, *, n: int = 180) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = _canonical_groups([5, 5, 5])
    X = _sample_design(n=int(n), groups=groups, rho_within=0.5, rho_between=0.05, rng=rng)
    beta = np.zeros(X.shape[1], dtype=float)
    beta[0] = 3.0
    beta[1] = 3.0
    beta[10] = 0.5
    beta[11] = 0.5
    y = _sample_logistic_response(X, beta, rng)
    return {"X": X, "y": y, "beta": beta, "groups": groups, "task": "logistic"}


def _phase_signal_dataset(p_g: int, xi_inf: float, *, seed: int, rho: float) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = _canonical_groups([p_g])
    X = _sample_design(n=5 * p_g, groups=groups, rho_within=rho, rho_between=0.0, rng=rng)
    mu = float(xi_inf) * float(p_g)
    beta = _distributed_group_shape(groups, 0, target_norm_sq=2.0 * mu)
    y = _sample_gaussian_response(X, beta, 1.0, rng)
    return {"X": X, "y": y, "beta": beta, "groups": groups}


def _null_sequence_dataset(p_g: int, *, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {"y_group": rng.normal(size=p_g), "p_g": p_g}


def _adaptive_localization_dataset(p_g: int, *, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = _canonical_groups([p_g])
    X = _sample_design(n=5 * p_g, groups=groups, rho_within=0.0, rho_between=0.0, rng=rng)
    mu = 2.0 * (float(p_g) ** 0.75)
    beta = _distributed_group_shape(groups, 0, target_norm_sq=2.0 * mu)
    y = _sample_gaussian_response(X, beta, 1.0, rng)
    return {"X": X, "y": y, "beta": beta, "groups": groups, "mu": mu}


def _run_theory_section(outdir: Path, *, quick: bool, seed: int) -> Dict[str, Any]:
    profile = RevisionQuickProfile()
    reps = profile.theory_reps if quick else 500
    rng = np.random.default_rng(seed)
    null_grid = [10, 20, 50] if quick else [10, 20, 50, 100, 200, 500]
    adaptive_grid = [20, 50] if quick else [20, 50, 100, 200, 500]
    phase_xi = [0.01, 0.04, 0.05] if quick else [0.005, 0.01, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.1, 0.2, 0.5]
    phase_pg = [30, 60] if quick else [30, 60, 120]
    phase_tau = [0.3] if quick else [0.1, 0.3, 0.5]
    adaptive_alpha_kappa = 1.0
    adaptive_beta_kappa = 1.0
    adaptive_rho = 0.1
    x_lower, x_upper = _theorem3p30_ratio_bounds(adaptive_rho, adaptive_alpha_kappa, adaptive_beta_kappa)
    adaptive_band = {"x_lower": x_lower, "x_upper": x_upper}

    null_rows: list[dict[str, float]] = []
    for p_g in null_grid:
        posterior_means = []
        beta_post_mean_norm_sq = []
        tail_prob_by_M: dict[int, list[float]] = {2: [], 3: [], 5: []}
        for _ in range(reps):
            ds = _null_sequence_dataset(p_g, seed=int(rng.integers(1_000_000)))
            obj = profile_kappa_log_posterior_grid(ds["y_group"], tau=0.1, alpha_kappa=1.0, beta_kappa=1.0)
            kappa_mean = float(np.trapezoid(obj["kappa"] * obj["density"], obj["kappa"]))
            posterior_means.append(kappa_mean)
            for M in (2, 3, 5):
                thresh = float(M) / math.sqrt(float(p_g))
                mask = obj["kappa"] > thresh
                tail_prob = float(np.trapezoid(obj["density"][mask], obj["kappa"][mask])) if np.any(mask) else 0.0
                tail_prob_by_M[M].append(tail_prob)
            beta_post_mean_norm_sq.append(
                profile_beta_posterior_mean_norm_sq(
                    ds["y_group"],
                    tau=0.1,
                    sigma=1.0,
                    alpha_kappa=1.0,
                    beta_kappa=1.0,
                )
            )
        mean_kappa = float(np.mean(posterior_means))
        null_rows.append(
            {
                "p_g": float(p_g),
                "median_posterior_mean": float(np.median(posterior_means)),
                "mean_posterior_mean": mean_kappa,
                "sqrt_pg_times_mean_kappa": float(math.sqrt(float(p_g)) * mean_kappa),
                "mean_tail_prob_M2": float(np.mean(tail_prob_by_M[2])),
                "mean_tail_prob_M3": float(np.mean(tail_prob_by_M[3])),
                "mean_tail_prob_M5": float(np.mean(tail_prob_by_M[5])),
                "mean_beta_post_mean_norm_sq": float(np.mean(beta_post_mean_norm_sq)),
                "mean_beta_post_mean_norm_sq_over_sigma2": float(np.mean(beta_post_mean_norm_sq)),
            }
        )
    null_df = pd.DataFrame(null_rows)
    null_df["mean_tail_prob"] = null_df["mean_tail_prob_M2"]
    slope = float(np.polyfit(np.log(null_df["p_g"]), np.log(np.maximum(null_df["median_posterior_mean"], 1e-12)), 1)[0])
    null_df["loglog_slope_global"] = slope
    null_df.to_csv(outdir / "theory_null_contraction.csv", index=False)

    adaptive_rows: list[dict[str, float]] = []
    adaptive_appendix_rows: list[dict[str, float]] = []
    for signal_type in ("distributed", "sparse"):
        for p_g in adaptive_grid:
            ratios = []
            band_mass = []
            for _ in range(reps):
                mu = 2.0 * (float(p_g) ** 0.75)
                ds = _sample_profile_signal_group(p_g, mu, seed=int(rng.integers(1_000_000)), sigma=1.0, signal_type=signal_type)
                obj = profile_kappa_log_posterior_grid(
                    ds["y_group"],
                    tau=0.1,
                    sigma=1.0,
                    alpha_kappa=adaptive_alpha_kappa,
                    beta_kappa=adaptive_beta_kappa,
                )
                kappa_mean = float(np.trapezoid(obj["kappa"] * obj["density"], obj["kappa"]))
                target = float(ds["mu"]) / float(p_g)
                ratios.append(float(kappa_mean / max(target, _EPS)))
                lo = adaptive_band["x_lower"] * target
                hi = adaptive_band["x_upper"] * target
                mask = (obj["kappa"] >= lo) & (obj["kappa"] <= hi)
                band_mass.append(float(np.trapezoid(obj["density"][mask], obj["kappa"][mask])) if np.any(mask) else 0.0)
            adaptive_rows.append(
                {
                    "p_g": float(p_g),
                    "signal_type": signal_type,
                    "ratio_median": float(np.nanmedian(ratios)),
                    "ratio_iqr_low": float(np.nanquantile(ratios, 0.25)),
                    "ratio_iqr_high": float(np.nanquantile(ratios, 0.75)),
                }
            )
            adaptive_appendix_rows.append(
                {
                    "p_g": float(p_g),
                    "signal_type": signal_type,
                    "appendix_band_x_lower": float(adaptive_band["x_lower"]),
                    "appendix_band_x_upper": float(adaptive_band["x_upper"]),
                    "appendix_band_mass_mean": float(np.nanmean(band_mass)),
                }
            )
    pd.DataFrame(adaptive_rows).to_csv(outdir / "theory_adaptive_localization.csv", index=False)
    pd.DataFrame(adaptive_appendix_rows).to_csv(outdir / "theory_adaptive_localization_appendix_band.csv", index=False)

    u0 = 0.5
    phase_rows: list[dict[str, float]] = []
    bridge_rows: list[dict[str, float | str]] = []
    for tau in phase_tau:
        rho = float(tau) / 1.0
        xi_crit_tau = (u0 * (rho**2)) / (2.0 * (u0 + (1.0 - u0) * (rho**2)))
        for xi_inf in phase_xi:
            for p_g in phase_pg:
                probs = []
                for _ in range(max(2, reps // 2)):
                    mu = float(xi_inf) * float(p_g)
                    ds = _sample_profile_signal_group(p_g, mu, seed=int(rng.integers(1_000_000)), sigma=1.0, signal_type="distributed")
                    obj = profile_kappa_log_posterior_grid(ds["y_group"], tau=float(tau), sigma=1.0, alpha_kappa=1.0, beta_kappa=1.0)
                    mask = obj["kappa"] > u0
                    probs.append(float(np.trapezoid(obj["density"][mask], obj["kappa"][mask])) if np.any(mask) else 0.0)
                phase_rows.append(
                    {
                        "tau": float(tau),
                        "rho": float(rho),
                        "xi_inf": float(xi_inf),
                        "p_g": float(p_g),
                        "prob_kappa_gt_u0": float(np.nanmean(probs)),
                        "xi_crit": float(xi_crit_tau),
                    }
                )
        if not quick:
            for xi_inf in [0.04, 0.06]:
                for p_g in [phase_pg[-1]]:
                    ds_profile = _sample_profile_signal_group(
                        p_g,
                        float(xi_inf) * float(p_g),
                        seed=int(rng.integers(1_000_000)),
                        sigma=1.0,
                        signal_type="distributed",
                    )
                    obj = profile_kappa_log_posterior_grid(ds_profile["y_group"], tau=float(tau), sigma=1.0, alpha_kappa=1.0, beta_kappa=1.0)
                    mask = obj["kappa"] > u0
                    p_profile = float(np.trapezoid(obj["density"][mask], obj["kappa"][mask])) if np.any(mask) else 0.0
                    ds_full = _phase_signal_dataset(p_g=p_g, xi_inf=float(xi_inf), seed=int(rng.integers(1_000_000)), rho=min(float(rho), 0.95))
                    fit = _fit_method("grrhs", ds_full["X"], ds_full["y"], ds_full["groups"], task="gaussian", seed=int(rng.integers(1_000_000)), quick=True, p0=1)
                    post = _posterior_draws(fit)
                    kap_draws = _flatten_draws(post.get("kappa"), scalar=False)
                    p_full = float(np.mean(kap_draws[:, 0] > u0)) if kap_draws is not None else float("nan")
                    bridge_rows.append(
                        {
                            "tau": float(tau),
                            "xi_inf": float(xi_inf),
                            "p_g": float(p_g),
                            "profile_prob_kappa_gt_u0": float(p_profile),
                            "full_nuts_prob_kappa_gt_u0": float(p_full),
                            "abs_gap": float(abs(p_profile - p_full)) if np.isfinite(p_full) else float("nan"),
                        }
                    )
    pd.DataFrame(phase_rows).to_csv(outdir / "theory_strong_signal_phase.csv", index=False)
    pd.DataFrame(bridge_rows).to_csv(outdir / "theory_phase_profile_vs_full_bridge.csv", index=False)
    manifest = {
        "validation_layer": "profile_specialization_normal_means",
        "null_uses_n": False,
        "adaptive_appendix_band": adaptive_band,
        "adaptive_ratio_band_source": "lemma_3_28_parameterized_proxy",
        "adaptive_ratio_band_inputs": {"rho": adaptive_rho, "alpha_kappa": adaptive_alpha_kappa, "beta_kappa": adaptive_beta_kappa},
        "strong_signal_u0": u0,
        "strong_signal_tau_grid": phase_tau,
    }
    (outdir / "theory_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "null_contraction": str(outdir / "theory_null_contraction.csv"),
        "adaptive_localization": str(outdir / "theory_adaptive_localization.csv"),
        "adaptive_localization_appendix_band": str(outdir / "theory_adaptive_localization_appendix_band.csv"),
        "strong_signal_phase": str(outdir / "theory_strong_signal_phase.csv"),
        "phase_profile_vs_full_bridge": str(outdir / "theory_phase_profile_vs_full_bridge.csv"),
        "manifest": str(outdir / "theory_manifest.json"),
        "loglog_slope": slope,
        "xi_crit": float((u0 * (0.3**2)) / (2.0 * (u0 + (1.0 - u0) * (0.3**2)))),
    }


def _run_benchmark_section(outdir: Path, *, quick: bool, seed: int, methods: Sequence[str]) -> Dict[str, Any]:
    profile = RevisionQuickProfile()
    reps = profile.benchmark_reps if quick else 400
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for setting in ["L0"] + [f"S{k}" for k in range(1, 13)]:
        for rep in range(reps):
            ds = _make_benchmark_dataset(setting, seed=int(rng.integers(1_000_000)))
            active_groups = np.asarray(ds["setting"]["active_groups"], dtype=int)
            fits = _fit_many_methods(methods, ds["X"], ds["y"], ds["groups"], task="gaussian", base_seed=int(rng.integers(1_000_000)), quick=quick, p0=max(len(active_groups), 1))
            for method, fit in fits.items():
                if isinstance(fit, Exception):
                    rows.append({"setting": setting, "rep": rep, "method": method, "status": "error", "error": type(fit).__name__})
                    continue
                posterior = _posterior_draws(fit)
                beta_hat = _posterior_mean(posterior.get("beta"))
                width, coverage = _ci_length_and_coverage(ds["beta"], posterior.get("beta"))
                conv = _convergence_gate(fit)
                mse = _mse_partition(ds["beta"], beta_hat)
                sigma2 = float(ds["sigma"]) ** 2
                scores = _group_scores_from_model(fit, ds["groups"])
                t31_holds = float("nan")
                t31_margin = float("nan")
                if "kappa_mean" in scores:
                    lhs_all = []
                    rhs_all = []
                    for gid, group in enumerate(ds["groups"]):
                        idx = np.asarray(group, dtype=int)
                        lhs = float(np.linalg.norm(beta_hat[idx], ord=2))
                        rhs = float(scores["kappa_mean"][gid] * np.linalg.norm(ds["X"][:, idx].T @ ds["y"], ord=2))
                        lhs_all.append(lhs)
                        rhs_all.append(rhs)
                    lhs_arr = np.asarray(lhs_all, dtype=float)
                    rhs_arr = np.asarray(rhs_all, dtype=float)
                    t31_holds = float(np.mean(lhs_arr <= rhs_arr + 1e-8))
                    t31_margin = float(np.min(rhs_arr - lhs_arr))
                rows.append(
                    {
                        "setting": setting,
                        "rep": rep,
                        "method": method,
                        "status": "ok",
                        "CI_length": width,
                        "Coverage": coverage,
                        **mse,
                        "MSE_overall_over_sigma2": float(mse["MSE_overall"] / max(sigma2, _EPS)),
                        "MSE_signal_over_sigma2": float(mse["MSE_signal"] / max(sigma2, _EPS)) if np.isfinite(mse["MSE_signal"]) else float("nan"),
                        "MSE_null_over_sigma2": float(mse["MSE_null"] / max(sigma2, _EPS)) if np.isfinite(mse["MSE_null"]) else float("nan"),
                        "theorem3p1_lift_holds_rate": t31_holds,
                        "theorem3p1_lift_min_margin": t31_margin,
                        "convergence_passed": bool(conv["passed"]),
                        "max_rhat": conv["max_rhat"],
                        "min_bulk_ess": conv["min_bulk_ess"],
                        "divergence_rate": conv["divergence_rate"],
                    }
                )
                if setting in {"S9", "S10", "S11", "S12"}:
                    kappa_mean = scores.get("kappa_mean")
                    for gid, group in enumerate(ds["groups"]):
                        idx = np.asarray(group, dtype=int)
                        group_rows.append({"setting": setting, "rep": rep, "method": method, "group_id": gid, "group_mse": float(np.sum((beta_hat[idx] - ds["beta"][idx]) ** 2)), "kappa_mean": float(kappa_mean[gid]) if kappa_mean is not None else float("nan"), "is_active_group": bool(gid in active_groups)})
    pd.DataFrame(rows).to_csv(outdir / "benchmark_main.csv", index=False)
    if group_rows:
        pd.DataFrame(group_rows).to_csv(outdir / "benchmark_group_level.csv", index=False)
    return {"main_table": str(outdir / "benchmark_main.csv"), "group_table": str(outdir / "benchmark_group_level.csv") if group_rows else None}


def _run_heterogeneity_section(outdir: Path, *, quick: bool, seed: int, methods: Sequence[str]) -> Dict[str, Any]:
    profile = RevisionQuickProfile()
    reps = profile.heterogeneity_reps if quick else 400
    rng = np.random.default_rng(seed)
    score_rows: list[dict[str, Any]] = []
    mse_rows: list[dict[str, Any]] = []
    mono_rows: list[dict[str, Any]] = []
    subset = list(methods)
    group_width_grid = [10, 50] if quick else [10, 50, 100]
    for rep in range(reps):
        for group_width in group_width_grid:
            ds = _make_heterogeneity_dataset(seed=int(rng.integers(1_000_000)), group_width=int(group_width))
            fits = _fit_many_methods(
                subset,
                ds["X"],
                ds["y"],
                ds["groups"],
                task="gaussian",
                base_seed=int(rng.integers(1_000_000)),
                quick=quick,
                p0=4,
            )
            for method, fit in fits.items():
                if isinstance(fit, Exception):
                    continue
                posterior = _posterior_draws(fit)
                beta_hat = _posterior_mean(posterior.get("beta"))
                mse = _mse_partition(ds["beta"], beta_hat)
                sigma2 = float(ds["sigma"]) ** 2
                mse_rows.append(
                    {
                        "rep": rep,
                        "method": method,
                        "group_width": int(group_width),
                        **mse,
                        "MSE_overall_over_sigma2": float(mse["MSE_overall"] / max(sigma2, _EPS)),
                        "MSE_signal_over_sigma2": float(mse["MSE_signal"] / max(sigma2, _EPS)) if np.isfinite(mse["MSE_signal"]) else float("nan"),
                        "MSE_null_over_sigma2": float(mse["MSE_null"] / max(sigma2, _EPS)) if np.isfinite(mse["MSE_null"]) else float("nan"),
                    }
                )
                scores = _group_scores_from_model(fit, ds["groups"])
                t31_holds = float("nan")
                if "kappa_mean" in scores:
                    lhs = []
                    rhs = []
                    for gid, group in enumerate(ds["groups"]):
                        idx = np.asarray(group, dtype=int)
                        lhs.append(float(np.linalg.norm(beta_hat[idx], ord=2)))
                        rhs.append(float(scores["kappa_mean"][gid] * np.linalg.norm(ds["X"][:, idx].T @ ds["y"], ord=2)))
                    lhs_arr = np.asarray(lhs, dtype=float)
                    rhs_arr = np.asarray(rhs, dtype=float)
                    t31_holds = float(np.mean(lhs_arr <= rhs_arr + 1e-8))
                mse_rows[-1]["theorem3p1_lift_holds_rate"] = t31_holds
                is_grrhs_variant = str(method).startswith("grrhs")
                for gid, mu_g in enumerate(ds["mu"]):
                    if "kappa_mean" in scores:
                        score_rows.append(
                            {
                                "rep": rep,
                                "method": method,
                                "group_width": int(group_width),
                                "group_id": gid,
                                "group_signal_mu": float(mu_g),
                                "score_name": "kappa_mean",
                                "score_value": float(scores["kappa_mean"][gid]),
                                "score_role": "primary_inferential_target" if is_grrhs_variant else "derived_reference",
                            }
                        )
                    if "gamma2_mean" in scores and gid < len(scores["gamma2_mean"]):
                        score_rows.append(
                            {
                                "rep": rep,
                                "method": method,
                                "group_width": int(group_width),
                                "group_id": gid,
                                "group_signal_mu": float(mu_g),
                                "score_name": "gamma2_mean",
                                "score_value": float(scores["gamma2_mean"][gid]),
                                "score_role": "auxiliary_reference" if method in {"gigg_mmle", "gigg_fixed"} else "derived_reference",
                            }
                        )
                    if is_grrhs_variant and "kappa_mean" in scores:
                        mono_rows.append(
                            {
                                "rep": rep,
                                "group_width": int(group_width),
                                "group_id": gid,
                                "group_signal_mu": float(mu_g),
                                "kappa_mean": float(scores["kappa_mean"][gid]),
                            }
                        )
    pd.DataFrame(score_rows).to_csv(outdir / "heterogeneity_scores.csv", index=False)
    pd.DataFrame(mse_rows).to_csv(outdir / "heterogeneity_mse.csv", index=False)
    pd.DataFrame(mono_rows).to_csv(outdir / "heterogeneity_monotone.csv", index=False)
    return {"score_table": str(outdir / "heterogeneity_scores.csv"), "mse_table": str(outdir / "heterogeneity_mse.csv"), "monotone_table": str(outdir / "heterogeneity_monotone.csv")}


def _run_inferential_section(outdir: Path, *, quick: bool, seed: int, methods: Sequence[str]) -> Dict[str, Any]:
    profile = RevisionQuickProfile()
    reps = profile.inferential_reps if quick else 500
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    for rep in range(reps):
        ds = _make_inferential_dataset(seed=int(rng.integers(1_000_000)))
        fits = _fit_many_methods(list(methods), ds["X"], ds["y"], ds["groups"], task="gaussian", base_seed=int(rng.integers(1_000_000)), quick=quick, p0=3)
        labels = ds["signal_labels"]
        for method, fit in fits.items():
            if isinstance(fit, Exception):
                continue
            scores = _group_scores_from_model(fit, ds["groups"])
            if "kappa_mean" in scores:
                rows.append({"rep": rep, "method": method, "score": "kappa_mean", "auroc": float(roc_auc_score(labels, scores["kappa_mean"]))})
                posterior = _posterior_draws(fit)
                beta_mean = _posterior_mean(posterior.get("beta"))
                for gid, group in enumerate(ds["groups"]):
                    lhs = float(np.linalg.norm(beta_mean[np.asarray(group, dtype=int)], ord=2))
                    rhs = float(scores["kappa_mean"][gid] * np.linalg.norm(ds["y"]))
                    bridge_rows.append({"rep": rep, "group_id": gid, "lhs_beta_norm": lhs, "rhs_bound": rhs, "bridge_holds": bool(lhs <= rhs + 1e-8)})
            if "beta_norm_sq" in scores:
                rows.append({"rep": rep, "method": method, "score": "beta_norm_sq", "auroc": float(roc_auc_score(labels, scores["beta_norm_sq"]))})
            if "gamma2_mean" in scores:
                rows.append({"rep": rep, "method": method, "score": "gamma2_mean", "auroc": float(roc_auc_score(labels, scores["gamma2_mean"]))})
    pd.DataFrame(rows).to_csv(outdir / "inferential_auroc.csv", index=False)
    pd.DataFrame(bridge_rows).to_csv(outdir / "inferential_bridge.csv", index=False)
    auroc_df = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    if not auroc_df.empty:
        for (method, score), sub in auroc_df.groupby(["method", "score"], dropna=False):
            vals = sub["auroc"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            lo = float(np.nan)
            hi = float(np.nan)
            if vals.size >= 2:
                brng = np.random.default_rng(seed + 991 + len(summary_rows))
                boots = []
                for _ in range(200):
                    draw = brng.choice(vals, size=vals.size, replace=True)
                    boots.append(float(np.mean(draw)))
                lo = float(np.quantile(boots, 0.025))
                hi = float(np.quantile(boots, 0.975))
            summary_rows.append(
                {
                    "method": str(method),
                    "score": str(score),
                    "auroc_mean": float(np.mean(vals)),
                    "auroc_std": float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0,
                    "auroc_bootstrap_ci_low": lo,
                    "auroc_bootstrap_ci_high": hi,
                    "n_reps": int(vals.size),
                }
            )
    summary_path = outdir / "inferential_auroc_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return {
        "auroc_table": str(outdir / "inferential_auroc.csv"),
        "auroc_summary_table": str(summary_path),
        "bridge_table": str(outdir / "inferential_bridge.csv"),
    }


def _run_logistic_section(outdir: Path, *, quick: bool, seed: int, methods: Sequence[str]) -> Dict[str, Any]:
    profile = RevisionQuickProfile()
    reps = profile.logistic_reps if quick else 200
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    chosen = [m for m in methods if m in {"grrhs", "rhs", "hs"}]
    n_obs = 150 if quick else 180
    for rep in range(reps):
        ds = _make_grouped_logistic_dataset(seed=int(rng.integers(1_000_000)), n=n_obs)
        fits = _fit_many_methods(chosen, ds["X"], ds["y"], ds["groups"], task="logistic", base_seed=int(rng.integers(1_000_000)), quick=quick, p0=2)
        for method, fit in fits.items():
            if isinstance(fit, Exception):
                rows.append({"rep": rep, "method": method, "status": "error", "error": type(fit).__name__})
                continue
            posterior = _posterior_draws(fit)
            beta_hat = _posterior_mean(posterior.get("beta"))
            conv = _convergence_gate(fit)
            prob_g1 = float(np.mean(_flatten_draws(posterior.get("kappa"), scalar=False)[:, 0] > 0.5)) if posterior.get("kappa") is not None else float("nan")
            prob_g2 = float(np.mean(_flatten_draws(posterior.get("kappa"), scalar=False)[:, 1] > 0.5)) if posterior.get("kappa") is not None else float("nan")
            rows.append({"rep": rep, "method": method, "status": "ok", "separator_bias": float(np.mean(beta_hat[:2] - ds["beta"][:2])) if beta_hat is not None else float("nan"), "separator_var": float(np.var(beta_hat[:2])) if beta_hat is not None else float("nan"), "weak_bias": float(np.mean(beta_hat[10:12] - ds["beta"][10:12])) if beta_hat is not None else float("nan"), "divergence_rate": conv["divergence_rate"], "max_rhat": conv["max_rhat"], "prob_kappa_g1_gt_05": prob_g1, "prob_kappa_g2_gt_05": prob_g2})
    pd.DataFrame(rows).to_csv(outdir / "logistic_weak_identification.csv", index=False)
    meta = {
        "coverage_scope": "exploratory_out_of_gaussian_theory",
        "note": "Logistic regression experiment is empirical extension and not a direct theorem-validation section.",
        "n_observations": int(n_obs),
    }
    (outdir / "logistic_manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"table": str(outdir / "logistic_weak_identification.csv"), "manifest": str(outdir / "logistic_manifest.json")}


def _run_ablation_section(outdir: Path, *, quick: bool, seed: int) -> Dict[str, Any]:
    profile = RevisionQuickProfile()
    reps = profile.ablation_reps if quick else 400
    rng = np.random.default_rng(seed)
    variants = {
        "full_grrhs": {"method": "grrhs", "grrhs_options": {"alpha_kappa": 0.5, "beta_kappa": 1.0, "eta": 0.5, "use_group_scale": True, "shared_kappa": False}},
        "no_a_layer": {"method": "grrhs", "grrhs_options": {"alpha_kappa": 0.5, "beta_kappa": 1.0, "eta": 0.5, "use_group_scale": False, "shared_kappa": False}},
        "no_local_scales": {"method": "grrhs", "grrhs_options": {"alpha_kappa": 0.5, "beta_kappa": 1.0, "eta": 0.5, "use_group_scale": True, "use_local_scale": False, "shared_kappa": False}},
        "shared_kappa": {"method": "grrhs", "grrhs_options": {"alpha_kappa": 0.5, "beta_kappa": 1.0, "eta": 0.5, "use_group_scale": True, "shared_kappa": True}},
        "rhs": {"method": "rhs", "grrhs_options": {}},
    }
    rows: list[dict[str, Any]] = []
    for rep in range(reps):
        ds = _make_heterogeneity_dataset(seed=int(rng.integers(1_000_000)))
        for label, spec in variants.items():
            fit = _fit_many_methods([spec["method"]], ds["X"], ds["y"], ds["groups"], task="gaussian", base_seed=int(rng.integers(1_000_000)), quick=quick, p0=4, grrhs_options=spec["grrhs_options"])[spec["method"]]
            if isinstance(fit, Exception):
                rows.append({"rep": rep, "variant": label, "status": "error", "error": type(fit).__name__})
                continue
            beta_hat = _posterior_mean(_posterior_draws(fit).get("beta"))
            rows.append({"rep": rep, "variant": label, "status": "ok", **_mse_partition(ds["beta"], beta_hat)})

        one_coord_groups = _canonical_groups([1] * 12)
        X1 = _sample_design(n=120 if quick else 250, groups=one_coord_groups, rho_within=0.0, rho_between=0.05, rng=rng)
        beta_shape = np.zeros(X1.shape[1], dtype=float)
        beta_shape[[0, 3, 8]] = [1.2, -1.0, 0.8]
        beta1, sigma1 = _solve_signal_scale(X1, beta_shape, snr=0.7)
        y1 = _sample_gaussian_response(X1, beta1, sigma1, rng)
        fit_grrhs = _fit_method("grrhs", X1, y1, one_coord_groups, task="gaussian", seed=int(rng.integers(1_000_000)), quick=quick, p0=3)
        fit_rhs = _fit_method("rhs", X1, y1, one_coord_groups, task="gaussian", seed=int(rng.integers(1_000_000)), quick=quick, p0=3)
        beta_grrhs = _posterior_mean(_posterior_draws(fit_grrhs).get("beta"))
        beta_rhs = _posterior_mean(_posterior_draws(fit_rhs).get("beta"))
        rows.append({
            "rep": rep,
            "variant": "one_coordinate_reduction",
            "status": "ok",
            "coef_l2_gap_to_rhs": float(np.linalg.norm(beta_grrhs - beta_rhs)),
            **_mse_partition(beta1, beta_grrhs),
        })
    pd.DataFrame(rows).to_csv(outdir / "ablation.csv", index=False)
    ablation_df = pd.DataFrame(rows)
    ok_df = ablation_df.loc[ablation_df["status"] == "ok"].copy()
    contrast_rows: list[dict[str, Any]] = []
    if not ok_df.empty:
        variant_means = ok_df.groupby("variant", dropna=False)[["MSE_null", "MSE_signal", "MSE_overall"]].mean(numeric_only=True)
        if "full_grrhs" in variant_means.index and "shared_kappa" in variant_means.index:
            contrast_rows.append({
                "contrast": "full_grrhs_vs_shared_kappa",
                "MSE_null_gap": float(variant_means.loc["shared_kappa", "MSE_null"] - variant_means.loc["full_grrhs", "MSE_null"]),
                "MSE_signal_gap": float(variant_means.loc["shared_kappa", "MSE_signal"] - variant_means.loc["full_grrhs", "MSE_signal"]),
                "MSE_overall_gap": float(variant_means.loc["shared_kappa", "MSE_overall"] - variant_means.loc["full_grrhs", "MSE_overall"]),
            })
        if "full_grrhs" in variant_means.index and "no_a_layer" in variant_means.index:
            contrast_rows.append({
                "contrast": "full_grrhs_vs_no_a_layer",
                "MSE_null_gap": float(variant_means.loc["no_a_layer", "MSE_null"] - variant_means.loc["full_grrhs", "MSE_null"]),
                "MSE_signal_gap": float(variant_means.loc["no_a_layer", "MSE_signal"] - variant_means.loc["full_grrhs", "MSE_signal"]),
                "MSE_overall_gap": float(variant_means.loc["no_a_layer", "MSE_overall"] - variant_means.loc["full_grrhs", "MSE_overall"]),
            })
        if "full_grrhs" in variant_means.index and "no_local_scales" in variant_means.index:
            contrast_rows.append({
                "contrast": "full_grrhs_vs_no_local_scales",
                "MSE_null_gap": float(variant_means.loc["no_local_scales", "MSE_null"] - variant_means.loc["full_grrhs", "MSE_null"]),
                "MSE_signal_gap": float(variant_means.loc["no_local_scales", "MSE_signal"] - variant_means.loc["full_grrhs", "MSE_signal"]),
                "MSE_overall_gap": float(variant_means.loc["no_local_scales", "MSE_overall"] - variant_means.loc["full_grrhs", "MSE_overall"]),
            })
    contrast_path = outdir / "ablation_priority_contrasts.csv"
    pd.DataFrame(contrast_rows).to_csv(contrast_path, index=False)
    return {"table": str(outdir / "ablation.csv"), "priority_contrasts": str(contrast_path)}


def _tau_prior_meff_draws(*, p: int, p0: int, n: int, rng: np.random.Generator, num_draws: int) -> pd.DataFrame:
    tau0 = GRRHS_NUTS.calibrate_tau0(p0=p0, D=p, n=n, sigma_ref=1.0)
    rows: list[dict[str, float]] = []
    labels = {
        "fixed_tau0": np.full(num_draws, tau0, dtype=float),
        "recommended_halfcauchy": np.abs(rng.standard_cauchy(num_draws)) * tau0,
        "wide_halfcauchy": np.abs(rng.standard_cauchy(num_draws)),
    }
    for label, tau in labels.items():
        tau_abs = np.clip(np.asarray(tau, dtype=float), 1e-8, 20.0)
        m_eff = (float(p) * tau_abs) / (1.0 + tau_abs)
        for value in m_eff:
            rows.append({"prior_name": label, "m_eff": float(value)})
    return pd.DataFrame(rows)


def _sample_profile_marginal_beta_prior(
    *,
    alpha_kappa: float,
    beta_kappa: float,
    tau: float,
    sigma: float,
    num_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    kappa = rng.beta(float(alpha_kappa), float(beta_kappa), size=int(num_draws))
    var = profile_prior_variance(kappa, tau=float(tau), sigma=float(sigma))
    return rng.normal(loc=0.0, scale=np.sqrt(np.maximum(var, _EPS)), size=int(num_draws))


def _estimate_tail_density_exponent(abs_samples: np.ndarray, *, q_low: float = 0.95, q_high: float = 0.995) -> float:
    vals = np.sort(np.asarray(abs_samples, dtype=float))
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size < 50:
        return float("nan")
    lo = float(np.quantile(vals, q_low))
    hi = float(np.quantile(vals, q_high))
    tail = vals[(vals >= lo) & (vals <= hi)]
    if tail.size < 20:
        return float("nan")
    n = float(vals.size)
    surv = np.arange(tail.size, 0, -1, dtype=float) / n
    x = np.log(np.maximum(tail, _EPS))
    y = np.log(np.maximum(surv, _EPS))
    slope = float(np.polyfit(x, y, 1)[0])
    survival_exp = -slope
    return float(survival_exp + 1.0)


def _run_hyperparameter_section(outdir: Path, *, quick: bool, seed: int) -> Dict[str, Any]:
    profile = RevisionQuickProfile()
    reps = 1 if quick else 400
    rng = np.random.default_rng(seed)
    tau_df = _tau_prior_meff_draws(p=50, p0=2, n=300 if quick else 500, rng=rng, num_draws=250 if quick else 4000)
    tau_df.to_csv(outdir / "hyper_tau_meff.csv", index=False)

    tau_eval_rows: list[dict[str, Any]] = []
    tau0_base = float(GRRHS_NUTS.calibrate_tau0(p0=4, D=60, n=300 if quick else 500, sigma_ref=1.0))
    tau_strategies = {
        "tau0_narrow": {"tau0_override": 0.5 * tau0_base, "auto_calibrate_tau": False},
        "tau0_recommended": {"tau0_override": tau0_base, "auto_calibrate_tau": False},
        "tau0_wide": {"tau0_override": 2.0 * tau0_base, "auto_calibrate_tau": False},
    }
    if quick:
        for label, opts in tau_strategies.items():
            tau_eval_rows.append(
                {
                    "rep": 0,
                    "tau_strategy": label,
                    "dataset": "quick_mode_placeholder",
                    "status": "skipped_in_quick_mode",
                    "tau0_override": float(opts["tau0_override"]),
                }
            )
    else:
        for rep in range(reps):
            ds_null = _make_heterogeneity_dataset(seed=int(rng.integers(1_000_000)), group_width=10)
            ds_sparse = _make_heterogeneity_dataset(seed=int(rng.integers(1_000_000)), group_width=50)
            for label, opts in tau_strategies.items():
                for ds_name, ds in [("heterogeneity_small", ds_null), ("heterogeneity_large", ds_sparse)]:
                    try:
                        fit = _fit_method(
                            "grrhs",
                            ds["X"],
                            ds["y"],
                            ds["groups"],
                            task="gaussian",
                            seed=int(rng.integers(1_000_000)),
                            quick=quick,
                            p0=4,
                            alpha_kappa=0.5,
                            beta_kappa=1.0,
                            tau0_override=float(opts["tau0_override"]),
                            auto_calibrate_tau=bool(opts["auto_calibrate_tau"]),
                        )
                    except Exception as exc:
                        tau_eval_rows.append({"rep": rep, "tau_strategy": label, "dataset": ds_name, "status": "error", "error": type(exc).__name__})
                        continue
                    post = _posterior_draws(fit)
                    beta_hat = _posterior_mean(post.get("beta"))
                    mse = _mse_partition(ds["beta"], beta_hat)
                    scores = _group_scores_from_model(fit, ds["groups"])
                    tau_eval_rows.append(
                        {
                            "rep": rep,
                            "tau_strategy": label,
                            "dataset": ds_name,
                            "status": "ok",
                            "tau0_override": float(opts["tau0_override"]),
                            **mse,
                            "kappa_mean_null_groups": float(np.mean(np.asarray(scores.get("kappa_mean", np.nan))[:2])) if "kappa_mean" in scores else float("nan"),
                        }
                    )
    tau_eval_path = outdir / "hyper_tau_inference_comparison.csv"
    pd.DataFrame(tau_eval_rows).to_csv(tau_eval_path, index=False)

    rows: list[dict[str, Any]] = []
    if quick:
        for alpha_kappa, beta_kappa in [(0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (0.5, 1.0)]:
            rows.append(
                {
                    "rep": 0,
                    "alpha_kappa": alpha_kappa,
                    "beta_kappa": beta_kappa,
                    "status": "skipped_in_quick_mode",
                    "p_g": 10,
                    "alpha_gt_pg_over_2": bool(float(alpha_kappa) > 5.0),
                }
            )
    else:
        for rep in range(reps):
            ds = _make_heterogeneity_dataset(seed=int(rng.integers(1_000_000)))
            labels = np.asarray(ds["mu"] > 0.0, dtype=int)
            for alpha_kappa, beta_kappa in [(0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (0.5, 1.0)]:
                try:
                    fit = _fit_method("grrhs", ds["X"], ds["y"], ds["groups"], task="gaussian", seed=int(rng.integers(1_000_000)), quick=quick, p0=4, alpha_kappa=alpha_kappa, beta_kappa=beta_kappa)
                except Exception as exc:
                    rows.append({"rep": rep, "alpha_kappa": alpha_kappa, "beta_kappa": beta_kappa, "status": "error", "error": type(exc).__name__})
                    continue
                conv = _convergence_gate(fit)
                scores = _group_scores_from_model(fit, ds["groups"])
                beta_hat = _posterior_mean(_posterior_draws(fit).get("beta"))
                pg = len(ds["groups"][0])
                rows.append(
                    {
                        "rep": rep,
                        "alpha_kappa": alpha_kappa,
                        "beta_kappa": beta_kappa,
                        "p_g": int(pg),
                        "alpha_gt_pg_over_2": bool(float(alpha_kappa) > float(pg) / 2.0),
                        **_mse_partition(ds["beta"], beta_hat),
                        "kappa_auroc": float(roc_auc_score(labels, scores["kappa_mean"])) if "kappa_mean" in scores else float("nan"),
                        "divergence_rate": conv["divergence_rate"],
                        "max_rhat": conv["max_rhat"],
                        "min_bulk_ess": conv["min_bulk_ess"],
                    }
                )
    beta_grid_path = outdir / "hyper_beta_prior_grid.csv"
    pd.DataFrame(rows).to_csv(beta_grid_path, index=False)

    tail_rows: list[dict[str, Any]] = []
    for beta_kappa in [0.5, 1.0, 2.0]:
        draws = _sample_profile_marginal_beta_prior(
            alpha_kappa=1.0,
            beta_kappa=float(beta_kappa),
            tau=0.3,
            sigma=1.0,
            num_draws=3000 if quick else 30000,
            rng=rng,
        )
        est = _estimate_tail_density_exponent(np.abs(draws))
        tail_rows.append(
            {
                "alpha_kappa": 1.0,
                "beta_kappa": float(beta_kappa),
                "estimated_density_tail_exponent": float(est),
                "theory_density_tail_exponent": float(2.0 * float(beta_kappa) + 2.0),
            }
        )
    tail_path = outdir / "hyper_marginal_prior_tail_check.csv"
    pd.DataFrame(tail_rows).to_csv(tail_path, index=False)
    return {
        "tau_table": str(outdir / "hyper_tau_meff.csv"),
        "tau_inference_table": str(tau_eval_path),
        "beta_grid_table": str(beta_grid_path),
        "marginal_tail_table": str(tail_path),
    }


def run_revision_suite(
    *,
    output_dir: str | Path,
    sections: Optional[Sequence[str]] = None,
    quick: bool = True,
    seed: int = 20260416,
    methods: Sequence[str] = DEFAULT_METHODS,
) -> Dict[str, Any]:
    outdir = Path(output_dir).resolve()
    _ensure_dir(outdir)
    requested = {str(x).strip().lower() for x in (sections or ["theory", "benchmark", "heterogeneity", "inferential", "logistic", "ablation", "hyper"])}
    summary: Dict[str, Any] = {"output_dir": str(outdir), "quick": bool(quick), "seed": int(seed), "methods": list(methods)}
    if "theory" in requested:
        theory_dir = outdir / "theory"
        _ensure_dir(theory_dir)
        summary["theory"] = _run_theory_section(theory_dir, quick=quick, seed=seed + 101)
    if "benchmark" in requested:
        bench_dir = outdir / "benchmark"
        _ensure_dir(bench_dir)
        summary["benchmark"] = _run_benchmark_section(bench_dir, quick=quick, seed=seed + 202, methods=methods)
    if "heterogeneity" in requested:
        het_dir = outdir / "heterogeneity"
        _ensure_dir(het_dir)
        summary["heterogeneity"] = _run_heterogeneity_section(het_dir, quick=quick, seed=seed + 303, methods=methods)
    if "inferential" in requested:
        inf_dir = outdir / "inferential"
        _ensure_dir(inf_dir)
        summary["inferential"] = _run_inferential_section(inf_dir, quick=quick, seed=seed + 404, methods=methods)
    if "logistic" in requested:
        log_dir = outdir / "logistic"
        _ensure_dir(log_dir)
        summary["logistic"] = _run_logistic_section(log_dir, quick=quick, seed=seed + 505, methods=methods)
    if "ablation" in requested:
        abl_dir = outdir / "ablation"
        _ensure_dir(abl_dir)
        summary["ablation"] = _run_ablation_section(abl_dir, quick=quick, seed=seed + 606)
    if "hyper" in requested:
        hyp_dir = outdir / "hyper"
        _ensure_dir(hyp_dir)
        summary["hyper"] = _run_hyperparameter_section(hyp_dir, quick=quick, seed=seed + 707)
    summary_path = outdir / "revision_suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
