"""Comprehensive evaluation utilities for Bayesian regression models.

This module assembles predictive accuracy, variable selection, uncertainty
calibration, shrinkage diagnostics, and basic inference diagnostics. It leans
on mature scientific Python libraries (NumPy, scikit-learn, ArviZ, SciPy) to
ensure well-tested implementations of common metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    auc,
    f1_score,
    mean_squared_error,
    precision_recall_curve,
)

try:  # Optional plotting libraries (not used in automated metrics)
    import matplotlib.pyplot as plt  # noqa: F401
    import seaborn as sns  # noqa: F401
except Exception:  # pragma: no cover
    plt = None  # type: ignore
    sns = None  # type: ignore

from scipy.stats import norm, probplot  # noqa: F401

from grwhs.diagnostics.postprocess import compute_diagnostics_from_samples


def _as_numpy(x: Optional[ArrayLike]) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    return arr


def predictive_metrics(
    y_true: Optional[ArrayLike],
    y_pred: Optional[ArrayLike],
    loglik_samples: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """Compute predictive metrics (RMSE, predictive log-likelihood)."""

    metrics: Dict[str, Optional[float]] = {"RMSE": None, "PredictiveLogLikelihood": None}

    y = _as_numpy(y_true)
    pred = _as_numpy(y_pred)
    if y is None or pred is None or y.shape[0] == 0:
        return metrics

    metrics["RMSE"] = float(np.sqrt(mean_squared_error(y, pred)))

    if loglik_samples is not None:
        ll = np.asarray(loglik_samples, dtype=float)
        if ll.ndim == 1:
            metrics["PredictiveLogLikelihood"] = float(np.mean(ll))
        elif ll.ndim == 2:
            metrics["PredictiveLogLikelihood"] = float(ll.mean())
    return metrics


def selection_metrics(
    y_true_binary: Optional[ArrayLike],
    scores: Optional[ArrayLike],
    threshold: float = 0.5,
) -> Dict[str, Optional[float]]:
    """Compute variable selection metrics (AUC-PR and F1)."""

    metrics: Dict[str, Optional[float]] = {"AUC-PR": None, "F1": None}

    y = _as_numpy(y_true_binary)
    s = _as_numpy(scores)
    if y is None or s is None or y.ndim != 1 or y.size == 0:
        return metrics

    # Need both classes present for meaningful metrics
    if np.unique(y).size < 2:
        return metrics

    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y, s)
        if precision_curve.size > 1 and recall_curve.size > 1:
            metrics["AUC-PR"] = float(auc(recall_curve, precision_curve))
    except Exception:  # pragma: no cover - handles edge cases
        metrics["AUC-PR"] = None

    try:
        preds = (s >= threshold).astype(int)
        metrics["F1"] = float(f1_score(y, preds, zero_division=0))
    except Exception:  # pragma: no cover
        metrics["F1"] = None

    return metrics


def interval_metrics(
    y_true: Optional[ArrayLike],
    intervals: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    """Empirical coverage and average width for predictive intervals."""

    metrics: Dict[str, Optional[float]] = {"Coverage90": None, "IntervalWidth90": None}

    y = _as_numpy(y_true)
    if intervals is None or y is None or y.size == 0:
        return metrics

    arr = np.asarray(intervals, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return metrics

    lower, upper = arr[:, 0], arr[:, 1]
    covered = (y >= lower) & (y <= upper)
    metrics["Coverage90"] = float(np.mean(covered))
    metrics["IntervalWidth90"] = float(np.mean(upper - lower))
    return metrics


def shrinkage_metrics(
    kappa: Optional[ArrayLike],
    effective_dof: Optional[float],
) -> Dict[str, Optional[float]]:
    """Mean shrinkage factor and total effective degrees of freedom."""

    kap = _as_numpy(kappa)
    mean_kappa = float(np.mean(kap)) if kap is not None else None
    edf_val = None if effective_dof is None else float(effective_dof)
    return {"MeanKappa": mean_kappa, "EffectiveDoF": edf_val}


@dataclass
class PosteriorSamples:
    coef: Optional[np.ndarray] = None  # (S, p)
    sigma: Optional[np.ndarray] = None  # (S,)
    lambda_: Optional[np.ndarray] = None  # (S, p)
    tau: Optional[np.ndarray] = None  # (S,)
    phi: Optional[np.ndarray] = None  # (S, G)


def _prepare_posterior_samples(model: Any) -> PosteriorSamples:
    coef = _as_numpy(getattr(model, "coef_samples_", None))
    if coef is not None and coef.ndim == 1:
        coef = coef.reshape(1, -1)

    sigma = _as_numpy(getattr(model, "sigma_samples_", None))
    if sigma is None:
        sigma2 = _as_numpy(getattr(model, "sigma2_samples_", None))
        if sigma2 is not None:
            sigma = np.sqrt(np.maximum(sigma2, 1e-12))
    if sigma is not None and sigma.ndim > 1:
        sigma = sigma.reshape(sigma.shape[0], -1).mean(axis=1)

    lambda_samples = _as_numpy(getattr(model, "lambda_samples_", None))
    if lambda_samples is not None and lambda_samples.ndim == 1:
        lambda_samples = lambda_samples.reshape(-1, 1)

    tau_samples = _as_numpy(getattr(model, "tau_samples_", None))
    if tau_samples is not None and tau_samples.ndim > 1:
        tau_samples = tau_samples.reshape(tau_samples.shape[0], -1).mean(axis=1)

    phi_samples = _as_numpy(getattr(model, "phi_samples_", None))
    if phi_samples is not None and phi_samples.ndim == 1:
        phi_samples = phi_samples.reshape(-1, 1)

    return PosteriorSamples(coef=coef, sigma=sigma, lambda_=lambda_samples, tau=tau_samples, phi=phi_samples)


def _predictive_draws(
    X: np.ndarray,
    coef_samples: Optional[np.ndarray],
    intercept: float | np.ndarray,
) -> Optional[np.ndarray]:
    if coef_samples is None:
        return None
    preds = (X @ coef_samples.T).T  # (S, n)
    if np.isscalar(intercept):
        preds = preds + float(intercept)
    else:
        preds = preds + np.asarray(intercept)[..., np.newaxis]
    return preds


def _log_likelihood_samples(
    y_true: np.ndarray,
    pred_draws: Optional[np.ndarray],
    sigma_samples: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if pred_draws is None or sigma_samples is None:
        return None
    S, n = pred_draws.shape
    sigma = np.asarray(sigma_samples, dtype=float).reshape(-1)
    if sigma.size == 1:
        sigma = np.repeat(sigma, S)
    elif sigma.size != S:
        sigma = sigma[:S]
    sigma = np.maximum(sigma, 1e-8)
    resid = y_true[np.newaxis, :] - pred_draws
    log_sigma = np.log(sigma)[:, np.newaxis]
    loglik = -0.5 * ((resid**2) / (sigma[:, np.newaxis] ** 2)) - np.log(np.sqrt(2 * np.pi)) - log_sigma
    return loglik


def evaluate_model_metrics(
    *,
    model: Any,
    X_train: Optional[np.ndarray],
    X_test: Optional[np.ndarray],
    y_train: Optional[np.ndarray],  # unused but kept for extensibility
    y_test: Optional[np.ndarray],
    beta_truth: Optional[np.ndarray] = None,
    group_index: Optional[np.ndarray] = None,
    coverage_level: float = 0.9,
    slab_width: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Evaluate a fitted model across predictive, selection, calibration, and shrinkage metrics."""

    metrics: Dict[str, Optional[float]] = {}

    Xte = _as_numpy(X_test)
    yte = _as_numpy(y_test)

    try:
        pred_mean = None if Xte is None else np.asarray(model.predict(Xte))
    except Exception:  # Some deterministic baselines might not implement predict
        pred_mean = None

    posterior = _prepare_posterior_samples(model)

    coef_hat = _as_numpy(getattr(model, "coef_", None))
    if coef_hat is not None and posterior.coef is None:
        posterior.coef = coef_hat.reshape(1, -1)

    sigma_mean = getattr(model, "sigma_mean_", None)
    if posterior.sigma is None and sigma_mean is not None:
        posterior.sigma = np.asarray([max(float(sigma_mean), 1e-8)])

    intercept = getattr(model, "intercept_", 0.0)
    if np.isscalar(intercept):
        intercept_val = float(intercept)
    else:
        intercept_val = np.asarray(intercept)

    pred_draws = None
    if Xte is not None and posterior.coef is not None:
        pred_draws = _predictive_draws(Xte, posterior.coef, intercept_val)

    loglik_samples = None
    if yte is not None and pred_draws is not None and posterior.sigma is not None:
        loglik_samples = _log_likelihood_samples(yte, pred_draws, posterior.sigma)

    metrics.update(predictive_metrics(yte, pred_mean, loglik_samples))

    # Variable selection metrics
    beta_truth_bin = None
    if beta_truth is not None:
        beta_truth_bin = (np.abs(beta_truth) > 1e-8).astype(int)

    prob_scores = None
    if posterior.coef is not None:
        prob_scores = np.mean(np.abs(posterior.coef) > 1e-6, axis=0)
    elif coef_hat is not None:
        scaled = np.abs(coef_hat)
        if scaled.max() > 0:
            prob_scores = scaled / scaled.max()
        else:
            prob_scores = scaled

    metrics.update(selection_metrics(beta_truth_bin, prob_scores))

    # Predictive intervals
    interval_array = None
    if pred_draws is not None and pred_draws.shape[0] > 1:
        lower = np.quantile(pred_draws, (1 - coverage_level) / 2, axis=0)
        upper = np.quantile(pred_draws, 1 - (1 - coverage_level) / 2, axis=0)
        interval_array = np.column_stack([lower, upper])
    elif pred_mean is not None and posterior.sigma is not None:
        sigma = float(posterior.sigma[0])
        z = norm.ppf(0.5 + coverage_level / 2)
        lower = pred_mean - z * sigma
        upper = pred_mean + z * sigma
        interval_array = np.column_stack([lower, upper])

    metrics.update(interval_metrics(yte, interval_array))

    # Shrinkage diagnostics (mean kappa)
    mean_kappa = None
    edf_total = None
    if (
        X_train is not None
        and posterior.lambda_ is not None
        and posterior.tau is not None
        and posterior.phi is not None
        and posterior.sigma is not None
        and group_index is not None
    ):
        try:
            diag_res = compute_diagnostics_from_samples(
                X=np.asarray(X_train),
                group_index=np.asarray(group_index),
                c=float(slab_width or getattr(model, "c", 1.0)),
                eps=1e-8,
                lambda_=posterior.lambda_,
                tau=posterior.tau,
                phi=posterior.phi,
                sigma=posterior.sigma,
            )
            mean_kappa = float(np.mean(diag_res.per_coeff["kappa"]))
            if "edf" in diag_res.per_group:
                edf_total = float(np.sum(diag_res.per_group["edf"]))
        except Exception:  # pragma: no cover - diagnostics may fail for deterministics
            mean_kappa = None
            edf_total = None

    metrics.update(shrinkage_metrics(mean_kappa, edf_total))

    return metrics
