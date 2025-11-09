"""Comprehensive evaluation utilities for Bayesian regression models.

This module assembles predictive accuracy, variable selection, uncertainty
calibration, shrinkage diagnostics, and basic inference diagnostics. It leans
on mature scientific Python libraries (NumPy, scikit-learn, ArviZ, SciPy) to
ensure well-tested implementations of common metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    auc,
    mean_squared_error,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

try:  # Optional plotting libraries (not used in automated metrics)
    import matplotlib.pyplot as plt  # noqa: F401
    import seaborn as sns  # noqa: F401
except Exception:  # pragma: no cover
    plt = None  # type: ignore
    sns = None  # type: ignore

from scipy.special import expit, logsumexp
from scipy.stats import norm, probplot  # noqa: F401

from grwhs.diagnostics.postprocess import compute_diagnostics_from_samples


def _as_numpy(x: Optional[ArrayLike]) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    return arr


def _mlpd_from_loglik_samples(loglik_samples: np.ndarray) -> Optional[float]:
    """
    Compute mean log predictive density via log-sum-exp across posterior draws.

    Args:
        loglik_samples: array with shape (S, n) or (n,), containing per-draw log-likelihoods.

    Returns:
        Scalar MLPD estimate or None if insufficient data.
    """
    arr = np.asarray(loglik_samples, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        return float(np.mean(arr))
    if arr.ndim == 2 and arr.shape[0] > 0:
        lpd_i = logsumexp(arr, axis=0) - np.log(arr.shape[0])
        return float(np.mean(lpd_i))
    return None


def predictive_metrics(
    y_true: Optional[ArrayLike],
    y_pred: Optional[ArrayLike],
    loglik_samples: Optional[np.ndarray] = None,
    *,
    pseudo_sigma2: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Compute predictive metrics (RMSE, predictive log-likelihood)."""

    metrics: Dict[str, Optional[float]] = {
        "RMSE": None,
        "PredictiveLogLikelihood": None,
        "MLPD": None,
    }

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
        mlpd_val = _mlpd_from_loglik_samples(ll)
        metrics["MLPD"] = None if mlpd_val is None else float(mlpd_val)
        if metrics["MLPD"] is not None:
            metrics["MLPD_source"] = "posterior_draws"
    elif (
        pseudo_sigma2 is not None
        and pseudo_sigma2 > 0
        and y is not None
        and pred is not None
        and y.shape[0] == pred.shape[0]
    ):
        sigma2 = float(max(pseudo_sigma2, 1e-8))
        residual = y - pred
        log_terms = -0.5 * (np.log(2 * np.pi * sigma2) + (residual**2) / sigma2)
        pseudo_loglik = float(np.mean(log_terms))
        metrics["PredictiveLogLikelihood"] = pseudo_loglik
        metrics["MLPD"] = pseudo_loglik
        metrics["MLPD_source"] = "gaussian_residual_proxy"
    return metrics


def classification_metrics(
    y_true: Optional[ArrayLike],
    prob_positive: Optional[ArrayLike],
    pred_labels: Optional[ArrayLike],
    loglik_samples: Optional[np.ndarray] = None,
    *,
    threshold: float = 0.5,
) -> Dict[str, Optional[float]]:
    """Compute binary classification metrics from probabilities or labels."""

    metrics: Dict[str, Optional[float]] = {
        "ClassAccuracy": None,
        "ClassF1": None,
        "ClassLogLoss": None,
        "ClassBrier": None,
        "ClassAUROC": None,
        "ClassAveragePrecision": None,
        "MLPD": None,
    }

    y = _as_numpy(y_true)
    if y is None or y.size == 0:
        return metrics

    y = y.reshape(-1)

    prob: Optional[np.ndarray]
    if prob_positive is None:
        prob = None
    else:
        prob_arr = np.asarray(prob_positive, dtype=float)
        if prob_arr.ndim == 2 and prob_arr.shape[1] >= 2:
            prob = prob_arr[:, -1]
        else:
            prob = prob_arr.reshape(-1)
        if prob.shape[0] != y.shape[0]:
            prob = None

    preds: Optional[np.ndarray]
    if pred_labels is None:
        preds = None
    else:
        preds_arr = np.asarray(pred_labels).reshape(-1)
        preds = preds_arr if preds_arr.shape[0] == y.shape[0] else None

    if prob is not None:
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        pred_binary = (prob >= threshold).astype(int)
        metrics["ClassAccuracy"] = float(accuracy_score(y, pred_binary))
        uniques = np.unique(y)
        if uniques.size == 2:
            try:
                metrics["ClassF1"] = float(f1_score(y, pred_binary))
            except ValueError:
                metrics["ClassF1"] = None
            try:
                metrics["ClassAUROC"] = float(roc_auc_score(y, prob))
            except ValueError:
                metrics["ClassAUROC"] = None
            try:
                metrics["ClassAveragePrecision"] = float(average_precision_score(y, prob))
            except ValueError:
                metrics["ClassAveragePrecision"] = None
        try:
            metrics["ClassLogLoss"] = float(log_loss(y, np.column_stack([1.0 - prob, prob])))
        except ValueError:
            metrics["ClassLogLoss"] = None
        try:
            metrics["ClassBrier"] = float(brier_score_loss(y, prob))
        except ValueError:
            metrics["ClassBrier"] = None
        if preds is None:
            preds = pred_binary
    elif preds is not None:
        preds_binary = preds.astype(int)
        metrics["ClassAccuracy"] = float(accuracy_score(y, preds_binary))
        if np.unique(y).size == 2:
            try:
                metrics["ClassF1"] = float(f1_score(y, preds_binary))
            except ValueError:
                metrics["ClassF1"] = None

    if loglik_samples is not None:
        ll = np.asarray(loglik_samples, dtype=float)
        mlpd_val = _mlpd_from_loglik_samples(ll)
        metrics["MLPD"] = None if mlpd_val is None else float(mlpd_val)
        if metrics["MLPD"] is not None:
            metrics["ClassLogLoss"] = float(-metrics["MLPD"])
    return metrics


def _bernoulli_log_likelihood_samples(
    y_true: Optional[ArrayLike],
    logits_draws: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Return per-draw Bernoulli log-likelihoods for classification tasks."""

    y = _as_numpy(y_true)
    if y is None or y.size == 0:
        return None

    if logits_draws is None:
        return None

    draws = np.asarray(logits_draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != y.shape[0]:
        return None

    y = y.reshape(1, -1)
    safe_logits = np.clip(draws, -60.0, 60.0)
    log_p1 = -np.logaddexp(0.0, -safe_logits)
    log_p0 = -np.logaddexp(0.0, safe_logits)
    return y * log_p1 + (1.0 - y) * log_p0

def selection_metrics(
    y_true_binary: Optional[ArrayLike],
    scores: Optional[ArrayLike],
) -> Dict[str, Optional[float]]:
    """Compute variable selection metrics (AUC-PR and best-threshold F1)."""

    metrics: Dict[str, Optional[float]] = {"AUC-PR": None, "F1": None, "F1_threshold": None}

    y = _as_numpy(y_true_binary)
    s = _as_numpy(scores)
    if y is None or s is None or y.ndim != 1 or y.size == 0:
        return metrics

    # Need both classes present for meaningful metrics
    if np.unique(y).size < 2:
        return metrics

    try:
        precision_curve, recall_curve, thresholds = precision_recall_curve(y, s)
        if precision_curve.size > 1 and recall_curve.size > 1:
            metrics["AUC-PR"] = float(auc(recall_curve, precision_curve))
        if precision_curve.size > 1 and recall_curve.size > 1:
            denom = precision_curve[1:] + recall_curve[1:]
            denom = np.where(denom == 0.0, 1.0, denom)
            f1_scores = 2.0 * precision_curve[1:] * recall_curve[1:] / denom
            if f1_scores.size > 0:
                idx = int(np.nanargmax(f1_scores))
                metrics["F1"] = float(f1_scores[idx])
                if thresholds.size > 0:
                    capped_idx = min(idx, thresholds.size - 1)
                    metrics["F1_threshold"] = float(thresholds[capped_idx])
    except Exception:  # pragma: no cover - handles edge cases
        metrics["AUC-PR"] = None

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
    mean_effective_nonzeros: Optional[float],
) -> Dict[str, Optional[float]]:
    """Mean shrinkage factor, effective degrees of freedom, and active complexity."""

    kap = _as_numpy(kappa)
    mean_kappa = float(np.mean(kap)) if kap is not None else None
    edf_val = None if effective_dof is None else float(effective_dof)
    eff_nz_val = None if mean_effective_nonzeros is None else float(mean_effective_nonzeros)
    return {
        "MeanKappa": mean_kappa,
        "EffectiveDoF": edf_val,
        "MeanEffectiveNonzeros": eff_nz_val,
    }


def _proxy_effective_counts(
    coef_draws: Optional[np.ndarray],
    coef_point: Optional[np.ndarray],
    group_index: Optional[np.ndarray],
    *,
    rel_threshold: float = 1e-2,
    abs_threshold: float = 1e-6,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Backstop heuristic for models without posterior shrinkage diagnostics.

    Args:
        coef_draws: optional posterior draws (S, p) or (p,)
        coef_point: optional single vector (p,)
        group_index: optional array mapping features -> group id
        rel_threshold: relative threshold w.r.t. max |beta|
        abs_threshold: absolute lower bound on threshold

    Returns:
        Tuple (effective_nonzeros, effective_dof) using magnitude thresholds.
    """

    source: Optional[np.ndarray] = None
    if coef_draws is not None:
        arr = np.asarray(coef_draws, dtype=float)
        if arr.ndim == 1:
            source = arr.reshape(-1)
        elif arr.ndim >= 2:
            source = np.mean(arr.reshape(arr.shape[0], -1), axis=0)
    if source is None and coef_point is not None:
        source = np.asarray(coef_point, dtype=float).reshape(-1)
    if source is None:
        return None, None

    magnitudes = np.abs(source)
    if magnitudes.size == 0:
        return None, None
    max_mag = float(np.max(magnitudes))
    if max_mag <= 0.0:
        return 0.0, 0.0

    threshold = max(abs_threshold, rel_threshold * max_mag)
    active_mask = magnitudes >= threshold
    effective_nonzeros = float(np.sum(active_mask))

    effective_dof = effective_nonzeros
    if group_index is not None and group_index.shape[0] == active_mask.shape[0]:
        unique_groups = np.unique(group_index)
        group_activity = 0.0
        for gid in unique_groups:
            members = group_index == gid
            if np.any(active_mask[members]):
                group_activity += 1.0
        effective_dof = group_activity

    return effective_nonzeros, effective_dof


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
    sigma_samples: Optional[np.ndarray] = None,
    *,
    rng_seed: int = 0,
) -> Optional[np.ndarray]:
    if coef_samples is None:
        return None
    preds = (X @ coef_samples.T).T  # (S, n)
    if np.isscalar(intercept):
        preds = preds + float(intercept)
    else:
        preds = preds + np.asarray(intercept)[..., np.newaxis]
    if sigma_samples is not None:
        sigma = np.asarray(sigma_samples, dtype=float).reshape(-1)
        S, _ = preds.shape
        if sigma.size == 1:
            sigma = np.repeat(sigma, S)
        elif sigma.size < S:
            sigma = np.pad(sigma, (0, S - sigma.size), mode="edge")
        elif sigma.size > S:
            sigma = sigma[:S]
        sigma = np.maximum(sigma, 1e-8)
        rng = np.random.default_rng(rng_seed)
        noise = rng.standard_normal(preds.shape) * sigma[:, np.newaxis]
        preds = preds + noise
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
    task: str = "regression",
    classification_threshold: float = 0.5,
) -> Dict[str, Optional[float]]:
    """Evaluate a fitted model across predictive, classification, selection, and calibration metrics."""

    task_label = str(task).lower()
    if task_label in {"binary", "binary_classification", "cls"}:
        task_label = "classification"
    if task_label not in {"regression", "classification"}:
        raise ValueError(f"Unsupported task '{task}'. Expected 'regression' or 'classification'.")

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

    prob_positive = None
    prob_draw_mean: Optional[np.ndarray] = None
    class_loglik_samples: Optional[np.ndarray] = None
    if task_label == "classification" and Xte is not None:
        proba_method = getattr(model, "predict_proba", None)
        if callable(proba_method):
            try:
                proba = np.asarray(proba_method(Xte))
                if proba.ndim == 1:
                    prob_positive = proba
                elif proba.ndim == 2 and proba.shape[1] >= 2:
                    prob_positive = proba[:, -1]
            except Exception:
                prob_positive = None
        if prob_positive is None:
            decision_method = getattr(model, "decision_function", None)
            if callable(decision_method):
                try:
                    decision_scores = np.asarray(decision_method(Xte), dtype=float).reshape(-1)
                    prob_positive = 1.0 / (1.0 + np.exp(-np.clip(decision_scores, -60.0, 60.0)))
                except Exception:
                    prob_positive = None

    if task_label == "classification" and Xte is not None and posterior.coef is not None:
        try:
            logits_draws = _predictive_draws(
                Xte,
                posterior.coef,
                intercept_val,
                sigma_samples=None,
            )
            if logits_draws is not None and logits_draws.size > 0:
                prob_draws = expit(np.clip(logits_draws, -60.0, 60.0))
                prob_draw_mean = prob_draws.mean(axis=0)
                if yte is not None:
                    class_loglik_samples = _bernoulli_log_likelihood_samples(yte, logits_draws)
        except Exception:
            prob_draw_mean = None
            class_loglik_samples = None

    if prob_draw_mean is not None:
        prob_positive = prob_draw_mean

    pseudo_sigma2: Optional[float] = None
    if task_label == "regression":
        Xtr = _as_numpy(X_train)
        ytr = _as_numpy(y_train)
        if Xtr is not None and ytr is not None and ytr.size > 1:
            try:
                train_preds = np.asarray(model.predict(Xtr))
                train_preds = train_preds.reshape(-1)
                ytr_vec = ytr.reshape(-1)
                if train_preds.shape[0] == ytr_vec.shape[0]:
                    residuals = ytr_vec - train_preds
                    if residuals.size > 1:
                        sigma2 = float(np.var(residuals, ddof=1))
                    else:
                        sigma2 = float(np.var(residuals))
                    if sigma2 > 0:
                        pseudo_sigma2 = sigma2
            except Exception:
                pseudo_sigma2 = None

    pred_draws = None
    loglik_samples = None
    if task_label == "regression":
        if Xte is not None and posterior.coef is not None:
            pred_draws = _predictive_draws(
                Xte,
                posterior.coef,
                intercept_val,
                sigma_samples=posterior.sigma,
            )
        if yte is not None and pred_draws is not None and posterior.sigma is not None:
            loglik_samples = _log_likelihood_samples(yte, pred_draws, posterior.sigma)
        metrics.update(
            predictive_metrics(
                yte,
                pred_mean,
                loglik_samples,
                pseudo_sigma2=pseudo_sigma2,
            )
        )
    else:
        metrics.update(
            classification_metrics(
                yte,
                prob_positive,
                pred_mean,
                loglik_samples=class_loglik_samples,
                threshold=classification_threshold,
            )
        )

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

    if task_label == "regression":
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
    mean_effective_nonzeros = None
    phi_samples = posterior.phi
    lambda_samples = posterior.lambda_
    tau_samples = posterior.tau
    sigma_samples = posterior.sigma
    if sigma_samples is None and task_label == "classification":
        sample_count = None
        if lambda_samples is not None:
            sample_count = lambda_samples.shape[0]
        elif posterior.coef is not None:
            sample_count = posterior.coef.shape[0]
        elif tau_samples is not None:
            sample_count = tau_samples.shape[0]
        if sample_count is not None and sample_count > 0:
            sigma_samples = np.full(int(sample_count), 2.0, dtype=float)
    if (
        X_train is not None
        and lambda_samples is not None
        and tau_samples is not None
        and sigma_samples is not None
        and group_index is not None
    ):
        try:
            G = int(np.max(group_index)) + 1 if group_index.size else 1
            if phi_samples is None:
                phi_samples = np.ones((lambda_samples.shape[0], G), dtype=float)
            elif phi_samples.shape[1] < G:
                # Expand to full group count if necessary
                S = phi_samples.shape[0]
                expanded = np.ones((S, G), dtype=float)
                expanded[:, :phi_samples.shape[1]] = phi_samples
                phi_samples = expanded
            slab_width = (
                slab_width
                or getattr(model, "slab_scale", None)
                or getattr(model, "c", None)
                or 2.0
            )
            diag_res = compute_diagnostics_from_samples(
                X=np.asarray(X_train),
                group_index=np.asarray(group_index),
                c=float(slab_width),
                eps=1e-8,
                lambda_=lambda_samples,
                tau=tau_samples,
                phi=phi_samples,
                sigma=sigma_samples,
            )
            mean_kappa = float(np.mean(diag_res.per_coeff["kappa"]))
            if "edf" in diag_res.per_group:
                edf_total = float(np.sum(diag_res.per_group["edf"]))
            if "effective_nonzeros_mean" in diag_res.meta:
                mean_effective_nonzeros = float(diag_res.meta["effective_nonzeros_mean"])
        except Exception:  # pragma: no cover - diagnostics may fail for deterministics
            mean_kappa = None
            edf_total = None
            mean_effective_nonzeros = None

    if mean_effective_nonzeros is None or edf_total is None:
        proxy_eff, proxy_edf = _proxy_effective_counts(
            posterior.coef,
            coef_hat,
            group_index,
        )
        if mean_effective_nonzeros is None and proxy_eff is not None:
            mean_effective_nonzeros = proxy_eff
        if edf_total is None and proxy_edf is not None:
            edf_total = proxy_edf

    metrics.update(shrinkage_metrics(mean_kappa, edf_total, mean_effective_nonzeros))

    return metrics
