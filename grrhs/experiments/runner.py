"""
Experiment orchestration entry points for GRRHS experiments.

This module implements the nested cross-validation workflow described in the
project brief:

* Synthetic or loader-backed dataset preparation with train-only standardisation.
* Outer K-fold (optionally repeated) evaluation with shared splits across methods.
* Inner CV hyper-parameter selection for frequentist baselines (GL/SGL).
* Optional τ calibration for GRRHS variants using the expected effective complexity.
* Exhaustive metric logging plus posterior export for Bayesian models.

The function :func:`run_experiment` is the public entry point invoked by the CLI
(`python -m grrhs.cli.run_experiment`).  It expects a fully merged configuration
dictionary and an output directory path where all artefacts will be written.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import kstest, norm

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    _HAS_SKLEARN_GP = True
except Exception:  # pragma: no cover - scikit-learn is a dependency but keep fallback
    GaussianProcessRegressor = None  # type: ignore
    Matern = None  # type: ignore
    WhiteKernel = None  # type: ignore
    _HAS_SKLEARN_GP = False

try:  # Optional dependency used for parquet export
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except Exception:  # pragma: no cover - pandas is listed as dependency but keep fallback
    pd = None  # type: ignore
    _HAS_PANDAS = False

from data.generators import generate_synthetic, synthetic_config_from_dict, make_groups
from data.loaders import load_real_dataset, LoadedDataset
from data.preprocess import (
    StandardizationConfig,
    apply_standardization,
    apply_standardizer,
    apply_y_mean,
)
from data.splits import OuterFold, holdout_splits, outer_kfold_splits
from grrhs.diagnostics.convergence import summarize_convergence
from grrhs.experiments.registry import build_from_config, get_model_name_from_config
from grrhs.metrics.evaluation import evaluate_model_metrics
from grrhs.utils.logging_utils import progress


StratifyOption = Optional[Union[bool, str]]


class ExperimentError(RuntimeError):
    """Raised when a configuration-driven experiment cannot be executed."""


DEFAULT_CONVERGENCE_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "max_rhat": 1.01,
    "min_ess": 100.0,
    "min_ess_by_block": {
        "beta": 400.0,
        "tau": 1000.0,
        "phi": 400.0,
        "gamma": 400.0,
        "lambda": 200.0,
        "b": 200.0,
    },
    "max_mcse_over_sd": 0.10,
    "max_retries": 1,
    "retry_scale": 2.0,
    "parameters": ["beta", "tau"],
    "expected_blocks": {
        "default": ["beta", "tau"],
        "grrhs_gibbs": ["beta", "tau", "phi", "lambda"],
        "grrhs_gibbs_logistic": ["beta", "tau", "phi", "lambda"],
        "grrhs_logistic": ["beta", "tau", "phi", "lambda"],
        "grrhs_gibbs_cls": ["beta", "tau", "phi", "lambda"],
        "gigg": ["beta", "tau", "gamma", "lambda"],
        "gigg_regression": ["beta", "tau", "gamma", "lambda"],
        "regularized_horseshoe": ["beta", "tau", "lambda"],
        "rhs": ["beta", "tau", "lambda"],
        "regularised_horseshoe": ["beta", "tau", "lambda"],
        "horseshoe": ["beta", "tau", "lambda"],
        "hs": ["beta", "tau", "lambda"],
    },
    "missing_policy": "warn",
    "require_valid_diagnostics": True,
    "min_chains_for_rhat": 4,
    "hmc": {
        "enabled": True,
        "models": ["regularized_horseshoe", "rhs", "regularised_horseshoe"],
        "max_divergences": 0,
        "min_ebfmi": 0.3,
        "max_treedepth_hits": 0,
        "require_present": True,
    },
}

DEFAULT_BAYESIAN_FAIRNESS_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "disable_inner_cv": True,
    "enforce_shared_sampling_budget": True,
    "require_posterior_mean_summary": True,
    "disable_budget_retry": True,
    "sampling_budget": {
        "burn_in": 1000,
        "kept_draws": 1000,
        "thinning": 1,
        "num_chains": 4,
    },
}

DEFAULT_POSTERIOR_VALIDATION_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "apply_to_bayesian_only": True,
    "sbc": {
        "enabled": True,
        "ks_pvalue_min": 0.05,
        "coverage_level": 0.9,
        "coverage_tolerance": 0.15,
        "min_coefficients": 8,
        "max_coefficients": 128,
        "fail_on_missing_truth": False,
        "fail_on_missing_draws": True,
    },
    "ppc": {
        "enabled": True,
        "tail_prob": 0.025,
        "min_draws": 200,
        "fail_on_missing_draws": True,
    },
    "seed_stability": {
        "enabled": True,
        "num_restarts": 2,
        "seed_stride": 1009,
        "max_beta_rel_l2": 0.15,
        "min_beta_cosine": 0.98,
        "max_tau_rel_sd": 0.20,
        "fail_on_missing_tau": False,
    },
}

_BAYESIAN_MODEL_NAMES = {
    "grrhs_gibbs",
    "grrhs_gibbs_logistic",
    "grrhs_logistic",
    "grrhs_gibbs_cls",
    "gigg",
    "gigg_regression",
    "regularized_horseshoe",
    "rhs",
    "regularised_horseshoe",
}

_BAYESIAN_HYPERPRIOR_LABELS: Dict[str, Dict[str, str]] = {
    "grrhs_gibbs": {
        "tau": "tau ~ C+(0, 1) via calibrated tau0 heuristic",
        "group": "group scales use HalfNormal/Cauchy-style grouped shrinkage defaults",
    },
    "grrhs_gibbs_logistic": {
        "tau": "tau ~ C+(0, 1) via calibrated tau0 heuristic",
        "group": "group scales use HalfNormal/Cauchy-style grouped shrinkage defaults",
    },
    "gigg": {
        "group": "a_g = 1/n when a_value is null",
        "mmle": "b_g updated by paper_lambda_only MMLE path",
    },
    "gigg_regression": {
        "group": "a_g = 1/n when a_value is null",
        "mmle": "b_g updated by paper_lambda_only MMLE path",
    },
    "regularized_horseshoe": {
        "tau": "tau ~ C+(0, 1)",
        "slab": "c^2 ~ Inv-Gamma(nu/2, nu s^2 / 2)",
    },
    "rhs": {
        "tau": "tau ~ C+(0, 1)",
        "slab": "c^2 ~ Inv-Gamma(nu/2, nu s^2 / 2)",
    },
    "regularised_horseshoe": {
        "tau": "tau ~ C+(0, 1)",
        "slab": "c^2 ~ Inv-Gamma(nu/2, nu s^2 / 2)",
    },
    "horseshoe": {
        "tau": "tau ~ C+(0, 1)",
    },
    "hs": {
        "tau": "tau ~ C+(0, 1)",
    },
}


class _ConstantClassificationModel:
    """Fallback classifier emitting constant probabilities when training data is degenerate."""

    def __init__(self, prob_positive: float, n_features: int) -> None:
        probability = float(np.clip(prob_positive, 0.0, 1.0))
        self.prob_positive = probability
        self._label = float(1.0 if probability >= 0.5 else 0.0)
        eps = 1e-6
        logit = math.log((probability + eps) / (1.0 - probability + eps))
        self.coef_ = np.zeros((1, n_features), dtype=float)
        self.intercept_ = np.array([logit], dtype=float)
        self.classes_ = np.array([0.0, 1.0], dtype=np.float32)

    def fit(self, X: Any, y: Any, **_: Any) -> "_ConstantClassificationModel":
        return self

    def predict(self, X: Any) -> np.ndarray:
        n = 0 if X is None else np.asarray(X).shape[0]
        return np.full(n, self._label, dtype=np.float32)

    def predict_proba(self, X: Any) -> np.ndarray:
        n = 0 if X is None else np.asarray(X).shape[0]
        proba = np.empty((n, 2), dtype=float)
        proba[:, 1] = self.prob_positive
        proba[:, 0] = 1.0 - self.prob_positive
        return proba

    def decision_function(self, X: Any) -> np.ndarray:
        n = 0 if X is None else np.asarray(X).shape[0]
        return np.full(n, float(self.intercept_[0]), dtype=float)


def _to_serializable(value: Any) -> Any:
    """Convert NumPy / Path rich objects into JSON serialisable forms."""
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_task(config: Mapping[str, Any]) -> str:
    """Resolve task label from config, normalising common aliases."""
    task = str(
        config.get("task")
        or config.get("data", {}).get("task")
        or config.get("experiments", {}).get("task", "regression")
    ).lower()
    aliases = {"binary": "classification", "binary_classification": "classification", "cls": "classification"}
    return aliases.get(task, task)


def _resolve_model_name(config: Mapping[str, Any]) -> str:
    model_cfg = config.get("model", {}) or {}
    name = model_cfg.get("name", model_cfg.get("type", ""))
    return str(name).strip().lower()


def _is_bayesian_model_name(name: str) -> bool:
    return str(name).strip().lower() in _BAYESIAN_MODEL_NAMES


def _is_bayesian_config(config: Mapping[str, Any]) -> bool:
    return _is_bayesian_model_name(_resolve_model_name(config))


def _uses_joint_covariates_model_name(name: str) -> bool:
    key = str(name).strip().lower()
    return key in {"gigg", "gigg_regression"}


def _fit_model_dispatch(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Sequence[Sequence[int]],
    C: Optional[np.ndarray] = None,
) -> None:
    if C is not None:
        try:
            model.fit(X, y, groups=groups, C=C)
            return
        except TypeError:
            pass
    try:
        model.fit(X, y, groups=groups)
    except TypeError:
        model.fit(X, y)


def _bayesian_fairness_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    experiments_cfg = config.get("experiments", {}) or {}
    raw = experiments_cfg.get("bayesian_fairness", {}) or {}
    resolved = deepcopy(DEFAULT_BAYESIAN_FAIRNESS_CONFIG)
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if key == "sampling_budget" and isinstance(value, Mapping):
                resolved["sampling_budget"].update(dict(value))
            else:
                resolved[key] = value
    return resolved


def _sampling_budget_from_config(config: Mapping[str, Any]) -> Dict[str, int]:
    fairness_cfg = _bayesian_fairness_config(config)
    raw = fairness_cfg.get("sampling_budget", {}) or {}
    burn_in = max(0, int(raw.get("burn_in", 4000)))
    kept_draws = max(1, int(raw.get("kept_draws", 2000)))
    thinning = max(1, int(raw.get("thinning", 2)))
    num_chains = max(1, int(raw.get("num_chains", 4)))
    return {
        "burn_in": burn_in,
        "kept_draws": kept_draws,
        "thinning": thinning,
        "num_chains": num_chains,
    }


def _posterior_validation_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    experiments_cfg = config.get("experiments", {}) or {}
    raw = experiments_cfg.get("posterior_validation", {}) or {}
    resolved = deepcopy(DEFAULT_POSTERIOR_VALIDATION_CONFIG)
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if key in {"sbc", "ppc", "seed_stability"} and isinstance(value, Mapping):
                resolved[key].update(dict(value))
            else:
                resolved[key] = value
    resolved["enabled"] = bool(resolved.get("enabled", False))
    resolved["apply_to_bayesian_only"] = bool(resolved.get("apply_to_bayesian_only", True))
    sbc_cfg = resolved.get("sbc", {}) or {}
    ppc_cfg = resolved.get("ppc", {}) or {}
    seed_cfg = resolved.get("seed_stability", {}) or {}
    resolved["sbc"] = {
        "enabled": bool(sbc_cfg.get("enabled", True)),
        "ks_pvalue_min": float(sbc_cfg.get("ks_pvalue_min", 0.05)),
        "coverage_level": float(sbc_cfg.get("coverage_level", 0.9)),
        "coverage_tolerance": float(sbc_cfg.get("coverage_tolerance", 0.15)),
        "min_coefficients": max(1, int(sbc_cfg.get("min_coefficients", 8))),
        "max_coefficients": max(1, int(sbc_cfg.get("max_coefficients", 128))),
        "fail_on_missing_truth": bool(sbc_cfg.get("fail_on_missing_truth", False)),
        "fail_on_missing_draws": bool(sbc_cfg.get("fail_on_missing_draws", True)),
    }
    resolved["ppc"] = {
        "enabled": bool(ppc_cfg.get("enabled", True)),
        "tail_prob": float(ppc_cfg.get("tail_prob", 0.025)),
        "min_draws": max(20, int(ppc_cfg.get("min_draws", 200))),
        "fail_on_missing_draws": bool(ppc_cfg.get("fail_on_missing_draws", True)),
    }
    resolved["seed_stability"] = {
        "enabled": bool(seed_cfg.get("enabled", True)),
        "num_restarts": max(1, int(seed_cfg.get("num_restarts", 2))),
        "seed_stride": max(1, int(seed_cfg.get("seed_stride", 1009))),
        "max_beta_rel_l2": max(0.0, float(seed_cfg.get("max_beta_rel_l2", 0.15))),
        "min_beta_cosine": float(seed_cfg.get("min_beta_cosine", 0.98)),
        "max_tau_rel_sd": max(0.0, float(seed_cfg.get("max_tau_rel_sd", 0.20))),
        "fail_on_missing_tau": bool(seed_cfg.get("fail_on_missing_tau", False)),
    }
    return resolved


def _apply_bayesian_sampling_budget(config: MutableMapping[str, Any]) -> None:
    if not _is_bayesian_config(config):
        return
    fairness_cfg = _bayesian_fairness_config(config)
    if not bool(fairness_cfg.get("enabled", True)):
        return
    if not bool(fairness_cfg.get("enforce_shared_sampling_budget", True)):
        return

    budget = _sampling_budget_from_config(config)
    model_cfg = config.setdefault("model", {})
    inference_cfg = config.setdefault("inference", {})
    model_name = _resolve_model_name(config)

    if model_name in {"grrhs_gibbs", "grrhs_gibbs_logistic", "grrhs_logistic", "grrhs_gibbs_cls", "gigg", "gigg_regression"}:
        gibbs_cfg = inference_cfg.setdefault("gibbs", {})
        gibbs_cfg["burn_in"] = int(budget["burn_in"])
        gibbs_cfg["thin"] = int(budget["thinning"])
        gibbs_cfg["num_chains"] = int(budget["num_chains"])
        model_cfg["iters"] = int(budget["burn_in"] + budget["kept_draws"] * budget["thinning"])
        return

    if model_name in {"horseshoe", "hs", "regularized_horseshoe", "rhs", "regularised_horseshoe"}:
        nuts_cfg = inference_cfg.setdefault("nuts", {})
        nuts_cfg["num_warmup"] = int(budget["burn_in"])
        nuts_cfg["num_samples"] = int(budget["kept_draws"] * budget["thinning"])
        nuts_cfg["num_chains"] = int(budget["num_chains"])
        nuts_cfg["thinning"] = int(budget["thinning"])
        model_cfg["num_warmup"] = int(budget["burn_in"])
        model_cfg["num_samples"] = int(budget["kept_draws"] * budget["thinning"])
        model_cfg["num_chains"] = int(budget["num_chains"])
        model_cfg["thinning"] = int(budget["thinning"])


def _bayesian_protocol_summary(
    config: Mapping[str, Any],
    *,
    std_cfg: StandardizationConfig,
    groups: Sequence[Sequence[int]],
    p: int,
) -> Optional[Dict[str, Any]]:
    if not _is_bayesian_config(config):
        return None
    fairness_cfg = _bayesian_fairness_config(config)
    if not bool(fairness_cfg.get("enabled", True)):
        return None

    model_name = _resolve_model_name(config)
    budget = _sampling_budget_from_config(config)
    experiments_cfg = config.get("experiments", {}) or {}
    metrics_cfg = experiments_cfg.get("metrics", {}) or {}
    reporting_cfg = experiments_cfg.get("reporting", {}) or {}
    comparison_cfg = reporting_cfg.get("comparison_metrics", {}) or {}
    regression_metrics = comparison_cfg.get("regression", []) if isinstance(comparison_cfg, Mapping) else []
    primary_metric = regression_metrics[0] if isinstance(regression_metrics, Sequence) and regression_metrics else "RMSE"
    return {
        "same_likelihood": "gaussian_regression",
        "shared_data_representation": {
            "X_standardization": std_cfg.X,
            "y_center": bool(std_cfg.y_center),
            "group_count": int(len(groups)),
            "feature_count": int(p),
        },
        "posterior_sampling_budget": {
            "burn_in": int(budget["burn_in"]),
            "kept_draws": int(budget["kept_draws"]),
            "thinning": int(budget["thinning"]),
            "num_chains": int(budget["num_chains"]),
            "retry_disabled_for_fairness": bool(fairness_cfg.get("disable_budget_retry", True)),
        },
        "hyperprior_policy": {
            "mode": "paper_default_hyperpriors",
            "details": deepcopy(_BAYESIAN_HYPERPRIOR_LABELS.get(model_name, {})),
        },
        "posterior_summary_rule": "posterior_mean",
        "primary_evaluation_metric": str(primary_metric),
        "evaluation_metrics": deepcopy(metrics_cfg),
        "convergence_required": True,
    }


def _parse_stratify_option(value: Any, *, default: StratifyOption) -> StratifyOption:
    """Normalise config stratify setting into {True, False, 'strict', None}."""

    if value is None:
        return default
    if isinstance(value, str):
        label = value.strip().lower()
        if label in {"", "auto"}:
            return default
        if label in {"strict"}:
            return "strict"
        if label in {"true", "1", "yes"}:
            return True
        if label in {"false", "0", "no"}:
            return False
        return default
    return bool(value)


def _standardization_from_config(cfg: Mapping[str, Any], task: str) -> StandardizationConfig:
    std_cfg = cfg.get("standardization", {}) or {}
    X_method = std_cfg.get("X", "unit_variance")
    if "y_center" in std_cfg:
        y_center = bool(std_cfg.get("y_center"))
    else:
        y_center = task != "classification"
    return StandardizationConfig(X=X_method, y_center=y_center)


def _require_nested_cv(config: Mapping[str, Any]) -> None:
    """Validate that both outer and inner nested CV splits are configured."""

    splits = config.get("splits")
    if not isinstance(splits, Mapping):
        raise ExperimentError(
            "Nested CV is required. Provide both splits.outer and splits.inner with n_splits >= 2."
        )

    def _coerce_splits(section: Optional[Mapping[str, Any]], label: str) -> int:
        if not isinstance(section, Mapping):
            raise ExperimentError(
                f"Nested CV requires splits.{label}.n_splits >= 2, but it was missing."
            )
        value = section.get("n_splits", section.get("folds"))
        try:
            n_splits = int(value)
        except (TypeError, ValueError):
            raise ExperimentError(
                f"splits.{label}.n_splits must be an integer >= 2 for nested CV."
            )
        if n_splits < 2:
            raise ExperimentError(
                f"splits.{label}.n_splits must be >= 2 for nested CV."
            )
        return n_splits

    _coerce_splits(splits.get("outer"), "outer")
    _coerce_splits(splits.get("inner"), "inner")


def _adjust_seeds_for_repeat(cfg: MutableMapping[str, Any], offset: int) -> None:
    if offset == 0:
        return
    inference_cfg = cfg.get("inference")
    if isinstance(inference_cfg, Mapping):
        for section in inference_cfg.values():
            if isinstance(section, MutableMapping) and "seed" in section:
                try:
                    section["seed"] = int(section["seed"]) + offset
                except (TypeError, ValueError):
                    continue
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, MutableMapping) and "seed" in model_cfg:
        try:
            model_cfg["seed"] = int(model_cfg["seed"]) + offset
        except (TypeError, ValueError):
            pass
    runtime_cfg = cfg.get("runtime")
    if isinstance(runtime_cfg, MutableMapping) and "seed" in runtime_cfg:
        try:
            runtime_cfg["seed"] = int(runtime_cfg["seed"]) + offset
        except (TypeError, ValueError):
            pass


def _normalise_binary_labels(y: np.ndarray) -> np.ndarray:
    """Ensure binary labels are encoded as {0, 1}."""
    values = np.unique(y)
    if values.size != 2:
        return y.astype(np.float32, copy=False)
    low, high = float(values[0]), float(values[-1])
    if math.isclose(low, 0.0) and math.isclose(high, 1.0):
        return y.astype(np.float32, copy=False)
    mapped = np.where(np.isclose(y, high), 1.0, 0.0)
    return mapped.astype(np.float32, copy=False)


def _resolve_seed(*candidates: Any) -> Optional[int]:
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _convergence_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    experiments_cfg = config.get("experiments", {}) or {}
    raw = experiments_cfg.get("convergence", {}) or {}
    resolved = dict(DEFAULT_CONVERGENCE_CONFIG)
    if isinstance(raw, Mapping):
        resolved.update(raw)
    resolved["enabled"] = bool(resolved.get("enabled", True))
    resolved["max_rhat"] = float(resolved.get("max_rhat", 1.01))
    resolved["min_ess"] = max(0.0, float(resolved.get("min_ess", 0.0)))
    resolved["max_mcse_over_sd"] = float(resolved.get("max_mcse_over_sd", 1.0))
    resolved["max_retries"] = max(0, int(resolved.get("max_retries", 1)))
    resolved["retry_scale"] = max(1.0, float(resolved.get("retry_scale", 2.0)))
    resolved["missing_policy"] = str(resolved.get("missing_policy", "warn")).strip().lower()
    resolved["require_valid_diagnostics"] = bool(resolved.get("require_valid_diagnostics", True))
    resolved["min_chains_for_rhat"] = max(1, int(resolved.get("min_chains_for_rhat", 4)))
    min_ess_by_block = resolved.get("min_ess_by_block", {}) or {}
    if isinstance(min_ess_by_block, Mapping):
        resolved["min_ess_by_block"] = {
            str(name).lower(): max(0.0, float(value))
            for name, value in min_ess_by_block.items()
            if value is not None
        }
    else:
        resolved["min_ess_by_block"] = {}
    hmc_raw = resolved.get("hmc", {}) or {}
    if not isinstance(hmc_raw, Mapping):
        hmc_raw = {}
    models = hmc_raw.get("models", [])
    if isinstance(models, str):
        models = [models]
    resolved["hmc"] = {
        "enabled": bool(hmc_raw.get("enabled", True)),
        "models": [str(item).lower() for item in models if str(item).strip()],
        "max_divergences": max(0, int(hmc_raw.get("max_divergences", 0))),
        "min_ebfmi": float(hmc_raw.get("min_ebfmi", 0.3)),
        "max_treedepth_hits": max(0, int(hmc_raw.get("max_treedepth_hits", 0))),
        "require_present": bool(hmc_raw.get("require_present", True)),
    }
    parameters = resolved.get("parameters", ["beta", "tau", "phi", "lambda"])
    if isinstance(parameters, str):
        parameters = [parameters]
    resolved["parameters"] = [str(name) for name in parameters]
    expected_blocks = resolved.get("expected_blocks", {}) or {}
    if not isinstance(expected_blocks, Mapping):
        expected_blocks = {}
    resolved["expected_blocks"] = {
        str(name).lower(): [str(param) for param in value]
        for name, value in expected_blocks.items()
        if isinstance(value, (list, tuple))
    }
    return resolved


def _classify_fold_status(status: str) -> bool:
    return not str(status).upper().startswith("INVALID")


def _expected_convergence_blocks(model_name: str, cfg: Mapping[str, Any]) -> List[str]:
    expected = cfg.get("expected_blocks", {}) or {}
    if isinstance(expected, Mapping):
        name_key = str(model_name).lower()
        if isinstance(expected.get(name_key), Sequence):
            return [str(item) for item in expected[name_key]]
        if isinstance(expected.get("default"), Sequence):
            return [str(item) for item in expected["default"]]
    parameters = cfg.get("parameters", []) or []
    return [str(item) for item in parameters]


def _check_convergence(
    summary: Mapping[str, Mapping[str, Any]] | None,
    cfg: Mapping[str, Any],
    *,
    model_name: str,
    sampler_diagnostics: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    if not summary:
        return True, []
    if not bool(cfg.get("enabled", True)):
        return True, []

    max_rhat = float(cfg.get("max_rhat", 1.01))
    min_ess_default = max(0.0, float(cfg.get("min_ess", 0.0)))
    min_ess_by_block = cfg.get("min_ess_by_block", {}) or {}
    max_mcse_over_sd = float(cfg.get("max_mcse_over_sd", 1.0))
    expected_blocks = _expected_convergence_blocks(model_name, cfg)
    missing_policy = str(cfg.get("missing_policy", "warn")).strip().lower()
    require_valid_diagnostics = bool(cfg.get("require_valid_diagnostics", True))
    failures: List[str] = []
    for name in expected_blocks:
        block = summary.get(name)
        if not isinstance(block, Mapping):
            if missing_policy == "fail":
                failures.append(f"missing.{name}")
            continue
        if require_valid_diagnostics and not bool(block.get("diagnostic_valid", False)):
            failures.append(f"{name}.diagnostic_valid=false")
        try:
            rhat_max = float(block.get("rhat_max"))
        except (TypeError, ValueError):
            if "error" in block:
                failures.append(f"{name}.error={block['error']}")
            continue
        if not np.isfinite(rhat_max) or rhat_max > max_rhat:
            failures.append(f"{name}.rhat_max={rhat_max:.3f}>{max_rhat:.3f}")
        try:
            ess_min = float(block.get("ess_min"))
        except (TypeError, ValueError):
            ess_min = np.nan
        min_ess = min_ess_default
        if isinstance(min_ess_by_block, Mapping):
            min_ess = float(min_ess_by_block.get(str(name).lower(), min_ess_default))
        if np.isfinite(min_ess) and min_ess > 0.0:
            if not np.isfinite(ess_min) or ess_min < min_ess:
                failures.append(f"{name}.ess_min={ess_min:.1f}<{min_ess:.1f}")
        try:
            mcse_over_sd_max = float(block.get("mcse_over_sd_max"))
        except (TypeError, ValueError):
            mcse_over_sd_max = np.nan
        if np.isfinite(max_mcse_over_sd) and max_mcse_over_sd > 0.0:
            if not np.isfinite(mcse_over_sd_max) or mcse_over_sd_max > max_mcse_over_sd:
                failures.append(
                    f"{name}.mcse_over_sd_max={mcse_over_sd_max:.3f}>{max_mcse_over_sd:.3f}"
                )

    hmc_cfg = cfg.get("hmc", {}) or {}
    if isinstance(hmc_cfg, Mapping) and bool(hmc_cfg.get("enabled", True)):
        hmc_models = {
            str(item).lower()
            for item in (hmc_cfg.get("models", []) or [])
        }
        if str(model_name).lower() in hmc_models:
            hmc_diag = (
                sampler_diagnostics.get("hmc")
                if isinstance(sampler_diagnostics, Mapping)
                else None
            )
            require_present = bool(hmc_cfg.get("require_present", True))
            if not isinstance(hmc_diag, Mapping):
                if require_present:
                    failures.append("hmc.diagnostics_missing")
            else:
                max_divergences = int(hmc_cfg.get("max_divergences", 0))
                min_ebfmi = float(hmc_cfg.get("min_ebfmi", 0.3))
                max_treedepth_hits = int(hmc_cfg.get("max_treedepth_hits", 0))
                divergences = int(hmc_diag.get("divergences", -1))
                ebfmi_min = float(hmc_diag.get("ebfmi_min", np.nan))
                treedepth_hits = int(hmc_diag.get("treedepth_hits", -1))
                if divergences < 0:
                    failures.append("hmc.divergences_missing")
                elif divergences > max_divergences:
                    failures.append(f"hmc.divergences={divergences}>{max_divergences}")
                if not np.isfinite(ebfmi_min):
                    failures.append("hmc.ebfmi_missing")
                elif ebfmi_min < min_ebfmi:
                    failures.append(f"hmc.ebfmi_min={ebfmi_min:.3f}<{min_ebfmi:.3f}")
                if treedepth_hits < 0:
                    failures.append("hmc.treedepth_hits_missing")
                elif treedepth_hits > max_treedepth_hits:
                    failures.append(f"hmc.treedepth_hits={treedepth_hits}>{max_treedepth_hits}")
    return len(failures) == 0, failures


def _scale_bayesian_runtime(
    model_config: MutableMapping[str, Any],
    inference_cfg: MutableMapping[str, Any],
    scale: float,
) -> None:
    model_name = str(model_config.get("name", "")).lower()

    if model_name in {"grrhs_gibbs", "grrhs_gibbs_logistic", "grrhs_logistic", "grrhs_gibbs_cls", "gigg", "gigg_regression"}:
        if "iters" in model_config:
            model_config["iters"] = max(4, int(math.ceil(float(model_config["iters"]) * scale)))
        gibbs_cfg = inference_cfg.get("gibbs")
        if isinstance(gibbs_cfg, MutableMapping) and "burn_in" in gibbs_cfg:
            gibbs_cfg["burn_in"] = max(2, int(math.ceil(float(gibbs_cfg["burn_in"]) * scale)))
        return

    if model_name in {"horseshoe", "hs", "regularized_horseshoe", "rhs", "regularised_horseshoe"}:
        for key in ("num_warmup", "num_samples"):
            if key in model_config:
                model_config[key] = max(100, int(math.ceil(float(model_config[key]) * scale)))
        return


def _summarize_convergence_compat(
    arrays: Mapping[str, np.ndarray],
    *,
    min_chains_for_rhat: int,
) -> Dict[str, Dict[str, Any]]:
    try:
        return summarize_convergence(arrays, min_chains_for_rhat=min_chains_for_rhat)
    except TypeError:
        # Backward-compatible path for tests or local monkeypatches that still
        # expose the older signature.
        return summarize_convergence(arrays)  # type: ignore[misc]


def _fit_model_with_retry(
    model_config: Mapping[str, Any],
    groups: Sequence[Sequence[int]],
    p: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    C_train: Optional[np.ndarray],
    task: str,
    std_cfg: StandardizationConfig,
    convergence_cfg: Mapping[str, Any],
) -> Tuple[
    Any,
    Dict[str, np.ndarray],
    Optional[Dict[str, Dict[str, Any]]],
    List[Dict[str, Any]],
    Mapping[str, Any],
    Dict[str, Any],
]:
    attempts: List[Dict[str, Any]] = []
    last_model = None
    last_arrays: Dict[str, np.ndarray] = {}
    last_summary: Optional[Dict[str, Dict[str, Any]]] = None
    last_effective_config: Mapping[str, Any] = model_config
    last_sampler_diagnostics: Dict[str, Any] = {}

    fairness_cfg = _bayesian_fairness_config(model_config)
    max_retries = int(convergence_cfg.get("max_retries", 1))
    if _is_bayesian_config(model_config) and bool(fairness_cfg.get("enabled", True)) and bool(
        fairness_cfg.get("disable_budget_retry", True)
    ):
        max_retries = 0
    total_attempts = 1 + max_retries
    scale = float(convergence_cfg.get("retry_scale", 2.0))

    for attempt in range(total_attempts):
        attempt_config = deepcopy(model_config)
        attempt_model_cfg = attempt_config.setdefault("model", {})
        attempt_inference_cfg = attempt_config.setdefault("inference", {})
        _apply_bayesian_sampling_budget(attempt_config)
        if attempt > 0:
            _scale_bayesian_runtime(attempt_model_cfg, attempt_inference_cfg, scale ** attempt)

        _maybe_calibrate_tau(attempt_model_cfg, std_cfg, X_train, y_train, groups, task)
        model = _instantiate_model(attempt_config, groups, p)
        _fit_model_dispatch(model, X_train, y_train, groups=groups, C=C_train)

        arrays = _collect_posterior_arrays(model)
        sampler_diagnostics = _collect_sampler_diagnostics(model)
        summary: Optional[Dict[str, Dict[str, Any]]] = None
        ok = True
        failures: List[str] = []
        if arrays:
            try:
                summary = _summarize_convergence_compat(
                    arrays,
                    min_chains_for_rhat=int(convergence_cfg.get("min_chains_for_rhat", 2)),
                )
            except Exception as exc:  # pragma: no cover - defensive guard for short/invalid chains
                ok = False
                failures = [f"convergence_error={type(exc).__name__}"]
            else:
                ok, failures = _check_convergence(
                    summary,
                    convergence_cfg,
                    model_name=str(attempt_model_cfg.get("name", "")),
                    sampler_diagnostics=sampler_diagnostics,
                )

        attempts.append(
            {
                "attempt": attempt + 1,
                "iters": attempt_model_cfg.get("iters"),
                "burn_in": ((attempt_inference_cfg.get("gibbs") or {}).get("burn_in") if isinstance(attempt_inference_cfg.get("gibbs"), Mapping) else None),
                "num_warmup": attempt_model_cfg.get("num_warmup"),
                "num_samples": attempt_model_cfg.get("num_samples"),
                "converged": ok,
                "failures": failures,
                "sampler_diagnostics": sampler_diagnostics,
            }
        )

        last_model = model
        last_arrays = arrays
        last_summary = summary
        last_effective_config = attempt_config
        last_sampler_diagnostics = sampler_diagnostics
        if ok:
            break

    if last_model is None:
        raise ExperimentError("Model fitting did not produce a fitted model.")
    return last_model, last_arrays, last_summary, attempts, last_effective_config, last_sampler_diagnostics


def _resolve_groups(config: Mapping[str, Any], groups: Optional[Sequence[Sequence[int]]], p: int) -> List[List[int]]:
    """Return explicit group structure for models that require it."""
    if groups:
        return [list(map(int, g)) for g in groups]
    data_cfg = config.get("data", {}) or {}
    if data_cfg.get("groups"):
        return [list(map(int, g)) for g in data_cfg["groups"]]
    G = data_cfg.get("G")
    group_sizes = data_cfg.get("group_sizes")
    return make_groups(int(p), G, group_sizes)


def _override_model_groups(
    true_groups: Sequence[Sequence[int]],
    n_features: int,
    override_cfg: Mapping[str, Any] | None,
    *,
    seed: Optional[int],
) -> Optional[List[List[int]]]:
    """Optionally perturb model-facing group assignments to simulate misspecification."""

    if not override_cfg:
        return None

    mode = str(override_cfg.get("mode", "none")).lower()
    if mode in {"none", "false", "", "off"}:
        return None

    rng_seed = override_cfg.get("seed", seed)
    rng = np.random.default_rng(None if rng_seed is None else int(rng_seed))

    assignments = np.full(int(n_features), -1, dtype=int)
    for gid, members in enumerate(true_groups):
        assignments[np.asarray(members, dtype=int)] = gid

    missing = np.where(assignments < 0)[0]
    if missing.size > 0 and len(true_groups) > 0:
        filler = rng.integers(0, len(true_groups), size=missing.size)
        assignments[missing] = filler

    fraction = float(override_cfg.get("fraction", 1.0))
    fraction = min(max(fraction, 0.0), 1.0)
    total = assignments.size
    if fraction <= 0.0 or len(true_groups) <= 1:
        indices_to_shuffle = np.array([], dtype=int)
    elif fraction >= 1.0:
        indices_to_shuffle = np.arange(total)
    else:
        count = max(1, int(round(fraction * total)))
        indices_to_shuffle = rng.choice(total, size=count, replace=False)

    group_count = len(true_groups)
    for idx in indices_to_shuffle:
        original = assignments[idx]
        if original < 0:
            continue
        new_gid = original
        attempts = 0
        while new_gid == original and attempts < 10:
            new_gid = int(rng.integers(0, group_count))
            attempts += 1
        if new_gid == original:
            new_gid = (original + 1) % group_count
        assignments[idx] = new_gid

    counts = np.bincount(assignments, minlength=group_count)
    for gid in range(group_count):
        if counts[gid] > 0:
            continue
        donors = np.where(counts > 1)[0]
        if donors.size == 0:
            continue
        donor = int(rng.choice(donors))
        donor_indices = np.where(assignments == donor)[0]
        donor_choice = int(rng.choice(donor_indices))
        assignments[donor_choice] = gid
        counts[donor] -= 1
        counts[gid] += 1

    overridden = [[] for _ in range(group_count)]
    for feat_idx, gid in enumerate(assignments):
        overridden[int(gid)].append(int(feat_idx))
    return overridden


def _prepare_dataset_bundle(
    config: Mapping[str, Any],
    *,
    task: str,
    repeat_index: int,
) -> Dict[str, Any]:
    """Generate or load dataset according to configuration."""

    data_cfg = deepcopy(config.get("data", {}))
    data_type = str(data_cfg.get("type", "synthetic")).lower()
    seeds_cfg = config.get("seeds", {}) or {}

    def _candidate_seed(values: Iterable[Any]) -> Optional[int]:
        for val in values:
            if val is None:
                continue
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
        return None

    base_seed = _candidate_seed(
        (
            data_cfg.get("seed"),
            seeds_cfg.get("data_generation"),
            config.get("seed"),
            seeds_cfg.get("experiment"),
        )
    )
    if data_type == "synthetic":
        if base_seed is not None:
            dataset_seed = int(base_seed) + int(repeat_index)
        else:
            dataset_seed = repeat_index if repeat_index > 0 else None
    else:
        dataset_seed = base_seed

    if data_type == "synthetic":
        response_override: Optional[Mapping[str, Any]] = None
        if task == "classification":
            override: Dict[str, Any] = {}
            classification_cfg = data_cfg.get("classification")
            if isinstance(classification_cfg, Mapping):
                override.update(classification_cfg)
            classification_global = config.get("classification")
            if isinstance(classification_global, Mapping):
                override.update(classification_global)
            if override:
                response_override = override
        synth_cfg = synthetic_config_from_dict(
            data_cfg,
            seed=dataset_seed,
            name=config.get("name"),
            task=task,
            response_override=response_override,
        )
        generated = generate_synthetic(synth_cfg)
        X = np.asarray(generated.X, dtype=np.float32, copy=False)
        y = np.asarray(generated.y, dtype=np.float32, copy=False).reshape(-1)
        if task == "classification":
            y = _normalise_binary_labels(y)
        groups = [list(map(int, g)) for g in generated.groups]
        override_cfg = data_cfg.get("model_groups_override") or data_cfg.get("group_override")
        model_groups = _override_model_groups(groups, X.shape[1], override_cfg, seed=dataset_seed)
        if model_groups is None:
            model_groups = groups
        bundle = {
            "X": X,
            "C": None,
            "y": y,
            "beta": None if generated.beta is None else np.asarray(generated.beta, dtype=np.float32),
            "groups": groups,
            "model_groups": model_groups,
            "feature_names": None,
            "covariate_feature_names": None,
            "metadata": {
                "type": "synthetic",
                "seed": synth_cfg.seed,
                "name": synth_cfg.name,
                "noise_sigma": generated.noise_sigma,
                "info": generated.info,
            },
        }
        return bundle

    if data_type == "loader":
        loader_cfg = data_cfg.get("loader")
        if not isinstance(loader_cfg, Mapping):
            raise ExperimentError("data.type=loader requires a 'loader' mapping in config.")
        io_cfg = config.get("io", {}) or {}
        base_dir_value = data_cfg.get("base_dir") or io_cfg.get("base_dir")
        base_dir = Path(base_dir_value).expanduser().resolve() if base_dir_value else None
        loaded: LoadedDataset = load_real_dataset(loader_cfg, base_dir=base_dir)
        if loaded.y is None:
            raise ExperimentError("Loader dataset must provide targets (loader.path_y) for supervised tasks.")
        X = np.asarray(loaded.X, dtype=np.float32, copy=False)
        y = np.asarray(loaded.y, dtype=np.float32, copy=False).reshape(-1)
        if task == "classification":
            y = _normalise_binary_labels(y)
        groups = _resolve_groups(config, loaded.groups, X.shape[1])
        override_cfg = data_cfg.get("model_groups_override") or data_cfg.get("group_override")
        model_groups = _override_model_groups(groups, X.shape[1], override_cfg, seed=dataset_seed)
        if model_groups is None:
            model_groups = groups
        bundle = {
            "X": X,
            "C": None if loaded.C is None else np.asarray(loaded.C, dtype=np.float32, copy=False),
            "y": y,
            "beta": None if loaded.beta is None else np.asarray(loaded.beta, dtype=np.float32),
            "groups": groups,
            "model_groups": model_groups,
            "feature_names": loaded.feature_names,
            "covariate_feature_names": loaded.covariate_feature_names,
            "metadata": {
                "type": "loader",
                "seed": dataset_seed,
                "paths": {k: loader_cfg[k] for k in sorted(loader_cfg.keys()) if isinstance(k, str)},
                "feature_names_path": loader_cfg.get("path_feature_names"),
                "group_map_path": loader_cfg.get("path_group_map"),
            },
        }
        return bundle

    raise ExperimentError(f"Unsupported data.type '{data_type}'.")


def _prepare_outer_folds(
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    task: str,
    repeat_index: int,
) -> List[OuterFold]:
    """Create outer cross-validation folds according to configuration."""

    outer_cfg = config.get("splits", {}).get("outer", {}) or {}
    mode = str(outer_cfg.get("mode", "kfold")).lower()
    n_total = int(dataset["X"].shape[0])

    if mode in {"holdout", "hold-out", "train_test", "train-test"}:
        train_size = outer_cfg.get("train_size")
        test_size = outer_cfg.get("test_size")
        if train_size is None or test_size is None:
            raise ExperimentError("holdout mode requires 'train_size' and 'test_size'.")
        n_repeats = int(outer_cfg.get("n_repeats", outer_cfg.get("repeats", 1)) or 1)
        shuffle = bool(outer_cfg.get("shuffle", True))
        seed = outer_cfg.get("seed")
        if seed is not None:
            seed = int(seed) + repeat_index
        return holdout_splits(
            n=n_total,
            train_size=int(train_size),
            test_size=int(test_size),
            n_repeats=n_repeats,
            seed=seed,
            shuffle=shuffle,
        )

    n_splits_raw = outer_cfg.get("n_splits", outer_cfg.get("folds"))
    try:
        n_splits = int(n_splits_raw)
    except (TypeError, ValueError):
        raise ExperimentError("splits.outer.n_splits must be provided for nested CV.")
    if n_splits < 2:
        raise ExperimentError("splits.outer.n_splits must be >= 2 for nested CV.")

    n_repeats = int(outer_cfg.get("n_repeats", outer_cfg.get("repeats", 1)) or 1)
    shuffle = bool(outer_cfg.get("shuffle", True))

    default_outer_strat = True if task == "classification" else False
    stratify = _parse_stratify_option(outer_cfg.get("stratify"), default=default_outer_strat)

    seed = outer_cfg.get("seed")
    if seed is not None:
        seed = int(seed) + repeat_index

    y_for_strat = None
    if task == "classification" and stratify is not False:
        y_for_strat = np.asarray(dataset["y"], dtype=int)

    folds = outer_kfold_splits(
        n=n_total,
        y=y_for_strat,
        task=task,
        n_splits=n_splits,
        n_repeats=n_repeats,
        shuffle=shuffle,
        seed=seed,
        stratify=stratify,
    )
    return folds


def _instantiate_model(config: Mapping[str, Any], groups: Sequence[Sequence[int]], p: int) -> Any:
    cfg = deepcopy(config)
    _apply_bayesian_sampling_budget(cfg)
    cfg.setdefault("data", {})
    cfg["data"]["groups"] = [list(map(int, g)) for g in groups]
    cfg["data"]["p"] = int(p)
    return build_from_config(cfg)


def _maybe_calibrate_tau(
    model_cfg: MutableMapping[str, Any],
    std_cfg: StandardizationConfig,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    task: str,
) -> None:
    """Calibrate τ_0 using expected effective sparsity if requested."""

    tau_cfg = model_cfg.get("tau")
    if not isinstance(tau_cfg, Mapping):
        return

    mode = str(tau_cfg.get("mode", "")).lower()
    if mode == "fixed":
        value = tau_cfg.get("value", model_cfg.get("tau0"))
        if value is not None:
            model_cfg["tau0"] = float(value)
        return
    if mode != "calibrated":
        return

    if isinstance(tau_cfg.get("p0"), Mapping):
        p0_value = tau_cfg["p0"].get("value")
        p0_grid = tau_cfg["p0"].get("grid")
    else:
        p0_value = tau_cfg.get("p0")
        p0_grid = tau_cfg.get("p0_grid") or tau_cfg.get("grid")

    if p0_value is not None:
        candidates = [float(p0_value)]
    elif p0_grid:
        candidates = [float(v) for v in p0_grid]
    else:
        candidates = [float(min(max(len(groups), 1), 5))]

    target = str(tau_cfg.get("target", "groups")).lower()
    D = len(groups) if target == "groups" else X.shape[1]
    if D <= 0:
        raise ExperimentError("Cannot calibrate τ: feature/group dimension is zero.")

    if task == "classification":
        sigma_classification = tau_cfg.get("sigma_classification", "auto")
        if sigma_classification in {None, "auto"}:
            y_bin = np.asarray(y, dtype=float).reshape(-1)
            if y_bin.size == 0:
                sigma_ref = 2.0
            else:
                p_hat = float(np.mean(y_bin))
                p_hat = float(min(max(p_hat, 1e-6), 1.0 - 1e-6))
                sigma_ref = float(1.0 / math.sqrt(p_hat * (1.0 - p_hat)))
        else:
            sigma_ref = float(sigma_classification)
    else:
        sigma_reference = tau_cfg.get("sigma_reference", 1.0)
        if sigma_reference in {None, "auto"}:
            y_scale = float(np.std(y, ddof=1)) if y.size else 1.0
            sigma_ref = y_scale
        else:
            sigma_ref = float(sigma_reference)

    n = X.shape[0]
    p0 = max(1.0, candidates[0])
    tau0 = (p0 / max(D - p0, 1e-8)) * (sigma_ref / math.sqrt(max(n, 1)))
    model_cfg["tau0"] = float(max(tau0, 1e-8))


def _compute_inner_metric(
    task: str,
    y_true: np.ndarray,
    model: Any,
    X_val: np.ndarray,
    *,
    class_labels: Optional[np.ndarray] = None,
) -> float:
    """Return scalar metric used for inner CV model selection."""
    if task == "classification":
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_val)
            if isinstance(prob, np.ndarray):
                prob = prob[:, -1] if prob.ndim == 2 else prob
        else:
            preds = model.predict(X_val)
            prob = 1.0 / (1.0 + np.exp(-preds))
        prob = np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)
        from sklearn.metrics import log_loss

        log_loss_kwargs: Dict[str, Any] = {}
        if class_labels is not None and class_labels.size >= 2:
            log_loss_kwargs["labels"] = class_labels
        return float(log_loss(y_true, prob, **log_loss_kwargs))

    from sklearn.metrics import mean_squared_error

    preds = model.predict(X_val)
    return float(mean_squared_error(y_true, preds))


def _compute_lasso_lambda_max(X: np.ndarray, y: np.ndarray) -> float:
    """Return λ_max for standardised Lasso path."""
    if X.size == 0 or y.size == 0:
        return 1.0
    lam = float(np.max(np.abs(X.T @ y))) / max(X.shape[0], 1)
    return max(lam, 1e-8)


def _set_nested_config_value(root: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """Assign a value into a nested mapping using dotted path notation."""
    parts = [part for part in str(dotted_key).split(".") if part]
    if not parts:
        raise ExperimentError("Search parameter name cannot be empty.")
    cursor: MutableMapping[str, Any] = root
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, MutableMapping):
            current = {}
            cursor[part] = current
        cursor = current
    cursor[parts[-1]] = value


def _build_candidate_config(base_config: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply candidate hyper-parameters, including dotted nested paths, to a config."""
    candidate_cfg = deepcopy(base_config)
    model_cfg = candidate_cfg.setdefault("model", {})
    if not isinstance(model_cfg, MutableMapping):
        raise ExperimentError("Expected config['model'] to be a mapping.")
    for key, value in candidate.items():
        if "." in str(key):
            _set_nested_config_value(model_cfg, str(key), value)
        else:
            model_cfg[str(key)] = value
    model_cfg.pop("search", None)
    return candidate_cfg


def _apply_mapping_overrides(root: MutableMapping[str, Any], overrides: Mapping[str, Any], prefix: str = "") -> None:
    """Recursively apply nested mapping overrides using dotted-path assignment."""
    for key, value in overrides.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _apply_mapping_overrides(root, value, prefix=path)
        else:
            _set_nested_config_value(root, path, value)


def _expand_search_values(values: Any, key: str, X_ref: np.ndarray, y_ref: np.ndarray) -> List[float]:
    """Expand declarative search specifications into explicit numeric grids."""
    if isinstance(values, Mapping):
        mode = str(values.get("mode", "")).lower()
        if mode in {"logspace", "geomspace"}:
            start = float(values.get("start", values.get("high", 1.0)))
            stop = float(values.get("stop", values.get("low", 1e-3)))
            num = int(values.get("num", values.get("points", 10)))
            if num <= 0:
                raise ExperimentError(f"search grid for '{key}' requires a positive 'num'.")
            if start <= 0 or stop <= 0:
                raise ExperimentError(f"logspace search for '{key}' requires positive bounds.")
            return list(np.geomspace(start, stop, num))
        if mode in {"lasso_path", "lasso"}:
            min_ratio = float(values.get("min_ratio", values.get("ratio", 1e-3)))
            num = int(values.get("num", values.get("points", 50)))
            if num <= 0:
                raise ExperimentError(f"lasso_path search for '{key}' requires a positive 'num'.")
            lam_max = _compute_lasso_lambda_max(X_ref, y_ref)
            stop = lam_max * max(min_ratio, 1e-6)
            return list(np.geomspace(lam_max, stop, num))
        raise ExperimentError(f"Unsupported search mode '{mode}' for '{key}'.")
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    raise ExperimentError(f"search grid for '{key}' must be a list or mapping.")


def _split_search_config(search_cfg: Mapping[str, Any]) -> Tuple[str, Mapping[str, Any], Dict[str, Any]]:
    """Separate search strategy, parameter space, and control options."""
    if "strategy" not in search_cfg:
        return "grid", search_cfg, {}

    strategy = str(search_cfg.get("strategy", "grid")).strip().lower()
    space = search_cfg.get("space")
    if not isinstance(space, Mapping) or not space:
        raise ExperimentError("model.search.space must be a non-empty mapping when strategy is specified.")
    controls = {
        key: deepcopy(value)
        for key, value in search_cfg.items()
        if key not in {"strategy", "space"}
    }
    return strategy, space, controls


def _coerce_search_float(value: Any, *, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentError(f"Search parameter '{key}' must be numeric; received {value!r}.") from exc


def _build_bayes_dimension(spec: Any, key: str, X_ref: np.ndarray, y_ref: np.ndarray) -> Dict[str, Any]:
    """Normalize one BO search-dimension specification."""
    if isinstance(spec, (list, tuple)):
        choices = [_coerce_search_float(value, key=key) for value in spec]
        if not choices:
            raise ExperimentError(f"Bayesian search choices for '{key}' cannot be empty.")
        return {"kind": "choice", "choices": choices}
    if not isinstance(spec, Mapping):
        raise ExperimentError(f"Bayesian search spec for '{key}' must be a mapping or list.")

    mode = str(spec.get("mode", "uniform")).strip().lower()
    if mode in {"lasso_path", "lasso"}:
        lam_max = _compute_lasso_lambda_max(X_ref, y_ref)
        min_ratio = float(spec.get("min_ratio", spec.get("ratio", 1e-3)))
        low = lam_max * max(min_ratio, 1e-6)
        high = lam_max
        return {"kind": "float", "scale": "log", "low": float(min(low, high)), "high": float(max(low, high))}
    if mode in {"logspace", "geomspace", "loguniform"}:
        low = spec.get("low", spec.get("stop", spec.get("min", None)))
        high = spec.get("high", spec.get("start", spec.get("max", None)))
        if low is None or high is None:
            raise ExperimentError(f"Bayesian log-space search for '{key}' requires 'low'/'high' bounds.")
        low_val = _coerce_search_float(low, key=key)
        high_val = _coerce_search_float(high, key=key)
        if low_val <= 0 or high_val <= 0:
            raise ExperimentError(f"Bayesian log-space search for '{key}' requires positive bounds.")
        return {"kind": "float", "scale": "log", "low": float(min(low_val, high_val)), "high": float(max(low_val, high_val))}
    if mode in {"uniform", "float", "linear"}:
        low = spec.get("low", spec.get("min", None))
        high = spec.get("high", spec.get("max", None))
        if low is None or high is None:
            raise ExperimentError(f"Bayesian search for '{key}' requires 'low'/'high' bounds.")
        low_val = _coerce_search_float(low, key=key)
        high_val = _coerce_search_float(high, key=key)
        return {"kind": "float", "scale": "linear", "low": float(min(low_val, high_val)), "high": float(max(low_val, high_val))}
    if mode in {"int", "integer"}:
        low = spec.get("low", spec.get("min", None))
        high = spec.get("high", spec.get("max", None))
        if low is None or high is None:
            raise ExperimentError(f"Bayesian integer search for '{key}' requires 'low'/'high' bounds.")
        low_val = int(round(_coerce_search_float(low, key=key)))
        high_val = int(round(_coerce_search_float(high, key=key)))
        if low_val > high_val:
            low_val, high_val = high_val, low_val
        return {"kind": "int", "scale": "linear", "low": low_val, "high": high_val}
    raise ExperimentError(f"Unsupported Bayesian search mode '{mode}' for '{key}'.")


def _sample_dimension(spec: Mapping[str, Any], rng: np.random.Generator, *, n_samples: int) -> np.ndarray:
    """Sample candidate values directly in parameter space."""
    kind = str(spec["kind"])
    if kind == "choice":
        choices = np.asarray(spec["choices"], dtype=float)
        idx = rng.integers(0, len(choices), size=n_samples)
        return choices[idx]
    low = float(spec["low"])
    high = float(spec["high"])
    if kind == "int":
        return rng.integers(int(low), int(high) + 1, size=n_samples).astype(float)
    if str(spec.get("scale", "linear")) == "log":
        return np.exp(rng.uniform(np.log(low), np.log(high), size=n_samples))
    return rng.uniform(low, high, size=n_samples)


def _transform_dimension_values(values: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    """Map parameter values to a smoother space for GP fitting."""
    arr = np.asarray(values, dtype=float)
    if str(spec["kind"]) == "choice":
        choices = np.asarray(spec["choices"], dtype=float)
        return np.searchsorted(choices, arr).astype(float)
    if str(spec.get("scale", "linear")) == "log":
        return np.log(np.clip(arr, 1e-12, None))
    return arr


def _decode_dimension_value(value: float, spec: Mapping[str, Any]) -> Any:
    """Map one proposed value back to parameter space."""
    kind = str(spec["kind"])
    if kind == "choice":
        choices = list(spec["choices"])
        idx = int(np.clip(round(float(value)), 0, len(choices) - 1))
        return float(choices[idx])
    if kind == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        return int(np.clip(round(float(value)), low, high))
    return float(np.clip(value, float(spec["low"]), float(spec["high"])))


def _candidate_signature(candidate: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Hashable representation used to avoid duplicate BO evaluations."""
    return tuple(sorted((str(key), _to_serializable(value)) for key, value in candidate.items()))


def _evaluate_inner_candidate(
    base_config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    C: Optional[np.ndarray],
    task: str,
    std_cfg: StandardizationConfig,
    inner_folds: Sequence[OuterFold],
    class_labels: Optional[np.ndarray],
    runtime_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate one inner-CV candidate and return score plus diagnostics."""
    fold_scores: List[float] = []
    skipped_folds = 0

    model_name = _resolve_model_name(base_config)
    use_joint_covariates = _uses_joint_covariates_model_name(model_name)
    for inner_fold in inner_folds:
        train_idx = inner_fold.train
        val_idx = inner_fold.test

        std_train = apply_standardization(X[train_idx], y[train_idx], std_cfg)
        X_train = std_train.X
        y_train = std_train.y
        X_val = apply_standardizer(X[val_idx], x_mean=std_train.x_mean, x_scale=std_train.x_scale)
        if std_cfg.y_center:
            y_val = apply_y_mean(y[val_idx], mean=std_train.y_mean)
        else:
            y_val = np.asarray(y[val_idx], dtype=np.float32)

        C_train_model = None
        if C is not None:
            C_train_raw = np.asarray(C[train_idx], dtype=np.float32)
            C_val_raw = np.asarray(C[val_idx], dtype=np.float32)
            C_train, C_val, _, _, _ = _standardize_covariates_train_test(C_train_raw, C_val_raw)
            C_train_model = C_train
            if not use_joint_covariates:
                X_train, X_val, _ = _residualize_against_covariates(X_train, X_val, C_train, C_val)
                y_train, y_val, _ = _residualize_against_covariates(y_train, y_val, C_train, C_val)

        if task == "classification":
            unique_classes = np.unique(y_train)
            if unique_classes.size < 2:
                skipped_folds += 1
                continue

        candidate_cfg = _build_candidate_config(base_config, candidate)
        if runtime_overrides:
            _apply_mapping_overrides(candidate_cfg, runtime_overrides)
        _maybe_calibrate_tau(candidate_cfg["model"], std_cfg, X_train, y_train, groups, task)
        model = _instantiate_model(candidate_cfg, groups, X.shape[1])
        _fit_model_dispatch(model, X_train, y_train, groups=groups, C=C_train_model if use_joint_covariates else None)

        score = _compute_inner_metric(task, y_val, model, X_val, class_labels=class_labels)
        fold_scores.append(score)

    avg_score = math.inf if not fold_scores else float(np.mean(fold_scores))
    result: Dict[str, Any] = {"params": dict(candidate), "score": avg_score}
    if skipped_folds:
        result["skipped_folds"] = skipped_folds
    return result


def _perform_inner_bayes_opt(
    base_config: Mapping[str, Any],
    search_space: Mapping[str, Any],
    controls: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    C: Optional[np.ndarray],
    task: str,
    std_cfg: StandardizationConfig,
    inner_folds: Sequence[OuterFold],
    class_labels: Optional[np.ndarray],
    X_ref: np.ndarray,
    y_ref: np.ndarray,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Low-dimensional GP-based Bayesian optimization for expensive models."""
    if not _HAS_SKLEARN_GP:
        raise ExperimentError("Bayesian optimization requires scikit-learn GaussianProcessRegressor.")

    dims = {key: _build_bayes_dimension(spec, key, X_ref, y_ref) for key, spec in search_space.items()}
    if len(dims) == 0:
        return {}, []
    if len(dims) > 4:
        raise ExperimentError("Bayesian optimization in this runner supports up to 4 search dimensions.")

    runtime_cfg = base_config.get("runtime", {})
    search_seed = _resolve_seed(
        controls.get("seed"),
        base_config.get("seed"),
        runtime_cfg.get("seed") if isinstance(runtime_cfg, Mapping) else None,
    )
    rng = np.random.default_rng(search_seed)
    keys = list(sorted(dims.keys()))
    budget = max(1, int(controls.get("budget", max(8, 3 * len(keys)))))
    init_points = max(2, int(controls.get("init_points", min(budget, 2 * len(keys) + 1))))
    random_candidates = max(64, int(controls.get("random_candidates", 256)))

    history: List[Dict[str, Any]] = []
    evaluated: Dict[Tuple[Tuple[str, Any], ...], Dict[str, Any]] = {}
    runtime_overrides = controls.get("runtime_overrides")
    if runtime_overrides is not None and not isinstance(runtime_overrides, Mapping):
        raise ExperimentError("model.search.runtime_overrides must be a mapping when provided.")

    def sample_candidate() -> Dict[str, Any]:
        candidate: Dict[str, Any] = {}
        for key in keys:
            raw = _sample_dimension(dims[key], rng, n_samples=1)[0]
            candidate[key] = _decode_dimension_value(raw, dims[key])
        return candidate

    def evaluate(candidate: Dict[str, Any], *, stage: str) -> Dict[str, Any]:
        signature = _candidate_signature(candidate)
        if signature in evaluated:
            return evaluated[signature]
        result = _evaluate_inner_candidate(
            base_config,
            candidate,
            X,
            y,
            groups,
            C=C,
            task=task,
            std_cfg=std_cfg,
            inner_folds=inner_folds,
            class_labels=class_labels,
            runtime_overrides=runtime_overrides if isinstance(runtime_overrides, Mapping) else None,
        )
        result["stage"] = stage
        evaluated[signature] = result
        history.append(result)
        return result

    init_target = min(init_points, budget)
    init_attempts = 0
    while len(history) < init_target and init_attempts < max(16, 8 * init_target):
        before = len(history)
        evaluate(sample_candidate(), stage="init")
        init_attempts += 1
        if len(history) == before and len(evaluated) == before:
            continue

    while len(history) < budget:
        X_obs = np.column_stack(
            [
                _transform_dimension_values(
                    np.asarray([entry["params"][key] for entry in history], dtype=float),
                    dims[key],
                )
                for key in keys
            ]
        )
        y_obs = np.asarray([float(entry["score"]) for entry in history], dtype=float)

        kernel = Matern(length_scale=np.ones(len(keys)), nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=search_seed,
        )
        gp.fit(X_obs, y_obs)

        raw_candidates = {key: _sample_dimension(dims[key], rng, n_samples=random_candidates) for key in keys}
        X_pool = np.column_stack([_transform_dimension_values(raw_candidates[key], dims[key]) for key in keys])
        mean, std = gp.predict(X_pool, return_std=True)
        std = np.maximum(std, 1e-12)
        best_so_far = float(np.min(y_obs))
        z = (best_so_far - mean) / std
        expected_improvement = (best_so_far - mean) * norm.cdf(z) + std * norm.pdf(z)

        proposal: Optional[Dict[str, Any]] = None
        for idx in np.argsort(-expected_improvement):
            candidate = {key: _decode_dimension_value(raw_candidates[key][idx], dims[key]) for key in keys}
            if _candidate_signature(candidate) not in evaluated:
                proposal = candidate
                break
        if proposal is None:
            retry_limit = max(16, random_candidates // 4)
            for _ in range(retry_limit):
                candidate = sample_candidate()
                if _candidate_signature(candidate) not in evaluated:
                    proposal = candidate
                    break
        if proposal is None:
            break
        evaluate(proposal, stage="bayes")

    best_entry = min(history, key=lambda entry: float(entry["score"]))
    return dict(best_entry["params"]), history


def _perform_inner_cv(
    base_config: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    C: Optional[np.ndarray] = None,
    task: str,
    std_cfg: StandardizationConfig,
) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    """Select inner-CV hyper-parameters via grid search or low-dimensional BO."""

    search_cfg = deepcopy(base_config.get("model", {}).get("search"))
    if search_cfg and _is_bayesian_config(base_config):
        fairness_cfg = _bayesian_fairness_config(base_config)
        if bool(fairness_cfg.get("enabled", True)) and bool(fairness_cfg.get("disable_inner_cv", True)):
            return {}, [{"reason": "disabled_for_bayesian_fairness"}]
    if not isinstance(search_cfg, Mapping) or not search_cfg:
        return {}, None

    std_all = apply_standardization(X, y, std_cfg)
    X_ref = std_all.X
    y_ref = std_all.y

    strategy, param_space, controls = _split_search_config(search_cfg)

    inner_cfg = deepcopy(base_config.get("splits", {}).get("inner", {}) or {})
    inner_splits = max(2, int(inner_cfg.get("n_splits", 5) or 5))
    inner_seed = inner_cfg.get("seed")
    default_inner_strat = True if task == "classification" else False
    inner_stratify = _parse_stratify_option(inner_cfg.get("stratify"), default=default_inner_strat)
    inner_folds = outer_kfold_splits(
        n=X.shape[0],
        y=y if (task == "classification") else None,
        task=task,
        n_splits=inner_splits,
        n_repeats=1,
        shuffle=bool(inner_cfg.get("shuffle", True)),
        seed=inner_seed,
        stratify=inner_stratify,
    )

    class_labels: Optional[np.ndarray] = None
    if task == "classification":
        labels = np.unique(np.asarray(y, dtype=float))
        if labels.size < 2:
            raise ExperimentError(
                "Classification inner CV requires at least two classes overall."
            )
        class_labels = labels

    if strategy == "bayes":
        best_candidate, history = _perform_inner_bayes_opt(
            base_config,
            param_space,
            controls,
            X,
            y,
            groups,
            C=C,
            task=task,
            std_cfg=std_cfg,
            inner_folds=inner_folds,
            class_labels=class_labels,
            X_ref=X_ref,
            y_ref=y_ref,
        )
        return best_candidate, history
    if strategy != "grid":
        raise ExperimentError(f"Unsupported search strategy '{strategy}'.")

    keys = sorted(param_space.keys())
    grid_values: List[List[float]] = []
    for key in keys:
        values = param_space[key]
        expanded = _expand_search_values(values, key, X_ref, y_ref)
        grid_values.append(expanded)

    history: List[Dict[str, Any]] = []
    best_candidate: Optional[Dict[str, float]] = None
    best_score = math.inf

    for candidate_values in np.array(np.meshgrid(*grid_values, indexing="ij")).T.reshape(-1, len(keys)):
        candidate = {key: float(val) for key, val in zip(keys, candidate_values)}
        history_entry = _evaluate_inner_candidate(
            base_config,
            candidate,
            X,
            y,
            groups,
            C=C,
            task=task,
            std_cfg=std_cfg,
            inner_folds=inner_folds,
            class_labels=class_labels,
        )
        history_entry["stage"] = "grid"
        history.append(history_entry)
        avg_score = float(history_entry["score"])
        if avg_score < best_score:
            best_score = avg_score
            best_candidate = candidate

    if best_candidate is None:
        return {}, history
    return best_candidate, history


def _collect_posterior_arrays(model: Any) -> Dict[str, np.ndarray]:
    """Collect posterior sample arrays exposed by fitted models if available."""
    arrays: Dict[str, np.ndarray] = {}
    attr_map = {
        "coef_samples_": "beta",
        "alpha_samples_": "alpha",
        "intercept_samples_": "intercept",
        "sigma_samples_": "sigma",
        "sigma2_samples_": "sigma2",
        "tau_samples_": "tau",
        "phi_samples_": "phi",
        "gamma_samples_": "gamma",
        "lambda_samples_": "lambda",
        "lambda_group_samples_": "group_lambda",
        "b_samples_": "b",
        "c_samples_": "c",
        "loglik_samples_": "loglik",
    }
    for attr, key in attr_map.items():
        value = getattr(model, attr, None)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.size == 0:
            continue
        scalar_keys = {"sigma", "sigma2", "tau", "tau2", "c", "beta0", "intercept"}
        if key not in scalar_keys and arr.ndim == 2:
            ref_map = {
                "beta": ("coef_mean_", "coef_"),
                "lambda": ("lambda_mean_",),
                "group_lambda": ("group_lambda_mean_", "lambda_group_mean_", "phi_mean_"),
                "phi": ("phi_mean_",),
                "gamma": ("gamma_mean_",),
                "b": ("b_mean_",),
                "alpha": ("alpha_mean_",),
            }
            param_len = None
            for attr_name in ref_map.get(key, ()):
                ref = getattr(model, attr_name, None)
                if ref is None:
                    continue
                ref_arr = np.asarray(ref).reshape(-1)
                if ref_arr.size > 0:
                    param_len = int(ref_arr.size)
                    break
            if param_len is not None:
                # Normalize common (params, draws) layout to (draws, params).
                if arr.shape[0] == param_len and arr.shape[1] != param_len:
                    arr = arr.T
        arrays[key] = arr
    return arrays


def _collect_sampler_diagnostics(model: Any) -> Dict[str, Any]:
    diagnostics = getattr(model, "sampler_diagnostics_", None)
    if not isinstance(diagnostics, Mapping):
        return {}
    return {str(key): value for key, value in diagnostics.items()}


def _flatten_scalar_draws(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    data = np.asarray(arr, dtype=float)
    return data.reshape(-1)


def _flatten_param_draws(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    data = np.asarray(arr, dtype=float)
    if data.ndim == 0:
        return data.reshape(1, 1)
    if data.ndim == 1:
        return data.reshape(1, -1)
    if data.ndim == 2:
        return data
    return data.reshape(-1, *data.shape[2:])


def _offset_all_seeds(cfg: MutableMapping[str, Any], *, offset: int) -> None:
    if offset == 0:
        return
    stack: List[MutableMapping[str, Any]] = [cfg]
    while stack:
        node = stack.pop()
        for key, value in node.items():
            if isinstance(value, MutableMapping):
                stack.append(value)
                continue
            if isinstance(key, str) and key.lower() == "seed":
                try:
                    node[key] = int(value) + int(offset)
                except (TypeError, ValueError):
                    continue


def _posterior_validation_failures(
    result: Mapping[str, Any],
) -> List[str]:
    failures = []
    if not isinstance(result, Mapping):
        return failures
    for key in ("sbc", "ppc", "seed_stability"):
        block = result.get(key)
        if not isinstance(block, Mapping):
            continue
        if str(block.get("status", "")).lower() == "fail":
            for reason in block.get("reasons", []) or []:
                failures.append(f"{key}.{reason}")
            if not block.get("reasons"):
                failures.append(f"{key}.failed")
    return failures


def _run_sbc_validation(
    *,
    beta_truth: Optional[np.ndarray],
    posterior_arrays: Mapping[str, np.ndarray],
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": "skip", "reasons": []}
    if beta_truth is None:
        if bool(cfg.get("fail_on_missing_truth", False)):
            out["status"] = "fail"
            out["reasons"] = ["missing_beta_truth"]
        else:
            out["reasons"] = ["missing_beta_truth"]
        return out
    beta_draws = _flatten_param_draws(
        None if "beta" not in posterior_arrays else np.asarray(posterior_arrays["beta"], dtype=float)
    )
    if beta_draws is None or beta_draws.size == 0:
        if bool(cfg.get("fail_on_missing_draws", True)):
            out["status"] = "fail"
            out["reasons"] = ["missing_beta_draws"]
        else:
            out["reasons"] = ["missing_beta_draws"]
        return out
    truth = np.asarray(beta_truth, dtype=float).reshape(-1)
    if beta_draws.ndim != 2 or beta_draws.shape[1] != truth.shape[0]:
        out["status"] = "fail"
        out["reasons"] = [f"beta_shape_mismatch:{beta_draws.shape} vs {truth.shape}"]
        return out

    draws = beta_draws.shape[0]
    if draws < 20:
        out["status"] = "fail"
        out["reasons"] = [f"insufficient_draws:{draws}"]
        return out

    abs_truth = np.abs(truth)
    order = np.argsort(-abs_truth)
    min_coeffs = int(cfg.get("min_coefficients", 8))
    max_coeffs = int(cfg.get("max_coefficients", 128))
    use_count = min(max(min_coeffs, min(max_coeffs, truth.shape[0])), truth.shape[0])
    idx = order[:use_count]
    selected_draws = beta_draws[:, idx]
    selected_truth = truth[idx]

    less_counts = np.sum(selected_draws < selected_truth[None, :], axis=0)
    ranks = (less_counts + 1.0) / (draws + 1.0)
    ks = kstest(ranks, "uniform")
    level = float(cfg.get("coverage_level", 0.9))
    lo = 0.5 * (1.0 - level)
    hi = 1.0 - lo
    q_lo = np.quantile(selected_draws, lo, axis=0)
    q_hi = np.quantile(selected_draws, hi, axis=0)
    coverage = float(np.mean((selected_truth >= q_lo) & (selected_truth <= q_hi)))
    tol = float(cfg.get("coverage_tolerance", 0.15))
    p_min = float(cfg.get("ks_pvalue_min", 0.05))

    reasons: List[str] = []
    if float(ks.pvalue) < p_min:
        reasons.append(f"ks_pvalue={float(ks.pvalue):.4f}<{p_min:.4f}")
    if abs(coverage - level) > tol:
        reasons.append(f"coverage_gap={abs(coverage - level):.3f}>{tol:.3f}")

    out.update(
        {
            "status": "fail" if reasons else "pass",
            "reasons": reasons,
            "draw_count": int(draws),
            "coefficient_count": int(use_count),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "target_coverage": float(level),
            "empirical_coverage": float(coverage),
        }
    )
    return out


def _run_ppc_validation(
    *,
    task: str,
    model: Any,
    posterior_arrays: Mapping[str, np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": "skip", "reasons": []}
    beta_draws = _flatten_param_draws(
        None if "beta" not in posterior_arrays else np.asarray(posterior_arrays["beta"], dtype=float)
    )
    if beta_draws is None or beta_draws.size == 0:
        if bool(cfg.get("fail_on_missing_draws", True)):
            out["status"] = "fail"
            out["reasons"] = ["missing_beta_draws"]
        else:
            out["reasons"] = ["missing_beta_draws"]
        return out
    min_draws = int(cfg.get("min_draws", 200))
    if beta_draws.shape[0] < min_draws:
        out["status"] = "fail"
        out["reasons"] = [f"insufficient_draws:{beta_draws.shape[0]}<{min_draws}"]
        return out
    tail_prob = float(cfg.get("tail_prob", 0.025))
    lower = tail_prob
    upper = 1.0 - tail_prob

    X_arr = np.asarray(X_test, dtype=float)
    y_arr = np.asarray(y_test, dtype=float).reshape(-1)
    intercept_draws = _flatten_scalar_draws(
        None if "intercept" not in posterior_arrays else np.asarray(posterior_arrays["intercept"], dtype=float)
    )
    if intercept_draws is None or intercept_draws.size == 0:
        intercept_const = getattr(model, "intercept_", 0.0)
        intercept_draws = np.full(beta_draws.shape[0], float(intercept_const), dtype=float)
    elif intercept_draws.shape[0] != beta_draws.shape[0]:
        intercept_draws = np.full(beta_draws.shape[0], float(np.mean(intercept_draws)), dtype=float)

    rng = np.random.default_rng(0)
    n_draws = beta_draws.shape[0]
    y_rep_mean = np.zeros(n_draws, dtype=float)
    y_rep_var = np.zeros(n_draws, dtype=float)

    if task == "classification":
        logits = X_arr @ beta_draws.T + intercept_draws[None, :]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        y_rep = rng.binomial(1, probs).astype(float)
        y_rep_mean[:] = y_rep.mean(axis=0)
        y_rep_var[:] = y_rep.var(axis=0)
    else:
        sigma_draws = _flatten_scalar_draws(
            None if "sigma" not in posterior_arrays else np.asarray(posterior_arrays["sigma"], dtype=float)
        )
        if sigma_draws is None or sigma_draws.size == 0:
            sigma2_draws = _flatten_scalar_draws(
                None if "sigma2" not in posterior_arrays else np.asarray(posterior_arrays["sigma2"], dtype=float)
            )
            if sigma2_draws is not None and sigma2_draws.size > 0:
                sigma_draws = np.sqrt(np.maximum(sigma2_draws, 0.0))
        if sigma_draws is None or sigma_draws.size == 0:
            sigma_draws = np.zeros(n_draws, dtype=float)
        if sigma_draws.shape[0] != n_draws:
            sigma_draws = np.full(n_draws, float(np.mean(sigma_draws)), dtype=float)

        mu = X_arr @ beta_draws.T + intercept_draws[None, :]
        noise = rng.normal(loc=0.0, scale=np.maximum(sigma_draws[None, :], 1e-12), size=mu.shape)
        y_rep = mu + noise
        y_rep_mean[:] = y_rep.mean(axis=0)
        y_rep_var[:] = y_rep.var(axis=0)

    obs_mean = float(np.mean(y_arr))
    obs_var = float(np.var(y_arr))
    p_mean = float(np.mean(y_rep_mean <= obs_mean))
    p_var = float(np.mean(y_rep_var <= obs_var))
    reasons: List[str] = []
    if not (lower <= p_mean <= upper):
        reasons.append(f"p_mean={p_mean:.4f} outside [{lower:.4f},{upper:.4f}]")
    if not (lower <= p_var <= upper):
        reasons.append(f"p_var={p_var:.4f} outside [{lower:.4f},{upper:.4f}]")
    out.update(
        {
            "status": "fail" if reasons else "pass",
            "reasons": reasons,
            "draw_count": int(n_draws),
            "tail_prob": float(tail_prob),
            "p_mean": p_mean,
            "p_var": p_var,
            "obs_mean": obs_mean,
            "obs_var": obs_var,
        }
    )
    return out


def _run_seed_stability_validation(
    *,
    base_model_config: Mapping[str, Any],
    groups: Sequence[Sequence[int]],
    p: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    C_train: Optional[np.ndarray],
    task: str,
    std_cfg: StandardizationConfig,
    convergence_cfg: Mapping[str, Any],
    baseline_arrays: Mapping[str, np.ndarray],
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": "skip", "reasons": []}
    baseline_beta = _flatten_param_draws(
        None if "beta" not in baseline_arrays else np.asarray(baseline_arrays["beta"], dtype=float)
    )
    if baseline_beta is None or baseline_beta.size == 0:
        out["status"] = "fail"
        out["reasons"] = ["missing_baseline_beta_draws"]
        return out
    baseline_beta_mean = baseline_beta.mean(axis=0)
    baseline_tau = _flatten_scalar_draws(
        None if "tau" not in baseline_arrays else np.asarray(baseline_arrays["tau"], dtype=float)
    )
    baseline_tau_mean = float(np.mean(baseline_tau)) if baseline_tau is not None and baseline_tau.size else None

    num_restarts = int(cfg.get("num_restarts", 2))
    stride = int(cfg.get("seed_stride", 1009))
    max_beta_rel_l2 = float(cfg.get("max_beta_rel_l2", 0.15))
    min_beta_cosine = float(cfg.get("min_beta_cosine", 0.98))
    max_tau_rel_sd = float(cfg.get("max_tau_rel_sd", 0.20))
    fail_on_missing_tau = bool(cfg.get("fail_on_missing_tau", False))

    beta_rel_l2_values: List[float] = []
    beta_cos_values: List[float] = []
    tau_values: List[float] = []
    reasons: List[str] = []
    for ridx in range(num_restarts):
        restart_cfg = deepcopy(base_model_config)
        _offset_all_seeds(restart_cfg, offset=(ridx + 1) * stride)
        try:
            (
                _model,
                arrays,
                _summary,
                _attempts,
                _effective,
                _sampler_diag,
            ) = _fit_model_with_retry(
                restart_cfg,
                groups,
                p,
                X_train,
                y_train,
                C_train,
                task,
                std_cfg,
                convergence_cfg,
            )
        except Exception as exc:
            reasons.append(f"restart_{ridx+1}_error={type(exc).__name__}")
            continue

        beta_draws = _flatten_param_draws(
            None if "beta" not in arrays else np.asarray(arrays["beta"], dtype=float)
        )
        if beta_draws is None or beta_draws.size == 0:
            reasons.append(f"restart_{ridx+1}_missing_beta")
            continue
        beta_mean = beta_draws.mean(axis=0)
        denom = float(np.linalg.norm(baseline_beta_mean))
        numer = float(np.linalg.norm(beta_mean - baseline_beta_mean))
        rel_l2 = numer / max(1e-12, denom)
        beta_rel_l2_values.append(rel_l2)
        cos_den = float(np.linalg.norm(beta_mean) * np.linalg.norm(baseline_beta_mean))
        cosine = 1.0 if cos_den <= 1e-12 else float(np.dot(beta_mean, baseline_beta_mean) / cos_den)
        beta_cos_values.append(cosine)

        tau_draws = _flatten_scalar_draws(None if "tau" not in arrays else np.asarray(arrays["tau"], dtype=float))
        if tau_draws is not None and tau_draws.size:
            tau_values.append(float(np.mean(tau_draws)))

    if not beta_rel_l2_values or not beta_cos_values:
        reasons.append("no_valid_restarts")
    else:
        if max(beta_rel_l2_values) > max_beta_rel_l2:
            reasons.append(f"beta_rel_l2_max={max(beta_rel_l2_values):.3f}>{max_beta_rel_l2:.3f}")
        if min(beta_cos_values) < min_beta_cosine:
            reasons.append(f"beta_cosine_min={min(beta_cos_values):.3f}<{min_beta_cosine:.3f}")

    tau_rel_sd = None
    if baseline_tau_mean is not None and tau_values:
        all_tau = np.asarray([baseline_tau_mean] + tau_values, dtype=float)
        tau_rel_sd = float(np.std(all_tau, ddof=1) / max(1e-12, abs(np.mean(all_tau))))
        if tau_rel_sd > max_tau_rel_sd:
            reasons.append(f"tau_rel_sd={tau_rel_sd:.3f}>{max_tau_rel_sd:.3f}")
    elif fail_on_missing_tau:
        reasons.append("missing_tau_for_stability")

    out.update(
        {
            "status": "fail" if reasons else "pass",
            "reasons": reasons,
            "num_restarts": int(num_restarts),
            "beta_rel_l2_max": None if not beta_rel_l2_values else float(max(beta_rel_l2_values)),
            "beta_cosine_min": None if not beta_cos_values else float(min(beta_cos_values)),
            "tau_rel_sd": tau_rel_sd,
        }
    )
    return out


def _run_posterior_validation(
    *,
    config: Mapping[str, Any],
    model: Any,
    posterior_arrays: Mapping[str, np.ndarray],
    task: str,
    beta_truth: Optional[np.ndarray],
    X_test_prediction: np.ndarray,
    y_test_eval: np.ndarray,
    groups: Sequence[Sequence[int]],
    p: int,
    X_train_model: np.ndarray,
    C_train_model: Optional[np.ndarray],
    y_train: np.ndarray,
    std_cfg: StandardizationConfig,
    convergence_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    cfg = _posterior_validation_config(config)
    result: Dict[str, Any] = {"enabled": bool(cfg.get("enabled", False))}
    if not bool(cfg.get("enabled", False)):
        result["status"] = "skip"
        return result
    if bool(cfg.get("apply_to_bayesian_only", True)) and not _is_bayesian_config(config):
        result["status"] = "skip"
        result["reason"] = "not_bayesian_model"
        return result

    sbc_cfg = cfg.get("sbc", {}) or {}
    ppc_cfg = cfg.get("ppc", {}) or {}
    seed_cfg = cfg.get("seed_stability", {}) or {}

    if bool(sbc_cfg.get("enabled", True)):
        result["sbc"] = _run_sbc_validation(beta_truth=beta_truth, posterior_arrays=posterior_arrays, cfg=sbc_cfg)
    if bool(ppc_cfg.get("enabled", True)):
        result["ppc"] = _run_ppc_validation(
            task=task,
            model=model,
            posterior_arrays=posterior_arrays,
            X_test=X_test_prediction,
            y_test=y_test_eval,
            cfg=ppc_cfg,
        )
    if bool(seed_cfg.get("enabled", True)):
        result["seed_stability"] = _run_seed_stability_validation(
            base_model_config=config,
            groups=groups,
            p=p,
            X_train=X_train_model,
            y_train=y_train,
            C_train=C_train_model,
            task=task,
            std_cfg=std_cfg,
            convergence_cfg=convergence_cfg,
            baseline_arrays=posterior_arrays,
            cfg=seed_cfg,
        )

    failures = _posterior_validation_failures(result)
    result["status"] = "fail" if failures else "pass"
    result["failures"] = failures
    return result


def _summarise_posterior(arrays: Mapping[str, np.ndarray]) -> Optional["pd.DataFrame"]:
    """Build summary statistics for posterior samples."""
    if not arrays or not _HAS_PANDAS:
        return None

    scalar_parameters = {"sigma", "sigma2", "tau", "tau2", "c"}
    records: List[Dict[str, Any]] = []
    for name, arr in arrays.items():
        data = np.asarray(arr)
        if data.ndim == 1:
            data = data[:, None]
        elif data.ndim == 2 and name in scalar_parameters:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            data = data.reshape(data.shape[0] * data.shape[1], -1)
        for idx in range(data.shape[1]):
            col = data[:, idx]
            if col.size == 0:
                continue
            record = {
                "parameter": name,
                "index": int(idx),
                "mean": float(col.mean()),
                "sd": float(col.std(ddof=1)) if col.size > 1 else 0.0,
                "q05": float(np.quantile(col, 0.05)),
                "q50": float(np.quantile(col, 0.50)),
                "q95": float(np.quantile(col, 0.95)),
                "min": float(col.min()),
                "max": float(col.max()),
            }
            records.append(record)
    if not records:
        return None
    return pd.DataFrame.from_records(records)


def _save_posterior_bundle(
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    include_convergence: bool = True,
    convergence_summary: Optional[Mapping[str, Mapping[str, Any]]] = None,
    min_chains_for_rhat: int = 2,
) -> Dict[str, Optional[str]]:
    """Persist posterior arrays and diagnostics if available."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not arrays:
        return {"posterior": None, "convergence": None, "summary": None}

    posterior_path = output_dir / "posterior_samples.npz"
    np.savez_compressed(posterior_path, **{k: np.asarray(v) for k, v in arrays.items()})

    convergence_path: Optional[Path] = None
    if include_convergence:
        convergence = (
            convergence_summary
            if convergence_summary is not None
            else _summarize_convergence_compat(arrays, min_chains_for_rhat=min_chains_for_rhat)
        )
        convergence_path = output_dir / "convergence.json"
        convergence_path.write_text(json.dumps(_to_serializable(convergence), indent=2), encoding="utf-8")

    summary_path: Optional[Path] = None
    summary_df = _summarise_posterior(arrays)
    if summary_df is not None:
        summary_path = output_dir / "posterior_summary.parquet"
        try:
            summary_df.to_parquet(summary_path)
        except Exception:  # pragma: no cover - parquet backend missing
            csv_path = summary_path.with_suffix(".csv")
            summary_df.to_csv(csv_path, index=False)
            summary_path = csv_path

    return {
        "posterior": str(posterior_path),
        "convergence": None if convergence_path is None else str(convergence_path),
        "summary": None if summary_path is None else str(summary_path),
    }


def _group_index(groups: Sequence[Sequence[int]], p: int) -> Optional[np.ndarray]:
    if not groups:
        return None
    index = np.zeros(p, dtype=int)
    for gid, block in enumerate(groups):
        index[np.asarray(block, dtype=int)] = gid
    return index


def _append_metrics_record(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_to_serializable(record)) + "\n")


def _standardize_covariates_train_test(
    C_train: np.ndarray,
    C_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize continuous covariates while leaving dummy columns untouched."""

    train = np.asarray(C_train, dtype=float)
    test = np.asarray(C_test, dtype=float)
    if train.ndim != 2 or test.ndim != 2:
        raise ExperimentError("Covariate matrices must be two-dimensional.")
    if train.shape[1] != test.shape[1]:
        raise ExperimentError("Train/test covariate matrices must share the same number of columns.")
    if train.shape[1] == 0:
        empty = np.zeros(0, dtype=np.float32)
        mask = np.zeros(0, dtype=bool)
        return train.astype(np.float32), test.astype(np.float32), empty, empty, mask

    means = train.mean(axis=0)
    scales = train.std(axis=0, ddof=0)
    binary_mask = np.zeros(train.shape[1], dtype=bool)
    for idx in range(train.shape[1]):
        values = np.unique(train[:, idx])
        binary_mask[idx] = values.size <= 2 and np.all(np.isin(values, [0.0, 1.0]))

    C_train_std = train.copy()
    C_test_std = test.copy()
    continuous_mask = ~binary_mask
    if np.any(continuous_mask):
        safe_scales = np.maximum(scales[continuous_mask], 1e-8)
        C_train_std[:, continuous_mask] = (train[:, continuous_mask] - means[continuous_mask]) / safe_scales
        C_test_std[:, continuous_mask] = (test[:, continuous_mask] - means[continuous_mask]) / safe_scales
        scales = np.where(continuous_mask, np.maximum(scales, 1e-8), 1.0)
    else:
        scales = np.ones_like(scales)

    return (
        C_train_std.astype(np.float32, copy=False),
        C_test_std.astype(np.float32, copy=False),
        means.astype(np.float32, copy=False),
        scales.astype(np.float32, copy=False),
        binary_mask,
    )


def _augment_with_intercept(C: np.ndarray) -> np.ndarray:
    arr = np.asarray(C, dtype=float)
    if arr.ndim != 2:
        raise ExperimentError("Covariate matrix must be two-dimensional.")
    intercept = np.ones((arr.shape[0], 1), dtype=float)
    return np.hstack([intercept, arr])


def _residualize_against_covariates(
    target_train: np.ndarray,
    target_test: np.ndarray,
    C_train: np.ndarray,
    C_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project targets onto covariates on the training fold and return residuals."""

    C_train_aug = _augment_with_intercept(C_train)
    C_test_aug = _augment_with_intercept(C_test)
    train_arr = np.asarray(target_train, dtype=float)
    test_arr = np.asarray(target_test, dtype=float)

    coef, _, _, _ = np.linalg.lstsq(C_train_aug, train_arr, rcond=None)
    resid_train = train_arr - C_train_aug @ coef
    resid_test = test_arr - C_test_aug @ coef
    return resid_train.astype(np.float32, copy=False), resid_test.astype(np.float32, copy=False), coef


def _estimate_covariate_effects(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    C_train: np.ndarray,
    C_test: np.ndarray,
    beta_point: np.ndarray,
    beta_draws: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Estimate nuisance covariate coefficients conditional on exposure effects."""

    C_train_aug = _augment_with_intercept(C_train)
    C_test_aug = _augment_with_intercept(C_test)
    beta_vec = np.asarray(beta_point, dtype=float).reshape(-1)
    alpha_hat, _, _, _ = np.linalg.lstsq(C_train_aug, np.asarray(y_train, dtype=float) - np.asarray(X_train, dtype=float) @ beta_vec, rcond=None)
    y_pred = (np.asarray(X_test, dtype=float) @ beta_vec) + (C_test_aug @ alpha_hat)

    alpha_draws: Optional[np.ndarray] = None
    pred_draws: Optional[np.ndarray] = None
    if beta_draws is not None:
        coef_draws = np.asarray(beta_draws, dtype=float)
        p = beta_vec.shape[0]
        if coef_draws.ndim == 1:
            if coef_draws.shape[0] != p:
                raise ExperimentError(
                    f"Posterior coefficient draw length {coef_draws.shape[0]} does not match feature count {p}."
                )
            coef_draws = coef_draws.reshape(1, p)
        elif coef_draws.ndim == 2:
            if coef_draws.shape[1] == p:
                pass
            elif coef_draws.shape[0] == p:
                coef_draws = coef_draws.T
            else:
                raise ExperimentError(
                    f"Posterior coefficient draws shape {coef_draws.shape} is incompatible with feature count {p}."
                )
        else:
            feature_axes = [axis for axis, size in enumerate(coef_draws.shape) if size == p]
            if not feature_axes:
                raise ExperimentError(
                    f"Posterior coefficient draws shape {coef_draws.shape} has no axis matching feature count {p}."
                )
            coef_draws = np.moveaxis(coef_draws, feature_axes[-1], -1).reshape(-1, p)

        alpha_draws = np.zeros((coef_draws.shape[0], C_train_aug.shape[1]), dtype=float)
        pred_draws = np.zeros((coef_draws.shape[0], X_test.shape[0]), dtype=float)
        y_train_arr = np.asarray(y_train, dtype=float)
        X_train_arr = np.asarray(X_train, dtype=float)
        X_test_arr = np.asarray(X_test, dtype=float)
        for idx, beta_draw in enumerate(coef_draws):
            alpha_draw, _, _, _ = np.linalg.lstsq(C_train_aug, y_train_arr - X_train_arr @ beta_draw, rcond=None)
            alpha_draws[idx] = alpha_draw
            pred_draws[idx] = X_test_arr @ beta_draw + C_test_aug @ alpha_draw

    return (
        y_pred.astype(np.float32, copy=False),
        None if pred_draws is None else pred_draws.astype(np.float32, copy=False),
        alpha_hat.astype(np.float32, copy=False),
        None if alpha_draws is None else alpha_draws.astype(np.float32, copy=False),
    )


def _run_fold_nested(
    base_config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    fold: OuterFold,
    *,
    fold_dir: Path,
    task: str,
    std_cfg: StandardizationConfig,
) -> Dict[str, Any]:
    """Run a single outer fold and capture metrics / posterior artefacts."""

    fold_dir.mkdir(parents=True, exist_ok=True)
    X = dataset["X"]
    C = dataset.get("C")
    y = dataset["y"]
    groups_true = dataset["groups"]
    model_groups = dataset.get("model_groups", groups_true)
    beta = dataset.get("beta")
    model_name = _resolve_model_name(base_config)
    use_joint_covariates = _uses_joint_covariates_model_name(model_name)

    train_idx = np.asarray(fold.train, dtype=int)
    test_idx = np.asarray(fold.test, dtype=int)

    std_train = apply_standardization(X[train_idx], y[train_idx], std_cfg)
    X_train = std_train.X
    y_train = std_train.y
    X_test = apply_standardizer(X[test_idx], x_mean=std_train.x_mean, x_scale=std_train.x_scale)
    if std_cfg.y_center:
        y_test = apply_y_mean(y[test_idx], mean=std_train.y_mean)
    else:
        y_test = np.asarray(y[test_idx], dtype=np.float32)
    y_train_eval = np.asarray(y_train, dtype=np.float32).copy()
    y_test_eval = np.asarray(y_test, dtype=np.float32).copy()

    C_train = None
    C_test = None
    C_train_raw = None
    C_test_raw = None
    covariate_mean = None
    covariate_scale = None
    covariate_binary_mask = None
    X_train_model = X_train
    X_test_model = X_test
    X_train_prediction = X_train
    X_test_prediction = X_test
    covariate_alpha_hat = None
    covariate_alpha_draws = None
    y_pred_override = None
    pred_draws_override = None

    if C is not None:
        C_train_raw = np.asarray(C[train_idx], dtype=np.float32)
        C_test_raw = np.asarray(C[test_idx], dtype=np.float32)
        C_train, C_test, covariate_mean, covariate_scale, covariate_binary_mask = _standardize_covariates_train_test(
            C_train_raw,
            C_test_raw,
        )
        if not use_joint_covariates:
            X_train_model, X_test_model, _ = _residualize_against_covariates(X_train, X_test, C_train, C_test)
            y_train, y_test, _ = _residualize_against_covariates(y_train, y_test, C_train, C_test)

    degenerate_model: Optional[_ConstantClassificationModel] = None
    degenerate_label_value: Optional[float] = None
    tuning_history: Optional[List[Dict[str, Any]]] = None
    inner_params: Dict[str, float] = {}

    if task == "classification":
        y_train = np.round(np.clip(y_train, 0.0, 1.0)).astype(np.float32)
        y_test = np.round(np.clip(y_test, 0.0, 1.0)).astype(np.float32)
        train_classes = np.unique(y_train)
        if train_classes.size < 2:
            degenerate_label_value = float(train_classes[0]) if train_classes.size else 0.0
            degenerate_model = _ConstantClassificationModel(
                prob_positive=degenerate_label_value,
                n_features=X_train_model.shape[1],
            )
            degenerate_model.fit(X_train_model, y_train)
            tuning_history = [
                {
                    "params": {},
                    "score": math.inf,
                    "skipped_folds": "all",
                    "reason": "single_class_outer_train",
                }
            ]

    model_config = deepcopy(base_config)
    model_config.setdefault("model", {})
    model_config["model"].pop("search", None)
    _apply_bayesian_sampling_budget(model_config)

    convergence_cfg = _convergence_config(base_config)
    convergence_summary: Optional[Dict[str, Dict[str, Any]]] = None
    convergence_attempts: List[Dict[str, Any]] = []
    sampler_diagnostics: Dict[str, Any] = {}
    bayesian_protocol = _bayesian_protocol_summary(
        model_config,
        std_cfg=std_cfg,
        groups=model_groups,
        p=int(X_train_model.shape[1]),
    )

    if degenerate_model is None:
        inner_params, tuning_history = _perform_inner_cv(
            base_config,
            X_train if C_train_raw is not None else X_train_model,
            y_train_eval if C_train_raw is not None else y_train,
            model_groups,
            C=C_train_raw,
            task=task,
            std_cfg=std_cfg,
        )
        for key, value in inner_params.items():
            if "." in str(key):
                _set_nested_config_value(model_config["model"], str(key), value)
            else:
                model_config["model"][key] = value
        (
            model,
            posterior_arrays_prefit,
            convergence_summary,
            convergence_attempts,
            effective_model_config,
            sampler_diagnostics,
        ) = _fit_model_with_retry(
            model_config,
            model_groups,
            X_train_model.shape[1],
            X_train_model,
            y_train,
            C_train if use_joint_covariates else None,
            task,
            std_cfg,
            convergence_cfg,
        )
        model_config = deepcopy(effective_model_config)
        posterior_arrays = posterior_arrays_prefit
        converged, convergence_failures = _check_convergence(
            convergence_summary,
            convergence_cfg,
            model_name=str(model_config["model"].get("name", "")),
            sampler_diagnostics=sampler_diagnostics,
        )
        fold_status = "OK" if converged else "INVALID_CONVERGENCE"
    else:
        model = degenerate_model
        fold_status = "DEGENERATE_LABELS"
        posterior_arrays = {}

    tuning_history = tuning_history or []

    experiments_cfg = base_config.get("experiments", {}) or {}
    coverage_level = float(experiments_cfg.get("coverage_level", 0.9))
    classification_threshold = float(experiments_cfg.get("classification_threshold", 0.5))
    evaluation_cfg = experiments_cfg.get("evaluation", {}) or {}
    predictive_density_mode = str(evaluation_cfg.get("predictive_density_mode", "mixed"))

    if C_train is not None and task != "classification":
        beta_point = getattr(model, "coef_mean_", None)
        if beta_point is None:
            beta_point = getattr(model, "coef_", None)
        if beta_point is None:
            raise ExperimentError("Covariate-adjusted regression requires fitted exposure coefficients.")
        y_pred_override, pred_draws_override, covariate_alpha_hat, covariate_alpha_draws = _estimate_covariate_effects(
            X_train,
            y_train_eval,
            X_test,
            C_train,
            C_test,
            np.asarray(beta_point, dtype=np.float32),
            getattr(model, "coef_samples_", None),
        )
        X_train_prediction = X_train
        X_test_prediction = X_test
    else:
        X_train_prediction = X_train_model
        X_test_prediction = X_test_model

    metrics = evaluate_model_metrics(
        model=model,
        X_train=X_train_model,
        X_test=X_test_prediction,
        y_train=y_train,
        y_test=y_test_eval,
        beta_truth=beta,
        group_index=_group_index(groups_true, X.shape[1]),
        coverage_level=coverage_level,
        slab_width=model_config["model"].get("c"),
        task=task,
        classification_threshold=classification_threshold,
        predictive_density_mode=predictive_density_mode,
        y_pred_override=y_pred_override,
        pred_draws_override=pred_draws_override,
    )

    metrics_path = fold_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_to_serializable(metrics), indent=2), encoding="utf-8")

    np.savez_compressed(
        fold_dir / "fold_arrays.npz",
        train_idx=train_idx,
        test_idx=test_idx,
        x_mean=np.asarray([]) if std_train.x_mean is None else std_train.x_mean,
        x_scale=np.asarray([]) if std_train.x_scale is None else std_train.x_scale,
        y_mean=np.asarray([]) if std_train.y_mean is None else np.array([std_train.y_mean], dtype=np.float32),
        covariate_mean=np.asarray([]) if covariate_mean is None else covariate_mean,
        covariate_scale=np.asarray([]) if covariate_scale is None else covariate_scale,
        covariate_binary_mask=np.asarray([]) if covariate_binary_mask is None else covariate_binary_mask.astype(np.int8),
    )

    save_posterior = bool(experiments_cfg.get("save_posterior", True))
    posterior_arrays_for_validation: Dict[str, np.ndarray] = (
        posterior_arrays if isinstance(posterior_arrays, Mapping) and posterior_arrays else _collect_posterior_arrays(model)
    )
    posterior_validation: Optional[Dict[str, Any]] = None
    if degenerate_model is None:
        posterior_validation = _run_posterior_validation(
            config=model_config,
            model=model,
            posterior_arrays=posterior_arrays_for_validation,
            task=task,
            beta_truth=beta,
            X_test_prediction=X_test_prediction,
            y_test_eval=y_test_eval,
            groups=model_groups,
            p=int(X_train_model.shape[1]),
            X_train_model=X_train_model,
            C_train_model=C_train if use_joint_covariates else None,
            y_train=y_train,
            std_cfg=std_cfg,
            convergence_cfg=convergence_cfg,
        )
        if str(posterior_validation.get("status", "")).lower() == "fail" and fold_status == "OK":
            fold_status = "INVALID_POSTERIOR_VALIDATION"

    if degenerate_model is None and posterior_arrays:
        posterior_arrays = posterior_arrays if save_posterior else {}
    else:
        posterior_arrays = _collect_posterior_arrays(model) if save_posterior else {}
    posterior_paths = None
    if posterior_arrays:
        posterior_paths = _save_posterior_bundle(
            fold_dir,
            posterior_arrays,
            convergence_summary=convergence_summary,
            min_chains_for_rhat=int(convergence_cfg.get("min_chains_for_rhat", 2)),
        )
    if covariate_alpha_hat is not None:
        alpha_payload: Dict[str, Any] = {"alpha_hat": covariate_alpha_hat}
        if covariate_alpha_draws is not None:
            alpha_payload["alpha_draws"] = covariate_alpha_draws
        np.savez_compressed(fold_dir / "covariate_adjustment.npz", **alpha_payload)
    if posterior_validation is not None:
        (fold_dir / "posterior_validation.json").write_text(
            json.dumps(_to_serializable(posterior_validation), indent=2),
            encoding="utf-8",
        )

    fold_summary = {
        "status": fold_status,
        "repeat": int(fold.repeat),
        "fold": int(fold.fold),
        "hash": fold.hash,
        "metrics": metrics,
        "best_params": inner_params,
        "tuning_history": tuning_history,
        "posterior_files": posterior_paths,
    }
    if convergence_summary is not None:
        fold_summary["convergence"] = convergence_summary
    if degenerate_model is None and convergence_summary is not None:
        fold_summary["expected_convergence_blocks"] = _expected_convergence_blocks(
            str(model_config["model"].get("name", "")),
            convergence_cfg,
        )
    if convergence_attempts:
        fold_summary["convergence_attempts"] = convergence_attempts
    if sampler_diagnostics:
        fold_summary["sampler_diagnostics"] = sampler_diagnostics
    if bayesian_protocol is not None:
        fold_summary["bayesian_fairness"] = bayesian_protocol
    if posterior_validation is not None:
        fold_summary["posterior_validation"] = posterior_validation
    if covariate_alpha_hat is not None:
        fold_summary["covariate_adjustment_file"] = str(fold_dir / "covariate_adjustment.npz")
    if degenerate_label_value is not None:
        fold_summary["degenerate_label"] = degenerate_label_value
    (fold_dir / "fold_summary.json").write_text(json.dumps(_to_serializable(fold_summary), indent=2), encoding="utf-8")

    fold_summary["posterior_arrays"] = posterior_arrays if save_posterior else {}
    return fold_summary


def _aggregate_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    include_invalid: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Aggregate scalar metrics across folds."""

    collector: Dict[str, List[float]] = defaultdict(list)
    for entry in records:
        if not include_invalid and not _classify_fold_status(str(entry.get("status", "OK"))):
            continue
        metrics = entry.get("metrics", {})
        for key, value in metrics.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            collector[key].append(numeric_value)

    mean_metrics = {key: float(np.mean(values)) for key, values in collector.items() if values}
    summary: Dict[str, Dict[str, float]] = {}
    for key, values in collector.items():
        arr = np.asarray(values, dtype=float)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "stderr": float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "count": float(arr.size),
        }
    return mean_metrics, summary


def _aggregate_metric_sources(
    records: Sequence[Mapping[str, Any]],
    *,
    include_invalid: bool = False,
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in records:
        if not include_invalid and not _classify_fold_status(str(entry.get("status", "OK"))):
            continue
        metrics = entry.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        for key, value in metrics.items():
            if not str(key).endswith("_source"):
                continue
            metric_name = str(key)[: -len("_source")]
            counts[metric_name][str(value)] += 1
    return {metric: dict(source_counts) for metric, source_counts in counts.items()}


def _extract_inference_seeds(cfg: Mapping[str, Any]) -> Dict[str, int]:
    seeds: Dict[str, int] = {}
    inference_cfg = cfg.get("inference")
    if not isinstance(inference_cfg, Mapping):
        return seeds
    for key, section in inference_cfg.items():
        if isinstance(section, Mapping) and "seed" in section:
            try:
                seeds[key] = int(section["seed"])
            except (TypeError, ValueError):
                continue
    return seeds


def run_experiment(config: Mapping[str, Any], output_dir: Path | str) -> Dict[str, Any]:
    """
    Execute nested CV experiment described by ``config``.

    Args:
        config: Fully merged experiment configuration.
        output_dir: Directory where artefacts should be written.

    Returns:
        Dictionary with aggregated metrics and bookkeeping information.
    """

    _require_nested_cv(config)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    effective_config = deepcopy(config)
    task = _resolve_task(effective_config)
    std_cfg = _standardization_from_config(effective_config, task)
    experiments_cfg = effective_config.get("experiments", {}) or {}
    repeats = max(1, int(experiments_cfg.get("repeats", 1) or 1))

    metrics_jsonl = output_path / "metrics.jsonl"
    if metrics_jsonl.exists():
        metrics_jsonl.unlink()

    all_fold_records: List[Dict[str, Any]] = []
    all_valid_fold_records: List[Dict[str, Any]] = []
    posterior_accumulator: Dict[str, List[np.ndarray]] = defaultdict(list)
    repeat_summaries: List[Dict[str, Any]] = []
    repeat_dir_paths: List[str] = []

    model_name = get_model_name_from_config(effective_config)
    _apply_bayesian_sampling_budget(effective_config)
    run_bayesian_protocol = _bayesian_protocol_summary(
        effective_config,
        std_cfg=std_cfg,
        groups=_resolve_groups(effective_config, effective_config.get("data", {}).get("groups"), 0)
        if effective_config.get("data", {}).get("groups")
        else [],
        p=int((effective_config.get("data", {}) or {}).get("p", 0) or 0),
    )

    for repeat_idx in range(repeats):
        repeat_config = deepcopy(effective_config)
        _adjust_seeds_for_repeat(repeat_config, repeat_idx)
        dataset = _prepare_dataset_bundle(repeat_config, task=task, repeat_index=repeat_idx)
        repeat_dir = output_path / f"repeat_{repeat_idx + 1:03d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        repeat_dir_paths.append(str(repeat_dir))

        dataset_meta = {
            "repeat_index": repeat_idx + 1,
            "task": task,
            "n": int(dataset["X"].shape[0]),
            "p": int(dataset["X"].shape[1]),
            "covariate_p": 0 if dataset.get("C") is None else int(dataset["C"].shape[1]),
            "groups": dataset["groups"],
            "model_groups": dataset.get("model_groups"),
            "feature_names": dataset.get("feature_names"),
            "covariate_feature_names": dataset.get("covariate_feature_names"),
            "metadata": dataset.get("metadata"),
        }
        (repeat_dir / "dataset_meta.json").write_text(json.dumps(_to_serializable(dataset_meta), indent=2), encoding="utf-8")

        outer_folds = _prepare_outer_folds(repeat_config, dataset, task=task, repeat_index=repeat_idx)
        repeat_records: List[Dict[str, Any]] = []

        for fold in progress(outer_folds, total=len(outer_folds), desc=f"repeat {repeat_idx + 1}/{repeats}: outer CV"):
            fold_dir = repeat_dir / f"fold_{fold.fold:02d}"
            fold_result = _run_fold_nested(
                repeat_config,
                dataset,
                fold,
                fold_dir=fold_dir,
                task=task,
                std_cfg=std_cfg,
            )

            record = {
                "status": fold_result.get("status", "OK"),
                "repeat": repeat_idx + 1,
                "outer_repeat": fold_result.get("repeat"),
                "fold": fold_result.get("fold"),
                "hash": fold_result.get("hash"),
                "metrics": fold_result.get("metrics", {}),
                "best_params": fold_result.get("best_params"),
            }
            _append_metrics_record(metrics_jsonl, record)

            posterior_arrays = fold_result.pop("posterior_arrays", {})
            if posterior_arrays and _classify_fold_status(record["status"]):
                for key, arr in posterior_arrays.items():
                    posterior_accumulator[key].append(np.asarray(arr))

            repeat_records.append(fold_result)
            all_fold_records.append({**record, "tuning_history": fold_result.get("tuning_history")})
            if _classify_fold_status(record["status"]):
                all_valid_fold_records.append({**record, "tuning_history": fold_result.get("tuning_history")})

        repeat_mean, repeat_summary = _aggregate_metrics(repeat_records)
        repeat_mean_all, repeat_summary_all = _aggregate_metrics(repeat_records, include_invalid=True)
        repeat_seeds = {}
        inference_seeds = _extract_inference_seeds(repeat_config)
        if inference_seeds:
            repeat_seeds["inference"] = inference_seeds
        data_seed = dataset.get("metadata", {}).get("seed")
        if data_seed is not None:
            try:
                repeat_seeds["data_generation"] = int(data_seed)
            except (TypeError, ValueError):
                pass
        repeat_payload = {
            "repeat_index": repeat_idx + 1,
            "metrics": repeat_mean,
            "metrics_summary": repeat_summary,
            "metric_sources": _aggregate_metric_sources(repeat_records),
            "metrics_all_folds": repeat_mean_all,
            "metrics_summary_all_folds": repeat_summary_all,
            "metric_sources_all_folds": _aggregate_metric_sources(repeat_records, include_invalid=True),
            "folds": [
                {
                    "status": entry.get("status"),
                    "repeat": entry.get("repeat"),
                    "fold": entry.get("fold"),
                    "hash": entry.get("hash"),
                    "metrics": entry.get("metrics"),
                    "best_params": entry.get("best_params"),
                    "posterior_files": entry.get("posterior_files"),
                }
                for entry in repeat_records
            ],
            "valid_folds": int(sum(1 for entry in repeat_records if _classify_fold_status(str(entry.get("status", "OK"))))),
            "invalid_folds": int(sum(1 for entry in repeat_records if not _classify_fold_status(str(entry.get("status", "OK"))))),
        }
        if repeat_seeds:
            repeat_payload["seeds"] = repeat_seeds
        (repeat_dir / "repeat_summary.json").write_text(json.dumps(_to_serializable(repeat_payload), indent=2), encoding="utf-8")
        repeat_summaries.append(repeat_payload)

    aggregated_metrics, aggregated_summary = _aggregate_metrics(all_fold_records)
    aggregated_metrics_all, aggregated_summary_all = _aggregate_metrics(all_fold_records, include_invalid=True)

    if posterior_accumulator:
        combined: Dict[str, np.ndarray] = {}
        for key, arrs in posterior_accumulator.items():
            if not arrs or np.asarray(arrs[0]).size == 0:
                continue
            target = np.asarray(arrs[0])
            kept: List[np.ndarray] = [target]
            skipped = 0
            for arr in arrs[1:]:
                candidate = np.asarray(arr)
                if candidate.ndim != target.ndim:
                    skipped += 1
                    continue
                if candidate.shape[1:] != target.shape[1:]:
                    if candidate.ndim == 2 and candidate.T.shape[1:] == target.shape[1:]:
                        candidate = candidate.T
                    else:
                        skipped += 1
                        continue
                kept.append(candidate)
            if skipped:
                print(f"[WARN] Skipped {skipped} posterior chunks for '{key}' due to shape mismatch.")
            combined[key] = np.concatenate(kept, axis=0)
        if combined:
            _save_posterior_bundle(
                output_path,
                combined,
                include_convergence=False,
                min_chains_for_rhat=int(_convergence_config(effective_config).get("min_chains_for_rhat", 2)),
            )

    summary_payload = {
        "status": "OK" if len(all_valid_fold_records) == len(all_fold_records) else "PARTIAL",
        "model": model_name,
        "task": task,
        "repeats": repeats,
        "outer_folds_per_repeat": len(repeat_summaries[0]["folds"]) if repeat_summaries else 0,
        "metrics": aggregated_metrics,
        "metrics_summary": aggregated_summary,
        "metric_sources": _aggregate_metric_sources(all_fold_records),
        "metrics_all_folds": aggregated_metrics_all,
        "metrics_summary_all_folds": aggregated_summary_all,
        "metric_sources_all_folds": _aggregate_metric_sources(all_fold_records, include_invalid=True),
        "valid_fold_count": int(len(all_valid_fold_records)),
        "invalid_fold_count": int(len(all_fold_records) - len(all_valid_fold_records)),
        "repeat_summaries": repeat_summaries,
        "artifacts": {
            "repeat_dirs": repeat_dir_paths,
        },
    }
    if run_bayesian_protocol is not None:
        summary_payload["bayesian_fairness"] = run_bayesian_protocol

    (output_path / "summary.json").write_text(json.dumps(_to_serializable(summary_payload), indent=2), encoding="utf-8")
    (output_path / "metrics.json").write_text(json.dumps(_to_serializable(aggregated_metrics), indent=2), encoding="utf-8")

    return summary_payload


__all__ = ["run_experiment"]
