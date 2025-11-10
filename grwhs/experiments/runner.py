"""
Experiment orchestration entry points for GRwHS experiments.

This module implements the nested cross-validation workflow described in the
project brief:

* Synthetic or loader-backed dataset preparation with train-only standardisation.
* Outer K-fold (optionally repeated) evaluation with shared splits across methods.
* Inner CV hyper-parameter selection for frequentist baselines (GL/SGL).
* Optional τ calibration for GRwHS variants using the expected effective complexity.
* Exhaustive metric logging plus posterior export for Bayesian models.

The function :func:`run_experiment` is the public entry point invoked by the CLI
(`python -m grwhs.cli.run_experiment`).  It expects a fully merged configuration
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
from grwhs.diagnostics.convergence import summarize_convergence
from grwhs.experiments.registry import build_from_config, get_model_name_from_config
from grwhs.metrics.evaluation import evaluate_model_metrics
from grwhs.utils.logging_utils import progress


StratifyOption = Optional[Union[bool, str]]


class ExperimentError(RuntimeError):
    """Raised when a configuration-driven experiment cannot be executed."""


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
    if base_seed is not None:
        dataset_seed = base_seed
    else:
        dataset_seed = repeat_index if repeat_index > 0 else None

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
            "y": y,
            "beta": None if generated.beta is None else np.asarray(generated.beta, dtype=np.float32),
            "groups": groups,
            "model_groups": model_groups,
            "feature_names": None,
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
            "y": y,
            "beta": None if loaded.beta is None else np.asarray(loaded.beta, dtype=np.float32),
            "groups": groups,
            "model_groups": model_groups,
            "feature_names": loaded.feature_names,
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
        sigma_ref = float(tau_cfg.get("sigma_classification", 2.0))
    else:
        if std_cfg.y_center:
            y_scale = float(np.std(y, ddof=1)) if y.size else 1.0
        else:
            y_scale = float(np.std(y, ddof=1)) if y.size else 1.0
        sigma_ref = float(tau_cfg.get("sigma_reference", y_scale))

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


def _perform_inner_cv(
    base_config: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    task: str,
    std_cfg: StandardizationConfig,
) -> Tuple[Dict[str, float], Optional[List[Dict[str, Any]]]]:
    """Grid-search hyper-parameters for frequentist baselines."""

    search_cfg = deepcopy(base_config.get("model", {}).get("search"))
    if not isinstance(search_cfg, Mapping) or not search_cfg:
        return {}, None

    std_all = apply_standardization(X, y, std_cfg)
    X_ref = std_all.X
    y_ref = std_all.y

    keys = sorted(search_cfg.keys())
    grid_values: List[List[float]] = []
    for key in keys:
        values = search_cfg[key]
        expanded = _expand_search_values(values, key, X_ref, y_ref)
        grid_values.append(expanded)

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

    history: List[Dict[str, Any]] = []
    best_candidate: Optional[Dict[str, float]] = None
    best_score = math.inf

    for candidate_values in np.array(np.meshgrid(*grid_values, indexing="ij")).T.reshape(-1, len(keys)):
        candidate = {key: float(val) for key, val in zip(keys, candidate_values)}
        fold_scores: List[float] = []
        skipped_folds = 0

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

            candidate_cfg = deepcopy(base_config)
            candidate_cfg.setdefault("model", {})
            candidate_cfg = deepcopy(candidate_cfg)
            candidate_cfg["model"] = {**candidate_cfg["model"], **candidate}
            candidate_cfg["model"].pop("search", None)

            if task == "classification":
                unique_classes = np.unique(y_train)
                if unique_classes.size < 2:
                    skipped_folds += 1
                    continue

            model = _instantiate_model(candidate_cfg, groups, X.shape[1])
            try:
                model.fit(X_train, y_train, groups=groups)
            except TypeError:
                model.fit(X_train, y_train)

            score = _compute_inner_metric(
                task,
                y_val,
                model,
                X_val,
                class_labels=class_labels,
            )
            fold_scores.append(score)

        if not fold_scores:
            avg_score = math.inf
        else:
            avg_score = float(np.mean(fold_scores))

        history_entry: Dict[str, Any] = {"params": candidate, "score": avg_score}
        if skipped_folds:
            history_entry["skipped_folds"] = skipped_folds
        history.append(history_entry)
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
        "sigma_samples_": "sigma",
        "sigma2_samples_": "sigma2",
        "tau_samples_": "tau",
        "phi_samples_": "phi",
        "lambda_samples_": "lambda",
        "loglik_samples_": "loglik",
    }
    for attr, key in attr_map.items():
        value = getattr(model, attr, None)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.size == 0:
            continue
        arrays[key] = arr
    return arrays


def _summarise_posterior(arrays: Mapping[str, np.ndarray]) -> Optional["pd.DataFrame"]:
    """Build summary statistics for posterior samples."""
    if not arrays or not _HAS_PANDAS:
        return None

    records: List[Dict[str, Any]] = []
    for name, arr in arrays.items():
        data = np.asarray(arr)
        if data.ndim == 1:
            data = data[:, None]
        elif data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
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
    output_dir: Path, arrays: Mapping[str, np.ndarray], *, include_convergence: bool = True
) -> Dict[str, Optional[str]]:
    """Persist posterior arrays and diagnostics if available."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not arrays:
        return {"posterior": None, "convergence": None, "summary": None}

    posterior_path = output_dir / "posterior_samples.npz"
    np.savez_compressed(posterior_path, **{k: np.asarray(v) for k, v in arrays.items()})

    convergence_path: Optional[Path] = None
    if include_convergence:
        convergence = summarize_convergence(arrays)
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
    y = dataset["y"]
    groups_true = dataset["groups"]
    model_groups = dataset.get("model_groups", groups_true)
    beta = dataset.get("beta")

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
                n_features=X_train.shape[1],
            )
            degenerate_model.fit(X_train, y_train)
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

    if degenerate_model is None:
        inner_params, tuning_history = _perform_inner_cv(
            base_config,
            X_train,
            y_train,
            model_groups,
            task=task,
            std_cfg=std_cfg,
        )
        for key, value in inner_params.items():
            model_config["model"][key] = value
        _maybe_calibrate_tau(model_config["model"], std_cfg, X_train, y_train, model_groups, task)
        model = _instantiate_model(model_config, model_groups, X_train.shape[1])
        try:
            model.fit(X_train, y_train, groups=model_groups)
        except TypeError:
            model.fit(X_train, y_train)
        fold_status = "OK"
    else:
        model = degenerate_model
        fold_status = "DEGENERATE_LABELS"

    tuning_history = tuning_history or []

    experiments_cfg = base_config.get("experiments", {}) or {}
    coverage_level = float(experiments_cfg.get("coverage_level", 0.9))
    classification_threshold = float(experiments_cfg.get("classification_threshold", 0.5))

    metrics = evaluate_model_metrics(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        beta_truth=beta,
        group_index=_group_index(groups_true, X.shape[1]),
        coverage_level=coverage_level,
        slab_width=model_config["model"].get("c"),
        task=task,
        classification_threshold=classification_threshold,
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
    )

    save_posterior = bool(experiments_cfg.get("save_posterior", True))
    posterior_arrays = _collect_posterior_arrays(model) if save_posterior else {}
    posterior_paths = None
    if posterior_arrays:
        posterior_paths = _save_posterior_bundle(fold_dir, posterior_arrays)

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
    if degenerate_label_value is not None:
        fold_summary["degenerate_label"] = degenerate_label_value
    (fold_dir / "fold_summary.json").write_text(json.dumps(_to_serializable(fold_summary), indent=2), encoding="utf-8")

    fold_summary["posterior_arrays"] = posterior_arrays if save_posterior else {}
    return fold_summary


def _aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Aggregate scalar metrics across folds."""

    collector: Dict[str, List[float]] = defaultdict(list)
    for entry in records:
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
    posterior_accumulator: Dict[str, List[np.ndarray]] = defaultdict(list)
    repeat_summaries: List[Dict[str, Any]] = []
    repeat_dir_paths: List[str] = []

    model_name = get_model_name_from_config(effective_config)

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
            "groups": dataset["groups"],
            "model_groups": dataset.get("model_groups"),
            "feature_names": dataset.get("feature_names"),
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
            if posterior_arrays:
                for key, arr in posterior_arrays.items():
                    posterior_accumulator[key].append(np.asarray(arr))

            repeat_records.append(fold_result)
            all_fold_records.append({**record, "tuning_history": fold_result.get("tuning_history")})

        repeat_mean, repeat_summary = _aggregate_metrics(repeat_records)
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
            "folds": [
                {
                    "repeat": entry.get("repeat"),
                    "fold": entry.get("fold"),
                    "hash": entry.get("hash"),
                    "metrics": entry.get("metrics"),
                    "best_params": entry.get("best_params"),
                    "posterior_files": entry.get("posterior_files"),
                }
                for entry in repeat_records
            ],
        }
        if repeat_seeds:
            repeat_payload["seeds"] = repeat_seeds
        (repeat_dir / "repeat_summary.json").write_text(json.dumps(_to_serializable(repeat_payload), indent=2), encoding="utf-8")
        repeat_summaries.append(repeat_payload)

    aggregated_metrics, aggregated_summary = _aggregate_metrics(all_fold_records)

    if posterior_accumulator:
        combined = {
            key: np.concatenate(arrs, axis=0)
            for key, arrs in posterior_accumulator.items()
            if arrs and arrs[0].size > 0
        }
        if combined:
            _save_posterior_bundle(output_path, combined, include_convergence=False)

    summary_payload = {
        "status": "OK",
        "model": model_name,
        "task": task,
        "repeats": repeats,
        "outer_folds_per_repeat": len(repeat_summaries[0]["folds"]) if repeat_summaries else 0,
        "metrics": aggregated_metrics,
        "metrics_summary": aggregated_summary,
        "repeat_summaries": repeat_summaries,
        "artifacts": {
            "repeat_dirs": repeat_dir_paths,
        },
    }

    (output_path / "summary.json").write_text(json.dumps(_to_serializable(summary_payload), indent=2), encoding="utf-8")
    (output_path / "metrics.json").write_text(json.dumps(_to_serializable(aggregated_metrics), indent=2), encoding="utf-8")

    return summary_payload


__all__ = ["run_experiment"]
