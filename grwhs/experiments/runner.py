
"""Experiment orchestration entry points."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from data.generators import generate_synthetic, synthetic_config_from_dict, make_groups
from data.preprocess import StandardizationConfig, apply_standardization
from data.splits import train_val_test_split
from data.loaders import load_real_dataset
from grwhs.experiments.registry import build_from_config, get_model_name_from_config
from grwhs.metrics import regression
from grwhs.models.baselines import Ridge
from grwhs.diagnostics.convergence import summarize_convergence


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = int(y.sum())
    neg = int(y.size - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.arange(1, y.size + 1)
    sum_pos = ranks[order][y[order] == 1].sum()
    auc = (sum_pos - pos * (pos + 1) / 2) / (pos * neg)
    return float(auc)


def _selection_predictions(beta_hat: np.ndarray, threshold_cfg: Dict[str, Any]) -> np.ndarray | None:
    kind = str(threshold_cfg.get("type", "magnitude")).lower()
    if kind != "magnitude":
        return None
    value = float(threshold_cfg.get("value", 0.0))
    return (np.abs(beta_hat) > value).astype(int)


def _false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    tn = np.logical_and(y_true == 0, y_pred == 0).sum()
    return float(fp / (fp + tn + 1e-8))


def run_experiment(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Execute an experiment (synthetic or loader-backed) based on configuration."""

    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config.get("data", {})
    data_type = str(data_cfg.get("type", "synthetic")).lower()
    base_seed = config.get("seed")

    dataset_metadata: Dict[str, Any] = {}
    feature_names: Optional[List[str]] = None

    strong_idx = weak_idx = active_idx = None

    X_source: np.ndarray
    y_source: Optional[np.ndarray]
    beta_truth: Optional[np.ndarray]
    groups: List[List[int]] | None
    experiment_seed: Optional[int] = base_seed

    if data_type == "synthetic":
        syn_cfg = synthetic_config_from_dict(data_cfg, seed=base_seed, name=config.get("name"))
        dataset = generate_synthetic(syn_cfg)
        X_source = dataset.X
        y_source = dataset.y
        beta_truth = dataset.beta
        groups = dataset.groups
        dataset_metadata.update(dataset.info)
        dataset_metadata.setdefault("scenario", syn_cfg.name)
        strong_idx = dataset.info.get("strong_idx")
        weak_idx = dataset.info.get("weak_idx")
        active_idx = dataset.info.get("active_idx")
        experiment_seed = syn_cfg.seed
    elif data_type == "loader":
        loader_cfg = data_cfg.get("loader", {})
        loader_base = loader_cfg.get("base_dir") or data_cfg.get("base_dir") or config.get("data_root")
        base_dir_path = Path(loader_base).expanduser() if loader_base else None
        loaded = load_real_dataset(loader_cfg, base_dir=base_dir_path)
        X_source = loaded.X
        y_source = loaded.y
        if y_source is None:
            raise ValueError("Real dataset loader requires targets (provide loader.path_y).")
        beta_truth = loaded.beta
        groups = loaded.groups
        feature_names = loaded.feature_names
        dataset_metadata.update(loaded.metadata)
        if groups is None:
            if data_cfg.get("groups") is not None:
                groups = [[int(idx) for idx in group] for group in data_cfg["groups"]]
            else:
                p = X_source.shape[1]
                group_sizes = data_cfg.get("group_sizes")
                G = data_cfg.get("G")
                if group_sizes is not None or G is not None:
                    groups = make_groups(p, G, group_sizes)
                else:
                    groups = [[j] for j in range(p)]
    else:
        raise ValueError(f"Unsupported data.type '{data_type}'.")

    if groups is None:
        groups = [[j] for j in range(X_source.shape[1])]

    std_cfg_dict = config.get("standardization", {})
    std_cfg = StandardizationConfig(
        X=std_cfg_dict.get("X", "unit_variance"),
        y_center=bool(std_cfg_dict.get("y_center", True)),
    )
    std_result = apply_standardization(X_source, y_source, std_cfg)

    splits = train_val_test_split(
        n=std_result.X.shape[0],
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        test_ratio=float(data_cfg.get("test_ratio", 0.2)),
        seed=base_seed,
    )

    def _slice(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr[idx]
        return arr[idx, :]

    X_train = _slice(std_result.X, splits.train)
    y_train = None if std_result.y is None else _slice(std_result.y, splits.train)
    X_val = _slice(std_result.X, splits.val) if splits.val.size else np.empty((0, std_result.X.shape[1]), dtype=std_result.X.dtype)
    y_val = None if std_result.y is None else (_slice(std_result.y, splits.val) if splits.val.size else np.empty((0,), dtype=std_result.y.dtype))
    X_test = _slice(std_result.X, splits.test) if splits.test.size else np.empty((0, std_result.X.shape[1]), dtype=std_result.X.dtype)
    y_test = None if std_result.y is None else (_slice(std_result.y, splits.test) if splits.test.size else np.empty((0,), dtype=std_result.y.dtype))

    try:
        model = build_from_config(config)
        model_name = get_model_name_from_config(config)
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"[WARN] Using Ridge fallback because model construction failed: {exc}")
        fit_intercept = bool(config.get("model", {}).get("fit_intercept", False))
        model = Ridge(alpha=1.0, fit_intercept=fit_intercept)
        model_name = "ridge_fallback"

    if y_train is None:
        raise ValueError("Experiment requires targets after preprocessing.")
    model.fit(X_train, y_train)

    metrics_cfg = config.get("experiments", {})
    requested_metrics: List[str] = list(metrics_cfg.get("metrics", ["mse", "r2"]))
    threshold_cfg = metrics_cfg.get("threshold", {"type": "magnitude", "value": 0.0})

    results: Dict[str, Any] = {}
    if y_test is not None and X_test.size:
        y_pred = model.predict(X_test)
        for metric in requested_metrics:
            name = metric.lower()
            if name == "mse":
                results[metric] = regression.mse(y_test, y_pred)
            elif name == "mae":
                results[metric] = regression.mae(y_test, y_pred)
            elif name == "r2":
                results[metric] = regression.r2(y_test, y_pred)
            else:
                results[metric] = None
    else:
        y_pred = None
        for metric in requested_metrics:
            results[metric] = None

    beta_hat = getattr(model, "coef_", None)
    selection_truth = None if beta_truth is None else (np.abs(beta_truth) > 1e-8).astype(int)
    if beta_hat is not None and selection_truth is not None:
        preds_binary = _selection_predictions(beta_hat, threshold_cfg)
        if preds_binary is not None:
            if "tpr" in requested_metrics:
                tp = np.logical_and(selection_truth == 1, preds_binary == 1).sum()
                fn = np.logical_and(selection_truth == 1, preds_binary == 0).sum()
                results["tpr"] = float(tp / (tp + fn + 1e-8))
            if "fpr" in requested_metrics:
                results["fpr"] = _false_positive_rate(selection_truth, preds_binary)
        if "auc" in requested_metrics:
            auc = _auc_from_scores(selection_truth, np.abs(beta_hat))
            results["auc"] = auc
    else:
        if "auc" in requested_metrics:
            results.setdefault("auc", None)
        if "tpr" in requested_metrics:
            results.setdefault("tpr", None)
        if "fpr" in requested_metrics:
            results.setdefault("fpr", None)

    metrics_serializable = {k: _to_serializable(v) for k, v in results.items()}

    dataset_path = output_dir / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        beta_true=np.array([], dtype=np.float32) if beta_truth is None else beta_truth,
        x_mean=np.array([]) if std_result.x_mean is None else std_result.x_mean,
        x_scale=np.array([]) if std_result.x_scale is None else std_result.x_scale,
        y_mean=np.array([]) if std_result.y_mean is None else np.array([std_result.y_mean], dtype=np.float32),
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_serializable, indent=2), encoding="utf-8")

    posterior_arrays: Dict[str, np.ndarray] = {}
    convergence_summary: Dict[str, Dict[str, float]] = {}
    posterior_path: Path | None = None
    if metrics_cfg.get("save_posterior", False):
        sample_attrs = [
            ("coef_samples_", "beta"),
            ("sigma_samples_", "sigma"),
            ("sigma2_samples_", "sigma2"),
            ("tau_samples_", "tau"),
            ("phi_samples_", "phi"),
            ("lambda_samples_", "lambda"),
        ]
        for attr, key in sample_attrs:
            value = getattr(model, attr, None)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.size == 0:
                continue
            posterior_arrays[key] = arr
        if posterior_arrays:
            posterior_path = output_dir / "posterior_samples.npz"
            np.savez_compressed(posterior_path, **{k: np.asarray(v) for k, v in posterior_arrays.items()})
            convergence_summary = summarize_convergence(posterior_arrays)
            conv_path = output_dir / "convergence.json"
            conv_path.write_text(json.dumps({k: _to_serializable(v) for k, v in convergence_summary.items()}, indent=2), encoding="utf-8")

    metadata = {
        "n": int(X_source.shape[0]),
        "p": int(X_source.shape[1]),
        "groups": groups,
        "seed": experiment_seed,
        "split": {
            "train": splits.train.tolist(),
            "val": splits.val.tolist(),
            "test": splits.test.tolist(),
        },
        "standardization": {
            "X": std_cfg.X,
            "y_center": std_cfg.y_center,
        },
        "model": model_name,
        "dataset_path": dataset_path.name,
        "posterior": {
            "saved": bool(posterior_arrays),
            "path": posterior_path.name if posterior_path else None,
            "convergence": convergence_summary,
        },
        "data": _to_serializable(dataset_metadata),
    }
    if strong_idx is not None:
        metadata["strong_idx"] = _to_serializable(strong_idx)
    if weak_idx is not None:
        metadata["weak_idx"] = _to_serializable(weak_idx)
    if active_idx is not None:
        metadata["active_idx"] = _to_serializable(active_idx)
    if feature_names is not None:
        metadata["feature_names"] = feature_names
    (output_dir / "dataset_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "status": "OK",
        "model": model_name,
        "metrics": metrics_serializable,
        "artifacts": {"dataset": dataset_path.name},
    }
