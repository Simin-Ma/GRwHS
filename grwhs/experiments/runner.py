
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
from grwhs.metrics.evaluation import evaluate_model_metrics
from grwhs.models.baselines import Ridge
from grwhs.diagnostics.convergence import summarize_convergence
from grwhs.utils.logging_utils import progress


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

    for _ in progress(range(1), total=1, desc=f"Training {model_name}"):
        try:
            model.fit(X_train, y_train, groups=groups)
        except TypeError:
            model.fit(X_train, y_train)

    metrics_cfg = config.get("experiments", {})

    coverage_level = float(metrics_cfg.get("coverage_level", 0.9))
    group_index = np.zeros(X_train.shape[1], dtype=int)
    for gid, idxs in enumerate(groups):
        group_index[np.asarray(idxs, dtype=int)] = gid

    slab_width = config.get("model", {}).get("c", None)

    results = evaluate_model_metrics(
        model=model,
        X_train=X_train,
        X_test=X_test if X_test.size else None,
        y_train=y_train,
        y_test=y_test,
        beta_truth=beta_truth,
        group_index=group_index,
        coverage_level=coverage_level,
        slab_width=slab_width,
    )

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
