
"""Experiment orchestration entry points."""
from __future__ import annotations

import json
import secrets
from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping

import numpy as np

from data.generators import generate_synthetic, synthetic_config_from_dict, make_groups
from data.preprocess import StandardizationConfig, apply_standardization
from data.splits import train_val_test_split
from data.loaders import load_real_dataset
from grwhs.experiments.registry import build_from_config, get_model_name_from_config
from grwhs.metrics.evaluation import evaluate_model_metrics
from grwhs.models.baselines import Ridge, LogisticRegressionClassifier
from grwhs.diagnostics.convergence import summarize_convergence
from grwhs.utils.logging_utils import progress
from grwhs.postprocess.debias import apply_debias_refit


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

def _bump_seed(value: Any, offset: int) -> Any:
    if offset == 0 or value is None:
        return value
    try:
        return int(value) + offset
    except (TypeError, ValueError):
        return value


def _adjust_seeds_for_repeat(cfg: Dict[str, Any], offset: int) -> None:
    """Offset inference seeds for later repeats while keeping data seeds fixed."""
    if offset == 0:
        return
    inference_cfg = cfg.get("inference")
    if isinstance(inference_cfg, dict):
        for section in inference_cfg.values():
            if isinstance(section, dict) and "seed" in section:
                section["seed"] = _bump_seed(section.get("seed"), offset)



def _run_single_experiment(
    config: Dict[str, Any],
    output_dir: Path,
    *,
    repeat_index: int = 1,
    total_repeats: int = 1,
) -> Dict[str, Any]:
    """Execute a single experiment (synthetic or loader-backed) based on configuration."""

    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config.get("data", {})
    data_type = str(data_cfg.get("type", "synthetic")).lower()
    seed_cfg = config.get("seeds", {})

    task_aliases = {
        "binary": "classification",
        "binary_classification": "classification",
        "cls": "classification",
    }
    task_raw = config.get("task", data_cfg.get("task", "regression"))
    task_norm = str(task_raw).lower()
    task = task_aliases.get(task_norm, task_norm)
    if task not in {"regression", "classification"}:
        raise ValueError(f"Unsupported task '{task_raw}'. Expected 'regression' or 'classification'.")

    def _resolve_seed(*candidates: Any) -> Optional[int]:
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue
        return None

    base_seed = _resolve_seed(seed_cfg.get("experiment"), config.get("seed"))
    split_seed = _resolve_seed(seed_cfg.get("split"), base_seed)

    dataset_metadata: Dict[str, Any] = {}
    feature_names: Optional[List[str]] = None

    strong_idx = weak_idx = active_idx = None

    X_source: np.ndarray
    y_source: Optional[np.ndarray]
    beta_truth: Optional[np.ndarray]
    groups: List[List[int]] | None
    experiment_seed: Optional[int] = base_seed
    data_seed: Optional[int] = _resolve_seed(data_cfg.get("seed"), seed_cfg.get("data_generation"))
    if data_seed is None:
        data_seed = int(secrets.randbits(32))

    if data_type == "synthetic":
        response_override: Dict[str, Any] | None = None
        if task == "classification":
            override_dict: Dict[str, Any] = {}
            class_cfg_data = data_cfg.get("classification")
            if isinstance(class_cfg_data, Mapping):
                override_dict.update(dict(class_cfg_data))
            class_cfg_global = config.get("classification")
            if isinstance(class_cfg_global, Mapping):
                override_dict.update(dict(class_cfg_global))
            if override_dict:
                response_override = override_dict
        syn_cfg = synthetic_config_from_dict(
            data_cfg,
            seed=data_seed,
            name=config.get("name"),
            task=task,
            response_override=response_override,
        )
        dataset = generate_synthetic(syn_cfg)
        X_source = dataset.X
        y_source = dataset.y
        beta_truth = dataset.beta
        groups = dataset.groups
        dataset_metadata.update(dataset.info)
        dataset_metadata.setdefault("scenario", syn_cfg.name)
        dataset_metadata["task"] = task
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
        dataset_metadata["task"] = task
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
    if "y_center" in std_cfg_dict:
        y_center_flag = bool(std_cfg_dict.get("y_center"))
    else:
        y_center_flag = task != "classification"
    std_cfg = StandardizationConfig(
        X=std_cfg_dict.get("X", "unit_variance"),
        y_center=y_center_flag,
    )
    std_result = apply_standardization(X_source, y_source, std_cfg)

    splits = train_val_test_split(
        n=std_result.X.shape[0],
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        test_ratio=float(data_cfg.get("test_ratio", 0.2)),
        seed=split_seed,
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
        print(f"[WARN] Model construction failed ({exc}); using fallback estimator.")
        model_cfg = config.get("model", {})
        if task == "classification":
            penalty = str(model_cfg.get("penalty", "l2"))
            solver = str(model_cfg.get("solver", "lbfgs"))
            max_iter = int(model_cfg.get("max_iter", 200))
            model = LogisticRegressionClassifier(
                penalty=penalty,
                solver=solver,
                max_iter=max_iter,
            )
            model_name = "logistic_regression_fallback"
        else:
            fit_intercept = bool(model_cfg.get("fit_intercept", False))
            alpha = float(model_cfg.get("alpha", 1.0))
            model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
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
    classification_threshold = float(metrics_cfg.get("classification_threshold", 0.5))
    classification_threshold = min(max(classification_threshold, 0.0), 1.0)
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
        task=task,
        classification_threshold=classification_threshold,
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
    dataset_arrays = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    metrics_path = output_dir / "metrics.json"

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

    postprocess_cfg = config.get("postprocess", {})
    debias_summary: Dict[str, Any] | None = None
    if isinstance(postprocess_cfg, dict):
        debias_cfg = postprocess_cfg.get("debias")
        if isinstance(debias_cfg, dict) and debias_cfg.get("enabled", False):
            if not posterior_arrays:
                debias_summary = {"error": "posterior_samples not available (set experiments.save_posterior=true)"}
            else:
                try:
                    debias_summary = apply_debias_refit(
                        run_dir=output_dir,
                        config=debias_cfg,
                        dataset=dataset_arrays,
                        posterior_arrays=posterior_arrays,
                    )
                except Exception as exc:  # pragma: no cover - safety net
                    debias_summary = {"error": str(exc)}
            if debias_summary:
                if "rmse_test_debiased" in debias_summary:
                    metrics_serializable["RMSE_DebiasTest"] = _to_serializable(debias_summary["rmse_test_debiased"])
                if "rmse_val_debiased" in debias_summary:
                    metrics_serializable["RMSE_DebiasVal"] = _to_serializable(debias_summary["rmse_val_debiased"])
                if "rmse_test_gain_pct" in debias_summary:
                    metrics_serializable["RMSE_DebiasGainPct"] = _to_serializable(debias_summary["rmse_test_gain_pct"])

    metrics_path.write_text(json.dumps(metrics_serializable, indent=2), encoding="utf-8")

    inference_seeds: Dict[str, int] = {}
    inference_cfg = config.get("inference", {})
    if isinstance(inference_cfg, dict):
        for key, section in inference_cfg.items():
            if not isinstance(section, dict):
                continue
            stage_seed = _resolve_seed(section.get("seed"))
            if stage_seed is not None:
                inference_seeds[key] = stage_seed

    runtime_cfg = config.get("runtime", {})
    runtime_seed = _resolve_seed(runtime_cfg.get("seed"))
    if runtime_seed is not None and "runtime" not in inference_seeds:
        inference_seeds["runtime"] = runtime_seed

    model_seed = _resolve_seed(config.get("model", {}).get("seed"))
    if model_seed is not None:
        inference_seeds.setdefault("model", model_seed)

    split_seed_log: Dict[str, int] = {}
    if split_seed is not None:
        split_seed_log["train_test"] = split_seed
        split_seed_log["train_val"] = split_seed + 1

    data_seed_logged = _resolve_seed(dataset_metadata.get("seed"), data_seed, base_seed)

    config_seed_defaults: Dict[str, int] = {}
    if isinstance(seed_cfg, dict):
        for key, value in seed_cfg.items():
            resolved = _resolve_seed(value)
            if resolved is not None:
                config_seed_defaults[key] = resolved

    seed_log: Dict[str, Any] = {}
    if experiment_seed is not None:
        seed_log["experiment"] = int(experiment_seed)
    if data_seed_logged is not None:
        seed_log["data_generation"] = int(data_seed_logged)
    if split_seed_log:
        seed_log["split"] = split_seed_log
    if inference_seeds:
        seed_log["inference"] = inference_seeds
    if config_seed_defaults:
        seed_log["config"] = config_seed_defaults

    metadata = {
        "n": int(X_source.shape[0]),
        "p": int(X_source.shape[1]),
        "groups": groups,
        "seed": experiment_seed,
        "task": task,
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
        "repeat": {
            "index": int(repeat_index),
            "total": int(total_repeats),
        },
        "posterior": {
            "saved": bool(posterior_arrays),
            "path": posterior_path.name if posterior_path else None,
            "convergence": convergence_summary,
        },
        "data": _to_serializable(dataset_metadata),
    }
    if seed_log:
        metadata["seeds"] = seed_log
    if strong_idx is not None:
        metadata["strong_idx"] = _to_serializable(strong_idx)
    if weak_idx is not None:
        metadata["weak_idx"] = _to_serializable(weak_idx)
    if active_idx is not None:
        metadata["active_idx"] = _to_serializable(active_idx)
    if feature_names is not None:
        metadata["feature_names"] = feature_names
    if debias_summary is not None:
        metadata.setdefault("postprocess", {})["debias"] = _to_serializable(debias_summary)
    (output_dir / "dataset_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    result: Dict[str, Any] = {
        "status": "OK",
        "model": model_name,
        "metrics": metrics_serializable,
        "artifacts": {"dataset": dataset_path.name},
        "repeat": {"index": int(repeat_index), "total": int(total_repeats)},
    }
    if debias_summary is not None:
        result.setdefault("postprocess", {})["debias"] = _to_serializable(debias_summary)
    if seed_log:
        result["seeds"] = seed_log
    result.setdefault("artifacts", {}).setdefault("repeat_dirs", []).append(str(output_dir))
    return result


def run_experiment(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Execute one or multiple experiment repeats based on configuration.

    When ``experiments.repeats`` is greater than 1, this function launches
    independent runs under ``output_dir/repeat_XXX`` directories and returns
    an aggregated metrics summary alongside per-repeat metrics.
    """

    experiments_cfg = config.get("experiments", {})
    repeats_raw = experiments_cfg.get("repeats", 1)
    try:
        repeats = int(repeats_raw)
    except (TypeError, ValueError):
        repeats = 1
    repeats = max(repeats, 1)

    output_dir = Path(output_dir)

    if repeats == 1:
        single_result = _run_single_experiment(
            deepcopy(config),
            output_dir,
            repeat_index=1,
            total_repeats=1,
        )
        single_result.setdefault(
            "repeat_metrics",
            [
                {
                    "repeat": 1,
                    "output_dir": str(output_dir),
                    "metrics": single_result.get("metrics", {}),
                    "status": single_result.get("status", "OK"),
                }
            ],
        )
        single_result.setdefault("repeats", 1)
        return single_result

    output_dir.mkdir(parents=True, exist_ok=True)

    repeat_entries: List[Dict[str, Any]] = []
    numeric_metrics: Dict[str, List[float]] = {}
    statuses: List[str] = []
    model_name: Optional[str] = None

    for idx in range(repeats):
        repeat_dir = output_dir / f"repeat_{idx + 1:03d}"
        repeat_config = deepcopy(config)
        base_name = repeat_config.get("name")
        if isinstance(base_name, str) and base_name:
            repeat_config["name"] = f"{base_name}_repeat{idx + 1}"

        _adjust_seeds_for_repeat(repeat_config, idx)

        single_result = _run_single_experiment(
            repeat_config,
            repeat_dir,
            repeat_index=idx + 1,
            total_repeats=repeats,
        )

        if model_name is None:
            model_name = single_result.get("model")

        statuses.append(single_result.get("status", "OK"))

        entry: Dict[str, Any] = {
            "repeat": idx + 1,
            "output_dir": str(repeat_dir),
            "metrics": single_result.get("metrics", {}),
            "status": single_result.get("status", "OK"),
        }
        if "seeds" in single_result:
            entry["seeds"] = single_result["seeds"]
        repeat_entries.append(entry)

        for key, value in single_result.get("metrics", {}).items():
            if isinstance(value, Number):
                numeric_metrics.setdefault(key, []).append(float(value))

    summary_metrics = {
        key: (sum(values) / len(values))
        for key, values in numeric_metrics.items()
        if values
    }

    status = "OK"
    for st in statuses:
        if st != "OK":
            status = st
            break

    result: Dict[str, Any] = {
        "status": status,
        "metrics": summary_metrics,
        "repeat_metrics": repeat_entries,
        "artifacts": {"repeat_dirs": [entry["output_dir"] for entry in repeat_entries]},
        "repeats": repeats,
    }
    if model_name is not None:
        result["model"] = model_name
    return result
