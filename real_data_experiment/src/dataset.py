from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from data.loaders import load_real_dataset

from .schemas import DatasetSpec, PreparedRealDataset, PreparedSplit
from .utils import center_train_test, standardize_train_test


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_dataset_summary(loader: dict[str, object]) -> dict[str, object]:
    path_x = loader.get("path_X")
    if not path_x:
        return {}
    x_path = (PROJECT_ROOT / str(path_x)).resolve()
    candidate = x_path.parents[1] / "dataset_summary.json"
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _default_feature_names(p: int) -> list[str]:
    return [f"x_{j + 1:03d}" for j in range(int(p))]


def _default_group_labels(groups: list[list[int]]) -> list[str]:
    return [f"group_{gid + 1}" for gid in range(len(groups))]


def _resolve_group_labels(spec: DatasetSpec, n_groups: int) -> list[str]:
    labels = [str(item) for item in spec.group_labels]
    if labels and len(labels) == int(n_groups):
        return labels
    return [f"group_{gid + 1}" for gid in range(int(n_groups))]


def load_prepared_real_dataset(spec: DatasetSpec) -> PreparedRealDataset:
    loaded = load_real_dataset(spec.loader, base_dir=PROJECT_ROOT)
    if loaded.y is None:
        raise ValueError(f"Dataset '{spec.dataset_id}' does not provide y; real-data runner requires supervised data.")
    if not loaded.groups:
        raise ValueError(f"Dataset '{spec.dataset_id}' does not provide groups; grouped comparisons require group metadata.")

    X = np.asarray(loaded.X, dtype=float)
    y = np.asarray(loaded.y, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"Dataset '{spec.dataset_id}' must have 2D X; received shape={X.shape}.")
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"Dataset '{spec.dataset_id}' has incompatible shapes: X rows={X.shape[0]} vs y rows={y.shape[0]}."
        )

    covariates = None if loaded.C is None else np.asarray(loaded.C, dtype=float)
    if covariates is not None and covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    feature_names = list(loaded.feature_names or _default_feature_names(X.shape[1]))
    groups = [[int(idx) for idx in group] for group in (loaded.groups or [])]
    group_labels = _resolve_group_labels(spec, len(groups)) if groups else _default_group_labels(groups)
    covariate_feature_names = None
    if covariates is not None:
        covariate_feature_names = list(
            loaded.covariate_feature_names or [f"c_{j + 1:03d}" for j in range(covariates.shape[1])]
        )

    summary_payload = _read_dataset_summary(spec.loader)
    metadata = dict(loaded.metadata)
    if summary_payload:
        metadata["dataset_summary"] = summary_payload

    return PreparedRealDataset(
        dataset_spec=spec,
        dataset_id=str(spec.dataset_id),
        label=str(spec.label),
        X=X,
        y=y,
        groups=groups,
        group_labels=group_labels,
        feature_names=feature_names,
        covariates=covariates,
        covariate_feature_names=covariate_feature_names,
        metadata=metadata,
    )


def _split_sizes(spec: DatasetSpec, n_samples: int) -> tuple[int, int]:
    test_size = spec.test_size
    train_size = spec.train_size
    if test_size is None and train_size is None:
        test_size = max(1, int(round(float(spec.test_fraction) * int(n_samples))))
    if test_size is None:
        test_size = int(n_samples) - int(train_size)
    if train_size is None:
        train_size = int(n_samples) - int(test_size)
    test_size = max(1, int(test_size))
    train_size = max(1, int(train_size))
    if train_size + test_size > int(n_samples):
        raise ValueError(
            f"Dataset '{spec.dataset_id}' requested train_size + test_size > n: "
            f"{train_size} + {test_size} > {n_samples}."
        )
    return train_size, test_size


def _replicate_seed(master_seed: int, dataset_id: str, replicate_id: int) -> int:
    digest = hashlib.sha1(f"{dataset_id}|{replicate_id}|{master_seed}".encode("utf-8")).hexdigest()
    return int(master_seed) + int(digest[:8], 16)


def _split_hash(train_idx: np.ndarray, test_idx: np.ndarray) -> str:
    key = (
        ",".join(str(int(i)) for i in np.asarray(train_idx, dtype=int).tolist())
        + "|"
        + ",".join(str(int(i)) for i in np.asarray(test_idx, dtype=int).tolist())
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _sample_train_test_indices(
    *,
    n_samples: int,
    train_size: int,
    test_size: int,
    seed: int,
    shuffle: bool,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(int(n_samples), dtype=int)
    if bool(shuffle):
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(indices)
    else:
        perm = indices
    train_idx = np.sort(np.asarray(perm[:train_size], dtype=int))
    test_idx = np.sort(np.asarray(perm[train_size:train_size + test_size], dtype=int))
    if train_idx.size != int(train_size) or test_idx.size != int(test_size):
        raise ValueError("Failed to construct train/test indices with the requested sizes.")
    return train_idx, test_idx


def _fit_linear_projection(train_design: np.ndarray, target: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(train_design, target, rcond=None)
    return np.asarray(coef, dtype=float)


def _residualize_against_covariates(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    C_train: np.ndarray,
    C_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c_train = np.asarray(C_train, dtype=float)
    c_test = np.asarray(C_test, dtype=float)
    design_train = np.column_stack([np.ones(c_train.shape[0], dtype=float), c_train])
    design_test = np.column_stack([np.ones(c_test.shape[0], dtype=float), c_test])

    coef_x = _fit_linear_projection(design_train, np.asarray(X_train, dtype=float))
    coef_y = _fit_linear_projection(design_train, np.asarray(y_train, dtype=float).reshape(-1, 1)).reshape(-1)

    x_hat_train = design_train @ coef_x
    x_hat_test = design_test @ coef_x
    y_hat_train = design_train @ coef_y
    y_hat_test = design_test @ coef_y

    return (
        np.asarray(X_train, dtype=float) - x_hat_train,
        np.asarray(X_test, dtype=float) - x_hat_test,
        np.asarray(y_train, dtype=float).reshape(-1) - y_hat_train,
        np.asarray(y_test, dtype=float).reshape(-1) - y_hat_test,
        np.asarray(y_hat_train, dtype=float).reshape(-1),
        np.asarray(y_hat_test, dtype=float).reshape(-1),
    )


def prepare_split(
    dataset: PreparedRealDataset,
    *,
    replicate_id: int,
    master_seed: int,
) -> PreparedSplit:
    spec = dataset.dataset_spec
    train_size, test_size = _split_sizes(spec, dataset.X.shape[0])
    seed = _replicate_seed(master_seed, dataset.dataset_id, int(replicate_id))
    train_idx, test_idx = _sample_train_test_indices(
        n_samples=dataset.X.shape[0],
        train_size=train_size,
        test_size=test_size,
        seed=seed,
        shuffle=bool(spec.shuffle),
    )

    X_train = np.asarray(dataset.X[train_idx], dtype=float)
    X_test = np.asarray(dataset.X[test_idx], dtype=float)
    y_train = np.asarray(dataset.y[train_idx], dtype=float).reshape(-1)
    y_test = np.asarray(dataset.y[test_idx], dtype=float).reshape(-1)
    C_train = None if dataset.covariates is None else np.asarray(dataset.covariates[train_idx], dtype=float)
    C_test = None if dataset.covariates is None else np.asarray(dataset.covariates[test_idx], dtype=float)

    covariate_mode = str(spec.covariate_mode).strip().lower()
    if covariate_mode == "none":
        X_model_train_raw = X_train
        X_model_test_raw = X_test
        y_model_train_raw = y_train
        y_model_test_raw = y_test
        pred_offset_train = np.zeros_like(y_train, dtype=float)
        pred_offset_test = np.zeros_like(y_test, dtype=float)
    elif covariate_mode == "residualize":
        if C_train is None or C_test is None:
            raise ValueError(
                f"Dataset '{dataset.dataset_id}' requested covariate_mode='residualize' but no covariates were loaded."
            )
        (
            X_model_train_raw,
            X_model_test_raw,
            y_model_train_raw,
            y_model_test_raw,
            pred_offset_train,
            pred_offset_test,
        ) = _residualize_against_covariates(X_train, X_test, y_train, y_test, C_train, C_test)
    else:
        raise ValueError(
            f"Unsupported covariate_mode for dataset '{dataset.dataset_id}': {spec.covariate_mode!r}."
        )

    X_train_used, X_test_used, x_center, x_scale = standardize_train_test(
        X_model_train_raw,
        X_model_test_raw,
        center=True,
        scale=True,
    )

    response_mode = str(spec.response_standardization).strip().lower()
    if response_mode == "train_center":
        y_train_used, y_test_used, y_offset = center_train_test(y_model_train_raw, y_model_test_raw)
        y_scale = 1.0
    elif response_mode == "none":
        y_train_used = np.asarray(y_model_train_raw, dtype=float).reshape(-1)
        y_test_used = np.asarray(y_model_test_raw, dtype=float).reshape(-1)
        y_offset = 0.0
        y_scale = 1.0
    else:
        raise ValueError(
            f"Unsupported response_standardization for dataset '{dataset.dataset_id}': "
            f"{spec.response_standardization!r}."
        )

    return PreparedSplit(
        dataset=dataset,
        replicate_id=int(replicate_id),
        seed=int(seed),
        split_hash=_split_hash(train_idx, test_idx),
        train_idx=train_idx,
        test_idx=test_idx,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        groups=[[int(idx) for idx in group] for group in dataset.groups],
        covariates_train=C_train,
        covariates_test=C_test,
        X_train_used=np.asarray(X_train_used, dtype=float),
        X_test_used=np.asarray(X_test_used, dtype=float),
        y_train_used=np.asarray(y_train_used, dtype=float).reshape(-1),
        y_test_used=np.asarray(y_test_used, dtype=float).reshape(-1),
        prediction_offset_train=np.asarray(pred_offset_train, dtype=float).reshape(-1),
        prediction_offset_test=np.asarray(pred_offset_test, dtype=float).reshape(-1),
        x_center=np.asarray(x_center, dtype=float).reshape(-1),
        x_scale=np.asarray(x_scale, dtype=float).reshape(-1),
        y_offset=float(y_offset),
        y_scale=float(y_scale),
        preprocessing={
            "covariate_mode": str(spec.covariate_mode),
            "response_standardization": str(spec.response_standardization),
            "train_size": int(train_size),
            "test_size": int(test_size),
        },
        metadata={
            "dataset_id": str(dataset.dataset_id),
            "dataset_label": str(dataset.label),
            "target_label": str(spec.target_label),
            "feature_count": int(dataset.X.shape[1]),
            "group_count": int(len(dataset.groups)),
            "sample_count": int(dataset.X.shape[0]),
        },
    )


def save_prepared_split(split: PreparedSplit, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_idx.npy", np.asarray(split.train_idx, dtype=int))
    np.save(out_dir / "test_idx.npy", np.asarray(split.test_idx, dtype=int))
    np.save(out_dir / "X_train_used.npy", np.asarray(split.X_train_used, dtype=float))
    np.save(out_dir / "X_test_used.npy", np.asarray(split.X_test_used, dtype=float))
    np.save(out_dir / "y_train_used.npy", np.asarray(split.y_train_used, dtype=float))
    np.save(out_dir / "y_test_used.npy", np.asarray(split.y_test_used, dtype=float))
    manifest = {
        "dataset_id": str(split.dataset.dataset_id),
        "dataset_label": str(split.dataset.label),
        "replicate_id": int(split.replicate_id),
        "seed": int(split.seed),
        "split_hash": str(split.split_hash),
        "train_size": int(split.train_idx.size),
        "test_size": int(split.test_idx.size),
        "feature_count": int(split.X_train.shape[1]),
        "group_count": int(len(split.groups)),
        "group_sizes": [int(len(group)) for group in split.groups],
        "group_labels": list(split.dataset.group_labels),
        "preprocessing": dict(split.preprocessing),
        "metadata": dict(split.metadata),
    }
    path = out_dir / "split_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "split_manifest": str(path),
        "train_idx": str(out_dir / "train_idx.npy"),
        "test_idx": str(out_dir / "test_idx.npy"),
        "X_train_used": str(out_dir / "X_train_used.npy"),
        "X_test_used": str(out_dir / "X_test_used.npy"),
        "y_train_used": str(out_dir / "y_train_used.npy"),
        "y_test_used": str(out_dir / "y_test_used.npy"),
    }
