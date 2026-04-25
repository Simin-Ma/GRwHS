from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np

from .blueprint import sample_signal_blueprint
from .schemas import FamilySpec, GroupedRegressionDataset, SettingSpec
from .utils import canonical_groups, ensure_dir, sample_correlated_design, save_json, setting_replicate_seed


def sigma2_for_target_r2(beta: np.ndarray, cov_x: np.ndarray, target_r2: float) -> Tuple[float, float]:
    beta_arr = np.asarray(beta, dtype=float).reshape(-1)
    signal_var = float(beta_arr.T @ np.asarray(cov_x, dtype=float) @ beta_arr)
    if signal_var <= 1e-12:
        signal_var = 1e-12
    r2 = min(max(float(target_r2), 1e-6), 1.0 - 1e-6)
    sigma2 = signal_var * (1.0 - r2) / r2
    return float(sigma2), float(signal_var)


def generate_grouped_dataset(
    setting: SettingSpec,
    *,
    replicate_id: int = 1,
    master_seed: int = 20260425,
    family_specs: Mapping[str, FamilySpec] | None = None,
) -> GroupedRegressionDataset:
    seed = setting_replicate_seed(setting.setting_id, replicate_id, master_seed=master_seed)
    signal_rng = np.random.default_rng(seed)
    design_seed = int(seed + 101)
    test_seed = int(seed + 202)
    noise_seed = int(seed + 303)
    test_noise_seed = int(seed + 404)

    signal_draw = sample_signal_blueprint(setting, signal_rng, family_specs=family_specs)
    X_train, cov_x = sample_correlated_design(
        n=setting.n_train,
        group_sizes=setting.group_sizes,
        rho_within=setting.rho_within,
        rho_between=setting.rho_between,
        seed=design_seed,
    )
    X_test, _ = sample_correlated_design(
        n=setting.n_test,
        group_sizes=setting.group_sizes,
        rho_within=setting.rho_within,
        rho_between=setting.rho_between,
        seed=test_seed,
    )

    sigma2, signal_var = sigma2_for_target_r2(signal_draw.beta, cov_x, setting.target_r2)
    noise_rng = np.random.default_rng(noise_seed)
    test_noise_rng = np.random.default_rng(test_noise_seed)
    y_train = X_train @ signal_draw.beta + noise_rng.normal(0.0, np.sqrt(sigma2), size=setting.n_train)
    y_test = X_test @ signal_draw.beta + test_noise_rng.normal(0.0, np.sqrt(sigma2), size=setting.n_test)

    groups = canonical_groups(setting.group_sizes)
    metadata: Dict[str, object] = {
        "seed": int(seed),
        "replicate_id": int(replicate_id),
        "signal_variance_population": float(signal_var),
        "target_r2": float(setting.target_r2),
        "implied_population_r2": float(signal_var / (signal_var + sigma2)),
        "rho_within": float(setting.rho_within),
        "rho_between": float(setting.rho_between),
        "n_train": int(setting.n_train),
        "n_test": int(setting.n_test),
    }

    return GroupedRegressionDataset(
        setting=setting,
        X_train=X_train.astype(np.float64, copy=False),
        y_train=y_train.astype(np.float64, copy=False),
        X_test=X_test.astype(np.float64, copy=False),
        y_test=y_test.astype(np.float64, copy=False),
        beta=signal_draw.beta.astype(np.float64, copy=False),
        sigma2=float(sigma2),
        cov_x=cov_x.astype(np.float64, copy=False),
        groups=groups,
        signal_draw=signal_draw,
        metadata=metadata,
    )


def save_grouped_dataset(dataset: GroupedRegressionDataset, out_dir: Path | str) -> Dict[str, str]:
    root = ensure_dir(out_dir)
    prefix = f"{dataset.setting.setting_id}_rep{int(dataset.metadata.get('replicate_id', 1)):03d}"
    arrays_path = root / f"{prefix}.npz"
    meta_path = root / f"{prefix}.json"

    np.savez_compressed(
        arrays_path,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        beta=dataset.beta,
        cov_x=dataset.cov_x,
    )
    save_json(dataset.summary(), meta_path)

    return {
        "arrays": str(arrays_path),
        "metadata": str(meta_path),
    }
