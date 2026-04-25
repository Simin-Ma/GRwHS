from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np

from .schemas import MechanismDataset, MechanismSettingSpec, active_group_mask
from .utils import (
    canonical_groups,
    ensure_dir,
    nearest_positive_definite,
    sample_correlated_design,
    save_json,
    setting_replicate_seed,
)


def sigma2_for_target_snr(beta: np.ndarray, cov_x: np.ndarray, target_snr: float) -> tuple[float, float]:
    beta_arr = np.asarray(beta, dtype=float).reshape(-1)
    signal_var = float(beta_arr.T @ np.asarray(cov_x, dtype=float) @ beta_arr)
    if signal_var <= 1e-12:
        signal_var = 1e-12
    snr = max(float(target_snr), 1e-8)
    sigma2 = signal_var / snr
    return float(sigma2), float(signal_var)


def _sample_mixed_decoy_design(
    *,
    n: int,
    group_sizes: tuple[int, ...],
    active_groups: tuple[int, ...],
    rho_within: float,
    rho_between: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    groups = canonical_groups(group_sizes)
    active = sorted({int(g) for g in active_groups if 0 <= int(g) < len(groups)})
    null_candidates = [gid for gid in range(len(groups)) if gid not in set(active)]
    decoy_group = int(null_candidates[0]) if null_candidates else None

    from .utils import block_correlation, standardize_columns

    cov = block_correlation(group_sizes, rho_within=float(rho_within), rho_between=float(rho_between))
    if active and decoy_group is not None:
        primary_idx = np.asarray(groups[int(active[0])], dtype=int)
        decoy_idx = np.asarray(groups[int(decoy_group)], dtype=int)
        if primary_idx.size and decoy_idx.size:
            load = np.zeros(int(sum(group_sizes)), dtype=float)
            load[primary_idx] = 1.0 / np.sqrt(float(primary_idx.size))
            load[decoy_idx] = 0.85 / np.sqrt(float(decoy_idx.size))
            cov = cov + float(max(rho_within - rho_between, 0.05)) * 0.35 * np.outer(load, load)
            cov = nearest_positive_definite(cov)

    rng = np.random.default_rng(int(seed))
    X = rng.multivariate_normal(np.zeros(cov.shape[0], dtype=float), cov, size=int(n))
    X = standardize_columns(X)
    return X, cov, decoy_group


def _build_correlation_stress_beta(setting: MechanismSettingSpec) -> np.ndarray:
    groups = canonical_groups(setting.group_sizes)
    beta = np.zeros(int(sum(setting.group_sizes)), dtype=float)
    active = sorted({int(g) for g in setting.active_groups if 0 <= int(g) < len(groups)})
    if not active:
        raise ValueError("M2 requires at least one active group.")

    pattern = str(setting.within_group_pattern).strip().lower()
    for gid in active:
        idx = np.asarray(groups[gid], dtype=int)
        if idx.size == 0:
            continue
        if pattern == "concentrated":
            beta[idx[0]] = 1.0
        elif pattern == "distributed":
            beta[idx] = 1.0 / np.sqrt(float(idx.size))
        elif pattern == "mixed_decoy":
            beta[idx[0]] = 1.0
            if idx.size > 1:
                beta[idx[1:]] = 0.25
        else:
            raise ValueError(f"Unknown within_group_pattern for M2: {setting.within_group_pattern!r}")
    return beta


def _build_complexity_mismatch_beta(setting: MechanismSettingSpec) -> np.ndarray:
    groups = canonical_groups(setting.group_sizes)
    total_active = max(1, int(setting.total_active_coeff))
    beta = np.zeros(int(sum(setting.group_sizes)), dtype=float)
    pattern = str(setting.complexity_pattern).strip().lower()
    within = str(setting.within_group_pattern).strip().lower()

    if pattern == "few_groups":
        active_group_ids = [0]
    elif pattern == "many_groups":
        max_active_groups = max(2, len(groups) - 2)
        active_count = min(max_active_groups, max(2, total_active))
        active_group_ids = list(range(active_count))
    else:
        raise ValueError(f"Unknown complexity_pattern: {setting.complexity_pattern!r}")

    if within == "concentrated":
        remaining = int(total_active)
        per_group = int(math.ceil(total_active / max(len(active_group_ids), 1)))
        for pos, gid in enumerate(active_group_ids):
            idx = np.asarray(groups[gid], dtype=int)
            if idx.size == 0:
                continue
            take = min(int(idx.size), max(1, remaining if pos == len(active_group_ids) - 1 else per_group))
            beta[idx[:take]] = 1.0
            remaining = max(0, remaining - take)
            if remaining <= 0:
                break
    elif within == "distributed":
        weights = np.zeros(len(groups), dtype=int)
        base = max(1, int(total_active // max(len(active_group_ids), 1)))
        rem = int(total_active)
        for gid in active_group_ids:
            weights[gid] = min(int(len(groups[gid])), base)
            rem -= weights[gid]
        ptr = 0
        while rem > 0 and active_group_ids:
            gid = active_group_ids[ptr % len(active_group_ids)]
            if weights[gid] < len(groups[gid]):
                weights[gid] += 1
                rem -= 1
            ptr += 1
            if ptr > 10000:
                break
        for gid in active_group_ids:
            idx = np.asarray(groups[gid], dtype=int)
            k = int(max(0, min(weights[gid], idx.size)))
            if k > 0:
                beta[idx[:k]] = 1.0 / np.sqrt(float(k))
    else:
        raise ValueError(f"Unknown within_group_pattern for M3: {setting.within_group_pattern!r}")
    return beta


def _generate_group_separation_dataset(setting: MechanismSettingSpec, *, seed: int, replicate_id: int) -> MechanismDataset:
    groups = canonical_groups(setting.group_sizes)
    X_train, cov_x = sample_correlated_design(
        n=setting.n_train,
        group_sizes=setting.group_sizes,
        rho_within=setting.rho_within,
        rho_between=setting.rho_between,
        seed=seed + 101,
    )
    X_test, _ = sample_correlated_design(
        n=setting.n_test,
        group_sizes=setting.group_sizes,
        rho_within=setting.rho_within,
        rho_between=setting.rho_between,
        seed=seed + 202,
    )

    sigma2 = float(setting.sigma2 if setting.sigma2 is not None else 1.0)
    beta = np.zeros(int(sum(setting.group_sizes)), dtype=float)
    for gid, mu_g in enumerate(setting.mu):
        idx = np.asarray(groups[gid], dtype=int)
        if float(mu_g) <= 0.0:
            continue
        beta[idx] = math.sqrt((2.0 * sigma2 * float(mu_g)) / max(int(idx.size), 1))

    rng_train = np.random.default_rng(seed + 303)
    rng_test = np.random.default_rng(seed + 404)
    y_train = X_train @ beta + rng_train.normal(0.0, np.sqrt(sigma2), size=setting.n_train)
    y_test = X_test @ beta + rng_test.normal(0.0, np.sqrt(sigma2), size=setting.n_test)
    signal_var = float(beta.T @ cov_x @ beta)
    active_mask = active_group_mask(beta, groups)

    metadata = {
        "seed": int(seed),
        "replicate_id": int(replicate_id),
        "signal_variance_population": float(signal_var),
        "target_snr": float(signal_var / max(sigma2, 1e-12)),
        "implied_population_snr": float(signal_var / max(sigma2, 1e-12)),
        "active_group_ids": [int(i) for i in np.flatnonzero(active_mask)],
        "p0_true": int(np.count_nonzero(beta)),
        "p0_groups_true": int(active_mask.sum()),
        "decoy_group": -1,
    }
    return MechanismDataset(
        setting=setting,
        X_train=X_train.astype(np.float64, copy=False),
        y_train=y_train.astype(np.float64, copy=False),
        X_test=X_test.astype(np.float64, copy=False),
        y_test=y_test.astype(np.float64, copy=False),
        beta=beta.astype(np.float64, copy=False),
        sigma2=float(sigma2),
        cov_x=cov_x.astype(np.float64, copy=False),
        groups=groups,
        metadata=metadata,
    )


def _generate_correlation_stress_dataset(setting: MechanismSettingSpec, *, seed: int, replicate_id: int) -> MechanismDataset:
    groups = canonical_groups(setting.group_sizes)
    beta = _build_correlation_stress_beta(setting)
    pattern = str(setting.within_group_pattern).strip().lower()
    decoy_group = -1
    if pattern == "mixed_decoy":
        X_train, cov_x, decoy = _sample_mixed_decoy_design(
            n=setting.n_train,
            group_sizes=setting.group_sizes,
            active_groups=setting.active_groups,
            rho_within=setting.rho_within,
            rho_between=setting.rho_between,
            seed=seed + 101,
        )
        X_test, _, _ = _sample_mixed_decoy_design(
            n=setting.n_test,
            group_sizes=setting.group_sizes,
            active_groups=setting.active_groups,
            rho_within=setting.rho_within,
            rho_between=setting.rho_between,
            seed=seed + 202,
        )
        decoy_group = -1 if decoy is None else int(decoy)
    else:
        X_train, cov_x = sample_correlated_design(
            n=setting.n_train,
            group_sizes=setting.group_sizes,
            rho_within=setting.rho_within,
            rho_between=setting.rho_between,
            seed=seed + 101,
        )
        X_test, _ = sample_correlated_design(
            n=setting.n_test,
            group_sizes=setting.group_sizes,
            rho_within=setting.rho_within,
            rho_between=setting.rho_between,
            seed=seed + 202,
        )
    sigma2, signal_var = sigma2_for_target_snr(beta, cov_x, setting.target_snr)
    rng_train = np.random.default_rng(seed + 303)
    rng_test = np.random.default_rng(seed + 404)
    y_train = X_train @ beta + rng_train.normal(0.0, np.sqrt(sigma2), size=setting.n_train)
    y_test = X_test @ beta + rng_test.normal(0.0, np.sqrt(sigma2), size=setting.n_test)
    active_mask = active_group_mask(beta, groups)

    metadata = {
        "seed": int(seed),
        "replicate_id": int(replicate_id),
        "signal_variance_population": float(signal_var),
        "target_snr": float(setting.target_snr),
        "implied_population_snr": float(signal_var / max(sigma2, 1e-12)),
        "active_group_ids": [int(i) for i in np.flatnonzero(active_mask)],
        "p0_true": int(np.count_nonzero(beta)),
        "p0_groups_true": int(active_mask.sum()),
        "decoy_group": int(decoy_group),
    }
    return MechanismDataset(
        setting=setting,
        X_train=X_train.astype(np.float64, copy=False),
        y_train=y_train.astype(np.float64, copy=False),
        X_test=X_test.astype(np.float64, copy=False),
        y_test=y_test.astype(np.float64, copy=False),
        beta=beta.astype(np.float64, copy=False),
        sigma2=float(sigma2),
        cov_x=np.asarray(cov_x, dtype=np.float64),
        groups=groups,
        metadata=metadata,
    )


def _generate_complexity_dataset(setting: MechanismSettingSpec, *, seed: int, replicate_id: int) -> MechanismDataset:
    groups = canonical_groups(setting.group_sizes)
    beta = _build_complexity_mismatch_beta(setting)
    X_train, cov_x = sample_correlated_design(
        n=setting.n_train,
        group_sizes=setting.group_sizes,
        rho_within=setting.rho_within,
        rho_between=setting.rho_between,
        seed=seed + 101,
    )
    X_test, _ = sample_correlated_design(
        n=setting.n_test,
        group_sizes=setting.group_sizes,
        rho_within=setting.rho_within,
        rho_between=setting.rho_between,
        seed=seed + 202,
    )
    sigma2, signal_var = sigma2_for_target_snr(beta, cov_x, setting.target_snr)
    rng_train = np.random.default_rng(seed + 303)
    rng_test = np.random.default_rng(seed + 404)
    y_train = X_train @ beta + rng_train.normal(0.0, np.sqrt(sigma2), size=setting.n_train)
    y_test = X_test @ beta + rng_test.normal(0.0, np.sqrt(sigma2), size=setting.n_test)
    active_mask = active_group_mask(beta, groups)

    metadata = {
        "seed": int(seed),
        "replicate_id": int(replicate_id),
        "signal_variance_population": float(signal_var),
        "target_snr": float(setting.target_snr),
        "implied_population_snr": float(signal_var / max(sigma2, 1e-12)),
        "active_group_ids": [int(i) for i in np.flatnonzero(active_mask)],
        "p0_true": int(np.count_nonzero(beta)),
        "p0_groups_true": int(active_mask.sum()),
        "decoy_group": -1,
    }
    return MechanismDataset(
        setting=setting,
        X_train=X_train.astype(np.float64, copy=False),
        y_train=y_train.astype(np.float64, copy=False),
        X_test=X_test.astype(np.float64, copy=False),
        y_test=y_test.astype(np.float64, copy=False),
        beta=beta.astype(np.float64, copy=False),
        sigma2=float(sigma2),
        cov_x=np.asarray(cov_x, dtype=np.float64),
        groups=groups,
        metadata=metadata,
    )


def _generate_ablation_dataset(setting: MechanismSettingSpec, *, seed: int, replicate_id: int) -> MechanismDataset:
    groups = canonical_groups(setting.group_sizes)
    X_train, cov_x = sample_correlated_design(
        n=setting.n_train,
        group_sizes=setting.group_sizes,
        rho_within=setting.rho_within,
        rho_between=setting.rho_between,
        seed=seed + 101,
    )
    X_test, _ = sample_correlated_design(
        n=setting.n_test,
        group_sizes=setting.group_sizes,
        rho_within=setting.rho_within,
        rho_between=setting.rho_between,
        seed=seed + 202,
    )

    rng = np.random.default_rng(seed + 303)
    beta = np.zeros(int(sum(setting.group_sizes)), dtype=float)
    p0_true = max(1, int(setting.total_active_coeff))
    active = rng.choice(np.arange(beta.size), size=p0_true, replace=False)
    n_strong = max(1, int(math.ceil(0.5 * p0_true)))
    beta[active[:n_strong]] = 2.0
    if active[n_strong:].size > 0:
        beta[active[n_strong:]] = 0.5

    sigma2 = float(setting.sigma2 if setting.sigma2 is not None else 1.0)
    rng_train = np.random.default_rng(seed + 404)
    rng_test = np.random.default_rng(seed + 505)
    y_train = X_train @ beta + rng_train.normal(0.0, np.sqrt(sigma2), size=setting.n_train)
    y_test = X_test @ beta + rng_test.normal(0.0, np.sqrt(sigma2), size=setting.n_test)
    signal_var = float(beta.T @ cov_x @ beta)
    active_mask = active_group_mask(beta, groups)

    metadata = {
        "seed": int(seed),
        "replicate_id": int(replicate_id),
        "signal_variance_population": float(signal_var),
        "target_snr": float(signal_var / max(sigma2, 1e-12)),
        "implied_population_snr": float(signal_var / max(sigma2, 1e-12)),
        "active_group_ids": [int(i) for i in np.flatnonzero(active_mask)],
        "p0_true": int(np.count_nonzero(beta)),
        "p0_groups_true": int(active_mask.sum()),
        "decoy_group": -1,
    }
    return MechanismDataset(
        setting=setting,
        X_train=X_train.astype(np.float64, copy=False),
        y_train=y_train.astype(np.float64, copy=False),
        X_test=X_test.astype(np.float64, copy=False),
        y_test=y_test.astype(np.float64, copy=False),
        beta=beta.astype(np.float64, copy=False),
        sigma2=float(sigma2),
        cov_x=np.asarray(cov_x, dtype=np.float64),
        groups=groups,
        metadata=metadata,
    )


def generate_mechanism_dataset(
    setting: MechanismSettingSpec,
    *,
    replicate_id: int = 1,
    master_seed: int = 20260425,
) -> MechanismDataset:
    seed = setting_replicate_seed(setting.setting_id, replicate_id, master_seed=master_seed)
    kind = str(setting.experiment_kind).strip().lower()
    if kind == "group_separation":
        return _generate_group_separation_dataset(setting, seed=seed, replicate_id=replicate_id)
    if kind == "correlation_stress":
        return _generate_correlation_stress_dataset(setting, seed=seed, replicate_id=replicate_id)
    if kind == "complexity_mismatch":
        return _generate_complexity_dataset(setting, seed=seed, replicate_id=replicate_id)
    if kind == "ablation":
        return _generate_ablation_dataset(setting, seed=seed, replicate_id=replicate_id)
    raise ValueError(f"Unknown mechanism experiment kind: {setting.experiment_kind!r}")


def save_mechanism_dataset(dataset: MechanismDataset, out_dir: Path | str) -> Dict[str, str]:
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
