from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

from .utils import canonical_groups, sample_correlated_design


def build_linear_beta(setting_name: str, group_sizes: Sequence[int]) -> np.ndarray:
    groups = canonical_groups(group_sizes)
    p = int(sum(group_sizes))
    beta = np.zeros(p, dtype=float)

    name = str(setting_name).upper()
    if name == "L1":
        for gid in [0, 1]:
            beta[groups[gid][0]] = 1.0
    elif name == "L2":
        for gid in [0, 1]:
            idx = np.asarray(groups[gid], dtype=int)
            beta[idx] = 1.0 / math.sqrt(len(idx))
    elif name == "L3":
        beta[groups[0][0]] = 1.0
    elif name == "L4":
        for gid in [3, 4]:
            idx = np.asarray(groups[gid], dtype=int)
            beta[idx] = 1.0 / math.sqrt(len(idx))
    else:
        raise ValueError(f"unknown linear setting: {setting_name}")
    return beta


def sigma2_for_target_snr(beta: np.ndarray, cov_x: np.ndarray, target_snr: float) -> float:
    signal_var = float(np.asarray(beta, dtype=float).T @ np.asarray(cov_x, dtype=float) @ np.asarray(beta, dtype=float))
    snr = float(target_snr)
    if signal_var <= 1e-12:
        return 1.0
    return float(signal_var * (1.0 - snr) / max(snr, 1e-8))


def generate_grouped_linear_dataset(
    n: int,
    group_sizes: Sequence[int],
    rho_within: float,
    rho_between: float,
    beta_shape: np.ndarray,
    seed: int,
    target_snr: float = 0.7,
) -> Dict[str, np.ndarray]:
    X, cov_x = sample_correlated_design(
        n=n,
        group_sizes=group_sizes,
        rho_within=rho_within,
        rho_between=rho_between,
        seed=seed,
    )
    beta = np.asarray(beta_shape, dtype=float).reshape(-1)
    sigma2 = sigma2_for_target_snr(beta=beta, cov_x=cov_x, target_snr=target_snr)

    rng = np.random.default_rng(int(seed) + 17)
    y = X @ beta + rng.normal(loc=0.0, scale=math.sqrt(sigma2), size=int(n))

    return {
        "X": X,
        "y": y,
        "beta0": beta,
        "sigma2": float(sigma2),
        "cov_x": cov_x,
        "groups": canonical_groups(group_sizes),
    }


def generate_heterogeneity_dataset(
    n: int,
    group_sizes: Sequence[int],
    rho_within: float,
    rho_between: float,
    sigma2: float,
    mu: Sequence[float],
    seed: int,
) -> Dict[str, np.ndarray]:
    X, cov_x = sample_correlated_design(
        n=n,
        group_sizes=group_sizes,
        rho_within=rho_within,
        rho_between=rho_between,
        seed=seed,
    )
    groups = canonical_groups(group_sizes)
    beta = np.zeros(sum(group_sizes), dtype=float)

    for gid, mu_g in enumerate(mu):
        idx = np.asarray(groups[gid], dtype=int)
        if float(mu_g) <= 0.0:
            continue
        beta[idx] = math.sqrt((2.0 * float(sigma2) * float(mu_g)) / max(len(idx), 1))

    rng = np.random.default_rng(int(seed) + 23)
    y = X @ beta + rng.normal(loc=0.0, scale=math.sqrt(float(sigma2)), size=int(n))

    return {
        "X": X,
        "y": y,
        "beta0": beta,
        "sigma2": float(sigma2),
        "cov_x": cov_x,
        "groups": groups,
        "mu": np.asarray(mu, dtype=float),
    }
