from __future__ import annotations

import numpy as np

from .utils import canonical_groups, sample_correlated_design


def generate_grouped_logistic_dataset(
    n: int,
    group_sizes: list[int],
    rho_within: float,
    rho_between: float,
    beta0: np.ndarray,
    seed: int,
    min_separator_auc: float = 0.9,
    max_attempts: int = 20,
) -> dict:
    from sklearn.metrics import roc_auc_score

    beta = np.asarray(beta0, dtype=float).reshape(-1)
    groups = canonical_groups(group_sizes)

    for k in range(int(max_attempts)):
        local_seed = int(seed) + 101 * k
        X, cov = sample_correlated_design(
            n=n,
            group_sizes=group_sizes,
            rho_within=rho_within,
            rho_between=rho_between,
            seed=local_seed,
        )
        logits = np.clip(X @ beta, -25.0, 25.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        rng = np.random.default_rng(local_seed + 7)
        y = rng.binomial(1, prob, size=int(n)).astype(float)

        try:
            auc = float(roc_auc_score(y, X[:, groups[0][0]]))
        except Exception:
            auc = 0.5
        if auc >= float(min_separator_auc):
            return {
                "X": X,
                "y": y,
                "beta0": beta,
                "groups": groups,
                "separator_auc": auc,
                "cov_x": cov,
                "attempts": k + 1,
            }

    return {
        "X": X,
        "y": y,
        "beta0": beta,
        "groups": groups,
        "separator_auc": auc,
        "cov_x": cov,
        "attempts": int(max_attempts),
    }
