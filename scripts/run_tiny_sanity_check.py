from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from data.generators import generate_synthetic, synthetic_config_from_dict
from grrhs.models.baselines import HorseshoeRegression, Lasso, Ridge, SparseGroupLasso
from grrhs.models.grrhs_gibbs import GRRHS_Gibbs


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _cv_select_and_fit(
    X: np.ndarray,
    y: np.ndarray,
    make_model: Callable[..., Any],
    param_grid: Iterable[Dict[str, Any]],
    *,
    cv: int,
    seed: int,
) -> Tuple[Any, Dict[str, Any], float]:
    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    best_score = np.inf
    best_params: Dict[str, Any] = {}

    for params in param_grid:
        fold_mse: List[float] = []
        for train_idx, val_idx in splitter.split(X):
            model = make_model(**params)
            model.fit(X[train_idx], y[train_idx])
            pred = np.asarray(model.predict(X[val_idx]), dtype=float)
            mse = float(np.mean((y[val_idx] - pred) ** 2))
            fold_mse.append(mse)
        score = float(np.mean(fold_mse))
        if score < best_score:
            best_score = score
            best_params = dict(params)

    best_model = make_model(**best_params)
    best_model.fit(X, y)
    return best_model, best_params, best_score


def _tiny_dataset_spec(
    n: int,
    p: int,
    sigma_noise: float,
    seed: int,
    profile: str = "fair_mixed_uneven",
) -> Dict[str, Any]:
    # High-dimensional sanity check (p >> n) for variable/group selection and coefficient recovery.
    # Keep n small, increase p, enforce grouped correlation, and use a blueprint with explicit tags
    # including many exact zeros ("null") so selection metrics are well-defined.
    p = int(p)
    if p < 120:
        raise ValueError(f"Expected p >= 120 for high-dimensional sanity profiles, got p={p}.")
    profile = str(profile).lower().strip()
    rng = np.random.default_rng(int(seed) + 2026)

    def _bucket_sizes(total: int, low: int, high: int) -> List[int]:
        sizes_local: List[int] = []
        remaining = int(total)
        while remaining > 0:
            if remaining <= high:
                if remaining < low and sizes_local:
                    sizes_local[-1] += remaining
                else:
                    sizes_local.append(remaining)
                break
            draw = int(rng.integers(low, high + 1))
            if 0 < (remaining - draw) < low:
                draw = remaining - low
            sizes_local.append(draw)
            remaining -= draw
        return sizes_local

    if profile == "fair_within_group_mixed_signal":
        # Explicitly uneven small/medium/large groups.
        small_total = int(round(0.22 * p))
        medium_total = int(round(0.33 * p))
        large_total = int(p - small_total - medium_total)
        group_records: List[Tuple[int, str]] = []
        group_records.extend((size, "small") for size in _bucket_sizes(small_total, 5, 10))
        group_records.extend((size, "medium") for size in _bucket_sizes(medium_total, 20, 30))
        group_records.extend((size, "large") for size in _bucket_sizes(large_total, 40, 80))
        rng.shuffle(group_records)
        sizes = np.array([int(size) for size, _ in group_records], dtype=int)
        group_scale = [tag for _, tag in group_records]
    else:
        sizes = np.array([2, 2, 2, 3, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 53], dtype=int)
        if int(np.sum(sizes)) != p:
            scale = float(p) / float(np.sum(sizes))
            sizes = np.maximum(2, np.round(sizes * scale).astype(int))
            diff = int(p - int(np.sum(sizes)))
            cursor = 0
            while diff != 0:
                idx = cursor % int(sizes.size)
                if diff > 0:
                    sizes[idx] += 1
                    diff -= 1
                elif sizes[idx] > 2:
                    sizes[idx] -= 1
                    diff += 1
                cursor += 1
        group_scale = ["small" if s <= 10 else "medium" if s <= 30 else "large" for s in sizes.tolist()]
    G = int(sizes.size)

    corr_extra: Dict[str, Any] = {}
    if profile == "fair_sparse_strong":
        signal_groups = [8, 12, 14]
        rho_in_range = [0.70, 0.92]
        rho_out_range = [0.04, 0.10]
    elif profile == "fair_dense_weak_grouped":
        signal_groups = [11, 12, 13, 14, 15]
        rho_in_range = [0.86, 0.98]
        rho_out_range = [0.02, 0.08]
    elif profile == "fair_hetero_mixed_corr":
        signal_groups = [10, 12, 14, 15]
        rho_in_range = [0.20, 0.95]  # global fallback only
        rho_out_range = [0.03, 0.09]
        # Group-wise heterogeneity: some strongly correlated, some weak, some mixed mid.
        corr_extra = {
            "rho_in_group_ranges": {
                0: [0.15, 0.35],
                1: [0.20, 0.40],
                2: [0.25, 0.45],
                3: [0.30, 0.50],
                4: [0.35, 0.55],
                5: [0.40, 0.60],
                6: [0.45, 0.65],
                7: [0.50, 0.70],
                8: [0.55, 0.75],
                9: [0.20, 0.45],
                10: [0.65, 0.90],
                11: [0.25, 0.60],
                12: [0.70, 0.94],
                13: [0.30, 0.70],
                14: [0.75, 0.96],
                15: [0.55, 0.95],
            },
            "within_group_mix_fraction": 0.55,
            "within_group_strong_range": [0.70, 0.98],
            "within_group_weak_range": [0.12, 0.45],
        }
    elif profile == "fair_within_group_mixed_signal":
        candidate_active = [idx for idx, tag in enumerate(group_scale) if tag in {"medium", "large"}]
        active_count = int(min(max(3, round(0.2 * G)), max(3, len(candidate_active) // 2)))
        signal_groups = sorted(int(v) for v in rng.choice(np.asarray(candidate_active, dtype=int), size=active_count, replace=False).tolist())
        rho_in_range = [0.20, 0.95]
        rho_out_range = [0.02, 0.08]

        group_ranges: Dict[str, List[float]] = {}
        for gid, tag in enumerate(group_scale):
            if tag == "small":
                group_ranges[str(gid)] = [0.18, 0.45]
            elif tag == "medium":
                group_ranges[str(gid)] = [0.40, 0.72]
            else:
                group_ranges[str(gid)] = [0.62, 0.94]
        corr_extra = {
            "rho_in_group_ranges": group_ranges,
            "within_group_mix_fraction": 0.55,
            "within_group_strong_range": [0.70, 0.98],
            "within_group_weak_range": [0.12, 0.50],
        }
    else:
        # default fair mixed uneven profile
        signal_groups = [11, 12, 13, 14, 15]
        rho_in_range = [0.82, 0.97]
        rho_out_range = [0.03, 0.10]

    blueprint: List[Dict[str, Any]] = []
    for gid in range(G):
        if gid in signal_groups:
            group_size = int(sizes[gid])
            if profile == "fair_sparse_strong":
                components = [
                    {"name": "strong", "count": 2, "distribution": "uniform", "low": 2.0, "high": 3.4, "sign": "mixed", "tag": "strong"},
                    {"name": "weak", "count": 1, "distribution": "uniform", "low": 0.18, "high": 0.35, "sign": "mixed", "tag": "weak"},
                    {"name": "near_zero", "count": 1, "distribution": "uniform", "low": 0.01, "high": 0.03, "sign": "mixed", "tag": "near_zero"},
                    {"name": "null_rest", "fraction": 1.0, "distribution": "constant", "value": 0.0, "sign": "mixed", "tag": "null"},
                ]
            elif profile == "fair_within_group_mixed_signal":
                strong_count = 1 if group_size < 30 else 2
                moderate_count = max(2, int(round(group_size * 0.12)))
                weak_count = max(3, int(round(group_size * 0.28)))
                near_zero_count = max(2, int(round(group_size * 0.15)))
                components = [
                    {"name": "strong", "count": strong_count, "distribution": "uniform", "low": 2.0, "high": 3.3, "sign": "mixed", "tag": "strong"},
                    {"name": "moderate", "count": moderate_count, "distribution": "uniform", "low": 0.60, "high": 1.20, "sign": "mixed", "tag": "medium"},
                    {"name": "weak", "count": weak_count, "distribution": "uniform", "low": 0.10, "high": 0.35, "sign": "mixed", "tag": "weak"},
                    {"name": "near_zero", "count": near_zero_count, "distribution": "uniform", "low": 0.01, "high": 0.05, "sign": "mixed", "tag": "near_zero"},
                    {"name": "null_rest", "fraction": 1.0, "distribution": "constant", "value": 0.0, "sign": "mixed", "tag": "null"},
                ]
            else:
                if gid == 15:
                    components = [
                        {"name": "strong", "count": 1, "distribution": "uniform", "low": 2.4, "high": 3.4, "sign": "mixed", "tag": "strong"},
                        {"name": "weak", "count": 24, "distribution": "uniform", "low": 0.10, "high": 0.32, "sign": "mixed", "tag": "weak"},
                        {"name": "near_zero", "count": 8, "distribution": "uniform", "low": 0.01, "high": 0.04, "sign": "mixed", "tag": "near_zero"},
                        {"name": "null_rest", "fraction": 1.0, "distribution": "constant", "value": 0.0, "sign": "mixed", "tag": "null"},
                    ]
                elif gid == 14:
                    components = [
                        {"name": "strong", "count": 1, "distribution": "uniform", "low": 2.0, "high": 3.0, "sign": "mixed", "tag": "strong"},
                        {"name": "weak", "count": 13, "distribution": "uniform", "low": 0.10, "high": 0.30, "sign": "mixed", "tag": "weak"},
                        {"name": "near_zero", "count": 5, "distribution": "uniform", "low": 0.01, "high": 0.04, "sign": "mixed", "tag": "near_zero"},
                        {"name": "null_rest", "fraction": 1.0, "distribution": "constant", "value": 0.0, "sign": "mixed", "tag": "null"},
                    ]
                elif gid == 13:
                    components = [
                        {"name": "strong", "count": 1, "distribution": "uniform", "low": 1.8, "high": 2.8, "sign": "mixed", "tag": "strong"},
                        {"name": "weak", "count": 10, "distribution": "uniform", "low": 0.10, "high": 0.28, "sign": "mixed", "tag": "weak"},
                        {"name": "near_zero", "count": 4, "distribution": "uniform", "low": 0.01, "high": 0.03, "sign": "mixed", "tag": "near_zero"},
                        {"name": "null_rest", "fraction": 1.0, "distribution": "constant", "value": 0.0, "sign": "mixed", "tag": "null"},
                    ]
                elif gid == 12:
                    components = [
                        {"name": "strong", "count": 1, "distribution": "uniform", "low": 1.8, "high": 2.8, "sign": "mixed", "tag": "strong"},
                        {"name": "weak", "count": 8, "distribution": "uniform", "low": 0.10, "high": 0.26, "sign": "mixed", "tag": "weak"},
                        {"name": "near_zero", "count": 3, "distribution": "uniform", "low": 0.01, "high": 0.03, "sign": "mixed", "tag": "near_zero"},
                        {"name": "null_rest", "fraction": 1.0, "distribution": "constant", "value": 0.0, "sign": "mixed", "tag": "null"},
                    ]
                else:
                    components = [
                        {"name": "strong", "count": 1, "distribution": "uniform", "low": 1.8, "high": 2.8, "sign": "mixed", "tag": "strong"},
                        {"name": "weak", "count": 6, "distribution": "uniform", "low": 0.10, "high": 0.26, "sign": "mixed", "tag": "weak"},
                        {"name": "near_zero", "count": 2, "distribution": "uniform", "low": 0.01, "high": 0.03, "sign": "mixed", "tag": "near_zero"},
                        {"name": "null_rest", "fraction": 1.0, "distribution": "constant", "value": 0.0, "sign": "mixed", "tag": "null"},
                    ]
            blueprint.append({"label": f"signal_group_{gid}", "groups": [gid], "components": components})
        else:
            if profile == "fair_sparse_strong":
                components = [
                    {"name": "noise_like", "count": 1, "distribution": "constant", "value": 0.0, "sign": "mixed", "tag": "noise_like"},
                    {"name": "null_rest", "fraction": 1.0, "distribution": "constant", "value": 0.0, "sign": "mixed", "tag": "null"},
                ]
            elif profile == "fair_within_group_mixed_signal":
                group_size = int(sizes[gid])
                near_zero_count = max(1, int(round(group_size * 0.06)))
                components = [
                    {"name": "noise_like", "count": 1, "distribution": "constant", "value": 0.0, "sign": "mixed", "tag": "noise_like"},
                    {"name": "near_zero", "count": near_zero_count, "distribution": "uniform", "low": 0.01, "high": 0.04, "sign": "mixed", "tag": "near_zero"},
                    {"name": "null_rest", "fraction": 1.0, "distribution": "constant", "value": 0.0, "sign": "mixed", "tag": "null"},
                ]
            else:
                components = [
                    {"name": "noise_like", "count": 1, "distribution": "constant", "value": 0.0, "sign": "mixed", "tag": "noise_like"},
                    {"name": "near_zero", "count": 1, "distribution": "uniform", "low": 0.01, "high": 0.03, "sign": "mixed", "tag": "near_zero"},
                    {"name": "null_rest", "fraction": 1.0, "distribution": "constant", "value": 0.0, "sign": "mixed", "tag": "null"},
                ]
            blueprint.append({"label": f"noise_group_{gid}", "groups": [gid], "components": components})

    return {
        "n": int(n),
        "p": int(p),
        "G": int(G),
        "group_sizes": [int(s) for s in sizes.tolist()],
        "seed": int(seed),
        "noise_sigma": float(sigma_noise),
        "correlation": {
            "type": "group_block_random",
            "rho_in_range": rho_in_range,
            "rho_out_range": rho_out_range,
            **corr_extra,
        },
        "signal": {
            # Keep exact zeros exact so selection metrics are meaningful.
            "background_noise": {"scale": 0.0},
            "blueprint": blueprint,
        },
        "profile": profile,
    }


def _toy_mixed_signal_dataset(
    n: int,
    sigma_noise: float,
    seed: int,
    *,
    A: float = 3.0,
    plus_corr: bool = False,
) -> Dict[str, Any]:
    # Mechanism-isolating toy example:
    # n=60, p=60, G=6 (10 vars/group), equal group sizes.
    # Default is near-orthogonal design to isolate global/local/group shrinkage behavior.
    rng = np.random.default_rng(int(seed))
    p = 60
    gsize = 10
    groups = [list(range(g * gsize, (g + 1) * gsize)) for g in range(6)]

    if not plus_corr:
        # Near-orthogonal design (X^T X ~= n I) to isolate shrinkage mechanisms.
        if int(n) < p:
            raise ValueError(f"toy_orthogonal_mixed_signal requires n >= p (got n={n}, p={p}).")
        z = rng.normal(0.0, 1.0, size=(int(n), p))
        q, _ = np.linalg.qr(z, mode="reduced")
        X = np.sqrt(float(n)) * q[:, :p]
    else:
        # Toy-plus: mild within-group correlation, still simple and controlled.
        sigma = np.zeros((p, p), dtype=float)
        np.fill_diagonal(sigma, 1.0)

        def _set_equicorr(block: List[int], rho: float) -> None:
            for i in block:
                for j in block:
                    if i != j:
                        sigma[i, j] = float(rho)

        _set_equicorr(groups[0], 0.20)
        _set_equicorr(groups[1], 0.20)
        _set_equicorr(groups[2], 0.40)
        _set_equicorr(groups[4], 0.20)
        _set_equicorr(groups[5], 0.20)

        g4 = groups[3]
        g4_a = g4[:5]
        g4_b = g4[5:]
        _set_equicorr(g4_a, 0.60)
        _set_equicorr(g4_b, 0.20)
        for i in g4_a:
            for j in g4_b:
                sigma[i, j] = 0.10
                sigma[j, i] = 0.10

        sigma = (sigma + sigma.T) / 2.0
        eig_min = float(np.min(np.linalg.eigvalsh(sigma)))
        if eig_min <= 1e-8:
            sigma += np.eye(p) * (1e-6 - eig_min)
        X = rng.multivariate_normal(mean=np.zeros(p, dtype=float), cov=sigma, size=int(n))

    beta = np.zeros(p, dtype=float)
    # Group 2: isolated strong signal.
    beta[10] = float(A)
    # Group 3: dense-weak signal.
    beta[20:30] = np.array([0.30, 0.28, 0.25, 0.22, 0.20, 0.18, 0.0, 0.0, 0.0, 0.0], dtype=float)
    # Group 4: mixed-signal group.
    beta[30:40] = np.array([0.8 * A, 0.35 * A, 0.22, 0.18, 0.15, 0.10, 0.0, 0.0, 0.0, 0.0], dtype=float)
    # Group 5: near-zero distractors.
    beta[40:50] = np.array([0.06, 0.05, 0.04, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    # Group 6: sparse medium signal.
    beta[50:60] = np.array([0.45, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    y = X @ beta + rng.normal(0.0, float(sigma_noise), size=int(n))

    tags: Dict[str, List[int]] = {
        "strong": [10, 30],
        "medium": [31, 50, 51],
        "weak": [20, 21, 22, 23, 24, 25, 32, 33, 34, 35],
        "near_zero": [40, 41, 42, 43],
        "null": [j for j in range(p) if j not in {10, 30, 31, 50, 51, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 40, 41, 42, 43}],
    }
    active_idx = sorted(int(j) for j in np.where(np.abs(beta) > 0.0)[0].tolist())
    signal_meta = {
        "assignments": [
            {"entry": "group2", "group": 1, "component": "strong", "indices": [10]},
            {"entry": "group3", "group": 2, "component": "weak_dense", "indices": [20, 21, 22, 23, 24, 25]},
            {"entry": "group4", "group": 3, "component": "mixed_signal", "indices": [30, 31, 32, 33, 34, 35]},
            {"entry": "group5", "group": 4, "component": "near_zero", "indices": [40, 41, 42, 43]},
            {"entry": "group6", "group": 5, "component": "sparse_medium", "indices": [50, 51]},
        ],
        "active_idx": active_idx,
        "tags": tags,
        "mechanism_sets": {
            "strong_idx": [10, 30],
            "weak_idx": [20, 21, 22, 23, 24, 25, 32, 33, 34, 35],
            "near_zero_idx": [40, 41, 42, 43],
            "null_idx": [int(j) for j in np.where(np.abs(beta) <= 1e-12)[0].tolist()],
        },
    }
    toy_profile_name = "toy_plus_mixed_signal_6g" if plus_corr else "toy_orthogonal_mixed_signal"
    data_cfg = {
        "profile": toy_profile_name,
        "n": int(n),
        "p": int(p),
        "G": 6,
        "group_sizes": [10, 10, 10, 10, 10, 10],
        "seed": int(seed),
        "noise_sigma": float(sigma_noise),
        "correlation": {
            "type": "near_orthogonal" if not plus_corr else "toy_plus_light_correlation",
            "group_rho": {
                "group1": 0.20 if plus_corr else 0.0,
                "group2": 0.20 if plus_corr else 0.0,
                "group3": 0.40 if plus_corr else 0.0,
                "group4_blockA": 0.60 if plus_corr else 0.0,
                "group4_blockB": 0.20,
                "group4_cross": 0.10 if plus_corr else 0.0,
                "group5": 0.20 if plus_corr else 0.0,
                "group6": 0.20 if plus_corr else 0.0,
            },
        },
        "signal": {
            "description": "orthogonal grouped mixed-signal toy",
            "A": float(A),
            "blueprint": signal_meta["assignments"],
        },
    }
    return {
        "X": X.astype(np.float64, copy=False),
        "y": y.astype(np.float64, copy=False),
        "beta": beta.astype(np.float64, copy=False),
        "groups": groups,
        "signal_meta": signal_meta,
        "data_cfg": data_cfg,
    }


def run_tiny_sanity(
    *,
    seed: int = 196,
    n: int = 20,
    p: int = 200,
    sigma_noise: float = 0.8,
    outdir: Path = Path("outputs/tiny_sanity"),
    hs_scale_global: float = 0.3,
    hs_num_warmup: int = 300,
    hs_num_samples: int = 300,
    grrhs_c: float = 5.0,
    grrhs_tau0: float | None = None,
    grrhs_tau0_multiplier: float = 1.0,
    grrhs_eta: float = 1.0,
    grrhs_s0: float = 1.0,
    grrhs_iters: int = 1200,
    grrhs_burnin: int = 600,
    grrhs_tau_slice_w: float = 0.25,
    grrhs_tau_slice_m: int = 180,
    profile: str = "fair_mixed_uneven",
    toy_a: float = 3.0,
    toy_plus_corr: bool = False,
) -> Dict[str, Any]:
    if profile in {"toy_mixed_signal_6g", "toy_orthogonal_mixed_signal"}:
        toy_payload = _toy_mixed_signal_dataset(
            n=n,
            sigma_noise=sigma_noise,
            seed=seed,
            A=float(toy_a),
            plus_corr=bool(toy_plus_corr),
        )
        data_cfg = toy_payload["data_cfg"]
        X_raw = np.asarray(toy_payload["X"], dtype=np.float64)
        y_raw = np.asarray(toy_payload["y"], dtype=np.float64)
        beta_true = np.asarray(toy_payload["beta"], dtype=np.float64)
        groups = [list(map(int, g)) for g in toy_payload["groups"]]
        signal_blueprint_meta = dict(toy_payload["signal_meta"])
    else:
        data_cfg = _tiny_dataset_spec(n=n, p=p, sigma_noise=sigma_noise, seed=seed, profile=profile)
        synth_cfg = synthetic_config_from_dict(data_cfg, seed=seed, name="tiny_sanity_5group")
        dataset = generate_synthetic(synth_cfg)
        X_raw = np.asarray(dataset.X, dtype=np.float64)
        y_raw = np.asarray(dataset.y, dtype=np.float64)
        beta_true = np.asarray(dataset.beta, dtype=np.float64)
        groups = [list(map(int, g)) for g in dataset.groups]
        signal_blueprint_meta = (
            dataset.info.get("signal_blueprint", {}) if isinstance(dataset.info, dict) else {}
        )
    p = int(X_raw.shape[1])

    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(X_raw).astype(np.float64)
    y = (y_raw - float(np.mean(y_raw))).astype(np.float64)

    beta_true_std = beta_true * scaler.scale_

    # Pearson correlation of features (invariant to per-feature affine scaling).
    corr = np.corrcoef(X, rowvar=False)
    fig_corr, ax_corr = plt.subplots(figsize=(8.8, 7.6))
    im = ax_corr.imshow(corr, vmin=-1.0, vmax=1.0, cmap="RdBu_r", interpolation="nearest")
    cbar = fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", rotation=90)
    step = 10 if p >= 80 else 1
    tick_pos = np.arange(0, p, step, dtype=int)
    tick_labels = [f"X{i + 1}" for i in tick_pos.tolist()]
    ax_corr.set_xticks(tick_pos)
    ax_corr.set_yticks(tick_pos)
    ax_corr.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax_corr.set_yticklabels(tick_labels, fontsize=7)
    ax_corr.set_title("Tiny Sanity: Feature Pearson Correlation (X)")

    cursor = 0
    for size in [len(g) for g in groups[:-1]]:
        cursor += size
        ax_corr.axhline(cursor - 0.5, color="#111827", linewidth=1.1)
        ax_corr.axvline(cursor - 0.5, color="#111827", linewidth=1.1)

    fig_corr.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    corr_fig_path = outdir / "tiny_sanity_X_pearson_corr.png"
    fig_corr.savefig(corr_fig_path, dpi=200)
    plt.close(fig_corr)
    corr_csv_path = outdir / "tiny_sanity_X_pearson_corr.csv"
    np.savetxt(corr_csv_path, corr, delimiter=",", fmt="%.6f")

    lasso_grid = [{"alpha": float(a)} for a in np.logspace(-2.3, 0.5, 20)]
    ridge_grid = [{"alpha": float(a)} for a in np.logspace(-3.0, 2.0, 24)]
    sgl_grid = [
        {"alpha": float(a), "l1_ratio": float(r)}
        for a in np.logspace(-2.5, 0.6, 16)
        for r in (0.3, 0.5, 0.7)
    ]

    lasso, lasso_params, lasso_cv = _cv_select_and_fit(
        X,
        y,
        lambda **kw: Lasso(fit_intercept=False, max_iter=120, max_epochs=60_000, tol=1e-6, warm_start=True, **kw),
        lasso_grid,
        cv=5,
        seed=seed,
    )
    ridge, ridge_params, ridge_cv = _cv_select_and_fit(
        X,
        y,
        lambda **kw: Ridge(fit_intercept=False, **kw),
        ridge_grid,
        cv=5,
        seed=seed,
    )
    sgl, sgl_params, sgl_cv = _cv_select_and_fit(
        X,
        y,
        lambda **kw: SparseGroupLasso(
            groups=groups,
            fit_intercept=False,
            max_iter=150,
            max_epochs=60_000,
            tol=1e-6,
            warm_start=True,
            **kw,
        ),
        sgl_grid,
        cv=5,
        seed=seed,
    )

    hs = HorseshoeRegression(
        scale_global=float(hs_scale_global),
        num_warmup=int(hs_num_warmup),
        num_samples=int(hs_num_samples),
        num_chains=1,
        thinning=1,
        target_accept_prob=0.9,
        progress_bar=False,
        seed=seed + 11,
    ).fit(X, y)

    # Calibrate tau0 from an expected number of nonzero coefficients (Piironen–Vehtari style heuristic).
    # Use blueprint tags so this remains valid even when "null" coefficients are exactly zero.
    tags_pre = signal_blueprint_meta.get("tags", {}) if isinstance(signal_blueprint_meta, dict) else {}
    p0 = 0
    if isinstance(tags_pre, dict):
        for key in ("strong", "weak", "medium"):
            idx = tags_pre.get(key, [])
            if isinstance(idx, (list, tuple)):
                p0 += len(idx)
    p0 = int(max(1, min(p - 1, p0)))
    tau0_auto = (p0 / (p - p0)) * float(sigma_noise) / float(np.sqrt(max(n, 1)))
    tau0_used = float(
        tau0_auto * max(float(grrhs_tau0_multiplier), 1e-8) if grrhs_tau0 is None else grrhs_tau0
    )

    grrhs = GRRHS_Gibbs(
        c=float(grrhs_c),
        tau0=float(tau0_used),
        eta=float(grrhs_eta),
        s0=float(grrhs_s0),
        use_groups=True,
        iters=int(grrhs_iters),
        burnin=int(grrhs_burnin),
        thin=1,
        seed=seed + 23,
        num_chains=1,
        tau_slice_w=float(grrhs_tau_slice_w),
        tau_slice_m=int(grrhs_tau_slice_m),
    ).fit(X, y, groups=groups)

    estimates = {
        "True": beta_true_std,
        "Lasso": np.asarray(lasso.coef_, dtype=float),
        "Ridge": np.asarray(ridge.coef_, dtype=float),
        "Sparse Group Lasso": np.asarray(sgl.coef_, dtype=float),
        "Horseshoe": np.asarray(hs.coef_, dtype=float),
        "GR-RHS": np.asarray(grrhs.coef_mean_, dtype=float),
    }

    order = ["True", "Lasso", "Ridge", "Sparse Group Lasso", "Horseshoe", "GR-RHS"]
    model_order = [name for name in order if name != "True"]
    colors = {
        "True": "#8e1b1b",
        "Lasso": "#2b6cb0",
        "Ridge": "#f2a900",
        "Sparse Group Lasso": "#15803d",
        "Horseshoe": "#7c3aed",
        "GR-RHS": "#111827",
    }

    signal_meta = signal_blueprint_meta if isinstance(signal_blueprint_meta, dict) else {}
    tag_map_raw = signal_meta.get("tags", {}) if isinstance(signal_meta, dict) else {}
    tag_to_idx: Dict[str, List[int]] = {}
    if isinstance(tag_map_raw, dict):
        for tag, indices in tag_map_raw.items():
            if not isinstance(indices, (list, tuple)):
                continue
            tag_to_idx[str(tag)] = sorted({int(j) for j in indices if 0 <= int(j) < p})

    signal_tag_names = {"strong", "weak", "medium"}
    noise_tag_names = {"near_zero", "noise_like", "null", "noise"}
    signal_idx = sorted({j for tag, idx in tag_to_idx.items() if tag in signal_tag_names for j in idx})
    noise_idx = sorted({j for tag, idx in tag_to_idx.items() if tag in noise_tag_names for j in idx})

    tagged_idx = set(signal_idx).union(noise_idx)
    # Any feature not explicitly tagged as signal/noise is treated as noise by default.
    for j in range(p):
        if j not in tagged_idx:
            noise_idx.append(j)
    noise_idx = sorted(set(noise_idx))

    signal_mask = np.zeros(p, dtype=bool)
    noise_mask = np.zeros(p, dtype=bool)
    signal_mask[np.asarray(signal_idx, dtype=int)] = True
    noise_mask[np.asarray(noise_idx, dtype=int)] = True

    # Safety fallback: avoid empty subsets if metadata is unavailable.
    if not np.any(signal_mask) or not np.any(noise_mask):
        eps = 1e-10
        signal_mask = np.abs(beta_true_std) > eps
        noise_mask = ~signal_mask

    # Feature- and group-selection metrics using |beta_hat| as the ranking signal.
    # Positives are blueprint-tagged strong/weak/medium coefficients.
    feature_metrics: List[Dict[str, object]] = []
    group_metrics: List[Dict[str, object]] = []
    pos_feature = np.asarray(signal_mask, dtype=bool)
    pos_count = int(np.sum(pos_feature))
    k = int(max(1, pos_count))

    # True positive groups: any group containing at least one positive feature.
    pos_group = np.zeros(len(groups), dtype=bool)
    for gid, idx in enumerate(groups):
        if np.any(pos_feature[np.asarray(idx, dtype=int)]):
            pos_group[gid] = True
    pos_group_count = int(np.sum(pos_group))

    def _precision_recall_at_k(scores: np.ndarray, positives: np.ndarray, k: int) -> tuple[float, float, float]:
        k = int(max(1, min(scores.size, k)))
        top = np.argpartition(-scores, kth=k - 1)[:k]
        tp = int(np.sum(positives[top]))
        prec = float(tp / k)
        rec = float(tp / max(int(np.sum(positives)), 1))
        fdr = float(1.0 - prec)
        return prec, rec, fdr

    def _group_scores(beta_hat: np.ndarray, mode: str) -> np.ndarray:
        if mode == "mean_abs":
            return np.array([float(np.mean(np.abs(beta_hat[idx]))) for idx in groups], dtype=float)
        if mode == "sum_abs":
            return np.array([float(np.sum(np.abs(beta_hat[idx]))) for idx in groups], dtype=float)
        if mode == "size_adjusted_sum":
            return np.array([float(np.sum(np.abs(beta_hat[idx])) / np.sqrt(max(len(idx), 1))) for idx in groups], dtype=float)
        if mode == "l2_norm":
            return np.array([float(np.linalg.norm(beta_hat[idx], ord=2)) for idx in groups], dtype=float)
        raise ValueError(f"Unsupported group scoring mode: {mode}")

    group_score_modes = ["mean_abs", "sum_abs", "size_adjusted_sum", "l2_norm"]

    for name in model_order:
        scores = np.abs(np.asarray(estimates[name], dtype=float))
        auprc = float(average_precision_score(pos_feature.astype(int), scores))
        prec_k, rec_k, fdr_k = _precision_recall_at_k(scores, pos_feature, k)
        feature_metrics.append(
            {
                "method": name,
                "pos_feature_count": int(pos_count),
                "AUPRC": auprc,
                f"precision@{k}": prec_k,
                f"recall@{k}": rec_k,
                f"fdr@{k}": fdr_k,
            }
        )

        beta_hat = np.asarray(estimates[name], dtype=float)
        m = int(max(1, pos_group_count))
        mode_payload: Dict[str, object] = {}
        for mode in group_score_modes:
            mode_scores = _group_scores(beta_hat, mode)
            mode_auprc = float(average_precision_score(pos_group.astype(int), mode_scores))
            topm = np.argpartition(-mode_scores, kth=m - 1)[:m]
            hit = int(np.sum(pos_group[topm]))
            mode_payload[mode] = {
                "AUPRC_group": mode_auprc,
                f"top{m}_group_hit": hit,
            }
        group_metrics.append(
            {
                "method": name,
                "pos_group_count": int(pos_group_count),
                "scores": mode_payload,
            }
        )

    # PR curve plot (feature-level).
    fig_pr, ax_pr = plt.subplots(figsize=(7.8, 5.4))
    for name in model_order:
        scores = np.abs(np.asarray(estimates[name], dtype=float))
        precision, recall, _ = precision_recall_curve(pos_feature.astype(int), scores)
        ax_pr.plot(recall, precision, linewidth=2.0, color=colors[name], label=name)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Tiny Sanity: Feature Selection Precision-Recall")
    ax_pr.grid(alpha=0.22)
    ax_pr.legend(frameon=False, fontsize=9)
    fig_pr.tight_layout()
    pr_fig_path = outdir / "tiny_sanity_feature_pr_curve.png"
    fig_pr.savefig(pr_fig_path, dpi=180)
    plt.close(fig_pr)

    summary_rows = []
    for name, coef in estimates.items():
        if name == "True":
            continue
        beta_rmse = float(np.sqrt(np.mean((coef - beta_true_std) ** 2)))
        summary_rows.append({"method": name, "beta_rmse": beta_rmse})

    outdir.mkdir(parents=True, exist_ok=True)

    coef_path = outdir / "tiny_sanity_coefficients.json"
    coef_path.write_text(json.dumps({k: v.tolist() for k, v in estimates.items()}, indent=2), encoding="utf-8")

    summary_path = outdir / "tiny_sanity_summary.json"
    mechanism_sets_raw = signal_meta.get("mechanism_sets", {}) if isinstance(signal_meta, dict) else {}
    mechanism_sets: Dict[str, np.ndarray] = {}
    if isinstance(mechanism_sets_raw, dict):
        for key in ("strong_idx", "weak_idx", "null_idx", "near_zero_idx"):
            vals = mechanism_sets_raw.get(key, [])
            if isinstance(vals, (list, tuple)):
                mechanism_sets[key] = np.array([int(v) for v in vals if 0 <= int(v) < p], dtype=int)
    if "strong_idx" not in mechanism_sets:
        mechanism_sets["strong_idx"] = np.array([], dtype=int)
    if "weak_idx" not in mechanism_sets:
        mechanism_sets["weak_idx"] = np.array([], dtype=int)
    if "null_idx" not in mechanism_sets:
        mechanism_sets["null_idx"] = np.array(np.where(np.abs(beta_true_std) <= 1e-12)[0], dtype=int)
    if "near_zero_idx" not in mechanism_sets:
        mechanism_sets["near_zero_idx"] = np.array([], dtype=int)

    summary_payload = {
        "seed": int(seed),
        "profile": str(profile),
        "n": int(n),
        "p": int(p),
        "sigma_noise": float(sigma_noise),
        "groups": groups,
        "group_sizes": [len(g) for g in groups],
        "correlation": data_cfg["correlation"],
        "signal_blueprint": data_cfg["signal"]["blueprint"],
        "signal_metadata": signal_blueprint_meta,
        "beta_true_raw": beta_true.tolist(),
        "beta_true_standardized": beta_true_std.tolist(),
        "cv": {
            "lasso": {"best_params": lasso_params, "cv_mse": lasso_cv},
            "ridge": {"best_params": ridge_params, "cv_mse": ridge_cv},
            "sparse_group_lasso": {"best_params": sgl_params, "cv_mse": sgl_cv},
        },
        "bayes_hyperparams": {
            "horseshoe": {
                "scale_global": float(hs_scale_global),
                "num_warmup": int(hs_num_warmup),
                "num_samples": int(hs_num_samples),
            },
            "grrhs": {
                "c": float(grrhs_c),
                "tau0": float(tau0_used),
                "tau0_auto": float(tau0_auto),
                "tau0_auto_multiplier": float(grrhs_tau0_multiplier),
                "p0_used_for_tau0": int(p0),
                "eta": float(grrhs_eta),
                "s0": float(grrhs_s0),
                "iters": int(grrhs_iters),
                "burnin": int(grrhs_burnin),
                "tau_slice_w": float(grrhs_tau_slice_w),
                "tau_slice_m": int(grrhs_tau_slice_m),
            },
        },
        "beta_rmse": summary_rows,
        "coef_error_partition": {
            "mode": "blueprint_tags",
            "signal_tags": sorted(signal_tag_names),
            "noise_tags": sorted(noise_tag_names),
            "signal_indices": [int(j) for j in np.flatnonzero(signal_mask)],
            "noise_indices": [int(j) for j in np.flatnonzero(noise_mask)],
        },
        "feature_selection": feature_metrics,
        "group_selection": group_metrics,
    }
    summary_path.write_text(json.dumps(_to_serializable(summary_payload), indent=2), encoding="utf-8")

    x = np.arange(p)

    rmse_rows: List[Dict[str, float]] = []
    mechanism_rmse_rows: List[Dict[str, float]] = []
    for name in model_order:
        diff = np.asarray(estimates[name], dtype=float) - beta_true_std
        total_rmse = float(np.sqrt(np.mean(diff ** 2)))
        signal_rmse = float(np.sqrt(np.mean((diff[signal_mask]) ** 2))) if np.any(signal_mask) else float("nan")
        noise_rmse = float(np.sqrt(np.mean((diff[noise_mask]) ** 2))) if np.any(noise_mask) else float("nan")
        rmse_rows.append(
            {
                "method": name,
                "rmse_total": total_rmse,
                "rmse_signal": signal_rmse,
                "rmse_noise": noise_rmse,
            }
        )
        strong_idx = mechanism_sets["strong_idx"]
        weak_idx = mechanism_sets["weak_idx"]
        null_idx = mechanism_sets["null_idx"]
        mechanism_rmse_rows.append(
            {
                "method": name,
                "rmse_strong": float(np.sqrt(np.mean((diff[strong_idx]) ** 2))) if strong_idx.size > 0 else float("nan"),
                "rmse_weak": float(np.sqrt(np.mean((diff[weak_idx]) ** 2))) if weak_idx.size > 0 else float("nan"),
                "rmse_null": float(np.sqrt(np.mean((diff[null_idx]) ** 2))) if null_idx.size > 0 else float("nan"),
            }
        )
    rmse_rows.sort(key=lambda row: row["rmse_total"])
    mechanism_rmse_rows.sort(
        key=lambda row: (
            row["rmse_weak"] if np.isfinite(row["rmse_weak"]) else np.inf,
            row["rmse_strong"] if np.isfinite(row["rmse_strong"]) else np.inf,
        )
    )
    summary_payload["mechanism_rmse"] = mechanism_rmse_rows
    summary_path.write_text(json.dumps(_to_serializable(summary_payload), indent=2), encoding="utf-8")

    # Figure 1: RMSE main chart with signal/noise decomposition.
    labels = [row["method"] for row in rmse_rows]
    pos = np.arange(len(labels))
    width = 0.24
    fig_rmse, ax_rmse = plt.subplots(figsize=(10.5, 5.2))
    ax_rmse.bar(pos - width, [row["rmse_total"] for row in rmse_rows], width=width, color="#111827", label="Total RMSE", alpha=0.92)
    ax_rmse.bar(pos, [row["rmse_signal"] for row in rmse_rows], width=width, color="#b91c1c", label="Signal RMSE (blueprint tags)", alpha=0.88)
    ax_rmse.bar(pos + width, [row["rmse_noise"] for row in rmse_rows], width=width, color="#0369a1", label="Noise RMSE (blueprint tags)", alpha=0.88)
    ax_rmse.set_xticks(pos)
    ax_rmse.set_xticklabels(labels, rotation=20, ha="right")
    ax_rmse.set_ylabel("RMSE on coefficients")
    ax_rmse.set_title("Tiny Sanity: Coefficient Recovery Error by Model (Tag-defined Signal/Noise)")
    ax_rmse.grid(axis="y", alpha=0.22)
    ax_rmse.legend(frameon=False, fontsize=9)
    fig_rmse.tight_layout()
    rmse_fig_path = outdir / "tiny_sanity_coef_rmse_bar.png"
    fig_rmse.savefig(rmse_fig_path, dpi=180)
    plt.close(fig_rmse)

    # Backward-compatible alias to old figure name.
    fig_alias_path = outdir / "tiny_sanity_coef_recovery.png"
    fig_rmse_alias, ax_rmse_alias = plt.subplots(figsize=(10.5, 5.2))
    ax_rmse_alias.bar(pos - width, [row["rmse_total"] for row in rmse_rows], width=width, color="#111827", label="Total RMSE", alpha=0.92)
    ax_rmse_alias.bar(pos, [row["rmse_signal"] for row in rmse_rows], width=width, color="#b91c1c", label="Signal RMSE (blueprint tags)", alpha=0.88)
    ax_rmse_alias.bar(pos + width, [row["rmse_noise"] for row in rmse_rows], width=width, color="#0369a1", label="Noise RMSE (blueprint tags)", alpha=0.88)
    ax_rmse_alias.set_xticks(pos)
    ax_rmse_alias.set_xticklabels(labels, rotation=20, ha="right")
    ax_rmse_alias.set_ylabel("RMSE on coefficients")
    ax_rmse_alias.set_title("Tiny Sanity: Coefficient Recovery Error by Model (Tag-defined Signal/Noise)")
    ax_rmse_alias.grid(axis="y", alpha=0.22)
    ax_rmse_alias.legend(frameon=False, fontsize=9)
    fig_rmse_alias.tight_layout()
    fig_rmse_alias.savefig(fig_alias_path, dpi=180)
    plt.close(fig_rmse_alias)

    # Figure 2: True vs Estimated scatter (facet by model).
    n_models = len(model_order)
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))
    fig_scatter, axes_scatter = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14.0, 4.8 * nrows), squeeze=False)
    min_val = float(min(np.min(beta_true_std), *(np.min(estimates[name]) for name in model_order)))
    max_val = float(max(np.max(beta_true_std), *(np.max(estimates[name]) for name in model_order)))
    pad = 0.08 * (max_val - min_val + 1e-12)
    lo, hi = min_val - pad, max_val + pad
    for idx, name in enumerate(model_order):
        ax = axes_scatter[idx // ncols][idx % ncols]
        ax.scatter(beta_true_std, estimates[name], s=30, alpha=0.85, color=colors[name], edgecolor="white", linewidth=0.4)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="#6b7280")
        ax.axhline(0.0, color="#e5e7eb", linewidth=0.9)
        ax.axvline(0.0, color="#e5e7eb", linewidth=0.9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(name)
        ax.set_xlabel("True coefficient")
        ax.set_ylabel("Estimated coefficient")
        ax.grid(alpha=0.2)
    for idx in range(n_models, nrows * ncols):
        axes_scatter[idx // ncols][idx % ncols].axis("off")
    fig_scatter.suptitle("Tiny Sanity: True vs Estimated Coefficients", y=0.99, fontsize=14)
    fig_scatter.tight_layout(rect=[0, 0, 1, 0.98])
    scatter_fig_path = outdir / "tiny_sanity_true_vs_estimated_facet.png"
    fig_scatter.savefig(scatter_fig_path, dpi=180)
    plt.close(fig_scatter)

    # Figure 3: Feature-level absolute error line plot.
    fig_err, ax_err = plt.subplots(figsize=(13.5, 5.6))
    for name in model_order:
        abs_err = np.abs(np.asarray(estimates[name], dtype=float) - beta_true_std)
        ax_err.plot(x, abs_err, marker="o", linewidth=1.8, markersize=3.8, color=colors[name], alpha=0.9, label=name)

    cursor = 0
    for size in [len(g) for g in groups[:-1]]:
        cursor += size
        ax_err.axvline(cursor - 0.5, color="#9ca3af", linestyle="--", linewidth=1)

    step = 10 if p >= 80 else 1
    tick_pos = np.arange(0, p, step, dtype=int)
    tick_labels = [f"X{i + 1}" for i in tick_pos.tolist()]
    ax_err.set_xticks(tick_pos)
    ax_err.set_xticklabels(tick_labels, rotation=90)
    ax_err.set_ylabel(r"Absolute Error $|\hat{\beta}_j - \beta_j|$")
    ax_err.set_xlabel("Feature")
    ax_err.set_title("Tiny Sanity: Feature-level Coefficient Absolute Error")
    ax_err.grid(axis="y", alpha=0.22)
    ax_err.legend(ncol=3, frameon=False, fontsize=9)
    fig_err.tight_layout()
    error_fig_path = outdir / "tiny_sanity_feature_abs_error.png"
    fig_err.savefig(error_fig_path, dpi=180)
    plt.close(fig_err)

    rmse_table_path = outdir / "tiny_sanity_coef_error_breakdown.json"
    rmse_table_path.write_text(json.dumps(_to_serializable(rmse_rows), indent=2), encoding="utf-8")
    mechanism_rmse_path = outdir / "tiny_sanity_mechanism_metrics.json"
    mechanism_rmse_payload = {
        "sets": {
            "strong_idx": [int(v) for v in mechanism_sets["strong_idx"].tolist()],
            "weak_idx": [int(v) for v in mechanism_sets["weak_idx"].tolist()],
            "null_idx": [int(v) for v in mechanism_sets["null_idx"].tolist()],
            "near_zero_idx": [int(v) for v in mechanism_sets["near_zero_idx"].tolist()],
        },
        "metrics": mechanism_rmse_rows,
    }
    mechanism_rmse_path.write_text(json.dumps(_to_serializable(mechanism_rmse_payload), indent=2), encoding="utf-8")
    (outdir / "tiny_sanity_feature_selection.json").write_text(json.dumps(_to_serializable(feature_metrics), indent=2), encoding="utf-8")
    (outdir / "tiny_sanity_group_selection.json").write_text(json.dumps(_to_serializable(group_metrics), indent=2), encoding="utf-8")

    print("=== Tiny sanity check finished ===")
    print(f"figure (rmse): {rmse_fig_path}")
    print(f"figure (scatter): {scatter_fig_path}")
    print(f"figure (feature_error): {error_fig_path}")
    print(f"figure (legacy_alias): {fig_alias_path}")
    print(f"figure (X corr): {corr_fig_path}")
    print(f"figure (PR curve): {pr_fig_path}")
    print(f"summary: {summary_path}")
    print(f"coefficients: {coef_path}")
    print(f"error breakdown: {rmse_table_path}")
    print(f"mechanism metrics: {mechanism_rmse_path}")
    print("beta RMSE:")
    for row in sorted(summary_rows, key=lambda r: r["beta_rmse"]):
        print(f"  {row['method']:<20s} {row['beta_rmse']:.4f}")

    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny grouped-signal sanity check for coefficient recovery.")
    parser.add_argument("--seed", type=int, default=196)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--p", type=int, default=200)
    parser.add_argument("--sigma-noise", type=float, default=0.8)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/tiny_sanity"))
    parser.add_argument("--hs-scale-global", type=float, default=0.3)
    parser.add_argument("--hs-num-warmup", type=int, default=300)
    parser.add_argument("--hs-num-samples", type=int, default=300)
    parser.add_argument("--grrhs-c", type=float, default=5.0)
    parser.add_argument("--grrhs-tau0", type=float, default=None, help="Override GR-RHS tau0. Default: auto-calibrated from blueprint p0.")
    parser.add_argument("--grrhs-tau0-multiplier", type=float, default=1.0, help="Multiplier applied to auto-calibrated tau0 when --grrhs-tau0 is not set.")
    parser.add_argument("--grrhs-eta", type=float, default=1.0)
    parser.add_argument("--grrhs-s0", type=float, default=1.0)
    parser.add_argument("--grrhs-iters", type=int, default=1200)
    parser.add_argument("--grrhs-burnin", type=int, default=600)
    parser.add_argument("--grrhs-tau-slice-w", type=float, default=0.25)
    parser.add_argument("--grrhs-tau-slice-m", type=int, default=180)
    parser.add_argument(
        "--profile",
        type=str,
        default="fair_mixed_uneven",
        choices=[
            "fair_mixed_uneven",
            "fair_dense_weak_grouped",
            "fair_sparse_strong",
            "fair_hetero_mixed_corr",
            "fair_within_group_mixed_signal",
            "toy_mixed_signal_6g",
            "toy_orthogonal_mixed_signal",
        ],
    )
    parser.add_argument("--toy-a", type=float, default=3.0, help="Strong-signal amplitude A for toy profiles.")
    parser.add_argument("--toy-plus-corr", action="store_true", help="Use toy-plus mild within-group correlation.")
    args = parser.parse_args()

    run_tiny_sanity(
        seed=args.seed,
        n=args.n,
        p=args.p,
        sigma_noise=args.sigma_noise,
        outdir=args.outdir,
        hs_scale_global=args.hs_scale_global,
        hs_num_warmup=args.hs_num_warmup,
        hs_num_samples=args.hs_num_samples,
        grrhs_c=args.grrhs_c,
        grrhs_tau0=args.grrhs_tau0,
        grrhs_tau0_multiplier=args.grrhs_tau0_multiplier,
        grrhs_eta=args.grrhs_eta,
        grrhs_s0=args.grrhs_s0,
        grrhs_iters=args.grrhs_iters,
        grrhs_burnin=args.grrhs_burnin,
        grrhs_tau_slice_w=args.grrhs_tau_slice_w,
        grrhs_tau_slice_m=args.grrhs_tau_slice_m,
        profile=args.profile,
        toy_a=args.toy_a,
        toy_plus_corr=args.toy_plus_corr,
    )


if __name__ == "__main__":
    main()
