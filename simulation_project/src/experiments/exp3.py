from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .evaluation import _bridge_ratio_diagnostics, _evaluate_row, _kappa_group_means, _kappa_group_prob_gt
from .fitting import _fit_all_methods
from .reporting import _finalize_experiment_run, _paired_converged_subset, _record_produced_paths
from .runtime import (
    _BAYESIAN_DEFAULT_CHAINS,
    _attempts_used,
    _exp3_gigg_config_for_mode,
    _gigg_config_default,
    _is_bayesian_method,
    _normalize_exp3_gigg_mode,
    _parallel_rows,
    _resolve_convergence_retry_limit,
    _result_diag_fields,
    _sampler_for_standard,
    xi_crit_u0_rho,
)
from ..utils import (
    MASTER_SEED,
    SamplerConfig,
    ensure_dir,
    experiment_seed,
    load_pandas,
    method_result_label,
    print_experiment_result,
    save_dataframe,
    save_json,
    setup_logger,
)

# ---------------------------------------------------------------------------
# EXP3 - Linear Benchmark
#
# Paper-aligned paths:
#   Exp3a: fixed-coefficient settings from GIGG Section 5.1 / Table 1
#   Exp3c: paper-style random-coefficient settings
#
# Legacy GR-RHS stress paths retained for continuity:
#   Exp3b: boundary signal benchmark
#   Exp3d: boundary-focused stress benchmark
#
# Semantic convention used throughout the paper-aligned paths:
#   concentrated = signal in few regressors within an active group
#   distributed  = signal shared across many regressors within an active group
# ---------------------------------------------------------------------------

_BOUNDARY_U0 = 0.5
_BOUNDARY_XI_RATIO = 1.2
_SIGMA2_BOUNDARY = 1.0
_WITHIN_GROUP_MIXED_STRONG = 1.0
_WITHIN_GROUP_MIXED_WEAK = 0.25
_EXP3_HEAVY_METHODS = {"GIGG_MMLE", "GHS_plus"}
_PAPER_FIXED_TARGET_R2 = 0.7
_PAPER_FIXED_TARGET_SNR = _PAPER_FIXED_TARGET_R2 / max(1.0 - _PAPER_FIXED_TARGET_R2, 1e-12)
_PAPER_RANDOM_GROUP_SIZE = 10
_PAPER_RANDOM_DISTRIBUTED_BETA = 0.25

# ---------------------------------------------------------------------------
# Default group configurations for the legacy generic Exp3 builder used by
# boundary-focused stress variants.
#
# Each entry:
#   name          - short label used in output CSV / meta JSON
#   group_sizes   - list of per-group sizes (sum = p)
#   active_groups - group indices containing signal (rest are null)
#
# G10x5 : 5 equal groups of size 10 (p=50)
# CL    : [30,10,5,3,2], signal in large groups
# CS    : [30,10,5,3,2], signal in small groups
# ---------------------------------------------------------------------------
_DEFAULT_EXP3_GROUP_CONFIGS: list[dict[str, Any]] = [
    {"name": "G10x5", "group_sizes": [10, 10, 10, 10, 10],  "active_groups": [0, 1]},
    {"name": "CL",    "group_sizes": [30, 10, 5, 3, 2],     "active_groups": [0, 1]},
    {"name": "CS",    "group_sizes": [30, 10, 5, 3, 2],     "active_groups": [3, 4]},
]


def _paper_fixed_exp3_group_configs() -> list[dict[str, Any]]:
    """Fixed-coefficient settings from the GIGG paper (Section 5.1 / Table 1)."""
    return [
        {
            "name": "C10H",
            "group_sizes": [10, 10, 10, 10, 10],
            "active_groups": [0, 1, 2, 3, 4],
            "paper_pattern": "one_per_group",
            "allowed_signals": ["concentrated"],
            "allowed_env_ids": ["PAPER_RHO08"],
        },
        {
            "name": "D10H",
            "group_sizes": [10, 10, 10, 10, 10],
            "active_groups": [0],
            "paper_pattern": "all_within_groups",
            "allowed_signals": ["distributed"],
            "allowed_env_ids": ["PAPER_RHO08"],
        },
        {
            "name": "C10M",
            "group_sizes": [10, 10, 10, 10, 10],
            "active_groups": [0, 1, 2, 3, 4],
            "paper_pattern": "one_per_group",
            "allowed_signals": ["concentrated"],
            "allowed_env_ids": ["PAPER_RHO06"],
        },
        {
            "name": "D10M",
            "group_sizes": [10, 10, 10, 10, 10],
            "active_groups": [0],
            "paper_pattern": "all_within_groups",
            "allowed_signals": ["distributed"],
            "allowed_env_ids": ["PAPER_RHO06"],
        },
        {
            "name": "C5",
            "group_sizes": [5] * 10,
            "active_groups": [0, 1, 2, 3, 4],
            "paper_pattern": "one_per_group",
            "allowed_signals": ["concentrated"],
            "allowed_env_ids": ["PAPER_RHO08"],
        },
        {
            "name": "D5",
            "group_sizes": [5] * 10,
            "active_groups": [0, 1],
            "paper_pattern": "all_within_groups",
            "allowed_signals": ["distributed"],
            "allowed_env_ids": ["PAPER_RHO08"],
        },
        {
            "name": "C25",
            "group_sizes": [25, 25],
            "active_groups": [0, 1],
            "paper_pattern": "custom_counts",
            "paper_active_counts": [3, 2],
            "allowed_signals": ["concentrated"],
            "allowed_env_ids": ["PAPER_RHO08"],
        },
        {
            "name": "D25",
            "group_sizes": [25, 25],
            "active_groups": [0],
            "paper_pattern": "first_k_in_first_active_group",
            "paper_first_k": 10,
            "allowed_signals": ["distributed"],
            "allowed_env_ids": ["PAPER_RHO08"],
        },
        {
            "name": "CL",
            "group_sizes": [30, 10, 5, 3, 2],
            "active_groups": [0, 1],
            "paper_pattern": "one_per_group",
            "allowed_signals": ["concentrated"],
            "allowed_env_ids": ["PAPER_RHO08"],
        },
        {
            "name": "DL",
            "group_sizes": [30, 10, 5, 3, 2],
            "active_groups": [0],
            "paper_pattern": "all_within_groups",
            "allowed_signals": ["distributed"],
            "allowed_env_ids": ["PAPER_RHO08"],
        },
        {
            "name": "CS",
            "group_sizes": [30, 10, 5, 3, 2],
            "active_groups": [3, 4],
            "paper_pattern": "one_per_group",
            "allowed_signals": ["concentrated"],
            "allowed_env_ids": ["PAPER_RHO08"],
        },
        {
            "name": "DS",
            "group_sizes": [30, 10, 5, 3, 2],
            "active_groups": [2, 3, 4],
            "paper_pattern": "all_within_groups",
            "allowed_signals": ["distributed"],
            "allowed_env_ids": ["PAPER_RHO08"],
        },
    ]


def _paper_fixed_exp3_env_points() -> list[dict[str, Any]]:
    return [
        {
            "env_id": "PAPER_RHO08",
            "setting_block": "paper_fixed_coeff",
            "rho_within": 0.8,
            "rho_between": 0.2,
            "target_snr": float(_PAPER_FIXED_TARGET_SNR),
            "signals": ["concentrated", "distributed"],
        },
        {
            "env_id": "PAPER_RHO06",
            "setting_block": "paper_fixed_coeff",
            "rho_within": 0.6,
            "rho_between": 0.2,
            "target_snr": float(_PAPER_FIXED_TARGET_SNR),
            "signals": ["concentrated", "distributed"],
        },
    ]


def _paper_random_exp3_group_configs(*, total_groups: int = 50) -> list[dict[str, Any]]:
    return [
        {
            "name": f"RND_G{total_groups}x{_PAPER_RANDOM_GROUP_SIZE}",
            "group_sizes": [_PAPER_RANDOM_GROUP_SIZE] * int(total_groups),
            "active_groups": [0],
            "paper_random_coefficients": True,
        }
    ]


def _paper_random_exp3_env_points() -> list[dict[str, Any]]:
    return [
        {
            "env_id": "PAPER_RANDOM_RHO08",
            "setting_block": "paper_random_coeff",
            "rho_within": 0.8,
            "rho_between": 0.2,
            "target_snr": float(_PAPER_FIXED_TARGET_SNR),
            "signals": ["random_coefficient"],
        }
    ]


def _paper_random_concentrated_beta_from_formula(
    *,
    group_size: int,
    rho_within: float,
    beta_distributed: float = _PAPER_RANDOM_DISTRIBUTED_BETA,
) -> float:
    """
    Match the within-group contribution to beta' Sigma beta between:
      distributed: beta_g = a * 1_k
      concentrated: beta_g = (b, 0, ..., 0)

    Under exchangeable Sigma_g = (1-rho) I_k + rho 11',
    this requires:
      b^2 = a^2 * [(1-rho)k + rho k^2].
    """
    k = int(group_size)
    if k <= 0:
        return 0.0
    rho = float(rho_within)
    a = float(beta_distributed)
    return abs(a) * math.sqrt((1.0 - rho) * k + rho * (k ** 2))

def _default_exp3_env_points() -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for rw in [0.8]:
        for snr in [0.2, 1.0, 5.0]:
            points.append(
                {
                    "env_id": f"RW{int(round(rw*10)):02d}_SNR{int(round(snr*10)):02d}",
                    "setting_block": "core_axis",
                    "rho_within": float(rw),
                    "rho_between": 0.2,
                    "target_snr": float(snr),
                    "signals": ["concentrated", "distributed", "boundary"],
                }
            )
    return points




def _exp3_keep_env_point_rw_gt_rb(ep: dict[str, Any]) -> bool:
    """Keep Exp3 environment point iff within-group correlation is strictly larger."""
    rw = float(ep.get("rho_within", float("nan")))
    rb = float(ep.get("rho_between", float("nan")))
    return bool(np.isfinite(rw) and np.isfinite(rb) and (rw > rb))


def _exp3_filter_env_points_rw_gt_rb(points: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for ep in points:
        epi = dict(ep)
        if _exp3_keep_env_point_rw_gt_rb(epi):
            kept.append(epi)
        else:
            dropped.append(epi)
    return kept, dropped


def _exp3_is_anchor_setting(
    *,
    group_cfg: dict[str, Any],
    env_id: str,
    signal: str,
    boundary_xi_ratio: float,
) -> bool:
    group_name = str(group_cfg.get("name", "")).strip()
    env_name = str(env_id).strip()
    sig = str(signal).strip().lower()
    if group_name == "G10x5" and env_name == "RW08_SNR10":
        if sig == "boundary":
            return abs(float(boundary_xi_ratio) - float(_BOUNDARY_XI_RATIO)) < 1e-9
        return sig in {"concentrated", "distributed"}
    if group_name in {"C10H", "D10H"} and env_name == "PAPER_RHO08":
        return sig in {"concentrated", "distributed"}
    if sig == "boundary":
        return False
    return False


def _build_paper_random_beta(
    group_sizes: Sequence[int],
    *,
    rng: np.random.Generator,
    rho_within: float,
) -> np.ndarray:
    from ..utils import canonical_groups

    groups = canonical_groups(group_sizes)
    beta = np.zeros(sum(group_sizes), dtype=float)
    if not groups:
        return beta

    def _apply_concentrated(gid: int) -> None:
        idx = np.asarray(groups[gid], dtype=int)
        if idx.size:
            beta[idx[0]] = _paper_random_concentrated_beta_from_formula(
                group_size=int(idx.size),
                rho_within=float(rho_within),
            )

    def _apply_distributed(gid: int) -> None:
        idx = np.asarray(groups[gid], dtype=int)
        if idx.size:
            beta[idx] = float(_PAPER_RANDOM_DISTRIBUTED_BETA)

    if float(rng.uniform()) < 0.5:
        _apply_concentrated(0)
    else:
        _apply_distributed(0)

    for gid in range(1, len(groups)):
        u = float(rng.uniform())
        if u < 0.2:
            _apply_concentrated(gid)
        elif u < 0.4:
            _apply_distributed(gid)
    return beta


def _build_benchmark_beta(
    signal: str,
    group_sizes: Sequence[int],
    *,
    group_cfg: dict[str, Any] | None = None,
    active_groups: Sequence[int] | None = None,
    sigma2: float = 1.0,
    p: int | None = None,
    boundary_u0: float = _BOUNDARY_U0,
    boundary_xi_ratio: float = _BOUNDARY_XI_RATIO,
    boundary_rho_profile: float | None = None,
    rho_within: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Construct beta for each benchmark signal structure.

    Public Exp3 paths use concentrated, distributed, boundary, and
    random_coefficient. A few older helper-only signal labels are retained
    below for backward-compatible internal stress utilities.
    """
    from ..utils import canonical_groups
    groups = canonical_groups(group_sizes)
    total_p = int(p or sum(group_sizes))
    beta = np.zeros(total_p, dtype=float)
    _active = list(active_groups) if active_groups is not None else [0, 1]
    cfg = dict(group_cfg or {})
    paper_pattern = str(cfg.get("paper_pattern", "")).strip().lower()

    if signal == "random_coefficient":
        rng_local = rng if rng is not None else np.random.default_rng(12345)
        if rho_within is None:
            raise ValueError("rho_within must be provided for random_coefficient signal generation.")
        return _build_paper_random_beta(group_sizes, rng=rng_local, rho_within=float(rho_within))

    if paper_pattern and signal in {"concentrated", "distributed"}:
        if paper_pattern == "one_per_group":
            for gid in _active:
                idx = np.asarray(groups[gid], dtype=int)
                if idx.size:
                    beta[idx[0]] = 1.0
            return beta
        if paper_pattern == "all_within_groups":
            for gid in _active:
                idx = np.asarray(groups[gid], dtype=int)
                if idx.size:
                    beta[idx] = 1.0
            return beta
        if paper_pattern == "custom_counts":
            counts = list(cfg.get("paper_active_counts", []))
            for gid, k in zip(_active, counts):
                idx = np.asarray(groups[gid], dtype=int)
                k_use = int(max(0, min(int(k), int(idx.size))))
                if k_use > 0:
                    beta[idx[:k_use]] = 1.0
            return beta
        if paper_pattern == "first_k_in_first_active_group":
            if _active:
                gid = int(_active[0])
                idx = np.asarray(groups[gid], dtype=int)
                k_use = int(max(0, min(int(cfg.get("paper_first_k", 0)), int(idx.size))))
                if k_use > 0:
                    beta[idx[:k_use]] = 1.0
            return beta

    if signal == "concentrated":
        for gid in _active:
            idx = np.asarray(groups[gid], dtype=int)
            if idx.size:
                beta[idx[0]] = 1.0
    elif signal == "distributed":
        for gid in _active:
            idx = np.asarray(groups[gid], dtype=int)
            if idx.size:
                beta[idx] = 1.0 / math.sqrt(len(idx))
    elif signal == "boundary":
        if boundary_rho_profile is None:
            raise ValueError("boundary_rho_profile must be provided for boundary signal calibration.")
        xi_c = xi_crit_u0_rho(u0=float(boundary_u0), rho=float(boundary_rho_profile))
        for gid in _active:
            idx = np.asarray(groups[gid], dtype=int)
            pg = len(idx)
            mu_g = float(boundary_xi_ratio) * xi_c * pg
            beta_val = math.sqrt(2.0 * float(sigma2) * mu_g / pg)
            beta[idx] = beta_val
    elif signal == "within_group_mixed":
        # Controlled heterogeneity: one strong coefficient plus weak nonzeros
        # within each active group, useful for stress-testing group+local shrinkage.
        strong = float(_WITHIN_GROUP_MIXED_STRONG)
        weak = float(_WITHIN_GROUP_MIXED_WEAK)
        for gid in _active:
            idx = np.asarray(groups[gid], dtype=int)
            if idx.size == 0:
                continue
            beta[idx[0]] = strong
            if idx.size > 1:
                beta[idx[1:]] = weak
    elif signal in {"half_dense", "dense"}:
        rng_local = rng if rng is not None else np.random.default_rng(12345)
        density = 0.2 if signal == "half_dense" else 0.6
        n_active = max(1, int(round(total_p * density)))
        active_idx = rng_local.choice(np.arange(total_p), size=n_active, replace=False)
        mags = rng_local.uniform(0.3, 1.2, size=n_active)
        signs = rng_local.choice([-1.0, 1.0], size=n_active)
        beta[active_idx] = mags * signs
    else:
        raise ValueError(f"unknown signal structure: {signal!r}")
    return beta


def _exp3_worker(
    task: dict[str, Any] | tuple,
) -> list[dict[str, Any]]:
    from .dgp.grouped_linear import generate_orthonormal_block_design
    from ..utils import canonical_groups, sample_correlated_design

    if isinstance(task, dict):
        sid = int(task["setting_id"])
        signal = str(task["signal"])
        group_cfg = dict(task["group_cfg"])
        setting_block = str(task["setting_block"])
        env_id = str(task["env_id"])
        design_type = str(task["design_type"])
        rho_within = float(task["rho_within"])
        rho_between = float(task["rho_between"])
        target_snr = float(task["target_snr"])
        boundary_xi_ratio = float(task.get("boundary_xi_ratio", _BOUNDARY_XI_RATIO))
        r = int(task["replicate_id"])
        seed_base = int(task["seed_base"])
        n_train = int(task.get("n_train", 100))
        n_test = int(task["n_test"])
        sampler = task["sampler"]
        methods_raw = task.get("methods", None)
        if methods_raw is None:
            methods = [str(task["method"])]
        else:
            methods = [str(m) for m in methods_raw]
        if not methods:
            raise ValueError("Exp3 task must include at least one method.")
        gigg_config = dict(task["gigg_config"])
        gigg_mode = str(task.get("gigg_mode", "paper_ref"))
        bayes_min_chains = task.get("bayes_min_chains")
        method_jobs = int(task.get("method_jobs", 1))
        enforce_conv = bool(task["enforce_bayes_convergence"])
        max_retries = int(task["max_convergence_retries"])
        grrhs_kwargs = dict(task["grrhs_kwargs"])
        log_path = str(task.get("log_path", "")).strip() or None
    else:
        if len(task) == 19:
            sid, signal, group_cfg, setting_block, env_id, design_type, rho_within, rho_between, target_snr, r, seed_base, n_test, sampler, methods, gigg_config, bayes_min_chains, enforce_conv, max_retries, grrhs_kwargs = task
            boundary_xi_ratio = float(_BOUNDARY_XI_RATIO)
            n_train = 100
            method_jobs = 1
            log_path = None
        elif len(task) == 20:
            sid, signal, group_cfg, setting_block, env_id, design_type, rho_within, rho_between, target_snr, boundary_xi_ratio, r, seed_base, n_test, sampler, methods, gigg_config, bayes_min_chains, enforce_conv, max_retries, grrhs_kwargs = task
            n_train = 100
            method_jobs = 1
            log_path = None
        else:
            sid, signal, group_cfg, setting_block, env_id, design_type, rho_within, rho_between, target_snr, boundary_xi_ratio, r, seed_base, n_train, n_test, sampler, methods, gigg_config, bayes_min_chains, method_jobs, enforce_conv, max_retries, grrhs_kwargs = task
            log_path = None
        methods = [str(m) for m in methods]
        gigg_mode = "paper_ref"
    group_cfg_name: str = str(group_cfg["name"])
    methods_upper = {m.upper() for m in methods}
    gigg_config = dict(gigg_config)
    gigg_mode_name = _normalize_exp3_gigg_mode(gigg_mode)
    if "GIGG_MMLE" in methods_upper and not bool(gigg_config.get("allow_budget_retry", False)):
        gigg_config["extra_retry"] = 0
        gigg_config.pop("retry_cap", None)
    s = experiment_seed(3, int(sid), r, master_seed=int(seed_base))

    group_sizes: list[int] = list(group_cfg["group_sizes"])
    active_groups: list[int] = list(group_cfg["active_groups"])

    sigma2_boundary = float(_SIGMA2_BOUNDARY)
    boundary_rho_profile = float(rho_within) / math.sqrt(max(sigma2_boundary, 1e-12))
    boundary_xi_crit = float("nan")
    boundary_xi = float("nan")
    if signal == "boundary":
        boundary_xi_crit = xi_crit_u0_rho(u0=float(_BOUNDARY_U0), rho=boundary_rho_profile)
        boundary_xi = float(boundary_xi_ratio) * boundary_xi_crit

    beta0 = _build_benchmark_beta(
        signal,
        group_sizes,
        group_cfg=group_cfg,
        active_groups=active_groups,
        sigma2=sigma2_boundary if signal == "boundary" else 1.0,
        boundary_u0=float(_BOUNDARY_U0),
        boundary_xi_ratio=float(boundary_xi_ratio),
        boundary_rho_profile=boundary_rho_profile if signal == "boundary" else None,
        rho_within=float(rho_within),
        rng=np.random.default_rng(s + 101),
    )
    p = int(sum(group_sizes))

    # Construct training dataset
    if str(design_type) == "orthonormal":
        from .dgp.grouped_linear import generate_orthonormal_block_design
        X_train = generate_orthonormal_block_design(n=n_train, group_sizes=group_sizes, seed=s)
        cov_x = np.eye(p, dtype=float)
    else:
        X_train, cov_x = sample_correlated_design(n=n_train, group_sizes=group_sizes, rho_within=rho_within, rho_between=rho_between, seed=s)

    # sigma2: SNR-calibrated for concentrated/distributed; fixed for boundary
    if signal == "boundary":
        sigma2 = sigma2_boundary
    else:
        from .dgp.grouped_linear import sigma2_for_target_snr as _s2
        sigma2 = _s2(beta=beta0, cov_x=cov_x, target_snr=float(target_snr))

    rng_y = np.random.default_rng(s + 17)
    y_train = X_train @ beta0 + rng_y.normal(0.0, math.sqrt(sigma2), n_train)

    # Test set: fresh X and noise, same DGP parameters
    if str(design_type) == "orthonormal":
        X_test = generate_orthonormal_block_design(n=n_test, group_sizes=group_sizes, seed=s + 77777)
    else:
        X_test, _ = sample_correlated_design(n=n_test, group_sizes=group_sizes, rho_within=rho_within, rho_between=rho_between, seed=s + 77777)
    rng_yt = np.random.default_rng(s + 88888)
    y_test = X_test @ beta0 + rng_yt.normal(0.0, math.sqrt(sigma2), n_test)

    groups = canonical_groups(group_sizes)
    p0 = int(np.sum(np.abs(beta0) > 1e-12))
    p0_signal_groups = int(
        np.sum(
            [
                int(np.any(np.abs(beta0[np.asarray(g, dtype=int)]) > 1e-12))
                for g in groups
            ]
        )
    )
    n_groups = len(group_sizes)
    if signal in {"half_dense", "dense", "random_coefficient"}:
        active_group_set = {
            gid for gid, g in enumerate(groups)
            if np.any(np.abs(beta0[np.asarray(g, dtype=int)]) > 1e-12)
        }
        active_groups = sorted(active_group_set)
    else:
        active_group_set = set(active_groups)
    null_groups = [g for g in range(n_groups) if g not in active_group_set]
    signal_group_mask = np.asarray([g in active_group_set for g in range(n_groups)], dtype=bool)

    fits = _fit_all_methods(
        X_train, y_train, groups,
        task="gaussian", seed=s, p0=p0,
        p0_groups=p0_signal_groups,
        sampler=sampler, methods=methods, gigg_config=gigg_config,
        bayes_min_chains=int(bayes_min_chains) if bayes_min_chains is not None else None,
        grrhs_kwargs=grrhs_kwargs or {},
        enforce_bayes_convergence=bool(enforce_conv),
        max_convergence_retries=int(max_retries),
        method_jobs=int(method_jobs),
    )

    out_rows: list[dict[str, Any]] = []
    for method, res in fits.items():
        metrics = _evaluate_row(res, beta0, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        bridge_diag = _bridge_ratio_diagnostics(
            res,
            groups=groups,
            X=X_train,
            y=y_train,
            signal_group_mask=signal_group_mask,
        )
        kappa_null_mean = float("nan")
        kappa_signal_mean = float("nan")
        kappa_null_prob_gt_u0 = float("nan")
        kappa_signal_prob_gt_u0 = float("nan")
        mmle_q_json = ""
        mmle_b_json = ""
        mmle_a_json = ""
        mmle_q_mean = float("nan")
        mmle_q_std = float("nan")
        mmle_q_min = float("nan")
        mmle_q_max = float("nan")
        mmle_q_n_groups = 0
        if method == "GR_RHS" and res.beta_mean is not None:
            km = _kappa_group_means(res, n_groups)
            kp = _kappa_group_prob_gt(res, n_groups, threshold=float(_BOUNDARY_U0))
            _sig_vals = [km[g] for g in active_groups if not np.isnan(km[g])]
            _null_vals = [km[g] for g in null_groups if not np.isnan(km[g])]
            _sig_probs = [kp[g] for g in active_groups if not np.isnan(kp[g])]
            _null_probs = [kp[g] for g in null_groups if not np.isnan(kp[g])]
            kappa_signal_mean = float(np.mean(_sig_vals)) if _sig_vals else float("nan")
            kappa_null_mean = float(np.mean(_null_vals)) if _null_vals else float("nan")
            kappa_signal_prob_gt_u0 = float(np.mean(_sig_probs)) if _sig_probs else float("nan")
            kappa_null_prob_gt_u0 = float(np.mean(_null_probs)) if _null_probs else float("nan")
        if method == "GIGG_MMLE":
            diag = dict(res.diagnostics or {})
            mmle_est = diag.get("mmle_estimate", {}) if isinstance(diag, dict) else {}
            q_vals = mmle_est.get("q_estimate", []) if isinstance(mmle_est, dict) else []
            a_vals = mmle_est.get("a_estimate", []) if isinstance(mmle_est, dict) else []
            if isinstance(q_vals, list):
                mmle_q_json = json.dumps(q_vals, ensure_ascii=False)
                mmle_b_json = mmle_q_json
                q_arr = np.asarray(q_vals, dtype=float)
                q_arr = q_arr[np.isfinite(q_arr)]
                if q_arr.size:
                    mmle_q_mean = float(np.mean(q_arr))
                    mmle_q_std = float(np.std(q_arr, ddof=0))
                    mmle_q_min = float(np.min(q_arr))
                    mmle_q_max = float(np.max(q_arr))
                    mmle_q_n_groups = int(q_arr.size)
            if isinstance(a_vals, list):
                mmle_a_json = json.dumps(a_vals, ensure_ascii=False)
        out_rows.append({
            "setting_id": int(sid),
            "gigg_mode": str(gigg_mode_name),
            "group_config": group_cfg_name,
            "signal": signal,
            "setting_block": str(setting_block),
            "env_id": str(env_id),
            "design_type": str(design_type),
            "rho_within": float(rho_within),
            "rho_between": float(rho_between),
            "target_snr": float(target_snr),
            "target_r2": float(target_snr / (1.0 + target_snr)) if np.isfinite(target_snr) and target_snr > 0.0 else float("nan"),
            "sigma2": float(sigma2),
            "boundary_u0": float(_BOUNDARY_U0) if signal == "boundary" else float("nan"),
            "boundary_xi_ratio": float(boundary_xi_ratio) if signal == "boundary" else float(_BOUNDARY_XI_RATIO),
            "boundary_rho_profile": boundary_rho_profile if signal == "boundary" else float("nan"),
            "boundary_xi_crit": boundary_xi_crit,
            "boundary_xi": boundary_xi,
            "replicate_id": int(r),
            "method": method,
            "status": res.status,
            "converged": bool(res.converged),
            "fit_attempts": _attempts_used(res),
            "kappa_null_mean": kappa_null_mean,
            "kappa_signal_mean": kappa_signal_mean,
            "kappa_null_prob_gt_u0": kappa_null_prob_gt_u0,
            "kappa_signal_prob_gt_u0": kappa_signal_prob_gt_u0,
            "mmle_q_estimate_json": mmle_q_json,
            "mmle_b_estimate_json": mmle_b_json,
            "mmle_a_estimate_json": mmle_a_json,
            "mmle_q_mean": mmle_q_mean,
            "mmle_q_std": mmle_q_std,
            "mmle_q_min": mmle_q_min,
            "mmle_q_max": mmle_q_max,
            "mmle_q_n_groups": int(mmle_q_n_groups),
            **_result_diag_fields(res),
            **bridge_diag,
            **metrics,
        })
        print_experiment_result(
            "Exp3",
            out_rows[-1],
            context_keys=["setting_id", "replicate_id", "method", "signal", "group_config", "env_id"],
            metric_keys=["mse_overall", "mse_null", "mse_signal", "lpd_test", "group_auroc"],
            log_path=log_path,
        )
    return out_rows


def run_exp3_linear_benchmark(
    n_jobs: int = 1,
    method_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 20,
    save_dir: str = "outputs/simulation_project",
    *,
    skip_run_analysis: bool = False,
    archive_artifacts: bool = True,
    signal_types: Sequence[str] | None = None,
    boundary_xi_ratio_list: Sequence[float] | None = None,
    env_points: Sequence[dict[str, Any]] | None = None,
    bayes_min_chains: int | None = None,
    group_configs: list[dict[str, Any]] | None = None,
    methods: Sequence[str] | None = None,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
    n_train: int = 100,
    n_test: int = 30,
    grrhs_extra_kwargs: dict | None = None,
    sampler_overrides: dict[str, Any] | None = None,
    gigg_mode: str = "paper_ref",
    heavy_methods_anchor_only: bool = False,
    result_dir_name: str = "exp3_linear_benchmark",
    exp_key: str = "exp3",
) -> Dict[str, str]:
    """
    Generic Exp3 benchmark runner.

    Paper-aligned experiments call this helper with explicit setting tables
    (`Exp3a`) or random-coefficient settings (`Exp3c`). Legacy boundary-focused
    stress experiments (`Exp3b`, `Exp3d`) also reuse this runner.

    Signal types (default ["concentrated", "distributed", "boundary"]):
      concentrated: few active regressors within each active group.
      distributed: many active regressors within each active group.
      boundary: signal set to xi_ratio * xi_crit(u0=0.5, rho_profile),
                with rho_profile = rho_within / sqrt(sigma2_boundary).
                xi_ratio values come from boundary_xi_ratio_list (default [1.2]).
      random_coefficient: paper 5.1 random group-level mixture with
                first group forced active and remaining groups sampled from
                concentrated/distributed/null with probabilities 0.2/0.2/0.6.

    bayes_min_chains:
      Minimum number of chains for Bayesian methods in Exp3.
      Default: 4.

    Methods:
      GR_RHS, GHS_plus, GIGG_MMLE, RHS, LASSO_CV, OLS.

    gigg_mode:
      paper_ref: strict gigg-master-aligned reference mode.

    sampler_overrides:
      Optional overrides for SamplerConfig fields such as chains, warmup,
      post_warmup_draws, adapt_delta, and max_treedepth.
    """
    pd = load_pandas()
    produced: set[Path] = set()

    base = Path(save_dir)
    out_name = str(result_dir_name).strip() or "exp3_linear_benchmark"
    exp_key_name = str(exp_key).strip().lower() or "exp3"
    out_dir = ensure_dir(base / "results" / out_name)
    fig_dir = ensure_dir(base / "figures" / out_name)
    tab_dir = ensure_dir(base / "tables" / out_name)
    log = setup_logger(str(exp_key_name), base / "logs" / f"{out_name}.log")
    log_path = str(base / "logs" / f"{out_name}.log")

    sampler = _sampler_for_standard()
    if sampler_overrides:
        sampler = SamplerConfig(**{**sampler.__dict__, **dict(sampler_overrides)})
    log.info("Exp3 sampler config: %s", sampler)
    bayes_min_chains_use = int(bayes_min_chains) if bayes_min_chains is not None else int(_BAYESIAN_DEFAULT_CHAINS)
    bayes_min_chains_use = max(1, int(bayes_min_chains_use))
    _exp3_methods = ["GR_RHS", "GHS_plus", "GIGG_MMLE", "RHS", "LASSO_CV", "OLS"]
    _exp3_methods_set = set(_exp3_methods)
    methods_use = [m for m in (methods or _exp3_methods) if m in _exp3_methods_set]
    if not methods_use:
        methods_use = list(_exp3_methods)
    bayes_methods_use = [m for m in methods_use if _is_bayesian_method(m)]
    classical_methods_use = [m for m in methods_use if not _is_bayesian_method(m)]
    gigg_mode_name = _normalize_exp3_gigg_mode(gigg_mode)
    gigg_cfg = _exp3_gigg_config_for_mode(_gigg_config_default(), gigg_mode=gigg_mode_name)
    retry_limit = _resolve_convergence_retry_limit(
        max_convergence_retries,
        until_bayes_converged=bool(until_bayes_converged),
    )
    design_mode = "standard"

    signals = list(signal_types or ["concentrated", "distributed", "boundary"])
    gc_list: list[dict[str, Any]] = list(group_configs) if group_configs is not None else list(_DEFAULT_EXP3_GROUP_CONFIGS)

    boundary_xi_ratios = sorted(
        {
            float(v)
            for v in (boundary_xi_ratio_list or [_BOUNDARY_XI_RATIO])
            if np.isfinite(float(v)) and float(v) > 0.0
        }
    )
    if not boundary_xi_ratios:
        raise ValueError("boundary_xi_ratio_list must contain at least one positive finite value.")

    # settings:
    #   sid, signal, group_cfg, setting_block, env_id, design_type, rho_within,
    #   rho_between, target_snr, boundary_xi_ratio
    settings: list[tuple[int, str, dict, str, str, str, float, float, float, float]] = []
    sid = 0
    env_points_used: list[dict[str, Any]] = []

    points_raw = list(env_points) if env_points is not None else _default_exp3_env_points()
    points, dropped_points = _exp3_filter_env_points_rw_gt_rb(points_raw)
    if dropped_points:
        log.warning(
            "Exp3: dropped %d env point(s) with rw<=rb: %s",
            len(dropped_points),
            [
                (str(ep.get("env_id", "?")), float(ep.get("rho_within", float("nan"))), float(ep.get("rho_between", float("nan"))))
                for ep in dropped_points
            ],
        )
    if not points:
        raise ValueError("No valid Exp3 env points remain after enforcing rw>rb.")
    for ep in points:
        env_points_used.append(
            {
                "env_id": str(ep["env_id"]),
                "setting_block": str(ep.get("setting_block", "custom")),
                "rho_within": float(ep["rho_within"]),
                "rho_between": float(ep["rho_between"]),
                "target_snr": float(ep["target_snr"]),
                "signals": [str(s) for s in ep.get("signals", signals)],
            }
        )
    for gc in gc_list:
        allowed_signals = {str(s) for s in gc.get("allowed_signals", signals)}
        allowed_env_ids = {str(v) for v in gc.get("allowed_env_ids", [])}
        for ep in env_points_used:
            if allowed_env_ids and str(ep["env_id"]) not in allowed_env_ids:
                continue
            sig_set = set(ep.get("signals", signals))
            for signal in signals:
                if signal not in allowed_signals:
                    continue
                if signal not in sig_set:
                    continue
                rho = float(ep["rho_within"])
                rhob = float(ep["rho_between"])
                snr = float(ep["target_snr"])
                design = "orthonormal" if rho == 0.0 and rhob == 0.0 else "correlated"
                if signal == "boundary":
                    for xi_ratio_v in boundary_xi_ratios:
                        sid += 1
                        settings.append(
                            (
                                sid,
                                signal,
                                gc,
                                str(ep["setting_block"]),
                                str(ep["env_id"]),
                                design,
                                rho,
                                rhob,
                                snr,
                                float(xi_ratio_v),
                            )
                        )
                else:
                    sid += 1
                    settings.append(
                        (
                            sid,
                            signal,
                            gc,
                            str(ep["setting_block"]),
                            str(ep["env_id"]),
                            design,
                            rho,
                            rhob,
                            snr,
                            float(_BOUNDARY_XI_RATIO),
                        )
                    )

    grrhs_kw: dict = {"tau_target": "groups"}
    if grrhs_extra_kwargs:
        grrhs_kw.update(grrhs_extra_kwargs)
    bayes_tasks: list[dict[str, Any]] = []
    classical_tasks: list[dict[str, Any]] = []
    for (sid_v, signal_v, gc_v, block_v, env_v, dt_v, rho_v, rhob_v, snr_v, bxi_v) in settings:
        is_anchor = _exp3_is_anchor_setting(
            group_cfg=dict(gc_v),
            env_id=str(env_v),
            signal=str(signal_v),
            boundary_xi_ratio=float(bxi_v),
        )
        bayes_methods_for_setting = list(bayes_methods_use)
        if bool(heavy_methods_anchor_only) and not bool(is_anchor):
            bayes_methods_for_setting = [m for m in bayes_methods_for_setting if m not in _EXP3_HEAVY_METHODS]
        for r in range(1, int(repeats) + 1):
            base_task = {
                "setting_id": int(sid_v),
                "signal": str(signal_v),
                "group_cfg": dict(gc_v),
                "setting_block": str(block_v),
                "env_id": str(env_v),
                "design_type": str(dt_v),
                "rho_within": float(rho_v),
                "rho_between": float(rhob_v),
                "target_snr": float(snr_v),
                "boundary_xi_ratio": float(bxi_v),
                "replicate_id": int(r),
                "seed_base": int(seed),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "sampler": sampler,
                "gigg_config": dict(gigg_cfg),
                "gigg_mode": str(gigg_mode_name),
                "bayes_min_chains": int(bayes_min_chains_use),
                "method_jobs": int(method_jobs),
                "enforce_bayes_convergence": bool(enforce_bayes_convergence),
                "max_convergence_retries": int(retry_limit),
                "grrhs_kwargs": dict(grrhs_kw),
                "log_path": log_path,
            }
            if bayes_methods_for_setting:
                bayes_task = dict(base_task)
                bayes_task["methods"] = list(bayes_methods_for_setting)
                bayes_tasks.append(bayes_task)
            if classical_methods_use:
                classical_task = dict(base_task)
                classical_task["methods"] = list(classical_methods_use)
                classical_tasks.append(classical_task)

    n_data_batches = len(settings) * int(repeats)
    n_method_evals = n_data_batches * len(methods_use)
    setting_rows_by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    setting_progress_path = out_dir / "setting_progress.jsonl"
    bayes_tasks_expected_by_setting: dict[int, int] = defaultdict(int)
    bayes_tasks_done_by_setting: dict[int, int] = defaultdict(int)
    tasks_expected_by_setting: dict[int, int] = defaultdict(int)
    tasks_done_by_setting: dict[int, int] = defaultdict(int)
    bayes_settings_completed: set[int] = set()
    settings_completed: set[int] = set()
    settings_meta: dict[int, dict[str, Any]] = {}
    base_keys = [
        "gigg_mode",
        "group_config",
        "signal",
        "setting_block",
        "env_id",
        "design_type",
        "rho_within",
        "rho_between",
        "target_snr",
        "boundary_xi_ratio",
    ]

    for (sid_v, signal_v, gc_v, block_v, env_v, dt_v, rho_v, rhob_v, snr_v, bxi_v) in settings:
        settings_meta[int(sid_v)] = {
            "signal": str(signal_v),
            "group_config": str(gc_v["name"]),
            "env_id": str(env_v),
            "boundary_xi_ratio": float(bxi_v),
        }
    for task in bayes_tasks:
        bayes_tasks_expected_by_setting[int(task["setting_id"])] += 1
        tasks_expected_by_setting[int(task["setting_id"])] += 1
    for task in classical_tasks:
        tasks_expected_by_setting[int(task["setting_id"])] += 1

    def _summarize_completed_setting(sid_local: int) -> dict[str, Any]:
        rows_local = list(setting_rows_by_id.get(sid_local, []))
        raw_local = pd.DataFrame(rows_local)
        methods_present = sorted(set(raw_local["method"].astype(str).tolist())) if not raw_local.empty and "method" in raw_local.columns else []
        counts_local = pd.DataFrame()
        paired_local = pd.DataFrame()
        paired_stats_local = pd.DataFrame()
        metric_local = pd.DataFrame()
        common_reps = 0

        if not raw_local.empty:
            counts_local = raw_local.groupby("method", as_index=False).agg(
                n_reps_total=("replicate_id", "count"),
                n_reps_ok=("status", lambda s: int((s == "ok").sum())),
                n_reps_converged=("converged", lambda s: int(s.fillna(False).astype(bool).sum())),
            )
            paired_local, paired_stats_local = _paired_converged_subset(
                raw_local,
                group_cols=base_keys,
                method_col="method",
                replicate_col="replicate_id",
                converged_col="converged",
                required_cols=["mse_null", "mse_signal", "mse_overall", "lpd_test"],
                method_levels=None,
            )
            if not paired_stats_local.empty and "n_common_replicates" in paired_stats_local.columns:
                common_reps = int(paired_stats_local["n_common_replicates"].iloc[0])
            if not paired_local.empty:
                metric_local = paired_local.groupby("method", as_index=False).agg(
                    mse_null=("mse_null", "mean"),
                    mse_signal=("mse_signal", "mean"),
                    mse_overall=("mse_overall", "mean"),
                    lpd_test=("lpd_test", "mean"),
                    coverage_95=("coverage_95", "mean"),
                    avg_ci_length=("avg_ci_length", "mean"),
                    kappa_null_mean=("kappa_null_mean", "mean"),
                    kappa_signal_mean=("kappa_signal_mean", "mean"),
                    kappa_null_prob_gt_u0=("kappa_null_prob_gt_u0", "mean"),
                    kappa_signal_prob_gt_u0=("kappa_signal_prob_gt_u0", "mean"),
                )
                pair_counts_local = paired_local.groupby("method", as_index=False).agg(n_reps_paired=("replicate_id", "nunique"))
                metric_local = metric_local.merge(pair_counts_local, on="method", how="left")
            else:
                metric_local = counts_local.copy()
                metric_local["n_reps_paired"] = 0

        method_rows: list[dict[str, Any]] = []
        if not counts_local.empty:
            counts_lookup = {str(r["method"]): r for r in counts_local.to_dict(orient="records")}
            metric_lookup = {str(r["method"]): r for r in metric_local.to_dict(orient="records")} if not metric_local.empty else {}
            for method_name in methods_present:
                c_row = counts_lookup.get(method_name, {})
                m_row = metric_lookup.get(method_name, {})
                method_rows.append({
                    "method": method_name,
                    "method_label": method_result_label(method_name),
                    "n_reps_total": int(c_row.get("n_reps_total", 0) or 0),
                    "n_reps_ok": int(c_row.get("n_reps_ok", 0) or 0),
                    "n_reps_converged": int(c_row.get("n_reps_converged", 0) or 0),
                    "n_reps_paired": int(m_row.get("n_reps_paired", 0) or 0),
                    "mse_overall": float(m_row.get("mse_overall", float("nan"))),
                    "mse_null": float(m_row.get("mse_null", float("nan"))),
                    "mse_signal": float(m_row.get("mse_signal", float("nan"))),
                    "lpd_test": float(m_row.get("lpd_test", float("nan"))),
                    "coverage_95": float(m_row.get("coverage_95", float("nan"))),
                    "avg_ci_length": float(m_row.get("avg_ci_length", float("nan"))),
                    "kappa_null_mean": float(m_row.get("kappa_null_mean", float("nan"))),
                    "kappa_signal_mean": float(m_row.get("kappa_signal_mean", float("nan"))),
                    "kappa_null_prob_gt_u0": float(m_row.get("kappa_null_prob_gt_u0", float("nan"))),
                    "kappa_signal_prob_gt_u0": float(m_row.get("kappa_signal_prob_gt_u0", float("nan"))),
                })

        ranked_rows = [r for r in method_rows if np.isfinite(float(r.get("mse_overall", float("nan"))))]
        ranked_rows = sorted(ranked_rows, key=lambda r: float(r["mse_overall"]))
        best_method = ranked_rows[0]["method"] if ranked_rows else None

        gr_row = next((r for r in method_rows if r["method"] == "GR_RHS"), None)
        rhs_row = next((r for r in method_rows if r["method"] == "RHS"), None)
        comparison: dict[str, Any] = {}
        if gr_row is not None and rhs_row is not None:
            gr_mse = float(gr_row.get("mse_overall", float("nan")))
            rhs_mse = float(rhs_row.get("mse_overall", float("nan")))
            gr_lpd = float(gr_row.get("lpd_test", float("nan")))
            rhs_lpd = float(rhs_row.get("lpd_test", float("nan")))
            comparison = {
                "gr_rhs_vs_rhs_mse_diff": gr_mse - rhs_mse if np.isfinite(gr_mse) and np.isfinite(rhs_mse) else float("nan"),
                "gr_rhs_vs_rhs_lpd_diff": gr_lpd - rhs_lpd if np.isfinite(gr_lpd) and np.isfinite(rhs_lpd) else float("nan"),
            }

        return {
            "setting_id": int(sid_local),
            "meta": dict(settings_meta.get(sid_local, {})),
            "n_methods_present": int(len(method_rows)),
            "common_paired_reps": int(common_reps),
            "best_method_by_mse": best_method,
            "methods": method_rows,
            "comparison": comparison,
        }

    def _log_completed_setting_analysis(summary_obj: dict[str, Any]) -> None:
        meta = dict(summary_obj.get("meta", {}))
        method_rows = list(summary_obj.get("methods", []))
        ranked_rows = [r for r in method_rows if np.isfinite(float(r.get("mse_overall", float("nan"))))]
        ranked_rows = sorted(ranked_rows, key=lambda r: float(r["mse_overall"]))
        lead = (
            f"Exp3 analysis: setting_id={int(summary_obj.get('setting_id', -1))} "
            f"signal={str(meta.get('signal', '?'))} group={str(meta.get('group_config', '?'))} "
            f"env={str(meta.get('env_id', '?'))} boundary_xi_ratio={float(meta.get('boundary_xi_ratio', float('nan'))):.3f} "
            f"paired_reps={int(summary_obj.get('common_paired_reps', 0))}"
        )
        if ranked_rows:
            best = ranked_rows[0]
            lead += f" best_mse={best['method']}({float(best['mse_overall']):.5f})"
        log.info(lead)

        if ranked_rows:
            detail = " | ".join(
                f"{r['method']} mse={float(r['mse_overall']):.5f}, lpd={float(r.get('lpd_test', float('nan'))):.5f}, paired={int(r.get('n_reps_paired', 0))}"
                for r in ranked_rows
            )
            log.info("Exp3 analysis detail: %s", detail)
        else:
            fallback = " | ".join(
                f"{r['method']} ok={int(r.get('n_reps_ok', 0))}/{int(r.get('n_reps_total', 0))}, conv={int(r.get('n_reps_converged', 0))}"
                for r in method_rows
            )
            log.info("Exp3 analysis detail: no paired-converged summary yet; %s", fallback or "no method rows")

        gr_row = next((r for r in method_rows if r["method"] == "GR_RHS"), None)
        if gr_row is not None and np.isfinite(float(gr_row.get("kappa_null_mean", float("nan")))) and np.isfinite(float(gr_row.get("kappa_signal_mean", float("nan")))):
            log.info(
                "Exp3 analysis GR_RHS kappa: null_mean=%.4f signal_mean=%.4f prob_gt_u0(null)=%.4f prob_gt_u0(signal)=%.4f",
                float(gr_row.get("kappa_null_mean", float("nan"))),
                float(gr_row.get("kappa_signal_mean", float("nan"))),
                float(gr_row.get("kappa_null_prob_gt_u0", float("nan"))),
                float(gr_row.get("kappa_signal_prob_gt_u0", float("nan"))),
            )

        comparison = dict(summary_obj.get("comparison", {}))
        mse_diff = float(comparison.get("gr_rhs_vs_rhs_mse_diff", float("nan")))
        lpd_diff = float(comparison.get("gr_rhs_vs_rhs_lpd_diff", float("nan")))
        if np.isfinite(mse_diff) or np.isfinite(lpd_diff):
            log.info(
                "Exp3 analysis GR_RHS vs RHS: delta_mse=%.5f delta_lpd=%.5f",
                mse_diff,
                lpd_diff,
            )

        with setting_progress_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary_obj, ensure_ascii=False) + "\n")

    def _log_setting_progress(task: dict[str, Any], _result: Any) -> None:
        sid_local = int(task["setting_id"])
        setting_rows_by_id[sid_local].extend(list(_result or []))
        methods_local = [str(m) for m in task.get("methods", [])]
        if methods_local and all(_is_bayesian_method(m) for m in methods_local):
            bayes_tasks_done_by_setting[sid_local] += 1
            bayes_expected = int(bayes_tasks_expected_by_setting.get(sid_local, 0))
            bayes_done = int(bayes_tasks_done_by_setting[sid_local])
            if bayes_expected > 0 and bayes_done >= bayes_expected and sid_local not in bayes_settings_completed:
                bayes_settings_completed.add(sid_local)
                meta = settings_meta.get(sid_local, {})
                log.info(
                    "Exp3 progress: bayes-complete settings %d/%d (setting_id=%d, signal=%s, group=%s, env=%s, boundary_xi_ratio=%.3f)",
                    len(bayes_settings_completed),
                    len(settings),
                    sid_local,
                    str(meta.get("signal", "?")),
                    str(meta.get("group_config", "?")),
                    str(meta.get("env_id", "?")),
                    float(meta.get("boundary_xi_ratio", float("nan"))),
                )
        tasks_done_by_setting[sid_local] += 1
        expected = int(tasks_expected_by_setting.get(sid_local, 0))
        done = int(tasks_done_by_setting[sid_local])
        if expected <= 0 or done < expected or sid_local in settings_completed:
            return
        settings_completed.add(sid_local)
        meta = settings_meta.get(sid_local, {})
        log.info(
            "Exp3 progress: completed settings %d/%d (setting_id=%d, signal=%s, group=%s, env=%s, boundary_xi_ratio=%.3f)",
            len(settings_completed),
            len(settings),
            sid_local,
            str(meta.get("signal", "?")),
            str(meta.get("group_config", "?")),
            str(meta.get("env_id", "?")),
            float(meta.get("boundary_xi_ratio", float("nan"))),
        )
        _log_completed_setting_analysis(_summarize_completed_setting(sid_local))

    log.info(
        "Exp3[%s]: %d settings x %d repeats = %d data batches; %d methods => %d method-evals; "
        "scheduled tasks: bayes=%d, classical=%d "
        "(group_configs=%s, signals=%s, env_points=%s, boundary_xi_ratio_grid=%s), "
        "methods=%s, bayes_min_chains=%d, enforce=%s, retry_limit=%d, gigg_mode=%s",
        design_mode,
        len(settings), repeats, n_data_batches, len(methods_use), n_method_evals,
        len(bayes_tasks), len(classical_tasks),
        [gc["name"] for gc in gc_list], signals,
        [ep["env_id"] for ep in env_points_used],
        boundary_xi_ratios,
        methods_use, int(bayes_min_chains_use), bool(enforce_bayes_convergence), int(retry_limit), str(gigg_mode_name),
    )

    chunks_bayes: list[Any] = []
    chunks_classic: list[Any] = []
    if bayes_tasks:
        chunks_bayes = _parallel_rows(
            bayes_tasks,
            _exp3_worker,
            n_jobs=n_jobs,
            prefer_process=True,
            process_fallback="serial",
            progress_desc="Exp3 Linear Benchmark (Bayes)",
            on_task_done=_log_setting_progress,
        )
    if classical_tasks:
        chunks_classic = _parallel_rows(
            classical_tasks,
            _exp3_worker,
            n_jobs=n_jobs,
            prefer_process=False,
            progress_desc="Exp3 Linear Benchmark (Classical)",
            on_task_done=_log_setting_progress,
        )

    rows: list[dict] = []
    for chunk in list(chunks_bayes) + list(chunks_classic):
        rows.extend(chunk)

    raw = pd.DataFrame(rows)
    if not raw.empty and "method" in raw.columns:
        raw["method_label"] = raw["method"].map(method_result_label)

    group_keys = base_keys + [
        "method",
    ]

    counts_df = raw.groupby(group_keys, as_index=False).agg(
        n_reps_total=("replicate_id", "count"),
        n_reps_ok=("status", lambda s: int((s == "ok").sum())),
        n_reps_converged=("converged", lambda s: int(s.fillna(False).astype(bool).sum())),
    )

    paired_raw, paired_stats = _paired_converged_subset(
        raw,
        group_cols=base_keys,
        method_col="method",
        replicate_col="replicate_id",
        converged_col="converged",
        required_cols=["mse_null", "mse_signal", "mse_overall", "lpd_test"],
        method_levels=None if bool(heavy_methods_anchor_only) else methods_use,
    )

    metric_df = paired_raw.groupby(group_keys, as_index=False).agg(
        mse_null=("mse_null", "mean"),
        mse_signal=("mse_signal", "mean"),
        mse_overall=("mse_overall", "mean"),
        lpd_test=("lpd_test", "mean"),
        coverage_95=("coverage_95", "mean"),
        avg_ci_length=("avg_ci_length", "mean"),
        kappa_null_mean=("kappa_null_mean", "mean"),
        kappa_signal_mean=("kappa_signal_mean", "mean"),
        kappa_null_prob_gt_u0=("kappa_null_prob_gt_u0", "mean"),
        kappa_signal_prob_gt_u0=("kappa_signal_prob_gt_u0", "mean"),
        bridge_ratio_mean=("bridge_ratio_mean", "mean"),
        bridge_ratio_min=("bridge_ratio_min", "mean"),
        bridge_ratio_max=("bridge_ratio_max", "mean"),
        bridge_ratio_p95=("bridge_ratio_p95", "mean"),
        bridge_ratio_violations=("bridge_ratio_violations", "mean"),
        bridge_ratio_null_mean=("bridge_ratio_null_mean", "mean"),
        bridge_ratio_signal_mean=("bridge_ratio_signal_mean", "mean"),
    )
    if "lpd_test_ppd" in paired_raw.columns:
        metric_df = metric_df.merge(
            paired_raw.groupby(group_keys, as_index=False).agg(lpd_test_ppd=("lpd_test_ppd", "mean")),
            on=group_keys,
            how="left",
        )
    if "lpd_test_plugin" in paired_raw.columns:
        metric_df = metric_df.merge(
            paired_raw.groupby(group_keys, as_index=False).agg(lpd_test_plugin=("lpd_test_plugin", "mean")),
            on=group_keys,
            how="left",
        )

    agg_df = counts_df.merge(metric_df, on=group_keys, how="left")
    pair_counts = paired_raw.groupby(group_keys, as_index=False).agg(n_reps_paired=("replicate_id", "nunique"))
    agg_df = agg_df.merge(pair_counts, on=group_keys, how="left")
    agg_df["n_reps"] = agg_df["n_reps_paired"].fillna(0).astype(int)
    agg_df["method_label"] = agg_df["method"].map(method_result_label)

    delta_rows: list[dict[str, Any]] = []
    baseline_method = "RHS"
    metrics_for_delta = ["mse_null", "mse_signal", "mse_overall", "lpd_test"]
    for setting_vals, sub in paired_raw.groupby(base_keys, dropna=False):
        wide_setting = sub.pivot_table(index="replicate_id", columns="method", values=metrics_for_delta, aggfunc="mean")
        if baseline_method not in wide_setting.columns.get_level_values(1):
            continue
        for metric in metrics_for_delta:
            if metric not in wide_setting.columns.get_level_values(0):
                continue
            wide = wide_setting[metric]
            if baseline_method not in wide.columns:
                continue
            base_vec = wide[baseline_method]
            for m in [c for c in wide.columns if str(c) != baseline_method]:
                diff = (wide[m] - base_vec).dropna()
                n_eff = int(diff.shape[0])
                if n_eff == 0:
                    continue
                mean_v = float(diff.mean())
                sd_v = float(diff.std(ddof=1)) if n_eff > 1 else float("nan")
                se_v = float(sd_v / np.sqrt(n_eff)) if n_eff > 1 else float("nan")
                ci_lo = float(mean_v - 1.96 * se_v) if np.isfinite(se_v) else float("nan")
                ci_hi = float(mean_v + 1.96 * se_v) if np.isfinite(se_v) else float("nan")
                row = {
                    "method": str(m),
                    "baseline_method": baseline_method,
                    "method_label": method_result_label(str(m)),
                    "baseline_method_label": method_result_label(baseline_method),
                    "metric": metric,
                    "mean_diff": mean_v,
                    "std_diff": sd_v,
                    "se_diff": se_v,
                    "ci95_lo": ci_lo,
                    "ci95_hi": ci_hi,
                    "n_effective_pairs": n_eff,
                }
                for k, v in zip(base_keys, setting_vals if isinstance(setting_vals, tuple) else (setting_vals,)):
                    row[k] = v
                delta_rows.append(row)
    delta_df = pd.DataFrame(delta_rows)

    save_dataframe(raw, out_dir / "raw_results.csv")
    _record_produced_paths(produced, out_dir / "raw_results.csv")
    save_dataframe(agg_df, out_dir / "summary.csv")
    _record_produced_paths(produced, out_dir / "summary.csv")
    save_dataframe(agg_df, out_dir / "summary_paired.csv")
    _record_produced_paths(produced, out_dir / "summary_paired.csv")
    save_dataframe(delta_df, out_dir / "summary_paired_deltas.csv")
    _record_produced_paths(produced, out_dir / "summary_paired_deltas.csv")
    table_df = metric_df.merge(counts_df[group_keys + ["n_reps_converged"]], on=group_keys, how="left")
    table_df = table_df.rename(columns={"n_reps_converged": "n_reps"})
    save_dataframe(table_df, tab_dir / "table_linear_benchmark.csv")
    _record_produced_paths(produced, tab_dir / "table_linear_benchmark.csv")
    save_json({
        "exp3_design": design_mode,
        "gigg_mode": str(gigg_mode_name),
        "group_configs": [{"name": gc["name"], "group_sizes": gc["group_sizes"], "active_groups": gc["active_groups"]} for gc in gc_list],
        "signals": signals,
        "env_points": env_points_used,
        "boundary_calibration": {
            "u0": float(_BOUNDARY_U0),
            "xi_ratio_default": float(_BOUNDARY_XI_RATIO),
            "xi_ratio_grid": [float(v) for v in boundary_xi_ratios],
            "sigma2_boundary": float(_SIGMA2_BOUNDARY),
            "rho_profile_formula": "rho_within / sqrt(sigma2_boundary)",
        },
        "n_train": int(n_train),
        "n_test": int(n_test),
        "methods": methods_use,
        "bayes_min_chains": int(bayes_min_chains_use),
        "method_jobs": int(method_jobs),
        "heavy_methods_anchor_only": bool(heavy_methods_anchor_only),
        "n_settings": len(settings),
        "repeats": int(repeats),
        "enforce_bayes_convergence": bool(enforce_bayes_convergence),
        "max_convergence_retries": int(retry_limit),
        "until_bayes_converged": bool(until_bayes_converged),
        "paired_stats": paired_stats.to_dict(orient="records"),
        "pairing_note": "summary.csv uses paired-converged subset",
    }, out_dir / "exp3_meta.json")
    _record_produced_paths(produced, out_dir / "exp3_meta.json")

    try:
        from .analysis.plotting import plot_exp3_benchmark, plot_exp3_boundary_phase_transition
        if not table_df.empty:
            plot_exp3_benchmark(table_df, out_dir=fig_dir)
            _record_produced_paths(produced, fig_dir / "fig3a_mse_by_signal.png", fig_dir / "fig3b_lpd_by_signal.png", fig_dir / "fig3c_null_signal_scatter.png")
            if "boundary_xi_ratio" in paired_raw.columns:
                plot_exp3_boundary_phase_transition(
                    paired_raw,
                    out_path=fig_dir / "fig3d_boundary_phase_transition.png",
                    u0=float(_BOUNDARY_U0),
                )
                _record_produced_paths(produced, fig_dir / "fig3d_boundary_phase_transition.png")
        else:
            log.warning("Plot exp3 skipped: no converged rows available.")
    except Exception as exc:
        log.warning("Plot exp3 failed: %s", exc)

    log.info(
        "Exp3 done: %d rows, %d settings, %d paired rows used for metrics",
        len(rows),
        len(settings),
        int(paired_raw.shape[0]),
    )
    result_paths = {
        "raw": str(out_dir / "raw_results.csv"),
        "summary": str(out_dir / "summary.csv"),
        "summary_paired": str(out_dir / "summary_paired.csv"),
        "summary_paired_deltas": str(out_dir / "summary_paired_deltas.csv"),
        "table": str(tab_dir / "table_linear_benchmark.csv"),
        "meta": str(out_dir / "exp3_meta.json"),
        "fig3a_mse_by_signal": str(fig_dir / "fig3a_mse_by_signal.png"),
        "fig3b_lpd_by_signal": str(fig_dir / "fig3b_lpd_by_signal.png"),
        "fig3c_null_signal_scatter": str(fig_dir / "fig3c_null_signal_scatter.png"),
    }
    if (fig_dir / "fig3d_boundary_phase_transition.png").exists():
        result_paths["fig3d_boundary_phase_transition"] = str(fig_dir / "fig3d_boundary_phase_transition.png")
    return _finalize_experiment_run(
        exp_key=str(exp_key_name),
        save_dir=save_dir,
        results_dir=out_dir,
        produced_paths=produced,
        result_paths=result_paths,
        skip_run_analysis=bool(skip_run_analysis),
        archive_artifacts=bool(archive_artifacts),
    )


def run_exp3a_main_benchmark(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 100,
    save_dir: str = "outputs/simulation_project",
    **kwargs,
) -> Dict[str, str]:
    """Exp3a: paper-aligned fixed-coefficient benchmark (GIGG Section 5.1)."""
    return run_exp3_linear_benchmark(
        n_jobs=n_jobs,
        seed=seed,
        repeats=repeats,
        save_dir=save_dir,
        signal_types=["concentrated", "distributed"],
        group_configs=_paper_fixed_exp3_group_configs(),
        env_points=_paper_fixed_exp3_env_points(),
        result_dir_name="exp3a_main_benchmark",
        exp_key="exp3a",
        **kwargs,
    )


def run_exp3b_boundary_stress(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 20,
    save_dir: str = "outputs/simulation_project",
    boundary_xi_ratio_list: Sequence[float] | None = None,
    **kwargs,
) -> Dict[str, str]:
    """Exp3b: boundary-only stress benchmark."""
    boundary_grid = list(boundary_xi_ratio_list or [0.5, 0.8, 1.0, 1.1, 1.2, 1.5, 2.0])
    return run_exp3_linear_benchmark(
        n_jobs=n_jobs,
        seed=seed,
        repeats=repeats,
        save_dir=save_dir,
        signal_types=["boundary"],
        boundary_xi_ratio_list=boundary_grid,
        result_dir_name="exp3b_boundary_stress",
        exp_key="exp3b",
        **kwargs,
    )


def run_exp3c_highdim_stress(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 30,
    save_dir: str = "outputs/simulation_project",
    **kwargs,
) -> Dict[str, str]:
    """Exp3c: high-dimensional stress test with paper-style random coefficients."""
    return run_exp3_linear_benchmark(
        n_jobs=n_jobs,
        seed=seed,
        repeats=repeats,
        save_dir=save_dir,
        signal_types=["random_coefficient"],
        group_configs=_paper_random_exp3_group_configs(total_groups=50),
        env_points=_paper_random_exp3_env_points(),
        n_train=200,
        n_test=100,
        result_dir_name="exp3c_highdim_stress",
        exp_key="exp3c",
        **kwargs,
    )


def run_exp3d_within_group_mixed(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 20,
    save_dir: str = "outputs/simulation_project",
    **kwargs,
) -> Dict[str, str]:
    """Exp3d: legacy boundary-focused stress benchmark under the generic Exp3 protocol.

    The public name/path keeps the historical `within_group_mixed` label for
    backward compatibility, but the active design is boundary-only.
    """
    return run_exp3_linear_benchmark(
        n_jobs=n_jobs,
        seed=seed,
        repeats=repeats,
        save_dir=save_dir,
        signal_types=["boundary"],
        result_dir_name="exp3d_within_group_mixed",
        exp_key="exp3d",
        **kwargs,
    )




