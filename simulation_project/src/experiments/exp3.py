from __future__ import annotations

import math
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
    ensure_dir,
    experiment_seed,
    load_pandas,
    save_dataframe,
    save_json,
    setup_logger,
)

# ---------------------------------------------------------------------------
# EXP3 - Linear Benchmark: Concentrated vs. Distributed vs. Boundary
# ---------------------------------------------------------------------------
# Factor design directly testing the core GR-RHS hypothesis:
#   "kappa_g mechanism is most beneficial when signals are group-concentrated"
#
# Factors (default single design):
#   signal_structure: concentrated / distributed / boundary / within_group_mixed / half_dense / dense
#     concentrated: 2/5 groups fully active, beta_j = 1/sqrt(p_g) (dense in group)
#     distributed:  2/5 groups with one active variable each, beta_j = 1
#     boundary:     2/5 groups active, beta calibrated at xi_ratio * xi_crit(u0, rho_profile),
#                   where rho_profile = rho_within / sqrt(sigma2_boundary)
#     within_group_mixed: each active group has one strong coefficient and
#                   the remaining coefficients weak but nonzero
#     half_dense:   random active coefficients at 20% density
#     dense:        random active coefficients at 60% density
#   env points:
#     rw in {0.3, 0.6, 0.8}, rb=0.1, snr in {0.2, 1.0, 5.0}
#
# Prediction:
#   concentrated + moderate/high rho: GR-RHS wins on null_group_mse
#   distributed: RHS matches GR-RHS (individual-level shrinkage is sufficient)
#   boundary: GR-RHS separates null/signal groups; competitors may fail
# ---------------------------------------------------------------------------

_BOUNDARY_U0 = 0.5
_BOUNDARY_XI_RATIO = 1.2
_SIGMA2_BOUNDARY = 1.0
_WITHIN_GROUP_MIXED_STRONG = 1.0
_WITHIN_GROUP_MIXED_WEAK = 0.25
_EXP3_HEAVY_METHODS = {"GIGG_MMLE", "GHS_plus"}

# ---------------------------------------------------------------------------
# Default group configurations for Exp3 - mirrors GIGG paper Table 1 coverage,
# plus GR-RHS-favorable scenarios (large null blocks, rho_between > 0).
#
# Each entry:
#   name          - short label used in output CSV / meta JSON
#   group_sizes   - list of per-group sizes (sum = p)
#   active_groups - group indices containing signal (rest are null)
#
# G10x5 : 5 equal groups of size 10 (p=50)  - GIGG paper C10H/D10H baseline
# CL    : [30,10,5,3,2], signal in large groups - GIGG Table 3 CL/DL
# CS    : [30,10,5,3,2], signal in small groups - GR-RHS kappa_g advantage
#         (large null blocks enable strong collective contraction)
# ---------------------------------------------------------------------------
_DEFAULT_EXP3_GROUP_CONFIGS: list[dict[str, Any]] = [
    {"name": "G10x5", "group_sizes": [10, 10, 10, 10, 10],  "active_groups": [0, 1]},
    {"name": "CL",    "group_sizes": [30, 10, 5, 3, 2],     "active_groups": [0, 1]},
    {"name": "CS",    "group_sizes": [30, 10, 5, 3, 2],     "active_groups": [3, 4]},
]

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
    if group_name != "G10x5":
        return False
    if env_name != "RW08_SNR10":
        return False
    if sig == "boundary":
        return abs(float(boundary_xi_ratio) - float(_BOUNDARY_XI_RATIO)) < 1e-9
    return sig in {"concentrated", "distributed"}


def _build_benchmark_beta(
    signal: str,
    group_sizes: Sequence[int],
    *,
    active_groups: Sequence[int] | None = None,
    sigma2: float = 1.0,
    p: int | None = None,
    boundary_u0: float = _BOUNDARY_U0,
    boundary_xi_ratio: float = _BOUNDARY_XI_RATIO,
    boundary_rho_profile: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Construct beta for each benchmark signal structure.

    concentrated: all variables in active groups with equal weight (||beta_g||=1 per group)
    distributed:  first variable only in each active group (beta_j=1)
    boundary:     all vars in active groups, calibrated at xi_ratio * xi_crit(u0, rho_profile)
    within_group_mixed: one strong + (p_g-1) weak coefficients per active group
    half_dense:   random coefficients over all p with 20% active density
    dense:        random coefficients over all p with 60% active density
    """
    from ..utils import canonical_groups
    groups = canonical_groups(group_sizes)
    total_p = int(p or sum(group_sizes))
    beta = np.zeros(total_p, dtype=float)
    _active = list(active_groups) if active_groups is not None else [0, 1]

    if signal == "concentrated":
        for gid in _active:
            idx = np.asarray(groups[gid], dtype=int)
            beta[idx] = 1.0 / math.sqrt(len(idx))
    elif signal == "distributed":
        for gid in _active:
            beta[groups[gid][0]] = 1.0
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
        ghs_plus_profile = str(task.get("ghs_plus_profile", "default"))
        bayes_min_chains = task.get("bayes_min_chains")
        method_jobs = int(task.get("method_jobs", 1))
        enforce_conv = bool(task["enforce_bayes_convergence"])
        max_retries = int(task["max_convergence_retries"])
        grrhs_kwargs = dict(task["grrhs_kwargs"])
    else:
        if len(task) == 19:
            sid, signal, group_cfg, setting_block, env_id, design_type, rho_within, rho_between, target_snr, r, seed_base, n_test, sampler, methods, gigg_config, bayes_min_chains, enforce_conv, max_retries, grrhs_kwargs = task
            boundary_xi_ratio = float(_BOUNDARY_XI_RATIO)
            n_train = 100
            method_jobs = 1
            ghs_plus_profile = "default"
        elif len(task) == 20:
            sid, signal, group_cfg, setting_block, env_id, design_type, rho_within, rho_between, target_snr, boundary_xi_ratio, r, seed_base, n_test, sampler, methods, gigg_config, bayes_min_chains, enforce_conv, max_retries, grrhs_kwargs = task
            n_train = 100
            method_jobs = 1
            ghs_plus_profile = "default"
        else:
            sid, signal, group_cfg, setting_block, env_id, design_type, rho_within, rho_between, target_snr, boundary_xi_ratio, r, seed_base, n_train, n_test, sampler, methods, gigg_config, bayes_min_chains, method_jobs, ghs_plus_profile, enforce_conv, max_retries, grrhs_kwargs = task
        methods = [str(m) for m in methods]
        gigg_mode = "paper_ref"
    group_cfg_name: str = str(group_cfg["name"])
    methods_upper = {m.upper() for m in methods}
    gigg_config = dict(gigg_config)
    gigg_mode_name = _normalize_exp3_gigg_mode(gigg_mode)
    if "GIGG_MMLE" in methods_upper:
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
        active_groups=active_groups,
        sigma2=sigma2_boundary if signal == "boundary" else 1.0,
        boundary_u0=float(_BOUNDARY_U0),
        boundary_xi_ratio=float(boundary_xi_ratio),
        boundary_rho_profile=boundary_rho_profile if signal == "boundary" else None,
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
    if signal in {"half_dense", "dense"}:
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
        ghs_plus_profile=str(ghs_plus_profile),
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
            **_result_diag_fields(res),
            **bridge_diag,
            **metrics,
        })
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
    gigg_mode: str = "paper_ref",
    heavy_methods_anchor_only: bool = False,
    gigg_budget_profile: str = "default",
    ghs_plus_budget_profile: str = "default",
    result_dir_name: str = "exp3_linear_benchmark",
    exp_key: str = "exp3",
) -> Dict[str, str]:
    """
    Exp3 benchmark (single-default design; no laptop/full split).

    Signal types (default ["concentrated", "distributed", "boundary"]):
      concentrated: all nonzero beta in G0 and G1, G2/G3/G4 are null.
      distributed: one nonzero beta in each of G0 and G1.
      boundary: signal set to xi_ratio * xi_crit(u0=0.5, rho_profile),
                with rho_profile = rho_within / sqrt(sigma2_boundary).
                xi_ratio values come from boundary_xi_ratio_list (default [1.2]).

    bayes_min_chains:
      Minimum number of chains for Bayesian methods in Exp3.
      Default: 4.

    Methods:
      GR_RHS, GHS_plus, GIGG_MMLE, RHS, LASSO_CV, OLS.

    gigg_mode:
      paper_ref: strict gigg-master-aligned reference mode.
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

    sampler = _sampler_for_standard()
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
    gigg_cfg = _exp3_gigg_config_for_mode(_gigg_config_default(profile=str(gigg_budget_profile)), gigg_mode=gigg_mode_name)
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
        for ep in env_points_used:
            sig_set = set(ep.get("signals", signals))
            for signal in signals:
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
                "ghs_plus_profile": str(ghs_plus_budget_profile),
                "bayes_min_chains": int(bayes_min_chains_use),
                "method_jobs": int(method_jobs),
                "enforce_bayes_convergence": bool(enforce_bayes_convergence),
                "max_convergence_retries": int(retry_limit),
                "grrhs_kwargs": dict(grrhs_kw),
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
        )
    if classical_tasks:
        chunks_classic = _parallel_rows(
            classical_tasks,
            _exp3_worker,
            n_jobs=n_jobs,
            prefer_process=False,
            progress_desc="Exp3 Linear Benchmark (Classical)",
        )

    rows: list[dict] = []
    for chunk in list(chunks_bayes) + list(chunks_classic):
        rows.extend(chunk)

    raw = pd.DataFrame(rows)

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
        "gigg_budget_profile": str(gigg_budget_profile),
        "ghs_plus_budget_profile": str(ghs_plus_budget_profile),
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
    repeats: int = 20,
    save_dir: str = "outputs/simulation_project",
    **kwargs,
) -> Dict[str, str]:
    """Exp3a: main benchmark (concentrated + distributed only)."""
    return run_exp3_linear_benchmark(
        n_jobs=n_jobs,
        seed=seed,
        repeats=repeats,
        save_dir=save_dir,
        signal_types=["concentrated", "distributed"],
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
    """Exp3c: high-dimensional stress test (n=200, p=500) with sparse signals only."""
    group_configs = [
        {"name": "HD10x50", "group_sizes": [50] * 10, "active_groups": [0, 1]},
    ]
    env_points = []
    for rw in [0.8]:
        for snr in [0.2, 1.0, 5.0]:
            env_points.append(
                {
                    "env_id": f"HD_RW{int(round(rw*10)):02d}_SNR{int(round(snr*10)):02d}",
                    "setting_block": "highdim_axis",
                    "rho_within": float(rw),
                    "rho_between": 0.2,
                    "target_snr": float(snr),
                    "signals": ["concentrated", "distributed"],
                }
            )
    return run_exp3_linear_benchmark(
        n_jobs=n_jobs,
        seed=seed,
        repeats=repeats,
        save_dir=save_dir,
        signal_types=["concentrated", "distributed"],
        group_configs=group_configs,
        env_points=env_points,
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
    """Exp3d: boundary-focused stress benchmark under simplified Exp3 protocol."""
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




