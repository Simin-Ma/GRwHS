from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .evaluation import _evaluate_row, _kappa_group_means
from .fitting import _fit_all_methods
from .reporting import _finalize_experiment_run, _record_produced_paths
from .runtime import (
    _BAYESIAN_DEFAULT_CHAINS,
    _attempts_used,
    _exp3_gigg_config_for_mode,
    _gigg_config_for_profile,
    _is_bayesian_method,
    _normalize_compute_profile,
    _normalize_exp3_gigg_mode,
    _parallel_rows,
    _resolve_convergence_retry_limit,
    _result_diag_fields,
    _sampler_for_profile,
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
# Factors (default core30 design):
#   signal_structure: concentrated / distributed / boundary
#     concentrated: 2/5 groups fully active, beta_j = 1/sqrt(p_g) (dense in group)
#     distributed:  2/5 groups with one active variable each, beta_j = 1
#     boundary:     2/5 groups active, beta calibrated at xi_ratio * xi_crit(u0, rho_profile),
#                   where rho_profile = rho_within / sqrt(sigma2_boundary)
#   env points:
#     E0        : (rw=0.3, rb=0.1, snr=1.0) [all signals]
#     RW_PLUS   : (rw=0.8, rb=0.1, snr=1.0) [all signals]
#
# Prediction:
#   concentrated + moderate/high rho: GR-RHS wins on null_group_mse
#   distributed: RHS matches GR-RHS (individual-level shrinkage is sufficient)
#   boundary: GR-RHS separates null/signal groups; competitors may fail
# ---------------------------------------------------------------------------

_BOUNDARY_U0 = 0.5
_BOUNDARY_XI_RATIO = 1.2
_SIGMA2_BOUNDARY = 1.0

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

_DEFAULT_EXP3_ENV_POINTS_CORE30: list[dict[str, Any]] = [
    {
        "env_id": "E0",
        "setting_block": "anchor",
        "rho_within": 0.3,
        "rho_between": 0.1,
        "target_snr": 1.0,
        "signals": ["concentrated", "distributed", "boundary"],
    },
    {
        "env_id": "RW_PLUS",
        "setting_block": "rw_axis",
        "rho_within": 0.8,
        "rho_between": 0.1,
        "target_snr": 1.0,
        "signals": ["concentrated", "distributed", "boundary"],
    },
]


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
) -> np.ndarray:
    """Construct beta for each benchmark signal structure.

    concentrated: all variables in active groups with equal weight (||beta_g||=1 per group)
    distributed:  first variable only in each active group (beta_j=1)
    boundary:     all vars in active groups, calibrated at xi_ratio * xi_crit(u0, rho_profile)
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
        r = int(task["replicate_id"])
        seed_base = int(task["seed_base"])
        n_test = int(task["n_test"])
        sampler = task["sampler"]
        method_name = str(task["method"])
        gigg_config = dict(task["gigg_config"])
        gigg_mode = str(task.get("gigg_mode", "stable"))
        bayes_min_chains = task.get("bayes_min_chains")
        enforce_conv = bool(task["enforce_bayes_convergence"])
        max_retries = int(task["max_convergence_retries"])
        grrhs_kwargs = dict(task["grrhs_kwargs"])
        methods = [method_name]
    else:
        sid, signal, group_cfg, setting_block, env_id, design_type, rho_within, rho_between, target_snr, r, seed_base, n_test, sampler, methods, gigg_config, bayes_min_chains, enforce_conv, max_retries, grrhs_kwargs = task
        gigg_mode = "stable"
    gigg_config = dict(gigg_config)
    gigg_mode_name = _normalize_exp3_gigg_mode(gigg_mode)
    if str(method_name if isinstance(task, dict) else (methods[0] if methods else "")).upper() == "GIGG_MMLE":
        if gigg_mode_name == "paper_ref":
            gigg_config["extra_retry"] = 0
            gigg_config.pop("retry_cap", None)
        else:
            hard_setting = (group_cfg_name := str(group_cfg["name"])) in {"CL", "G10x5"} and signal in {"concentrated", "distributed"}
            if hard_setting:
                gigg_config["extra_retry"] = max(1, int(gigg_config.get("extra_retry", 0)))
                # Keep rescue behavior efficient while preserving robustness.
                gigg_config["retry_cap"] = 2
                # Full profile defaults to paper-locked no_retry; for hard Exp3 settings
                # we allow one bounded rescue attempt to improve benchmark completeness.
                if bool(gigg_config.get("no_retry", False)):
                    gigg_config["no_retry"] = False
                gigg_config["progress_bar"] = bool(gigg_config.get("progress_bar", False))
                # For difficult Exp3 settings, prefer stronger mixing from the first attempt.
                gigg_config["randomize_group_order"] = bool(gigg_config.get("randomize_group_order", True))
                gigg_config["lambda_vectorized_update"] = bool(gigg_config.get("lambda_vectorized_update", True))
                gigg_config["extra_beta_refresh_prob"] = max(float(gigg_config.get("extra_beta_refresh_prob", 0.0)), 0.08)
                gigg_config["init_scale_blend"] = max(float(gigg_config.get("init_scale_blend", 0.5)), 0.65)
                # Damped MMLE updates reduce q_g oscillation in difficult correlated regimes.
                gigg_config["mmle_step_size"] = min(max(float(gigg_config.get("mmle_step_size", 0.6)), 0.0), 1.0)
    s = experiment_seed(3, int(sid), r, master_seed=int(seed_base))

    group_sizes: list[int] = list(group_cfg["group_sizes"])
    active_groups: list[int] = list(group_cfg["active_groups"])
    group_cfg_name: str = str(group_cfg["name"])
    n_train = 100

    sigma2_boundary = float(_SIGMA2_BOUNDARY)
    boundary_rho_profile = float(rho_within) / math.sqrt(max(sigma2_boundary, 1e-12))
    boundary_xi_crit = float("nan")
    boundary_xi = float("nan")
    if signal == "boundary":
        boundary_xi_crit = xi_crit_u0_rho(u0=float(_BOUNDARY_U0), rho=boundary_rho_profile)
        boundary_xi = float(_BOUNDARY_XI_RATIO) * boundary_xi_crit

    beta0 = _build_benchmark_beta(
        signal,
        group_sizes,
        active_groups=active_groups,
        sigma2=sigma2_boundary if signal == "boundary" else 1.0,
        boundary_u0=float(_BOUNDARY_U0),
        boundary_xi_ratio=float(_BOUNDARY_XI_RATIO),
        boundary_rho_profile=boundary_rho_profile if signal == "boundary" else None,
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

    fits = _fit_all_methods(
        X_train, y_train, groups,
        task="gaussian", seed=s, p0=p0,
        p0_groups=p0_signal_groups,
        sampler=sampler, methods=methods, gigg_config=gigg_config,
        bayes_min_chains=int(bayes_min_chains) if bayes_min_chains is not None else None,
        grrhs_kwargs=grrhs_kwargs or {},
        enforce_bayes_convergence=bool(enforce_conv),
        max_convergence_retries=int(max_retries),
    )

    out_rows: list[dict[str, Any]] = []
    for method, res in fits.items():
        metrics = _evaluate_row(res, beta0, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        kappa_null_mean = float("nan")
        kappa_signal_mean = float("nan")
        if method == "GR_RHS" and res.beta_mean is not None:
            km = _kappa_group_means(res, n_groups)
            null_groups = [g for g in range(n_groups) if g not in set(active_groups)]
            _sig_vals = [km[g] for g in active_groups if not np.isnan(km[g])]
            _null_vals = [km[g] for g in null_groups if not np.isnan(km[g])]
            kappa_signal_mean = float(np.mean(_sig_vals)) if _sig_vals else float("nan")
            kappa_null_mean = float(np.mean(_null_vals)) if _null_vals else float("nan")
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
            "boundary_xi_ratio": float(_BOUNDARY_XI_RATIO) if signal == "boundary" else float("nan"),
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
            **_result_diag_fields(res),
            **metrics,
        })
    return out_rows


def run_exp3_linear_benchmark(
    n_jobs: int = 1,
    seed: int = MASTER_SEED,
    repeats: int = 20,
    save_dir: str = "outputs/simulation_project",
    *,
    signal_types: Sequence[str] | None = None,
    rho_within_values: Sequence[float] | None = None,
    snr_values: Sequence[float] | None = None,
    rho_between: float = 0.1,
    exp3_design: str = "core30",
    env_points: Sequence[dict[str, Any]] | None = None,
    bayes_min_chains: int | None = None,
    group_configs: list[dict[str, Any]] | None = None,
    profile: str = "full",
    methods: Sequence[str] | None = None,
    enforce_bayes_convergence: bool = True,
    max_convergence_retries: int | None = None,
    until_bayes_converged: bool = True,
    n_test: int = 30,
    sampler_backend: str = "nuts",
    grrhs_extra_kwargs: dict | None = None,
    gigg_mode: str = "stable",
    result_dir_name: str = "exp3_linear_benchmark",
    exp_key: str = "exp3",
) -> Dict[str, str]:
    """
    Exp3 benchmark with two design modes:

      core30 (default):
        compact, theory-aligned design with rw>rb constraint and no SNR axis.
        Under current defaults this yields 6 settings per group config
        (2 concentrated + 2 distributed + 2 boundary) = 18 total settings
        across [G10x5, CL, CS].

      legacy_factorial:
        factor design without SNR axis:
          signal_structure x group_config x rho_within x rho_between
        with automatic filtering of combinations that violate rw>rb.
        target_snr is fixed at 1.0 in this mode.

    Signal types (default ["concentrated", "distributed", "boundary"]):
      concentrated: all nonzero beta in G0 and G1, G2/G3/G4 are null.
      distributed: one nonzero beta in each of G0 and G1.
      boundary: signal set to xi_ratio * xi_crit(u0=0.5, rho_profile),
                with rho_profile = rho_within / sqrt(sigma2_boundary).

    bayes_min_chains:
      Minimum number of chains for Bayesian methods in Exp3.
      Default: 2 for laptop profile, 4 for full profile.

    Methods:
      GR_RHS, GHS_plus, GIGG_MMLE, RHS, LASSO_CV, OLS.

    gigg_mode:
      paper_ref: strict baseline mode (no Exp3 hard-setting rescue/stabilization).
      stable:    enhanced mode with bounded rescue/stabilization in hard settings.
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

    profile_name = _normalize_compute_profile(profile)
    sampler = _sampler_for_profile(profile_name)
    bayes_min_chains_use = int(bayes_min_chains) if bayes_min_chains is not None else (2 if profile_name == "laptop" else int(_BAYESIAN_DEFAULT_CHAINS))
    bayes_min_chains_use = max(1, int(bayes_min_chains_use))
    _exp3_methods = ["GR_RHS", "GHS_plus", "GIGG_MMLE", "RHS", "LASSO_CV", "OLS"]
    methods_use = [m for m in (methods or _exp3_methods) if m in set(_exp3_methods)]
    if not methods_use:
        methods_use = list(_exp3_methods)
    gigg_mode_name = _normalize_exp3_gigg_mode(gigg_mode)
    gigg_cfg = _exp3_gigg_config_for_mode(_gigg_config_for_profile(profile_name), gigg_mode=gigg_mode_name)
    retry_limit = _resolve_convergence_retry_limit(
        profile_name,
        max_convergence_retries,
        until_bayes_converged=bool(until_bayes_converged),
    )

    design_mode = str(exp3_design).strip().lower()
    if design_mode not in {"core30", "legacy_factorial"}:
        raise ValueError(f"unknown exp3_design: {exp3_design!r}. Use 'core30' or 'legacy_factorial'.")

    signals = list(signal_types or ["concentrated", "distributed", "boundary"])
    gc_list: list[dict[str, Any]] = list(group_configs) if group_configs is not None else list(_DEFAULT_EXP3_GROUP_CONFIGS)

    # settings:
    #   sid, signal, group_cfg, setting_block, env_id, design_type, rho_within, rho_between, target_snr
    settings: list[tuple[int, str, dict, str, str, str, float, float, float]] = []
    sid = 0
    env_points_used: list[dict[str, Any]] = []

    if design_mode == "core30":
        points_raw = list(env_points) if env_points is not None else list(_DEFAULT_EXP3_ENV_POINTS_CORE30)
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
                    sid += 1
                    rho = float(ep["rho_within"])
                    rhob = float(ep["rho_between"])
                    snr = float(ep["target_snr"])
                    design = "orthonormal" if rho == 0.0 and rhob == 0.0 else "correlated"
                    settings.append((sid, signal, gc, str(ep["setting_block"]), str(ep["env_id"]), design, rho, rhob, snr))
    else:
        rho_values = list(rho_within_values if rho_within_values is not None else [0.3, 0.8])
        if snr_values is not None:
            log.warning("Exp3 legacy_factorial ignores snr_values (SNR axis removed); using fixed target_snr=1.0.")
        snr_list = [1.0]
        rhob = float(rho_between)
        rho_values_valid = [float(rho) for rho in rho_values if float(rho) > rhob]
        rho_values_dropped = [float(rho) for rho in rho_values if float(rho) <= rhob]
        if rho_values_dropped:
            log.warning(
                "Exp3 legacy_factorial: dropped rho_within values that violate rw>rb (rb=%.3f): %s",
                rhob,
                rho_values_dropped,
            )
        if not rho_values_valid:
            raise ValueError(f"No valid rho_within values remain after enforcing rw>rb with rho_between={rhob}.")
        env_points_used = [
            {
                "env_id": f"LEGACY_RW{float(rho):.1f}_SNR{float(snr):.1f}",
                "setting_block": "legacy_factorial",
                "rho_within": float(rho),
                "rho_between": rhob,
                "target_snr": float(snr),
                "signals": list(signals),
            }
            for rho in rho_values_valid
            for snr in snr_list
        ]
        for gc in gc_list:
            for signal in signals:
                for rho in rho_values_valid:
                    for snr in snr_list:
                        sid += 1
                        design = "orthonormal" if float(rho) == 0.0 and rhob == 0.0 else "correlated"
                        settings.append(
                            (
                                sid,
                                signal,
                                gc,
                                "legacy_factorial",
                                f"LEGACY_RW{float(rho):.1f}_SNR{float(snr):.1f}",
                                design,
                                float(rho),
                                rhob,
                                float(snr),
                            )
                        )

    grrhs_kw: dict = {"backend": str(sampler_backend), "tau_target": "groups"}
    if grrhs_extra_kwargs:
        grrhs_kw.update(grrhs_extra_kwargs)
    tasks: list[dict[str, Any]] = []
    for (sid_v, signal_v, gc_v, block_v, env_v, dt_v, rho_v, rhob_v, snr_v) in settings:
        for r in range(1, int(repeats) + 1):
            for method in methods_use:
                tasks.append(
                    {
                        "setting_id": int(sid_v),
                        "signal": str(signal_v),
                        "group_cfg": dict(gc_v),
                        "setting_block": str(block_v),
                        "env_id": str(env_v),
                        "design_type": str(dt_v),
                        "rho_within": float(rho_v),
                        "rho_between": float(rhob_v),
                        "target_snr": float(snr_v),
                        "replicate_id": int(r),
                        "seed_base": int(seed),
                        "n_test": int(n_test),
                        "sampler": sampler,
                        "method": str(method),
                        "gigg_config": dict(gigg_cfg),
                        "gigg_mode": str(gigg_mode_name),
                        "bayes_min_chains": int(bayes_min_chains_use),
                        "enforce_bayes_convergence": bool(enforce_bayes_convergence),
                        "max_convergence_retries": int(retry_limit),
                        "grrhs_kwargs": dict(grrhs_kw),
                    }
                )

    log.info(
        "Exp3[%s]: %d settings x %d repeats x %d methods = %d tasks "
        "(group_configs=%s, signals=%s, env_points=%s), methods=%s, bayes_min_chains=%d, enforce=%s, retry_limit=%d, gigg_mode=%s",
        design_mode,
        len(settings), repeats, len(methods_use), len(tasks),
        [gc["name"] for gc in gc_list], signals,
        [ep["env_id"] for ep in env_points_used],
        methods_use, int(bayes_min_chains_use), bool(enforce_bayes_convergence), int(retry_limit), str(gigg_mode_name),
    )
    bayes_tasks: list[dict[str, Any]] = []
    classical_tasks: list[dict[str, Any]] = []
    for t in tasks:
        method_name = str(t.get("method", ""))
        if _is_bayesian_method(method_name):
            bayes_tasks.append(t)
        else:
            classical_tasks.append(t)

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

    ok_raw = raw.loc[raw["status"] == "ok"].copy()
    conv_raw = ok_raw.loc[ok_raw["converged"].fillna(False).astype(bool)].copy()
    group_keys = ["gigg_mode", "group_config", "signal", "setting_block", "env_id", "design_type", "rho_within", "rho_between", "target_snr", "method"]

    counts_df = raw.groupby(group_keys, as_index=False).agg(
        n_reps_total=("replicate_id", "count"),
        n_reps_ok=("status", lambda s: int((s == "ok").sum())),
        n_reps_converged=("converged", lambda s: int(s.fillna(False).astype(bool).sum())),
    )

    metric_df = conv_raw.groupby(group_keys, as_index=False).agg(
        mse_null=("mse_null", "mean"),
        mse_signal=("mse_signal", "mean"),
        mse_overall=("mse_overall", "mean"),
        lpd_test=("lpd_test", "mean"),
        coverage_95=("coverage_95", "mean"),
        avg_ci_length=("avg_ci_length", "mean"),
        kappa_null_mean=("kappa_null_mean", "mean"),
        kappa_signal_mean=("kappa_signal_mean", "mean"),
    )

    agg_df = counts_df.merge(metric_df, on=group_keys, how="left")
    agg_df["n_reps"] = agg_df["n_reps_converged"]

    save_dataframe(raw, out_dir / "raw_results.csv")
    _record_produced_paths(produced, out_dir / "raw_results.csv")
    save_dataframe(agg_df, out_dir / "summary.csv")
    _record_produced_paths(produced, out_dir / "summary.csv")
    table_df = metric_df.merge(counts_df[group_keys + ["n_reps_converged"]], on=group_keys, how="left")
    table_df = table_df.rename(columns={"n_reps_converged": "n_reps"})
    save_dataframe(table_df, tab_dir / "table_linear_benchmark.csv")
    _record_produced_paths(produced, tab_dir / "table_linear_benchmark.csv")
    save_json({
        "exp3_design": design_mode,
        "profile": profile_name,
        "gigg_mode": str(gigg_mode_name),
        "group_configs": [{"name": gc["name"], "group_sizes": gc["group_sizes"], "active_groups": gc["active_groups"]} for gc in gc_list],
        "signals": signals,
        "env_points": env_points_used,
        "boundary_calibration": {
            "u0": float(_BOUNDARY_U0),
            "xi_ratio": float(_BOUNDARY_XI_RATIO),
            "sigma2_boundary": float(_SIGMA2_BOUNDARY),
            "rho_profile_formula": "rho_within / sqrt(sigma2_boundary)",
        },
        "n_train": 100,
        "n_test": int(n_test),
        "methods": methods_use,
        "bayes_min_chains": int(bayes_min_chains_use),
        "n_settings": len(settings),
        "repeats": int(repeats),
        "enforce_bayes_convergence": bool(enforce_bayes_convergence),
        "max_convergence_retries": int(retry_limit),
        "until_bayes_converged": bool(until_bayes_converged),
    }, out_dir / "exp3_meta.json")
    _record_produced_paths(produced, out_dir / "exp3_meta.json")

    try:
        from .analysis.plotting import plot_exp3_benchmark
        if not table_df.empty:
            plot_exp3_benchmark(table_df, out_dir=fig_dir)
            _record_produced_paths(produced, fig_dir / "fig3a_mse_by_signal.png", fig_dir / "fig3b_lpd_by_signal.png", fig_dir / "fig3c_null_signal_scatter.png")
        else:
            log.warning("Plot exp3 skipped: no converged rows available.")
    except Exception as exc:
        log.warning("Plot exp3 failed: %s", exc)

    log.info(
        "Exp3 done: %d rows, %d settings, %d converged rows used for metrics",
        len(rows),
        len(settings),
        int(conv_raw.shape[0]),
    )
    result_paths = {
        "raw": str(out_dir / "raw_results.csv"),
        "summary": str(out_dir / "summary.csv"),
        "table": str(tab_dir / "table_linear_benchmark.csv"),
        "meta": str(out_dir / "exp3_meta.json"),
        "fig3a_mse_by_signal": str(fig_dir / "fig3a_mse_by_signal.png"),
        "fig3b_lpd_by_signal": str(fig_dir / "fig3b_lpd_by_signal.png"),
        "fig3c_null_signal_scatter": str(fig_dir / "fig3c_null_signal_scatter.png"),
    }
    return _finalize_experiment_run(
        exp_key=str(exp_key_name),
        save_dir=save_dir,
        results_dir=out_dir,
        produced_paths=produced,
        result_paths=result_paths,
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
    **kwargs,
) -> Dict[str, str]:
    """Exp3b: boundary-only stress benchmark."""
    return run_exp3_linear_benchmark(
        n_jobs=n_jobs,
        seed=seed,
        repeats=repeats,
        save_dir=save_dir,
        signal_types=["boundary"],
        result_dir_name="exp3b_boundary_stress",
        exp_key="exp3b",
        **kwargs,
    )




