from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

MASTER_SEED = 20260415


@dataclass(frozen=True)
class SamplerConfig:
    chains: int = 2
    warmup: int = 500
    post_warmup_draws: int = 500
    adapt_delta: float = 0.95
    max_treedepth: int = 12
    strict_adapt_delta: float = 0.99
    strict_max_treedepth: int = 14
    max_divergence_ratio: float = 0.005
    rhat_threshold: float = 1.01
    ess_threshold: float = 400.0


@dataclass
class FitResult:
    method: str
    status: str
    beta_mean: Optional[np.ndarray]
    beta_draws: Optional[np.ndarray]
    kappa_draws: Optional[np.ndarray]
    group_scale_draws: Optional[np.ndarray]
    runtime_seconds: float
    rhat_max: float
    bulk_ess_min: float
    divergence_ratio: float
    converged: bool
    tau_draws: Optional[np.ndarray] = None
    error: str = ""
    diagnostics: Optional[Dict[str, Any]] = None


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(name: str, log_file: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def experiment_seed(experiment_id: int, setting_id: int, replicate_id: int, master_seed: int = MASTER_SEED) -> int:
    return int(master_seed + 100000 * int(experiment_id) + 1000 * int(setting_id) + int(replicate_id))


def canonical_groups(group_sizes: Sequence[int]) -> List[List[int]]:
    groups: List[List[int]] = []
    start = 0
    for g in group_sizes:
        width = int(g)
        if width <= 0:
            raise ValueError("group sizes must be positive")
        groups.append(list(range(start, start + width)))
        start += width
    return groups


def standardize_columns(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    x = arr - arr.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd < 1e-10, 1.0, sd)
    return x / sd


def nearest_positive_definite(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, float(eps))
    out = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(out))
    d = np.where(d < eps, 1.0, d)
    out = out / np.outer(d, d)
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 1.0)
    return out


def block_correlation(group_sizes: Sequence[int], rho_within: float, rho_between: float) -> np.ndarray:
    groups = canonical_groups(group_sizes)
    p = int(sum(group_sizes))
    corr = np.full((p, p), float(rho_between), dtype=float)
    np.fill_diagonal(corr, 1.0)
    for g in groups:
        idx = np.asarray(g, dtype=int)
        for i in idx:
            for j in idx:
                if i != j:
                    corr[i, j] = float(rho_within)
    return nearest_positive_definite(corr)


def sample_correlated_design(
    n: int,
    group_sizes: Sequence[int],
    rho_within: float,
    rho_between: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    corr = block_correlation(group_sizes, rho_within=rho_within, rho_between=rho_between)
    X = rng.multivariate_normal(mean=np.zeros(corr.shape[0]), cov=corr, size=int(n))
    X = standardize_columns(X)
    return X, corr


def flatten_draws(draws: Optional[np.ndarray], *, scalar: bool = False) -> Optional[np.ndarray]:
    if draws is None:
        return None
    arr = np.asarray(draws, dtype=float)
    if scalar:
        return arr.reshape(-1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(-1, *arr.shape[2:])


def posterior_mean(draws: Optional[np.ndarray]) -> Optional[np.ndarray]:
    flat = flatten_draws(draws, scalar=False)
    if flat is None:
        return None
    return flat.mean(axis=0)


def posterior_ci95(draws: Optional[np.ndarray]) -> Optional[np.ndarray]:
    flat = flatten_draws(draws, scalar=False)
    if flat is None:
        return None
    return np.quantile(flat, [0.025, 0.975], axis=0)


def collect_model_draws(model: Any) -> Dict[str, Optional[np.ndarray]]:
    mapping = {
        "beta": "coef_samples_",
        "tau": "tau_samples_",
        "sigma": "sigma_samples_",
        "lambda": "lambda_samples_",
        "kappa": "kappa_samples_",
        "a": "a_samples_",
        "group_scale": "group_lambda_samples_",
        "gamma2": "gamma2_samples_",
    }
    out: Dict[str, Optional[np.ndarray]] = {}
    for k, attr in mapping.items():
        value = getattr(model, attr, None)
        out[k] = None if value is None else np.asarray(value, dtype=float)
    return out


def _diagnostic_from_hmc(model: Any, beta_draws: Optional[np.ndarray]) -> tuple[float, float, float]:
    diag = getattr(model, "sampler_diagnostics_", {})
    hmc = diag.get("hmc", {}) if isinstance(diag, Mapping) else {}
    divergences = int(hmc.get("divergences", -1)) if isinstance(hmc, Mapping) else -1
    n_draws = 0
    if beta_draws is not None:
        arr = np.asarray(beta_draws)
        if arr.ndim >= 3:
            n_draws = int(arr.shape[0] * arr.shape[1])
        elif arr.ndim == 2:
            n_draws = int(arr.shape[0])
    div_ratio = float(divergences / max(n_draws, 1)) if divergences >= 0 and n_draws > 0 else float("nan")
    pq = diag.get("posterior_quality", {}) if isinstance(diag, Mapping) else {}
    rhat = float(pq.get("max_rhat", float("nan"))) if isinstance(pq, Mapping) else float("nan")
    ess = float(pq.get("min_ess", float("nan"))) if isinstance(pq, Mapping) else float("nan")
    return rhat, ess, div_ratio


def diagnostics_summary_for_method(
    model: Any,
    tracked_params: Sequence[str],
    beta_draws: Optional[np.ndarray],
    config: SamplerConfig,
) -> tuple[float, float, float, bool, Dict[str, Any]]:
    from simulation_project.src.core.diagnostics.convergence import summarize_convergence

    draws = collect_model_draws(model)
    subset = {k: v for k, v in draws.items() if k in set(tracked_params) and v is not None}

    rhat_max = float("nan")
    ess_min = float("nan")
    detail = summarize_convergence(subset) if subset else {}
    rvals: List[float] = []
    evals: List[float] = []
    for item in detail.values():
        if isinstance(item, Mapping):
            rv = item.get("rhat_max", float("nan"))
            ev = item.get("ess_min", float("nan"))
            if np.isfinite(rv):
                rvals.append(float(rv))
            if np.isfinite(ev):
                evals.append(float(ev))
    if rvals:
        rhat_max = float(max(rvals))
    if evals:
        ess_min = float(min(evals))

    hmc_rhat, hmc_ess, div_ratio = _diagnostic_from_hmc(model, beta_draws=beta_draws)
    if np.isfinite(hmc_rhat):
        rhat_max = float(hmc_rhat)
    if np.isfinite(hmc_ess):
        ess_min = float(hmc_ess)

    # Single-chain runs do not have meaningful R-hat; treat it as unavailable
    # so convergence relies on ESS + divergence diagnostics.
    if int(getattr(config, "chains", 0)) < 2:
        rhat_max = float("nan")

    # Divergence diagnostics are HMC-specific. For samplers without this signal,
    # we gate convergence on R-hat/ESS only.
    div_ok = (not np.isfinite(div_ratio)) or (div_ratio < float(config.max_divergence_ratio))
    # R-hat requires >=2 chains; when unavailable (NaN from single-chain runs),
    # skip the R-hat gate and rely on ESS + divergence rate alone.
    rhat_ok = (not np.isfinite(rhat_max)) or (rhat_max < float(config.rhat_threshold))
    converged = bool(
        rhat_ok
        and np.isfinite(ess_min)
        and (ess_min > float(config.ess_threshold))
        and div_ok
    )

    merged = {
        "convergence_detail": detail,
        "sampler_diagnostics": getattr(model, "sampler_diagnostics_", {}),
    }
    return rhat_max, ess_min, div_ratio, converged, merged


def save_dataframe(df: Any, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def save_json(obj: Mapping[str, Any], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_pandas():
    """
    Import pandas with a Windows-safe platform.machine() shim.

    Some Windows environments can hang in platform.machine(), which pandas
    calls during import (pandas.compat._constants). We temporarily replace
    platform.machine() with a lightweight env-var based implementation.
    """
    import importlib
    import platform

    orig_machine = getattr(platform, "machine", None)

    def _fast_machine() -> str:
        return (
            os.environ.get("PROCESSOR_ARCHITECTURE")
            or os.environ.get("PROCESSOR_ARCHITEW6432")
            or "unknown"
        )

    patched = callable(orig_machine)
    if patched:
        platform.machine = _fast_machine  # type: ignore[assignment]
    try:
        return importlib.import_module("pandas")
    finally:
        if patched and orig_machine is not None:
            platform.machine = orig_machine  # type: ignore[assignment]


def timed_call(fn, *args, **kwargs) -> tuple[Any, float]:
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = max(time.perf_counter() - start, 1e-12)
    return out, elapsed


def rhs_style_tau0(n: int, p: int, p0: int) -> float:
    p0_use = max(int(p0), 1)
    denom = max(int(p) - p0_use, 1)
    return float((p0_use / denom) / math.sqrt(max(int(n), 1)))


def logistic_pseudo_sigma(y: np.ndarray, *, clip_eps: float = 1e-3) -> float:
    """Pseudo-sigma calibration for logistic RHS (Piironen & Vehtari, 2017)."""
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if y_arr.size == 0:
        return 2.0
    mu = float(np.mean(y_arr))
    eps = float(max(clip_eps, 1e-8))
    mu = float(min(max(mu, eps), 1.0 - eps))
    return float(1.0 / math.sqrt(mu * (1.0 - mu)))


def method_display_name(name: str) -> str:
    mapping = {
        "GR_RHS": "GR-RHS",
        "RHS": "RHS",
        "GIGG_MMLE": "GIGG-MMLE",
        "GIGG_b_small": "GIGG (b=1/n)",
        "GIGG_GHS": "GIGG-GHS (b=1/2)",
        "GIGG_b_large": "GIGG (b=1)",
        "GHS_plus": "Grouped Horseshoe+",
        "OLS": "OLS",
        "LASSO_CV": "Lasso (CV)",
        "GR_RHS_full": "GR-RHS (full)",
        "GR_RHS_no_ag": "GR-RHS (no a_g)",
        "GR_RHS_no_local_scales": "GR-RHS (lambda_j=1)",
        "GR_RHS_shared_kappa": "GR-RHS (shared kappa)",
        "GR_RHS_no_kappa": "GR-RHS (no kappa)",
    }
    return mapping.get(name, name)
