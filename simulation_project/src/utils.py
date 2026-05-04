from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

MASTER_SEED = 20260415


@dataclass(frozen=True)
class SamplerConfig:
    chains: int = 4
    warmup: int = 500
    post_warmup_draws: int = 500
    adapt_delta: float = 0.95
    max_treedepth: int = 12
    strict_adapt_delta: float = 0.99
    strict_max_treedepth: int = 14
    max_divergence_ratio: float = 0.005
    rhat_threshold: float = 1.015
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
    rho_w = float(rho_within)
    for g in groups:
        idx = np.asarray(g, dtype=int)
        corr[np.ix_(idx, idx)] = rho_w
    np.fill_diagonal(corr, 1.0)
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
    predictive_rhats: List[float] = []
    predictive_esses: List[float] = []
    global_scale_rhats: List[float] = []
    global_scale_esses: List[float] = []
    for name, item in detail.items():
        if not isinstance(item, Mapping):
            continue
        rv = item.get("rhat_max", float("nan"))
        ev = item.get("ess_min", float("nan"))
        if str(name) in {"beta", "kappa", "group_scale", "lambda"}:
            if np.isfinite(rv):
                predictive_rhats.append(float(rv))
            if np.isfinite(ev):
                predictive_esses.append(float(ev))
        if str(name) in {"tau", "tau2", "sigma", "sigma2"}:
            if np.isfinite(rv):
                global_scale_rhats.append(float(rv))
            if np.isfinite(ev):
                global_scale_esses.append(float(ev))
    merged["convergence_partition"] = {
        "predictive_rhat_max": float(max(predictive_rhats)) if predictive_rhats else float("nan"),
        "predictive_ess_min": float(min(predictive_esses)) if predictive_esses else float("nan"),
        "global_scale_rhat_max": float(max(global_scale_rhats)) if global_scale_rhats else float("nan"),
        "global_scale_ess_min": float(min(global_scale_esses)) if global_scale_esses else float("nan"),
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


def _artifact_safe_name(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    safe = safe.strip("._-")
    return safe or "unknown"


def _json_ready_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        val = float(value)
        if np.isnan(val):
            return None
        if np.isposinf(val):
            return "inf"
        if np.isneginf(val):
            return "-inf"
        return val
    return value


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    return _json_ready_scalar(value)


def append_jsonl_records(path: Path | str, records: Sequence[Mapping[str, Any]]) -> None:
    """Append JSON-serializable checkpoint records as JSON Lines."""
    if not records:
        return
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_ready(dict(record)), ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def save_array_bundle(
    out_path: Path | str,
    /,
    **arrays: Any,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(name): np.asarray(value)
        for name, value in arrays.items()
        if value is not None
    }
    np.savez_compressed(path, **payload)
    return path


def fit_result_summary_payload(
    result: FitResult,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    include_draw_shapes: bool = True,
) -> dict[str, Any]:
    draw_shapes: dict[str, list[int] | None] = {}
    if include_draw_shapes:
        draw_shapes = {
            "beta_mean": None if result.beta_mean is None else [int(x) for x in np.asarray(result.beta_mean).shape],
            "beta_draws": None if result.beta_draws is None else [int(x) for x in np.asarray(result.beta_draws).shape],
            "kappa_draws": None if result.kappa_draws is None else [int(x) for x in np.asarray(result.kappa_draws).shape],
            "group_scale_draws": None if result.group_scale_draws is None else [int(x) for x in np.asarray(result.group_scale_draws).shape],
            "tau_draws": None if result.tau_draws is None else [int(x) for x in np.asarray(result.tau_draws).shape],
        }
    payload: dict[str, Any] = {
        "method": str(result.method),
        "status": str(result.status),
        "converged": bool(result.converged),
        "runtime_seconds": float(result.runtime_seconds),
        "rhat_max": _json_ready_scalar(result.rhat_max),
        "bulk_ess_min": _json_ready_scalar(result.bulk_ess_min),
        "divergence_ratio": _json_ready_scalar(result.divergence_ratio),
        "error": str(result.error or ""),
        "has_beta_mean": bool(result.beta_mean is not None),
        "has_beta_draws": bool(result.beta_draws is not None),
        "has_kappa_draws": bool(result.kappa_draws is not None),
        "has_group_scale_draws": bool(result.group_scale_draws is not None),
        "has_tau_draws": bool(result.tau_draws is not None),
        "diagnostics": json_ready(result.diagnostics or {}),
    }
    if include_draw_shapes:
        payload["draw_shapes"] = draw_shapes
    if extra_metadata:
        payload["metadata"] = json_ready(dict(extra_metadata))
    if result.beta_mean is not None:
        beta_mean_arr = np.asarray(result.beta_mean, dtype=float).reshape(-1)
        payload["beta_mean_summary"] = {
            "dimension": int(beta_mean_arr.size),
            "l1_norm": float(np.sum(np.abs(beta_mean_arr))),
            "l2_norm": float(np.linalg.norm(beta_mean_arr)),
            "max_abs": float(np.max(np.abs(beta_mean_arr))) if beta_mean_arr.size else 0.0,
            "nonzero_abs_gt_1e-12": int(np.sum(np.abs(beta_mean_arr) > 1e-12)),
        }
    if result.tau_draws is not None:
        tau_arr = np.asarray(result.tau_draws, dtype=float).reshape(-1)
        if tau_arr.size:
            payload["tau_draws_summary"] = {
                "mean": float(np.mean(tau_arr)),
                "median": float(np.median(tau_arr)),
                "sd": float(np.std(tau_arr, ddof=0)),
                "q025": float(np.quantile(tau_arr, 0.025)),
                "q975": float(np.quantile(tau_arr, 0.975)),
            }
    return payload


def save_fit_result_artifacts(
    root_dir: Path | str,
    *,
    result: FitResult,
    run_context: Mapping[str, Any] | None = None,
    coefficient_truth: Any = None,
    dataset_arrays: Mapping[str, Any] | None = None,
    dataset_metadata: Mapping[str, Any] | None = None,
    extra_arrays: Mapping[str, Any] | None = None,
    extra_json: Mapping[str, Any] | None = None,
    save_dataset_bundle: bool = False,
) -> dict[str, str]:
    root = ensure_dir(root_dir)
    fit_dir = ensure_dir(root / "fit")

    summary_path = fit_dir / "fit_summary.json"
    draws_path = fit_dir / "posterior_draws.npz"
    beta_mean_path = fit_dir / "beta_mean.npy"
    coefficient_path = fit_dir / "coefficient_detail.csv"

    save_json(
        fit_result_summary_payload(
            result,
            extra_metadata=run_context,
        ),
        summary_path,
    )

    written: dict[str, str] = {
        "fit_dir": str(fit_dir),
        "fit_summary": str(summary_path),
    }

    if result.beta_mean is not None:
        np.save(beta_mean_path, np.asarray(result.beta_mean, dtype=float))
        written["beta_mean"] = str(beta_mean_path)

    draws_payload = {
        "beta_draws": result.beta_draws,
        "kappa_draws": result.kappa_draws,
        "group_scale_draws": result.group_scale_draws,
        "tau_draws": result.tau_draws,
    }
    if any(value is not None for value in draws_payload.values()):
        save_array_bundle(draws_path, **draws_payload)
        written["posterior_draws"] = str(draws_path)

    beta_mean_flat = None if result.beta_mean is None else np.asarray(result.beta_mean, dtype=float).reshape(-1)
    beta_true_flat = None if coefficient_truth is None else np.asarray(coefficient_truth, dtype=float).reshape(-1)
    if beta_mean_flat is not None or beta_true_flat is not None:
        p = max(
            0 if beta_mean_flat is None else int(beta_mean_flat.size),
            0 if beta_true_flat is None else int(beta_true_flat.size),
        )
        rows: list[dict[str, Any]] = []
        for idx in range(p):
            true_value = float(beta_true_flat[idx]) if beta_true_flat is not None and idx < beta_true_flat.size else float("nan")
            est_value = float(beta_mean_flat[idx]) if beta_mean_flat is not None and idx < beta_mean_flat.size else float("nan")
            error = est_value - true_value if np.isfinite(est_value) and np.isfinite(true_value) else float("nan")
            rows.append(
                {
                    "coefficient_index": int(idx),
                    "beta_true": None if not np.isfinite(true_value) else true_value,
                    "beta_estimate": None if not np.isfinite(est_value) else est_value,
                    "error": None if not np.isfinite(error) else float(error),
                    "abs_error": None if not np.isfinite(error) else float(abs(error)),
                    "is_true_nonzero": bool(np.isfinite(true_value) and abs(true_value) > 1e-12),
                    "is_est_nonzero": bool(np.isfinite(est_value) and abs(est_value) > 1e-12),
                }
            )
        pd = load_pandas()
        pd.DataFrame(rows).to_csv(coefficient_path, index=False)
        written["coefficient_detail"] = str(coefficient_path)

    if save_dataset_bundle or dataset_arrays or dataset_metadata:
        dataset_dir = ensure_dir(root / "dataset")
        if dataset_arrays:
            dataset_arrays_path = dataset_dir / "dataset_arrays.npz"
            save_array_bundle(dataset_arrays_path, **dict(dataset_arrays))
            written["dataset_arrays"] = str(dataset_arrays_path)
        if dataset_metadata is not None:
            dataset_meta_path = dataset_dir / "dataset_metadata.json"
            save_json(json_ready(dict(dataset_metadata)), dataset_meta_path)
            written["dataset_metadata"] = str(dataset_meta_path)

    if extra_arrays:
        extras_dir = ensure_dir(root / "extras")
        extra_arrays_path = extras_dir / "extra_arrays.npz"
        save_array_bundle(extra_arrays_path, **dict(extra_arrays))
        written["extra_arrays"] = str(extra_arrays_path)

    if extra_json:
        extras_dir = ensure_dir(root / "extras")
        extra_json_path = extras_dir / "extra_metadata.json"
        save_json(json_ready(dict(extra_json)), extra_json_path)
        written["extra_metadata"] = str(extra_json_path)

    return written


def load_pandas():
    """
    Import pandas with a Windows-safe platform.machine() shim.

    Some Windows environments can hang in platform.machine(), which pandas
    calls during import (pandas.compat._constants). We temporarily replace
    platform.machine() with a lightweight env-var based implementation.
    """
    cached = sys.modules.get("pandas")
    if cached is not None:
        return cached

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
        "GR_RHS_LowDim": "GR-RHS-LowDim",
        "GR_RHS_HighDim": "GR-RHS-HighDim",
        "RHS": "RHS",
        "RHS_LowDim": "RHS-LowDim",
        "RHS_HighDim": "RHS-HighDim",
        "RHS_Gibbs": "RHS-Gibbs",
        "GIGG_MMLE": "GIGG-MMLE",
        "GIGG_b_small": "GIGG (b=1/n)",
        "GIGG_GHS": "GIGG-GHS (b=1/2)",
        "GIGG_b_large": "GIGG (b=1)",
        "GHS_plus": "Grouped Horseshoe+",
        "OLS": "OLS",
        "LASSO_CV": "Lasso (CV)",
        "GR_RHS_no_local_scales": "GR-RHS (lambda_j=1)",
        "GR_RHS_shared_kappa": "GR-RHS (shared kappa)",
        "GR_RHS_no_kappa": "GR-RHS (no kappa)",
    }
    return mapping.get(name, name)


def method_result_label(name: str) -> str:
    mapping = {
        "GR_RHS": "GR-RHS",
        "GR_RHS_LowDim": "GR-RHS-LowDim [gibbs_staged]",
        "GR_RHS_HighDim": "GR-RHS-HighDim [collapsed_profile]",
        "RHS": "RHS",
        "RHS_LowDim": "RHS-LowDim [stan_rstanarm_hs]",
        "RHS_HighDim": "RHS-HighDim [woodbury_slice]",
        "RHS_Gibbs": "RHS-Gibbs [woodbury_slice]",
        "RHS_oracle": "RHS_oracle [stan_rstanarm_hs]",
    }
    return mapping.get(name, method_display_name(name))


def _progress_format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        val = float(value)
        if not np.isfinite(val):
            return "nan"
        return f"{val:.5f}"
    text = str(value).strip()
    return text or "NA"


def print_experiment_result(
    exp_name: str,
    row: Mapping[str, Any],
    *,
    context_keys: Sequence[str] | None = None,
    metric_keys: Sequence[str] | None = None,
    log_path: str | Path | None = None,
) -> None:
    ctx_keys = [str(k) for k in (context_keys or [])]
    metric_keys_use = [str(k) for k in (metric_keys or [])]
    parts: list[str] = [f"[{str(exp_name)}]"]
    for key in ctx_keys:
        if key in row:
            parts.append(f"{key}={_progress_format_value(row.get(key))}")
    for key in ("status", "converged", "fit_attempts", "runtime_seconds"):
        if key in row and key not in set(ctx_keys):
            parts.append(f"{key}={_progress_format_value(row.get(key))}")
    for key in metric_keys_use:
        if key in row:
            parts.append(f"{key}={_progress_format_value(row.get(key))}")
    line = " ".join(parts)
    print(line, flush=True)
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
