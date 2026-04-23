"""Posterior convergence diagnostics (R-hat, ESS)."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

Array = np.ndarray

_SCALAR_PARAMETER_NAMES = {
    "sigma",
    "sigma2",
    "tau",
    "tau2",
    "c",
    "beta0",
    "intercept",
}


def _safe_stat(values: Array, reducer: Any) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(reducer(finite))


def _reshape_samples(samples: Array, *, scalar_param: bool = False) -> Tuple[Array, Tuple[int, ...]]:
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 0:
        raise ValueError("samples must have at least one dimension (draws)")

    if scalar_param:
        if arr.ndim == 1:
            arr = arr.reshape(1, arr.shape[0], 1)
            param_shape = ()
        elif arr.ndim == 2:
            # Scalar parameters may be stored as (chains, draws).
            arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
            param_shape = ()
        else:
            trailing_shape = arr.shape[2:]
            if int(np.prod(trailing_shape, dtype=int)) == 1:
                arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
                param_shape = ()
            else:
                param_shape = trailing_shape
    elif arr.ndim == 1:
        arr = arr.reshape(1, arr.shape[0], 1)
        param_shape = ()
    elif arr.ndim == 2:
        # interpret as (draws, parameters)
        arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        param_shape = (arr.shape[2],)
    else:
        # expect (chains, draws, ...)
        arr = np.asarray(arr, dtype=float)
        if arr.shape[0] < 1:
            raise ValueError("expected chain axis with length >= 1")
        param_shape = arr.shape[2:]
    # ensure even draws for splitting
    draws = arr.shape[1]
    if draws < 4:
        raise ValueError("need at least 4 draws for convergence diagnostics")
    if draws % 2 == 1:
        arr = arr[:, :-1]
        draws -= 1
    if draws < 4:
        raise ValueError("not enough draws after adjustment; need >=4")
    return arr, param_shape


def _split_chains(chains: Array) -> Array:
    C, N = chains.shape[:2]
    half = N // 2
    left = chains[:, :half]
    right = chains[:, half:]
    return np.concatenate([left, right], axis=0)


def _rhat_from_chains(chains: Array) -> Array:
    chains = np.asarray(chains, dtype=float)
    C, N = chains.shape[:2]
    if C < 2:
        raise ValueError("split R-hat requires at least two chains after splitting")
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    W = chain_vars.mean(axis=0)
    B = N * chain_means.var(axis=0, ddof=1)
    var_hat = ((N - 1) / N) * W + B / N
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(np.where(W > 0, var_hat / W, 1.0))
    return rhat


def split_rhat(samples: Array, *, scalar_param: bool = False) -> Array:
    arr, param_shape = _reshape_samples(samples, scalar_param=scalar_param)
    split = _split_chains(arr)
    split2d = split.reshape(split.shape[0], split.shape[1], -1)
    finite_cols = np.all(np.isfinite(split2d), axis=(0, 1))
    rhat_flat = np.full(split2d.shape[2], np.nan, dtype=float)
    if np.any(finite_cols):
        good = split2d[:, :, finite_cols].reshape(split.shape[0], split.shape[1], -1)
        rhat_flat[finite_cols] = np.asarray(_rhat_from_chains(good), dtype=float).reshape(-1)
    rhat = rhat_flat.reshape(param_shape if param_shape else (1,))
    if param_shape:
        return rhat.reshape(param_shape)
    return np.squeeze(rhat)


def _autocorrelation(chain: Array) -> Array:
    chain = np.asarray(chain, dtype=float)
    n = chain.shape[0]
    if n < 3:
        return np.ones((n,) + chain.shape[1:])
    centered = chain - chain.mean(axis=0)
    var0 = np.mean(centered * centered, axis=0)
    zero_mask = var0 <= 1e-12
    if np.all(zero_mask):
        shape = (n,) + chain.shape[1:]
        ac = np.zeros(shape, dtype=float)
        ac[0] = 1.0
        return ac
    var0_safe = np.where(zero_mask, 1.0, var0)
    ac = np.empty((n,) + chain.shape[1:], dtype=float)
    for lag in range(n):
        prod = centered[: n - lag] * centered[lag:]
        ac[lag] = np.mean(prod, axis=0) / var0_safe
    if np.any(zero_mask):
        ac_flat = ac.reshape(n, -1)
        mask_flat = zero_mask.reshape(-1)
        ac_flat[:, mask_flat] = 0.0
        ac_flat[0, mask_flat] = 1.0
    return ac


def _ess_from_chains(chains: Array) -> Array:
    chains = np.asarray(chains, dtype=float)
    C, N = chains.shape[:2]
    chains2d = chains.reshape(C, N, -1)
    param_shape = chains.shape[2:]
    finite_mask = np.isfinite(chains2d)
    all_nonfinite = ~np.any(finite_mask, axis=(0, 1))
    sanitized = np.where(finite_mask, chains2d, np.nan)
    counts = np.sum(finite_mask, axis=1, keepdims=True)
    sums = np.nansum(sanitized, axis=1, keepdims=True)
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centered = np.where(finite_mask, chains2d - mean, 0.0)
    ac_avg = np.zeros((N,) + chains2d.shape[2:], dtype=float)
    for c in range(C):
        ac_avg += _autocorrelation(centered[c])
    ac_avg /= C
    rho = ac_avg[1:]
    ess = np.empty(ac_avg.shape[1:], dtype=float)
    for idx in np.ndindex(ac_avg.shape[1:]):
        ac_seq = rho[(slice(None),) + idx]
        total = 0.0
        for k in range(0, len(ac_seq), 2):
            pair = ac_seq[k]
            if k + 1 < len(ac_seq):
                pair += ac_seq[k + 1]
            if pair < 0:
                break
            total += pair
        ess[idx] = C * N / max(1.0, 1.0 + 2.0 * total)
        ess[idx] = min(ess[idx], C * N)
    if np.any(all_nonfinite):
        ess_flat = ess.reshape(-1)
        ess_flat[np.where(all_nonfinite)[0]] = 0.0
    return ess if param_shape else np.squeeze(ess)


def effective_sample_size(samples: Array, *, scalar_param: bool = False) -> Array:
    arr, param_shape = _reshape_samples(samples, scalar_param=scalar_param)
    split = _split_chains(arr)
    split2d = split.reshape(split.shape[0], split.shape[1], -1)
    finite_cols = np.all(np.isfinite(split2d), axis=(0, 1))
    ess_flat = np.full(split2d.shape[2], np.nan, dtype=float)
    if np.any(finite_cols):
        good = split2d[:, :, finite_cols].reshape(split.shape[0], split.shape[1], -1)
        ess_flat[finite_cols] = np.asarray(_ess_from_chains(good), dtype=float).reshape(-1)
    ess = ess_flat.reshape(param_shape if param_shape else (1,))
    if param_shape:
        return ess.reshape(param_shape)
    return np.squeeze(ess)


def _diagnostic_metadata(
    samples: Array,
    *,
    min_chains_for_rhat: int = 2,
    scalar_param: bool = False,
) -> Dict[str, Any]:
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 0:
        raise ValueError("samples must have at least one dimension (draws)")
    if arr.ndim == 1:
        raw_num_chains = 1
        raw_num_draws = int(arr.shape[0])
    elif arr.ndim == 2:
        if scalar_param:
            raw_num_chains = int(arr.shape[0])
            raw_num_draws = int(arr.shape[1])
        else:
            raw_num_chains = 1
            raw_num_draws = int(arr.shape[0])
    else:
        raw_num_chains = int(arr.shape[0])
        raw_num_draws = int(arr.shape[1])
    return {
        "raw_num_chains": raw_num_chains,
        "raw_num_draws": raw_num_draws,
        "diagnostic_valid": bool(raw_num_chains >= int(min_chains_for_rhat)),
    }


def summarize_convergence(
    samples: Dict[str, Array],
    *,
    min_chains_for_rhat: int = 2,
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for name, arr in samples.items():
        scalar_param = str(name).strip().lower() in _SCALAR_PARAMETER_NAMES
        try:
            meta = _diagnostic_metadata(
                arr,
                min_chains_for_rhat=min_chains_for_rhat,
                scalar_param=scalar_param,
            )
            rhat = split_rhat(arr, scalar_param=scalar_param)
            ess = effective_sample_size(arr, scalar_param=scalar_param)
            flat_rhat = np.asarray(rhat).ravel()
            flat_ess = np.asarray(ess).ravel()
            with np.errstate(divide="ignore", invalid="ignore"):
                # MCSE / posterior sd ~= sqrt(1 / ESS) for the posterior mean.
                mcse_over_sd = np.where(flat_ess > 0.0, np.sqrt(1.0 / flat_ess), np.inf)
            summary[name] = {
                "rhat_max": _safe_stat(flat_rhat, np.max),
                "rhat_median": _safe_stat(flat_rhat, np.median),
                "ess_min": _safe_stat(flat_ess, np.min),
                "ess_median": _safe_stat(flat_ess, np.median),
                "mcse_over_sd_max": _safe_stat(mcse_over_sd, np.max),
                "mcse_over_sd_median": _safe_stat(mcse_over_sd, np.median),
                **meta,
            }
        except ValueError as exc:
            summary[name] = {
                "error": str(exc),
                **_diagnostic_metadata(
                    arr,
                    min_chains_for_rhat=min_chains_for_rhat,
                    scalar_param=scalar_param,
                ),
            }
    return summary

