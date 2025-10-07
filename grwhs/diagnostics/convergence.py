"""Posterior convergence diagnostics (R-hat, ESS)."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

Array = np.ndarray


def _reshape_samples(samples: Array) -> Tuple[Array, Tuple[int, ...]]:
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 0:
        raise ValueError("samples must have at least one dimension (draws)")

    if arr.ndim == 1:
        arr = arr.reshape(1, arr.shape[0], 1)
        param_shape: Tuple[int, ...] = ()
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


def split_rhat(samples: Array) -> Array:
    arr, param_shape = _reshape_samples(samples)
    split = _split_chains(arr)
    rhat = _rhat_from_chains(split)
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
    centered = chains2d - chains2d.mean(axis=1, keepdims=True)
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
    return ess if param_shape else np.squeeze(ess)


def effective_sample_size(samples: Array) -> Array:
    arr, param_shape = _reshape_samples(samples)
    split = _split_chains(arr)
    ess = _ess_from_chains(split)
    if param_shape:
        return ess.reshape(param_shape)
    return np.squeeze(ess)


def summarize_convergence(samples: Dict[str, Array]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for name, arr in samples.items():
        try:
            rhat = split_rhat(arr)
            ess = effective_sample_size(arr)
            flat_rhat = np.asarray(rhat).ravel()
            flat_ess = np.asarray(ess).ravel()
            summary[name] = {
                "rhat_max": float(np.max(flat_rhat)),
                "rhat_median": float(np.median(flat_rhat)),
                "ess_min": float(np.min(flat_ess)),
                "ess_median": float(np.median(flat_ess)),
            }
        except ValueError as exc:
            summary[name] = {"error": str(exc)}
    return summary
