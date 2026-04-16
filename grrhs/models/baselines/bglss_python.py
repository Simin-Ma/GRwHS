"""Python-native Bayesian Group Lasso with Spike-and-Slab (BGLSS).

This implementation is designed as a practical benchmark baseline that mirrors
the high-level `MBSGS::BGLSS` interface while remaining fully Python-native.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import __main__
import os
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.stats import geninvgauss


GroupsLike = Sequence[Sequence[int]]


def _normalize_groups(groups: GroupsLike) -> list[np.ndarray]:
    if not groups:
        raise ValueError("At least one group must be specified.")
    out: list[np.ndarray] = []
    for i, g in enumerate(groups):
        idx = np.asarray(list(g), dtype=int).reshape(-1)
        if idx.size == 0:
            raise ValueError(f"Group {i} is empty.")
        out.append(idx)
    return out


def _check_partition(groups: list[np.ndarray], p: int) -> None:
    mask = np.zeros(p, dtype=int)
    for g in groups:
        if np.any(g < 0) or np.any(g >= p):
            raise ValueError("Group indices out of bounds.")
        mask[g] += 1
    if np.any(mask != 1):
        bad = np.where(mask != 1)[0].tolist()
        raise ValueError(f"Groups must form a partition of features; bad indices: {bad[:20]}")


def _build_group_index(groups: list[np.ndarray], p: int) -> np.ndarray:
    idx = np.empty(p, dtype=int)
    for g_id, block in enumerate(groups):
        idx[block] = g_id
    return idx


def _sample_mvn_precision(precision: np.ndarray, rhs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw x ~ N(precision^{-1} rhs, precision^{-1}) via Cholesky, reusing factorization."""
    p = precision.shape[0]
    jitter = 1e-10
    eye = np.eye(p)
    for _ in range(8):
        try:
            a = precision + jitter * eye
            L = np.linalg.cholesky(a)
            mu = cho_solve((L, True), rhs, check_finite=False)
            z = rng.normal(size=p)
            v = solve_triangular(L.T, z, lower=False, check_finite=False)
            return mu + v
        except np.linalg.LinAlgError:
            jitter *= 10.0

    vals, vecs = np.linalg.eigh(0.5 * (precision + precision.T))
    vals = np.maximum(vals, 1e-10)
    inv = vecs @ np.diag(1.0 / vals) @ vecs.T
    mu = inv @ rhs
    z = rng.normal(size=p)
    v = vecs @ (z / np.sqrt(vals))
    return mu + v


def _run_single_chain(
    *,
    chain_idx: int,
    seed: int,
    niter: int,
    burnin: int,
    thinning: int,
    store_beta_samples: bool,
    Xw: np.ndarray,
    yw: np.ndarray,
    XtX: np.ndarray,
    Xty: np.ndarray,
    group_index: np.ndarray,
    G: int,
    group_sizes: np.ndarray,
    a: float,
    b: float,
    pi_prior: bool,
    pi_init: float,
    alpha: float,
    gamma: float,
    lambda_slab2: float,
    lambda_spike2: float,
    update_tau: bool,
    num_update: int,
    niter_update: int,
    overdispersed_init: bool,
    init_beta_scale: float,
    init_tau2_log_sd: float,
    tempering_enabled: bool,
    tempering_initial_temp: float,
    tempering_fraction: float,
    spike_annealing_enabled: bool,
    spike_annealing_initial_spike2: float,
    spike_annealing_fraction: float,
    beta_center: Optional[np.ndarray],
    continuation_state: Optional[Mapping[str, Any]],
    replica_exchange_enabled: bool,
    replica_exchange_high_temp: float,
    replica_exchange_swap_interval: int,
    replica_exchange_burnin_only: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed) + 1009 * int(chain_idx))
    eps = 1e-10

    n, p = Xw.shape
    post_burn = max(0, int(niter) - int(burnin))
    kept = post_burn // max(1, int(thinning))

    if store_beta_samples and kept > 0:
        beta_draws = np.zeros((kept, p), dtype=float)
    else:
        beta_draws = None
    pi_draws = np.zeros(kept, dtype=float) if kept > 0 else None
    sigma2_draws = np.zeros(kept, dtype=float) if kept > 0 else None

    use_rex = bool(replica_exchange_enabled) and float(replica_exchange_high_temp) > 1.0
    temps = np.array([1.0, float(replica_exchange_high_temp)], dtype=float) if use_rex else np.array([1.0], dtype=float)
    n_reps = int(temps.shape[0])

    beta_states = np.zeros((n_reps, p), dtype=float)
    sigma2_states = np.zeros(n_reps, dtype=float)
    pi_states = np.zeros(n_reps, dtype=float)
    z_states = np.zeros((n_reps, G), dtype=int)
    tau2_states = np.zeros((n_reps, G), dtype=float)
    lambda_slab2_states = np.full(n_reps, float(lambda_slab2), dtype=float)
    loglik_states = np.zeros(n_reps, dtype=float)

    beta_center_arr = np.zeros(p, dtype=float) if beta_center is None else np.asarray(beta_center, dtype=float).reshape(p)
    resid_center = yw - Xw @ beta_center_arr
    var_center = float(np.var(resid_center))
    base_sigma2 = float(var_center if var_center > 1e-8 else 1.0)
    state_payload = continuation_state or {}
    for r in range(n_reps):
        pi_r = float(np.clip(pi_init, 1e-4, 1.0 - 1e-4))
        if state_payload:
            beta_init = np.asarray(state_payload.get("beta"), dtype=float).reshape(-1)
            if beta_init.shape[0] == p:
                beta_states[r] = beta_init.copy()
            else:
                beta_states[r] = beta_center_arr.copy()

            sigma2_init = state_payload.get("sigma2")
            sigma2_states[r] = float(max(eps, float(base_sigma2 if sigma2_init is None else sigma2_init)))

            pi_init_state = state_payload.get("pi")
            if pi_init_state is not None:
                pi_r = float(np.clip(float(pi_init_state), 1e-4, 1.0 - 1e-4))

            tau2_init = state_payload.get("tau2")
            if tau2_init is not None:
                tau2_arr = np.asarray(tau2_init, dtype=float).reshape(-1)
                if tau2_arr.shape[0] == G:
                    tau2_states[r] = np.maximum(tau2_arr, eps)
                else:
                    tau2_states[r] = np.full(G, 1.0, dtype=float)
            else:
                tau2_states[r] = np.full(G, 1.0, dtype=float)

            z_init = state_payload.get("z")
            if z_init is not None:
                z_arr = np.asarray(z_init, dtype=int).reshape(-1)
                if z_arr.shape[0] == G:
                    z_states[r] = (z_arr > 0).astype(int)
                else:
                    z_states[r] = rng.binomial(1, pi_r, size=G).astype(int)
            else:
                z_states[r] = rng.binomial(1, pi_r, size=G).astype(int)

            lambda_init = state_payload.get("lambda_slab2")
            if lambda_init is not None:
                lambda_slab2_states[r] = max(eps, float(lambda_init))
        else:
            if overdispersed_init:
                beta_states[r] = (
                    beta_center_arr + rng.normal(loc=0.0, scale=max(1e-6, float(init_beta_scale)), size=p)
                ).astype(float)
                sigma2_states[r] = float(base_sigma2 * np.exp(rng.normal(loc=0.0, scale=0.4)))
                pi_r = float(rng.beta(max(a, 1e-3), max(b, 1e-3)))
                tau2_states[r] = np.exp(
                    rng.normal(loc=0.0, scale=max(1e-6, float(init_tau2_log_sd)), size=G)
                ).astype(float)
            else:
                beta_states[r] = beta_center_arr.copy()
                sigma2_states[r] = base_sigma2
                tau2_states[r] = np.full(G, 1.0, dtype=float)
            z_states[r] = rng.binomial(1, pi_r, size=G).astype(int)
        pi_states[r] = pi_r

    rex_swap_attempts = 0
    rex_swap_accepts = 0
    lambda_spike2_base = float(lambda_spike2)

    temper_steps = int(max(0, round(int(burnin) * min(1.0, max(0.0, float(tempering_fraction))))))
    spike_steps = int(max(0, round(int(burnin) * min(1.0, max(0.0, float(spike_annealing_fraction))))))

    t0 = time.perf_counter()
    for t in range(int(niter)):
        if (not use_rex) and tempering_enabled and temper_steps > 1 and t < temper_steps:
            prog_t = float(t) / float(temper_steps - 1)
            temp_sched = float(tempering_initial_temp) - (float(tempering_initial_temp) - 1.0) * prog_t
            inv_temp_sched = 1.0 / max(1.0, temp_sched)
        else:
            inv_temp_sched = 1.0

        if spike_annealing_enabled and spike_steps > 1 and t < spike_steps:
            prog_s = float(t) / float(spike_steps - 1)
            start_s = max(eps, float(spike_annealing_initial_spike2))
            end_s = max(eps, float(lambda_spike2_base))
            lambda_spike2_t = float(np.exp((1.0 - prog_s) * np.log(start_s) + prog_s * np.log(end_s)))
        else:
            lambda_spike2_t = float(lambda_spike2_base)

        for r in range(n_reps):
            inv_temp = inv_temp_sched / float(temps[r])
            beta = beta_states[r]
            sigma2 = float(sigma2_states[r])
            tau2 = tau2_states[r]
            z = z_states[r]
            pi = float(pi_states[r])
            lambda_slab2_current = float(lambda_slab2_states[r])

            if update_tau and t < int(num_update) * int(niter_update):
                sel = np.where(z == 1)[0]
                if sel.size > 0:
                    m_tau = float(np.mean(tau2[sel]))
                    lambda_slab2_current = max(
                        eps,
                        float(np.mean(group_sizes[sel] + 1.0) / max(m_tau, eps)),
                    )

            inv_sigma2 = 1.0 / max(float(sigma2), eps)
            inv_tau = inv_sigma2 / np.maximum(tau2, eps)

            precision = np.array(XtX, copy=True) * (inv_temp * inv_sigma2)
            precision[np.diag_indices(p)] += inv_tau[group_index]
            rhs = Xty * (inv_temp * inv_sigma2)
            beta = _sample_mvn_precision(precision, rhs, rng)

            beta_sq = beta * beta
            group_norm2 = np.bincount(group_index, weights=beta_sq, minlength=G)

            resid = yw - Xw @ beta
            rss = float(resid @ resid)
            quad = float(np.sum(group_norm2 / np.maximum(tau2, eps)))

            shape = float(alpha) + 0.5 * (inv_temp * n + p)
            rate = float(gamma) + 0.5 * (inv_temp * rss + quad)
            sigma2 = 1.0 / rng.gamma(shape=shape, scale=1.0 / max(rate, eps))

            for g_idx in range(G):
                chi = max(float(group_norm2[g_idx]) / max(float(sigma2), eps), eps)
                psi = lambda_slab2_current if z[g_idx] == 1 else lambda_spike2_t
                psi = max(float(psi), eps)
                b_gig = np.sqrt(chi * psi)
                scale = np.sqrt(chi / psi)
                tau2[g_idx] = float(scale * geninvgauss.rvs(0.5, b_gig, random_state=rng))
                tau2[g_idx] = max(float(tau2[g_idx]), eps)

            for g_idx, pg in enumerate(group_sizes):
                shape_g = 0.5 * (float(pg) + 1.0)
                log_f1 = shape_g * np.log(max(lambda_slab2_current, eps)) - 0.5 * lambda_slab2_current * tau2[g_idx]
                log_f0 = shape_g * np.log(max(lambda_spike2_t, eps)) - 0.5 * lambda_spike2_t * tau2[g_idx]
                log_p1 = np.log(max(pi, eps)) + log_f1
                log_p0 = np.log(max(1.0 - pi, eps)) + log_f0
                m = max(log_p0, log_p1)
                p1 = np.exp(log_p1 - m) / (np.exp(log_p1 - m) + np.exp(log_p0 - m))
                z[g_idx] = int(rng.uniform() < p1)

            if pi_prior:
                pi = float(rng.beta(a + np.sum(z), b + G - np.sum(z)))
            else:
                pi = float(np.clip(pi_init, 1e-4, 1.0 - 1e-4))

            loglik_states[r] = -0.5 * n * np.log(max(float(sigma2), eps)) - 0.5 * rss / max(float(sigma2), eps)
            beta_states[r] = beta
            sigma2_states[r] = sigma2
            tau2_states[r] = tau2
            z_states[r] = z
            pi_states[r] = pi
            lambda_slab2_states[r] = lambda_slab2_current

        if use_rex and int(replica_exchange_swap_interval) > 0:
            do_swap = (t + 1) % int(replica_exchange_swap_interval) == 0
            if replica_exchange_burnin_only:
                do_swap = do_swap and (t < int(burnin))
            if do_swap:
                rex_swap_attempts += 1
                inv0 = 1.0 / float(temps[0])
                inv1 = 1.0 / float(temps[1])
                log_alpha = (inv0 - inv1) * (float(loglik_states[1]) - float(loglik_states[0]))
                if np.log(rng.uniform()) < min(0.0, log_alpha):
                    rex_swap_accepts += 1
                    beta_states[[0, 1]] = beta_states[[1, 0]]
                    sigma2_states[[0, 1]] = sigma2_states[[1, 0]]
                    pi_states[[0, 1]] = pi_states[[1, 0]]
                    z_states[[0, 1]] = z_states[[1, 0]]
                    tau2_states[[0, 1]] = tau2_states[[1, 0]]
                    lambda_slab2_states[[0, 1]] = lambda_slab2_states[[1, 0]]
                    loglik_states[[0, 1]] = loglik_states[[1, 0]]

        if t >= int(burnin) and ((t - int(burnin)) % int(thinning) == 0):
            k = (t - int(burnin)) // int(thinning)
            if k < kept:
                if beta_draws is not None:
                    beta_draws[k, :] = beta_states[0]
                if pi_draws is not None:
                    pi_draws[k] = float(pi_states[0])
                if sigma2_draws is not None:
                    sigma2_draws[k] = float(sigma2_states[0])

    elapsed_sec = float(time.perf_counter() - t0)
    return {
        "chain_idx": int(chain_idx),
        "beta_last": beta_states[0].copy(),
        "beta_draws": beta_draws,
        "pi_draws": pi_draws,
        "sigma2_draws": sigma2_draws,
        "elapsed_sec": elapsed_sec,
        "rex_accept_rate": 0.0 if rex_swap_attempts == 0 else float(rex_swap_accepts) / float(rex_swap_attempts),
        "final_state": {
            "beta": beta_states[0].copy(),
            "sigma2": float(sigma2_states[0]),
            "pi": float(pi_states[0]),
            "z": z_states[0].copy(),
            "tau2": tau2_states[0].copy(),
            "lambda_slab2": float(lambda_slab2_states[0]),
        },
    }


@dataclass
class BGLSSPythonRegression:
    groups: GroupsLike
    niter: int = 6000
    burnin: int = 2000
    seed: int = 2025
    fit_intercept: bool = False
    a: float = 1.0
    b: float = 1.0
    pi_prior: bool = True
    pi_init: float = 0.5
    alpha: float = 0.1
    gamma: float = 0.1
    lambda_slab2: float = 0.5
    lambda_spike2: float = 25.0
    update_tau: bool = False
    num_update: int = 100
    niter_update: int = 100
    store_beta_samples: bool = True
    num_chains: int = 1
    thinning: int = 1
    chain_method: str = "auto"
    overdispersed_init: bool = True
    init_beta_scale: float = 0.1
    init_tau2_log_sd: float = 0.5
    tempering_enabled: bool = False
    tempering_initial_temp: float = 2.0
    tempering_fraction: float = 0.6
    spike_annealing_enabled: bool = False
    spike_annealing_initial_spike2: float = 6.0
    spike_annealing_fraction: float = 0.7
    init_mode: str = "ridge"
    init_ridge_lambda: float = 1.0
    replica_exchange_enabled: bool = False
    replica_exchange_high_temp: float = 2.0
    replica_exchange_swap_interval: int = 10
    replica_exchange_burnin_only: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        self.groups = _normalize_groups(self.groups)
        cm = str(self.chain_method).strip().lower()
        if cm not in {"auto", "sequential", "parallel"}:
            raise ValueError("chain_method must be one of: auto|sequential|parallel")
        self.chain_method = cm
        init_mode = str(self.init_mode).strip().lower()
        if init_mode not in {"zero", "ridge"}:
            raise ValueError("init_mode must be one of: zero|ridge")
        self.init_mode = init_mode
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.pos_mean_: Optional[np.ndarray] = None
        self.pos_median_: Optional[np.ndarray] = None
        self.coef_samples_: Optional[np.ndarray] = None
        self.pi_samples_: Optional[np.ndarray] = None
        self.sigma2_samples_: Optional[np.ndarray] = None
        self.chain_elapsed_sec_: Optional[np.ndarray] = None
        self.replica_exchange_accept_rate_: Optional[np.ndarray] = None
        self.chain_final_states_: Optional[list[dict[str, Any]]] = None
        self.chain_initial_states_: Optional[list[Mapping[str, Any]]] = None

    def set_chain_initial_states(self, states: Sequence[Mapping[str, Any]]) -> None:
        normalized: list[Mapping[str, Any]] = []
        for state in states:
            if isinstance(state, Mapping):
                normalized.append(state)
        self.chain_initial_states_ = normalized if normalized else None

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> "BGLSSPythonRegression":
        del fit_kwargs
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D.")
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y dimensions are inconsistent.")
        n, p = X_arr.shape
        _check_partition(self.groups, p)

        if self.fit_intercept:
            x_mean = X_arr.mean(axis=0)
            y_mean = float(y_arr.mean())
            Xw = X_arr - x_mean
            yw = y_arr - y_mean
        else:
            x_mean = np.zeros(p, dtype=float)
            y_mean = 0.0
            Xw = X_arr
            yw = y_arr

        num_chains = max(1, int(self.num_chains))
        thinning = max(1, int(self.thinning))
        G = len(self.groups)
        group_sizes = np.array([g.size for g in self.groups], dtype=float)
        group_index = _build_group_index(self.groups, p)

        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw
        beta_center: Optional[np.ndarray] = None
        if self.init_mode == "ridge":
            lam = max(1e-8, float(self.init_ridge_lambda))
            precision = XtX + lam * np.eye(p, dtype=float)
            try:
                beta_center = np.linalg.solve(precision, Xty)
            except np.linalg.LinAlgError:
                beta_center = np.zeros(p, dtype=float)
        post_burn = max(0, int(self.niter) - int(self.burnin))
        kept = post_burn // thinning

        method = self.chain_method
        if method == "auto":
            method = "parallel" if num_chains > 1 else "sequential"

        worker_kwargs = dict(
            seed=int(self.seed),
            niter=int(self.niter),
            burnin=int(self.burnin),
            thinning=int(thinning),
            store_beta_samples=bool(self.store_beta_samples),
            Xw=Xw,
            yw=yw,
            XtX=XtX,
            Xty=Xty,
            group_index=group_index,
            G=int(G),
            group_sizes=group_sizes,
            a=float(self.a),
            b=float(self.b),
            pi_prior=bool(self.pi_prior),
            pi_init=float(self.pi_init),
            alpha=float(self.alpha),
            gamma=float(self.gamma),
            lambda_slab2=float(self.lambda_slab2),
            lambda_spike2=float(self.lambda_spike2),
            update_tau=bool(self.update_tau),
            num_update=int(self.num_update),
            niter_update=int(self.niter_update),
            overdispersed_init=bool(self.overdispersed_init),
            init_beta_scale=float(self.init_beta_scale),
            init_tau2_log_sd=float(self.init_tau2_log_sd),
            tempering_enabled=bool(self.tempering_enabled),
            tempering_initial_temp=float(self.tempering_initial_temp),
            tempering_fraction=float(self.tempering_fraction),
            spike_annealing_enabled=bool(self.spike_annealing_enabled),
            spike_annealing_initial_spike2=float(self.spike_annealing_initial_spike2),
            spike_annealing_fraction=float(self.spike_annealing_fraction),
            beta_center=None if beta_center is None else np.asarray(beta_center, dtype=float),
            replica_exchange_enabled=bool(self.replica_exchange_enabled),
            replica_exchange_high_temp=float(self.replica_exchange_high_temp),
            replica_exchange_swap_interval=max(1, int(self.replica_exchange_swap_interval)),
            replica_exchange_burnin_only=bool(self.replica_exchange_burnin_only),
        )

        results: list[dict[str, Any]] = []
        can_spawn = bool(getattr(__main__, "__file__", None))
        if method == "parallel" and num_chains > 1 and can_spawn:
            max_workers = min(num_chains, max(1, os.cpu_count() or 1))
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futures = [
                        ex.submit(
                            _run_single_chain,
                            chain_idx=chain_idx,
                            continuation_state=(
                                None
                                if not self.chain_initial_states_ or chain_idx >= len(self.chain_initial_states_)
                                else self.chain_initial_states_[chain_idx]
                            ),
                            **worker_kwargs,
                        )
                        for chain_idx in range(num_chains)
                    ]
                    for fut in as_completed(futures):
                        results.append(fut.result())
            except Exception:
                results = []
                for chain_idx in range(num_chains):
                    results.append(
                        _run_single_chain(
                            chain_idx=chain_idx,
                            continuation_state=(
                                None
                                if not self.chain_initial_states_ or chain_idx >= len(self.chain_initial_states_)
                                else self.chain_initial_states_[chain_idx]
                            ),
                            **worker_kwargs,
                        )
                    )
        else:
            for chain_idx in range(num_chains):
                results.append(
                    _run_single_chain(
                        chain_idx=chain_idx,
                        continuation_state=(
                            None
                            if not self.chain_initial_states_ or chain_idx >= len(self.chain_initial_states_)
                            else self.chain_initial_states_[chain_idx]
                        ),
                        **worker_kwargs,
                    )
                )

        results.sort(key=lambda x: int(x["chain_idx"]))

        beta_last = np.stack([np.asarray(r["beta_last"], dtype=float) for r in results], axis=0)
        self.chain_elapsed_sec_ = np.asarray([float(r["elapsed_sec"]) for r in results], dtype=float)
        self.replica_exchange_accept_rate_ = np.asarray(
            [float(r.get("rex_accept_rate", 0.0)) for r in results],
            dtype=float,
        )
        self.chain_final_states_ = [dict(r.get("final_state", {})) for r in results]

        if self.store_beta_samples and kept > 0:
            beta_draws = np.stack([np.asarray(r["beta_draws"], dtype=float) for r in results], axis=0)
        else:
            beta_draws = beta_last[:, None, :]

        if kept > 0:
            pi_draws = np.stack([np.asarray(r["pi_draws"], dtype=float) for r in results], axis=0)
            sigma2_draws = np.stack([np.asarray(r["sigma2_draws"], dtype=float) for r in results], axis=0)
        else:
            pi_draws = None
            sigma2_draws = None

        flat_beta = beta_draws.reshape(-1, p)
        self.pos_mean_ = flat_beta.mean(axis=0)
        self.pos_median_ = np.median(flat_beta, axis=0)
        self.coef_ = self.pos_mean_.copy()
        self.coef_samples_ = beta_draws.copy()
        self.pi_samples_ = None if pi_draws is None else pi_draws.copy()
        self.sigma2_samples_ = None if sigma2_draws is None else sigma2_draws.copy()

        if self.fit_intercept:
            self.intercept_ = float(y_mean - x_mean @ self.coef_)
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: Any, **predict_kwargs: Any) -> np.ndarray:
        del predict_kwargs
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        X_arr = np.asarray(X, dtype=float)
        return X_arr @ self.coef_ + float(self.intercept_)

    def get_posterior_summaries(self) -> dict[str, np.ndarray | float]:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before requesting summaries.")
        out: dict[str, np.ndarray | float] = {
            "coef": self.coef_.copy(),
            "intercept": float(self.intercept_),
        }
        if self.pos_mean_ is not None:
            out["pos_mean"] = self.pos_mean_.copy()
        if self.pos_median_ is not None:
            out["pos_median"] = self.pos_median_.copy()
        if self.coef_samples_ is not None:
            flat = self.coef_samples_.reshape(-1, self.coef_samples_.shape[-1])
            out["coef_sd"] = flat.std(axis=0)
        if self.chain_elapsed_sec_ is not None:
            out["chain_elapsed_sec"] = self.chain_elapsed_sec_.copy()
        if self.replica_exchange_accept_rate_ is not None:
            out["rex_accept_rate"] = self.replica_exchange_accept_rate_.copy()
        return out
