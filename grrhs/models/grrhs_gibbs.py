from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.random import Generator, default_rng
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from grrhs.utils.logging_utils import progress

_EPS = 1e-12
_MIN_POS = 1e-10


def _sample_invgamma(alpha: float, beta: float, rng: Generator) -> float:
    """InvGamma(alpha, beta), shape-scale parameterization."""
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("InvGamma requires alpha > 0 and beta > 0.")
    z = rng.gamma(shape=alpha, scale=1.0 / beta)
    return float(1.0 / max(z, _MIN_POS))


def _fit_grrhs_chain_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    model = GRRHS_Gibbs(
        c=float(payload["c"]),
        tau0=float(payload["tau0"]),
        eta=float(payload["eta"]),
        s0=float(payload["s0"]),
        use_groups=bool(payload["use_groups"]),
        alpha_c=float(payload["alpha_c"]),
        beta_c=float(payload["beta_c"]),
        iters=int(payload["iters"]),
        burnin=int(payload["burnin"]),
        thin=int(payload["thin"]),
        seed=int(payload["seed"]),
        num_chains=1,
        slice_w=float(payload["slice_w"]),
        slice_m=int(payload["slice_m"]),
        tau_slice_w=float(payload["tau_slice_w"]),
        tau_slice_m=int(payload["tau_slice_m"]),
        jitter=float(payload["jitter"]),
        mh_sd_log_tau2=float(payload["mh_sd_log_tau2"]),
        mh_sd_log_lambda=float(payload["mh_sd_log_lambda"]),
        mh_sd_log_a=float(payload["mh_sd_log_a"]),
        mh_sd_log_c2=float(payload["mh_sd_log_c2"]),
    )
    fitted = model.fit(
        np.asarray(payload["X"], dtype=float),
        np.asarray(payload["y"], dtype=float),
        groups=payload["groups"],
    )
    return {
        "groups": fitted.groups_,
        "group_id": None if fitted.group_id_ is None else np.asarray(fitted.group_id_, dtype=int),
        "group_sizes": None if fitted.group_sizes_ is None else np.asarray(fitted.group_sizes_, dtype=int),
        "coef_samples": None if fitted.coef_samples_ is None else np.asarray(fitted.coef_samples_),
        "sigma2_samples": None if fitted.sigma2_samples_ is None else np.asarray(fitted.sigma2_samples_),
        "tau_samples": None if fitted.tau_samples_ is None else np.asarray(fitted.tau_samples_),
        "phi_samples": None if fitted.phi_samples_ is None else np.asarray(fitted.phi_samples_),
        "lambda_samples": None if fitted.lambda_samples_ is None else np.asarray(fitted.lambda_samples_),
        "a_samples": None if fitted.a_samples_ is None else np.asarray(fitted.a_samples_),
        "c2_samples": None if fitted.c2_samples_ is None else np.asarray(fitted.c2_samples_),
        "coef_mean": None if fitted.coef_mean_ is None else np.asarray(fitted.coef_mean_),
        "intercept": float(fitted.intercept_),
    }


@dataclass
class GRRHS_Gibbs:
    """
    Group Regularized Horseshoe (GR-RHS) Metropolis-within-Gibbs sampler.

    Hierarchy implemented:
      beta_j | lambda_j, a_g, c_g^2, tau ~ N(0, tau^2 * tilde_lambda_{j,g}^2)
      tilde_lambda_{j,g}^2 = c_g^2 * lambda_j^2 * a_g^2 / (c_g^2 + tau^2 * lambda_j^2 * a_g^2)
      lambda_j ~ HC(0, 1)
      a_g ~ HN(0, s_{a,g}^2), with s_{a,g} = eta / sqrt(p_g)
      c_g^2 ~ IG(alpha_c, beta_c)
      tau ~ HC(0, tau0), via IG augmentation with nu
      p(sigma^2) ∝ 1 / sigma^2
    """

    # Backward-compatible public hyperparameters
    c: float = 1.0
    tau0: float = 0.1
    eta: float = 0.5
    s0: float = 1.0
    use_groups: bool = True

    # New slab prior parameters
    alpha_c: float = 2.0
    beta_c: float = 2.0

    # Sampling controls
    iters: int = 2000
    burnin: Optional[int] = None
    thin: int = 1
    seed: int = 42
    num_chains: int = 1

    # Kept for compatibility with existing configs; mapped to MH step sizes
    slice_w: float = 0.25
    slice_m: int = 100
    tau_slice_w: float = 0.2
    tau_slice_m: int = 200
    jitter: float = 1e-10

    # Explicit MH proposal SDs in log space
    mh_sd_log_tau2: Optional[float] = None
    mh_sd_log_lambda: Optional[float] = None
    mh_sd_log_a: Optional[float] = None
    mh_sd_log_c2: Optional[float] = None

    rng: Generator = field(init=False)
    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    phi_samples_: Optional[np.ndarray] = field(default=None, init=False)  # alias of a_samples_
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    a_samples_: Optional[np.ndarray] = field(default=None, init=False)
    c2_samples_: Optional[np.ndarray] = field(default=None, init=False)

    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)

    groups_: Optional[List[List[int]]] = field(default=None, init=False)
    group_id_: Optional[np.ndarray] = field(default=None, init=False)
    group_sizes_: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.rng = default_rng(self.seed)
        if self.burnin is None:
            self.burnin = self.iters // 2
        if self.burnin < 0 or self.burnin >= self.iters:
            raise ValueError("burnin must satisfy 0 <= burnin < iters.")
        if self.thin <= 0:
            raise ValueError("thin must be positive.")
        if self.c <= 0.0:
            raise ValueError("c must be positive.")
        if self.tau0 <= 0.0:
            raise ValueError("tau0 must be positive.")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive.")
        if self.alpha_c <= 0.0 or self.beta_c <= 0.0:
            raise ValueError("alpha_c and beta_c must be positive.")
        if self.num_chains <= 0:
            raise ValueError("num_chains must be positive.")
        if self.jitter <= 0.0:
            raise ValueError("jitter must be positive.")

        if self.mh_sd_log_tau2 is None:
            self.mh_sd_log_tau2 = float(max(self.tau_slice_w, 1e-3))
        if self.mh_sd_log_lambda is None:
            self.mh_sd_log_lambda = float(max(self.slice_w, 1e-3))
        if self.mh_sd_log_a is None:
            self.mh_sd_log_a = float(max(self.slice_w, 1e-3))
        if self.mh_sd_log_c2 is None:
            self.mh_sd_log_c2 = float(max(self.slice_w, 1e-3))

    @staticmethod
    def _flatten_scalar_draws(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        data = np.asarray(arr, dtype=float)
        return data.reshape(-1)

    @staticmethod
    def _flatten_param_draws(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        data = np.asarray(arr, dtype=float)
        if data.ndim <= 1:
            return data.reshape(1, -1)
        if data.ndim == 2:
            return data
        return data.reshape(-1, *data.shape[2:])

    @staticmethod
    def _stack_chain_draws(arrays: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        if not arrays or arrays[0] is None:
            return None
        return np.stack([np.asarray(arr) for arr in arrays], axis=0)

    def _spawn_single_chain(self, *, seed: int) -> "GRRHS_Gibbs":
        return type(self)(
            c=self.c,
            tau0=self.tau0,
            eta=self.eta,
            s0=self.s0,
            use_groups=self.use_groups,
            alpha_c=self.alpha_c,
            beta_c=self.beta_c,
            iters=self.iters,
            burnin=self.burnin,
            thin=self.thin,
            seed=seed,
            num_chains=1,
            slice_w=self.slice_w,
            slice_m=self.slice_m,
            tau_slice_w=self.tau_slice_w,
            tau_slice_m=self.tau_slice_m,
            jitter=self.jitter,
            mh_sd_log_tau2=self.mh_sd_log_tau2,
            mh_sd_log_lambda=self.mh_sd_log_lambda,
            mh_sd_log_a=self.mh_sd_log_a,
            mh_sd_log_c2=self.mh_sd_log_c2,
        )

    def _fit_multichain(self, X: np.ndarray, y: np.ndarray, groups: Optional[List[List[int]]] = None) -> "GRRHS_Gibbs":
        payloads: List[Dict[str, Any]] = []
        groups_payload = None if groups is None else [list(group) for group in groups]
        for chain_idx in range(int(self.num_chains)):
            payloads.append(
                {
                    "c": self.c,
                    "tau0": self.tau0,
                    "eta": self.eta,
                    "s0": self.s0,
                    "use_groups": self.use_groups,
                    "alpha_c": self.alpha_c,
                    "beta_c": self.beta_c,
                    "iters": self.iters,
                    "burnin": self.burnin,
                    "thin": self.thin,
                    "seed": int(self.seed) + chain_idx,
                    "slice_w": self.slice_w,
                    "slice_m": self.slice_m,
                    "tau_slice_w": self.tau_slice_w,
                    "tau_slice_m": self.tau_slice_m,
                    "jitter": self.jitter,
                    "mh_sd_log_tau2": self.mh_sd_log_tau2,
                    "mh_sd_log_lambda": self.mh_sd_log_lambda,
                    "mh_sd_log_a": self.mh_sd_log_a,
                    "mh_sd_log_c2": self.mh_sd_log_c2,
                    "X": np.asarray(X, dtype=float),
                    "y": np.asarray(y, dtype=float),
                    "groups": groups_payload,
                }
            )

        try:
            with ProcessPoolExecutor(max_workers=int(self.num_chains)) as executor:
                chain_results = list(executor.map(_fit_grrhs_chain_task, payloads))
        except Exception:
            chain_results = [_fit_grrhs_chain_task(payload) for payload in payloads]

        lead = chain_results[0]
        self.groups_ = lead["groups"]
        self.group_id_ = None if lead["group_id"] is None else np.asarray(lead["group_id"], dtype=int).copy()
        self.group_sizes_ = None if lead["group_sizes"] is None else np.asarray(lead["group_sizes"], dtype=int).copy()
        self.coef_samples_ = self._stack_chain_draws([item["coef_samples"] for item in chain_results])
        self.sigma2_samples_ = self._stack_chain_draws([item["sigma2_samples"] for item in chain_results])
        self.tau_samples_ = self._stack_chain_draws([item["tau_samples"] for item in chain_results])
        self.lambda_samples_ = self._stack_chain_draws([item["lambda_samples"] for item in chain_results])
        self.a_samples_ = self._stack_chain_draws([item["a_samples"] for item in chain_results])
        self.c2_samples_ = self._stack_chain_draws([item["c2_samples"] for item in chain_results])
        self.phi_samples_ = self._stack_chain_draws([item["phi_samples"] for item in chain_results])
        coef_draws = self._flatten_param_draws(self.coef_samples_)
        self.coef_mean_ = None if coef_draws is None else coef_draws.mean(axis=0)
        self.intercept_ = float(lead["intercept"])
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[List[List[int]]] = None) -> "GRRHS_Gibbs":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.num_chains > 1:
            return self._fit_multichain(X, y, groups)

        n, p = X.shape
        if groups is None:
            groups = [[j] for j in range(p)]
        self.groups_ = groups

        if self.use_groups:
            sampler_groups = [list(g) for g in groups]
        else:
            sampler_groups = [list(range(p))]
        G = len(sampler_groups)

        group_id = np.empty(p, dtype=int)
        group_sizes = np.zeros(G, dtype=int)
        for g, idxs in enumerate(sampler_groups):
            idx_arr = np.asarray(idxs, dtype=int)
            group_id[idx_arr] = g
            group_sizes[g] = len(idx_arr)
        self.group_id_ = group_id
        self.group_sizes_ = group_sizes

        s_a = self.eta / np.sqrt(np.maximum(group_sizes.astype(float), 1.0))

        XtX = X.T @ X
        Xty = X.T @ y

        try:
            beta = np.linalg.solve(XtX + 1e-3 * np.eye(p), Xty)
        except np.linalg.LinAlgError:
            beta = np.zeros(p, dtype=float)

        resid_init = y - X @ beta
        sigma2 = max(float(np.mean(resid_init * resid_init)), self.jitter)
        tau2 = max(self.tau0 * self.tau0, self.jitter)
        nu = max(self.tau0 * self.tau0, self.jitter)

        lam = np.ones(p, dtype=float)
        a = np.maximum(s_a, 1e-3)
        c2 = np.full(G, max(self.c * self.c, 1e-3), dtype=float)

        kept = max(0, (self.iters - self.burnin + self.thin - 1) // self.thin)
        beta_draws = np.zeros((kept, p), dtype=float)
        sigma2_draws = np.zeros(kept, dtype=float)
        tau_draws = np.zeros(kept, dtype=float)
        lambda_draws = np.zeros((kept, p), dtype=float)
        a_draws = np.zeros((kept, G), dtype=float)
        c2_draws = np.zeros((kept, G), dtype=float)
        keep_i = 0

        tau_trace: List[float] = []
        sigma_trace: List[float] = []

        for it in progress(range(self.iters), total=self.iters, desc="GR-RHS MWG"):
            d = self._prior_precision_vector(lam=lam, a=a, c2=c2, tau2=tau2, group_id=group_id)
            beta = self._sample_beta_conditional(XtX=XtX, Xty=Xty, sigma2=sigma2, prior_prec=d, rng=self.rng)

            resid = y - X @ beta
            rss = float(resid @ resid)
            sigma2 = _sample_invgamma(alpha=0.5 * n, beta=0.5 * max(rss, self.jitter), rng=self.rng)

            nu = _sample_invgamma(alpha=1.0, beta=(1.0 / (self.tau0 * self.tau0)) + (1.0 / max(tau2, self.jitter)), rng=self.rng)

            tau2 = self._mh_update_tau2(
                tau2=tau2,
                nu=nu,
                beta=beta,
                lam=lam,
                a=a,
                c2=c2,
                group_id=group_id,
            )

            for j in range(p):
                g = group_id[j]
                lam[j] = self._mh_update_lambda_j(
                    lam_j=lam[j],
                    beta_j=beta[j],
                    tau2=tau2,
                    a_g=a[g],
                    c2_g=c2[g],
                )

            for g in range(G):
                idx = np.asarray(sampler_groups[g], dtype=int)
                a[g] = self._mh_update_a_g(
                    a_g=a[g],
                    beta_g=beta[idx],
                    lam_g=lam[idx],
                    tau2=tau2,
                    c2_g=c2[g],
                    s_a_g=s_a[g],
                )

            for g in range(G):
                idx = np.asarray(sampler_groups[g], dtype=int)
                c2[g] = self._mh_update_c2_g(
                    c2_g=c2[g],
                    beta_g=beta[idx],
                    lam_g=lam[idx],
                    tau2=tau2,
                    a_g=a[g],
                )

            tau_trace.append(math.sqrt(max(tau2, self.jitter)))
            sigma_trace.append(math.sqrt(max(sigma2, self.jitter)))

            if it >= self.burnin and ((it - self.burnin) % self.thin == 0):
                beta_draws[keep_i] = beta
                sigma2_draws[keep_i] = sigma2
                tau_draws[keep_i] = math.sqrt(max(tau2, self.jitter))
                lambda_draws[keep_i] = lam
                a_draws[keep_i] = a
                c2_draws[keep_i] = c2
                keep_i += 1

        self.coef_samples_ = beta_draws
        self.sigma2_samples_ = sigma2_draws
        self.tau_samples_ = tau_draws
        self.lambda_samples_ = lambda_draws
        self.a_samples_ = a_draws
        self.c2_samples_ = c2_draws
        self.phi_samples_ = a_draws
        self.coef_mean_ = beta_draws.mean(axis=0) if kept > 0 else np.zeros(p, dtype=float)
        self.intercept_ = 0.0

        if tau_trace:
            logger = logging.getLogger(__name__)
            logger.info("tau_trace=%s", np.asarray(tau_trace))
            logger.info("sigma_trace=%s", np.asarray(sigma_trace))

        return self

    def predict(self, X: np.ndarray, use_posterior_mean: bool = True) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if use_posterior_mean:
            if self.coef_mean_ is None:
                raise RuntimeError("Model not fitted.")
            return X @ self.coef_mean_ + self.intercept_
        coef_draws = self._flatten_param_draws(self.coef_samples_)
        if coef_draws is None or coef_draws.shape[0] == 0:
            raise RuntimeError("No posterior samples available.")
        return X @ coef_draws[-1] + self.intercept_

    def get_posterior_summaries(self) -> Dict[str, Any]:
        if self.coef_samples_ is None:
            raise RuntimeError("Not fitted.")
        beta_draws = self._flatten_param_draws(self.coef_samples_)
        sigma2_draws = self._flatten_scalar_draws(self.sigma2_samples_)
        tau_draws = self._flatten_scalar_draws(self.tau_samples_)
        a_draws = self._flatten_param_draws(self.a_samples_)
        c2_draws = self._flatten_param_draws(self.c2_samples_)
        if beta_draws is None:
            raise RuntimeError("Posterior coefficient draws are unavailable.")
        return {
            "beta_mean": beta_draws.mean(axis=0),
            "beta_median": np.median(beta_draws, axis=0),
            "beta_ci95": np.quantile(beta_draws, [0.025, 0.975], axis=0),
            "sigma2_mean": float(sigma2_draws.mean()) if sigma2_draws is not None else None,
            "tau_mean": float(tau_draws.mean()) if tau_draws is not None else None,
            "phi_mean": a_draws.mean(axis=0) if a_draws is not None else None,
            "a_mean": a_draws.mean(axis=0) if a_draws is not None else None,
            "c2_mean": c2_draws.mean(axis=0) if c2_draws is not None else None,
        }

    def _prior_precision_vector(
        self,
        lam: np.ndarray,
        a: np.ndarray,
        c2: np.ndarray,
        tau2: float,
        group_id: np.ndarray,
    ) -> np.ndarray:
        a_map = a[group_id]
        c2_map = c2[group_id]
        tau2_safe = max(float(tau2), self.jitter)
        local = 1.0 / np.maximum(tau2_safe * (lam * lam) * (a_map * a_map), self.jitter)
        slab = 1.0 / np.maximum(c2_map, self.jitter)
        return local + slab

    def _beta_logprior_contrib(self, beta: np.ndarray, d: np.ndarray) -> float:
        return float(0.5 * np.log(np.maximum(d, self.jitter)).sum() - 0.5 * (d * beta * beta).sum())

    def _mh_accept(self, log_target_new: float, log_target_old: float) -> bool:
        log_alpha = log_target_new - log_target_old
        if log_alpha >= 0.0:
            return True
        u = max(self.rng.uniform(), _MIN_POS)
        return math.log(u) < log_alpha

    def _mh_update_tau2(
        self,
        tau2: float,
        nu: float,
        beta: np.ndarray,
        lam: np.ndarray,
        a: np.ndarray,
        c2: np.ndarray,
        group_id: np.ndarray,
    ) -> float:
        log_tau2 = math.log(max(tau2, self.jitter))
        log_prop = log_tau2 + self.rng.normal(scale=float(self.mh_sd_log_tau2))
        tau2_prop = max(math.exp(log_prop), self.jitter)

        d_old = self._prior_precision_vector(lam=lam, a=a, c2=c2, tau2=tau2, group_id=group_id)
        d_new = self._prior_precision_vector(lam=lam, a=a, c2=c2, tau2=tau2_prop, group_id=group_id)

        lp_old = self._beta_logprior_contrib(beta=beta, d=d_old) + self._log_prior_tau2_given_nu(tau2=tau2, nu=nu) + log_tau2
        lp_new = self._beta_logprior_contrib(beta=beta, d=d_new) + self._log_prior_tau2_given_nu(tau2=tau2_prop, nu=nu) + log_prop
        return tau2_prop if self._mh_accept(lp_new, lp_old) else tau2

    def _mh_update_lambda_j(
        self,
        lam_j: float,
        beta_j: float,
        tau2: float,
        a_g: float,
        c2_g: float,
    ) -> float:
        log_old = math.log(max(lam_j, self.jitter))
        log_new = log_old + self.rng.normal(scale=float(self.mh_sd_log_lambda))
        lam_new = max(math.exp(log_new), self.jitter)

        d_old = self._single_precision(lam=lam_j, a_g=a_g, c2_g=c2_g, tau2=tau2)
        d_new = self._single_precision(lam=lam_new, a_g=a_g, c2_g=c2_g, tau2=tau2)

        lp_old = 0.5 * math.log(d_old) - 0.5 * d_old * (beta_j * beta_j) + self._log_prior_lambda(lam_j) + log_old
        lp_new = 0.5 * math.log(d_new) - 0.5 * d_new * (beta_j * beta_j) + self._log_prior_lambda(lam_new) + log_new
        return lam_new if self._mh_accept(lp_new, lp_old) else lam_j

    def _mh_update_a_g(
        self,
        a_g: float,
        beta_g: np.ndarray,
        lam_g: np.ndarray,
        tau2: float,
        c2_g: float,
        s_a_g: float,
    ) -> float:
        log_old = math.log(max(a_g, self.jitter))
        log_new = log_old + self.rng.normal(scale=float(self.mh_sd_log_a))
        a_new = max(math.exp(log_new), self.jitter)

        d_old = self._group_precision(lam_g=lam_g, a_g=a_g, c2_g=c2_g, tau2=tau2)
        d_new = self._group_precision(lam_g=lam_g, a_g=a_new, c2_g=c2_g, tau2=tau2)

        lp_old = self._beta_logprior_contrib(beta=beta_g, d=d_old) + self._log_prior_a(a=a_g, s_a=s_a_g) + log_old
        lp_new = self._beta_logprior_contrib(beta=beta_g, d=d_new) + self._log_prior_a(a=a_new, s_a=s_a_g) + log_new
        return a_new if self._mh_accept(lp_new, lp_old) else a_g

    def _mh_update_c2_g(
        self,
        c2_g: float,
        beta_g: np.ndarray,
        lam_g: np.ndarray,
        tau2: float,
        a_g: float,
    ) -> float:
        log_old = math.log(max(c2_g, self.jitter))
        log_new = log_old + self.rng.normal(scale=float(self.mh_sd_log_c2))
        c2_new = max(math.exp(log_new), self.jitter)

        d_old = self._group_precision(lam_g=lam_g, a_g=a_g, c2_g=c2_g, tau2=tau2)
        d_new = self._group_precision(lam_g=lam_g, a_g=a_g, c2_g=c2_new, tau2=tau2)

        lp_old = self._beta_logprior_contrib(beta=beta_g, d=d_old) + self._log_prior_c2(c2_g) + log_old
        lp_new = self._beta_logprior_contrib(beta=beta_g, d=d_new) + self._log_prior_c2(c2_new) + log_new
        return c2_new if self._mh_accept(lp_new, lp_old) else c2_g

    def _log_prior_tau2_given_nu(self, tau2: float, nu: float) -> float:
        t = max(tau2, self.jitter)
        n = max(nu, self.jitter)
        alpha = 0.5
        beta = 1.0 / n
        return alpha * math.log(beta) - math.lgamma(alpha) - (alpha + 1.0) * math.log(t) - beta / t

    @staticmethod
    def _log_prior_lambda(lam: float) -> float:
        return -math.log1p(lam * lam)

    @staticmethod
    def _log_prior_a(a: float, s_a: float) -> float:
        ss = max(s_a, _MIN_POS)
        return -0.5 * (a * a) / (ss * ss)

    def _log_prior_c2(self, c2: float) -> float:
        x = max(c2, self.jitter)
        return (
            self.alpha_c * math.log(self.beta_c)
            - math.lgamma(self.alpha_c)
            - (self.alpha_c + 1.0) * math.log(x)
            - self.beta_c / x
        )

    def _single_precision(self, lam: float, a_g: float, c2_g: float, tau2: float) -> float:
        local = 1.0 / max(tau2 * lam * lam * a_g * a_g, self.jitter)
        slab = 1.0 / max(c2_g, self.jitter)
        return local + slab

    def _group_precision(self, lam_g: np.ndarray, a_g: float, c2_g: float, tau2: float) -> np.ndarray:
        local = 1.0 / np.maximum(tau2 * (lam_g * lam_g) * (a_g * a_g), self.jitter)
        slab = 1.0 / max(c2_g, self.jitter)
        return local + slab

    def _sample_beta_conditional(
        self,
        XtX: np.ndarray,
        Xty: np.ndarray,
        sigma2: float,
        prior_prec: np.ndarray,
        rng: Generator,
    ) -> np.ndarray:
        p = XtX.shape[0]
        sigma2_safe = max(float(sigma2), self.jitter)
        precision = (XtX / sigma2_safe).copy()
        precision.flat[:: p + 1] += np.maximum(prior_prec, self.jitter)

        chol, lower = cho_factor(precision, overwrite_a=False, check_finite=False)
        mean = cho_solve((chol, lower), Xty / sigma2_safe, check_finite=False)

        z = rng.standard_normal(p)
        if lower:
            noise = solve_triangular(chol.T, z, lower=False, check_finite=False)
        else:
            noise = solve_triangular(chol, z, lower=False, check_finite=False)
        return mean + noise
