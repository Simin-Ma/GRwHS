from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp
from jax import random
from jax.nn import sigmoid
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary as diagnostics_summary
from numpyro.infer import MCMC, NUTS
from scipy.special import betaln

_EPS = 1e-12


def _split_groups(p: int, groups: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    g = len(groups)
    gid = np.empty(p, dtype=np.int32)
    gsz = np.zeros(g, dtype=np.int32)
    for k, idxs in enumerate(groups):
        gid[np.asarray(idxs, dtype=np.int32)] = k
        gsz[k] = len(idxs)
    return gid, gsz


def _thin(arr: Optional[np.ndarray], step: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if arr.ndim == 0:
        return arr
    if arr.ndim == 1:
        return arr if step <= 1 else arr[::step]
    out = arr if step <= 1 else arr[:, ::step, ...]
    if out.shape[0] == 1:
        return out[0]
    return out


def _flatten_scalar(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    return np.asarray(arr, dtype=float).reshape(-1)


def _flatten_param(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    data = np.asarray(arr, dtype=float)
    if data.ndim == 1:
        return data.reshape(1, -1)
    if data.ndim == 2:
        return data
    return data.reshape(-1, *data.shape[2:])


@dataclass
class GRRHS_NUTS:
    """
    GR-RHS reference implementation:
    primitive hierarchy with transformed/non-centered NUTS.
    """

    tau0: Optional[float] = None
    eta: float = 0.5
    s0: float = 1.0
    alpha_kappa: float = 2.0
    beta_kappa: float = 8.0
    likelihood: str = "gaussian"
    use_group_scale: bool = True
    shared_kappa: bool = False
    auto_calibrate_tau: bool = True
    tau_target: str = "coefficients"  # coefficients | groups
    p0: Optional[float] = None
    sigma_reference: float = 1.0
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    thinning: int = 1
    target_accept_prob: float = 0.95
    max_tree_depth: int = 12
    dense_mass: bool = True
    chain_method: str = "sequential"
    progress_bar: bool = False
    seed: int = 42

    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    a_samples_: Optional[np.ndarray] = field(default=None, init=False)
    kappa_samples_: Optional[np.ndarray] = field(default=None, init=False)
    c2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    phi_samples_: Optional[np.ndarray] = field(default=None, init=False)  # alias of a_samples_
    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_mean_: Optional[float] = field(default=None, init=False)
    tau_mean_: Optional[float] = field(default=None, init=False)
    lambda_mean_: Optional[np.ndarray] = field(default=None, init=False)
    a_mean_: Optional[np.ndarray] = field(default=None, init=False)
    kappa_mean_: Optional[np.ndarray] = field(default=None, init=False)
    c2_mean_: Optional[np.ndarray] = field(default=None, init=False)
    phi_mean_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)
    groups_: Optional[List[List[int]]] = field(default=None, init=False)
    group_id_: Optional[np.ndarray] = field(default=None, init=False)
    group_sizes_: Optional[np.ndarray] = field(default=None, init=False)
    sampler_diagnostics_: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.eta <= 0.0:
            raise ValueError("eta must be positive.")
        if self.s0 <= 0.0:
            raise ValueError("s0 must be positive.")
        lik = str(self.likelihood).strip().lower()
        if lik in {"gaussian", "regression"}:
            self.likelihood = "gaussian"
        elif lik in {"logistic", "classification"}:
            self.likelihood = "logistic"
        else:
            raise ValueError("likelihood must be 'gaussian' or 'logistic'.")
        if self.tau0 is not None and float(self.tau0) <= 0.0:
            raise ValueError("tau0 must be positive when provided.")
        if float(self.alpha_kappa) <= 0.0 or float(self.beta_kappa) <= 0.0:
            raise ValueError("alpha_kappa and beta_kappa must be positive.")
        if self.num_warmup < 0 or self.num_samples <= 0:
            raise ValueError("num_warmup must be >=0 and num_samples must be >0.")
        if self.num_chains <= 0:
            raise ValueError("num_chains must be positive.")
        if self.thinning <= 0:
            raise ValueError("thinning must be positive.")
        if not (0.5 <= float(self.target_accept_prob) < 1.0):
            raise ValueError("target_accept_prob must be in [0.5, 1.0).")
        if int(self.max_tree_depth) <= 0:
            raise ValueError("max_tree_depth must be positive.")

    @staticmethod
    def calibrate_tau0(*, p0: float, D: int, n: int, sigma_ref: float = 1.0) -> float:
        p0_use = max(float(p0), 1.0)
        denom = max(float(D) - p0_use, 1e-8)
        return float(max((p0_use / denom) * (float(sigma_ref) / math.sqrt(max(int(n), 1))), 1e-8))

    @staticmethod
    def _log_half_cauchy_on_log(log_x: jnp.ndarray, scale: float) -> jnp.ndarray:
        x = jnp.exp(log_x)
        return jnp.log(2.0 / jnp.pi) - jnp.log(scale) - jnp.log1p((x / scale) ** 2) + log_x

    @staticmethod
    def _log_half_normal_on_log(log_x: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        x = jnp.exp(log_x)
        return 0.5 * jnp.log(2.0 / jnp.pi) - jnp.log(scale) - 0.5 * (x / scale) ** 2 + log_x

    @staticmethod
    def _log_beta_on_logit(logit_x: jnp.ndarray, alpha: float, beta: float) -> jnp.ndarray:
        x = jnp.clip(jnp.asarray(sigmoid(logit_x)), _EPS, 1.0 - _EPS)
        return (
            (alpha - 1.0) * jnp.log(x)
            + (beta - 1.0) * jnp.log1p(-x)
            - float(betaln(alpha, beta))
            + jnp.log(x)
            + jnp.log1p(-x)
        )

    def _model(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        group_id: jnp.ndarray,
        group_sizes: jnp.ndarray,
        tau0_eff: float,
    ) -> None:
        n, p = X.shape
        G = int(group_sizes.shape[0])

        if self.likelihood == "gaussian":
            log_sigma = numpyro.sample("log_sigma_raw", dist.Normal(0.0, 1.0))
            sigma = numpyro.deterministic("sigma", jnp.exp(log_sigma))
            numpyro.factor(
                "prior_log_sigma",
                self._log_half_cauchy_on_log(log_sigma, self.s0) - dist.Normal(0.0, 1.0).log_prob(log_sigma),
            )
        else:
            sigma = numpyro.deterministic("sigma", jnp.asarray(1.0, dtype=X.dtype))

        log_tau = numpyro.sample("log_tau_raw", dist.Normal(0.0, 1.0))
        tau = numpyro.deterministic("tau", jnp.exp(log_tau))
        numpyro.factor(
            "prior_log_tau",
            self._log_half_cauchy_on_log(log_tau, tau0_eff) - dist.Normal(0.0, 1.0).log_prob(log_tau),
        )

        log_lambda = numpyro.sample("log_lambda_raw", dist.Normal(jnp.zeros((p,)), jnp.ones((p,))).to_event(1))
        lam = numpyro.deterministic("lambda", jnp.exp(log_lambda))
        numpyro.factor(
            "prior_log_lambda",
            jnp.sum(self._log_half_cauchy_on_log(log_lambda, 1.0) - dist.Normal(0.0, 1.0).log_prob(log_lambda)),
        )

        s_a = self.eta / jnp.sqrt(jnp.maximum(group_sizes.astype(X.dtype), 1.0))
        if self.use_group_scale:
            log_a = numpyro.sample("log_a_raw", dist.Normal(jnp.zeros((G,)), jnp.ones((G,))).to_event(1))
            a = numpyro.deterministic("a", jnp.exp(log_a))
            numpyro.factor(
                "prior_log_a",
                jnp.sum(self._log_half_normal_on_log(log_a, s_a) - dist.Normal(0.0, 1.0).log_prob(log_a)),
            )
        else:
            a = numpyro.deterministic("a", jnp.ones((G,), dtype=X.dtype))

        if self.shared_kappa:
            logit_kappa_raw = numpyro.sample("logit_kappa_shared_raw", dist.Normal(0.0, 1.0))
            kappa_shared = sigmoid(logit_kappa_raw)
            kappa = numpyro.deterministic("kappa", jnp.full((G,), kappa_shared, dtype=X.dtype))
            numpyro.factor(
                "prior_logit_kappa",
                self._log_beta_on_logit(logit_kappa_raw, self.alpha_kappa, self.beta_kappa)
                - dist.Normal(0.0, 1.0).log_prob(logit_kappa_raw),
            )
        else:
            logit_kappa = numpyro.sample(
                "logit_kappa_raw",
                dist.Normal(jnp.zeros((G,)), jnp.ones((G,))).to_event(1),
            )
            kappa = numpyro.deterministic("kappa", sigmoid(logit_kappa))
            numpyro.factor(
                "prior_logit_kappa",
                jnp.sum(
                    self._log_beta_on_logit(logit_kappa, self.alpha_kappa, self.beta_kappa)
                    - dist.Normal(0.0, 1.0).log_prob(logit_kappa)
                ),
            )

        sigma2 = sigma * sigma
        c2 = numpyro.deterministic("c2", sigma2 * kappa / (1.0 - kappa + _EPS))

        a_j = a[group_id]
        kappa_j = kappa[group_id]
        lam2 = lam * lam
        a2 = a_j * a_j
        tau2 = tau * tau
        num = sigma2 * kappa_j * tau2 * lam2 * a2
        den = sigma2 * kappa_j + (1.0 - kappa_j) * tau2 * lam2 * a2 + _EPS
        beta_scale = jnp.sqrt(jnp.maximum(num / den, _EPS))

        beta_raw = numpyro.sample("beta_raw", dist.Normal(jnp.zeros((p,)), jnp.ones((p,))).to_event(1))
        beta = numpyro.deterministic("beta", beta_raw * beta_scale)

        mean = X @ beta
        with numpyro.plate("data", n):
            if self.likelihood == "gaussian":
                numpyro.sample("y", dist.Normal(mean, sigma), obs=y)
            else:
                numpyro.sample("y", dist.Bernoulli(logits=mean), obs=y)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[List[List[int]]] = None,
    ) -> "GRRHS_NUTS":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D.")
        if y_arr.ndim != 1 or y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("y must be a 1D vector aligned with X.")

        n, p = X_arr.shape
        if groups is None:
            groups = [[j] for j in range(p)]
        self.groups_ = [list(map(int, g)) for g in groups]
        gid, gsz = _split_groups(p, self.groups_)
        self.group_id_ = gid
        self.group_sizes_ = gsz

        if self.tau0 is not None:
            tau0_eff = float(self.tau0)
        elif self.auto_calibrate_tau:
            target_dim = p if str(self.tau_target).strip().lower() == "coefficients" else len(self.groups_)
            p0_use = float(self.p0) if self.p0 is not None else float(max(1, min(20, target_dim // 4)))
            sigma_ref = float(self.sigma_reference)
            if self.likelihood == "logistic" and sigma_ref <= 0.0:
                sigma_ref = 2.0
            tau0_eff = self.calibrate_tau0(
                p0=p0_use,
                D=target_dim,
                n=n,
                sigma_ref=sigma_ref,
            )
        else:
            tau0_eff = 0.1

        kernel = NUTS(
            self._model,
            target_accept_prob=float(self.target_accept_prob),
            dense_mass=bool(self.dense_mass),
            max_tree_depth=int(self.max_tree_depth),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=int(self.num_warmup),
            num_samples=int(self.num_samples),
            num_chains=int(self.num_chains),
            chain_method=str(self.chain_method),
            progress_bar=bool(self.progress_bar),
        )

        start = time.perf_counter()
        mcmc.run(
            random.PRNGKey(int(self.seed)),
            jnp.asarray(X_arr),
            jnp.asarray(y_arr),
            jnp.asarray(gid),
            jnp.asarray(gsz),
            float(tau0_eff),
            extra_fields=("diverging", "energy", "num_steps"),
        )
        runtime_sec = max(time.perf_counter() - start, 1e-12)
        samples = mcmc.get_samples(group_by_chain=True)
        self._store_samples(samples)
        self.sampler_diagnostics_ = self._extract_diagnostics(mcmc, runtime_sec=runtime_sec)
        self.sampler_diagnostics_["parameterization"] = {
            "primitive_hierarchy": True,
            "transformed_variables": ["log_tau", "log_sigma", "log_lambda", "log_a", "logit_kappa"],
            "non_centered_beta": True,
            "likelihood": str(self.likelihood),
            "use_group_scale": bool(self.use_group_scale),
            "shared_kappa": bool(self.shared_kappa),
            "tau0_effective": float(tau0_eff),
        }
        return self

    def _store_samples(self, samples: Dict[str, jnp.ndarray]) -> None:
        def get(name: str) -> Optional[np.ndarray]:
            if name not in samples:
                return None
            return _thin(np.asarray(samples[name], dtype=np.float64), int(self.thinning))

        self.coef_samples_ = get("beta")
        self.sigma_samples_ = get("sigma")
        self.tau_samples_ = get("tau")
        self.lambda_samples_ = get("lambda")
        self.a_samples_ = get("a")
        self.kappa_samples_ = get("kappa")
        self.c2_samples_ = get("c2")

        self.phi_samples_ = self.a_samples_

        coef_draws = _flatten_param(self.coef_samples_)
        self.coef_mean_ = None if coef_draws is None else coef_draws.mean(axis=0)
        sigma_draws = _flatten_scalar(self.sigma_samples_)
        tau_draws = _flatten_scalar(self.tau_samples_)
        lam_draws = _flatten_param(self.lambda_samples_)
        a_draws = _flatten_param(self.a_samples_)
        kappa_draws = _flatten_param(self.kappa_samples_)
        c2_draws = _flatten_param(self.c2_samples_)
        self.sigma_mean_ = None if sigma_draws is None else float(sigma_draws.mean())
        self.tau_mean_ = None if tau_draws is None else float(tau_draws.mean())
        self.lambda_mean_ = None if lam_draws is None else lam_draws.mean(axis=0)
        self.a_mean_ = None if a_draws is None else a_draws.mean(axis=0)
        self.kappa_mean_ = None if kappa_draws is None else kappa_draws.mean(axis=0)
        self.c2_mean_ = None if c2_draws is None else c2_draws.mean(axis=0)
        self.phi_mean_ = self.a_mean_
        self.intercept_ = 0.0

    def _extract_diagnostics(self, mcmc: MCMC, *, runtime_sec: float) -> Dict[str, Any]:
        out: Dict[str, Any] = {"backend": "numpyro_nuts", "runtime_sec": float(runtime_sec)}
        try:
            extra = mcmc.get_extra_fields(group_by_chain=True)
        except Exception:
            extra = {}

        diverging_raw = np.asarray(extra.get("diverging", []), dtype=float)
        num_steps_raw = np.asarray(extra.get("num_steps", []), dtype=float)
        energy_raw = np.asarray(extra.get("energy", []), dtype=float)
        if diverging_raw.ndim == 1 and diverging_raw.size:
            diverging_raw = diverging_raw.reshape(1, -1)
        if num_steps_raw.ndim == 1 and num_steps_raw.size:
            num_steps_raw = num_steps_raw.reshape(1, -1)
        if energy_raw.ndim == 1 and energy_raw.size:
            energy_raw = energy_raw.reshape(1, -1)

        divergences = int(np.sum(diverging_raw > 0.5)) if diverging_raw.size else -1
        if num_steps_raw.size:
            treedepth_limit = float(2 ** int(self.max_tree_depth))
            treedepth_hits = int(np.sum(num_steps_raw >= treedepth_limit))
            max_num_steps = int(np.max(num_steps_raw))
        else:
            treedepth_hits = -1
            max_num_steps = -1

        ebfmi_vals: list[float] = []
        if energy_raw.size:
            for chain_energy in energy_raw:
                if chain_energy.size < 3:
                    ebfmi_vals.append(float("nan"))
                    continue
                num = float(np.mean(np.diff(chain_energy) ** 2))
                den = float(np.var(chain_energy))
                ebfmi_vals.append(float("nan") if den <= 0.0 or not np.isfinite(den) else num / den)
        ebfmi_min = float(np.nanmin(ebfmi_vals)) if ebfmi_vals else float("nan")
        out["hmc"] = {
            "divergences": int(divergences),
            "treedepth_hits": int(treedepth_hits),
            "max_num_steps": int(max_num_steps),
            "ebfmi_per_chain": ebfmi_vals,
            "ebfmi_min": ebfmi_min,
        }

        try:
            samples = mcmc.get_samples(group_by_chain=True)
            diag = diagnostics_summary(samples, group_by_chain=True)
            blocks = ["beta", "tau", "a", "kappa", "lambda"]
            rhat_vals: list[float] = []
            ess_vals: list[float] = []
            for b in blocks:
                if b not in diag:
                    continue
                entry = diag[b]
                if isinstance(entry, dict):
                    if "r_hat" in entry:
                        rhat_vals.extend(np.asarray(entry["r_hat"], dtype=float).reshape(-1).tolist())
                    if "n_eff" in entry:
                        ess_vals.extend(np.asarray(entry["n_eff"], dtype=float).reshape(-1).tolist())
            rhat_vals = [float(v) for v in rhat_vals if np.isfinite(v)]
            ess_vals = [float(v) for v in ess_vals if np.isfinite(v)]
            min_ess = float(min(ess_vals)) if ess_vals else float("nan")
            out["posterior_quality"] = {
                "max_rhat": float(max(rhat_vals)) if rhat_vals else float("nan"),
                "min_ess": min_ess,
                "ess_per_sec": float(min_ess / runtime_sec) if np.isfinite(min_ess) else float("nan"),
            }
        except Exception:
            out["posterior_quality"] = {}
        return out

    def predict(self, X: np.ndarray, use_posterior_mean: bool = True) -> np.ndarray:
        if self.coef_mean_ is None:
            raise RuntimeError("Model not fitted.")
        X_arr = np.asarray(X, dtype=np.float32)
        if use_posterior_mean:
            linear = X_arr @ np.asarray(self.coef_mean_, dtype=float) + float(self.intercept_)
            if self.likelihood == "logistic":
                linear = np.clip(linear, -60.0, 60.0)
                return 1.0 / (1.0 + np.exp(-linear))
            return linear
        if self.coef_samples_ is None:
            raise RuntimeError("No posterior samples available.")
        last = np.asarray(self.coef_samples_)[-1]
        linear = X_arr @ np.asarray(last, dtype=float) + float(self.intercept_)
        if self.likelihood == "logistic":
            linear = np.clip(linear, -60.0, 60.0)
            return 1.0 / (1.0 + np.exp(-linear))
        return linear

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.likelihood != "logistic":
            raise RuntimeError("predict_proba is only available for logistic likelihood.")
        prob = np.asarray(self.predict(X, use_posterior_mean=True), dtype=float).reshape(-1)
        return np.column_stack([1.0 - prob, prob])

    def get_posterior_summaries(self) -> Dict[str, Any]:
        if self.coef_samples_ is None:
            raise RuntimeError("Model not fitted.")
        coef = _flatten_param(self.coef_samples_)
        if coef is None:
            raise RuntimeError("Coefficient draws are unavailable.")
        out: Dict[str, Any] = {
            "beta_mean": coef.mean(axis=0),
            "beta_median": np.median(coef, axis=0),
            "beta_ci95": np.quantile(coef, [0.025, 0.975], axis=0),
            "sigma_mean": self.sigma_mean_,
            "tau_mean": self.tau_mean_,
            "a_mean": self.a_mean_,
            "kappa_mean": self.kappa_mean_,
            "c2_mean": self.c2_mean_,
            "phi_mean": self.phi_mean_,
            "lambda_mean": self.lambda_mean_,
        }
        return out


GRRHS_HMC = GRRHS_NUTS
