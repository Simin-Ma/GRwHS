from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp
from jax import random
from jax.nn import sigmoid
import numpy as np
from numpy.random import Generator, default_rng
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary as diagnostics_summary
from numpyro.infer import MCMC, NUTS
from scipy.special import betaln

from simulation_project.src.core.inference.samplers import slice_sample_1d
from simulation_project.src.core.inference.woodbury import beta_sample_woodbury, beta_sample_cholesky

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


def _normalize_init_params(
    init_params: Optional[Dict[str, Any]],
    *,
    num_chains: int,
) -> Optional[Dict[str, jnp.ndarray]]:
    if not isinstance(init_params, dict) or not init_params:
        return None
    out: Dict[str, jnp.ndarray] = {}
    chains = max(1, int(num_chains))
    for key, value in init_params.items():
        arr = np.asarray(value, dtype=np.float32)
        if chains == 1:
            if arr.ndim > 0 and arr.shape[0] == 1:
                arr = arr[0]
        else:
            if arr.ndim == 0:
                arr = np.repeat(arr.reshape(1), chains, axis=0)
            elif arr.shape[0] != chains:
                arr = np.repeat(np.expand_dims(arr, axis=0), chains, axis=0)
        out[str(key)] = jnp.asarray(arr)
    return out or None


def _extract_last_init_params(
    samples: Dict[str, jnp.ndarray],
    *,
    latent_keys: Sequence[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key in latent_keys:
        if key not in samples:
            continue
        arr = np.asarray(samples[key], dtype=np.float32)
        if arr.ndim < 2:
            continue
        out[str(key)] = np.asarray(arr[:, -1, ...], dtype=np.float32)
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
    use_local_scale: bool = True
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
    dense_mass: bool = False
    use_init_to_median: bool = True
    chain_method: str = "sequential"
    progress_bar: bool = True
    seed: int = 42
    init_params: Optional[Dict[str, Any]] = None
    resume_no_warmup: bool = False

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
    last_init_params_: Optional[Dict[str, np.ndarray]] = field(default=None, init=False)

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
            sigma = numpyro.sample("sigma", dist.HalfCauchy(jnp.asarray(self.s0, dtype=X.dtype)))
        else:
            sigma = numpyro.deterministic("sigma", jnp.asarray(1.0, dtype=X.dtype))

        tau_scale = jnp.maximum(jnp.asarray(tau0_eff, dtype=X.dtype) * sigma, _EPS)
        tau = numpyro.sample("tau", dist.HalfCauchy(tau_scale))

        if self.use_local_scale:
            lam = numpyro.sample("lambda", dist.HalfCauchy(jnp.ones((p,), dtype=X.dtype)).to_event(1))
        else:
            lam = numpyro.deterministic("lambda", jnp.ones((p,), dtype=X.dtype))

        s_a = self.eta / jnp.sqrt(jnp.maximum(group_sizes.astype(X.dtype), 1.0))
        if self.use_group_scale:
            a = numpyro.sample("a", dist.HalfNormal(s_a).to_event(1))
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
        numpyro.deterministic("c2", sigma2 * kappa / (1.0 - kappa + _EPS))

        a_j = a[group_id]
        kappa_j = kappa[group_id]
        tau2 = tau * tau
        lam2 = lam * lam
        a2_j = a_j * a_j
        # Dimensionless ratio r_j = tau^2 * lambda_j^2 * a_g(j)^2 / sigma^2
        # keeps each factor O(1) and avoids overflow in products of scales.
        # v_j = sigma^2 * kappa_g(j) * r_j / (kappa_g(j) + (1-kappa_g(j)) * r_j)
        r = tau2 * lam2 * a2_j / (sigma2 + _EPS)
        beta_var = sigma2 * kappa_j * r / (kappa_j + (1.0 - kappa_j) * r + _EPS)
        beta_scale = jnp.sqrt(jnp.maximum(beta_var, _EPS))

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

        init_strategy = (
            numpyro.infer.init_to_median(num_samples=15)
            if bool(self.use_init_to_median)
            else numpyro.infer.init_to_uniform()
        )
        kernel = NUTS(
            self._model,
            target_accept_prob=float(self.target_accept_prob),
            dense_mass=bool(self.dense_mass),
            max_tree_depth=int(self.max_tree_depth),
            init_strategy=init_strategy,
        )
        init_params_use = _normalize_init_params(
            self.init_params,
            num_chains=int(self.num_chains),
        )
        warmup_use = int(self.num_warmup)
        if bool(self.resume_no_warmup) and init_params_use is not None:
            warmup_use = 0
        mcmc = MCMC(
            kernel,
            num_warmup=int(warmup_use),
            num_samples=int(self.num_samples),
            num_chains=int(self.num_chains),
            chain_method=str(self.chain_method),
            progress_bar=bool(self.progress_bar),
        )

        start = time.perf_counter()
        run_kwargs: Dict[str, Any] = {}
        if init_params_use is not None:
            run_kwargs["init_params"] = init_params_use
        mcmc.run(
            random.PRNGKey(int(self.seed)),
            jnp.asarray(X_arr),
            jnp.asarray(y_arr),
            jnp.asarray(gid),
            jnp.asarray(gsz),
            float(tau0_eff),
            extra_fields=("diverging", "energy", "num_steps"),
            **run_kwargs,
        )
        runtime_sec = max(time.perf_counter() - start, 1e-12)
        samples = mcmc.get_samples(group_by_chain=True)
        latent_keys = ["sigma", "tau", "beta_raw"]
        if bool(self.use_local_scale):
            latent_keys.append("lambda")
        if bool(self.use_group_scale):
            latent_keys.append("a")
        if bool(self.shared_kappa):
            latent_keys.append("logit_kappa_shared_raw")
        else:
            latent_keys.append("logit_kappa_raw")
        self.last_init_params_ = _extract_last_init_params(samples, latent_keys=latent_keys)
        self._store_samples(samples)
        self.sampler_diagnostics_ = self._extract_diagnostics(mcmc, runtime_sec=runtime_sec)
        transformed = ["logit_kappa"]
        self.sampler_diagnostics_["parameterization"] = {
            "primitive_hierarchy": True,
            "transformed_variables": transformed,
            "non_centered_beta": True,
            "likelihood": str(self.likelihood),
            "use_group_scale": bool(self.use_group_scale),
            "use_local_scale": bool(self.use_local_scale),
            "shared_kappa": bool(self.shared_kappa),
            "tau0_effective": float(tau0_eff),
            "init_to_median": bool(self.use_init_to_median),
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


# ---------------------------------------------------------------------------
# Gibbs + 1-D slice sampler for GR-RHS
# ---------------------------------------------------------------------------

def _inv_gamma_sample(shape: float, rate: float, rng: Generator, jitter: float) -> float:
    """Sample InvGamma(shape, rate) using shape/rate parameterisation."""
    z = rng.gamma(shape=max(shape, jitter), scale=1.0 / max(rate, jitter))
    return 1.0 / max(z, jitter)


@dataclass
class GRRHS_Gibbs:
    """Gibbs + 1-D slice sampler for GR-RHS (Gaussian likelihood only).

    Per-iteration complexity
    ------------------------
    beta update    : O(n^2 p + n^3) when n < p (Bhattacharya fast sampler),
                     O(p^3) otherwise (Cholesky)
    sigma^2 update : O(p) (1-D slice on log sigma)
    tau update     : O(p) (1-D slice on log tau)
    lambda_j update: O(1) x p (1-D slice per coefficient, if use_local_scale)
    a_g update     : O(p_g) x G (1-D slice per group, if use_group_scale)
    kappa_g update : O(p_g) x G (1-D slice per group)

    Profile specialisation (O5)
    ---------------------------
    When use_local_scale=False and use_group_scale=False the posterior
    factorises across groups (Prop. 3.16).  The algorithm exploits this
    automatically because every kappa_g slice step is conditionally
    independent of every other kappa_{g'} given beta, tau, and sigma^2.
    """

    # prior hyper-parameters
    tau0: Optional[float] = None
    eta: float = 0.5
    s0: float = 1.0
    alpha_kappa: float = 2.0
    beta_kappa: float = 8.0
    use_group_scale: bool = True
    use_local_scale: bool = True
    shared_kappa: bool = False
    auto_calibrate_tau: bool = True
    tau_target: str = "coefficients"
    p0: Optional[float] = None
    sigma_reference: float = 1.0
    # sampler settings
    iters: int = 2000
    burnin: int = 1000
    thin: int = 1
    seed: int = 42
    num_chains: int = 1
    jitter: float = 1e-10
    slice_width_log: float = 0.5
    slice_width_logit: float = 1.0
    slice_max_steps: int = 200
    progress_bar: bool = True
    initial_chain_states: Optional[List[Dict[str, Any]]] = None
    resume_no_burnin: bool = False

    # posterior storage (set by fit)
    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    a_samples_: Optional[np.ndarray] = field(default=None, init=False)
    kappa_samples_: Optional[np.ndarray] = field(default=None, init=False)
    c2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_mean_: Optional[float] = field(default=None, init=False)
    tau_mean_: Optional[float] = field(default=None, init=False)
    lambda_mean_: Optional[np.ndarray] = field(default=None, init=False)
    a_mean_: Optional[np.ndarray] = field(default=None, init=False)
    kappa_mean_: Optional[np.ndarray] = field(default=None, init=False)
    c2_mean_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)
    groups_: Optional[List[List[int]]] = field(default=None, init=False)
    group_id_: Optional[np.ndarray] = field(default=None, init=False)
    group_sizes_: Optional[np.ndarray] = field(default=None, init=False)
    sampler_diagnostics_: Dict[str, Any] = field(default_factory=dict, init=False)
    chain_last_states_: Optional[List[Dict[str, Any]]] = field(default=None, init=False)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _v_arr(
        sigma2: float,
        tau2: float,
        lam2: np.ndarray,
        a2_j: np.ndarray,
        kappa_j: np.ndarray,
        jitter: float,
    ) -> np.ndarray:
        """Per-coefficient prior variance:
        v_{j,g} = sigma^2 * kappa * r / (kappa + (1-kappa) * r),
        where r = tau^2 * lambda^2 * a^2 / sigma^2.
        """
        r = tau2 * lam2 * a2_j / (sigma2 + jitter)
        return np.maximum(sigma2 * kappa_j * r / (kappa_j + (1.0 - kappa_j) * r + jitter), jitter)

    @staticmethod
    def _ll_beta(beta: np.ndarray, v: np.ndarray) -> float:
        """Log N(0, diag(v)) density for beta (up to constants)."""
        return -0.5 * float(np.sum(np.log(v) + beta ** 2 / v))

    # ------------------------------------------------------------------ log conditionals

    def _lc_log_sigma(
        self,
        r: float,
        beta: np.ndarray,
        y: np.ndarray,
        Xbeta: np.ndarray,
        tau2: float,
        lam2: np.ndarray,
        a2_j: np.ndarray,
        kappa_j: np.ndarray,
    ) -> float:
        """Log-conditional for r = log(sigma), including Jacobian."""
        sigma2 = math.exp(2.0 * r)
        resid = y - Xbeta
        v = self._v_arr(sigma2, tau2, lam2, a2_j, kappa_j, self.jitter)
        n = float(len(y))
        # Gaussian likelihood + beta prior + half-Cauchy(s0) prior + Jacobian (+r)
        return (
            -n * r
            - 0.5 * float(np.dot(resid, resid)) / sigma2
            + self._ll_beta(beta, v)
            + r
            - math.log(max(self.s0 ** 2 + sigma2, self.jitter))
        )

    def _lc_log_tau(
        self,
        u: float,
        beta: np.ndarray,
        sigma2: float,
        lam2: np.ndarray,
        a2_j: np.ndarray,
        kappa_j: np.ndarray,
        tau_scale: float,
    ) -> float:
        """Log-conditional for u = log(tau)."""
        tau2 = math.exp(2.0 * u)
        v = self._v_arr(sigma2, tau2, lam2, a2_j, kappa_j, self.jitter)
        return (
            self._ll_beta(beta, v)
            + u
            - math.log(max(tau_scale ** 2 + tau2, self.jitter))
        )

    def _lc_log_lam_j(
        self,
        s: float,
        beta_j: float,
        sigma2: float,
        tau2: float,
        a2_j: float,
        kappa_j: float,
    ) -> float:
        """Log-conditional for s = log(lambda_j); affects coefficient j only."""
        lam2_j = math.exp(2.0 * s)
        r = tau2 * lam2_j * a2_j / (sigma2 + self.jitter)
        v_j = max(sigma2 * kappa_j * r / (kappa_j + (1.0 - kappa_j) * r + self.jitter), self.jitter)
        # half-Cauchy(1) prior on lambda_j + Jacobian
        return -0.5 * (math.log(v_j) + beta_j ** 2 / v_j) + s - math.log(max(1.0 + math.exp(2.0 * s), self.jitter))

    def _lc_log_a_g(
        self,
        t: float,
        beta_g: np.ndarray,
        sigma2: float,
        tau2: float,
        lam2_g: np.ndarray,
        kappa_g: float,
        s_a_g: float,
    ) -> float:
        """Log-conditional for t = log a_g (affects all j in group g)."""
        a2 = math.exp(2.0 * t)
        r = tau2 * lam2_g * a2 / (sigma2 + self.jitter)
        kg = np.full_like(r, kappa_g)
        v_g = np.maximum(sigma2 * kg * r / (kg + (1.0 - kg) * r + self.jitter), self.jitter)
        # half-Normal(s_a_g) prior on a_g + Jacobian
        return (
            self._ll_beta(beta_g, v_g)
            + t
            - 0.5 * math.exp(2.0 * t) / max(s_a_g ** 2, self.jitter)
        )

    def _lc_logit_kappa_g(
        self,
        w: float,
        beta_g: np.ndarray,
        sigma2: float,
        tau2: float,
        lam2_g: np.ndarray,
        a2_g: float,
    ) -> float:
        """Log-conditional for w = logit(kappa_g). Jacobian is included in the Beta prior term."""
        kappa_g = 1.0 / (1.0 + math.exp(-w))
        kg = np.full(len(beta_g), kappa_g)
        r = tau2 * lam2_g * a2_g / (sigma2 + self.jitter)
        v_g = np.maximum(sigma2 * kg * r / (kg + (1.0 - kg) * r + self.jitter), self.jitter)
        # Beta(alpha_kappa, beta_kappa) prior with logit Jacobian absorbed.
        return (
            self._ll_beta(beta_g, v_g)
            + self.alpha_kappa * math.log(max(kappa_g, self.jitter))
            + self.beta_kappa * math.log(max(1.0 - kappa_g, self.jitter))
        )

    # ------------------------------------------------------------------ single chain

    def _sample_chain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: List[List[int]],
        group_id: np.ndarray,
        group_sizes: np.ndarray,
        tau0_eff: float,
        seed: int,
        *,
        iters: int,
        burnin: int,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        rng = default_rng(int(seed))
        n, p = X.shape
        G = int(group_sizes.shape[0])
        XtX = X.T @ X
        Xty = X.T @ y

        # group-scale prior for a_g: a_g ~ HalfNormal(eta / sqrt(group_size_g))
        s_a = self.eta / np.sqrt(np.maximum(group_sizes.astype(float), 1.0))

        # initialise parameters
        try:
            beta = np.linalg.solve(XtX + 1e-3 * np.eye(p), Xty)
        except np.linalg.LinAlgError:
            beta = np.zeros(p)
        sigma2 = max(float(np.var(y - X @ beta)), self.jitter)
        tau = float(tau0_eff)
        lam = np.ones(p)
        a = np.ones(G)
        kappa = np.full(G, float(self.alpha_kappa) / (self.alpha_kappa + self.beta_kappa))
        log_sigma = 0.5 * math.log(max(sigma2, self.jitter))
        log_tau = math.log(max(tau, self.jitter))
        log_lam = np.zeros(p)
        log_a = np.zeros(G)
        logit_kappa = np.zeros(G)
        if isinstance(initial_state, dict):
            beta = np.asarray(initial_state.get("beta", beta), dtype=float).reshape(-1)
            if beta.size != p:
                beta = np.asarray(np.zeros(p), dtype=float)
            log_sigma = float(initial_state.get("log_sigma", log_sigma))
            log_tau = float(initial_state.get("log_tau", log_tau))
            log_lam = np.asarray(initial_state.get("log_lam", log_lam), dtype=float).reshape(-1)
            if log_lam.size != p:
                log_lam = np.zeros(p)
            log_a = np.asarray(initial_state.get("log_a", log_a), dtype=float).reshape(-1)
            if log_a.size != G:
                log_a = np.zeros(G)
            logit_kappa = np.asarray(initial_state.get("logit_kappa", logit_kappa), dtype=float).reshape(-1)
            if logit_kappa.size != G:
                logit_kappa = np.zeros(G)
            sigma2 = max(float(math.exp(2.0 * log_sigma)), self.jitter)
            tau = max(float(math.exp(log_tau)), self.jitter)
            lam = np.maximum(np.exp(log_lam), self.jitter)
            a = np.maximum(np.exp(log_a), self.jitter)
            kappa = 1.0 / (1.0 + np.exp(-logit_kappa))

        iters_use = int(max(1, iters))
        burnin_use = int(max(0, min(burnin, iters_use - 1)))
        kept = max(0, (iters_use - burnin_use + int(self.thin) - 1) // int(self.thin))
        beta_draws = np.zeros((kept, p))
        sigma_draws = np.zeros(kept)
        tau_draws = np.zeros(kept)
        lam_draws = np.ones((kept, p))
        a_draws = np.ones((kept, G))
        kappa_draws = np.zeros((kept, G))
        keep_i = 0

        iterator = range(iters_use)
        if bool(self.progress_bar):
            try:
                from simulation_project.src.core.utils.logging_utils import progress
                iterator = progress(iterator, total=iters_use, desc="GR-RHS Gibbs")
            except Exception:
                pass

        for it in iterator:
            tau2 = tau ** 2
            sigma2 = math.exp(2.0 * log_sigma)
            lam2 = lam ** 2
            a2_j = a[group_id] ** 2
            kappa_j = kappa[group_id]

            # ---- beta | rest  (Woodbury when n < p, else Cholesky) ----
            v = self._v_arr(sigma2, tau2, lam2, a2_j, kappa_j, self.jitter)
            if n < p:
                beta = beta_sample_woodbury(X, y, sigma2, v, rng, jitter=self.jitter)
            else:
                beta = beta_sample_cholesky(XtX, Xty, sigma2, v, rng, jitter=self.jitter)
            Xbeta = X @ beta

            # ---- log sigma | rest  (1-D slice) ----
            def _lc_s(r: float) -> float:
                return self._lc_log_sigma(r, beta, y, Xbeta, tau2, lam2, a2_j, kappa_j)
            log_sigma = slice_sample_1d(_lc_s, log_sigma, rng, width=self.slice_width_log, max_steps=self.slice_max_steps)
            sigma2 = math.exp(2.0 * log_sigma)

            # ---- log tau | rest  (1-D slice) ----
            tau_scale = float(tau0_eff) * math.sqrt(max(sigma2, self.jitter))
            def _lc_t(u: float) -> float:
                return self._lc_log_tau(u, beta, sigma2, lam2, a2_j, kappa_j, tau_scale)
            log_tau = slice_sample_1d(_lc_t, log_tau, rng, width=self.slice_width_log, max_steps=self.slice_max_steps)
            tau = math.exp(log_tau)
            tau2 = tau ** 2

            # ---- log lambda_j | rest  (1-D slice per coefficient) ----
            if self.use_local_scale:
                for j in range(p):
                    def _lc_lj(s: float, _j: int = j) -> float:
                        return self._lc_log_lam_j(s, float(beta[_j]), sigma2, tau2, float(a2_j[_j]), float(kappa_j[_j]))
                    log_lam[j] = slice_sample_1d(_lc_lj, log_lam[j], rng, width=self.slice_width_log, max_steps=self.slice_max_steps)
                lam = np.exp(log_lam)
                lam2 = lam ** 2

            # ---- log a_g | rest  (1-D slice per group) ----
            if self.use_group_scale:
                for g, members in enumerate(groups):
                    idx = np.asarray(members, dtype=int)
                    def _lc_ag(t: float, _g: int = g, _idx: np.ndarray = idx) -> float:
                        return self._lc_log_a_g(t, beta[_idx], sigma2, tau2, lam2[_idx], float(kappa[_g]), float(s_a[_g]))
                    log_a[g] = slice_sample_1d(_lc_ag, log_a[g], rng, width=self.slice_width_log, max_steps=self.slice_max_steps)
                a = np.exp(log_a)

            a2_j = a[group_id] ** 2

            # ---- logit kappa_g | rest  (1-D slice per group; factorizes in profile mode) ----
            if self.shared_kappa:
                def _lc_ksh(w: float) -> float:
                    return self._lc_logit_kappa_g(w, beta, sigma2, tau2, lam2, float(np.mean(a2_j)))
                logit_kappa[0] = slice_sample_1d(_lc_ksh, logit_kappa[0], rng, width=self.slice_width_logit, max_steps=self.slice_max_steps)
                kappa[:] = 1.0 / (1.0 + math.exp(-logit_kappa[0]))
            else:
                for g, members in enumerate(groups):
                    idx = np.asarray(members, dtype=int)
                    def _lc_kg(w: float, _g: int = g, _idx: np.ndarray = idx) -> float:
                        return self._lc_logit_kappa_g(w, beta[_idx], sigma2, tau2, lam2[_idx], float(np.mean(a2_j[_idx])))
                    logit_kappa[g] = slice_sample_1d(_lc_kg, logit_kappa[g], rng, width=self.slice_width_logit, max_steps=self.slice_max_steps)
                kappa = 1.0 / (1.0 + np.exp(-logit_kappa))

            kappa_j = kappa[group_id]

            if it >= burnin_use and (it - burnin_use) % int(self.thin) == 0:
                beta_draws[keep_i] = beta
                sigma_draws[keep_i] = math.exp(log_sigma)
                tau_draws[keep_i] = tau
                lam_draws[keep_i] = lam.copy()
                a_draws[keep_i] = a.copy()
                kappa_draws[keep_i] = kappa.copy()
                keep_i += 1

        return {
            "beta": beta_draws,
            "sigma": sigma_draws,
            "tau": tau_draws,
            "lambda": lam_draws,
            "a": a_draws,
            "kappa": kappa_draws,
            "last_state": {
                "beta": np.asarray(beta, dtype=float).copy(),
                "log_sigma": float(log_sigma),
                "log_tau": float(log_tau),
                "log_lam": np.asarray(log_lam, dtype=float).copy(),
                "log_a": np.asarray(log_a, dtype=float).copy(),
                "logit_kappa": np.asarray(logit_kappa, dtype=float).copy(),
            },
        }

    # ------------------------------------------------------------------ public API

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[List[List[int]]] = None,
    ) -> "GRRHS_Gibbs":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D.")
        n, p = X_arr.shape

        groups_use = [[j] for j in range(p)] if groups is None else [list(map(int, g)) for g in groups]
        self.groups_ = groups_use
        gid, gsz = _split_groups(p, groups_use)
        self.group_id_ = gid
        self.group_sizes_ = gsz

        if self.tau0 is not None:
            tau0_eff = float(self.tau0)
        elif self.auto_calibrate_tau:
            target_dim = p if str(self.tau_target).strip().lower() == "coefficients" else len(groups_use)
            p0_use = float(self.p0) if self.p0 is not None else float(max(1, min(20, target_dim // 4)))
            tau0_eff = GRRHS_NUTS.calibrate_tau0(p0=p0_use, D=target_dim, n=n, sigma_ref=float(self.sigma_reference))
        else:
            tau0_eff = 0.1

        initial_states: list[Optional[Dict[str, Any]]] = [None] * int(self.num_chains)
        if isinstance(self.initial_chain_states, list) and self.initial_chain_states:
            for ci in range(min(len(self.initial_chain_states), int(self.num_chains))):
                st = self.initial_chain_states[ci]
                if isinstance(st, dict):
                    initial_states[ci] = st
        burnin_use = int(self.burnin)
        if bool(self.resume_no_burnin) and any(s is not None for s in initial_states):
            burnin_use = 0

        start = time.perf_counter()
        all_chains = [
            self._sample_chain(
                X_arr,
                y_arr,
                groups_use,
                gid,
                gsz,
                tau0_eff,
                int(self.seed) + ci,
                iters=int(self.iters),
                burnin=int(burnin_use),
                initial_state=initial_states[ci],
            )
            for ci in range(int(self.num_chains))
        ]
        runtime_sec = max(time.perf_counter() - start, 1e-12)
        self.chain_last_states_ = [dict(c.get("last_state", {})) for c in all_chains]

        def _stack(key: str) -> np.ndarray:
            arrs = [c[key] for c in all_chains]
            return arrs[0] if len(arrs) == 1 else np.stack(arrs, axis=0)

        self.coef_samples_ = _stack("beta")
        self.sigma_samples_ = _stack("sigma")
        self.tau_samples_ = _stack("tau")
        self.lambda_samples_ = _stack("lambda")
        self.a_samples_ = _stack("a")
        self.kappa_samples_ = _stack("kappa")

        def _flat2(arr: np.ndarray) -> np.ndarray:
            d = np.asarray(arr, dtype=float)
            return d if d.ndim == 2 else d.reshape(-1, d.shape[-1])

        def _flat1(arr: np.ndarray) -> np.ndarray:
            return np.asarray(arr, dtype=float).reshape(-1)

        self.coef_mean_ = _flat2(self.coef_samples_).mean(axis=0)
        self.sigma_mean_ = float(_flat1(self.sigma_samples_).mean())
        self.tau_mean_ = float(_flat1(self.tau_samples_).mean())
        self.lambda_mean_ = _flat2(self.lambda_samples_).mean(axis=0)
        self.a_mean_ = _flat2(self.a_samples_).mean(axis=0)
        self.kappa_mean_ = _flat2(self.kappa_samples_).mean(axis=0)
        self.c2_samples_ = (np.asarray(self.sigma_samples_, dtype=float) ** 2)[..., None] * (
            np.asarray(self.kappa_samples_, dtype=float)
            / np.maximum(1.0 - np.asarray(self.kappa_samples_, dtype=float), 1e-12)
        )
        self.c2_mean_ = _flat2(self.c2_samples_).mean(axis=0)
        self.intercept_ = 0.0

        profile_mode = not bool(self.use_local_scale) and not bool(self.use_group_scale)
        self.sampler_diagnostics_ = {
            "backend": "simcore_gibbs_slice",
            "runtime_sec": float(runtime_sec),
            "num_chains": int(self.num_chains),
            "kept_draws_per_chain": int(self.coef_samples_.shape[-2] if np.asarray(self.coef_samples_).ndim >= 3 else np.asarray(self.coef_samples_).shape[0]),
            "beta_sampler": "woodbury_bhattacharya" if n < p else "cholesky",
            "profile_mode_factorised": bool(profile_mode),
            "use_local_scale": bool(self.use_local_scale),
            "use_group_scale": bool(self.use_group_scale),
            "tau0_effective": float(tau0_eff),
        }
        return self

    def predict(self, X: np.ndarray, use_posterior_mean: bool = True) -> np.ndarray:
        if self.coef_mean_ is None:
            raise RuntimeError("Model not fitted.")
        return np.asarray(X, dtype=float) @ np.asarray(self.coef_mean_, dtype=float) + float(self.intercept_)

    def get_posterior_summaries(self) -> Dict[str, Any]:
        if self.coef_samples_ is None:
            raise RuntimeError("Model not fitted.")
        coef = np.asarray(self.coef_samples_, dtype=float)
        if coef.ndim >= 3:
            coef = coef.reshape(-1, coef.shape[-1])
        return {
            "beta_mean": coef.mean(axis=0),
            "beta_median": np.median(coef, axis=0),
            "beta_ci95": np.quantile(coef, [0.025, 0.975], axis=0),
            "sigma_mean": self.sigma_mean_,
            "tau_mean": self.tau_mean_,
            "a_mean": self.a_mean_,
            "kappa_mean": self.kappa_mean_,
            "c2_mean": self.c2_mean_,
            "lambda_mean": self.lambda_mean_,
        }


# ---------------------------------------------------------------------------
# Collapsed NUTS: beta marginalized out, NUTS on hyperparameters only
# ---------------------------------------------------------------------------

@dataclass
class GRRHS_CollapsedNUTS:
    """GR-RHS sampler with beta analytically marginalized.

    Runs NUTS on the marginal posterior:
        p(theta | y)  proportional to  N(y; 0, X V(theta) X^T + sigma^2 I_n)
    then draws beta ~ p(beta | theta, y) via Woodbury Gibbs.

    Dimension reduction vs GRRHS_NUTS
    ----------------------------------
    Mode                   GRRHS_NUTS    Collapsed
    Full (lambda+a active)  2p+2G+2      p+2G+2
    Profile (lam=a=1)       p+G+2        G+2       <- orders-of-magnitude smaller

    Profile mode precomputes M_g = X_g X_g^T so every NUTS gradient step
    computes Sigma_y = sum_g v_g M_g + sigma^2 I in O(G n^2 + n^3) instead of O(n^2 p).

    Convergence guarantee: the marginal posterior is the same as the joint
    posterior marginalized over beta (Rao-Blackwell).  All standard NUTS
    validity properties apply.
    """

    tau0: Optional[float] = None
    eta: float = 0.5
    s0: float = 1.0
    alpha_kappa: float = 2.0
    beta_kappa: float = 8.0
    use_group_scale: bool = True
    use_local_scale: bool = True
    shared_kappa: bool = False
    auto_calibrate_tau: bool = True
    tau_target: str = "coefficients"
    p0: Optional[float] = None
    sigma_reference: float = 1.0
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    thinning: int = 1
    target_accept_prob: float = 0.95
    max_tree_depth: int = 12
    dense_mass: bool = False
    use_init_to_median: bool = True
    chain_method: str = "sequential"
    progress_bar: bool = True
    seed: int = 42
    init_params: Optional[Dict[str, Any]] = None
    resume_no_warmup: bool = False
    beta_draws_per_sample: int = 1   # posterior beta draws per hyperparameter sample
    sigma_jitter: float = 1e-6       # numerical jitter on Sigma_y diagonal

    # posterior storage
    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    a_samples_: Optional[np.ndarray] = field(default=None, init=False)
    kappa_samples_: Optional[np.ndarray] = field(default=None, init=False)
    c2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_mean_: Optional[float] = field(default=None, init=False)
    tau_mean_: Optional[float] = field(default=None, init=False)
    lambda_mean_: Optional[np.ndarray] = field(default=None, init=False)
    a_mean_: Optional[np.ndarray] = field(default=None, init=False)
    kappa_mean_: Optional[np.ndarray] = field(default=None, init=False)
    c2_mean_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)
    groups_: Optional[List[List[int]]] = field(default=None, init=False)
    group_id_: Optional[np.ndarray] = field(default=None, init=False)
    group_sizes_: Optional[np.ndarray] = field(default=None, init=False)
    sampler_diagnostics_: Dict[str, Any] = field(default_factory=dict, init=False)
    last_init_params_: Optional[Dict[str, np.ndarray]] = field(default=None, init=False)

    # ------------------------------------------------------------------ model

    def _build_model(
        self,
        profile_mode: bool,
        group_XXT: Optional[jnp.ndarray],  # (G, n, n) precomputed M_g for profile mode
    ):
        """Return a NumPyro model function parameterised by this instance."""
        s0 = float(self.s0)
        alpha_kappa = float(self.alpha_kappa)
        beta_kappa = float(self.beta_kappa)
        eta = float(self.eta)
        use_local = bool(self.use_local_scale)
        use_group = bool(self.use_group_scale)
        shared_kappa = bool(self.shared_kappa)
        sig_jit = float(self.sigma_jitter)
        # capture static helper as local reference (avoids self-capture inside JAX trace)
        _lbeta = GRRHS_NUTS._log_beta_on_logit

        def model(X, y, group_id, group_sizes, tau0_eff):
            n, p = X.shape
            G = int(group_sizes.shape[0])

            sigma = numpyro.sample("sigma", dist.HalfCauchy(jnp.asarray(s0, dtype=X.dtype)))
            sigma2 = sigma * sigma

            tau_scale = jnp.maximum(jnp.asarray(tau0_eff, dtype=X.dtype) * sigma, _EPS)
            tau = numpyro.sample("tau", dist.HalfCauchy(tau_scale))

            if use_local:
                lam = numpyro.sample("lambda", dist.HalfCauchy(jnp.ones(p, dtype=X.dtype)).to_event(1))
            else:
                lam = numpyro.deterministic("lambda", jnp.ones(p, dtype=X.dtype))

            s_a = eta / jnp.sqrt(jnp.maximum(group_sizes.astype(X.dtype), 1.0))
            if use_group:
                a = numpyro.sample("a", dist.HalfNormal(s_a).to_event(1))
            else:
                a = numpyro.deterministic("a", jnp.ones(G, dtype=X.dtype))

            if shared_kappa:
                logit_kappa_raw = numpyro.sample("logit_kappa_shared_raw", dist.Normal(0.0, 1.0))
                kappa_shared = sigmoid(logit_kappa_raw)
                kappa = numpyro.deterministic("kappa", jnp.full((G,), kappa_shared, dtype=X.dtype))
                numpyro.factor(
                    "prior_logit_kappa",
                    _lbeta(logit_kappa_raw, alpha_kappa, beta_kappa)
                    - dist.Normal(0.0, 1.0).log_prob(logit_kappa_raw),
                )
            else:
                logit_kappa = numpyro.sample("logit_kappa_raw", dist.Normal(jnp.zeros(G), jnp.ones(G)).to_event(1))
                kappa = numpyro.deterministic("kappa", sigmoid(logit_kappa))
                numpyro.factor(
                    "prior_logit_kappa",
                    jnp.sum(
                        _lbeta(logit_kappa, alpha_kappa, beta_kappa)
                        - dist.Normal(0.0, 1.0).log_prob(logit_kappa)
                    ),
                )

            numpyro.deterministic("c2", sigma2 * kappa / (1.0 - kappa + _EPS))

            tau2 = tau * tau

            # Marginal likelihood: y ~ N(0, X V X^T + sigma^2 I_n)
            if profile_mode and group_XXT is not None:
                # lam=1 and a=1 => v depends only on group:
                # v_g = sigma2*kappa_g*tau2 / (sigma2*kappa_g + (1-kappa_g)*tau2)
                r_g = tau2 / (sigma2 + _EPS)
                v_g = sigma2 * kappa * r_g / (kappa + (1.0 - kappa) * r_g + _EPS)  # (G,)
                # Sigma_y = sum_g v_g * M_g + sigma2 * I  [M_g = X_g X_g^T precomputed]
                Sigma_y = jnp.einsum("g,gnm->nm", v_g, group_XXT) + sigma2 * jnp.eye(n, dtype=X.dtype)
            else:
                a_j = a[group_id]
                kappa_j = kappa[group_id]
                lam2 = lam * lam
                a2_j = a_j * a_j
                r = tau2 * lam2 * a2_j / (sigma2 + _EPS)
                v = sigma2 * kappa_j * r / (kappa_j + (1.0 - kappa_j) * r + _EPS)  # (p,)
                XD = X * v        # n x p
                Sigma_y = XD @ X.T + sigma2 * jnp.eye(n, dtype=X.dtype)

            Sigma_y = Sigma_y + sig_jit * jnp.eye(n, dtype=X.dtype)
            numpyro.sample("y", dist.MultivariateNormal(jnp.zeros(n, dtype=X.dtype), Sigma_y), obs=y)

        return model

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[List[List[int]]] = None,
    ) -> "GRRHS_CollapsedNUTS":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        n, p = X_arr.shape

        groups_use = [[j] for j in range(p)] if groups is None else [list(map(int, g)) for g in groups]
        self.groups_ = groups_use
        gid, gsz = _split_groups(p, groups_use)
        self.group_id_ = gid
        self.group_sizes_ = gsz
        G = len(groups_use)

        if self.tau0 is not None:
            tau0_eff = float(self.tau0)
        elif self.auto_calibrate_tau:
            target_dim = p if str(self.tau_target).strip().lower() == "coefficients" else G
            p0_use = float(self.p0) if self.p0 is not None else float(max(1, min(20, target_dim // 4)))
            tau0_eff = GRRHS_NUTS.calibrate_tau0(p0=p0_use, D=target_dim, n=n, sigma_ref=float(self.sigma_reference))
        else:
            tau0_eff = 0.1

        profile_mode = not bool(self.use_local_scale) and not bool(self.use_group_scale)

        # Precompute M_g = X_g X_g^T for profile mode
        group_XXT_jnp = None
        if profile_mode:
            M_list = []
            for g_idx, members in enumerate(groups_use):
                Xg = X_arr[:, np.asarray(members, dtype=int)]
                M_list.append(Xg @ Xg.T)  # n x n
            group_XXT_jnp = jnp.asarray(np.stack(M_list, axis=0))  # G x n x n

        model_fn = self._build_model(profile_mode=profile_mode, group_XXT=group_XXT_jnp)

        init_strategy = (
            numpyro.infer.init_to_median(num_samples=15)
            if bool(self.use_init_to_median)
            else numpyro.infer.init_to_uniform()
        )
        kernel = NUTS(
            model_fn,
            target_accept_prob=float(self.target_accept_prob),
            dense_mass=bool(self.dense_mass),
            max_tree_depth=int(self.max_tree_depth),
            init_strategy=init_strategy,
        )
        init_params_use = _normalize_init_params(
            self.init_params,
            num_chains=int(self.num_chains),
        )
        warmup_use = int(self.num_warmup)
        if bool(self.resume_no_warmup) and init_params_use is not None:
            warmup_use = 0
        mcmc = MCMC(
            kernel,
            num_warmup=int(warmup_use),
            num_samples=int(self.num_samples),
            num_chains=int(self.num_chains),
            chain_method=str(self.chain_method),
            progress_bar=bool(self.progress_bar),
        )

        start = time.perf_counter()
        run_kwargs: Dict[str, Any] = {}
        if init_params_use is not None:
            run_kwargs["init_params"] = init_params_use
        mcmc.run(
            random.PRNGKey(int(self.seed)),
            jnp.asarray(X_arr),
            jnp.asarray(y_arr),
            jnp.asarray(gid),
            jnp.asarray(gsz),
            float(tau0_eff),
            extra_fields=("diverging", "energy", "num_steps"),
            **run_kwargs,
        )
        runtime_nuts = time.perf_counter() - start

        samples = mcmc.get_samples(group_by_chain=True)
        latent_keys = ["sigma", "tau"]
        if bool(self.use_local_scale):
            latent_keys.append("lambda")
        if bool(self.use_group_scale):
            latent_keys.append("a")
        if bool(self.shared_kappa):
            latent_keys.append("logit_kappa_shared_raw")
        else:
            latent_keys.append("logit_kappa_raw")
        self.last_init_params_ = _extract_last_init_params(samples, latent_keys=latent_keys)

        # ---- Draw beta from p(beta | theta_i, y) via Woodbury ----
        rng_post = default_rng(int(self.seed) + 1)
        X64 = X_arr.astype(np.float64)
        y64 = y_arr.astype(np.float64)

        def _get_flat(key: str) -> Optional[np.ndarray]:
            if key not in samples:
                return None
            arr = np.asarray(samples[key], dtype=np.float64)
            return arr.reshape(-1, *arr.shape[2:]) if arr.ndim >= 2 else arr

        sigma_flat = _get_flat("sigma")          # (S,)
        tau_flat = _get_flat("tau")              # (S,)
        lam_flat = _get_flat("lambda")           # (S, p)
        a_flat = _get_flat("a")                  # (S, G)
        kappa_flat = _get_flat("kappa")          # (S, G)

        S = sigma_flat.shape[0] if sigma_flat is not None else 0
        beta_draws = np.zeros((S, p))

        for i in range(S):
            sig2_i = float(sigma_flat[i]) ** 2 if sigma_flat is not None else 1.0
            tau2_i = float(tau_flat[i]) ** 2 if tau_flat is not None else 1.0
            lam2_i = lam_flat[i] ** 2 if lam_flat is not None else np.ones(p)
            a2_j_i = (a_flat[i][gid]) ** 2 if a_flat is not None else np.ones(p)
            kappa_j_i = kappa_flat[i][gid] if kappa_flat is not None else np.full(p, 0.5)

            r_i = tau2_i * lam2_i * a2_j_i / (sig2_i + 1e-12)
            v_i = sig2_i * kappa_j_i * r_i / (kappa_j_i + (1.0 - kappa_j_i) * r_i + 1e-12)
            v_i = np.maximum(v_i, 1e-12)

            if n < p:
                beta_draws[i] = beta_sample_woodbury(X64, y64, sig2_i, v_i, rng_post)
            else:
                XtX = X64.T @ X64
                Xty = X64.T @ y64
                beta_draws[i] = beta_sample_cholesky(XtX, Xty, sig2_i, v_i, rng_post)

        runtime_sec = time.perf_counter() - start

        # ---- Store samples ----
        def _thin_flat(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if arr is None:
                return None
            step = int(self.thinning)
            return arr if step <= 1 else arr[::step]

        self.coef_samples_ = _thin_flat(beta_draws)
        self.sigma_samples_ = _thin_flat(sigma_flat)
        self.tau_samples_ = _thin_flat(tau_flat)
        self.lambda_samples_ = _thin_flat(lam_flat)
        self.a_samples_ = _thin_flat(a_flat)
        self.kappa_samples_ = _thin_flat(kappa_flat)
        c2_flat = _get_flat("c2")
        self.c2_samples_ = _thin_flat(c2_flat)

        def _mean1(arr):
            return None if arr is None else float(np.asarray(arr).reshape(-1).mean())
        def _mean2(arr):
            return None if arr is None else np.asarray(arr, dtype=float).reshape(-1, arr.shape[-1]).mean(axis=0)

        self.coef_mean_ = np.asarray(self.coef_samples_, dtype=float).mean(axis=0) if self.coef_samples_ is not None else None
        self.sigma_mean_ = _mean1(self.sigma_samples_)
        self.tau_mean_ = _mean1(self.tau_samples_)
        self.lambda_mean_ = _mean2(self.lambda_samples_)
        self.a_mean_ = _mean2(self.a_samples_)
        self.kappa_mean_ = _mean2(self.kappa_samples_)
        self.c2_mean_ = _mean2(self.c2_samples_)
        self.intercept_ = 0.0

        extra = {}
        try:
            extra = mcmc.get_extra_fields(group_by_chain=True)
        except Exception:
            pass
        div = int(np.sum(np.asarray(extra.get("diverging", []), dtype=float) > 0.5)) if "diverging" in extra else -1
        self.sampler_diagnostics_ = {
            "backend": "simcore_collapsed_nuts",
            "runtime_sec": float(runtime_sec),
            "runtime_nuts_sec": float(runtime_nuts),
            "profile_mode": bool(profile_mode),
            "nuts_dim": G + 2 if profile_mode else (p + 2 * G + 2 if (self.use_local_scale and self.use_group_scale) else p + G + 2),
            "beta_sampler": "woodbury_bhattacharya" if n < p else "cholesky",
            "divergences": div,
            "tau0_effective": float(tau0_eff),
        }
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_mean_ is None:
            raise RuntimeError("Model not fitted.")
        return np.asarray(X, dtype=float) @ np.asarray(self.coef_mean_, dtype=float)

    def get_posterior_summaries(self) -> Dict[str, Any]:
        if self.coef_samples_ is None:
            raise RuntimeError("Model not fitted.")
        coef = np.asarray(self.coef_samples_, dtype=float)
        return {
            "beta_mean": coef.mean(axis=0),
            "beta_median": np.median(coef, axis=0),
            "beta_ci95": np.quantile(coef, [0.025, 0.975], axis=0),
            "sigma_mean": self.sigma_mean_,
            "tau_mean": self.tau_mean_,
            "a_mean": self.a_mean_,
            "kappa_mean": self.kappa_mean_,
            "c2_mean": self.c2_mean_,
            "lambda_mean": self.lambda_mean_,
        }


