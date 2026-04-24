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
from simulation_project.src.core.inference.woodbury import (
    beta_sample_block_cholesky,
    beta_sample_cholesky,
    beta_sample_woodbury,
)

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


def _should_use_median_init(
    *,
    use_init_to_median: bool,
    init_params: Optional[Dict[str, jnp.ndarray]],
) -> bool:
    # Resumed runs already provide explicit latent values, so a median-based
    # strategy just adds noisy "skipping initialization" warnings.
    return bool(use_init_to_median) and not bool(init_params)


def _safe_nanmin(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


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
    kappa_samples_: Optional[np.ndarray] = field(default=None, init=False)
    c2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_mean_: Optional[float] = field(default=None, init=False)
    tau_mean_: Optional[float] = field(default=None, init=False)
    lambda_mean_: Optional[np.ndarray] = field(default=None, init=False)
    kappa_mean_: Optional[np.ndarray] = field(default=None, init=False)
    c2_mean_: Optional[np.ndarray] = field(default=None, init=False)
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

        kappa_j = kappa[group_id]
        tau2 = tau * tau
        lam2 = lam * lam
        # Dimensionless ratio r_j = tau^2 * lambda_j^2 / sigma^2
        # keeps each factor O(1) and avoids overflow in products of scales.
        # v_j = sigma^2 * kappa_g(j) * r_j / (kappa_g(j) + (1-kappa_g(j)) * r_j)
        r = tau2 * lam2 / (sigma2 + _EPS)
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

        init_params_use = _normalize_init_params(
            self.init_params,
            num_chains=int(self.num_chains),
        )
        init_strategy = (
            numpyro.infer.init_to_median(num_samples=15)
            if _should_use_median_init(
                use_init_to_median=bool(self.use_init_to_median),
                init_params=init_params_use,
            )
            else numpyro.infer.init_to_uniform()
        )
        kernel = NUTS(
            self._model,
            target_accept_prob=float(self.target_accept_prob),
            dense_mass=bool(self.dense_mass),
            max_tree_depth=int(self.max_tree_depth),
            init_strategy=init_strategy,
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
        self.kappa_samples_ = get("kappa")
        self.c2_samples_ = get("c2")

        coef_draws = _flatten_param(self.coef_samples_)
        self.coef_mean_ = None if coef_draws is None else coef_draws.mean(axis=0)
        sigma_draws = _flatten_scalar(self.sigma_samples_)
        tau_draws = _flatten_scalar(self.tau_samples_)
        lam_draws = _flatten_param(self.lambda_samples_)
        kappa_draws = _flatten_param(self.kappa_samples_)
        c2_draws = _flatten_param(self.c2_samples_)
        self.sigma_mean_ = None if sigma_draws is None else float(sigma_draws.mean())
        self.tau_mean_ = None if tau_draws is None else float(tau_draws.mean())
        self.lambda_mean_ = None if lam_draws is None else lam_draws.mean(axis=0)
        self.kappa_mean_ = None if kappa_draws is None else kappa_draws.mean(axis=0)
        self.c2_mean_ = None if c2_draws is None else c2_draws.mean(axis=0)
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
        ebfmi_min = _safe_nanmin(ebfmi_vals)
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
            "kappa_mean": self.kappa_mean_,
            "c2_mean": self.c2_mean_,
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
    kappa_g update : O(p_g) x G (1-D slice per group)

    Profile specialisation (O5)
    ---------------------------
    When use_local_scale=False the posterior
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
    kappa_samples_: Optional[np.ndarray] = field(default=None, init=False)
    c2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_mean_: Optional[float] = field(default=None, init=False)
    tau_mean_: Optional[float] = field(default=None, init=False)
    lambda_mean_: Optional[np.ndarray] = field(default=None, init=False)
    kappa_mean_: Optional[np.ndarray] = field(default=None, init=False)
    c2_mean_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)
    groups_: Optional[List[List[int]]] = field(default=None, init=False)
    group_id_: Optional[np.ndarray] = field(default=None, init=False)
    group_sizes_: Optional[np.ndarray] = field(default=None, init=False)
    sampler_diagnostics_: Dict[str, Any] = field(default_factory=dict, init=False)
    chain_last_states_: Optional[List[Dict[str, Any]]] = field(default=None, init=False)
    chain_phase_infos_: Optional[List[Dict[str, Any]]] = field(default=None, init=False)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _v_arr(
        sigma2: float,
        tau2: float,
        lam2: np.ndarray,
        kappa_j: np.ndarray,
        jitter: float,
    ) -> np.ndarray:
        """Per-coefficient prior variance:
        v_{j,g} = sigma^2 * kappa * r / (kappa + (1-kappa) * r),
        where r = tau^2 * lambda^2 / sigma^2.
        """
        r = tau2 * lam2 / (sigma2 + jitter)
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
        kappa_j: np.ndarray,
    ) -> float:
        """Log-conditional for r = log(sigma), including Jacobian."""
        sigma2 = math.exp(2.0 * r)
        resid = y - Xbeta
        v = self._v_arr(sigma2, tau2, lam2, kappa_j, self.jitter)
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
        kappa_j: np.ndarray,
        tau_scale: float,
    ) -> float:
        """Log-conditional for u = log(tau)."""
        tau2 = math.exp(2.0 * u)
        v = self._v_arr(sigma2, tau2, lam2, kappa_j, self.jitter)
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
        kappa_j: float,
    ) -> float:
        """Log-conditional for s = log(lambda_j); affects coefficient j only."""
        lam2_j = math.exp(2.0 * s)
        r = tau2 * lam2_j / (sigma2 + self.jitter)
        v_j = max(sigma2 * kappa_j * r / (kappa_j + (1.0 - kappa_j) * r + self.jitter), self.jitter)
        # half-Cauchy(1) prior on lambda_j + Jacobian
        return -0.5 * (math.log(v_j) + beta_j ** 2 / v_j) + s - math.log(max(1.0 + math.exp(2.0 * s), self.jitter))

    def _lc_logit_kappa_g(
        self,
        w: float,
        beta_g: np.ndarray,
        sigma2: float,
        tau2: float,
        lam2_g: np.ndarray,
    ) -> float:
        """Log-conditional for w = logit(kappa_g). Jacobian is included in the Beta prior term."""
        kappa_g = 1.0 / (1.0 + math.exp(-w))
        kg = np.full(len(beta_g), kappa_g)
        r = tau2 * lam2_g / (sigma2 + self.jitter)
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

        # initialise parameters
        try:
            beta = np.linalg.solve(XtX + 1e-3 * np.eye(p), Xty)
        except np.linalg.LinAlgError:
            beta = np.zeros(p)
        sigma2 = max(float(np.var(y - X @ beta)), self.jitter)
        tau = float(tau0_eff)
        lam = np.ones(p)
        kappa = np.full(G, float(self.alpha_kappa) / (self.alpha_kappa + self.beta_kappa))
        log_sigma = 0.5 * math.log(max(sigma2, self.jitter))
        log_tau = math.log(max(tau, self.jitter))
        log_lam = np.zeros(p)
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
            logit_kappa = np.asarray(initial_state.get("logit_kappa", logit_kappa), dtype=float).reshape(-1)
            if logit_kappa.size != G:
                logit_kappa = np.zeros(G)
            sigma2 = max(float(math.exp(2.0 * log_sigma)), self.jitter)
            tau = max(float(math.exp(log_tau)), self.jitter)
            lam = np.maximum(np.exp(log_lam), self.jitter)
            kappa = 1.0 / (1.0 + np.exp(-logit_kappa))

        iters_use = int(max(1, iters))
        burnin_use = int(max(0, min(burnin, iters_use - 1)))
        kept = max(0, (iters_use - burnin_use + int(self.thin) - 1) // int(self.thin))
        beta_draws = np.zeros((kept, p))
        sigma_draws = np.zeros(kept)
        tau_draws = np.zeros(kept)
        lam_draws = np.ones((kept, p))
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
            kappa_j = kappa[group_id]

            # ---- beta | rest  (Woodbury when n < p, else Cholesky) ----
            v = self._v_arr(sigma2, tau2, lam2, kappa_j, self.jitter)
            if n < p:
                beta = beta_sample_woodbury(X, y, sigma2, v, rng, jitter=self.jitter)
            else:
                beta = beta_sample_cholesky(XtX, Xty, sigma2, v, rng, jitter=self.jitter)
            Xbeta = X @ beta

            # ---- log sigma | rest  (1-D slice) ----
            def _lc_s(r: float) -> float:
                return self._lc_log_sigma(r, beta, y, Xbeta, tau2, lam2, kappa_j)
            log_sigma = slice_sample_1d(_lc_s, log_sigma, rng, width=self.slice_width_log, max_steps=self.slice_max_steps)
            sigma2 = math.exp(2.0 * log_sigma)

            # ---- log tau | rest  (1-D slice) ----
            tau_scale = float(tau0_eff) * math.sqrt(max(sigma2, self.jitter))
            def _lc_t(u: float) -> float:
                return self._lc_log_tau(u, beta, sigma2, lam2, kappa_j, tau_scale)
            log_tau = slice_sample_1d(_lc_t, log_tau, rng, width=self.slice_width_log, max_steps=self.slice_max_steps)
            tau = math.exp(log_tau)
            tau2 = tau ** 2

            # ---- log lambda_j | rest  (1-D slice per coefficient) ----
            if self.use_local_scale:
                for j in range(p):
                    def _lc_lj(s: float, _j: int = j) -> float:
                        return self._lc_log_lam_j(s, float(beta[_j]), sigma2, tau2, float(kappa_j[_j]))
                    log_lam[j] = slice_sample_1d(_lc_lj, log_lam[j], rng, width=self.slice_width_log, max_steps=self.slice_max_steps)
                lam = np.exp(log_lam)
                lam2 = lam ** 2

            # ---- logit kappa_g | rest  (1-D slice per group; factorizes in profile mode) ----
            if self.shared_kappa:
                def _lc_ksh(w: float) -> float:
                    return self._lc_logit_kappa_g(w, beta, sigma2, tau2, lam2)
                logit_kappa[0] = slice_sample_1d(_lc_ksh, logit_kappa[0], rng, width=self.slice_width_logit, max_steps=self.slice_max_steps)
                kappa[:] = 1.0 / (1.0 + math.exp(-logit_kappa[0]))
            else:
                for g, members in enumerate(groups):
                    idx = np.asarray(members, dtype=int)
                    def _lc_kg(w: float, _g: int = g, _idx: np.ndarray = idx) -> float:
                        return self._lc_logit_kappa_g(w, beta[_idx], sigma2, tau2, lam2[_idx])
                    logit_kappa[g] = slice_sample_1d(_lc_kg, logit_kappa[g], rng, width=self.slice_width_logit, max_steps=self.slice_max_steps)
                kappa = 1.0 / (1.0 + np.exp(-logit_kappa))

            kappa_j = kappa[group_id]

            if it >= burnin_use and (it - burnin_use) % int(self.thin) == 0:
                beta_draws[keep_i] = beta
                sigma_draws[keep_i] = math.exp(log_sigma)
                tau_draws[keep_i] = tau
                lam_draws[keep_i] = lam.copy()
                kappa_draws[keep_i] = kappa.copy()
                keep_i += 1

        phase_a_converged_flag = bool(
            skip_adaptive or phase_a_iters < int(self.phase_a_max_iters) or warmup_warning != "geometry_not_stable"
        )
        phase_b_converged_flag = bool(
            skip_adaptive or phase_b_iters < int(self.phase_b_max_iters) or warmup_warning not in {"transition_not_stable"}
        )
        concentrated_phase_a_relax = bool(
            (not phase_a_converged_flag)
            and phase_b_converged_flag
            and bool(phase_a_checks_last.get("mean_kappa"))
            and bool(phase_a_checks_last.get("prior_var_ok"))
            and concentration_score >= 0.40
        )
        warmup_warning_final = str(warmup_warning)
        if concentrated_phase_a_relax and warmup_warning_final == "geometry_not_stable":
            phase_a_converged_flag = True
            warmup_warning_final = ""

        return {
            "beta": beta_draws,
            "sigma": sigma_draws,
            "tau": tau_draws,
            "lambda": lam_draws,
            "kappa": kappa_draws,
            "last_state": {
                "beta": np.asarray(beta, dtype=float).copy(),
                "log_sigma": float(log_sigma),
                "log_tau": float(log_tau),
                "log_lam": np.asarray(log_lam, dtype=float).copy(),
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
        self.chain_phase_infos_ = [dict(c.get("phase_info", {})) for c in all_chains]

        def _stack(key: str) -> np.ndarray:
            arrs = [c[key] for c in all_chains]
            return arrs[0] if len(arrs) == 1 else np.stack(arrs, axis=0)

        self.coef_samples_ = _stack("beta")
        self.sigma_samples_ = _stack("sigma")
        self.tau_samples_ = _stack("tau")
        self.lambda_samples_ = _stack("lambda")
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
        self.kappa_mean_ = _flat2(self.kappa_samples_).mean(axis=0)
        self.c2_samples_ = (np.asarray(self.sigma_samples_, dtype=float) ** 2)[..., None] * (
            np.asarray(self.kappa_samples_, dtype=float)
            / np.maximum(1.0 - np.asarray(self.kappa_samples_, dtype=float), 1e-12)
        )
        self.c2_mean_ = _flat2(self.c2_samples_).mean(axis=0)
        self.intercept_ = 0.0

        profile_mode = not bool(self.use_local_scale)
        self.sampler_diagnostics_ = {
            "backend": "simcore_gibbs_slice",
            "runtime_sec": float(runtime_sec),
            "num_chains": int(self.num_chains),
            "kept_draws_per_chain": int(self.coef_samples_.shape[-2] if np.asarray(self.coef_samples_).ndim >= 3 else np.asarray(self.coef_samples_).shape[0]),
            "beta_sampler": "woodbury_bhattacharya" if n < p else "cholesky",
            "profile_mode_factorised": bool(profile_mode),
            "use_local_scale": bool(self.use_local_scale),
            "tau0_effective": float(tau0_eff),
        }
        if isinstance(self.chain_phase_infos_, list) and self.chain_phase_infos_:
            self.sampler_diagnostics_["chain_phase_infos"] = self.chain_phase_infos_
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
            "kappa_mean": self.kappa_mean_,
            "c2_mean": self.c2_mean_,
            "lambda_mean": self.lambda_mean_,
        }


@dataclass
class GRRHS_Gibbs_Staged(GRRHS_Gibbs):
    """Staged Gaussian Gibbs sampler for GR-RHS.

    Phase A focuses on stabilizing posterior geometry (tau / lambda / kappa).
    Phase B checks transition stability of beta summaries.
    Phase C collects stationary draws.
    """

    adaptive_burnin: bool = True
    phase_a_max_iters: int = 400
    phase_b_max_iters: int = 300
    phase_a_hyper_only_iters: int = 120
    min_phase_a_iters: int = 100
    min_phase_b_iters: int = 100
    geometry_window: int = 100
    geometry_tol: float = 0.03
    transition_window: int = 100
    transition_tol: float = 0.05
    phase_a_late_beta_refresh: bool = True
    phase_b_extra_beta_refresh: bool = True
    phase_a_refresh_interval: int = 12
    phase_a_refresh_repeats: int = 1
    phase_b_initial_extra_refresh_steps: int = 20
    phase_b_initial_refresh_repeats: int = 1
    phase_a_core_min_pass: int = 2
    phase_a_support_min_pass: int = 1
    phase_a_score_min: int = 3
    phase_b_core_min_pass: int = 2
    phase_b_support_min_pass: int = 1
    phase_b_score_min: int = 3
    concentrated_block_refresh: bool = True
    concentrated_top_groups: int = 2
    concentrated_refresh_min_score: float = 0.08
    concentrated_refresh_baseline_ratio: float = 1.8
    phase_a_block_refresh_interval: int = 10
    phase_b_block_refresh_steps: int = 24
    phase_b_block_refresh_repeats: int = 1
    structure_aware_warmup: bool = True
    distributed_entropy_threshold: float = 0.85
    concentrated_score_threshold: float = 0.34
    concentrated_top2_mass_threshold: float = 0.62
    distributed_top2_mass_threshold: float = 0.52
    distributed_block_top_groups: int = 3
    distributed_phase_a_score_min: int = 3
    distributed_phase_b_score_min: int = 3

    @staticmethod
    def _group_norms(beta: np.ndarray, groups: List[List[int]]) -> np.ndarray:
        vals = np.zeros(len(groups), dtype=float)
        for gid, members in enumerate(groups):
            idx = np.asarray(members, dtype=int)
            vals[gid] = float(np.linalg.norm(beta[idx])) if idx.size else 0.0
        return vals

    @staticmethod
    def _concentration_score(group_norms: np.ndarray) -> float:
        arr = np.asarray(group_norms, dtype=float).reshape(-1)
        total = float(np.sum(arr))
        if (not np.isfinite(total)) or total <= 0.0:
            return 0.0
        return float(np.max(arr) / total)

    @staticmethod
    def _group_mass_weights(group_norms: np.ndarray) -> np.ndarray:
        arr = np.asarray(group_norms, dtype=float).reshape(-1)
        total = float(np.sum(arr))
        if arr.size == 0 or (not np.isfinite(total)) or total <= 0.0:
            return np.zeros_like(arr, dtype=float)
        w = arr / total
        return np.clip(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    @classmethod
    def _group_norm_entropy(cls, group_norms: np.ndarray) -> float:
        w = cls._group_mass_weights(group_norms)
        if w.size <= 1 or float(np.sum(w)) <= 0.0:
            return 0.0
        mask = w > 0.0
        ent = -float(np.sum(w[mask] * np.log(w[mask])))
        return float(ent / max(np.log(float(w.size)), 1e-12))

    @classmethod
    def _topk_group_mass(cls, group_norms: np.ndarray, k: int) -> float:
        w = cls._group_mass_weights(group_norms)
        if w.size == 0:
            return 0.0
        kk = int(max(1, min(int(k), int(w.size))))
        order = np.argsort(-w)
        return float(np.sum(w[order[:kk]]))

    @staticmethod
    def _safe_cv(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            return 0.0
        mean = float(np.mean(arr))
        sd = float(np.std(arr))
        if (not np.isfinite(mean)) or abs(mean) < 1e-12:
            return 0.0
        return float(sd / abs(mean))

    def _structure_mode(
        self,
        *,
        concentration_score: float,
        group_entropy: float,
        top2_mass: float,
    ) -> str:
        if not bool(self.structure_aware_warmup):
            return "concentrated"
        conc = float(concentration_score)
        ent = float(group_entropy)
        top2 = float(top2_mass)
        if conc >= float(self.concentrated_score_threshold) or top2 >= float(self.concentrated_top2_mass_threshold):
            return "concentrated"
        if (
            ent >= float(self.distributed_entropy_threshold)
            and conc <= float(self.concentrated_score_threshold) * 0.9
            and top2 <= float(self.distributed_top2_mass_threshold)
        ):
            return "distributed"
        return "unclear"

    def _global_beta_refresh(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        beta: np.ndarray,
        sigma2: float,
        tau2: float,
        lam2: np.ndarray,
        kappa: np.ndarray,
        group_id: np.ndarray,
        rng: Generator,
    ) -> np.ndarray:
        p = int(X.shape[1])
        kappa_j = kappa[group_id]
        v = self._v_arr(sigma2, tau2, lam2, kappa_j, self.jitter)
        if int(X.shape[0]) < p:
            return beta_sample_woodbury(X, y, sigma2, v, rng, jitter=self.jitter)
        XtX = X.T @ X
        Xty = X.T @ y
        return beta_sample_cholesky(XtX, Xty, sigma2, v, rng, jitter=self.jitter)

    def _target_group_ids(
        self,
        *,
        beta: np.ndarray,
        groups: List[List[int]],
        concentration_score: float,
    ) -> tuple[list[int], float]:
        if not bool(self.concentrated_block_refresh):
            return [], float("inf")
        g_count = max(int(len(groups)), 1)
        baseline = 1.0 / float(g_count)
        adaptive_threshold = max(
            baseline + float(self.concentrated_refresh_min_score),
            baseline * float(self.concentrated_refresh_baseline_ratio),
        )
        if float(concentration_score) < float(adaptive_threshold):
            return [], float(adaptive_threshold)
        norms = self._group_norms(beta, groups)
        if norms.size == 0 or not np.any(np.isfinite(norms)):
            return [], float(adaptive_threshold)
        order = np.argsort(-norms)
        top_k = int(max(0, min(int(self.concentrated_top_groups), int(order.size))))
        return [int(v) for v in order[:top_k]], float(adaptive_threshold)

    def _group_block_beta_refresh(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        beta: np.ndarray,
        sigma2: float,
        tau2: float,
        lam2: np.ndarray,
        kappa: np.ndarray,
        groups: List[List[int]],
        group_id: np.ndarray,
        target_groups: Sequence[int],
        rng: Generator,
    ) -> np.ndarray:
        if not target_groups:
            return beta
        beta_new = np.asarray(beta, dtype=float).copy()
        fitted = X @ beta_new
        for gid in target_groups:
            members = np.asarray(groups[int(gid)], dtype=int)
            if members.size == 0:
                continue
            Xg = X[:, members]
            fitted_without = fitted - Xg @ beta_new[members]
            resid_without = y - fitted_without
            kappa_g = float(kappa[int(gid)])
            prior_var_g = self._v_arr(
                sigma2,
                tau2,
                lam2[members],
                np.full(members.size, kappa_g, dtype=float),
                self.jitter,
            )
            beta_block = beta_sample_block_cholesky(
                Xg,
                resid_without,
                sigma2,
                prior_var_g,
                rng,
                jitter=self.jitter,
            )
            beta_new[members] = beta_block
            fitted = fitted_without + Xg @ beta_block
        return beta_new

    @staticmethod
    def _stable_window(trace: List[float], window: int, tol: float) -> bool:
        vals = np.asarray(trace, dtype=float)
        w = int(max(4, window))
        if vals.size < 2 * w:
            return False
        prev = vals[-2 * w : -w]
        curr = vals[-w:]
        if (not np.all(np.isfinite(prev))) or (not np.all(np.isfinite(curr))):
            return False
        prev_med = float(np.median(prev))
        curr_med = float(np.median(curr))
        scale = max(abs(prev_med), abs(curr_med), 1.0)

        prev_mad = float(np.median(np.abs(prev - prev_med)))
        curr_mad = float(np.median(np.abs(curr - curr_med)))
        prev_iqr = float(np.quantile(prev, 0.75) - np.quantile(prev, 0.25))
        curr_iqr = float(np.quantile(curr, 0.75) - np.quantile(curr, 0.25))

        center_ok = abs(curr_med - prev_med) <= float(tol) * scale
        mad_ok = abs(curr_mad - prev_mad) <= float(tol) * max(prev_mad, curr_mad, 1.0)
        iqr_ok = abs(curr_iqr - prev_iqr) <= float(tol) * max(prev_iqr, curr_iqr, 1.0)

        prev_half_1 = prev[: max(2, w // 2)]
        prev_half_2 = prev[-max(2, w // 2) :]
        curr_half_1 = curr[: max(2, w // 2)]
        curr_half_2 = curr[-max(2, w // 2) :]
        prev_shift = abs(float(np.mean(prev_half_2) - np.mean(prev_half_1)))
        curr_shift = abs(float(np.mean(curr_half_2) - np.mean(curr_half_1)))
        trend_ok = max(prev_shift, curr_shift) <= float(tol) * scale * 1.25
        spread_ok = bool(mad_ok or iqr_ok)
        return bool(center_ok and spread_ok and trend_ok)

    @staticmethod
    def _robust_trace_window(
        trace: List[float],
        window: int,
    ) -> Optional[Dict[str, np.ndarray | float]]:
        vals = np.asarray(trace, dtype=float)
        w = int(max(4, window))
        if vals.size < 2 * w:
            return None
        prev = vals[-2 * w : -w]
        curr = vals[-w:]
        if (not np.all(np.isfinite(prev))) or (not np.all(np.isfinite(curr))):
            return None
        prev_med = float(np.median(prev))
        curr_med = float(np.median(curr))
        prev_mad = float(np.median(np.abs(prev - prev_med)))
        curr_mad = float(np.median(np.abs(curr - curr_med)))
        prev_iqr = float(np.quantile(prev, 0.75) - np.quantile(prev, 0.25))
        curr_iqr = float(np.quantile(curr, 0.75) - np.quantile(curr, 0.25))
        prev_half_1 = prev[: max(2, w // 2)]
        prev_half_2 = prev[-max(2, w // 2) :]
        curr_half_1 = curr[: max(2, w // 2)]
        curr_half_2 = curr[-max(2, w // 2) :]
        prev_shift = abs(float(np.mean(prev_half_2) - np.mean(prev_half_1)))
        curr_shift = abs(float(np.mean(curr_half_2) - np.mean(curr_half_1)))
        return {
            "prev": prev,
            "curr": curr,
            "prev_med": prev_med,
            "curr_med": curr_med,
            "prev_mad": prev_mad,
            "curr_mad": curr_mad,
            "prev_iqr": prev_iqr,
            "curr_iqr": curr_iqr,
            "prev_shift": prev_shift,
            "curr_shift": curr_shift,
        }

    def _metric_stable(
        self,
        trace: List[float],
        *,
        window: int,
        tol: float,
        center_weight: float = 1.0,
        spread_weight: float = 1.0,
        trend_weight: float = 1.0,
    ) -> bool:
        stats = self._robust_trace_window(trace, window)
        if stats is None:
            return False
        prev_med = float(stats["prev_med"])
        curr_med = float(stats["curr_med"])
        prev_mad = float(stats["prev_mad"])
        curr_mad = float(stats["curr_mad"])
        prev_iqr = float(stats["prev_iqr"])
        curr_iqr = float(stats["curr_iqr"])
        prev_shift = float(stats["prev_shift"])
        curr_shift = float(stats["curr_shift"])
        scale = max(abs(prev_med), abs(curr_med), 1.0)
        center_ok = abs(curr_med - prev_med) <= float(tol) * scale * float(center_weight)
        spread_ok = (
            abs(curr_mad - prev_mad) <= float(tol) * max(prev_mad, curr_mad, 1.0) * float(spread_weight)
            or abs(curr_iqr - prev_iqr) <= float(tol) * max(prev_iqr, curr_iqr, 1.0) * float(spread_weight)
        )
        trend_ok = max(prev_shift, curr_shift) <= float(tol) * scale * 1.25 * float(trend_weight)
        return bool(center_ok and spread_ok and trend_ok)

    def _metric_stable_relaxed(
        self,
        trace: List[float],
        *,
        window: int,
        tol: float,
        center_weight: float = 1.0,
        spread_weight: float = 1.0,
        trend_weight: float = 1.0,
    ) -> bool:
        stats = self._robust_trace_window(trace, window)
        if stats is None:
            return False
        prev_med = float(stats["prev_med"])
        curr_med = float(stats["curr_med"])
        prev_mad = float(stats["prev_mad"])
        curr_mad = float(stats["curr_mad"])
        prev_iqr = float(stats["prev_iqr"])
        curr_iqr = float(stats["curr_iqr"])
        prev_shift = float(stats["prev_shift"])
        curr_shift = float(stats["curr_shift"])
        scale = max(abs(prev_med), abs(curr_med), 1.0)
        center_ok = abs(curr_med - prev_med) <= float(tol) * scale * float(center_weight)
        spread_ok = (
            abs(curr_mad - prev_mad) <= float(tol) * max(prev_mad, curr_mad, 1.0) * float(spread_weight)
            or abs(curr_iqr - prev_iqr) <= float(tol) * max(prev_iqr, curr_iqr, 1.0) * float(spread_weight)
        )
        trend_ok = max(prev_shift, curr_shift) <= float(tol) * scale * 1.25 * float(trend_weight)
        return bool(center_ok and (spread_ok or trend_ok))

    @staticmethod
    def _phase_score(
        *,
        checks: Dict[str, bool],
        core_keys: Sequence[str],
        support_keys: Sequence[str],
        core_min_pass: int,
        support_min_pass: int,
        score_keys: Sequence[str],
        score_min: int,
    ) -> tuple[bool, int, int, int]:
        core_pass = int(sum(bool(checks.get(k)) for k in core_keys))
        support_pass = int(sum(bool(checks.get(k)) for k in support_keys))
        score = int(sum(bool(checks.get(k)) for k in score_keys))
        ready = bool(
            core_pass >= int(max(0, core_min_pass))
            and support_pass >= int(max(0, support_min_pass))
            and score >= int(max(0, score_min))
        )
        return ready, core_pass, support_pass, score

    def _phase_a_ready(
        self,
        traces: Dict[str, List[float]],
        *,
        structure_mode: str,
    ) -> tuple[bool, Dict[str, bool], Dict[str, int]]:
        if str(structure_mode) == "distributed":
            checks = {
                "mean_kappa": self._metric_stable(
                    traces["mean_kappa"],
                    window=int(self.geometry_window),
                    tol=float(self.geometry_tol),
                ),
                "sd_kappa": self._metric_stable(
                    traces["sd_kappa"],
                    window=int(self.geometry_window),
                    tol=float(self.geometry_tol) * 1.15,
                    spread_weight=1.15,
                ),
                "mean_log_prior_var": self._metric_stable(
                    traces["mean_log_prior_var"],
                    window=int(self.geometry_window),
                    tol=float(self.geometry_tol) * 1.20,
                ),
                "sd_log_prior_var": self._metric_stable(
                    traces["sd_log_prior_var"],
                    window=int(self.geometry_window),
                    tol=float(self.geometry_tol) * 1.40,
                    spread_weight=1.20,
                ),
                "group_norm_cv": self._metric_stable(
                    traces["group_norm_cv"],
                    window=int(self.geometry_window),
                    tol=float(self.geometry_tol) * 1.20,
                    trend_weight=1.15,
                ),
                "group_norm_entropy": self._metric_stable(
                    traces["group_norm_entropy"],
                    window=int(self.geometry_window),
                    tol=float(self.geometry_tol) * 1.15,
                    trend_weight=1.10,
                ),
            }
            checks["prior_var_ok"] = bool(checks["mean_log_prior_var"] or checks["sd_log_prior_var"])
            ready, core_pass, support_pass, score = self._phase_score(
                checks=checks,
                core_keys=("mean_kappa", "prior_var_ok"),
                support_keys=("sd_kappa", "group_norm_cv", "group_norm_entropy"),
                core_min_pass=int(self.phase_a_core_min_pass),
                support_min_pass=int(self.phase_a_support_min_pass),
                score_keys=("mean_kappa", "prior_var_ok", "sd_kappa", "group_norm_cv", "group_norm_entropy"),
                score_min=int(self.distributed_phase_a_score_min),
            )
            meta = {
                "core_pass": int(core_pass),
                "support_pass": int(support_pass),
                "score": int(score),
            }
            return ready, checks, meta

        checks = {
            "log_tau": self._metric_stable(
                traces["log_tau"],
                window=int(self.geometry_window),
                tol=float(self.geometry_tol) * 1.80,
                trend_weight=1.30,
            ),
            "mean_kappa": self._metric_stable(
                traces["mean_kappa"],
                window=int(self.geometry_window),
                tol=float(self.geometry_tol),
            ),
            "mean_log_prior_var": self._metric_stable(
                traces["mean_log_prior_var"],
                window=int(self.geometry_window),
                tol=float(self.geometry_tol) * 1.20,
            ),
            "sd_log_prior_var": self._metric_stable(
                traces["sd_log_prior_var"],
                window=int(self.geometry_window),
                tol=float(self.geometry_tol) * 1.40,
                spread_weight=1.20,
            ),
            "median_group_norm": self._metric_stable(
                traces["median_group_norm"],
                window=int(self.geometry_window),
                tol=float(self.geometry_tol) * 1.25,
                trend_weight=1.20,
            ),
        }
        checks["prior_var_ok"] = bool(checks["mean_log_prior_var"] or checks["sd_log_prior_var"])
        ready, core_pass, support_pass, score = self._phase_score(
            checks=checks,
            core_keys=("mean_kappa", "prior_var_ok"),
            support_keys=("median_group_norm", "log_tau"),
            core_min_pass=int(self.phase_a_core_min_pass),
            support_min_pass=int(self.phase_a_support_min_pass),
            score_keys=("mean_kappa", "prior_var_ok", "median_group_norm", "log_tau"),
            score_min=int(self.phase_a_score_min),
        )
        meta = {
            "core_pass": int(core_pass),
            "support_pass": int(support_pass),
            "score": int(score),
        }
        return ready, checks, meta

    def _phase_b_ready(
        self,
        traces: Dict[str, List[float]],
        *,
        structure_mode: str,
    ) -> tuple[bool, Dict[str, bool], Dict[str, int]]:
        if str(structure_mode) == "distributed":
            checks = {
                "beta_norm": self._metric_stable(
                    traces["beta_norm"],
                    window=int(self.transition_window),
                    tol=float(self.transition_tol),
                ),
                "resid_norm": self._metric_stable_relaxed(
                    traces["resid_norm"],
                    window=int(self.transition_window),
                    tol=float(self.transition_tol) * 1.30,
                    center_weight=1.15,
                    trend_weight=1.25,
                ),
                "mean_group_norm": self._metric_stable_relaxed(
                    traces["mean_group_norm"],
                    window=int(self.transition_window),
                    tol=float(self.transition_tol) * 1.40,
                    center_weight=1.20,
                    trend_weight=1.30,
                ),
                "sd_group_norm": self._metric_stable(
                    traces["sd_group_norm"],
                    window=int(self.transition_window),
                    tol=float(self.transition_tol) * 1.35,
                    spread_weight=1.20,
                ),
                "group_norm_entropy": self._metric_stable(
                    traces["group_norm_entropy"],
                    window=int(self.transition_window),
                    tol=float(self.transition_tol) * 1.25,
                    trend_weight=1.15,
                ),
                "mean_kappa": self._metric_stable(
                    traces["mean_kappa"],
                    window=int(self.transition_window),
                    tol=float(self.transition_tol) * 1.25,
                ),
            }
            ready, core_pass, support_pass, score = self._phase_score(
                checks=checks,
                core_keys=("mean_group_norm", "resid_norm"),
                support_keys=("beta_norm", "mean_kappa", "sd_group_norm", "group_norm_entropy"),
                core_min_pass=2,
                support_min_pass=1,
                score_keys=("beta_norm", "resid_norm", "mean_group_norm", "sd_group_norm", "group_norm_entropy", "mean_kappa"),
                score_min=3,
            )
            meta = {
                "core_pass": int(core_pass),
                "support_pass": int(support_pass),
                "score": int(score),
            }
            return ready, checks, meta

        checks = {
            "beta_norm": self._metric_stable(
                traces["beta_norm"],
                window=int(self.transition_window),
                tol=float(self.transition_tol),
            ),
            "top_group_norm": self._metric_stable(
                traces["top_group_norm"],
                window=int(self.transition_window),
                tol=float(self.transition_tol) * 1.15,
                trend_weight=1.20,
            ),
            "resid_norm": self._metric_stable(
                traces["resid_norm"],
                window=int(self.transition_window),
                tol=float(self.transition_tol) * 1.10,
            ),
            "log_tau": self._metric_stable(
                traces["log_tau"],
                window=int(self.transition_window),
                tol=float(self.transition_tol) * 1.35,
                trend_weight=1.25,
            ),
            "mean_kappa": self._metric_stable(
                traces["mean_kappa"],
                window=int(self.transition_window),
                tol=float(self.transition_tol) * 1.15,
            ),
        }
        ready, core_pass, support_pass, score = self._phase_score(
            checks=checks,
            core_keys=("beta_norm", "resid_norm"),
            support_keys=("top_group_norm", "log_tau", "mean_kappa"),
            core_min_pass=int(self.phase_b_core_min_pass),
            support_min_pass=int(self.phase_b_support_min_pass),
            score_keys=("beta_norm", "resid_norm", "top_group_norm", "log_tau", "mean_kappa"),
            score_min=int(self.phase_b_score_min),
        )
        meta = {
            "core_pass": int(core_pass),
            "support_pass": int(support_pass),
            "score": int(score),
        }
        return ready, checks, meta

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

        try:
            beta = np.linalg.solve(XtX + 1e-3 * np.eye(p), Xty)
        except np.linalg.LinAlgError:
            beta = np.zeros(p)
        sigma2 = max(float(np.var(y - X @ beta)), self.jitter)
        tau = float(tau0_eff)
        lam = np.ones(p)
        kappa = np.full(G, float(self.alpha_kappa) / (self.alpha_kappa + self.beta_kappa))
        log_sigma = 0.5 * math.log(max(sigma2, self.jitter))
        log_tau = math.log(max(tau, self.jitter))
        log_lam = np.zeros(p)
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
            logit_kappa = np.asarray(initial_state.get("logit_kappa", logit_kappa), dtype=float).reshape(-1)
            if logit_kappa.size != G:
                logit_kappa = np.zeros(G)
            sigma2 = max(float(math.exp(2.0 * log_sigma)), self.jitter)
            tau = max(float(math.exp(log_tau)), self.jitter)
            lam = np.maximum(np.exp(log_lam), self.jitter)
            kappa = 1.0 / (1.0 + np.exp(-logit_kappa))

        sample_target = max(
            0,
            (int(max(1, iters)) - int(max(0, min(burnin, max(1, iters) - 1))) + int(self.thin) - 1) // int(self.thin),
        )
        beta_draws = np.zeros((sample_target, p))
        sigma_draws = np.zeros(sample_target)
        tau_draws = np.zeros(sample_target)
        lam_draws = np.ones((sample_target, p))
        kappa_draws = np.zeros((sample_target, G))
        keep_i = 0

        traces: Dict[str, List[float]] = {
            "log_tau": [],
            "mean_kappa": [],
            "mean_log_prior_var": [],
            "sd_log_prior_var": [],
            "median_group_norm": [],
            "mean_group_norm": [],
            "sd_group_norm": [],
            "group_norm_cv": [],
            "group_norm_entropy": [],
            "sd_kappa": [],
            "beta_norm": [],
            "top_group_norm": [],
            "resid_norm": [],
            "concentration_score": [],
        }

        skip_adaptive = bool(self.adaptive_burnin and int(burnin) <= 0 and initial_state is not None)
        phase_a_done = (not bool(self.adaptive_burnin)) or bool(skip_adaptive)
        phase_b_done = (not bool(self.adaptive_burnin)) or bool(skip_adaptive)
        phase_a_iters = 0
        phase_b_iters = 0
        burnin_total = 0
        total_iters = 0
        warmup_warning = ""
        phase_a_checks_last: Dict[str, bool] = {}
        phase_b_checks_last: Dict[str, bool] = {}
        phase_a_meta_last: Dict[str, int] = {}
        phase_b_meta_last: Dict[str, int] = {}
        block_refresh_count = 0
        block_refresh_group_hist = np.zeros(G, dtype=int)
        block_refresh_steps = 0
        adaptive_concentration_threshold_last = float("nan")
        structure_mode = "unclear"
        distributed_block_refresh_count = 0

        max_total = int(max(
            sample_target * int(self.thin) + 1,
            int(self.phase_a_max_iters) + int(self.phase_b_max_iters) + sample_target * int(self.thin),
        ))

        iterator = range(max_total)
        if bool(self.progress_bar):
            try:
                from simulation_project.src.core.utils.logging_utils import progress

                iterator = progress(iterator, total=max_total, desc="GR-RHS Gibbs staged")
            except Exception:
                pass

        for _ in iterator:
            total_iters += 1
            tau2 = tau ** 2
            sigma2 = math.exp(2.0 * log_sigma)
            lam2 = lam ** 2
            kappa_j = kappa[group_id]
            hyper_only_phase = bool(
                (not phase_a_done)
                and bool(self.adaptive_burnin)
                and phase_a_iters < int(max(0, self.phase_a_hyper_only_iters))
            )

            if not hyper_only_phase:
                v = self._v_arr(sigma2, tau2, lam2, kappa_j, self.jitter)
                if n < p:
                    beta = beta_sample_woodbury(X, y, sigma2, v, rng, jitter=self.jitter)
                else:
                    beta = beta_sample_cholesky(XtX, Xty, sigma2, v, rng, jitter=self.jitter)
                Xbeta = X @ beta

                def _lc_s(r: float) -> float:
                    return self._lc_log_sigma(r, beta, y, Xbeta, tau2, lam2, kappa_j)

                log_sigma = slice_sample_1d(_lc_s, log_sigma, rng, width=self.slice_width_log, max_steps=self.slice_max_steps)
                sigma2 = math.exp(2.0 * log_sigma)
            else:
                Xbeta = X @ beta

            tau_scale = float(tau0_eff) * math.sqrt(max(sigma2, self.jitter))

            def _lc_t(u: float) -> float:
                return self._lc_log_tau(u, beta, sigma2, lam2, kappa_j, tau_scale)

            log_tau = slice_sample_1d(_lc_t, log_tau, rng, width=self.slice_width_log, max_steps=self.slice_max_steps)
            tau = math.exp(log_tau)
            tau2 = tau ** 2

            if self.use_local_scale:
                for j in range(p):
                    def _lc_lj(s: float, _j: int = j) -> float:
                        return self._lc_log_lam_j(s, float(beta[_j]), sigma2, tau2, float(kappa_j[_j]))

                    log_lam[j] = slice_sample_1d(_lc_lj, log_lam[j], rng, width=self.slice_width_log, max_steps=self.slice_max_steps)
                lam = np.exp(log_lam)
                lam2 = lam ** 2

            if self.shared_kappa:
                def _lc_ksh(w: float) -> float:
                    return self._lc_logit_kappa_g(w, beta, sigma2, tau2, lam2)

                logit_kappa[0] = slice_sample_1d(_lc_ksh, logit_kappa[0], rng, width=self.slice_width_logit, max_steps=self.slice_max_steps)
                kappa[:] = 1.0 / (1.0 + math.exp(-logit_kappa[0]))
            else:
                for g, members in enumerate(groups):
                    idx = np.asarray(members, dtype=int)

                    def _lc_kg(w: float, _idx: np.ndarray = idx) -> float:
                        return self._lc_logit_kappa_g(w, beta[_idx], sigma2, tau2, lam2[_idx])

                    logit_kappa[g] = slice_sample_1d(_lc_kg, logit_kappa[g], rng, width=self.slice_width_logit, max_steps=self.slice_max_steps)
                kappa = 1.0 / (1.0 + np.exp(-logit_kappa))

            # Extra beta refresh to improve coupling in concentrated-signal regimes:
            # after hyper-parameters move, redraw beta again when one/few groups dominate.
            group_norms_pre = self._group_norms(beta, groups)
            concentration_score = self._concentration_score(group_norms_pre)
            group_entropy_pre = self._group_norm_entropy(group_norms_pre)
            top2_mass_pre = self._topk_group_mass(group_norms_pre, 2)
            structure_mode = self._structure_mode(
                concentration_score=concentration_score,
                group_entropy=group_entropy_pre,
                top2_mass=top2_mass_pre,
            )
            in_phase_a_late = bool(
                (not phase_a_done)
                and (not hyper_only_phase)
                and phase_a_iters >= max(int(self.phase_a_hyper_only_iters), int(self.min_phase_a_iters) // 2)
            )
            in_phase_b = bool(phase_a_done and (not phase_b_done))
            do_refresh = False
            refresh_repeats = 0
            if bool(self.phase_a_late_beta_refresh) and in_phase_a_late:
                interval = int(max(1, self.phase_a_refresh_interval))
                if phase_a_iters % interval == 0:
                    do_refresh = True
                    refresh_repeats = int(max(1, self.phase_a_refresh_repeats))
            elif bool(self.phase_b_extra_beta_refresh) and in_phase_b:
                if phase_b_iters <= int(max(0, self.phase_b_initial_extra_refresh_steps)):
                    do_refresh = True
                    refresh_repeats = int(max(1, self.phase_b_initial_refresh_repeats))

            if do_refresh:
                tau2 = tau ** 2
                sigma2 = math.exp(2.0 * log_sigma)
                lam2 = lam ** 2
                for _refresh in range(refresh_repeats):
                    beta = self._global_beta_refresh(
                        X=X,
                        y=y,
                        beta=beta,
                        sigma2=sigma2,
                        tau2=tau2,
                        lam2=lam2,
                        kappa=kappa,
                        group_id=group_id,
                        rng=rng,
                    )

            use_block_refresh = False
            block_refresh_repeats = 0
            target_groups: list[int] = []
            if bool(self.concentrated_block_refresh):
                if in_phase_a_late:
                    interval = int(max(1, self.phase_a_block_refresh_interval))
                    if phase_a_iters % interval == 0:
                        use_block_refresh = True
                        block_refresh_repeats = 1
                elif in_phase_b and phase_b_iters <= int(max(0, self.phase_b_block_refresh_steps)):
                    use_block_refresh = True
                    block_refresh_repeats = int(max(1, self.phase_b_block_refresh_repeats))

            if use_block_refresh:
                tau2 = tau ** 2
                sigma2 = math.exp(2.0 * log_sigma)
                lam2 = lam ** 2
                if str(structure_mode) == "distributed":
                    norms = self._group_norms(beta, groups)
                    order = np.argsort(-norms)
                    top_k = int(max(0, min(int(self.distributed_block_top_groups), int(order.size))))
                    target_groups = [int(v) for v in order[:top_k]]
                    adaptive_concentration_threshold_last = float("nan")
                    if target_groups:
                        distributed_block_refresh_count += 1
                else:
                    target_groups, adaptive_concentration_threshold_last = self._target_group_ids(
                        beta=beta,
                        groups=groups,
                        concentration_score=concentration_score,
                    )
                if target_groups:
                    block_refresh_steps += 1
                    block_refresh_count += int(len(target_groups)) * int(max(1, block_refresh_repeats))
                    for gid in target_groups:
                        block_refresh_group_hist[int(gid)] += int(max(1, block_refresh_repeats))
                for _ in range(block_refresh_repeats):
                    beta = self._group_block_beta_refresh(
                        X=X,
                        y=y,
                        beta=beta,
                        sigma2=sigma2,
                        tau2=tau2,
                        lam2=lam2,
                        kappa=kappa,
                        groups=groups,
                        group_id=group_id,
                        target_groups=target_groups,
                        rng=rng,
                    )

            group_norms = self._group_norms(beta, groups)
            top_group_norm = float(np.max(group_norms)) if group_norms.size else 0.0
            median_group_norm = float(np.median(group_norms)) if group_norms.size else 0.0
            mean_group_norm = float(np.mean(group_norms)) if group_norms.size else 0.0
            sd_group_norm = float(np.std(group_norms)) if group_norms.size else 0.0
            group_norm_cv = self._safe_cv(group_norms)
            group_norm_entropy = self._group_norm_entropy(group_norms)
            concentration_score = self._concentration_score(group_norms)
            top2_mass = self._topk_group_mass(group_norms, 2)
            resid_norm = float(np.linalg.norm(y - X @ beta) / math.sqrt(max(n, 1)))
            prior_var = self._v_arr(sigma2, tau2, lam2, kappa[group_id], self.jitter)
            log_prior_var = np.log(np.maximum(prior_var, self.jitter))
            traces["log_tau"].append(float(log_tau))
            traces["mean_kappa"].append(float(np.mean(kappa)))
            traces["sd_kappa"].append(float(np.std(kappa)))
            traces["mean_log_prior_var"].append(float(np.mean(log_prior_var)))
            traces["sd_log_prior_var"].append(float(np.std(log_prior_var)))
            traces["median_group_norm"].append(median_group_norm)
            traces["mean_group_norm"].append(mean_group_norm)
            traces["sd_group_norm"].append(sd_group_norm)
            traces["group_norm_cv"].append(group_norm_cv)
            traces["group_norm_entropy"].append(group_norm_entropy)
            traces["beta_norm"].append(float(np.linalg.norm(beta)))
            traces["top_group_norm"].append(top_group_norm)
            traces["resid_norm"].append(resid_norm)
            traces["concentration_score"].append(float(concentration_score))

            if not phase_a_done:
                phase_a_iters += 1
                burnin_total += 1
                if phase_a_iters >= int(self.min_phase_a_iters):
                    phase_a_done, phase_a_checks_last, phase_a_meta_last = self._phase_a_ready(
                        traces,
                        structure_mode=structure_mode,
                    )
                if (not phase_a_done) and phase_a_iters >= int(self.phase_a_max_iters):
                    if not phase_a_checks_last:
                        _ready, phase_a_checks_last, phase_a_meta_last = self._phase_a_ready(
                            traces,
                            structure_mode=structure_mode,
                        )
                    phase_a_done = True
                    warmup_warning = "geometry_not_stable"
                continue

            if not phase_b_done:
                phase_b_iters += 1
                burnin_total += 1
                if phase_b_iters >= int(self.min_phase_b_iters):
                    phase_b_done, phase_b_checks_last, phase_b_meta_last = self._phase_b_ready(
                        traces,
                        structure_mode=structure_mode,
                    )
                if (not phase_b_done) and phase_b_iters >= int(self.phase_b_max_iters):
                    if not phase_b_checks_last:
                        _ready, phase_b_checks_last, phase_b_meta_last = self._phase_b_ready(
                            traces,
                            structure_mode=structure_mode,
                        )
                    phase_b_done = True
                    if not warmup_warning:
                        warmup_warning = "transition_not_stable"
                continue

            stationary_idx = total_iters - burnin_total - 1
            if stationary_idx >= 0 and stationary_idx % int(self.thin) == 0 and keep_i < sample_target:
                beta_draws[keep_i] = beta
                sigma_draws[keep_i] = math.exp(log_sigma)
                tau_draws[keep_i] = tau
                lam_draws[keep_i] = lam.copy()
                kappa_draws[keep_i] = kappa.copy()
                keep_i += 1
                if keep_i >= sample_target:
                    break

        if keep_i < sample_target:
            beta_draws = beta_draws[:keep_i]
            sigma_draws = sigma_draws[:keep_i]
            tau_draws = tau_draws[:keep_i]
            lam_draws = lam_draws[:keep_i]
            kappa_draws = kappa_draws[:keep_i]

        phase_a_converged_flag = bool(
            skip_adaptive or phase_a_iters < int(self.phase_a_max_iters) or warmup_warning != "geometry_not_stable"
        )
        phase_b_converged_flag = bool(
            skip_adaptive or phase_b_iters < int(self.phase_b_max_iters) or warmup_warning not in {"transition_not_stable"}
        )
        concentrated_phase_a_relax = bool(
            (not phase_a_converged_flag)
            and phase_b_converged_flag
            and bool(phase_a_checks_last.get("mean_kappa"))
            and bool(phase_a_checks_last.get("prior_var_ok"))
            and concentration_score >= 0.40
        )
        warmup_warning_final = str(warmup_warning)
        if concentrated_phase_a_relax and warmup_warning_final == "geometry_not_stable":
            phase_a_converged_flag = True
            warmup_warning_final = ""

        return {
            "beta": beta_draws,
            "sigma": sigma_draws,
            "tau": tau_draws,
            "lambda": lam_draws,
            "kappa": kappa_draws,
            "last_state": {
                "beta": np.asarray(beta, dtype=float).copy(),
                "log_sigma": float(log_sigma),
                "log_tau": float(log_tau),
                "log_lam": np.asarray(log_lam, dtype=float).copy(),
                "logit_kappa": np.asarray(logit_kappa, dtype=float).copy(),
            },
            "phase_info": {
                "phase_a_iters": int(phase_a_iters),
                "phase_b_iters": int(phase_b_iters),
                "actual_burnin": int(burnin_total),
                "phase_a_converged": bool(phase_a_converged_flag),
                "phase_b_converged": bool(phase_b_converged_flag),
                "phase_a_hyper_only_iters": int(min(phase_a_iters, int(max(0, self.phase_a_hyper_only_iters)))),
                "phase_a_checks": dict(phase_a_checks_last),
                "phase_a_score": int(phase_a_meta_last.get("score", 0)),
                "phase_a_core_pass": int(phase_a_meta_last.get("core_pass", 0)),
                "phase_a_support_pass": int(phase_a_meta_last.get("support_pass", 0)),
                "phase_b_checks": dict(phase_b_checks_last),
                "phase_b_score": int(phase_b_meta_last.get("score", 0)),
                "phase_b_core_pass": int(phase_b_meta_last.get("core_pass", 0)),
                "phase_b_support_pass": int(phase_b_meta_last.get("support_pass", 0)),
                "structure_mode": str(structure_mode),
                "concentration_score_last": float(concentration_score),
                "top2_group_mass_last": float(top2_mass),
                "group_norm_entropy_last": float(group_norm_entropy),
                "adaptive_concentration_threshold_last": float(adaptive_concentration_threshold_last),
                "target_group_ids_last": list(target_groups),
                "block_refresh_count": int(block_refresh_count),
                "block_refresh_steps": int(block_refresh_steps),
                "block_refresh_group_histogram": np.asarray(block_refresh_group_hist, dtype=int).tolist(),
                "distributed_block_refresh_count": int(distributed_block_refresh_count),
                "concentrated_phase_a_relax_applied": bool(concentrated_phase_a_relax),
                "resume_no_burnin_used": bool(skip_adaptive),
                "warmup_warning": str(warmup_warning_final),
                "total_iters": int(total_iters),
            },
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[List[List[int]]] = None,
    ) -> "GRRHS_Gibbs_Staged":
        super().fit(X, y, groups=groups)
        self.sampler_diagnostics_["backend"] = "simcore_gibbs_staged"
        self.sampler_diagnostics_["adaptive_burnin"] = bool(self.adaptive_burnin)
        self.sampler_diagnostics_["phase_a_max_iters"] = int(self.phase_a_max_iters)
        self.sampler_diagnostics_["phase_b_max_iters"] = int(self.phase_b_max_iters)
        self.sampler_diagnostics_["phase_a_hyper_only_iters"] = int(self.phase_a_hyper_only_iters)
        self.sampler_diagnostics_["min_phase_a_iters"] = int(self.min_phase_a_iters)
        self.sampler_diagnostics_["min_phase_b_iters"] = int(self.min_phase_b_iters)
        self.sampler_diagnostics_["geometry_window"] = int(self.geometry_window)
        self.sampler_diagnostics_["transition_window"] = int(self.transition_window)
        self.sampler_diagnostics_["geometry_tol"] = float(self.geometry_tol)
        self.sampler_diagnostics_["transition_tol"] = float(self.transition_tol)
        self.sampler_diagnostics_["phase_a_late_beta_refresh"] = bool(self.phase_a_late_beta_refresh)
        self.sampler_diagnostics_["phase_b_extra_beta_refresh"] = bool(self.phase_b_extra_beta_refresh)
        self.sampler_diagnostics_["phase_a_refresh_interval"] = int(self.phase_a_refresh_interval)
        self.sampler_diagnostics_["phase_a_refresh_repeats"] = int(self.phase_a_refresh_repeats)
        self.sampler_diagnostics_["phase_b_initial_extra_refresh_steps"] = int(self.phase_b_initial_extra_refresh_steps)
        self.sampler_diagnostics_["phase_b_initial_refresh_repeats"] = int(self.phase_b_initial_refresh_repeats)
        self.sampler_diagnostics_["phase_a_core_min_pass"] = int(self.phase_a_core_min_pass)
        self.sampler_diagnostics_["phase_a_support_min_pass"] = int(self.phase_a_support_min_pass)
        self.sampler_diagnostics_["phase_a_score_min"] = int(self.phase_a_score_min)
        self.sampler_diagnostics_["phase_b_core_min_pass"] = int(self.phase_b_core_min_pass)
        self.sampler_diagnostics_["phase_b_support_min_pass"] = int(self.phase_b_support_min_pass)
        self.sampler_diagnostics_["phase_b_score_min"] = int(self.phase_b_score_min)
        self.sampler_diagnostics_["concentrated_block_refresh"] = bool(self.concentrated_block_refresh)
        self.sampler_diagnostics_["concentrated_top_groups"] = int(self.concentrated_top_groups)
        self.sampler_diagnostics_["concentrated_refresh_score_delta"] = float(self.concentrated_refresh_min_score)
        self.sampler_diagnostics_["concentrated_refresh_baseline_ratio"] = float(self.concentrated_refresh_baseline_ratio)
        self.sampler_diagnostics_["phase_a_block_refresh_interval"] = int(self.phase_a_block_refresh_interval)
        self.sampler_diagnostics_["phase_b_block_refresh_steps"] = int(self.phase_b_block_refresh_steps)
        self.sampler_diagnostics_["phase_b_block_refresh_repeats"] = int(self.phase_b_block_refresh_repeats)
        self.sampler_diagnostics_["structure_aware_warmup"] = bool(self.structure_aware_warmup)
        self.sampler_diagnostics_["distributed_entropy_threshold"] = float(self.distributed_entropy_threshold)
        self.sampler_diagnostics_["concentrated_score_threshold"] = float(self.concentrated_score_threshold)
        self.sampler_diagnostics_["concentrated_top2_mass_threshold"] = float(self.concentrated_top2_mass_threshold)
        self.sampler_diagnostics_["distributed_top2_mass_threshold"] = float(self.distributed_top2_mass_threshold)
        self.sampler_diagnostics_["distributed_block_top_groups"] = int(self.distributed_block_top_groups)
        self.sampler_diagnostics_["distributed_phase_a_score_min"] = int(self.distributed_phase_a_score_min)
        self.sampler_diagnostics_["distributed_phase_b_score_min"] = int(self.distributed_phase_b_score_min)
        if isinstance(self.chain_phase_infos_, list) and self.chain_phase_infos_:
            self.sampler_diagnostics_["warmup_warning"] = next(
                (str(info.get("warmup_warning", "")) for info in self.chain_phase_infos_ if str(info.get("warmup_warning", "")).strip()),
                "",
            )
        return self


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
    Full (lambda active)    2p+G+2       p+G+2
    Profile (lam=1)         p+G+2        G+2       <- orders-of-magnitude smaller

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
    kappa_samples_: Optional[np.ndarray] = field(default=None, init=False)
    c2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_mean_: Optional[float] = field(default=None, init=False)
    tau_mean_: Optional[float] = field(default=None, init=False)
    lambda_mean_: Optional[np.ndarray] = field(default=None, init=False)
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
        use_local = bool(self.use_local_scale)
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
                # lam=1 => v depends only on group:
                # v_g = sigma2*kappa_g*tau2 / (sigma2*kappa_g + (1-kappa_g)*tau2)
                r_g = tau2 / (sigma2 + _EPS)
                v_g = sigma2 * kappa * r_g / (kappa + (1.0 - kappa) * r_g + _EPS)  # (G,)
                # Sigma_y = sum_g v_g * M_g + sigma2 * I  [M_g = X_g X_g^T precomputed]
                Sigma_y = jnp.einsum("g,gnm->nm", v_g, group_XXT) + sigma2 * jnp.eye(n, dtype=X.dtype)
            else:
                kappa_j = kappa[group_id]
                lam2 = lam * lam
                r = tau2 * lam2 / (sigma2 + _EPS)
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

        profile_mode = not bool(self.use_local_scale)

        # Precompute M_g = X_g X_g^T for profile mode
        group_XXT_jnp = None
        if profile_mode:
            M_list = []
            for g_idx, members in enumerate(groups_use):
                Xg = X_arr[:, np.asarray(members, dtype=int)]
                M_list.append(Xg @ Xg.T)  # n x n
            group_XXT_jnp = jnp.asarray(np.stack(M_list, axis=0))  # G x n x n

        model_fn = self._build_model(profile_mode=profile_mode, group_XXT=group_XXT_jnp)

        init_params_use = _normalize_init_params(
            self.init_params,
            num_chains=int(self.num_chains),
        )
        init_strategy = (
            numpyro.infer.init_to_median(num_samples=15)
            if _should_use_median_init(
                use_init_to_median=bool(self.use_init_to_median),
                init_params=init_params_use,
            )
            else numpyro.infer.init_to_uniform()
        )
        kernel = NUTS(
            model_fn,
            target_accept_prob=float(self.target_accept_prob),
            dense_mass=bool(self.dense_mass),
            max_tree_depth=int(self.max_tree_depth),
            init_strategy=init_strategy,
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
        kappa_flat = _get_flat("kappa")          # (S, G)

        S = sigma_flat.shape[0] if sigma_flat is not None else 0
        beta_draws = np.zeros((S, p))

        for i in range(S):
            sig2_i = float(sigma_flat[i]) ** 2 if sigma_flat is not None else 1.0
            tau2_i = float(tau_flat[i]) ** 2 if tau_flat is not None else 1.0
            lam2_i = lam_flat[i] ** 2 if lam_flat is not None else np.ones(p)
            kappa_j_i = kappa_flat[i][gid] if kappa_flat is not None else np.full(p, 0.5)

            r_i = tau2_i * lam2_i / (sig2_i + 1e-12)
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
            "nuts_dim": G + 2 if profile_mode else p + G + 2,
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
            "kappa_mean": self.kappa_mean_,
            "c2_mean": self.c2_mean_,
            "lambda_mean": self.lambda_mean_,
        }


