from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from jax.nn import softplus
from jax.scipy.sparse.linalg import cg
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.svi import SVIState, _make_loss_fn
from numpyro.optim import Adam, _value_and_grad

from grrhs.utils.logging_utils import progress

_EPS = 1e-12


def _split_groups(p: int, groups: List[List[int]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    g = len(groups)
    gid = np.empty(p, dtype=np.int32)
    gsz = np.zeros(g, dtype=np.int32)
    for k, idxs in enumerate(groups):
        gid[idxs] = k
        gsz[k] = len(idxs)
    return jnp.array(gid), jnp.array(gsz)


def _gather_beta_from_groups(beta_groups: List[jnp.ndarray], groups: List[List[int]], p: int) -> jnp.ndarray:
    beta = jnp.zeros((p,))
    for g, idxs in enumerate(groups):
        beta = beta.at[jnp.array(idxs)].set(beta_groups[g])
    return beta


def _hutchinson_trace(matvec_fn, dim: int, key: jax.Array, num_samples: int) -> jax.Array:
    if num_samples <= 0:
        raise ValueError("num_samples for Hutchinson estimator must be positive")
    keys = random.split(key, num_samples)

    def single(k):
        z = random.rademacher(k, (dim,), dtype=jnp.float32)
        return jnp.dot(z, matvec_fn(z))

    return jnp.mean(jax.vmap(single)(keys))


@dataclass
class GRRHS_SVI_Numpyro:
    c: float = 1.0
    tau0: float = 0.1
    eta: float = 0.5
    s0: float = 1.0
    alpha_c: float = 2.0
    beta_c: float = 2.0

    num_steps: int = 3000
    lr: float = 1e-2
    seed: int = 42
    batch_size: Optional[int] = None
    use_hutchinson: bool = True
    hutchinson_samples: int = 8
    natural_gradient: bool = False
    cg_tol: float = 1e-5
    cg_max_iters: int = 50
    natgrad_damping: float = 1e-3
    coupling_clip: Optional[float] = 3.0
    covariance_damping: float = 0.0

    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_mean_: Optional[float] = field(default=None, init=False)
    tau_mean_: Optional[float] = field(default=None, init=False)
    a_mean_: Optional[np.ndarray] = field(default=None, init=False)
    c2_mean_: Optional[np.ndarray] = field(default=None, init=False)
    phi_mean_: Optional[np.ndarray] = field(default=None, init=False)  # alias of a_mean_
    lambda_mean_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)

    group_id_: Optional[jnp.ndarray] = field(default=None, init=False)
    group_sizes_: Optional[jnp.ndarray] = field(default=None, init=False)
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_samples_: Optional[np.ndarray] = field(default=None, init=False)
    a_samples_: Optional[np.ndarray] = field(default=None, init=False)
    c2_samples_: Optional[np.ndarray] = field(default=None, init=False)
    phi_samples_: Optional[np.ndarray] = field(default=None, init=False)  # alias of a_samples_

    def _model(self, X: jnp.ndarray, y: jnp.ndarray, groups: List[List[int]]):
        n, p = X.shape
        _, group_sizes = _split_groups(p, groups)
        eta_g = self.eta / jnp.sqrt(jnp.maximum(group_sizes, 1))

        sigma = numpyro.sample("sigma", dist.HalfCauchy(self.s0))
        tau = numpyro.sample("tau", dist.HalfCauchy(self.tau0))
        lam = numpyro.sample("lambda", dist.HalfCauchy(jnp.ones((p,))).to_event(1))

        beta_groups: List[jnp.ndarray] = []
        for g, idxs_raw in enumerate(groups):
            idxs = jnp.array(idxs_raw)
            lam_g = lam[idxs]
            a_g = numpyro.sample(f"a_g_{g}", dist.HalfNormal(eta_g[g]))
            c2_g = numpyro.sample(f"c2_g_{g}", dist.InverseGamma(self.alpha_c, self.beta_c))

            lam2 = lam_g * lam_g
            a2 = a_g * a_g
            num = c2_g * lam2 * a2
            den = c2_g + (tau * tau) * lam2 * a2 + _EPS
            tl2 = num / den
            beta_scale = tau * jnp.sqrt(jnp.maximum(tl2, _EPS))
            beta_g = numpyro.sample(f"beta_g_{g}", dist.Normal(jnp.zeros_like(beta_scale), beta_scale).to_event(1))
            beta_groups.append(beta_g)

        beta = _gather_beta_from_groups(beta_groups, groups, p)
        mean = X @ beta
        with numpyro.plate("data", n):
            numpyro.sample("y", dist.Normal(mean, sigma), obs=y)

    def _guide(self, X: jnp.ndarray, y: jnp.ndarray, groups: List[List[int]]):
        _, p = X.shape

        mu_log_tau = numpyro.param("mu_log_tau", jnp.array(0.0))
        rho_log_tau = numpyro.param("rho_log_tau", jnp.array(-1.0))
        std_log_tau = softplus(rho_log_tau)

        mu_log_sigma = numpyro.param("mu_log_sigma", jnp.array(0.0))
        rho_log_sigma = numpyro.param("rho_log_sigma", jnp.array(-1.0))
        std_log_sigma = softplus(rho_log_sigma)

        mu_log_lam = numpyro.param("mu_log_lambda", jnp.zeros((p,)))
        rho_log_lam = numpyro.param("rho_log_lambda", -jnp.ones((p,)))
        std_log_lam = softplus(rho_log_lam)

        if self.coupling_clip is not None:
            clip = float(self.coupling_clip)
            coupling_constraint = constraints.interval(-clip, clip)
        else:
            coupling_constraint = constraints.real

        coupling_a = numpyro.param("coupling_a", jnp.zeros((p,)), constraint=coupling_constraint)
        coupling_rho = numpyro.param("coupling_rho", jnp.array(0.0), constraint=coupling_constraint)

        u_shared = numpyro.sample("u_shared", dist.Normal(0.0, 1.0), infer={"is_auxiliary": True})

        log_tau = numpyro.sample(
            "log_tau_aux",
            dist.Normal(mu_log_tau + coupling_rho * u_shared, std_log_tau),
            infer={"is_auxiliary": True},
        )
        numpyro.sample("tau", dist.Delta(jnp.exp(log_tau)))

        log_sigma = numpyro.sample(
            "log_sigma_aux",
            dist.Normal(mu_log_sigma, std_log_sigma),
            infer={"is_auxiliary": True},
        )
        numpyro.sample("sigma", dist.Delta(jnp.exp(log_sigma)))

        log_lam = numpyro.sample(
            "log_lambda_aux",
            dist.Normal(mu_log_lam + coupling_a * u_shared, std_log_lam).to_event(1),
            infer={"is_auxiliary": True},
        )
        numpyro.sample("lambda", dist.Delta(jnp.exp(log_lam)).to_event(1))

        for g, idxs_raw in enumerate(groups):
            idxs = jnp.array(idxs_raw)
            pg = int(idxs.shape[0])

            loc_beta = numpyro.param(f"loc_beta_g_{g}", jnp.zeros((pg,)))
            L_flat = numpyro.param(f"L_beta_g_{g}", jnp.eye(pg))
            L = jnp.tril(L_flat)
            diag = jnp.diag(softplus(jnp.diag(L)) + 1e-5)
            L = L - jnp.diag(jnp.diag(L)) + diag
            beta_aux = numpyro.sample(
                f"beta_aux_g_{g}",
                dist.MultivariateNormal(loc_beta, scale_tril=L),
                infer={"is_auxiliary": True},
            )
            numpyro.sample(f"beta_g_{g}", dist.Delta(beta_aux).to_event(1))

            mu_log_a = numpyro.param(f"mu_log_a_g_{g}", jnp.array(0.0))
            rho_log_a = numpyro.param(f"rho_log_a_g_{g}", jnp.array(-1.0))
            log_a = numpyro.sample(
                f"log_a_aux_g_{g}",
                dist.Normal(mu_log_a, softplus(rho_log_a)),
                infer={"is_auxiliary": True},
            )
            numpyro.sample(f"a_g_{g}", dist.Delta(jnp.exp(log_a)))

            mu_log_c2 = numpyro.param(f"mu_log_c2_g_{g}", jnp.array(0.0))
            rho_log_c2 = numpyro.param(f"rho_log_c2_g_{g}", jnp.array(-1.0))
            log_c2 = numpyro.sample(
                f"log_c2_aux_g_{g}",
                dist.Normal(mu_log_c2, softplus(rho_log_c2)),
                infer={"is_auxiliary": True},
            )
            numpyro.sample(f"c2_g_{g}", dist.Delta(jnp.exp(log_c2)))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[List[List[int]]] = None,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
        num_samples_export: int = 1000,
    ) -> "GRRHS_SVI_Numpyro":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        _, p = X.shape
        if groups is None:
            groups = [[j] for j in range(p)]
        self.groups_ = groups
        gid, gsz = _split_groups(p, groups)
        self.group_id_ = gid
        self.group_sizes_ = gsz

        steps = int(num_steps or self.num_steps)
        learn_rate = float(lr or self.lr)
        rng_key = random.PRNGKey(self.seed)
        optimizer = Adam(learn_rate)

        num_particles = max(1, int(self.hutchinson_samples)) if self.use_hutchinson else 1
        svi = SVI(self._model, self._guide, optimizer, loss=Trace_ELBO(num_particles=num_particles))

        args = (jnp.array(X), jnp.array(y), groups)
        state = svi.init(rng_key, *args)

        for _ in progress(range(steps), total=steps, desc="SVI training"):
            state, loss = self._matrix_free_update(svi, state, args)
            if jnp.isnan(loss):
                raise FloatingPointError("SVI loss produced NaN")

        params = svi.get_params(state)
        return_sites = [
            "tau",
            "sigma",
            "lambda",
            *[f"a_g_{g}" for g in range(len(groups))],
            *[f"c2_g_{g}" for g in range(len(groups))],
            *[f"beta_g_{g}" for g in range(len(groups))],
        ]
        post = Predictive(self._guide, params=params, num_samples=num_samples_export, return_sites=return_sites)(
            rng_key, jnp.array(X), jnp.array(y), groups
        )

        beta_samples = []
        for s in range(num_samples_export):
            beta_groups = [post[f"beta_g_{g}"][s] for g in range(len(groups))]
            beta = _gather_beta_from_groups(beta_groups, groups, p)
            beta_samples.append(np.asarray(beta))
        coef_samples = np.stack(beta_samples, axis=0)
        self.coef_samples_ = coef_samples
        self.coef_mean_ = coef_samples.mean(axis=0)

        lambda_samples = np.asarray(post["lambda"])
        if lambda_samples.ndim == 1:
            lambda_samples = lambda_samples.reshape(-1, 1)
        self.lambda_samples_ = lambda_samples
        self.lambda_mean_ = lambda_samples.mean(axis=0)

        tau_samples = np.asarray(post["tau"]).reshape(-1)
        sigma_samples = np.asarray(post["sigma"]).reshape(-1)
        self.tau_samples_ = tau_samples
        self.sigma_samples_ = sigma_samples
        self.tau_mean_ = float(tau_samples.mean())
        self.sigma_mean_ = float(sigma_samples.mean())

        a_mat = np.stack([np.asarray(post[f"a_g_{g}"]).reshape(-1) for g in range(len(groups))], axis=1)
        c2_mat = np.stack([np.asarray(post[f"c2_g_{g}"]).reshape(-1) for g in range(len(groups))], axis=1)
        self.a_samples_ = a_mat
        self.c2_samples_ = c2_mat
        self.a_mean_ = a_mat.mean(axis=0)
        self.c2_mean_ = c2_mat.mean(axis=0)

        self.phi_samples_ = a_mat
        self.phi_mean_ = self.a_mean_
        self.intercept_ = 0.0
        return self

    def _matrix_free_update(self, svi: SVI, state: SVIState, args: Tuple[Any, ...]) -> Tuple[SVIState, jax.Array]:
        rng_key, rng_key_step = random.split(state.rng_key)
        loss_fn = _make_loss_fn(
            svi.loss,
            rng_key_step,
            svi.constrain_fn,
            svi.model,
            svi.guide,
            args,
            {},
            svi.static_kwargs,
            mutable_state=state.mutable_state,
        )
        params = svi.optim.get_params(state.optim_state)
        (loss, mutable_state), grads = _value_and_grad(loss_fn, x=params)
        grads = self._apply_matrix_free(loss_fn, params, grads, rng_key_step)
        optim_state = svi.optim.update(grads, state.optim_state, value=loss)
        new_state = SVIState(optim_state, mutable_state, rng_key)
        return new_state, loss

    def _apply_matrix_free(self, loss_fn, params, grads, rng_key):
        if self.natural_gradient:
            return self._apply_natural_gradient(loss_fn, params, grads, rng_key)
        return grads

    def _apply_natural_gradient(self, loss_fn, params, grads, rng_key):
        flat_params, unravel_params = ravel_pytree(params)
        flat_grads, unravel_grads = ravel_pytree(grads)

        grad_norm = jnp.linalg.norm(flat_grads)
        if not jnp.isfinite(grad_norm) or grad_norm < 1e-12:
            return grads

        def scalar_loss(flat_values):
            loss_value, _ = loss_fn(unravel_params(flat_values))
            return loss_value

        grad_loss_fn = jax.grad(scalar_loss)

        def fisher_vec(vec):
            hv = jax.jvp(grad_loss_fn, (flat_params,), (vec,))[1]
            if self.natgrad_damping > 0:
                hv = hv + self.natgrad_damping * vec
            return hv

        solution, info = cg(fisher_vec, flat_grads, tol=self.cg_tol, maxiter=self.cg_max_iters)
        info = jnp.asarray(info)
        solution = jax.lax.cond(info == 0, lambda val: val, lambda _: flat_grads, solution)

        if self.use_hutchinson and self.hutchinson_samples > 0:
            trace_est = _hutchinson_trace(fisher_vec, flat_params.shape[0], rng_key, self.hutchinson_samples)
            denom = trace_est / (flat_params.shape[0] + 1e-8)
            solution = solution / (denom + 1e-8)

        if self.covariance_damping > 0:
            gamma = jnp.clip(self.covariance_damping, 0.0, 1.0)
            solution = gamma * solution + (1.0 - gamma) * flat_grads

        return unravel_grads(solution)

    def predict(self, X: np.ndarray, use_posterior_mean: bool = True) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.coef_mean_ is None:
            raise RuntimeError("Model not fitted.")
        if use_posterior_mean:
            return X @ self.coef_mean_ + self.intercept_
        if self.coef_samples_ is None or self.coef_samples_.shape[0] == 0:
            raise RuntimeError("No posterior samples available.")
        return X @ self.coef_samples_[-1] + self.intercept_

    def get_posterior_summaries(self) -> Dict[str, Any]:
        if self.coef_samples_ is None:
            raise RuntimeError("Not fitted.")
        return {
            "beta_mean": self.coef_samples_.mean(axis=0),
            "beta_median": np.median(self.coef_samples_, axis=0),
            "beta_ci95": np.quantile(self.coef_samples_, [0.025, 0.975], axis=0),
            "sigma_mean": self.sigma_mean_,
            "tau_mean": self.tau_mean_,
            "a_mean": self.a_mean_,
            "c2_mean": self.c2_mean_,
            "phi_mean": self.phi_mean_,
            "lambda_mean": self.lambda_mean_,
        }


GRRHS_SVI = GRRHS_SVI_Numpyro
