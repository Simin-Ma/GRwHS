# grwhs/models/grwhs_svi_numpyro.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.nn import softplus
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.svi import SVIState, _make_loss_fn
from numpyro.optim import Adam, _value_and_grad

from grwhs.utils.logging_utils import progress

def _split_groups(p: int, groups: List[List[int]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      - group_id: shape [p], integer group index for each feature
      - group_sizes: shape [G], size of each group
    """
    G = len(groups)
    gid = np.empty(p, dtype=np.int32)
    gsz = np.zeros(G, dtype=np.int32)
    for g, idxs in enumerate(groups):
        gid[idxs] = g
        gsz[g] = len(idxs)
    return jnp.array(gid), jnp.array(gsz)


def _gather_beta_from_groups(beta_groups: List[jnp.ndarray], groups: List[List[int]], p: int) -> jnp.ndarray:
    """
    Assemble full beta vector of length p from per-group beta blocks.
    """
    beta = jnp.zeros((p,))
    for g, idxs in enumerate(groups):
        beta = beta.at[jnp.array(idxs)].set(beta_groups[g])
    return beta


def _hutchinson_trace(matvec_fn, dim: int, key: jax.Array, num_samples: int) -> jax.Array:
    """Approximate ``trace(A)`` for a matrix via matrix-vector product callback."""
    if num_samples <= 0:
        raise ValueError("num_samples for Hutchinson estimator must be positive")

    keys = random.split(key, num_samples)

    def single(k):
        z = random.rademacher(k, (dim,), dtype=jnp.float32)
        return jnp.dot(z, matvec_fn(z))

    estimates = jax.vmap(single)(keys)
    return jnp.mean(estimates)


@dataclass
class GRwHS_SVI_Numpyro:
    c: float = 1.0
    tau0: float = 0.1
    eta: float = 0.5
    s0: float = 1.0

    num_steps: int = 3000
    lr: float = 1e-2
    seed: int = 42
    batch_size: Optional[int] = None  # None = full batch
    use_hutchinson: bool = True
    hutchinson_samples: int = 8
    natural_gradient: bool = False
    cg_tol: float = 1e-5
    cg_max_iters: int = 50
    natgrad_damping: float = 1e-3
    coupling_clip: Optional[float] = 3.0
    covariance_damping: float = 0.0

    # Posterior mean of coefficients (computed after fit)
    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)  # [S, p]
    sigma_mean_: Optional[float] = field(default=None, init=False)
    tau_mean_: Optional[float] = field(default=None, init=False)
    phi_mean_: Optional[np.ndarray] = field(default=None, init=False)      # [G]
    lambda_mean_: Optional[np.ndarray] = field(default=None, init=False)   # [p]
    intercept_: float = field(default=0.0, init=False)

    # Group structure used during fitting
    group_id_: Optional[jnp.ndarray] = field(default=None, init=False)
    group_sizes_: Optional[jnp.ndarray] = field(default=None, init=False)
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False)
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)
    phi_samples_: Optional[np.ndarray] = field(default=None, init=False)
    sigma_samples_: Optional[np.ndarray] = field(default=None, init=False)

    def _model(self, X: jnp.ndarray, y: jnp.ndarray, groups: List[List[int]]):
        """
        Probabilistic model:
          - Samples group-specific local scales phi_g and coefficients beta_g by group
          - Applies regularized horseshoe-style shrinkage with group structure
        """
        n, p = X.shape
        G = len(groups)
        group_id, group_sizes = _split_groups(p, groups)
        eta_g = self.eta / jnp.sqrt(group_sizes)

        # scales
        sigma = numpyro.sample("sigma", dist.HalfCauchy(self.s0))
        tau = numpyro.sample("tau", dist.HalfCauchy(self.tau0))
        lam = numpyro.sample("lambda", dist.HalfCauchy(jnp.ones((p,))).to_event(1))

        # RHS regularization: tilde_lambda
        tl = (self.c * lam) / jnp.sqrt(self.c ** 2 + (tau ** 2) * (lam ** 2) + 1e-18)

        beta_groups = []
        phi_groups = []
        for g in range(G):
            idxs = jnp.array(groups[g])
            pg = idxs.shape[0]

            # ?_g ~ HalfNormal(?_g)
            phi_g = numpyro.sample(f"phi_g_{g}", dist.HalfNormal(eta_g[g]))
            phi_groups.append(phi_g)

            # ?_g prior scale: ?_g * ? * tl[idxs] * ?
            scale_g = phi_g * tau * tl[idxs] * sigma
            # ?_g | ? ~ N(0, diag(scale_g^2))
            numpyro.deterministic(f"beta_scale_g_{g}", scale_g)
            beta_g = numpyro.sample(f"beta_g_{g}", dist.Normal(jnp.zeros((pg,)), scale_g).to_event(1))
            beta_groups.append(beta_g)

        beta = _gather_beta_from_groups(beta_groups, groups, p)

        # likelihood
        mean = X @ beta
        with numpyro.plate("data", n):
            numpyro.sample("y", dist.Normal(mean, sigma), obs=y)

    def _guide(self, X: jnp.ndarray, y: jnp.ndarray, groups: List[List[int]]):
        """
        Variational guide:
          - Joint q([beta_g, log phi_g]) per group is MVN(loc, L L^T)
          - log lambda_j, log tau, log sigma ~ Normal (reparameterized via Delta(exp(.)))
        """
        n, p = X.shape
        G = len(groups)

        # ----- global scales (log-normal via Delta(exp(.)))
        mu_log_tau = numpyro.param("mu_log_tau", jnp.array(0.0))
        rho_log_tau = numpyro.param("rho_log_tau", jnp.array(-1.0))
        std_log_tau = softplus(rho_log_tau)

        mu_log_sigma = numpyro.param("mu_log_sigma", jnp.array(0.0))
        rho_log_sigma = numpyro.param("rho_log_sigma", jnp.array(-1.0))
        std_log_sigma = softplus(rho_log_sigma)

        # ----- local scales (log-normal with shared factor coupling)
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

        u_shared = numpyro.sample("u_shared", dist.Normal(0.0, 1.0))

        loc_tau = mu_log_tau + coupling_rho * u_shared
        log_tau = numpyro.sample("log_tau_aux", dist.Normal(loc_tau, std_log_tau))
        numpyro.sample("tau", dist.Delta(jnp.exp(log_tau)))

        log_sigma = numpyro.sample("log_sigma_aux", dist.Normal(mu_log_sigma, std_log_sigma))
        numpyro.sample("sigma", dist.Delta(jnp.exp(log_sigma)))

        loc_lambda = mu_log_lam + coupling_a * u_shared
        log_lam = numpyro.sample(
            "log_lambda_aux", dist.Normal(loc_lambda, std_log_lam).to_event(1)
        )
        numpyro.sample("lambda", dist.Delta(jnp.exp(log_lam)).to_event(1))

        # ----- group blocks: q([beta_g, log phi_g]) = MVN
        for g in range(G):
            idxs = jnp.array(groups[g])
            pg = int(idxs.shape[0])
            d = pg + 1  # concat [beta_g, log phi_g]

            loc = numpyro.param(f"loc_group_{g}", jnp.zeros((d,)))
            L_flat = numpyro.param(f"L_group_{g}", jnp.eye(d))
            L = jnp.tril(L_flat)
            diag = jnp.diag(softplus(jnp.diag(L)) + 1e-5)
            L = L - jnp.diag(jnp.diag(L)) + diag

            z = numpyro.sample(f"z_group_{g}", dist.MultivariateNormal(loc, scale_tril=L))
            beta_g = z[:pg]
            log_phi_g = z[pg]

            numpyro.sample(f"beta_g_{g}", dist.Delta(beta_g).to_event(1))
            numpyro.sample(f"phi_g_{g}", dist.Delta(jnp.exp(log_phi_g)))

    # ---------------------------
    # Public API
    # ---------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[List[List[int]]] = None,
            num_steps: Optional[int] = None, lr: Optional[float] = None, num_samples_export: int = 1000) -> "GRwHS_SVI_Numpyro":
        """
        Fit with SVI on full batch (or provided batch size). Inputs are assumed preprocessed to approximately unit variance.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n, p = X.shape
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

        X_jax = jnp.array(X)
        y_jax = jnp.array(y)
        args = (X_jax, y_jax, groups)

        # Initialize SVI state
        state = svi.init(rng_key, *args)

        for step_idx in progress(range(steps), total=steps, desc="SVI training"):
            state, loss = self._matrix_free_update(svi, state, args)
            if jnp.isnan(loss):
                raise FloatingPointError("SVI loss produced NaN")

        params = svi.get_params(state)

        # Sample from the guide using Predictive with learned params
        pred = Predictive(self._guide, params=params, num_samples=num_samples_export, return_sites=[
            "tau", "sigma", "lambda",
            # Export group-wise phi and beta
            *[f"phi_g_{g}" for g in range(len(groups))],
            *[f"beta_g_{g}" for g in range(len(groups))],
        ])

        post = pred(rng_key, jnp.array(X), jnp.array(y), groups)

        # Reconstruct per-sample full beta from group blocks
        beta_samples = []
        for s in range(num_samples_export):
            beta_groups = [post[f"beta_g_{g}"][s] for g in range(len(groups))]
            beta = _gather_beta_from_groups(beta_groups, groups, p)
            beta_samples.append(np.asarray(beta))

        coef_samples = np.stack(beta_samples, axis=0)  # [S, p]
        self.coef_samples_ = coef_samples
        self.coef_mean_ = coef_samples.mean(axis=0)

        lambda_samples = np.asarray(post["lambda"])
        if lambda_samples.ndim == 1:
            lambda_samples = lambda_samples.reshape(-1, 1)
        self.lambda_samples_ = lambda_samples
        self.lambda_mean_ = lambda_samples.mean(axis=0)

        tau_samples = np.asarray(post["tau"]).reshape(-1)
        self.tau_samples_ = tau_samples
        self.tau_mean_ = float(tau_samples.mean())

        sigma_samples = np.asarray(post["sigma"]).reshape(-1)
        self.sigma_samples_ = sigma_samples
        self.sigma_mean_ = float(sigma_samples.mean())

        phi_samples_list = []
        for g in range(len(groups)):
            arr = np.asarray(post[f"phi_g_{g}"])
            if arr.ndim == 1:
                phi_samples_list.append(arr)
            else:
                reshaped = arr.reshape(arr.shape[0], -1)
                phi_samples_list.append(reshaped[:, 0])
        phi_mat = np.stack(phi_samples_list, axis=1)
        self.phi_samples_ = phi_mat
        self.phi_mean_ = phi_mat.mean(axis=0)

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
            trace_est = _hutchinson_trace(
                fisher_vec,
                flat_params.shape[0],
                rng_key,
                self.hutchinson_samples,
            )
            denom = trace_est / (flat_params.shape[0] + 1e-8)
            solution = solution / (denom + 1e-8)

        if self.covariance_damping > 0:
            gamma = jnp.clip(self.covariance_damping, 0.0, 1.0)
            solution = gamma * solution + (1.0 - gamma) * flat_grads

        natural_grads = unravel_grads(solution)
        return natural_grads

    def predict(self, X: np.ndarray, use_posterior_mean: bool = True) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.coef_mean_ is None:
            raise RuntimeError("Model not fitted.")
        if use_posterior_mean:
            return X @ self.coef_mean_ + self.intercept_
        else:
            if self.coef_samples_ is None or self.coef_samples_.shape[0] == 0:
                raise RuntimeError("No posterior samples available.")
            return X @ self.coef_samples_[-1] + self.intercept_

    def get_posterior_summaries(self) -> Dict[str, Any]:
        if self.coef_samples_ is None:
            raise RuntimeError("Not fitted.")
        out = {
            "beta_mean": self.coef_samples_.mean(axis=0),
            "beta_median": np.median(self.coef_samples_, axis=0),
            "beta_ci95": np.quantile(self.coef_samples_, [0.025, 0.975], axis=0),
            "sigma_mean": self.sigma_mean_,
            "tau_mean": self.tau_mean_,
            "phi_mean": self.phi_mean_,
            "lambda_mean": self.lambda_mean_,
        }
        return out
GRwHS_SVI = GRwHS_SVI_Numpyro
