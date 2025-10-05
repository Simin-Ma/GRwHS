# grwhs/models/grwhs_svi_numpyro.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.nn import softplus
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.optim import Adam

from grwhs.utils.logging_utils import progress

def _split_groups(p: int, groups: List[List[int]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    ???:
      - group_id: shape [p], ????????????
      - group_sizes: shape [G]
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
    ?????? ?_g ??? ? ????????p??    """
    beta = jnp.zeros((p,))
    for g, idxs in enumerate(groups):
        beta = beta.at[jnp.array(idxs)].set(beta_groups[g])
    return beta


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

    # ????????    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)
    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)  # [S, p]
    sigma_mean_: Optional[float] = field(default=None, init=False)
    tau_mean_: Optional[float] = field(default=None, init=False)
    phi_mean_: Optional[np.ndarray] = field(default=None, init=False)      # [G]
    lambda_mean_: Optional[np.ndarray] = field(default=None, init=False)   # [p]
    intercept_: float = field(default=0.0, init=False)

    # ?????    groups_: Optional[List[List[int]]] = field(default=None, init=False)
    group_id_: Optional[jnp.ndarray] = field(default=None, init=False)
    group_sizes_: Optional[jnp.ndarray] = field(default=None, init=False)

    def _model(self, X: jnp.ndarray, y: jnp.ndarray, groups: List[List[int]]):
        """
        ?????????????????        - ??? sample ?_g ???_g???j ?????? ? ?????        """
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
        ???????????          - ????????joint q([?_g, log ?_g]) = MVN(loc, L L??)
          - log ?_j, log ?, log ? ~ Normal
        """
        n, p = X.shape
        G = len(groups)

        # ----- global scales (log-normal via Delta(exp(.)))
        mu_log_tau = numpyro.param("mu_log_tau", jnp.array(0.0))
        rho_log_tau = numpyro.param("rho_log_tau", jnp.array(-1.0))
        std_log_tau = softplus(rho_log_tau)
        log_tau = numpyro.sample("log_tau_aux", dist.Normal(mu_log_tau, std_log_tau))
        numpyro.sample("tau", dist.Delta(jnp.exp(log_tau)))

        mu_log_sigma = numpyro.param("mu_log_sigma", jnp.array(0.0))
        rho_log_sigma = numpyro.param("rho_log_sigma", jnp.array(-1.0))
        std_log_sigma = softplus(rho_log_sigma)
        log_sigma = numpyro.sample("log_sigma_aux", dist.Normal(mu_log_sigma, std_log_sigma))
        numpyro.sample("sigma", dist.Delta(jnp.exp(log_sigma)))

        # ----- local scales (log-normal)
        mu_log_lam = numpyro.param("mu_log_lambda", jnp.zeros((p,)))
        rho_log_lam = numpyro.param("rho_log_lambda", -jnp.ones((p,)))
        std_log_lam = softplus(rho_log_lam)
        log_lam = numpyro.sample("log_lambda_aux", dist.Normal(mu_log_lam, std_log_lam).to_event(1))
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
    # ??? API
    # ---------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[List[List[int]]] = None,
            num_steps: Optional[int] = None, lr: Optional[float] = None, num_samples_export: int = 1000) -> "GRwHS_SVI_Numpyro":
        """
        ??? SVI????????batch?? ?????unit-variance?? ???????????? preprocess ????????
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

        svi = SVI(self._model, self._guide, optimizer, loss=Trace_ELBO())
        # ??? batch????????????
        init_state = svi.init(rng_key, jnp.array(X), jnp.array(y), groups)

        # Track ELBO during optimization
        state = init_state
        for step_idx in progress(range(steps), total=steps, desc="SVI training"):
            state, loss = svi.update(state, jnp.array(X), jnp.array(y), groups)
            if jnp.isnan(loss):
                raise FloatingPointError("SVI loss produced NaN")

        params = svi.get_params(state)

        # ???????????? guide??        # ??? Predictive?????guide
        pred = Predictive(self._guide, params=params, num_samples=num_samples_export, return_sites=[
            "tau", "sigma", "lambda",
            # ?????? ?, ?
            *[f"phi_g_{g}" for g in range(len(groups))],
            *[f"beta_g_{g}" for g in range(len(groups))],
        ])

        post = pred(rng_key, jnp.array(X), jnp.array(y), groups)

        # ???
        beta_samples = []
        for s in range(num_samples_export):
            beta_groups = [post[f"beta_g_{g}"][s] for g in range(len(groups))]
            beta = _gather_beta_from_groups(beta_groups, groups, p)
            beta_samples.append(np.asarray(beta))

        coef_samples = np.stack(beta_samples, axis=0)  # [S, p]
        self.coef_samples_ = coef_samples
        self.coef_mean_ = coef_samples.mean(axis=0)

        # ??????????        self.sigma_mean_ = float(np.asarray(post["sigma"]).mean())
        self.tau_mean_ = float(np.asarray(post["tau"]).mean())
        self.lambda_mean_ = np.asarray(post["lambda"]).mean(axis=0)

        phi_samples = []
        for g in range(len(groups)):
            arr = np.asarray(post[f"phi_g_{g}"])
            if arr.ndim == 1:
                phi_samples.append(arr)
            else:
                reshaped = arr.reshape(arr.shape[0], -1)
                phi_samples.append(reshaped[:, 0])
        phi_mat = np.stack(phi_samples, axis=1)
        self.phi_mean_ = phi_mat.mean(axis=0)

        self.intercept_ = 0.0
        return self

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
