"""Logistic GRRHS Gibbs sampler with Pólya–Gamma augmentation."""
from __future__ import annotations

import logging
import math
from typing import Optional, List

import numpy as np

from grrhs.inference.pg import sample_pg
from grrhs.models.grrhs_gibbs import (
    GRRHS_Gibbs,
    _BURNIN_WARM,
    _PHI_ADAPT_COEFF,
    _PHI_BASE_FLOOR,
    _PHI_EPS,
    _FLOOR_MIN_WEIGHT,
    _TAU_MAX,
    _TAU_PENALTY,
    _burnin_weight,
    _sample_invgamma,
    _slice_sample_1d,
    sample_gig,
)
from grrhs.utils.logging_utils import progress


class GRRHS_Gibbs_Logistic(GRRHS_Gibbs):
    """Group-regularized Horseshoe Gibbs sampler for Bernoulli-logit likelihoods."""

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[List[List[int]]] = None) -> "GRRHS_Gibbs_Logistic":
        """
        Run the Gibbs sampler with Pólya–Gamma augmentation.

        Assumes X columns are standardized (unit variance) but y is binary {0, 1}.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have matching number of rows.")
        if np.any((y < -1e-8) | (y > 1 + 1e-8)):
            raise ValueError("Logistic GRRHS requires binary labels in {0, 1}.")
        y = np.clip(y, 0.0, 1.0)

        n, p = X.shape
        if groups is None:
            groups = [[j] for j in range(p)]
        self.groups_ = groups
        G = len(groups)
        group_id = np.empty(p, dtype=int)
        group_sizes = np.zeros(G, dtype=int)
        for g, idxs in enumerate(groups):
            group_id[idxs] = g
            group_sizes[g] = len(idxs)
        self.group_id_ = group_id
        self.group_sizes_ = group_sizes

        eta_g = self.eta / np.sqrt(group_sizes)

        # Initial state
        beta = np.zeros(p)
        tau = self.tau0
        phi = np.ones(G) * (self.eta / np.sqrt(np.mean(group_sizes)))
        lam = np.ones(p)

        xi_j = np.ones(p)
        xi_tau = 1.0

        kept = max(0, (self.iters - self.burnin + self.thin - 1) // self.thin)
        coef_draws = np.zeros((kept, p))
        tau_draws = np.zeros(kept)
        phi_draws = np.zeros((kept, G))
        lam_draws = np.zeros((kept, p))
        keep_i = 0

        tau_trace: List[float] = []
        mean_kappa_trace: List[float] = []

        for it in progress(range(self.iters), total=self.iters, desc="Gibbs sampling (logistic)"):
            burnin_w = _burnin_weight(it, self.burnin, _BURNIN_WARM)

            tilde_lam = (self.c * lam) / np.sqrt(self.c ** 2 + (tau ** 2) * (lam ** 2))
            phi_safe = np.maximum(phi, _PHI_EPS)
            phi_group_safe = phi_safe[group_id]
            v_prior = (phi_group_safe ** 2) * (tau ** 2) * (tilde_lam ** 2)
            v_prior = np.maximum(v_prior, self.jitter)

            logits = X @ beta
            omega = sample_pg(np.ones(n, dtype=float), logits, rng=self.rng)
            omega = np.asarray(omega, dtype=float).reshape(-1)
            omega = np.maximum(omega, self.jitter)
            sqrt_omega = np.sqrt(omega)
            z = (y - 0.5) / omega
            y_tilde = sqrt_omega * z
            X_tilde = X * sqrt_omega[:, None]
            beta = self._sample_beta_conditional(X_tilde, y_tilde, sigma2=1.0, v_prior=v_prior, rng=self.rng)
            beta2 = beta ** 2

            phi_g_map = phi_group_safe

            def make_logp_u(j: int):
                b2 = beta2[j]
                phi2 = phi_g_map[j] ** 2

                def _logp(u: float) -> float:
                    lam_j = math.exp(u)
                    num = self.c * lam_j
                    den = math.sqrt(self.c ** 2 + (tau ** 2) * (lam_j ** 2))
                    tl = num / den
                    val = 0.0
                    val += -math.log(max(tl, 1e-300))
                    denom = 2.0 * phi2 * (tau ** 2) * (tl ** 2)
                    val += -(b2 / max(denom, self.jitter))
                    val += -math.log(1.0 + lam_j ** 2)
                    val += u
                    return float(val)

                return _logp

            for j in range(p):
                logp = make_logp_u(j)
                u0 = math.log(lam[j])
                u_new = _slice_sample_1d(logp, u0, rng=self.rng, w=self.slice_w, m=self.slice_m)
                lam[j] = math.exp(u_new)
                if not math.isfinite(lam[j]):
                    lam[j] = self.jitter
                lam[j] = max(lam[j], self.jitter)

            for j in range(p):
                lam_j = max(lam[j], self.jitter)
                xi_j[j] = _sample_invgamma(alpha=1.0, beta=1.0 + 1.0 / (lam_j ** 2), rng=self.rng)

            inv_tilde2 = 1.0 / np.maximum(tilde_lam ** 2, self.jitter)
            S_g = np.zeros(G)
            for g in range(G):
                idxs = groups[g]
                S_g[g] = (beta[idxs] ** 2 * inv_tilde2[idxs]).sum()
            S_g *= 1.0 / (tau ** 2)
            lam_gig = 0.5 - 0.5 * group_sizes
            chi_gig = np.maximum(S_g, self.jitter)
            psi_gig = 1.0 / np.maximum(eta_g ** 2, self.jitter)
            theta = np.zeros(G)
            for g in range(G):
                theta[g] = float(sample_gig(lambda_param=lam_gig[g], chi=chi_gig[g], psi=psi_gig[g], size=1, rng=self.rng)[0])
            phi = np.sqrt(np.maximum(theta, 0.0))
            adaptive_floor = max(_PHI_BASE_FLOOR, _PHI_ADAPT_COEFF * tau)
            phi_floor_weight = max(_FLOOR_MIN_WEIGHT, burnin_w)
            phi_floor_effective = max(_PHI_EPS, adaptive_floor * phi_floor_weight)
            phi = np.maximum(phi, phi_floor_effective)
            phi_safe = np.maximum(phi, _PHI_EPS)
            phi_group_safe = phi_safe[group_id]

            def logp_v(v: float) -> float:
                t = math.exp(v)
                tl = (self.c * lam) / np.sqrt(self.c ** 2 + (t ** 2) * (lam ** 2))
                tl2 = np.maximum(tl ** 2, self.jitter)
                val = 0.0
                val += -p * v
                val += -np.log(np.maximum(tl, 1e-300)).sum()
                denom = (phi_group_safe ** 2) * (t ** 2) * tl2
                val += -(beta2 / (2.0 * denom)).sum()
                val += -math.log(1.0 + (t / self.tau0) ** 2)
                val += -_TAU_PENALTY * burnin_w * (t ** 2)
                val += v
                return float(val)

            v0 = math.log(tau)
            v_new = _slice_sample_1d(logp_v, v0, rng=self.rng, w=self.slice_w, m=self.slice_m)
            tau = math.exp(v_new)
            tau = min(tau, _TAU_MAX)

            xi_tau = _sample_invgamma(alpha=1.0, beta=(1.0 / (self.tau0 ** 2)) + 1.0 / (tau ** 2), rng=self.rng)

            xtwx_diag = np.einsum("i,ij,ij->j", omega, X, X)
            d_prior = 1.0 / v_prior
            denom = xtwx_diag + d_prior
            kappa = np.divide(xtwx_diag, denom, out=np.zeros_like(xtwx_diag), where=denom > 0)
            mean_kappa_trace.append(float(np.mean(kappa)))
            tau_trace.append(float(tau))

            if it >= self.burnin and ((it - self.burnin) % self.thin == 0):
                coef_draws[keep_i] = beta
                tau_draws[keep_i] = tau
                phi_draws[keep_i] = phi
                lam_draws[keep_i] = lam
                keep_i += 1

        self.coef_samples_ = coef_draws
        self.sigma2_samples_ = None
        self.tau_samples_ = tau_draws
        self.phi_samples_ = phi_draws
        self.lambda_samples_ = lam_draws
        self.coef_mean_ = coef_draws.mean(axis=0) if kept > 0 else np.zeros(p)
        self.intercept_ = 0.0

        if tau_trace:
            logger = logging.getLogger(__name__)
            logger.info("tau_trace=%s", np.asarray(tau_trace))
            logger.info("mean_kappa_trace=%s", np.asarray(mean_kappa_trace))

        return self

    def decision_function(self, X: np.ndarray, use_posterior_mean: bool = True) -> np.ndarray:
        """Return linear logits for downstream evaluation."""
        X = np.asarray(X, dtype=float)
        if use_posterior_mean:
            if self.coef_mean_ is None:
                raise RuntimeError("Model not fitted.")
            coef_vec = self.coef_mean_
        else:
            if self.coef_samples_ is None or self.coef_samples_.shape[0] == 0:
                raise RuntimeError("No posterior samples available.")
            coef_vec = self.coef_samples_[-1]
        return X @ coef_vec + self.intercept_

    def predict_proba(self, X: np.ndarray, use_posterior_mean: bool = True) -> np.ndarray:
        """Predict class probabilities using either the posterior mean or last draw."""
        logits = self.decision_function(X, use_posterior_mean=use_posterior_mean)
        logits = np.clip(logits, -60.0, 60.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        return np.vstack([1.0 - prob, prob]).T

    def predict(self, X: np.ndarray, use_posterior_mean: bool = True, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels by thresholding posterior-mean probabilities."""
        proba = self.predict_proba(X, use_posterior_mean=use_posterior_mean)[:, 1]
        return (proba >= threshold).astype(int)


__all__ = ["GRRHS_Gibbs_Logistic"]
