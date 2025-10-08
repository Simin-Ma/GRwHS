# grwhs/models/grwhs_gibbs.py
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from numpy.random import Generator, default_rng
from scipy.linalg import cho_factor, cho_solve

from grwhs.utils.logging_utils import progress

# Prefer the stable GIG sampler implemented in inference/gig.py
# Expected signature: sample_gig(lambda_param: float, chi: float, psi: float, size: int | tuple[int, ...] = 1, rng: Optional[Generator] = None) -> np.ndarray
from grwhs.inference.gig import sample_gig

_PHI_EPS = 1e-12
_PHI_BASE_FLOOR = 2e-5
_PHI_ADAPT_COEFF = 5e-7
_FLOOR_MIN_WEIGHT = 0.002
_TAU_MAX = 5e3
_SIGMA2_FACTOR = 2.0
_BURNIN_WARM = 1000
_RIDGE_ALPHA = 1e-3
_TAU_PENALTY = 5e-5


def _burnin_weight(step: int, burnin: Optional[int], warm: int = 1000) -> float:
    if burnin is None or burnin <= 0:
        return 0.0
    if step < burnin:
        return 1.0
    if step < burnin + warm:
        return 1.0 - (step - burnin) / float(max(warm, 1))
    return 0.0


# ------------------------------
# Helpers: sampling utilities
# ------------------------------
def _sample_invgamma(alpha: float, beta: float, rng: Generator) -> float:
    """
    InvGamma(alpha, beta) with shape–scale parameterization used in the paper:
        p(x) ∝ beta^alpha x^{-alpha-1} exp(-beta/x),  x > 0
    Sampling: if Z ~ Gamma(alpha, scale=1/beta), then X = 1/Z ~ InvGamma(alpha, beta).
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("InvGamma requires alpha,beta > 0")
    z = rng.gamma(shape=alpha, scale=1.0 / beta)
    return 1.0 / z


def _slice_sample_1d(logpdf, x0: float, rng: Generator, w: float = 1.0, m: int = 100, max_steps: int = 1000) -> float:
    """
    Robust 1-D slice sampler (Neal 2003 style with stepping-out and shrinkage).
    - logpdf: callable returning an unnormalized log density
    - x0:     current point (float)
    - w:      initial step size
    - m:      maximum stepping-out iterations
    - returns: new sample
    """
    logy = logpdf(x0) - rng.exponential(1.0)

    # Step out
    u = rng.uniform(0.0, 1.0)
    L = x0 - u * w
    R = L + w
    j = int(rng.integers(0, m))
    k = (m - 1) - j

    while j > 0 and logpdf(L) > logy:
        L -= w
        j -= 1
    while k > 0 and logpdf(R) > logy:
        R += w
        k -= 1

    # Shrinkage
    it = 0
    while it < max_steps:
        x1 = rng.uniform(L, R)
        if logpdf(x1) >= logy:
            return x1
        # shrink
        if x1 < x0:
            L = x1
        else:
            R = x1
        it += 1
    # Fallback
    return x0


# ------------------------------
# Core: GRwHS Gibbs sampler
# ------------------------------
@dataclass
class GRwHS_Gibbs:
    # Hyperparameters
    c: float = 1.0           # slab width
    tau0: float = 0.1        # global HC scale
    eta: float = 0.5         # base group HalfNormal scale (before size adjustment)
    s0: float = 1.0          # noise HC scale

    # Sampling controls
    iters: int = 2000
    burnin: Optional[int] = None
    thin: int = 1
    seed: int = 42

    # Numerical settings
    slice_w: float = 1.0
    slice_m: int = 100
    jitter: float = 1e-10    # Numerical jitter to avoid degeneracy

    # Runtime state (accessible after fit)
    rng: Generator = field(init=False)
    coef_samples_: Optional[np.ndarray] = field(default=None, init=False)   # [S, p]
    sigma2_samples_: Optional[np.ndarray] = field(default=None, init=False) # [S]
    tau_samples_: Optional[np.ndarray] = field(default=None, init=False)    # [S]
    phi_samples_: Optional[np.ndarray] = field(default=None, init=False)    # [S, G]
    lambda_samples_: Optional[np.ndarray] = field(default=None, init=False) # [S, p]

    coef_mean_: Optional[np.ndarray] = field(default=None, init=False)      # Posterior mean
    intercept_: float = field(default=0.0, init=False)

    # Data/group information
    groups_: Optional[List[List[int]]] = field(default=None, init=False)
    group_id_: Optional[np.ndarray] = field(default=None, init=False)    # len p
    group_sizes_: Optional[np.ndarray] = field(default=None, init=False) # len G

    def __post_init__(self):
        self.rng = default_rng(self.seed)
        if self.burnin is None:
            self.burnin = self.iters // 2
        if self.c <= 0:
            raise ValueError("c must be > 0")

    # ----------
    # API
    # ----------
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[List[List[int]]] = None) -> "GRwHS_Gibbs":
        """
        Run the Gibbs sampler. Assumes each column of X has unit variance and y is centered.

        Parameters
        ----------
        X: (n, p)
        y: (n,)
        groups: list of index lists. If None, treat each variable as its own group.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        y_var = float(np.var(y)) if y.size else 1.0
        sigma2_cap = _SIGMA2_FACTOR * max(y_var, self.jitter)

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

        # Size-adjusted HalfNormal prior for group scales: η_g = η / sqrt(p_g)
        eta_g = self.eta / np.sqrt(group_sizes)

        # Initialize parameters
        try:
            beta = np.linalg.solve(
                X.T @ X + _RIDGE_ALPHA * np.eye(p),
                X.T @ y,
            )
        except np.linalg.LinAlgError:
            beta = np.zeros(p)
        sigma2 = 1.0
        tau = self.tau0
        phi = np.ones(G) * (self.eta / np.sqrt(np.mean(group_sizes)))
        lam = np.ones(p)

        # Optional inverse-gamma auxiliaries
        xi_j = np.ones(p)
        xi_tau = 1.0
        xi_sigma = 1.0

        # Sampling controls
        kept = max(0, (self.iters - self.burnin + self.thin - 1) // self.thin)
        coef_draws = np.zeros((kept, p))
        sigma2_draws = np.zeros(kept)
        tau_draws = np.zeros(kept)
        phi_draws = np.zeros((kept, G))
        lam_draws = np.zeros((kept, p))

        keep_i = 0

        # Precompute matrices
        # Note: the algorithm repeatedly uses C0 = D_beta^{-1} = diag(φ_g^2 τ^2 \tildeλ_j^2 σ^2)
        #       and the Cholesky of M = X C0 X^T + σ^2 I_n
        xtx_diag = np.einsum("ij,ij->j", X, X)
        tau_trace: List[float] = []
        sigma_trace: List[float] = []
        mean_kappa_trace: List[float] = []
        for it in progress(range(self.iters), total=self.iters, desc="Gibbs sampling"):
            burnin_w = _burnin_weight(it, self.burnin, _BURNIN_WARM)
            # ---- 1) Compute tilde_lambda, prior precision d_j, and inverse C0 (prior covariance diag)
            tilde_lam = (self.c * lam) / np.sqrt(self.c ** 2 + (tau ** 2) * (lam ** 2))
            # prior variance v_j = φ_g^2 τ^2 \tildeλ_j^2 σ^2
            phi_safe = np.maximum(phi, _PHI_EPS)
            phi_group_safe = phi_safe[group_id]
            v_prior = (phi_group_safe ** 2) * (tau ** 2) * (tilde_lam ** 2) * sigma2
            # Guard against numerical issues
            v_prior = np.maximum(v_prior, self.jitter * sigma2)

            # prior precision d_jj:
            d_prior = 1.0 / v_prior

            # ---- 2) Conditional Gaussian block: sample β | rest
            beta = self._sample_beta_conditional(X, y, sigma2, v_prior, rng=self.rng)

            # ---- 3) Update λ_j: slice sample u_j = log λ_j (instead of MH)
            # log-target: -log(tildeλ_j) - β_j^2/(2 φ_g^2 τ^2 tildeλ_j^2 σ^2) - log(1+λ_j^2) + u_j
            phi_g_map = phi_group_safe

            def make_logp_u(j: int):
                b2 = beta[j] ** 2
                phi2 = (phi_g_map[j] ** 2)
                def _logp(u: float) -> float:
                    lam_j = math.exp(u)
                    # tilde
                    num = self.c * lam_j
                    den = math.sqrt(self.c ** 2 + (tau ** 2) * (lam_j ** 2))
                    tl = num / den
                    # Target
                    val = 0.0
                    val += -math.log(max(tl, 1e-300))
                    val += -(b2 / (2.0 * phi2 * (tau ** 2) * (tl ** 2) * sigma2))
                    # Half-Cauchy(0,1): -log(1+λ^2)
                    val += -math.log(1.0 + lam_j ** 2)
                    # Jacobian
                    val += u
                    return float(val)
                return _logp

            for j in range(p):
                logp = make_logp_u(j)
                u0 = math.log(lam[j])
                u_new = _slice_sample_1d(logp, u0, rng=self.rng, w=self.slice_w, m=self.slice_m)
                lam[j] = math.exp(u_new)

            # Optional: refresh ξ_j | λ_j^2
            for j in range(p):
                xi_j[j] = _sample_invgamma(alpha=1.0, beta=1.0 + 1.0 / (lam[j] ** 2), rng=self.rng)

            # ---- 4) Group scales φ_g: sample θ_g = φ_g^2 ~ GIG(λ=1/2 - p_g/2, χ=S_g, ψ=1/η_g^2)
            # S_g = (1/(τ^2 σ^2)) Σ_{j∈Gg} β_j^2 / \tildeλ_j^2
            inv_tilde2 = 1.0 / np.maximum(tilde_lam ** 2, self.jitter)
            # Aggregate by group
            S_g = np.zeros(G)
            for g in range(G):
                idxs = groups[g]
                S_g[g] = (beta[idxs] ** 2 * inv_tilde2[idxs]).sum()
            S_g *= 1.0 / (tau ** 2 * sigma2)
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

            # ---- 5) Global scale τ: slice sample v = log τ
            # log-target:
            #   - p log τ - sum_j log tildeλ_j
            #   - (1/(2σ^2)) * sum_j β_j^2 / (φ_{g(j)}^2 τ^2 \tildeλ_j^2)
            #   + log Half-Cauchy(τ | 0, τ0) + v
            beta2 = beta ** 2

            def logp_v(v: float) -> float:
                t = math.exp(v)
                # recompute tilde for given τ
                tl = (self.c * lam) / np.sqrt(self.c ** 2 + (t ** 2) * (lam ** 2))
                tl2 = np.maximum(tl ** 2, self.jitter)
                # terms
                val = 0.0
                val += -p * v  # -p log τ
                val += -np.log(np.maximum(tl, 1e-300)).sum()
                denom = (phi_group_safe ** 2) * (t ** 2) * tl2 * sigma2
                val += -(beta2 / (2.0 * denom)).sum()
                # Half-Cauchy prior on τ with scale τ0: -log(1 + (τ/τ0)^2)
                val += -math.log(1.0 + (t / self.tau0) ** 2)
                val += -_TAU_PENALTY * burnin_w * (t ** 2)
                # Jacobian
                val += v
                return float(val)

            v0 = math.log(tau)
            v_new = _slice_sample_1d(logp_v, v0, rng=self.rng, w=self.slice_w, m=self.slice_m)
            tau = math.exp(v_new)
            tau = min(tau, _TAU_MAX)

            # Optional: refresh ξ_tau | τ^2
            xi_tau = _sample_invgamma(alpha=1.0, beta=(1.0 / (self.tau0 ** 2)) + 1.0 / (tau ** 2), rng=self.rng)

            # ---- 6) Noise σ^2: inverse-gamma conditional
            # α = (n + p + 1)/2
            # β = 0.5 * ||y - Xβ||^2 + 0.5 * Σ β_j^2 / (φ_{g(j)}^2 τ^2 \tildeλ_j^2) + 1/ξ_σ
            resid = y - X @ beta
            RSS = float(resid @ resid)
            tl2 = np.maximum(((self.c * lam) / np.sqrt(self.c ** 2 + (tau ** 2) * (lam ** 2))) ** 2, self.jitter)
            prior_quad = float((beta2 / (phi_group_safe ** 2 * (tau ** 2) * tl2)).sum())
            alpha = 0.5 * (n + p + 1)
            beta_scale = 0.5 * (RSS + prior_quad) + 1.0 / xi_sigma
            sigma2 = _sample_invgamma(alpha=alpha, beta=beta_scale, rng=self.rng)
            sigma2 = min(sigma2, sigma2_cap)

            # Refresh ξ_σ | σ^2
            xi_sigma = _sample_invgamma(alpha=1.0, beta=(1.0 / (self.s0 ** 2)) + 1.0 / sigma2, rng=self.rng)

            # ---- Diagnostics trace logging (τ, σ, mean κ)
            tilde_lam_log = (self.c * lam) / np.sqrt(self.c ** 2 + (tau ** 2) * (lam ** 2))
            tilde_lam_log = np.maximum(tilde_lam_log, self.jitter)
            v_prior_log = (phi_group_safe ** 2) * (tau ** 2) * (tilde_lam_log ** 2) * sigma2
            v_prior_log = np.maximum(v_prior_log, self.jitter * sigma2)
            d_prior_log = 1.0 / v_prior_log
            q_diag = xtx_diag / sigma2
            denom = q_diag + d_prior_log
            kappa = np.divide(q_diag, denom, out=np.zeros_like(q_diag), where=denom > 0)
            mean_kappa_trace.append(float(np.mean(kappa)))
            tau_trace.append(float(tau))
            sigma_trace.append(float(math.sqrt(max(sigma2, self.jitter))))

            # ---- Store draws
            if it >= self.burnin and ((it - self.burnin) % self.thin == 0):
                coef_draws[keep_i] = beta
                sigma2_draws[keep_i] = sigma2
                tau_draws[keep_i] = tau
                phi_draws[keep_i] = phi
                lam_draws[keep_i] = lam
                keep_i += 1

        # Finalize: persist state
        self.coef_samples_ = coef_draws
        self.sigma2_samples_ = sigma2_draws
        self.tau_samples_ = tau_draws
        self.phi_samples_ = phi_draws
        self.lambda_samples_ = lam_draws
        self.coef_mean_ = coef_draws.mean(axis=0) if kept > 0 else np.zeros(p)
        self.intercept_ = 0.0  # X and y are assumed centered/standardized

        if tau_trace:
            logger = logging.getLogger(__name__)
            logger.info("tau_trace=%s", np.asarray(tau_trace))
            logger.info("sigma_trace=%s", np.asarray(sigma_trace))
            logger.info("mean_kappa_trace=%s", np.asarray(mean_kappa_trace))

        return self

    def predict(self, X: np.ndarray, use_posterior_mean: bool = True) -> np.ndarray:
        """Predict; uses the posterior mean by default."""
        X = np.asarray(X, dtype=float)
        if use_posterior_mean:
            if self.coef_mean_ is None:
                raise RuntimeError("Model not fitted.")
            return X @ self.coef_mean_ + self.intercept_
        else:
            # Use the most recent draw
            if self.coef_samples_ is None or self.coef_samples_.shape[0] == 0:
                raise RuntimeError("No posterior samples available.")
            return X @ self.coef_samples_[-1] + self.intercept_

    # ----------
    # Internal: conditional Gaussian sampler for β (Bhattacharya et al.)
    # ----------
    def _sample_beta_conditional(self, X: np.ndarray, y: np.ndarray, sigma2: float, v_prior: np.ndarray,
                                 rng: Generator) -> np.ndarray:
        """
        Sample β | rest ~ N(μ, Σ).
        Prior: β ~ N(0, C0) with C0 = diag(v_prior) (diagonal).
        Likelihood: y | β ~ N(Xβ, σ² I).
        Posterior: Σ = (C0^{-1} + XᵀX / σ²)^{-1}, μ = Σ Xᵀ y / σ².

        Sampling recipe (Rue/Bhattacharya):
          1) Sample u ~ N(0, C0)  => u = sqrt(v_prior) * z, z ~ N(0, I).
          2) Sample δ ~ N(0, I_n).
          3) Form M = X C0 Xᵀ + σ² I_n, then solve w = M^{-1} (y - X u + σ δ).
          4) β = u + C0 Xᵀ w.
        """
        n, p = X.shape
        sqrt_v = np.sqrt(np.maximum(v_prior, 1e-18))
        u = sqrt_v * rng.standard_normal(p)
        delta = rng.standard_normal(n)

        # X C0 Xᵀ = X * diag(v_prior) * Xᵀ
        XC = X * v_prior  # broadcasting each column scaled by v_prior_j
        M = XC @ X.T  # (n, n)
        # + σ² I
        M.flat[:: n + 1] += sigma2  # add to diag

        # rhs = y - X u + σ δ
        rhs = y - X @ u + math.sqrt(sigma2) * delta

        # Cholesky solve
        c, lower = cho_factor(M, overwrite_a=False, check_finite=False)
        w = cho_solve((c, lower), rhs, check_finite=False)

        # β = u + C0 Xᵀ w
        beta = u + v_prior * (X.T @ w)
        return beta

    # ----------
    # Convenience exports (diagnostics/visualization)
    # ----------
    def get_posterior_summaries(self) -> Dict[str, Any]:
        if self.coef_samples_ is None:
            raise RuntimeError("Not fitted.")
        out = {
            "beta_mean": self.coef_samples_.mean(axis=0),
            "beta_median": np.median(self.coef_samples_, axis=0),
            "beta_ci95": np.quantile(self.coef_samples_, [0.025, 0.975], axis=0),
            "sigma2_mean": float(self.sigma2_samples_.mean()) if self.sigma2_samples_ is not None else None,
            "tau_mean": float(self.tau_samples_.mean()) if self.tau_samples_ is not None else None,
            "phi_mean": self.phi_samples_.mean(axis=0) if self.phi_samples_ is not None else None,
        }
        return out
