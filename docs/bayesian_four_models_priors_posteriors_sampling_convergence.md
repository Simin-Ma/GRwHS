# Four Bayesian Regression Models: Priors, Posteriors, Sampling, Convergence, and Budgets (Code-Aligned)

This document is aligned with the current repository implementation.

## 0. Scope

- GR-RHS Gibbs: `grrhs/models/grrhs_gibbs.py`
- GR-RHS SVI: `grrhs/models/grrhs_svi_numpyro.py`
- GIGG: `grrhs/models/gigg_regression.py` and `grrhs/models/gigg_cran.py`
- RHS: `grrhs/models/baselines/stan/rhs_gaussian_regression.stan`
- Runner + convergence export: `grrhs/experiments/runner.py`

## 1. GR-RHS Gibbs (`grrhs_gibbs`)

### 1.1 Hierarchy

For `y = X beta + eps, eps ~ N(0, sigma^2 I)`:

- `beta_j | lambda_j, a_g, c_g^2, tau ~ N(0, tau^2 * tilde_lambda_{j,g}^2)`
- `tilde_lambda_{j,g}^2 = c_g^2 * lambda_j^2 * a_g^2 / (c_g^2 + tau^2 * lambda_j^2 * a_g^2)`
- `lambda_j ~ HalfCauchy(0, 1)`
- `a_g ~ HalfNormal(eta / sqrt(p_g))`
- `c_g^2 ~ InvGamma(alpha_c, beta_c)`
- `tau ~ HalfCauchy(0, tau0)` with auxiliary `nu`:
  - `tau^2 | nu ~ InvGamma(1/2, 1/nu)`
  - `nu | tau^2 ~ InvGamma(1, 1/tau0^2 + 1/tau^2)`
- `p(sigma^2) ∝ 1/sigma^2`

Useful precision decomposition used in code:

`1 / (tau^2 * tilde_lambda_{j,g}^2) = 1 / (tau^2 * lambda_j^2 * a_g^2) + 1 / c_g^2`.

### 1.2 Sampling

Metropolis-within-Gibbs:

- Closed form:
  - `beta | rest` Gaussian
  - `sigma^2 | rest` inverse-gamma
  - `nu | tau^2` inverse-gamma
- Log-scale random-walk MH:
  - `tau^2`
  - each `lambda_j`
  - each `a_g`
  - each `c_g^2`

### 1.3 Exported posterior arrays

- `coef_samples_ -> beta`
- `sigma2_samples_ -> sigma2`
- `tau_samples_ -> tau`
- `lambda_samples_ -> lambda`
- `a_samples_ -> a`
- `c2_samples_ -> c2`
- `phi_samples_ -> phi` (compatibility alias of `a_samples_`)

## 2. GR-RHS SVI (`grrhs_svi`)

SVI model now matches the same GR-RHS hierarchy (`a_g`, `c_g^2`, regularized variance map).

Guide is mean-field/block variational with log-parameterized positive scales. Exported arrays include:

- `coef_samples_`, `sigma_samples_`, `tau_samples_`, `lambda_samples_`, `a_samples_`, `c2_samples_`
- compatibility alias: `phi_samples_ = a_samples_`

## 3. GIGG and RHS

- GIGG and RHS implementations are unchanged in model definition.
- Fairness and diagnostics are compared under shared split and budget protocol.

## 4. Convergence blocks (runner defaults)

Current GR-RHS monitored blocks are:

- `beta`, `tau`, `a`, `c2`, `lambda`

Compatibility note: `phi` is still accepted as a legacy alias for grouped amplitude in exports and downstream plotting.

## 5. Budget policy

Bayesian fairness in `configs/base.yaml` enforces shared posterior budget by default (burn-in, kept draws, thinning, chains), with retry scaling rules handled by `runner.py`.
