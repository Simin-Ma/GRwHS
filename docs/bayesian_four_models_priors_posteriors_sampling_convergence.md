# Four Bayesian Regression Models: Priors, Posteriors, Sampling, Convergence, and Budgets (Code-Aligned)

This document is aligned with the current repository implementation.

## 0. Scope

- GR-RHS (NUTS): `grrhs/models/grrhs_nuts.py`
- GIGG: `grrhs/models/gigg_regression.py` and `grrhs/models/gigg_cran.py`
- RHS: `grrhs/models/baselines/stan/rhs_gaussian_regression.stan`
- Runner + convergence export: `grrhs/experiments/runner.py`

## 1. GR-RHS (`grrhs_nuts`)

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
- `p(sigma^2) 鈭?1/sigma^2`

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

## 2. Legacy Variants

Legacy GR-RHS variants have been retired in this repository; GR-RHS is consolidated to the NUTS implementation below.

## 2.1 GR-RHS NUTS (`grrhs_nuts`)

Reference path for paper-ready computation:

- Primitive hierarchy with transformed variables:
  - `log tau`, `log sigma`, `log lambda_j`, `log a_g`, `logit kappa_g`
- Non-centered coefficients:
  - `beta_j = beta_raw_j * sd_j`, `beta_raw_j ~ N(0,1)`
- Global shrinkage recommendation:
  - `tau0` via sparsity-aware calibration `tau0 = (p0 / (D - p0)) * (sigma_ref / sqrt(n))`
- Group-level defaults (allowance interpretation):
  - `kappa_g ~ Beta(alpha_kappa, beta_kappa)` with default `(2, 8)` (prior mass near low allowance)
  - `a_g ~ HalfNormal(eta/sqrt(p_g))`, default `eta=0.5`

Sampler diagnostics exported in `sampler_diagnostics_`:

- `hmc.divergences`
- `hmc.treedepth_hits`
- `hmc.ebfmi_min`
- `posterior_quality.max_rhat`
- `posterior_quality.min_ess`
- `posterior_quality.ess_per_sec`

## 3. GIGG and RHS

- GIGG and RHS implementations are unchanged in model definition.
- Fairness and diagnostics are compared under shared split and budget protocol.

## 4. Convergence blocks (runner defaults)

Current GR-RHS monitored blocks are:

- `beta`, `tau`, `a`, `c2`, `lambda`

Compatibility note: `phi` is still accepted as a legacy alias for grouped amplitude in exports and downstream plotting.

## 5. Budget policy

Bayesian fairness in `configs/base.yaml` enforces shared posterior budget by default (burn-in, kept draws, thinning, chains), with retry scaling rules handled by `runner.py`.

