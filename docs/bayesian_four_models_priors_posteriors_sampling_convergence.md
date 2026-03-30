# Four Bayesian Regression Models: Priors, Posteriors, Sampling, Convergence, and Budgets (Code-Aligned)

> Purpose: This document is strictly aligned with the current repository implementation. It summarizes (i) the priors, (ii) posterior / conditional-posterior forms actually sampled, (iii) exact sampling logic, (iv) convergence checks + thresholds, and (v) concrete sampling-budget numbers for the Bayesian regression models implemented in this repo.
>
> Scope note (runner gates):
> - The runner’s “Bayesian fairness / convergence / posterior_validation” gates only apply when `model.name` is one of:
>   `grrhs_gibbs`, `gigg` / `gigg_regression`, `regularized_horseshoe` / `rhs` / `regularised_horseshoe`
>   (see `grrhs/experiments/runner.py:_BAYESIAN_MODEL_NAMES`).
> - `grrhs_svi` (variational) is Bayesian (explicit priors) but is **not** treated as “Bayesian” by those gates unless `_BAYESIAN_MODEL_NAMES` is expanded.
>
> Code-as-truth sources:
> - Models: `grrhs/models/grrhs_gibbs.py`, `grrhs/models/gigg_regression.py`, `grrhs/models/grrhs_svi_numpyro.py`,
>   `grrhs/models/baselines/models.py`, `grrhs/models/baselines/stan/rhs_gaussian_regression.stan`
> - Runner + checks: `grrhs/experiments/runner.py`, `grrhs/experiments/registry.py`, `grrhs/diagnostics/convergence.py`
> - Defaults: `configs/base.yaml`, `configs/methods/grrhs_regression.yaml`, `configs/methods/gigg.yaml`, `configs/methods/regularized_horseshoe.yaml`

---

## 0. Notation

- Data: $X \in \mathbb{R}^{n \times p},\ y \in \mathbb{R}^n$
- Groups: $g(j)\in\{1,\dots,G\}$, group size $p_g$
- Local scale: $\lambda_j > 0$
- Group scale: $\phi_g > 0$ (in GIGG, group scale is $\gamma_g$)
- Global scale: $\tau > 0$
- Noise scale: $\sigma > 0$
- Regularized horseshoe local scale:
  $$\tilde\lambda_j=\frac{c\lambda_j}{\sqrt{c^2+\tau^2\lambda_j^2}}$$

---

## 1) `GRRHS_Gibbs` (Gaussian regression; Gibbs + slice)

Implementation: `grrhs/models/grrhs_gibbs.py:GRRHS_Gibbs`.

### 1.1 Parameters and defaults (code + method YAML)

Packages used:
- NumPy for linear algebra + RNG.
- No external MCMC engine (slice sampling + GIG draws are implemented inside `GRRHS_Gibbs`).

Code defaults:
- Hyperparameters: `c=1.0`, `tau0=0.1`, `eta=0.5`, `s0=1.0`, `use_groups=true`
- Sampling: `iters=2000`, `burnin=iters//2`, `thin=1`, `num_chains=1`, `seed=42`
- Slice samplers: local `slice_w=1.0, slice_m=100`; global tau `tau_slice_w=0.35, tau_slice_m=200`
- Numerical: `jitter=1e-10`

Method config: `configs/methods/grrhs_regression.yaml`
- `model.c=1.0, model.eta=0.5, model.s0=1.0, model.iters=2000`
- `inference.gibbs.burn_in=1000, thin=1, num_chains=4, jitter=1e-8, seed=321`
- `inference.gibbs.tau_slice_w=0.8, tau_slice_m=1200`
- `model.tau.mode=calibrated, target=coefficients, p0.value=20, sigma_reference=1.0`

Budget note (default repo behavior):
- `configs/base.yaml` sets `experiments.bayesian_fairness.enforce_shared_sampling_budget: true`, so the runner overwrites
  burn-in / total iterations / chain counts for Bayesian models (see §7).

### 1.2 Priors (as implemented)

1. $\tau \sim \text{Half-Cauchy}(0,\tau_0)$
2. $\lambda_j \sim \text{Half-Cauchy}(0,1)$
3. If `use_groups=true`: $\phi_g \sim \text{Half-Normal}(\eta_g)$ with $\eta_g=\eta/\sqrt{p_g}$
4. $\sigma \sim \text{Half-Cauchy}(0,s_0)$
5. Conditional coefficient prior:
   $$\beta_j\mid\tau,\lambda_j,\phi_{g(j)},\sigma \sim \mathcal N\!\left(0,\ \phi_{g(j)}^2\tau^2\tilde\lambda_j^2\sigma^2\right)$$

### 1.3 Conditional posteriors sampled (exact code shapes)

1. **$\beta\mid\cdot$ (Gaussian)**
   - $C_0=\mathrm{diag}(v_j)$ with $v_j=\phi_{g(j)}^2\tau^2\tilde\lambda_j^2\sigma^2$
   - $P=C_0^{-1}+\dfrac{X^TX}{\sigma^2}$, $\ \mu=P^{-1}\dfrac{X^Ty}{\sigma^2}$
   - Draw: $\beta\sim\mathcal N(\mu,P^{-1})$
   - Implementation: Cholesky on the $p\times p$ precision in `_sample_beta_conditional`.

2. **$\lambda_j\mid\cdot$ (slice on $u_j=\log\lambda_j$)**
   $$\log\pi(u_j\mid\cdot)=-\log\tilde\lambda_j-\frac{\beta_j^2}{2\phi_{g(j)}^2\tau^2\tilde\lambda_j^2\sigma^2}-\log(1+\lambda_j^2)+u_j,\quad \lambda_j=e^{u_j}$$

3. **$\phi_g^2\mid\cdot$ (GIG; then apply adaptive floor)**
   Let $\theta_g=\phi_g^2$:
   $$\theta_g\sim\text{GIG}(\lambda_g,\chi_g,\psi_g)$$
   $$\lambda_g=\frac12-\frac{p_g}{2},\quad
   \chi_g=\frac{1}{\tau^2\sigma^2}\sum_{j\in g}\frac{\beta_j^2}{\tilde\lambda_j^2},\quad
   \psi_g=\frac{1}{\eta_g^2}$$
   Then $\phi_g=\sqrt{\theta_g}$ and code enforces a floor that increases with $\tau$ during burn-in.

4. **$\tau\mid\cdot$ (slice on $v=\log\tau$)**
   $$\log\pi(v\mid\cdot)= -p\log\tau-\sum_j\log\tilde\lambda_j
   -\sum_j\frac{\beta_j^2}{2\phi_{g(j)}^2\tau^2\tilde\lambda_j^2\sigma^2}
   -\log\left(1+\frac{\tau^2}{\tau_0^2}\right)+v-\alpha_\tau\tau^2$$
   where $\alpha_\tau=\texttt{\_TAU\_PENALTY}\times\texttt{burnin\_w}$ and `_TAU_PENALTY=5e-5`. After sampling, code caps `tau <= _TAU_MAX` with `_TAU_MAX=5e3`.

5. **$\sigma^2\mid\cdot$ (inverse-gamma; Half-Cauchy via auxiliary refresh)**
   $$\sigma^2\sim\text{Inv-Gamma}\left(\alpha=\frac{n+p+1}{2},\ \beta=\frac12\|y-X\beta\|_2^2+\frac12\sum_j\frac{\beta_j^2}{\phi_{g(j)}^2\tau^2\tilde\lambda_j^2}+\frac{1}{\xi_\sigma}\right)$$
   $$\xi_\sigma\mid\sigma^2\sim\text{Inv-Gamma}\left(1,\ \frac{1}{s_0^2}+\frac{1}{\sigma^2}\right)$$

### 1.4 Sampling order (one chain)

Per iteration (see the `for it in ...` loop in `fit`):
1. Compute $\tilde\lambda$, prior variances `v_prior`
2. Sample $\beta$
3. Slice-sample each $\lambda_j$ (on $u_j=\log\lambda_j$)
4. Refresh per-$j$ auxiliary `xi_j` (inverse-gamma; currently not used elsewhere in the update)
5. Sample $\phi_g$ via GIG (and apply adaptive floor)
6. Slice-sample $\tau$ (on $v=\log\tau$), cap at `_TAU_MAX`
7. Refresh auxiliary `xi_tau` (inverse-gamma; currently not used elsewhere in the update)
8. Sample $\sigma^2$ and refresh $\xi_\sigma$; cap $\sigma^2$ at `_SIGMA2_FACTOR * var(y)`
9. Store draws after burn-in with thinning

Multi-chain:
- When `num_chains>1`, the class spawns independent single-chain fits (seeds `seed + chain_idx`) via `ProcessPoolExecutor`,
  then stacks arrays as `(chains, draws, ...)`.

### 1.5 Key numerical constants

From `grrhs/models/grrhs_gibbs.py`:
- `_PHI_EPS=1e-12`, `_PHI_BASE_FLOOR=2e-5`, `_PHI_ADAPT_COEFF=5e-7`, `_FLOOR_MIN_WEIGHT=0.002`
- `_TAU_MAX=5e3`, `_TAU_PENALTY=5e-5`
- `_SIGMA2_FACTOR=2.0`, `_BURNIN_WARM=1000`, `_RIDGE_ALPHA=1e-3`

---

## 2) `GIGGRegression` (Gaussian regression; Gibbs)

Implementation (default): `grrhs/models/gigg_regression.py:GIGGRegression` (pure NumPy Gibbs; repo-native).

Optional CRAN-compatible backend (Python port of CRAN `gigg` v0.2.1):
- `grrhs/models/gigg_cran.py:GIGGRegressionCRAN` (select via `model.backend: cran` for `model.name: gigg` / `gigg_regression`)
- Details: `docs/gigg_cran_python_port.md`

### 2.1 Parameters and defaults (code + method YAML)

Packages used:
- `GIGGRegression`: NumPy-only (no `scipy.stats`; scalar samplers are implemented directly for speed/portability).
- `GIGGRegressionCRAN`: NumPy-only (direct algorithmic port; includes a rejection-based scalar GIG sampler).

Code defaults:
- `method="mmle"`, `n_burn_in=500`, `n_samples=1000`, `n_thin=1`, `jitter=1e-8`, `seed=0`, `num_chains=1`
- `a_value=None`, `a_fixed_default=0.5`, `b_init=0.5`, `b_floor=1e-3`, `b_max=4.0`
- `tau_sq_init=1.0`, `sigma_sq_init=1.0`
- `mmle_update="paper_lambda_only"`, `mmle_burnin_only=true`
- `force_a_1_over_n=true` (in MMLE mode, keep $a_g=1/n$ as in the CRAN defaults; set false to use user-provided `a_value`)
- `fit_intercept=true` (if `C` is not passed, code uses `C=1` and samples `alpha`)
- `store_lambda=true`, `btrick=false`, `stable_solve=true`
- `lambda_constraint_mode="hard"`, caps via `lambda_cap` / `lambda_soft_cap`

Method config: `configs/methods/gigg.yaml`
- `model.method="fixed"`, `model.mmle_enabled=false` (so MMLE is off)
- `model.n_samples=1000`, `inference.gibbs.burn_in=500`, `inference.gibbs.thin=1`, `inference.gibbs.num_chains=4`
- `model.store_lambda=true`, `model.stable_solve=true`, `model.btrick=false`

Budget note:
- As with GRRHS, the runner overwrites burn-in / draws / chain counts under `experiments.bayesian_fairness` (see §7).

### 2.2 Hierarchy implied by the sampled $\beta$-block

The sampled $\beta$ conditional corresponds to:
$$y=C\alpha + X\beta + \varepsilon,\quad \varepsilon\sim\mathcal N(0,\sigma^2I)$$
$$\beta\mid\sigma^2,\tau^2,\gamma^2,\lambda^2\sim\mathcal N\!\left(0,\ \sigma^2\,\mathrm{diag}(\tau^2\gamma_{g(j)}^2\lambda_j^2)\right)$$
where `local_scale[j] = tau_sq * gamma_sq[group_id[j]] * lambda_sq[j]`.

### 2.3 Conditional updates (exact code forms)

One Gibbs step (`_gibbs_step`) updates:

1. **$\alpha\mid\cdot$ (Gaussian; only if `C` has columns)**
   - Uses `precision_alpha = C^T C + I * jitter`
   - Draw: $\alpha = \mu_\alpha + \sqrt{\sigma^2}\,\epsilon$ via Cholesky solve.

2. **$\beta\mid\cdot$ (Gaussian)**
   - $y_\text{tilde}=y-C\alpha$
   - Precision: $X^TX+\mathrm{diag}(1/\text{local\_scale})$
   - Mean: $(X^TX+\mathrm{diag}(1/\text{local\_scale}))^{-1}X^Ty_\text{tilde}$
   - Covariance: $\sigma^2(\cdot)^{-1}$
   - If `btrick=true`, uses the Bhattacharya et al. (2016) $n\times n$ draw.

3. **$\tau^2\mid\cdot$ (inverse-gamma)**
   - Shape: `tau_shape = 0.5*(p+1)`
   - Code uses
     `tau_rate = 0.5 * tau_sq * sum(beta^2 / local_scale) + 1/nu`
     (which simplifies because `local_scale` contains `tau_sq`).

4. **$\sigma^2\mid\cdot$ (inverse-gamma)**
   - Shape: `sigma_shape = 0.5*(n+1)`
   - Scale: `0.5 * ||y - X beta - C alpha||^2 + 1/nu`

5. **Per group $g$: $\gamma_g^2\mid\cdot$ (GIG with reciprocal branch)**
   - Code computes
     $$\psi_g^\ast=\frac{1}{\tau^2}\sum_{j\in g}\frac{\beta_j^2}{\lambda_j^2}$$
   - Then samples either $\gamma_g^2$ or $(\gamma_g^2)^{-1}$ with swapped GIG parameters depending on whether $p_g/2$ is smaller/larger than `p_vec[g]` (which depends on `method` / `a_value` / `a_fixed_default`).

6. **Per coefficient: $\lambda_j^2\mid\cdot$ (inverse-gamma)**
   - Shape: `lam_shape = q_vec[g] + 0.5` (where `q_vec` is the per-group $q_g$, initialized from `b_init`)
   - Scale: `eta[g] + beta[j]^2 / (2*tau_sq*gamma_sq[g])`
   - Implementation then enforces positivity and optional capping via `lambda_constraint_mode`.
   - `eta[g]` is set to `1.0` each step (`eta.fill(1.0)`).

7. **$\nu\mid\cdot$ (inverse-gamma augmentation)**
   $$\nu\sim\text{Inv-Gamma}\left(1,\ \frac{1}{\tau^2}+\frac{1}{\sigma^2}\right)$$

8. **MMLE update of $q_g$ (only when `method="mmle"`)**
   - Uses `_digamma_inv` on targets derived from `-mean(log(lambda_sq))`, with clipping to `[b_floor, b_max]`.
   - If `method="fixed"` (repo default), $q_g$ stays fixed at its initialization.

### 2.4 Sampling schedule

- Burn-in: run `_gibbs_step` exactly `n_burn_in` times.
- Sampling: run `_gibbs_step` exactly `n_samples * n_thin` times and store every `n_thin`.
- Multi-chain: same pattern as GRRHS (`ProcessPoolExecutor`, seeds `seed + chain_idx`, stack to `(chains, draws, ...)`).

### 2.5 CRAN parity note (answering “is my GIGG logic the CRAN logic?”)

- If you run `model.name: gigg` with the default backend, you are running `GIGGRegression` (repo-native Gibbs), which is *conceptually aligned* with a GIGG hierarchy but is not intended as a line-by-line CRAN clone.
- If you want the **same update order / helper routines / sampler structure** as the CRAN package you pasted (v0.2.1), use `model.backend: cran` to run `GIGGRegressionCRAN` (see `docs/gigg_cran_python_port.md`).

---

## 3) `RegularizedHorseshoeRegression` (Gaussian RHS; CmdStan HMC/NUTS)

Implementation: `grrhs/models/baselines/models.py:RegularizedHorseshoeRegression` + Stan file `grrhs/models/baselines/stan/rhs_gaussian_regression.stan`.

### 3.1 Method YAML defaults

Method config: `configs/methods/regularized_horseshoe.yaml`
- `scale_intercept=10.0`
- `nu_global=1.0`, `nu_local=1.0`
- `slab_scale=2.0`, `slab_df=4.0`
- `num_warmup=1000`, `num_samples=1000`, `num_chains=4`, `thinning=1`
- `target_accept_prob=0.999`, `max_tree_depth=14`, `progress_bar=false`, `seed=151`
- `tau.mode=calibrated`, `tau.target=coefficients`, `tau.p0.value=20`

### 3.2 Priors and parameterization (exact Stan program)

From `rhs_gaussian_regression.stan`:

- $y_{sd}=\max(\mathrm{sd}(y),10^{-9})$
- $z\sim\mathcal N(0,1)$ (vector length $d=p$)
- Intercept: $\beta_0\sim\mathcal N(0,\text{scale\_icept})$
- Noise: `logsigma ~ Normal(log(y_sd), 1.0)`, $\sigma=\max(\exp(\log\sigma),10^{-9})$
- Local scales:
  - `aux1_local[j] ~ Normal(0,1)` with `<lower=0>` (Half-Normal)
  - `aux2_local[j] ~ Inv-Gamma(nu_local/2, nu_local/2)`
  - $\lambda_j=\text{aux1}_{local,j}\sqrt{\text{aux2}_{local,j}}$
- Global scale:
  - `aux1_global ~ Normal(0,1)` with `<lower=0>` (Half-Normal)
  - `aux2_global ~ Inv-Gamma(nu_global/2, nu_global/2)`
  - $\tau=\text{aux1}_{global}\sqrt{\text{aux2}_{global}}\cdot\text{scale\_global}\cdot\sigma$
- Slab:
  - `caux ~ Inv-Gamma(slab_df/2, slab_df/2)`, $c=\text{slab\_scale}\sqrt{c_{aux}}$
- Regularized horseshoe:
  $$\tilde\lambda_j^2=\frac{c^2\lambda_j^2}{c^2+\tau^2\lambda_j^2+10^{-12}},\quad \beta_j=z_j\tilde\lambda_j\tau$$
- Likelihood: $y\sim\mathcal N(\beta_0+X\beta,\sigma)$

### 3.3 Sampling + diagnostics (Python wrapper)

- Packages used: `cmdstanpy` to run Stan HMC/NUTS (model code in `rhs_gaussian_regression.stan`).
- For gaussian likelihood, the class uses CmdStan (`_use_cmdstan_backend()` returns true).
- It runs HMC/NUTS via `CmdStanModel.sample(...)` with `iter_warmup=num_warmup`, `iter_sampling=num_samples`, `chains=num_chains`,
  `adapt_delta=target_accept_prob`, `max_treedepth=max_tree_depth`.
- Extracted posterior arrays (stored on the fitted model, then written by the runner):
  - `beta`, `beta0`, `sigma`, `tau`, `lambda`, `lambda_tilde`, `c`
- Extracted HMC diagnostics (used by runner hard checks):
  - `divergences`, `ebfmi_min`, `treedepth_hits` (computed from CmdStan columns `divergent__`, `energy__`, `treedepth__`).

---

## 4) `GRRHS_SVI_Numpyro` (Gaussian regression; variational inference)

Implementation: `grrhs/models/grrhs_svi_numpyro.py:GRRHS_SVI_Numpyro` (registered as `model.name="grrhs_svi"`).

Packages used:
- JAX + NumPyro (`numpyro.infer.SVI`, `Trace_ELBO`, `Predictive`) for variational inference and approximate posterior sampling.

### 4.1 Priors (model)

The SVI model uses the same prior family as GRRHS Gibbs:
- $\sigma\sim\text{Half-Cauchy}(0,s_0)$, $\tau\sim\text{Half-Cauchy}(0,\tau_0)$
- $\lambda_j\sim\text{Half-Cauchy}(0,1)$
- $\phi_g\sim\text{Half-Normal}(\eta/\sqrt{p_g})$
- $\beta_j\mid\cdot\sim\mathcal N(0,\phi_{g(j)}^2\tau^2\tilde\lambda_j^2\sigma^2)$
- $y\mid\beta,\sigma\sim\mathcal N(X\beta,\sigma)$

### 4.2 Guide (approximate posterior) and “sampling”

Guide structure (`_guide`):
- Uses log-normal auxiliaries for `tau`, `sigma`, `lambda` via `Normal(...)` on log-scale and `Delta(exp(.))`.
- Couples log-scales with a shared factor `u_shared ~ Normal(0,1)`.
- Per group: $q([\beta_g,\log\phi_g])$ is a multivariate normal with learned `loc_group_g` and lower-triangular `L_group_g`.

Optimization:
- Uses NumPyro `SVI` with `Trace_ELBO(num_particles=...)`.
- Defaults: `num_steps=3000`, `lr=1e-2`, `seed=42`.
- If `use_hutchinson=true`, sets `num_particles=hutchinson_samples` (default `8`).
- Only hard stop condition in code: raise if the ELBO loss becomes NaN.

Exported posterior-like draws:
- After training, code samples from the guide via `Predictive(guide, num_samples=num_samples_export)` (default `1000`).
- Exported arrays include `beta` (`coef_samples_`), `tau_samples_`, `sigma_samples_`, `lambda_samples_`, `phi_samples_`.
- These are samples from $q(\cdot)$ (not exact MCMC posterior draws).

---

## 5. `tau0` calibration (runner behavior)

In `grrhs/experiments/runner.py:_maybe_calibrate_tau`, when `model.tau.mode=calibrated`:
$$\tau_0=\frac{p_0}{D-p_0}\cdot\frac{\sigma_{ref}}{\sqrt{n}}$$

- $p_0$: `model.tau.p0.value` (or fallback) (method YAML defaults use `20`)
- $D$: number of groups if `target=groups`, else number of coefficients if `target=coefficients`
- Regression: $\sigma_{ref}$ is `sigma_reference` unless it is `"auto"`, in which case $\sigma_{ref}=\mathrm{sd}(y)$ on the training data.
- Lower bound: `tau0 >= 1e-8`
- Implementation detail: if a `p0.grid` is present, code currently builds `candidates` but uses only the first candidate (`candidates[0]`).

Where it is used:
- `GRRHS_Gibbs`: runner writes `model.tau0`.
- RHS: registry maps `model.tau0` → `scale_global` (see `grrhs/experiments/registry.py:_horseshoe_common_kwargs`).

---

## 6. Convergence diagnostics (R-hat / ESS / MCSE) and runner pass/fail

### 6.1 Diagnostics implementation (`grrhs/diagnostics/convergence.py`)

For each parameter block array, the code computes:
- Split-$\hat R$ by splitting each chain in half, doubling the chain count, then applying:
  $$W=\frac1C\sum_{c=1}^C s_c^2,\quad
  B=N\,\mathrm{Var}(\bar\theta_c),\quad
  \hat V=\frac{N-1}{N}W+\frac{1}{N}B,\quad
  \hat R=\sqrt{\hat V/W}$$
- ESS via per-lag autocorrelation with pair-sum truncation:
  $$ESS=\frac{CN}{1+2\sum_k\rho_k}$$
- MCSE/SD proxy for the posterior mean: $\sqrt{1/ESS}$

Shape requirements:
- Needs at least `draws >= 4`; if odd, drops one draw.
- `diagnostic_valid := raw_num_chains >= min_chains_for_rhat`.

### 6.2 Runner convergence gate (`grrhs/experiments/runner.py:_check_convergence`)

For Bayesian model names (see scope at top), runner checks:
- Block presence per `experiments.convergence.expected_blocks[model_name]`
- `diagnostic_valid` (when `require_valid_diagnostics=true`)
- `rhat_max <= max_rhat`
- `ess_min >= min_ess_by_block[block]` (fallback `min_ess`)
- `mcse_over_sd_max <= max_mcse_over_sd`

HMC hard checks (only for models listed in `experiments.convergence.hmc.models`):
- `divergences <= max_divergences`
- `ebfmi_min >= min_ebfmi`
- `treedepth_hits <= max_treedepth_hits`
- if `require_present=true`, missing HMC diagnostics causes failure.

### 6.3 Default thresholds (repo defaults in `configs/base.yaml`)

`experiments.convergence`:
- `enabled: true`
- `max_rhat: 1.01`
- `min_ess: 100`
- `min_ess_by_block: beta=400, tau=1000, phi=400, gamma=400, lambda=200, b=200`
- `max_mcse_over_sd: 0.10`
- `max_retries: 3`
- `retry_scale: 2.0`
- `missing_policy: fail`
- `require_valid_diagnostics: true`
- `min_chains_for_rhat: 4`
- `hmc.enabled: true`
- `hmc.models: ["regularized_horseshoe", "rhs", "regularised_horseshoe"]`
- `hmc.max_divergences: 0`, `hmc.min_ebfmi: 0.3`, `hmc.max_treedepth_hits: 0`, `hmc.require_present: true`

Expected blocks (base config):
- `grrhs_gibbs`: `beta, tau, phi, lambda`
- `gigg` / `gigg_regression`: `beta, tau, gamma, lambda`
- `regularized_horseshoe` / `rhs` / `regularised_horseshoe`: `beta, tau, lambda`

---

## 7. Sampling budgets (exact numbers)

### 7.1 Default shared fairness budget (repo defaults)

`configs/base.yaml -> experiments.bayesian_fairness`:
- `enabled: true`
- `enforce_shared_sampling_budget: true`
- `sampling_budget: burn_in=8000, kept_draws=8000, thinning=1, num_chains=4`
- `disable_budget_retry: false`

Runner application (`grrhs/experiments/runner.py:_apply_bayesian_sampling_budget`):
- GRRHS / GIGG:
  - `inference.gibbs.burn_in = 8000`, `thin = 1`, `num_chains = 4`
  - `model.iters = burn_in + kept_draws * thinning = 16000`
  - For GIGG: also sets `model.n_burn_in=8000`, `model.n_thin=1`, `model.n_samples=8000`
- RHS (NUTS / CmdStan interface):
  - `model.num_warmup = 8000`
  - `model.num_samples = kept_draws * thinning = 8000`
  - `model.num_chains = 4`, `model.thinning = 1`

### 7.2 Retry scaling (when convergence fails)

In `runner._fit_model_with_retry`:
- Total attempts = `1 + experiments.convergence.max_retries`, but:
  - if `experiments.bayesian_fairness.disable_budget_retry=true`, then `max_retries` is forced to `0`.
- Attempt `k` (0-indexed) scales runtime by `retry_scale ** k` after applying the shared budget.

### 7.3 Method-file budgets (checked-in, but overridden by default)

The method YAML files define smaller budgets, but they are overridden by the shared budget above unless
`experiments.bayesian_fairness.enforce_shared_sampling_budget=false`.

For reference:
- `configs/methods/grrhs_regression.yaml`: `iters=2000`, `burn_in=1000`, `thin=1`, `num_chains=4`
- `configs/methods/gigg.yaml`: `burn_in=500`, `n_samples=1000`, `thin=1`, `num_chains=4`
- `configs/methods/regularized_horseshoe.yaml`: `num_warmup=1000`, `num_samples=1000`, `num_chains=4`, `thinning=1`,
  `target_accept_prob=0.999`, `max_tree_depth=14`

---

## 8. Posterior artifacts written to disk

Per fold (when posterior arrays exist), runner writes:
- `posterior_samples.npz`
- `convergence.json`
- `posterior_summary.parquet` (or `.csv` fallback)
- `posterior_validation.json` (when enabled)

Array collection mapping (attribute → key in `posterior_samples.npz`), from `runner._collect_posterior_arrays`:
- `coef_samples_ -> beta`
- `alpha_samples_ -> alpha`
- `intercept_samples_ -> intercept`
- `sigma_samples_ -> sigma`
- `sigma2_samples_ -> sigma2`
- `tau_samples_ -> tau`
- `phi_samples_ -> phi`
- `gamma_samples_ -> gamma`
- `lambda_samples_ -> lambda`
- `lambda_group_samples_ -> group_lambda`
- `b_samples_ -> b`
- `c_samples_ -> c`
- `loglik_samples_ -> loglik`

Not every model exports every key; missing keys simply do not appear in the `.npz`.

---

## 9. Posterior Validation auto-fail flow (SBC / PPC / seed stability)

Enabled by default in `configs/base.yaml` (`experiments.posterior_validation.enabled: true`).

Runner behavior:
- When `apply_to_bayesian_only=true`, this gate is only applied to model names in `_BAYESIAN_MODEL_NAMES` (see top of doc).
- If it fails, fold status becomes `INVALID_POSTERIOR_VALIDATION` (runner logic near `_run_posterior_validation`).

### 9.1 SBC (simulation-based calibration proxy on coefficient truth)

Defaults from `configs/base.yaml -> experiments.posterior_validation.sbc`:
- `ks_pvalue_min: 0.05`
- `coverage_level: 0.9`
- `coverage_tolerance: 0.15`
- `min_coefficients: 8`, `max_coefficients: 128`
- `fail_on_missing_truth: false`, `fail_on_missing_draws: true`

### 9.2 PPC (posterior predictive checks)

Defaults from `configs/base.yaml -> experiments.posterior_validation.ppc`:
- `tail_prob: 0.025`
- `min_draws: 200`
- `fail_on_missing_draws: true`
- `use_train_data: true` (runner default unless overridden)

### 9.3 Seed stability (multi-restart)

Defaults from `configs/base.yaml -> experiments.posterior_validation.seed_stability`:
- `num_restarts: 2`, `seed_stride: 1009`
- `max_beta_rel_l2: 0.15`, `min_beta_cosine: 0.98`
- `tau_stability_mode: log_median`, `max_tau_log_sd: 0.35`, `max_tau_rel_sd: 0.20`
- `fail_on_missing_tau: false`

---

## 10. `sim1_bayes3_pass` profile (repo override file)

For a lighter-weight “Bayesian-3” pass-oriented configuration, the repo provides:
- `configs/overrides/sim1_bayes3_pass.yaml`

Key differences vs `configs/base.yaml`:
- Fairness sampling budget: `burn_in=1500`, `kept_draws=1500`, `num_chains=4`
- Convergence thresholds relaxed (e.g. `max_rhat=1.15`, `min_ess=50`, and `max_retries=0`)
- `disable_budget_retry=true` (so retries are forced off)
- NUTS tuning override: `inference.nuts.target_accept_prob=0.99`, `inference.nuts.max_tree_depth=13`

RHS implementation note:
- Stan stabilization lives in `grrhs/models/baselines/stan/rhs_gaussian_regression.stan` (e.g. `+ 1e-12` in the `lambda_tilde` denominator).
