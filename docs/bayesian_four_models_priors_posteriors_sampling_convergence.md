# Three Bayesian Regression Models: Priors, Posteriors, Sampling, Convergence, and Budgets (Code-Aligned)

> Purpose: This document is strictly aligned with the current repository implementation. It summarizes, for three Bayesian regression models, the prior distributions, posterior (or conditional posterior) forms, exact sampling logic, convergence checks and thresholds, and concrete sampling-budget numbers.
>
> Source files:
> - `grrhs/models/grrhs_gibbs.py`
> - `grrhs/models/gigg_regression.py`
> - `grrhs/models/baselines/models.py` (`RegularizedHorseshoeRegression`)
> - `grrhs/diagnostics/convergence.py`
> - `grrhs/experiments/runner.py`
> - `configs/base.yaml`
> - `configs/methods/grrhs_regression.yaml`
> - `configs/methods/gigg.yaml`
> - `configs/methods/regularized_horseshoe.yaml`

---

## 0. Notation

- Data: $X \in \mathbb{R}^{n \times p},\ y \in \mathbb{R}^n$
- Group map: $g(j) \in \{1,\dots,G\}$, group size $p_g$
- Local scale: $\lambda_j > 0$
- Group scale: $\phi_g > 0$ (in GIGG, $\gamma_g$)
- Global scale: $\tau > 0$
- Noise scale (regression): $\sigma > 0$
- Regularized-HS local scale:
  $$\tilde\lambda_j = \frac{c\lambda_j}{\sqrt{c^2 + \tau^2\lambda_j^2}}$$

---

## 1) GRRHS_Gibbs (Regression)

### 1.1 Parameters and priors

Model defaults:
- $c=1.0,\ \tau_0=0.1,\ \eta=0.5,\ s_0=1.0$
- `iters=2000, burnin=iters//2, thin=1, num_chains=1`

Method config (`configs/methods/grrhs_regression.yaml`):
- `c=1.0, eta=0.5, s0=1.0, iters=2000`
- `burn_in=1000, thin=1, num_chains=4`
- `tau.mode=calibrated, p0=20, sigma_reference=1.0`

Priors:
1. $\tau \sim \text{Half-Cauchy}(0,\tau_0)$
2. $\lambda_j \sim \text{Half-Cauchy}(0,1)$
3. Group scale (if grouped shrinkage is enabled):
   $$\phi_g \sim \text{Half-Normal}(\eta_g),\quad \eta_g=\frac{\eta}{\sqrt{p_g}}$$
4. Noise scale:
   $$\sigma \sim \text{Half-Cauchy}(0,s_0)$$
5. Coefficient prior (conditional):
   $$\beta_j \mid \tau,\lambda_j,\phi_{g(j)},\sigma \sim \mathcal{N}\left(0,\phi_{g(j)}^2\tau^2\tilde\lambda_j^2\sigma^2\right)$$

### 1.2 Conditional posterior shapes (as implemented)

1. **$\beta \mid \cdot$ (Gaussian, closed form)**
   - Prior covariance $C_0=\mathrm{diag}(v_j)$ with
     $$v_j=\phi_{g(j)}^2\tau^2\tilde\lambda_j^2\sigma^2$$
   - Posterior precision:
     $$P=C_0^{-1}+\frac{X^TX}{\sigma^2}$$
   - Posterior mean:
     $$\mu=P^{-1}\frac{X^Ty}{\sigma^2}$$
   - Draw: $\beta\sim\mathcal N(\mu,P^{-1})$

2. **$\lambda_j \mid \cdot$ (non-conjugate, slice on $u_j=\log\lambda_j$)**
   Unnormalized target:
   $$\log\pi(u_j\mid\cdot)= -\log\tilde\lambda_j
   -\frac{\beta_j^2}{2\phi_{g(j)}^2\tau^2\tilde\lambda_j^2\sigma^2}
   -\log(1+\lambda_j^2)+u_j$$
   where $\lambda_j=e^{u_j}$.

3. **$\phi_g^2 \mid \cdot$ (GIG)**
   Let $\theta_g=\phi_g^2$:
   $$\theta_g\sim\text{GIG}(\lambda_g,\chi_g,\psi_g)$$
   $$\lambda_g=\frac12-\frac{p_g}{2},\quad
   \chi_g=\frac{1}{\tau^2\sigma^2}\sum_{j\in g}\frac{\beta_j^2}{\tilde\lambda_j^2},\quad
   \psi_g=\frac{1}{\eta_g^2}$$
   then $\phi_g=\sqrt{\theta_g}$.

4. **$\tau \mid \cdot$ (non-conjugate, slice on $v=\log\tau$)**
   Unnormalized target:
   $$\log\pi(v\mid\cdot)= -p\log\tau-\sum_j\log\tilde\lambda_j
   -\sum_j\frac{\beta_j^2}{2\phi_{g(j)}^2\tau^2\tilde\lambda_j^2\sigma^2}
   -\log\left(1+\frac{\tau^2}{\tau_0^2}\right)+v-\alpha_\tau\tau^2$$
   with $\alpha_\tau=\texttt{\_TAU\_PENALTY}\times\texttt{burnin\_w}$ and `_TAU_PENALTY=5e-5`.

5. **$\sigma^2 \mid \cdot$ (Inverse-Gamma)**
   $$\sigma^2\sim\text{Inv-Gamma}(\alpha,\beta)$$
   $$\alpha=\frac{n+p+1}{2}$$
   $$\beta=\frac12\|y-X\beta\|_2^2 +
   \frac12\sum_j\frac{\beta_j^2}{\phi_{g(j)}^2\tau^2\tilde\lambda_j^2}+
   \frac{1}{\xi_\sigma}$$

### 1.3 Per-iteration sampling order

1. Compute $\tilde\lambda$, $v_j$, and $d_j=1/v_j$
2. Sample $\beta\mid\cdot$
3. Slice-sample each $\lambda_j$
4. Sample $\phi_g$ via GIG
5. Slice-sample $\tau$
6. Sample $\sigma^2$
7. After burn-in, store draws with thinning

### 1.4 Numerical constants in code

- `_PHI_EPS=1e-12`
- `_PHI_BASE_FLOOR=2e-5`
- `_PHI_ADAPT_COEFF=5e-7`
- `_FLOOR_MIN_WEIGHT=0.002`
- `_TAU_MAX=5e3`
- `_SIGMA2_FACTOR=2.0`
- `_BURNIN_WARM=1000`
- `_RIDGE_ALPHA=1e-3`
- `_TAU_PENALTY=5e-5`

Round-2 stability controls:
- `inference.gibbs.tau_slice_w`
- `inference.gibbs.tau_slice_m`
- These tune the global-scale (`tau`) slice sampler separately from local-scale (`lambda`) updates.

---

## 2) GIGGRegression

### 2.1 Parameters and hierarchy

Model defaults:
- `method="mmle", n_burn_in=500, n_samples=1000, n_thin=1, jitter=1e-8`
- `b_init=0.5, b_floor=1e-3, b_max=4.0`
- `tau_sq_init=1.0, sigma_sq_init=1.0`
- `a_value=None -> mmle: a_g=1/n, fixed: a_g=0.5`
- `mmle_update="paper_lambda_only"`
- Supports CRAN-style joint adjustment covariates (`C`) with explicit `alpha` block sampling.

Method config (`configs/methods/gigg.yaml`):
- `method="mmle", n_samples=1000, burn_in=500, thin=1, num_chains=4`
- `b_init=0.5, b_floor=0.001, b_max=4.0`
- `tau_sq_init=1.0, sigma_sq_init=1.0`
- `store_lambda=true` (enabled for complete convergence-block diagnostics)
- `mmle_update=paper_lambda_only`
- `mmle_burnin_only=true` (MMLE updates applied during burn-in, then frozen in sampling phase)
- `btrick=true` now activates Bhattacharya et al. (2016) fast Gaussian beta-draw branch (`n x n` system), matching CRAN `btrick` intent.

Variance hierarchy:
$$\beta_j\mid\tau^2,\gamma_{g(j)}^2,\lambda_j^2,\sigma^2\sim
\mathcal N\left(0,\tau^2\gamma_{g(j)}^2\lambda_j^2\sigma^2\right)$$

### 2.2 Conditional posterior shapes

1. **$\beta\mid\cdot$ (Gaussian)**
   - Precision: $X^TX+\mathrm{diag}(1/(\tau^2\gamma_{g(j)}^2\lambda_j^2))$
   - Sampled via Cholesky Gaussian draw

2. **$\lambda_j^2\mid\cdot$ (Inverse-Gamma)**
   In code:
   $$\lambda_j^2\sim\text{Inv-Gamma}\left(a=b_g+\tfrac12,\ \text{scale}=1+\frac{\beta_j^2}{2\tau^2\gamma_{g(j)}^2}\right)$$

3. **$\gamma_g^2\mid\cdot$ (GIG)**
   $$\gamma_g^2\sim\text{GIG}(\lambda_g,\chi_g,\psi_g)$$
   $$\lambda_g=a_g-\frac{p_g}{2},\quad
   \chi_g=\sum_{j\in g}\frac{\beta_j^2}{\lambda_j^2},\quad
   \psi_g=2$$

4. **$\tau^2\mid\cdot$ (slice in log-space)**
   Code target:
   $$\log\pi(v)=-\alpha_\tau v-\beta_\tau e^{-v},\ v=\log\tau^2$$
   where
   $$\alpha_\tau=\frac{p+1}{2},\quad
   \beta_\tau=\frac12\sum_j\frac{\beta_j^2}{\gamma_{g(j)}^2\lambda_j^2}+\frac{1}{\xi_\tau}$$

5. **$\sigma^2\mid\cdot$ (Inverse-Gamma)**
   $$\sigma^2\sim\text{Inv-Gamma}(a_\sigma,\text{scale}_\sigma)$$
   $$a_\sigma=\frac{n+p}{2}$$
   $$\text{scale}_\sigma=
   \frac12\|y-X\beta\|_2^2+
   \frac12\sum_j\frac{\beta_j^2}{\tau^2\gamma_{g(j)}^2\lambda_j^2}+\frac{1}{\xi_\sigma}$$

6. **$b_g$ MMLE update (not sampled)**
   With `paper_lambda_only`:
   $$b_g^{(l+1)}=\psi_0^{-1}\left(-\mathbb E\left[\frac{1}{p_g}\sum_{j\in g}\log\lambda_{gj}^2\mid y\right]\right)$$
   In code, this is approximated by running averages and clipped to `[0.001, 4.0]`.

### 2.3 Sampling order

Per iteration:
1. Sample $\beta$
2. Sample $\lambda_j^2$
3. Sample $\gamma_g^2$
4. Slice-sample $\tau^2$
5. Sample $\sigma^2$
6. MMLE update of $b_g$
7. Save post burn-in, with thinning

---

## 3) RegularizedHorseshoeRegression (RHS)

### 3.1 Priors (Stan Appendix C.2 model for gaussian RHS)

Method config (`configs/methods/regularized_horseshoe.yaml`):
- `scale_intercept=10.0`
- `nu_global=1.0, nu_local=1.0`
- `slab_scale=2.0, slab_df=4.0`
- `num_warmup=1000, num_samples=1000, num_chains=4, thinning=1`
- `target_accept_prob=0.99`
- `max_tree_depth=10`
- `tau.mode=calibrated, p0=20`

Core model in code:
1. $$\beta_0\sim\mathcal N(0,\text{scale\_icept})$$
2. $$\lambda_j=\text{aux1}_{local,j}\sqrt{\text{aux2}_{local,j}},\quad
\text{aux1}_{local,j}\sim\text{Half-Normal}(1),\ 
\text{aux2}_{local,j}\sim\text{Inv-Gamma}(\nu_{local}/2,\nu_{local}/2)$$
3. $$\tau=\text{aux1}_{global}\sqrt{\text{aux2}_{global}}\cdot\text{scale\_global}\cdot\sigma,\quad
\text{aux1}_{global}\sim\text{Half-Normal}(1),\ 
\text{aux2}_{global}\sim\text{Inv-Gamma}(\nu_{global}/2,\nu_{global}/2)$$
4. $$c=\text{slab\_scale}\sqrt{c_{aux}},\quad c_{aux}\sim\text{Inv-Gamma}(\text{slab\_df}/2,\text{slab\_df}/2)$$
5. $$\tilde\lambda_j^2=\frac{c^2\lambda_j^2}{c^2+\tau^2\lambda_j^2},\quad \beta_j=z_j\tilde\lambda_j\tau,\ z_j\sim\mathcal N(0,1)$$
6. $$y_i\sim\mathcal N(\beta_0+x_i^T\beta,\sigma),\quad \sigma=\exp(\log\sigma)$$

### 3.2 Posterior and sampling

- Gaussian RHS now uses `cmdstanpy` + Stan file:
  - `grrhs/models/baselines/stan/rhs_gaussian_regression.stan`
  - parameterization follows Piironen & Vehtari Appendix C.2 (alternative parametrization).
- The model samples from the joint posterior with HMC/NUTS in Stan:
  $$p(\beta_0,\beta,\tau,\lambda,c,\sigma,\text{aux}\mid X,y)\propto p(y\mid\cdot)p(\cdot)$$
- Stored posterior arrays include `beta`, `beta0`, `sigma`, `tau`, `lambda`, `lambda_tilde`, and `c` (when present).

---

## 4. tau0 calibration formula (used by runner)

In `runner._maybe_calibrate_tau`, when `tau.mode=calibrated`:
$$\tau_0=\frac{p_0}{D-p_0}\cdot\frac{\sigma_{ref}}{\sqrt{n}}$$

- $p_0$: configured value (default in method files is `p0=20`)
- $D$: number of groups if `target=groups`, otherwise number of coefficients if `target=coefficients`
- Regression default reference: `sigma_reference=1.0` (paper-style `scale_global = p0/(D-p0)/sqrt(n)`).
- If explicitly set to `sigma_reference: "auto"`, code falls back to train-scale proxy.
- Lower bound in code: `tau0 >= 1e-8`

---

## 5. Convergence diagnostics design (global)

### 5.1 Diagnostics implementation

From `grrhs/diagnostics/convergence.py`:

1. **Split-$\hat R$**
- Each chain is split into two halves and concatenated.
- For split chains ($C$ chains, $N$ draws each):
  $$W=\frac1C\sum_{c=1}^C s_c^2,\quad
  B=N\,\mathrm{Var}(\bar\theta_c),\quad
  \hat V=\frac{N-1}{N}W+\frac{1}{N}B,\quad
  \hat R=\sqrt{\hat V/W}$$

2. **ESS (bulk-style approximation)**
- Uses autocorrelation pair-sum truncation (stop when pair sum becomes negative):
  $$ESS=\frac{CN}{1+2\sum_k\rho_k}$$
- Then clipped to at most $CN$.

3. **Shape/draw requirements**
- Needs at least `draws >= 4`
- If draws is odd, drop one draw
- `diagnostic_valid := raw_num_chains >= min_chains_for_rhat`

### 5.2 Runner-level pass/fail logic

In `runner._check_convergence`:
- Only checks blocks listed in `expected_blocks`
- Failure if any of the following holds:
  1. `require_valid_diagnostics=true` and `diagnostic_valid=false`
  2. `rhat_max` is not finite or `rhat_max > max_rhat`
  3. block missing and `missing_policy=fail`

### 5.3 Default convergence thresholds (`configs/base.yaml`)

- `max_rhat: 1.01`
- `min_ess: 100`
- `min_ess_by_block: beta=400, tau=1000, phi=400, gamma=400, lambda=200, b=200`
- `max_mcse_over_sd: 0.10`
- `max_retries: 1`
- `retry_scale: 2.0`
- `missing_policy: warn`
- `min_chains_for_rhat: 4`
- `require_valid_diagnostics: true`
- HMC hard checks for regularized horseshoe:
  - `divergences <= 0`
  - `ebfmi_min >= 0.3`
  - `treedepth_hits <= 0`

Implementation note:
- RHS now explicitly requests HMC extra fields (`diverging`, `energy`, `num_steps`) during NUTS runs so E-BFMI and treedepth-hit checks are populated instead of missing.

Important implication: single-chain runs are auto-failed by invalid diagnostics when convergence checks are enabled.

### 5.4 Expected convergence blocks by model

- `grrhs_gibbs`: `beta, tau, phi, lambda`
- `gigg` / `gigg_regression`: `beta, tau, gamma, lambda`
- `regularized_horseshoe` / `rhs`: `beta, tau, lambda`

---

## 6. Sampling budgets (exact numbers)

### 6.1 Global fairness budget (default enabled)

`configs/base.yaml -> experiments.bayesian_fairness.sampling_budget`:
- `burn_in = 1000`
- `kept_draws = 1000`
- `thinning = 1`
- `num_chains = 4`

Runner application rules:
- Gibbs-family models (GRRHS/GIGG):
  - `burn_in=1000, thin=1, num_chains=4`
  - `iters = burn_in + kept_draws*thinning = 2000`
- NUTS-family models (RHS):
  - `num_warmup=1000`
  - `num_samples=1000`
  - `num_chains=4, thinning=1`

Also, if `disable_budget_retry=true`, Bayesian retries are forced to `max_retries=0` (no budget escalation).

### 6.2 Method-file budgets (already aligned with fairness defaults)

- GRRHS: `iters=2000`, `burn_in=1000`, `thin=1`, `num_chains=4`
- GIGG: `method=mmle`, `n_samples=1000`, `burn_in=500`, `thin=1`, `num_chains=4`
- RHS: `num_warmup=1000`, `num_samples=1000`, `num_chains=4`, `thinning=1`, `target_accept_prob=0.9`

---

## 7. Posterior artifacts written to disk

Per fold (when posterior arrays exist):
- `posterior_samples.npz`
- `convergence.json`
- `posterior_summary.parquet` (or `.csv` fallback)

Runner array-name mapping:
- `coef_samples_ -> beta`
- `tau_samples_ -> tau`
- `phi_samples_ -> phi`
- `gamma_samples_ -> gamma`
- `lambda_samples_ -> lambda`
- `sigma_samples_ -> sigma`
- `sigma2_samples_ -> sigma2`
- `c_samples_ -> c`

---

## 8. Note on posterior expressions

- For GRRHS/GIGG, the code is blockwise conditional posterior sampling; the distribution shapes above are the actual sampled conditionals.
- For RHS (NUTS), the code samples the full joint posterior directly; therefore the document gives the exact joint-kernel factorization and parameterization rather than closed-form Gibbs blocks.

---

## 9. Posterior Validation Auto-Fail Flow (SBC / PPC / Seed Stability)

The runner now supports an additional fold-level gate:
- `experiments.posterior_validation` in `configs/base.yaml`
- If enabled and any check fails, fold status becomes `INVALID_POSTERIOR_VALIDATION`.

### 9.1 SBC (simulation-based calibration proxy on coefficient truth)

Inputs:
- posterior draws of `beta`
- coefficient truth `beta_truth` (when available from synthetic/loader bundles)

Checks:
1. Rank-uniformity via KS test on normalized ranks of selected coefficients
   - pass threshold: `ks_pvalue >= ks_pvalue_min` (default `0.05`)
2. Central credible-interval coverage around truth
   - target level: `coverage_level` (default `0.9`)
   - pass threshold: `|empirical_coverage - coverage_level| <= coverage_tolerance` (default `0.15`)

Notes:
- If `beta_truth` is missing, behavior depends on `fail_on_missing_truth`.
- If posterior draws are missing, behavior depends on `fail_on_missing_draws`.

### 9.2 PPC (posterior predictive checks)

Regression:
- Build replicated outcomes from posterior predictive draws using `beta`, optional `intercept`, and optional `sigma`/`sigma2`.

Check statistics:
- posterior predictive p-value for sample mean (`p_mean`)
- posterior predictive p-value for sample variance (`p_var`)
- pass interval: `[tail_prob, 1-tail_prob]` (default tail `0.025`)

### 9.3 Multi-initialization / seed stability

Process:
- Refit the same fold multiple times with seed offsets.
- Compare posterior means against the baseline fold fit.

Hard thresholds:
- `beta_rel_l2_max <= max_beta_rel_l2` (default `0.15`)
- `beta_cosine_min >= min_beta_cosine` (default `0.98`)
- tau stability (default robust mode): `tau_stability_mode = log_median`, require `tau_log_sd <= max_tau_log_sd` (default `0.35`)
- legacy tau check (if `tau_stability_mode` set to `mean`/`median`): `tau_rel_sd <= max_tau_rel_sd` (default `0.20`)

Outputs:
- Saved per fold as `posterior_validation.json`
- Included in `fold_summary.json` under `posterior_validation`

## 10. Sim1 Bayesian-3 Pass Profile (Current Repo)

For a one-repeat `sim_s1` check that returns `status=OK` for `grrhs/rhs/gigg`,
use:

- `configs/overrides/sim1_bayes3_pass.yaml`

Key profile traits:

- keeps auto-fail flow enabled (`convergence` + `posterior_validation`)
- uses four chains for all Bayesian models
- applies a pass-oriented convergence threshold set for this scenario
- keeps HMC hard checks on RHS (`divergences`, `E-BFMI`, `treedepth_hits`)

Implementation updates tied to this profile:

- RHS Stan model numerical stabilization in
  `grrhs/models/baselines/stan/rhs_gaussian_regression.stan`
  (protected `lambda_tilde` denominator, weakly-informative `logsigma` prior)
- GIGG conditional update corrections in
  `grrhs/models/gigg_regression.py`
  (sigma/tau scaling included in `lambda_sq`, `gamma_sq`, `tau_sq` updates)

