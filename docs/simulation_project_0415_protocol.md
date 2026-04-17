# Simulation Project 0415 Protocol (Detailed)

This document specifies the implementation-level behavior of the standalone simulation pipeline at:

- `simulation_project/src/run_experiment.py`

It is intended as a reproducibility companion for manuscript preparation.

---

## 1. Scope

The standalone pipeline executes 9 experiments:

1. Null-group contraction (profile specialization)
2. Adaptive localization (profile specialization)
3. Strong-signal phase diagram (profile specialization)
4. Grouped linear benchmark (full fitting)
5. Heterogeneity and group allowance (full fitting)
6. Grouped logistic weak-identification (full fitting)
7. Ablation (full fitting)
8. Global-scale calibration (cheap prior simulation)
9. Beta prior sensitivity for kappa (full fitting, small scale)

Primary benchmark methods:

- `GR_RHS`
- `RHS`
- `GIGG_MMLE`
- `GIGG_b_small`
- `GIGG_GHS`
- `GIGG_b_large`
- `GHS_plus`
- `OLS`
- `LASSO_CV`

`laptop` profile default uses a reduced subset:

- `GR_RHS`, `RHS`, `GIGG_MMLE`, `GHS_plus`, `OLS`, `LASSO_CV`

Implementation notes for current revision:

- `exp4` accepts `snr_list` and writes `target_snr` into `raw_results.csv` and `summary.csv`.
- `exp4` includes extra design settings: `L6` (unequal groups with distributed within-group signal) and `L4B20` (higher cross-group correlation, `rho_between=0.2`).
- `exp6` uses `min_separator_auc=0.8` by default (configurable in API).
- `exp8` tau calibration uses mixed strong/weak active coefficients instead of a single fixed active magnitude.
- `exp8` reuses one synthetic dataset per `(p0, replicate)` across all tau modes for paired comparisons and reduced duplicated compute.
- `exp9` evaluates all prior pairs on one shared scenario-replicate dataset (paired prior sensitivity).
- benchmark experiments (`exp4/5/6`) support method subsets and compute profiles (`full`, `laptop`).
- GIGG fitting enables Bhattacharya fast beta draw (`btrick=True`) with profile-specific iteration budgets.
- convergence-enforced mode retries Bayesian fits with expanded sampling budgets.
- optional "until converged" mode (`--until-bayes-converged`) keeps adding budget until convergence, with internal safety hard cap (currently 12 retries) to avoid infinite loops.
- current CLI default for `exp4-9`: when convergence enforcement is active and `--max-convergence-retries` is omitted, it runs in until-converged mode.
- non-converged fits after the retry budget/hard-cap are marked failed and excluded from posterior-trust metrics.
- benchmark summaries include `n_total_runs`, `n_effective`, and `valid_rate`.
- `exp4` emits both `summary_all.csv` and `summary_converged.csv` for all-runs vs converged-only reporting.
- `exp4-9` emit paired-converged summaries using the common replicate intersection across compared methods/configs.
- default table outputs (`table_*.csv`) now prefer paired-converged summaries when available.
- `exp4-9` emit convergence audit files: per-method `convergence_audit.csv` and intersection-level `paired_convergence_audit.csv`.

---

## 2. Reproducibility Rules

### 2.1 Seed scheme

- `MASTER_SEED = 20260415`
- Per-run seed formula:
  - `seed = MASTER_SEED + 100000 * experiment_id + 1000 * setting_id + replicate_id`

### 2.2 Pairing policy

Within each replicate, all methods are fit on the same dataset draw.

### 2.3 Dataset preprocessing

- Grouped linear/logistic: feature columns are centered and standardized.
- Grouped normal means: no extra normalization.

---

## 3. Bayesian compute budget and diagnostics

### 3.1 Default budget

- chains: `2`
- warmup: `500`
- posterior draws: `500`
- adapt_delta: `0.95`
- max_treedepth: `12`

### 3.1.1 Laptop profile budget

- non-`exp8`: `chains=1`, `warmup=250`, `post_warmup_draws=250`
- `exp8`: `chains=2`, `warmup=300`, `post_warmup_draws=300`
- relaxed convergence gate in laptop mode: `Rhat < 1.03`, `ESS > 120`, divergence ratio `< 1%`

### 3.2 Retry budget

If divergence ratio exceeds 0.5%, rerun with:

- adapt_delta: `0.99`
- max_treedepth: `14`

### 3.2.1 Convergence-enforced retries

- Available for Bayesian methods in `exp4-9`.
- Controlled by CLI:
  - `--max-convergence-retries K`
  - `--until-bayes-converged`
  - `--no-enforce-bayes-convergence` (default is enforce).
- Retry strategy scales warmup/draws and increases NUTS controls (`adapt_delta`, `max_treedepth`) per attempt.
- If still not converged after all attempts (or safety cap in until-converged mode), that fit is marked as failed (posterior deemed not trustworthy).

### 3.3 Convergence gate

A fit is marked valid only if all hold:

- `Rhat < 1.01`
- `Bulk ESS > 400`
- divergence ratio `< 0.5%`

Failed fits remain missing; datasets are not replaced.

---

## 4. Parallel execution model

The standalone pipeline supports true replicate-level parallelism via `n_jobs`:

- `n_jobs=1`: sequential
- `n_jobs>1`: process-based workers (`ProcessPoolExecutor`)

Implementation choices:

- workers are top-level functions to support Windows pickling
- work is partitioned by replicate (or setting-replicate pair)
- method comparison remains paired within replicate
- progress bars are shown via `tqdm`:
  - per-experiment task progress
  - all-experiments global progress when running `--experiment all`

---

## 4.1 Runtime expectation (rough)

Approximate laptop-level runtime at default repeats:

- Exp1: minutes
- Exp2: minutes
- Exp3: minutes
- Exp4: hours
- Exp5: hours
- Exp6: hours
- Exp7: hours
- Exp8: under one minute
- Exp9: hours

Combined full run is typically on the order of tens of hours (hardware dependent).

---

## 5. Exp3 threshold interface

Phase diagram uses explicit threshold interface:

- `theta_u0_rho(u0, rho)`
- `xi_crit_u0_rho(u0, rho) = theta_u0_rho / 2`

Concrete function (0415 paper eq. 65, evaluated at `kappa = u0`):

- `theta_u0_rho(u0, rho) = u0 * rho^2 / (u0 + (1 - u0) * rho^2)`
- `xi_crit_u0_rho(u0, rho) = theta_u0_rho(u0, rho) / 2`

where `rho = tau / sqrt(sigma2)`.

Metadata is written to:

- `simulation_project/results/exp3_phase_diagram/phase_threshold_meta.json`

---

## 6. Output contract

Each experiment emits:

- `raw_results.csv`
- `summary.csv` (or split summaries)
- figures under `simulation_project/figures`
- logs under `simulation_project/logs`

Tables expected from the suite:

- `table_benchmark_linear.csv`
- `table_benchmark_linear_all.csv`
- `table_benchmark_linear_converged.csv`
- `table_benchmark_linear_paired_converged.csv`
- `table_heterogeneity_auroc.csv`
- `table_heterogeneity_auroc_all.csv`
- `table_heterogeneity_auroc_paired_converged.csv`
- `table_ablation.csv`
- `table_ablation_all.csv`
- `table_ablation_paired_converged.csv`
- `table_beta_prior_sensitivity.csv`
- `table_beta_prior_sensitivity_all.csv`
- `table_beta_prior_sensitivity_paired_converged.csv`

---

## 7. Main commands

Run all:

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2
```

Run all with laptop profile:

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2 --profile laptop
```

Run with stricter convergence retries:

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2 --profile full --max-convergence-retries 3
```

Run in until-converged mode:

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2 --profile full --until-bayes-converged
```

Run one experiment:

```bash
python -m simulation_project.src.run_experiment --experiment 5 --save-dir simulation_project --repeats 100 --n-jobs 2
```

Programmatic calls for newly exposed controls:

```python
from simulation_project.src.run_experiment import run_exp4_benchmark_linear, run_exp6_grouped_logistic

run_exp4_benchmark_linear(
    save_dir="simulation_project",
    repeats=30,
    n_jobs=2,
    snr_list=[0.2, 1.0, 5.0],
    profile="laptop",
)
run_exp6_grouped_logistic(
    save_dir="simulation_project",
    repeats=50,
    n_jobs=2,
    min_separator_auc=0.8,
    profile="laptop",
)
```

Experiment id mapping:

- `1`: exp1 null contraction
- `2`: exp2 adaptive localization
- `3`: exp3 phase diagram
- `4`: exp4 benchmark linear
- `5`: exp5 heterogeneity
- `6`: exp6 grouped logistic
- `7`: exp7 ablation
- `8`: exp8 tau calibration
- `9`: exp9 beta prior sensitivity

---

## 8. Repository status

This repository is intentionally trimmed to the standalone 0415 simulation path.

`grrhs` now only retains the minimum model/diagnostic components required by
`simulation_project`, and the legacy runner/sweep stack has been removed.
