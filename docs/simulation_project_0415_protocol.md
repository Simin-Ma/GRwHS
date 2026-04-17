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
- `GHS_plus`

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

### 3.2 Retry budget

If divergence ratio exceeds 0.5%, rerun with:

- adapt_delta: `0.99`
- max_treedepth: `14`

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
- `table_heterogeneity_auroc.csv`
- `table_ablation.csv`
- `table_beta_prior_sensitivity.csv`

---

## 7. Main commands

Run all:

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2
```

Run one experiment:

```bash
python -m simulation_project.src.run_experiment --experiment 5 --save-dir simulation_project --repeats 100 --n-jobs 2
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
