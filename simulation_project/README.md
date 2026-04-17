# Simulation Project (0415 Spec Implementation)

This folder contains the standalone 9-experiment simulation pipeline built from the 0415 design:

- 3 theory-to-practice profile-specialization experiments (normal means, cheap layer)
- 4 full model benchmark experiments (grouped linear/logistic, heterogeneity, ablation)
- 2 prior-guidance cheap experiments (tau calibration, Beta prior sensitivity)

Methods included in the benchmark layer:

- `GR_RHS`
- `RHS`
- `GIGG_MMLE`
- `GHS_plus`

No pure HS, no GRASP, no OLS/Lasso as headline benchmark methods in this pipeline.

---

## 1. Directory Layout

```text
simulation_project/
  config/
    global_config.yaml
    methods.yaml
    experiments.yaml
  src/
    dgp_normal_means.py
    dgp_grouped_linear.py
    dgp_grouped_logistic.py
    fit_gr_rhs.py
    fit_rhs.py
    fit_gigg.py
    fit_ghs_plus.py
    metrics.py
    plotting.py
    utils.py
    run_experiment.py
  results/
    exp1_null_contraction/
    exp2_adaptive_localization/
    exp3_phase_diagram/
    exp4_benchmark_linear/
    exp5_heterogeneity/
    exp6_grouped_logistic/
    exp7_ablation/
    exp8_tau_calibration/
    exp9_beta_prior_sensitivity/
  figures/
  tables/
  logs/
```

---

## 2. Reproducibility & Seeds

Global seed:

- `MASTER_SEED = 20260415`

Per experiment/setting/replicate seed:

- `seed = MASTER_SEED + 100000 * experiment_id + 1000 * setting_id + replicate_id`

All methods in the same replicate share the same generated dataset.

---

## 3. Sampling Budget and Convergence Gate

Default MCMC budget:

- `chains = 2`
- `warmup = 500`
- `post_warmup_draws = 500`
- `adapt_delta = 0.95`
- `max_treedepth = 12`

Auto retry policy:

- if divergence ratio `>= 0.5%`, rerun with
- `adapt_delta = 0.99`, `max_treedepth = 14`

Convergence inclusion gate:

- `Rhat < 1.01`
- `Bulk ESS > 400`
- divergence ratio `< 0.5%`

Failed fits are logged and kept missing (no dataset replacement).

---

## 4. Parallel Execution (Now Process-Based)

`n_jobs` is active and runs replicate-level parallelism using process pools.

- `n_jobs=1`: sequential
- `n_jobs>1`: `ProcessPoolExecutor` worker fan-out

Implementation note:

- worker functions are module-top-level (Windows-safe pickling)
- computationally heavy experiments (`exp4/5/6/7/9`) now run with true multi-process task distribution

---

## 5. Strong-Signal Threshold Interface (`exp3`)

The phase-diagram experiment exposes:

- `theta_u0_rho(u0, rho)`
- `xi_crit_u0_rho(u0, rho) = theta_u0_rho / 2`

Current implementation (0415 paper eq. 65, evaluated at `kappa = u0`):

- `theta_u0_rho(u0, rho) = u0 * rho^2 / (u0 + (1 - u0) * rho^2)`
- `xi_crit_u0_rho(u0, rho) = theta_u0_rho(u0, rho) / 2`

where `rho = tau / sqrt(sigma2)`.

Metadata output:

- `results/exp3_phase_diagram/phase_threshold_meta.json`

---

## 6. CLI Usage

Run all 9 experiments:

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2
```

Run a single experiment:

```bash
python -m simulation_project.src.run_experiment --experiment 4 --save-dir simulation_project --repeats 50 --n-jobs 2
```

Select by id:

- `1` -> null contraction
- `2` -> adaptive localization
- `3` -> phase diagram
- `4` -> benchmark linear
- `5` -> heterogeneity
- `6` -> grouped logistic
- `7` -> ablation
- `8` -> tau calibration
- `9` -> beta prior sensitivity

---

## 7. Rough Runtime (Laptop-Level, Approximate)

Runtime depends strongly on CPU cores, RAM, and whether HMC retries are triggered. The table below is a practical rough guide for defaults in this project.

Assumption:

- modern laptop CPU (6-12 logical cores)
- Python env warm already (no first-time heavy JIT delay)
- `n_jobs=2` for heavier experiments

Approximate duration:

- Exp1 (`repeats=500`): ~1 to 4 min
- Exp2 (`repeats=500`): ~1 to 5 min
- Exp3 (`repeats=200`): ~2 to 8 min
- Exp4 (`repeats=50`): ~3 to 10 h
- Exp5 (`repeats=100`): ~2 to 7 h
- Exp6 (`repeats=50`): ~1 to 4 h
- Exp7 (`repeats=50`): ~2 to 8 h
- Exp8 (`repeats=5000`): < 1 min
- Exp9 (`repeats=30`): ~1 to 4 h

All 9 experiments together (default repeats):

- roughly ~10 to 35 h (`n_jobs=2`, laptop-level estimate)

If you need a faster pilot run, reduce `repeats` first for Exp4/5/6/7/9.

---

## 8. Expected Outputs

Each experiment writes:

- `raw_results.csv`
- `summary.csv` (or split summaries where appropriate)
- figures into `simulation_project/figures/`
- logs into `simulation_project/logs/`

Publication-target filenames currently generated include:

- `fig1_null_contraction.png`
- `fig2_adaptive_localization.png`
- `fig3_phase_heatmap.png`
- `fig3_phase_curves.png`
- `fig4_benchmark_overall_mse.png`
- `fig5_kappa_stratification.png`
- `fig5_null_signal_mse.png`
- `fig5_group_ranking.png`
- `fig6_logistic_coefficients.png`
- `fig6_logistic_null_group.png`
- `fig6_logistic_diagnostics.png`
- `fig6_kappa_logistic.png`
- `fig7_tau_calibration.png`

Tables:

- `table_benchmark_linear.csv`
- `table_heterogeneity_auroc.csv`
- `table_ablation.csv`
- `table_beta_prior_sensitivity.csv`

---

## 9. Experiment API

All experiment functions are in:

- `simulation_project/src/run_experiment.py`

and share signature:

- `run_expX(..., n_jobs, seed, repeats, save_dir)`

Functions:

- `run_exp1_null_contraction`
- `run_exp2_adaptive_localization`
- `run_exp3_phase_diagram`
- `run_exp4_benchmark_linear`
- `run_exp5_heterogeneity`
- `run_exp6_grouped_logistic`
- `run_exp7_ablation`
- `run_exp8_tau_calibration`
- `run_exp9_beta_prior_sensitivity`

---

## 10. Relationship to Main Repo Runner

This pipeline is intentionally standalone and is now the primary simulation framework in this repository:

- it follows the 0415 paper-facing simulation spec directly
- it uses its own fixed output layout under `simulation_project/`
- it is optimized for transparent reproduction of the 9-study manuscript block

If you need the broader legacy benchmark stack, use the root-level CLI and configs documented in the repository root README.

Detailed protocol companion:

- [docs/simulation_project_0415_protocol.md](D:/FilesP/GR-RHS/docs/simulation_project_0415_protocol.md)
