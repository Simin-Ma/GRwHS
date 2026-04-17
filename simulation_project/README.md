# Simulation Project (0415 Spec Implementation)

This folder contains the standalone 9-experiment simulation pipeline built from the 0415 design:

- 3 theory-to-practice profile-specialization experiments (normal means, cheap layer)
- 4 full model benchmark experiments (grouped linear/logistic, heterogeneity, ablation)
- 2 prior-guidance cheap experiments (tau calibration, Beta prior sensitivity)

Methods included in the benchmark layer:

- `GR_RHS`
- `RHS`
- `GIGG_MMLE`
- `GIGG_b_small`
- `GIGG_GHS`
- `GIGG_b_large`
- `GHS_plus`
- `OLS`
- `LASSO_CV`

`laptop` profile default method set is a cheaper subset:

- `GR_RHS`, `RHS`, `GIGG_MMLE`, `GHS_plus`, `OLS`, `LASSO_CV`

No pure HS and no GRASP as headline Bayesian methods in this pipeline.

Recent protocol updates:

- `exp4` supports SNR sweeps via `snr_list` and includes extra settings for unequal-group distributed signals (`L6`) and stronger cross-group coupling (`L4B20`).
- `exp6` enforces `min_separator_auc` filtering by default (`0.8`).
- `exp7` removes duplicated RHS-equivalent ablation labels.
- `exp8` uses mixed strong/weak active coefficients (`2.0` / `0.5`) for tau-calibration robustness.
- `exp8` now reuses the same synthetic dataset across tau modes within each `(p0, replicate)` block for paired comparison and lower compute overhead.
- `exp9` now evaluates all `(alpha_kappa, beta_kappa)` priors on the same scenario-replicate dataset (paired prior sensitivity).
- benchmark layer supports compute profiles: `full` and `laptop` (`--profile laptop` in CLI).
- GIGG uses Bhattacharya fast beta draw (`btrick=True`) and profile-specific iteration budgets.
- Bayesian methods support convergence-enforced retries. In current CLI defaults, if convergence enforcement is on and `--max-convergence-retries` is not set, the runner uses "until converged" mode (bounded by internal safety cap, currently 12 retries, to avoid infinite loops).
- benchmark summaries now expose run-quality columns such as `n_total_runs`, `n_effective`, and `valid_rate`.
- `exp4` writes both `summary_all.csv` and `summary_converged.csv` (plus matching tables) for dual reporting.
- `exp4-9` now additionally write paired-converged summaries (common replicate intersection across compared methods/configs), and default table files prefer this paired-converged view when available.
- `exp4-9` now write convergence audits: per-method `convergence_audit.csv` and intersection-level `paired_convergence_audit.csv`.

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

Laptop profile budget (`--profile laptop`, exp8 slightly heavier than others):

- most experiments: `chains=1`, `warmup=250`, `post_warmup_draws=250`
- exp8: `chains=2`, `warmup=300`, `post_warmup_draws=300`

Auto retry policy:

- if divergence ratio `>= 0.5%`, rerun with
- `adapt_delta = 0.99`, `max_treedepth = 14`

Convergence inclusion gate:

- `Rhat < 1.01`
- `Bulk ESS > 400`
- divergence ratio `< 0.5%`

Failed fits are logged and kept missing (no dataset replacement).

---

## 4. Parallel Execution

`n_jobs` is active with mixed strategy:

- `exp1/2/3` (theory layer): setting-level thread parallelism (fast startup, low overhead)
- `exp4+` (full model layer): process/thread fallback controlled in runner

This avoids Windows startup overhead and avoids forcing `pandas` import on cheap experiments.

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

Run all with laptop profile:

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2 --profile laptop
```

Run with strict convergence enforcement and extra retry budget:

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2 --profile full --max-convergence-retries 3
```

Force "until converged" mode explicitly (auto-add budget until convergence, with internal hard cap):

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2 --profile full --until-bayes-converged
```

Disable convergence enforcement (not recommended for final reporting):

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2 --no-enforce-bayes-convergence
```

Run a single experiment:

```bash
python -m simulation_project.src.run_experiment --experiment 4 --save-dir simulation_project --repeats 50 --n-jobs 2
```

Programmatic examples for new knobs:

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

Approximate duration (very hardware-dependent):

- Exp1 (`repeats=500`): seconds to minutes
- Exp2 (`repeats=500`): seconds to minutes
- Exp3 (`repeats=200`): seconds to minutes
- Exp4 (`repeats=100`, full profile): hours
- Exp5 (`repeats=100`, full profile): hours
- Exp6 (`repeats=50`, full profile): hours
- Exp7 (`repeats=100`, full profile): hours
- Exp8 (`repeats=100`, full profile): hours
- Exp9 (`repeats=120`, full profile): hours

All 9 experiments together:

- `full` profile: roughly ~10 to 35 h (`n_jobs=2`, laptop-level estimate)
- `laptop` profile: usually much shorter (pilot-grade); final tables should still be rerun under `full`

If you need a faster pilot run, reduce `repeats` first for Exp4/5/6/7/9.

---

## 9. Pandas Import Hang (Windows) - Permanent Workaround

If your system Python has very slow/blocked `import pandas`, theory experiments still run fast because `exp1/2/3` no longer depend on pandas for core execution.

Quick diagnosis:

```bash
python scripts/diagnose_pandas_import.py
```

Output file:

- `simulation_project/logs/pandas_import_diagnosis.json`

If diagnosis shows `import_pandas` timeout:

1. Keep running `exp1/2/3` normally (they are isolated from pandas).
2. For `exp4+`, run in a clean Python environment (recommended) or fix the system Python install.
3. Before long runs, clean stale python workers:
   - PowerShell: `Get-Process | ? {$_.ProcessName -like 'python*'} | Stop-Process -Force`

---

## 8. Expected Outputs

Each experiment writes:

- `raw_results.csv`
- `summary.csv` (or split summaries where appropriate)
- quality counts (`n_total_runs`, `n_effective`, `valid_rate`) in benchmark-style summaries
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
