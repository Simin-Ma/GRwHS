# Fair Benchmark Protocol

This note documents the fairness contract implemented by the current regression benchmark stack and clarifies where comparisons are strict, approximate, or model-specific.

## 0. Shared contract
- **Standardization**: every run inherits [configs/base.yaml](D:/FilesP/GR-RHS/configs/base.yaml), which centers `y` and scales `X` to unit variance by default.
- **Identical splits**: all methods consume the same outer/inner split seeds and the same `OuterFold` objects materialized by the runner.
- **Shared data generation inside sweeps**: `run_sweep` pins one common `data.seed` / `seeds.data_generation` across all variations in the same sweep.
- **Shared group metadata**: grouped methods read the same contiguous group layout from the experiment config.
- **Nested CV scope**: inner CV is only used for methods that declare `model.search`. In the main paper-style sweeps that means the frequentist baselines; Bayesian grouped-sparsity models run with fixed, predeclared defaults.
- **Bayesian fairness guardrail**: for `RHS`, `GR-RHS`, and `GIGG`, `experiments.bayesian_fairness` disables Bayesian inner CV, enforces one shared posterior budget, requires posterior-mean summaries, and disables budget-escalation retries in the headline benchmark.

## 1. Metric fairness

### Point-prediction metrics
- `RMSE` and `MAE` are directly comparable across all methods.
- Selection and coefficient-recovery metrics are comparable whenever synthetic truth is available.
- Synthetic selection metrics use one shared ranking signal across methods: absolute fitted coefficient magnitude
  (posterior mean for Bayesian models, point estimate for deterministic models).

### Predictive density metrics
- The default benchmark mode is now `experiments.evaluation.predictive_density_mode: "strict"`.
- In `strict` mode, `MLPD` / `PredictiveLogLikelihood` are only reported when the evaluator can form exact held-out log-likelihood draws under the model's own likelihood
  (either from explicit `loglik_samples_` or from posterior parameter draws that determine that likelihood exactly).
- Deterministic baselines therefore return `MLPD_source: "disabled"` instead of a pseudo-likelihood.
- Regression density metrics are computed from `log p(y_test | theta)` under posterior draws, not by adding predictive noise first and scoring that noisy draw a second time.
- If you explicitly switch to `predictive_density_mode: "mixed"`, deterministic baselines may use a Gaussian proxy; those folds are marked with `MLPD_source: "gaussian_proxy"`.

Paper guidance: use `RMSE` as the main predictive ranking metric unless every compared model reports predictive densities under the same likelihood construction.

## 2. Hyper-parameter fairness

### Frequentist baselines
- Ridge, Lasso, and Sparse Group Lasso use nested CV on the outer-train split only.
- The selected hyper-parameters are refit on the full outer-train fold before testing.

Current canonical grids:
- Ridge: `alpha ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000}` from [configs/methods/ridge.yaml](D:/FilesP/GR-RHS/configs/methods/ridge.yaml)

### Bayesian grouped-sparsity models
- Main-benchmark sweeps do not run outer-loop hyper-parameter search.
- For the main leaderboard, fairness is grounded in the shared task definition, shared splits, shared evaluation rules,
  and declared compute budget for each method family rather than forcing Bayesian and frequentist methods through one identical tuning loop.
- Fairness is defined as **paper-faithful defaults under a shared data/split/evaluation protocol**, not “force every Bayesian method into the same search routine”.
- GRRHS prior sensitivity remains separated from the main leaderboard.
- The default shared Bayesian sampling budget is `burn_in=1000`, `kept_draws=1000`, `thinning=1`, `num_chains=1`.

## 3. Convergence fairness

### Model-specific monitored blocks
- GRRHS: `beta`, `tau`, `phi`, `lambda`
- GIGG: `beta`, `tau`, `gamma`, `lambda`
- Regularized Horseshoe / Horseshoe: `beta`, `tau`, `lambda`

### Decision rule
- Outside the fairness guardrail, the runner retries once with a larger budget when the configured `max_rhat` threshold is missed.
- Inside the fairness guardrail, retries are disabled so no Bayesian method silently receives extra posterior budget.
- Folds that still fail are marked `INVALID_CONVERGENCE` and excluded from aggregate metric summaries.
- Missing monitored blocks are recorded in `convergence_attempts`. The default policy is `missing_policy: "warn"` rather than immediate failure.
- Sweep comparison artifacts re-aggregate metrics on the intersection of valid outer folds across compared runs so
  models are ranked on the same realized fold set.

### Diagnostic validity
- `convergence.json` now records `raw_num_chains`, `raw_num_draws`, and `diagnostic_valid`.
- `diagnostic_valid=false` means the summary was computed from a chain layout that is weaker than the requested multi-chain standard.
- Current defaults require `require_valid_diagnostics: true`, so single-chain Gibbs runs no longer enter the headline benchmark unless that guardrail is explicitly relaxed.
- Multi-chain NUTS baselines now retain their chain axis in saved posterior arrays, so `raw_num_chains` reflects the configured chain count for models such as RHS.

Paper guidance: report convergence with model-specific shrinkage blocks, and keep headline comparisons restricted to runs with multi-chain-valid diagnostics.

## 4. Seed robustness and runtime robustness

Main benchmark:
- Shared data-generation seed within each sweep
- Fixed split seeds from [configs/base.yaml](D:/FilesP/GR-RHS/configs/base.yaml)
- Fixed model defaults for Bayesian methods

Additional audit sweeps:
- [configs/sweeps/audit_seed_stability_bayesian_methods.yaml](D:/FilesP/GR-RHS/configs/sweeps/audit_seed_stability_bayesian_methods.yaml): re-runs GRRHS, GIGG, and RHS with matched per-method seed perturbations on the same benchmark task
- [configs/sweeps/audit_budget_sensitivity_bayesian_methods.yaml](D:/FilesP/GR-RHS/configs/sweeps/audit_budget_sensitivity_bayesian_methods.yaml): compares 1x / 2x / 4x runtime budgets for the same Bayesian roster under method-appropriate budget knobs
- Legacy GRRHS-only audits remain available at [configs/sweeps/audit_seed_stability.yaml](D:/FilesP/GR-RHS/configs/sweeps/audit_seed_stability.yaml) and [configs/sweeps/audit_budget_sensitivity.yaml](D:/FilesP/GR-RHS/configs/sweeps/audit_budget_sensitivity.yaml)

These audit sweeps are not part of the headline benchmark table; they are intended to support robustness claims.

## 5. Practical checklist
1. Use [configs/experiments/exp1_group_regression.yaml](D:/FilesP/GR-RHS/configs/experiments/exp1_group_regression.yaml) or the NHANES experiment configs as the shared task definition.
2. Run the canonical benchmark sweeps from [configs/sweeps/exp1_methods.yaml](D:/FilesP/GR-RHS/configs/sweeps/exp1_methods.yaml) and [configs/sweeps/real_nhanes_2003_2004_ggt_methods.yaml](D:/FilesP/GR-RHS/configs/sweeps/real_nhanes_2003_2004_ggt_methods.yaml).
3. Inspect `repeat_*/fold_*/convergence.json` together with `fold_summary.json`, not just aggregate metrics.
4. Use the audit sweeps before making claims about seed robustness or “fully converged” runtime stability.
5. Treat `MLPD` as a strict-comparison metric only when every compared model reports the same density source.
