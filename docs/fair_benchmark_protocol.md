# Fair Benchmark Protocol

This note documents the fairness contract implemented by the current regression benchmark stack and clarifies where comparisons are strict, approximate, or model-specific.

## 0. Shared contract
- **Standardization**: every run inherits [configs/base.yaml](D:/FilesP/GR-RHS/configs/base.yaml), which centers `y` and scales `X` to unit variance by default.
- **Identical splits**: all methods consume the same outer/inner split seeds and the same `OuterFold` objects materialized by the runner.
- **Shared data generation inside sweeps**: `run_sweep` pins one common `data.seed` / `seeds.data_generation` across all variations in the same sweep.
- **Shared group metadata**: grouped methods read the same contiguous group layout from the experiment config.
- **Nested CV scope**: inner CV is only used for methods that declare `model.search`. In the main paper-style sweeps that means the frequentist baselines; Bayesian grouped-sparsity models run with fixed, predeclared defaults.

## 1. Metric fairness

### Point-prediction metrics
- `RMSE` and `MAE` are directly comparable across all methods.
- Selection and coefficient-recovery metrics are comparable whenever synthetic truth is available.

### Predictive density metrics
- The default benchmark mode is now `experiments.evaluation.predictive_density_mode: "strict"`.
- In `strict` mode, `MLPD` / `PredictiveLogLikelihood` are only reported when the model exposes posterior predictive log-likelihood draws.
- Deterministic baselines therefore return `MLPD_source: "disabled"` instead of a pseudo-likelihood.
- If you explicitly switch to `predictive_density_mode: "mixed"`, deterministic baselines may use a Gaussian proxy; those folds are marked with `MLPD_source: "gaussian_proxy"`.

Paper guidance: use `RMSE` as the main predictive ranking metric unless every compared model reports predictive densities under the same likelihood construction.

## 2. Hyper-parameter fairness

### Frequentist baselines
- Ridge, Lasso, Group Lasso, and Sparse Group Lasso use nested CV on the outer-train split only.
- The selected hyper-parameters are refit on the full outer-train fold before testing.

Current canonical grids:
- Ridge: `alpha ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000}` from [configs/methods/ridge.yaml](D:/FilesP/GR-RHS/configs/methods/ridge.yaml)
- Group Lasso: `alpha ∈ {1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1}` from [configs/methods/group_lasso.yaml](D:/FilesP/GR-RHS/configs/methods/group_lasso.yaml)

### Bayesian grouped-sparsity models
- Main-benchmark sweeps do not run outer-loop hyper-parameter search.
- Fairness is defined as **paper-faithful defaults under a shared data/split/evaluation protocol**, not “force every Bayesian method into the same search routine”.
- GRRHS prior sensitivity remains separated from the main leaderboard.

## 3. Convergence fairness

### Model-specific monitored blocks
- GRRHS: `beta`, `tau`, `phi`, `lambda`
- GIGG: `beta`, `tau`, `gamma`, `lambda`
- Regularized Horseshoe / Horseshoe: `beta`, `tau`, `lambda`
- Group Horseshoe: `beta`, `tau`, `group_lambda`

### Decision rule
- The runner retries once with a larger budget when the configured `max_rhat` threshold is missed.
- Folds that still fail are marked `INVALID_CONVERGENCE` and excluded from aggregate metric summaries.
- Missing monitored blocks are recorded in `convergence_attempts`. The default policy is `missing_policy: "warn"` rather than immediate failure.

### Diagnostic validity
- `convergence.json` now records `raw_num_chains`, `raw_num_draws`, and `diagnostic_valid`.
- `diagnostic_valid=false` means the summary was computed from a chain layout that is weaker than the requested multi-chain standard.
- Current defaults keep `require_valid_diagnostics: false`, so single-chain Gibbs runs are retained but explicitly flagged as heuristic diagnostics rather than full multi-chain evidence.
- Multi-chain NUTS baselines now retain their chain axis in saved posterior arrays, so `raw_num_chains` reflects the configured chain count for models such as Group Horseshoe and RHS.

Paper guidance: report convergence with model-specific shrinkage blocks, and separately disclose whether diagnostics were multi-chain-valid or split-chain heuristics.

## 4. Seed robustness and runtime robustness

Main benchmark:
- Shared data-generation seed within each sweep
- Fixed split seeds from [configs/base.yaml](D:/FilesP/GR-RHS/configs/base.yaml)
- Fixed model defaults for Bayesian methods

Additional audit sweeps:
- [configs/sweeps/audit_seed_stability_bayesian_methods.yaml](D:/FilesP/GR-RHS/configs/sweeps/audit_seed_stability_bayesian_methods.yaml): re-runs GRRHS, GIGG, Group Horseshoe, and RHS with matched per-method seed perturbations on the same benchmark task
- [configs/sweeps/audit_budget_sensitivity_bayesian_methods.yaml](D:/FilesP/GR-RHS/configs/sweeps/audit_budget_sensitivity_bayesian_methods.yaml): compares 1x / 2x / 4x runtime budgets for the same Bayesian roster under method-appropriate budget knobs
- Legacy GRRHS-only audits remain available at [configs/sweeps/audit_seed_stability.yaml](D:/FilesP/GR-RHS/configs/sweeps/audit_seed_stability.yaml) and [configs/sweeps/audit_budget_sensitivity.yaml](D:/FilesP/GR-RHS/configs/sweeps/audit_budget_sensitivity.yaml)

These audit sweeps are not part of the headline benchmark table; they are intended to support robustness claims.

## 5. Practical checklist
1. Use [configs/experiments/exp1_group_regression.yaml](D:/FilesP/GR-RHS/configs/experiments/exp1_group_regression.yaml) or the NHANES experiment configs as the shared task definition.
2. Run the canonical benchmark sweeps from [configs/sweeps/exp1_methods.yaml](D:/FilesP/GR-RHS/configs/sweeps/exp1_methods.yaml) and [configs/sweeps/real_nhanes_2003_2004_ggt_methods.yaml](D:/FilesP/GR-RHS/configs/sweeps/real_nhanes_2003_2004_ggt_methods.yaml).
3. Inspect `repeat_*/fold_*/convergence.json` together with `fold_summary.json`, not just aggregate metrics.
4. Use the audit sweeps before making claims about seed robustness or “fully converged” runtime stability.
5. Treat `MLPD` as a strict-comparison metric only when every compared model reports the same density source.
