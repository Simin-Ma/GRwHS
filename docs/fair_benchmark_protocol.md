# Fair Benchmark Protocol

This note captures the fairness rules baked into the current regression-focused configs so the Exp1 and Exp4 sweeps can be cited directly in the paper and reproduced from the repo.

## 0. Shared fairness contract
- **Standardization** - every run inherits `configs/base.yaml`, which enforces mean-zero `y` and unit-variance columns for `X`.
- **Identical splits** - `configs/base.yaml` also pins nested CV to outer 5-fold (with optional repeats) and inner 5-fold CV. All models consume the exact same `splits` object materialized inside `outputs/.../repeat_*/fold_*`.
- **Group metadata** - a single set of contiguous groups comes from the experiment YAML (e.g. `configs/experiments/exp1_group_regression.yaml`), and the runner passes that same grouping to Group Lasso, Group Horseshoe, and GRwHS.
- **Reported metrics** - regression runs always log RMSE, predictive log-likelihood, MLPD, AUC-PR, F1, Coverage90, IntervalWidth90, shrinkage diagnostics, and group-level stats (see `grwhs/metrics/evaluation.py`).
- **Convergence checks** - Bayesian models must satisfy `rhat_max <= 1.05` before a fold is deemed valid. The runner writes `convergence.json` per fold so we can audit RHS/GH/GRwHS side by side.

## 1. Frequentist baselines (nested CV)

### Ridge
- Config: `configs/methods/ridge.yaml`.
- Hyper-parameter: L2 penalty `alpha` in `{1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.05, 0.1, 0.5, 1, 5, 10}`.
- Procedure: for each outer fold we run 5-fold inner CV on the training split using MSE, select the best `alpha`, refit on the full outer-train, then assess on outer-test.

### Group Lasso
- Config: `configs/methods/group_lasso.yaml`.
- Uses `group_weight_mode: "size"` so large and small groups are treated symmetrically.
- Hyper-parameter: grouped penalty `alpha` in `{1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.05, 0.1, 0.5, 1}`.
- Inner-CV schedule matches Ridge (5x1 nested CV inside each outer-train split).

## 2. Horseshoe-family baselines

All horseshoe-like models now share:
- Expected signal size `s = 30` (`model.tau.p0.value`) which stays constant across Exp1/Exp4.
- Automatic tau heuristic: `_maybe_calibrate_tau` turns `s` into `tau0 = (s / (p - s)) / sqrt(n)` when `standardization.X = unit_variance`.
- Common slab width `c = 1.5` (regularized HS and GRwHS).
- Identical NUTS settings for RHS and Group HS (2k warmup + 2k posterior draws, 1 chain, thinning 1, `target_accept = 0.9`).

### Regularized Horseshoe (RHS)
- Config: `configs/methods/regularized_horseshoe.yaml`.
- Relies on the calibrated `tau0`, slab scale 1.5, and the shared Half-Cauchy noise prior (`sigma_scale = 1.0`).

### Group Horseshoe (GH)
- Config: `configs/methods/group_horseshoe.yaml`.
- Shares the same `s = 30` prior via `model.tau`, the same NUTS budget, and uses the same groups as GRwHS.

### GRwHS (our model)
- Config: `configs/methods/grwhs_regression.yaml` (`c = 1.5`, `eta = 0.5`, `tau.p0.value = 30`).
- Gibbs sampler runs 20k iterations (10k burn-in) with identical jitter/seed per fold.

_Classification-oriented configs (Exp2/Exp3 and the logistic method presets) were removed to keep the repository focused on regression. Recover them from git history if you need them again._

## 3. Practical checklist
1. Start from `configs/experiments/exp1_group_regression.yaml` to guarantee identical data generation and group layout.
2. Run the sweeps defined in `configs/sweeps/exp1_methods.yaml` and `configs/sweeps/exp4_*.yaml`; they already enumerate Ridge, Group Lasso, RHS, GH, and GRwHS.
3. Synthetic sweeps now fix `experiments.repeats = 3` to balance runtime with variance reduction; because the runner still shares the exact same splits per repeat/fold, reducing the count preserves fairness (bump it back up if you need tighter CIs).
4. After each sweep, inspect `outputs/sweeps/.../fold_*/convergence.json`; any R-hat outside `[1, 1.05]` triggers a rerun with more samples.
5. Use `scripts/plot_diagnostics.py` or the notebook templates to overlay RMSE/log-likelihood and group-selection metrics; because splits and preprocessing are shared, the comparisons are apples-to-apples.
