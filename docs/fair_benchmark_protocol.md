# Fair Benchmark Protocol

This note captures the “fair comparison” rules baked into the current configs so the Exp1/Exp2 sweeps can be cited directly in the paper and reproduced from the repo.

## 0. Shared fairness contract
- **Standardization** – every run inherits `configs/base.yaml`, which enforces mean-zero `y` and unit-variance columns for `X`.
- **Identical splits** – `configs/base.yaml` also pins nested CV to outer 5-fold (with optional repeats) and inner 5-fold CV. All models consume the exact same `splits` object materialized inside `outputs/.../repeat_*/fold_*`.
- **Group metadata** – a single set of contiguous groups comes from the experiment YAML (e.g. `configs/experiments/exp1_group_regression.yaml`), and the runner passes that same grouping to Group Lasso, Group Horseshoe, and GRwHS.
- **Reported metrics** – regression runs always log RMSE, predictive log-likelihood, MLPD, AUC-PR, F1, Coverage90, IntervalWidth90, shrinkage diagnostics, and group-level stats (see `grwhs/metrics/evaluation.py`).
- **Convergence checks** – Bayesian models must satisfy `rhat_max <= 1.05` (or 1.10 in the rare logistic corner cases) before a fold is deemed valid. The runner writes `convergence.json` per fold so we can audit RHS/GH/GRwHS side by side.

## 1. Frequentist baselines (nested CV)

### Ridge
- Config: `configs/methods/ridge.yaml`.
- Hyper-parameter: L2 penalty `alpha ∈ {1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.05, 0.1, 0.5, 1, 5, 10}`.
- Procedure: for each outer fold we run 5-fold inner CV on the training split using MSE/log-loss, select the best `alpha`, refit on the full outer-train, then assess on outer-test.

### Group Lasso
- Config: `configs/methods/group_lasso.yaml`.
- Uses `group_weight_mode: "size"` so large and small groups are treated symmetrically.
- Hyper-parameter: grouped penalty `alpha ∈ {1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.05, 0.1, 0.5, 1}`.
- Inner-CV schedule matches Ridge (5×1 nested CV inside each outer-train split).

## 2. Horseshoe-family baselines

All horseshoe-like models now share:
- Expected signal size `s = 30` (`model.tau.p0.value`) which stays constant across Exp1/Exp2.
- Automatic τ heuristic: `_maybe_calibrate_tau` turns `s` into `tau0 = (s / (p - s)) / sqrt(n)` when `standardization.X = unit_variance`.
- Common slab width `c = 1.5` (regularized HS and GRwHS).
- Identical NUTS settings for RHS and Group HS (2 000 warmup + 2 000 posterior draws, 1 chain, thinning 1, `target_accept = 0.9`).

### Regularized Horseshoe (RHS)
- Regression config: `configs/methods/regularized_horseshoe.yaml`.
- Classification config: `configs/methods/regularized_horseshoe_logistic.yaml`.
- Both rely on the calibrated `tau0`, slab scale 1.5, and the shared Half-Cauchy noise prior (`sigma_scale = 1.0`).

### Group Horseshoe (GH)
- Regression config: `configs/methods/group_horseshoe.yaml`.
- Classification config: `configs/methods/group_horseshoe_logistic.yaml`.
- Shares the same `s = 30` prior via `model.tau`, same NUTS budget, and uses the same groups as GRwHS.

### GRwHS (our model)
- Regression config: `configs/methods/grwhs_regression.yaml` (`c = 1.5`, `eta = 0.5`, `tau.p0.value = 30`).
- Logistic config: `configs/methods/grwhs_logistic.yaml` (same hyper-parameters, classification noise proxy `sigma_classification = 2.0`).
- Gibbs sampler runs 20k iterations (10k burn-in) with identical jitter/seed per fold.

## 3. Practical checklist
1. Start from `configs/experiments/exp1_group_regression.yaml` (or the logistic variant) to guarantee identical data generation and group layout.
2. Run the sweep defined in `configs/sweeps/exp1_methods.yaml`; it already enumerates Ridge, Group Lasso, RHS, GH, and GRwHS.
3. After each sweep, inspect `outputs/sweeps/.../fold_*/convergence.json`; any R-hat outside [1, 1.05] triggers a rerun with more samples.
4. Use `scripts/plot_diagnostics.py` or the notebook templates to overlay RMSE/log-likelihood and group-selection metrics; because splits and preprocessing are shared, the comparisons are apples-to-apples.
