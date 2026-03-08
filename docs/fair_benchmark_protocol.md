# Fair Benchmark Protocol

This note captures the fairness rules baked into the current regression-focused configs so the Exp1 and Exp4 sweeps can be cited directly in the paper and reproduced from the repo.

## 0. Shared fairness contract
- **Standardization** - every run inherits `configs/base.yaml`, which enforces mean-zero `y` and unit-variance columns for `X`.
- **Identical splits** - `configs/base.yaml` also pins nested CV to outer 5-fold (with optional repeats) and inner 5-fold CV. All models consume the exact same `splits` object materialized inside `outputs/.../repeat_*/fold_*`.
- **Group metadata** - a single set of contiguous groups comes from the experiment YAML (for example `configs/experiments/exp1_group_regression.yaml`), and the runner passes that same grouping to Group Lasso, Group Horseshoe, GIGG, and GRRHS.
- **Reported metrics** - regression runs always log RMSE, predictive log-likelihood, MLPD, AUC-PR, F1, Coverage90, IntervalWidth90, shrinkage diagnostics, and group-level stats (see `grrhs/metrics/evaluation.py`).
- **Convergence checks** - Bayesian models must satisfy `rhat_max <= 1.05` on the monitored blocks (`beta`, `tau`, `phi`, `lambda`) before a fold is deemed valid. The runner writes `convergence.json` per fold, automatically retries once with a larger budget when the threshold is missed, and excludes persistently invalid folds from aggregated headline metrics.

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

## 2. Bayesian grouped-sparsity models

All Bayesian grouped-sparsity models now share:
- Expected signal size `s = 20` (`model.tau.p0.value`) where the config exposes a calibrated `tau0` path.
- Automatic tau heuristic: `_maybe_calibrate_tau` turns `s` into `tau0 = (s / (p - s)) / sqrt(n)` when `standardization.X = unit_variance`.
- No outer-loop hyperparameter search in the benchmark sweeps.
- The same split objects, preprocessing, metric extraction, and convergence filtering as the convex baselines.

### Paper-faithful configuration provenance

| Model | Config | Source rationale | What is fixed vs learned | Why this is the fair benchmark path |
| --- | --- | --- | --- | --- |
| `RHS` | `configs/methods/regularized_horseshoe.yaml` | Piironen and Vehtari recommend calibrating global shrinkage from expected sparsity and using a finite regularized slab instead of the pure horseshoe tail. | Fixed: `p0 = 20`, `slab_scale = 2.0`, `slab_df = 4.0`, `sigma_scale = 1.0`, NUTS budget. Learned: `tau`, `lambda_j`, `sigma`, and slab scale `c` through `c^2 ~ Inv-Gamma(nu/2, nu s^2 / 2)`. | Matches the paper's operating regime: calibrated sparsity prior plus sampled regularized slab, with no extra outer-loop tuning. |
| `GIGG` | `configs/methods/gigg.yaml` | Boss et al. recommend fixing `a_g` close to zero and focusing estimation on `b_g`, preferably with MMLE; in practice they use `a_g = 1/n` and estimate `b_g`. | Fixed: `a_g = 1/n` implicitly, `b_init = 1.0`, `b_floor = 0.001`, `b_max = 4.0`, Gibbs budget. Learned: group-specific `b_g` by MMLE, plus posterior `tau`, `gamma_g`, `lambda_gj`, `sigma`. | Uses the authors' recommended empirical-Bayes path instead of suppressing GIGG's built-in adaptive mechanism or adding ad hoc external search. |
| `Group Horseshoe` | `configs/methods/group_horseshoe.yaml` | This baseline is the grouped horseshoe-style comparator in the repository, run with grouped shrinkage, shared calibrated `tau0`, and NUTS inference without outer tuning. | Fixed: `p0 = 20`, `group_scale = 1.0`, `sigma_scale = 1.0`, NUTS budget. Learned: `tau`, group-level shrinkage scales, `beta`, `sigma`. | Fair because it is evaluated in its native grouped-Bayesian regime with the same preprocessing and convergence rules as other Bayesian baselines. |
| `GRRHS` | `configs/methods/grrhs_regression.yaml` | Our method is meant to be judged primarily through its prior structure, not through dataset-specific search. The main-paper path is fixed defaults plus separate prior sensitivity analysis. | Fixed: `c = 1.0`, `eta = 0.5`, `p0 = 20`, Gibbs budget. Learned: posterior `tau`, `phi_g`, `lambda_j`, `sigma`. Separate sensitivity sweep varies only `p0`, `eta`, `c`. | Fair because the main table uses one predeclared default configuration, while prior sensitivity is reported separately rather than folded back into the benchmark leaderboard. |

Interpretation rule for the paper: fairness means each Bayesian method is run in the way its own methodology recommends, under a shared experimental protocol. It does not mean forcing all Bayesian methods into the same hyperparameter treatment.

### Regularized Horseshoe (RHS)
- Config: `configs/methods/regularized_horseshoe.yaml`.
- Relies on the Piironen-Vehtari calibrated `tau0` heuristic, the sampled regularized slab `c^2 ~ Inv-Gamma(nu/2, nu s^2 / 2)` with `s = 2.0, nu = 4.0`, and the shared Half-Cauchy noise prior (`sigma_scale = 1.0`).

### GIGG
- Config: `configs/methods/gigg.yaml`.
- Uses the Boss et al. Bayesian Analysis (2024) recommendation: `a_g = 1/n` fixed implicitly (`a_value: null`) and group-specific `b_g` estimated by MMLE via `mmle_update: "paper_lambda_only"`, with clipping to `[0.001, 4.0]` for numerical stability.

### Group Horseshoe (GH)
- Config: `configs/methods/group_horseshoe.yaml`.
- Shares the same `s = 20` prior via `model.tau`, uses 2k warmup + 2k posterior draws (1 chain, thinning 1, `target_accept = 0.9`), and uses the same groups as GRRHS.

### GRRHS (our model)
- Config: `configs/methods/grrhs_regression.yaml` (`c = 1.0`, `eta = 0.5`, `tau.p0.value = 20`).
- Gibbs sampler runs 8k iterations (4k burn-in) with identical jitter and seed conventions per fold.
- Prior sensitivity is not folded into the main benchmark. It is reported separately via `configs/sweeps/real_nhanes_2003_2004_grrhs_sensitivity.yaml`.

_Classification-oriented configs (Exp2/Exp3 and the logistic method presets) were removed to keep the repository focused on regression. Recover them from git history if you need them again._

## 3. Practical checklist
1. Start from `configs/experiments/exp1_group_regression.yaml` to guarantee identical data generation and group layout.
2. Run the canonical main-benchmark sweeps in `configs/sweeps/exp1_methods.yaml`, `configs/sweeps/sim_s1.yaml`, `configs/sweeps/sim_s2.yaml`, `configs/sweeps/sim_s3.yaml`, and `configs/sweeps/real_nhanes_2003_2004_ggt_methods.yaml`; they now enumerate the full roster `GRRHS / RHS / GIGG / Group Horseshoe / Group Lasso / SGL / Lasso / Ridge` wherever the dataset supports them.
3. Synthetic sweeps now fix `experiments.repeats = 3` to balance runtime with variance reduction; because the runner still shares the exact same splits per repeat/fold, reducing the count preserves fairness.
4. After each sweep, inspect `outputs/sweeps/.../fold_*/convergence.json`; the runner retries once automatically when `R-hat > 1.05`, and any fold that still fails is flagged as `INVALID_CONVERGENCE` and excluded from aggregate benchmark summaries.
5. Use `scripts/plot_diagnostics.py` or the notebook templates to overlay RMSE, log-likelihood, and shrinkage diagnostics; because splits and preprocessing are shared, the comparisons are apples-to-apples.
