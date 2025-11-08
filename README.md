# GRwHS Experimentation Toolkit

Comprehensive infrastructure for benchmarking generalized regularized horseshoe (GRwHS) models across structured toy studies (regression/classification grids, overlap ablations) and real datasets. The toolkit helps you generate data, train multiple model families, evaluate metrics, track posterior convergence, and produce reproducible reports and plots.

---

## 1. Installation & Environment

### 1.1 Requirements
- Python 3.9+
- pip
- (optional) virtualenv/venv/conda for isolation

### 1.2 Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell / cmd
# or source .venv/bin/activate   # macOS / Linux

pip install --upgrade pip
pip install -e .[dev]
```
`.[dev]` installs runtime dependencies (numpy, jax, numpyro, torch, scipy, matplotlib, pandas, etc.) and development tools (pytest, ruff, pre-commit).

### 1.3 Optional Tools
- `pre-commit install` for formatting/lint hooks
- Jupyter (already included) for interactive analysis (`notebooks/`)

---

## 2. Repository Layout & Flow

### 2.1 Top-level map

| Path | Purpose / Notes |
|------|-----------------|
| `configs/` | Layered YAML stack. `base.yaml` holds global defaults; `experiments/` define datasets, `methods/` capture hyper-parameters/priors, and `sweeps/` enumerate grid searches or method comparisons. |
| `data/` | Source-of-truth for synthetic data generators (`generators.py`), preprocessing (`preprocess.py`), loaders for real datasets (`loaders.py`), and reproducible split helpers (`splits.py`). |
| `grwhs/` | Installable package that implements the CLI, runner, models, diagnostics, metrics, visualization, and utilities described in Section 2.2. |
| `scripts/` | Task-oriented automation: diagnostics/plotting (`plot_check.py`, `plot_diagnostics.py`), sweep utilities (`random_sweep_selector.py`), calibration helpers (`calibrate_logistic.py`), etc. |
| `notebooks/` | Scratch space for exploratory analysis or report-ready figures that build on saved outputs. |
| `outputs/` | Auto-generated artifacts. Single runs live under `outputs/runs/<name>-<timestamp>/`, sweeps under `outputs/sweeps/<sweep_id>/<variant>-<timestamp>/`, and aggregated exports under `outputs/reports/`. |
| `tests/` | Pytest suite covering generators, inference kernels, diagnostics, and smoke tests for the CLI pipeline. |
| `random_sweep_selector.py` | Optional entry-point that samples subsets of a sweep spec and executes them (useful for stochastic benchmarking sessions). |
| `pyproject.toml` / `.pre-commit-config.yaml` | Toolchain configuration (packaging metadata, lint/test hooks). |

### 2.2 Core package modules (`grwhs/`)

- `grwhs/cli/`
  - `run_experiment.py` merges any number of config files/overrides, stamps the resolved YAML, and kicks off the runner.
  - `run_sweep.py` iterates over `configs/sweeps/*` definitions, managing per-variant overrides and destination folders.
  - `make_report.py` aggregates finished run directories into JSON/CSV summaries in `outputs/reports/`.
- `grwhs/experiments/`
  - `runner.py` is the orchestration hub: it calls dataset generators, instantiates models, evaluates metrics, and writes artifacts.
  - `registry.py` exposes the `@register` decorator used by every model/baseline so the runner can request them by name.
  - `sweeps.py` loads sweep templates and materializes per-run configurations; `aggregator.py` consolidates fold-level results.
- `grwhs/models/`
  - Contains the Gibbs (`grwhs_gibbs.py`), SVI (`grwhs_svi_numpyro.py`), and convex baselines (lasso/ridge/GL/SGL) implementations.
  - Models rely on inference helpers (sampling routines, Woodbury solvers) and populate posterior buffers used downstream.
- `grwhs/inference/`
  - Linear algebra kernels, proximal updates, and Generalized Inverse Gaussian samplers shared by multiple models.
  - Encapsulates numerical safeguards (jitter, reparameterisations) so models stay focused on statistical logic.
- `grwhs/metrics/`
  - Regression, classification, selection, and calibration metrics consumed by the runner and by `make_report`.
- `grwhs/diagnostics/`
  - `convergence.py`, `shrinkage.py`, and `postprocess.py` compute R-hat/ESS, group shrinkage summaries, and EDF-style diagnostics.
- `grwhs/postprocess/`
  - Currently `debias.py`, which adjusts posterior draws/point-estimates before reporting when requested by configs.
- `grwhs/utils/`
  - Shared infrastructure: config parsing/validation, structured logging, filesystem helpers, and dataclass-like containers.
- `grwhs/viz/`
  - Plotting/table building blocks consumed both by CLI scripts and notebooks (scatter plots, coverage curves, LaTeX-ready tables).

### 2.3 Execution flow (config -> artifacts)

1. Compose a config stack (`configs/base.yaml` + dataset + method + optional overrides) and hand it to `grwhs.cli.run_experiment` or `grwhs.cli.run_sweep`.
2. The CLI resolves/validates the merged YAML, persists `resolved_config.yaml`, and hands control to `grwhs.experiments.runner.Runner`.
3. The runner calls `data/generators.py` & `data/preprocess.py` to build standardised folds, then acquires the requested estimator from `grwhs.experiments.registry`.
4. Models under `grwhs/models/` call into `grwhs/inference/` primitives, emit predictions/posterior draws, and register any auxiliary diagnostics.
5. Metrics from `grwhs/metrics/` and convergence summaries from `grwhs/diagnostics/` are computed before `runner` writes datasets, metrics, posterior arrays, and plots into `outputs/runs/...` (or the sweep-specific subfolder).
6. Reporting/visualisation layers (`grwhs.cli.make_report`, `scripts/plot_check.py`, `scripts/plot_diagnostics.py`, notebooks) consume those artifacts, while `tests/` assert the whole pathway stays stable.

```
configs -> grwhs.cli (run_experiment/run_sweep) -> grwhs.experiments.runner ->
registry/models/inference -> metrics + diagnostics -> outputs/(runs|sweeps|reports) -> scripts/notebooks/tests
```

---

## 3. Configuration Layers

The renewed experiment plan is organised into three composable layers: dataset descriptors (`configs/experiments`), method presets (`configs/methods`), and sweep specifications (`configs/sweeps`). Each run merges `configs/base.yaml` with one (or more) files from those directories before the nested-CV runner executes.

### 3.1 Base config (`configs/base.yaml`)

`base.yaml` encapsulates protocol-wide defaults:

- **Splits** - `splits.outer` defines the outer K-fold (with optional repeats and auto stratification); `splits.inner` sets the inner CV used for convex baselines.
- **Standardisation** - feature scaling and optional response centring (default: regressions centre `y`, classification leaves labels untouched).
- **Model** - GRwHS prior hyperparameters (`c`, `eta`, `s0`) and a `tau` block that can be `mode: calibrated` (m_eff heuristic with p0 grid) or `mode: fixed`.
- **Inference** - Gibbs sampler defaults (iterations, burn-in, jitter).
- **Metrics** - canonical regression/classification metrics evaluated on the outer test fold only.

Every other configuration inherits from this foundation.

### 3.2 Dataset descriptors (`configs/experiments/*.yaml`)

Dataset files adjust only the `data` (and occasionally `standardization` or `splits`) section:

- `toy_regression.yaml` - heterogeneous groups with controllable intra-group correlation and signal strength.
- `toy_regression_overlap.yaml` - injects 20% overlapping memberships to stress overlapping-group priors.
- `toy_classification.yaml` - logistic generator with tunable separation (`classification.scale`) and optional noise.
- `toy_classification_imbalance.yaml` / `toy_classification_separable.yaml` - variants for class imbalance and near separability.
- `real_*_template.yaml` - placeholders for plugging in actual loaders (`path_X`, `path_y`, `path_group_map`).

### 3.3 Method presets (`configs/methods/*.yaml`)

Method presets collect model-specific hyperparameters and tuning instructions:

- `grwhs_full.yaml` - Gibbs sampler with calibrated tau (p0 grid, df=2).
- `grwhs_fixed_tau.yaml` - GRwHS with tau fixed to a supplied value (for the calibration ablation).
- `grwhs_no_group.yaml` - feature-wise Regularised Horseshoe baseline (no group layer).
- `grwhs_full_logistic.yaml` / `grwhs_fixed_tau_logistic.yaml` - logistic GRwHS variants.
- `group_lasso.yaml` - skglm Group Lasso plus an alpha grid for inner-CV model selection.
- `sparse_group_lasso.yaml` - Sparse Group Lasso with joint (alpha, l1 ratio) search.

These files can be stacked (dataset + method + ablation override) by passing multiple `--config` arguments or via sweep `config_files`.

### 3.4 Sweep specs (`configs/sweeps/*.yaml`)

Sweeps combine datasets and methods into experiment suites:

- `toy_regression_methods.yaml` - GRwHS vs GL/SGL vs ablations on the base toy regression setup.
- `toy_regression_overlap_methods.yaml` - the same comparison under overlapping groups.
- `toy_regression_grid.yaml` - GRwHS full model across `{n in {100, 200}} x {p in {500, 1000}} x {rho_in in {0, 0.3, 0.6}} x {A in {2, 4, 6, 8, 10}}`.
- `toy_classification_methods.yaml` / `toy_classification_variants.yaml` - method comparisons for balanced, imbalanced, and near-separable logistic tasks.
- `toy_classification_grid.yaml` - logistic GRwHS across correlation/separation grids.

Each variation may specify extra overrides (e.g. seeds, priors) or add method files via `config_files`.

### 3.5 Running a single experiment

```bash
# GRwHS (calibrated tau) on toy regression
python -m grwhs.cli.run_experiment \
  --config configs/base.yaml \
          configs/experiments/toy_regression.yaml \
          configs/methods/grwhs_full.yaml \
  --name toy_regression_grwhs

# Sparse Group Lasso on the same dataset
python -m grwhs.cli.run_experiment \
  --config configs/base.yaml \
          configs/experiments/toy_regression.yaml \
          configs/methods/sparse_group_lasso.yaml \
  --name toy_regression_sgl
```

### 3.6 Launching a sweep

```bash
# Compare GRwHS / GL / SGL on overlapping groups
python -m grwhs.cli.run_sweep \
  --base-config configs/base.yaml \
  --sweep-config configs/sweeps/toy_regression_overlap_methods.yaml \
  --jobs 4
```

`run_sweep` merges `base.yaml`, all `common_config_files`, then each variation's `config_files` and `overrides`. Fully resolved configs are written to `<outdir>/.../resolved_config.yaml`.

### 3.7 Classification-specific notes

- Set `task: classification` in the dataset descriptor and disable response centring (`standardization.y_center: false`).
- Use the logistic GRwHS presets (`configs/methods/grwhs_full_logistic.yaml`, etc.).
- Convex baselines reuse the same wrappers; the runner maps their linear predictors through sigma(x) to obtain probabilities for log-loss/Brier/ECE.
- Outer/inner splits automatically stratify unless explicitly disabled.

---

## 4. Benchmark Workflow

Follow this checklist to reproduce the experiment suite summarised in the proposal:

1. **Fix dataset seeds** - the sweep runner now stamps a single `data.seed` / `seeds.data_generation` across all variations to guarantee identical outer folds for every method.
2. **Toy regression (structure focus)**
   - Hyper-grid for GRwHS only:
     ```bash
     python -m grwhs.cli.run_sweep \
       --base-config configs/base.yaml \
       --sweep-config configs/sweeps/toy_regression_grid.yaml \
       --jobs 4
     ```
   - Method comparison at the default setting:
     ```bash
     python -m grwhs.cli.run_sweep \
       --base-config configs/base.yaml \
       --sweep-config configs/sweeps/toy_regression_methods.yaml \
       --jobs 4
     ```
   - Overlapping groups ablation:
     ```bash
     python -m grwhs.cli.run_sweep \
       --base-config configs/base.yaml \
       --sweep-config configs/sweeps/toy_regression_overlap_methods.yaml \
       --jobs 4
     ```
3. **Toy classification (uncertainty focus)**
   - Correlation/separation grid for GRwHS logistic:
     ```bash
     python -m grwhs.cli.run_sweep \
       --base-config configs/base.yaml \
       --sweep-config configs/sweeps/toy_classification_grid.yaml \
       --jobs 4
     ```
   - Method comparison across balanced/imbalanced/separable regimes:
     ```bash
     python -m grwhs.cli.run_sweep \
       --base-config configs/base.yaml \
       --sweep-config configs/sweeps/toy_classification_variants.yaml \
       --jobs 4
     ```
4. **Real data** - copy one of the `real_*_template.yaml` files, fill in `data.loader.*` paths (and group maps, if available), then run single experiments or create a sweep analogous to the toy setups.
5. **Summaries** - aggregate results with `python -m grwhs.cli.make_report --runs <path/glob>` to produce combined CSV/JSON tables (outer-fold means and standard-errors are already stored inside each run directory).

Each run directory records fold-level metrics (`fold_*` subdirectories), resolved configuration, and full metadata so reports can cross-reference calibration statistics (`tau_summary.json`) or tuning diagnostics (`tuning_summary.json`).

## 5. Outputs & Artifacts

Each run directory (`outputs/runs/<name-timestamp>/`) contains:
- `dataset.npz`: standardized train/val/test splits, true coefficients, means/scales.
- `dataset_meta.json`: metadata (n, p, group mapping, splits, model, posterior info).
- `metrics.json`: metrics (`mse`, `r2`, `tpr`, `fpr`, `auc`, etc.) serialized to JSON-friendly types.
- `posterior_samples.npz`: posterior arrays (coefficients, tau, phi, lambda, sigma) if available.
- `convergence.json`: split R-hat & ESS summary computed from posterior arrays.
- `plots_check/`: generated plots (prediction scatter/histograms & posterior traces).
- `resolved_config.yaml`: final merged configuration for reproducibility.

---

## 6. Visualisation & Posterior Inspection

### 6.1 Plot Script
`scripts/plot_check.py` builds standard and posterior plots for any run:
```bash
python scripts/plot_check.py outputs/runs/grwhs_svi_B-<timestamp>
```
Outputs include:
- `scatter_pred_vs_truth.png`
- `residual_hist.png`
- `prediction_over_index.png`
- `coefficients_sorted.png`
- `posterior_trace_beta0.png`, `posterior_hist_beta0.png`
- `posterior_trace_tau.png`, `posterior_hist_tau.png`, etc., depending on available draws.

### 6.2 Custom Plots
Import `grwhs.viz.plots` directly for advanced plotting (e.g., multiple coefficients, overlay comparisons).

---

## 7. Reporting & Aggregation

### 7.1 make_report CLI
Summarize one or more runs into JSON:
```bash
python -m grwhs.cli.make_report \
  --run outputs/runs/toy_regression_grwhs-<timestamp> \
  --run outputs/runs/toy_classification_grwhs-<timestamp>
```
Creates per-run summaries and a consolidated `summary_index.json` in `outputs/reports/`. Each summary contains metrics, dataset stats, posterior metadata, and convergence results.

Integrate into scripts/notebooks to compare models across the new toy grids, overlapping-group variants, and any real datasets you plug in.

### 7.2 Tables & Plots
- Format metrics for publication via `grwhs.viz.tables` (extend as needed).
- Combine with `grwhs.viz.plots` or custom plotting to build comparative figures.

---

## 8. Posterior Diagnostics & Convergence

- `grwhs/diagnostics/postprocess.py`: shrinkage diagnostics (,  budgets, EDF) from posterior draws.
- `grwhs/diagnostics/convergence.py`: `split_rhat`, `effective_sample_size`, `summarize_convergence` utilities.

`convergence.json` example:
```json
{
  "beta": {"rhat_max": 1.03, "rhat_median": 1.01, "ess_min": 150.4, "ess_median": 210.7},
  "tau": {"rhat_max": 1.02, "ess_min": 180.5}
}
```
Use these to monitor mixing; consider raising iterations or modifying hyperparameters if R-hat exceeds ~1.1 or ESS is low.

### 8.1 Diagnostics Plots CLI

Use `scripts/plot_diagnostics.py` to produce the five reviewer-facing panels (trace, autocorrelation, beta densities, group-level _g shrinkage, coverage-width calibration) with configurable options:

```bash
python scripts/plot_diagnostics.py \
  --run-dir outputs/sweeps/toy_regression_methods/grwhs_full-<timestamp> \
  --burn-in 1000 --max-lag 120 \
  --strong-count 4 --weak-count 4 \
  --groups-to-plot 10 \
  --coverage-levels 0.5 0.7 0.8 0.9 0.95 \
  --dest figures/toy_regression_grwhs --dpi 150
```

- `--run-dir` points to the target run (expects `posterior_samples.npz`, `dataset.npz`, and metadata).
- `--burn-in` (or `--burn-in-frac`) trims early samples; the burn-in split is marked on trace plots.
- `--strong-count` / `--weak-count` select how many strong vs. weak coefficients (with truth overlays) are shown.
- `--groups-to-plot` limits the number of _g violins, sorted by posterior median to highlight selective shrinkage.
- `--coverage-levels` sets the interval grid for the coverage-width calibration curve; the nominal target is annotated automatically.
- Figures default to `<run>/figures/`, but `--dest` can redirect outputs to a publication assets folder.
- The CLI also produces `posterior_reconstruction.png`, overlaying observed responses (gray crosses), posterior mean reconstructions (black dots), and the true signal trace (red) for quick visual assessment of recovery quality.
- Additional hierarchy-focused outputs include:
  * `group_shrinkage_landscape.png` - _g means/credible intervals with signal groups highlighted.
  * `group_coefficient_heatmap.png` - per-group |beta_j| posterior means with _g bars and truth boxes.
  * `group_vs_individual_scatter.png` - co-variation of group-level _g and within-group lambda_j medians.

The script reads group structure, seeds, and posterior arrays directly from the stored artifacts, so the same command works for any dataset/method combination without hard-coded indices.

### 8.2 Randomized Sweep Selector

Use `scripts/random_sweep_selector.py` to randomly subsample `configs/sweeps/mixed_signal_grid.yaml`, execute the corresponding sweeps, and report the best (lowest) RMSE achieved by `grwhs_gibbs`:

```bash
python scripts/random_sweep_selector.py \
  --base-config configs/experiments/toy_regression.yaml \
  --sweep-config configs/sweeps/mixed_signal_grid.yaml \
  --outdir outputs/sweeps/random_mixed \
  --samples 5 \
  --subset-size 4 \
  --seed 2025
```

- Each sampled sweep writes to `outputs/sweeps/random_mixed/<name>`, producing its own `sweep_summary_*.json`.
- After all runs finish, the script parses the summaries, locates the GRwHS run with the smallest RMSE, and prints the winning run directory and summary path.
- Adjust `--samples`, `--subset-size`, and `--seed` to explore different random subsets or increase coverage.

---

## 9. Testing & Validation

### 9.1 Full Test Suite
```bash
python -m pytest
```
Runs unit tests for data generation, inference (SVI/Gibbs), GIG sampler, convergence utilities, visualization scaffolding, and overall smoke tests.

### 9.2 Manual Validation Checklist
1. Execute sweeps for the toy grids (regression/classification) and any real datasets you enable.
2. Inspect metrics via `metrics.json` and aggregated reports.
3. Check `convergence.json` for acceptable R-hat/ESS.
4. Generate plots with `scripts/plot_check.py` and review posterior traces/histograms.
5. Ensure `posterior_samples.npz` exists when `save_posterior=true`.

---

## 10. Extending the Toolkit

### 10.1 New Model/Baseline
- Implement under `grwhs/models/`.
- Register via `@register("name")` in `grwhs/experiments/registry.py`.
- Populate posterior attributes (e.g., `coef_samples_`) if you want convergence diagnostics and posterior plots.

### 10.2 Additional Datasets
- Add YAML configs (synthetic) or loader adapters (real data).
- Modify CLI or scripts to include the new toy/real dataset combinations in benchmark loops.

### 10.3 Diagnostics/Visualization Extensions
- Add new diagnostics to `grwhs/diagnostics/` (e.g., running means, autocorrelation plots).
- Extend `scripts/plot_check.py` or create new scripts for custom visual analytics (e.g., comparing multiple runs on a single figure).

### 10.4 Reporting Enhancements
- Enhance `grwhs/cli/make_report.py` to produce Markdown/HTML, embed plots, or compute aggregated statistics across runs.

---

## 11. Tips & Troubleshooting

- **Imports fail when running scripts**: ensure repo root is on `sys.path`. Provided scripts handle this automatically.
- **JAX warnings about guide variables**: informational; tune the guide or mark auxiliaries if customizing SVI.
- **Posterior files missing**: confirm `experiments.save_posterior=true` and that the model implements posterior arrays.
- **Slow experiments**: reduce `data.n/p`, `experiments.repeats`, or inference iterations via overrides.
- **Poor convergence (R-hat > 1.1)**: increase iterations, adjust learning rates, re-check model hyperparameters.

---

## 12. Command Cheat-Sheet

| Task | Command |
|------|---------|
| Install deps | `pip install -e .[dev]` |
| Run toy regression (GRwHS) | `python -m grwhs.cli.run_experiment --config configs/base.yaml configs/experiments/toy_regression.yaml configs/methods/grwhs_full.yaml --name grwhs_gr_toy` |
| Run toy classification (logistic GRwHS) | `python -m grwhs.cli.run_experiment --config configs/base.yaml configs/experiments/toy_classification.yaml configs/methods/grwhs_full_logistic.yaml --name grwhs_cls_toy` |
| Run real dataset suite | `for model in ridge lasso elastic_net grwhs_gibbs grwhs_svi; do python -m grwhs.cli.run_experiment --config configs/real_dataset.yaml --name ${model}_real --override model.name=${model}; done` |
| Generate plots | `python scripts/plot_check.py <run_dir>` |
| Summarize runs | `python -m grwhs.cli.make_report --run <run_dir> [--run <run_dir>]` |
| Run tests | `python -m pytest` |
| Clean outputs (PowerShell) | `Remove-Item outputs\runs -Recurse -Force` |

---

## 13. Contribution Guidelines

- Keep documentation in sync with code (README, configs, CLI help).
- Run `python -m pytest` before submitting changes.
- Add targeted tests for new models/diagnostics.
- Use issues/PRs to coordinate new features or bug fixes.

By following the steps above you can benchmark all supported models across the toy regression/classification grids, overlap ablations, and any real datasets you plug in, inspecting metrics, posterior behavior, and convergence diagnostics end-to-end.












