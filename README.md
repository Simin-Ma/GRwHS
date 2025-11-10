# GRwHS Experimentation Toolkit

Comprehensive infrastructure for benchmarking generalized regularized horseshoe (GRwHS) models across synthetic group regression benchmarks, real pathway-aware regression-style datasets, and targeted robustness/ablation studies. The toolkit helps you generate data, train multiple model families, evaluate metrics, track posterior convergence, and produce reproducible reports and plots.

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
  - Contains the Gibbs (`grwhs_gibbs.py`), SVI (`grwhs_svi_numpyro.py`), and convex baselines (ridge/Group Lasso/SGL) implementations.
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

Dataset files adjust only the `data` (and occasionally `standardization` or `splits`) section. The regression campaign now centres on the following descriptors:

- `sim_s1.yaml` *(Group-sparse strong signal)* – n=1300 pool with 300/1000 hold-out splits, 8 uneven groups showing strongly activated blocks in G1/G3, SNR swept via overrides.
- `sim_s2.yaml` *(Dense but weak)* – same structural prior, but 30–50% of features within every group carry 0.2–0.5 effects so global shrinkage must stay gentle.
- `sim_s3.yaml` *(Mixed strong/weak/noise)* – combines an 80% active strong group, a medium group, and sparse weak groups to stress simultaneous group + feature shrinkage.
- `exp4_ablation.yaml` / `exp4_group_misspec.yaml` – legacy configs kept for ablation plots (RHS vs GRwHS, or shuffled groups).
- `real_<dataset>.yaml` – add one YAML per real dataset, pointing to loader entries (see Section 4.2) or direct CSV/NPZ paths plus group maps.

### 3.3 Method presets (`configs/methods/*.yaml`)

Method presets collect model-specific hyperparameters and tuning instructions:

- `grwhs_regression.yaml` – GRwHS Gibbs sampler with RHS-matched global/local priors (τ₀ from s₀=20, η=0.7, 3k iters / 1.5k burn-in).
- `regularized_horseshoe.yaml` – RHS baseline sharing the same τ₀/slab width *and* the exact Gibbs kernel as GRwHS, but with the group layer disabled (`use_groups=false`), so any runtime advantage does not come from better numerics.
- `gigg.yaml` – GIGG Gibbs sampler (fixed a_g=1/n, EB-updated b_g via digamma inverse) with Woodbury accelerations baked into the model class.
- `sparse_group_lasso.yaml` – skglm SGL with log-spaced α grid × {0.2,0.5,0.8} ℓ₁ ratios.
- `lasso.yaml` – classic L1 path using auto-computed λ_max → 10⁻³ λ_max.
- `ridge.yaml` – eight-point L2 grid spanning 1e-4…1e3.
- `group_lasso.yaml` / `group_horseshoe.yaml` – still available for auxiliary ablations.

*(Retired)* Logistic presets (`*_logistic.yaml`) remain in git history if binary tasks return later.

These files can be stacked (dataset + method + ablation override) by passing multiple `--config` arguments or via sweep `config_files`.

### 3.4 Sweep specs (`configs/sweeps/*.yaml`)

Sweeps combine datasets and methods into experiment suites:

- `sim_s1.yaml`, `sim_s2.yaml`, `sim_s3.yaml` – final benchmark sweeps; each variation pairs one of the three SNR overrides (`configs/overrides/snr_{0p5,1p0,3p0}.yaml`) with the six locked baselines (GRwHS, RHS, GIGG, SGL, Lasso, Ridge). Every sweep repeats data generation 30× and standardises the 300/1000 hold-out split.
- `exp4_ablation.yaml` / `exp4_group_misspec.yaml` – optional add-ons for interpretability sections (structure removal or shuffled groups).
- `real_<dataset>_methods.yaml` – create one per real dataset once you configure loaders; follow the synthetic sweeps as a template (same six methods, identical preprocessing, repeated 70/30 shuffles).

Logistic sweeps (`exp2_methods.yaml`, `exp3_real_methods.yaml`) were deleted when the repo was narrowed to regression-only studies.

Each variation may specify extra overrides (e.g. seeds, priors) or add method files via `config_files`.

### 3.5 Running a single experiment

```bash
# GRwHS on Scenario S1 @ SNR=1
python -m grwhs.cli.run_experiment ^
  --config configs/base.yaml ^
          configs/experiments/sim_s1.yaml ^
          configs/overrides/snr_1p0.yaml ^
          configs/methods/grwhs_regression.yaml ^
  --name sim_s1_snr1_grwhs

# GIGG baseline on the same draw @ SNR=0.5
python -m grwhs.cli.run_experiment ^
  --config configs/base.yaml ^
          configs/experiments/sim_s1.yaml ^
          configs/overrides/snr_0p5.yaml ^
          configs/methods/gigg.yaml ^
  --name sim_s1_snr0p5_gigg
```

### 3.6 Launching a sweep

```bash
# Scenario S2 sweep (all 3 SNR levels × 6 methods)
python -m grwhs.cli.run_sweep ^
  --base-config configs/base.yaml ^
  --sweep-config configs/sweeps/sim_s2.yaml ^
  --jobs 6

# Optional: group-misspec ablation
python -m grwhs.cli.run_sweep ^
  --base-config configs/base.yaml ^
  --sweep-config configs/sweeps/exp4_group_misspec.yaml ^
  --jobs 2
```

`run_sweep` merges `base.yaml`, all `common_config_files`, then each variation's `config_files` and `overrides`. Fully resolved configs are written to `<outdir>/.../resolved_config.yaml`.

---

## 4. Benchmark Workflow

Follow this checklist to reproduce the final regression study (three synthetic suites + real data + ablations):

1. **Synthetic Scenarios (S1–S3)**
   - Launch `configs/sweeps/sim_s1.yaml`, `sim_s2.yaml`, and `sim_s3.yaml`. Each sweep iterates over SNR ∈ {0.5, 1, 3} via the override files, repeats the data draw 30 times, and evaluates the six locked baselines (GRwHS, RHS, GIGG, SGL, Lasso, Ridge) under identical preprocessing and nested-CV settings.
   - Outputs live under `outputs/sweeps/sim_s*/` with per-variant resolved configs that document the chosen SNR, seeds, and tuning decisions.

2. **Real Data Benchmarks**
   - Create `configs/experiments/real_<dataset>.yaml` for each dataset (specify loader module + split policy, or furnish `path_X`, `path_y`, and `path_group_map`). Keep the same preprocessing as synthetic runs (train-only standardisation, shared splits across methods).
   - Mirror the synthetic sweeps by adding `configs/sweeps/real_<dataset>_methods.yaml`: six methods, no hyperparameter fiddling beyond what is already codified in `configs/methods/`, and repeated 70/30 (or 5× CV) splits for standard errors.
   - Typical candidates: a crisis-omics regression task with pathway groupings + a second dataset where group structure is curated (e.g. proteomics pathways). Store raw data under `data/real/` and point YAMLs to the loader helper you wire up in `data/loaders.py`.

3. **Ablations & Robustness**
   - `configs/sweeps/exp4_ablation.yaml` isolates the “remove group layer” comparison (GRwHS vs RHS) on the S1 blueprint.
   - `configs/sweeps/exp4_group_misspec.yaml` probes shuffled group assignments (35% of features reassigned before fitting).

4. **Summaries**
   - Use `python -m grwhs.cli.make_report --runs <glob>` to aggregate metrics across all sweeps and real-data repeats.
   - Every run already stores fold-level JSON/NPZ artefacts plus posterior diagnostics, so you can filter by scenario/SNR/model when building tables.

Each run directory records fold-level metrics (`fold_*` subdirectories), resolved configuration, and full metadata so reports can cross-reference calibration statistics (`tau_summary.json`) or tuning diagnostics (`tuning_summary.json`).

> Need a quick reference for the fairness contract (identical preprocessing, nested CV grids, τ heuristics, R-hat checks)? See `docs/fair_benchmark_protocol.md`.
>
> **Implementation details.** `grwhs.experiments.runner` draws the outer folds once per dataset repeat and hands the exact same `OuterFold` objects to every model (no per-model shuffling). The nested inner CV routine is only invoked when a method config supplies a `model.search` grid (ridge/Lasso/SGL); Bayesian baselines (GRwHS, RHS, GIGG, etc.) never read `splits.inner`, so they do not indirectly peek at outer-test data.

## 5. Outputs & Artifacts

Each run directory (`outputs/runs/<name-timestamp>/`) contains:
- `dataset.npz`: standardized train/val/test splits, true coefficients, means/scales.
- `dataset_meta.json`: metadata (n, p, group mapping, splits, model, posterior info).
- `metrics.json`: metrics (`mse`, `r2`, `tpr`, `fpr`, `auc`, etc.) serialized to JSON-friendly types.
- `posterior_samples.npz`: posterior arrays (coefficients, tau, phi, lambda, sigma) if available.
- `repeat_*/fold_*/convergence.json`: fold-level R-hat & ESS summaries computed from posterior arrays (no aggregate file).
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
  --run outputs/runs/sim_s1_snr1_grwhs-<timestamp> \
  --run outputs/runs/real_crisisdata_grwhs-<timestamp>
```
Creates per-run summaries and a consolidated `summary_index.json` in `outputs/reports/`. Each summary contains metrics, dataset stats, posterior metadata, and convergence results.

Integrate into scripts/notebooks to compare models across the four benchmark suites, the ablations, and any real datasets you plug in.

### 7.2 Tables & Plots
- Format metrics for publication via `grwhs.viz.tables` (extend as needed).
- Combine with `grwhs.viz.plots` or custom plotting to build comparative figures.

### 7.3 Sweep comparison summaries
- Every `run_sweep` invocation now emits `sweep_comparison_<timestamp>.json/.csv/.md` alongside `sweep_summary_<timestamp>.json` in the sweep output directory.
- The CSV/Markdown tables list each variation (model) with its aggregated metrics so you can paste them straight into spreadsheets or docs.
- The JSON payload also includes `metric_extrema`, recording which variation achieved the min/max value for every metric; use it to programmatically pick winners or trigger alerts.
- Metrics that a model cannot provide (e.g., MLPD for deterministic baselines) are rendered as `N/A`, so “missing” can’t be confused with “zero”.
- Re-run `run_sweep` with `--dry-run` to preview planned jobs without executing; comparison files are written only for completed sweeps.
- Convex baselines now populate the `MLPD` column via a Gaussian-residual proxy (residual variance measured on the inner training folds, then applied to the outer test residuals). Each fold’s `metrics.json` flags this with `MLPD_source: "gaussian_residual_proxy"` so you can disclose the approximation in the paper.

---

## 8. Posterior Diagnostics & Convergence

- `grwhs/diagnostics/postprocess.py`: shrinkage diagnostics (,  budgets, EDF) from posterior draws.
- `grwhs/diagnostics/convergence.py`: `split_rhat`, `effective_sample_size`, `summarize_convergence` utilities.

Fold-level `convergence.json` example:
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
  --run-dir outputs/sweeps/sim_s1/grwhs_snr1-<timestamp> \
  --burn-in 1000 --max-lag 120 \
  --strong-count 4 --weak-count 4 \
  --groups-to-plot 10 \
  --coverage-levels 0.5 0.7 0.8 0.9 0.95 \
  --dest figures/sim_s1_grwhs --dpi 150
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
  --base-config configs/experiments/sim_s3.yaml \
  --sweep-config configs/sweeps/sim_s3.yaml \
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
1. Execute the S1–S3 sweeps (all SNRs), the real-data sweeps you wired up, and optional Exp4 ablations.
2. Inspect metrics via `metrics.json` and aggregated reports.
3. Check each `repeat_*/fold_*/convergence.json` for acceptable R-hat/ESS.
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
- Modify CLI or scripts to include new benchmark scenarios or real datasets in loops/sweeps.

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
| Run S1 @ SNR=1 (GRwHS) | `python -m grwhs.cli.run_experiment --config configs/base.yaml configs/experiments/sim_s1.yaml configs/overrides/snr_1p0.yaml configs/methods/grwhs_regression.yaml --name sim_s1_snr1_grwhs` |
| Run S2 sweep | `python -m grwhs.cli.run_sweep --base-config configs/base.yaml --sweep-config configs/sweeps/sim_s2.yaml --jobs 6` |
| Run Exp4 misspec sweep | `python -m grwhs.cli.run_sweep --base-config configs/base.yaml --sweep-config configs/sweeps/exp4_group_misspec.yaml --jobs 2` |
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

By following the steps above you can benchmark all supported models across the regression suites, targeted ablations, and any real datasets you plug in, inspecting metrics, posterior behavior, and convergence diagnostics end-to-end.












