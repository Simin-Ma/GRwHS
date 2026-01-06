# GRRHS Experimentation Toolkit

Comprehensive infrastructure for benchmarking generalized regularized horseshoe (GRRHS) models across synthetic group regression benchmarks, real pathway-aware regression-style datasets, and targeted robustness/ablation studies. The toolkit helps you generate data, train multiple model families, evaluate metrics, track posterior convergence, and produce reproducible reports and plots.

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
| `grrhs/` | Installable package that implements the CLI, runner, models, diagnostics, metrics, visualization, and utilities described in Section 2.2. |
| `scripts/` | Task-oriented automation: diagnostics/plotting (`plot_check.py`, `plot_diagnostics.py`), sweep utilities (`random_sweep_selector.py`), calibration helpers (`calibrate_logistic.py`), etc. |
| `notebooks/` | Scratch space for exploratory analysis or report-ready figures that build on saved outputs. |
| `outputs/` | Auto-generated artifacts. Single runs live under `outputs/runs/<name>-<timestamp>/`, sweeps under `outputs/sweeps/<sweep_id>/<variant>-<timestamp>/`, and aggregated exports under `outputs/reports/`. |
| `tests/` | Pytest suite covering generators, inference kernels, diagnostics, and smoke tests for the CLI pipeline. |
| `random_sweep_selector.py` | Optional entry-point that samples subsets of a sweep spec and executes them (useful for stochastic benchmarking sessions). |
| `pyproject.toml` / `.pre-commit-config.yaml` | Toolchain configuration (packaging metadata, lint/test hooks). |

### 2.2 Core package modules (`grrhs/`)

- `grrhs/cli/`
  - `run_experiment.py` merges any number of config files/overrides, stamps the resolved YAML, and kicks off the runner.
  - `run_sweep.py` iterates over `configs/sweeps/*` definitions, managing per-variant overrides and destination folders.
  - `make_report.py` aggregates finished run directories into JSON/CSV summaries in `outputs/reports/`.
- `grrhs/experiments/`
  - `runner.py` is the orchestration hub: it calls dataset generators, instantiates models, evaluates metrics, and writes artifacts.
  - `registry.py` exposes the `@register` decorator used by every model/baseline so the runner can request them by name.
  - `sweeps.py` loads sweep templates and materializes per-run configurations; `aggregator.py` consolidates fold-level results.
- `grrhs/models/`
  - Contains the Gibbs (`grrhs_gibbs.py`), SVI (`grrhs_svi_numpyro.py`), and convex baselines (ridge/Group Lasso/SGL) implementations.
  - Models rely on inference helpers (sampling routines, Woodbury solvers) and populate posterior buffers used downstream.
- `grrhs/inference/`
  - Linear algebra kernels, proximal updates, and Generalized Inverse Gaussian samplers shared by multiple models.
  - Encapsulates numerical safeguards (jitter, reparameterisations) so models stay focused on statistical logic.
- `grrhs/metrics/`
  - Regression, classification, selection, and calibration metrics consumed by the runner and by `make_report`.
- `grrhs/diagnostics/`
  - `convergence.py`, `shrinkage.py`, and `postprocess.py` compute R-hat/ESS, group shrinkage summaries, and EDF-style diagnostics.
- `grrhs/postprocess/`
  - Currently `debias.py`, which adjusts posterior draws/point-estimates before reporting when requested by configs.
- `grrhs/utils/`
  - Shared infrastructure: config parsing/validation, structured logging, filesystem helpers, and dataclass-like containers.
- `grrhs/viz/`
  - Plotting/table building blocks consumed both by CLI scripts and notebooks (scatter plots, coverage curves, LaTeX-ready tables).

### 2.3 Execution flow (config -> artifacts)

1. Compose a config stack (`configs/base.yaml` + dataset + method + optional overrides) and hand it to `grrhs.cli.run_experiment` or `grrhs.cli.run_sweep`.
2. The CLI resolves/validates the merged YAML, persists `resolved_config.yaml`, and hands control to `grrhs.experiments.runner.Runner`.
3. The runner calls `data/generators.py` & `data/preprocess.py` to build standardised folds, then acquires the requested estimator from `grrhs.experiments.registry`.
4. Models under `grrhs/models/` call into `grrhs/inference/` primitives, emit predictions/posterior draws, and register any auxiliary diagnostics.
5. Metrics from `grrhs/metrics/` and convergence summaries from `grrhs/diagnostics/` are computed before `runner` writes datasets, metrics, posterior arrays, and plots into `outputs/runs/...` (or the sweep-specific subfolder).
6. Reporting/visualisation layers (`grrhs.cli.make_report`, `scripts/plot_check.py`, `scripts/plot_diagnostics.py`, notebooks) consume those artifacts, while `tests/` assert the whole pathway stays stable.

```
configs -> grrhs.cli (run_experiment/run_sweep) -> grrhs.experiments.runner ->
registry/models/inference -> metrics + diagnostics -> outputs/(runs|sweeps|reports) -> scripts/notebooks/tests
```

---

## 3. Configuration Layers

The renewed experiment plan is organised into three composable layers: dataset descriptors (`configs/experiments`), method presets (`configs/methods`), and sweep specifications (`configs/sweeps`). Each run merges `configs/base.yaml` with one (or more) files from those directories before the nested-CV runner executes.

### 3.1 Base config (`configs/base.yaml`)

`base.yaml` encapsulates protocol-wide defaults:

- **Splits** - `splits.outer` defines the outer K-fold (with optional repeats and auto stratification); `splits.inner` sets the inner CV used for convex baselines.
- **Standardisation** - feature scaling and optional response centring (default: regressions centre `y`, classification leaves labels untouched).
- **Model** - GRRHS prior hyperparameters (`c`, `eta`, `s0`) and a `tau` block that can be `mode: calibrated` (m_eff heuristic with p0 grid) or `mode: fixed`.
- **Inference** - Gibbs sampler defaults (iterations, burn-in, jitter).
- **Metrics** - canonical regression/classification metrics evaluated on the outer test fold only.

Every other configuration inherits from this foundation.

### 3.2 Dataset descriptors (`configs/experiments/*.yaml`)

Dataset files adjust only the `data` (and occasionally `standardization` or `splits`) section. The regression campaign now centres on the following descriptors:

- `sim_s1.yaml` *(Group-sparse strong signal)* – n=1300 pool with 300/1000 hold-out splits, 8 uneven groups showing strongly activated blocks in G1/G3, SNR swept via overrides.
- `sim_s2.yaml` *(Dense but weak)* – same structural prior, but 30–50% of features within every group carry 0.2–0.5 effects so global shrinkage must stay gentle.
- `sim_s3.yaml` *(Mixed strong/weak/noise)* – combines an 80% active strong group, a medium group, and sparse weak groups to stress simultaneous group + feature shrinkage.
- `exp4_ablation.yaml` / `exp4_group_misspec.yaml` – legacy configs kept for ablation plots (RHS vs GRRHS, or shuffled groups).
- `real_<dataset>.yaml` – add one YAML per real dataset, pointing to loader entries (see Section 4.2) or direct CSV/NPZ paths plus group maps.

### 3.3 Method presets (`configs/methods/*.yaml`)

Method presets collect model-specific hyperparameters and tuning instructions:

- `grrhs_regression.yaml` – GRRHS Gibbs sampler with RHS-matched global/local priors (τ₀ from s₀=20, η=0.7, 3k iters / 1.5k burn-in).
- `regularized_horseshoe.yaml` – RHS baseline sharing the same τ₀/slab width *and* the exact Gibbs kernel as GRRHS, but with the group layer disabled (`use_groups=false`), so any runtime advantage does not come from better numerics.
- `gigg.yaml` – GIGG Gibbs sampler (fixed a_g=1/n, EB-updated b_g via digamma inverse) with Woodbury accelerations baked into the model class.
- `sparse_group_lasso.yaml` – skglm SGL with log-spaced α grid × {0.2,0.5,0.8} ℓ₁ ratios.
- `lasso.yaml` – classic L1 path using auto-computed λ_max → 10⁻³ λ_max.
- `ridge.yaml` – eight-point L2 grid spanning 1e-4…1e3.
- `group_lasso.yaml` / `group_horseshoe.yaml` – still available for auxiliary ablations; the horseshoe presets now reuse the same Gibbs kernel as GRRHS (with `use_groups=true/false`) so numerical behavior is aligned.

*(Retired)* Logistic presets (`*_logistic.yaml`) remain in git history if binary tasks return later.

These files can be stacked (dataset + method + ablation override) by passing multiple `--config` arguments or via sweep `config_files`.

### 3.4 Sweep specs (`configs/sweeps/*.yaml`)

Sweeps combine datasets and methods into experiment suites:

- `sim_s1.yaml`, `sim_s2.yaml`, `sim_s3.yaml` – final benchmark sweeps; each variation pairs one of the three SNR overrides (`configs/overrides/snr_{0p5,1p0,3p0}.yaml`) with the six locked baselines (GRRHS, RHS, GIGG, SGL, Lasso, Ridge). Every sweep now repeats data generation 3× and standardises the 300/1000 hold-out split to trade minor variance for faster turnarounds.
- `exp4_ablation.yaml` / `exp4_group_misspec.yaml` – optional add-ons for interpretability sections (structure removal or shuffled groups).
- `real_<dataset>_methods.yaml` – create one per real dataset once you configure loaders; follow the synthetic sweeps as a template (same six methods, identical preprocessing, repeated 70/30 shuffles).

Logistic sweeps (`exp2_methods.yaml`, `exp3_real_methods.yaml`) were deleted when the repo was narrowed to regression-only studies.

Each variation may specify extra overrides (e.g. seeds, priors) or add method files via `config_files`.

### 3.5 Running a single experiment

```bash
# GRRHS on Scenario S1 @ SNR=1
python -m grrhs.cli.run_experiment ^
  --config configs/base.yaml ^
          configs/experiments/sim_s1.yaml ^
          configs/overrides/snr_1p0.yaml ^
          configs/methods/grrhs_regression.yaml ^
  --name sim_s1_snr1_grrhs

# GIGG baseline on the same draw @ SNR=0.5
python -m grrhs.cli.run_experiment ^
  --config configs/base.yaml ^
          configs/experiments/sim_s1.yaml ^
          configs/overrides/snr_0p5.yaml ^
          configs/methods/gigg.yaml ^
  --name sim_s1_snr0p5_gigg
```

### 3.6 Launching a sweep

```bash
# Scenario S2 sweep (all 3 SNR levels × 6 methods)
python -m grrhs.cli.run_sweep ^
  --base-config configs/base.yaml ^
  --sweep-config configs/sweeps/sim_s2.yaml ^
  --jobs 6

# Optional: group-misspec ablation
python -m grrhs.cli.run_sweep ^
  --base-config configs/base.yaml ^
  --sweep-config configs/sweeps/exp4_group_misspec.yaml ^
  --jobs 2
```

`run_sweep` merges `base.yaml`, all `common_config_files`, then each variation's `config_files` and `overrides`. Fully resolved configs are written to `<outdir>/.../resolved_config.yaml`.

---

## 4. Benchmark Workflow

Follow this checklist to reproduce the final regression study (three synthetic suites + real data + ablations):

1. **Synthetic Scenarios (S1–S3)**
   - Launch `configs/sweeps/sim_s1.yaml`, `sim_s2.yaml`, and `sim_s3.yaml`. Each sweep iterates over SNR ∈ {0.1, 0.5, 1, 3} via the override files, repeats the data draw 3 times, and evaluates the six locked baselines (GRRHS, RHS, GIGG, SGL, Lasso, Ridge) under identical preprocessing and nested-CV settings.
   - Outputs live under `outputs/sweeps/sim_s*/` with per-variant resolved configs that document the chosen SNR, seeds, and tuning decisions.

2. **Real Data Benchmarks**
   - Create `configs/experiments/real_<dataset>.yaml` for each dataset (specify loader module + split policy, or furnish `path_X`, `path_y`, and `path_group_map`). Keep the same preprocessing as synthetic runs (train-only standardisation, shared splits across methods).
   - Mirror the synthetic sweeps by adding `configs/sweeps/real_<dataset>_methods.yaml`: six methods, no hyperparameter fiddling beyond what is already codified in `configs/methods/`, and repeated 70/30 (or 5× CV) splits for standard errors.
   - Typical candidates: a crisis-omics regression task with pathway groupings + a second dataset where group structure is curated (e.g. proteomics pathways). Store raw data under `data/real/` and point YAMLs to the loader helper you wire up in `data/loaders.py`.

3. **Ablations & Robustness**
   - `configs/sweeps/exp4_ablation.yaml` isolates the “remove group layer” comparison (GRRHS vs RHS) on the S1 blueprint.
   - `configs/sweeps/exp4_group_misspec.yaml` probes shuffled group assignments (35% of features reassigned before fitting).

4. **Summaries**
   - Use `python -m grrhs.cli.make_report --runs <glob>` to aggregate metrics across all sweeps and real-data repeats.
   - Every run already stores fold-level JSON/NPZ artefacts plus posterior diagnostics, so you can filter by scenario/SNR/model when building tables.

Each run directory records fold-level metrics (`fold_*` subdirectories), resolved configuration, and full metadata so reports can cross-reference calibration statistics (`tau_summary.json`) or tuning diagnostics (`tuning_summary.json`).

> Need a quick reference for the fairness contract (identical preprocessing, nested CV grids, τ heuristics, R-hat checks)? See `docs/fair_benchmark_protocol.md`.
>
> **Implementation details.** `grrhs.experiments.runner` draws the outer folds once per dataset repeat and hands the exact same `OuterFold` objects to every model (no per-model shuffling). The nested inner CV routine is only invoked when a method config supplies a `model.search` grid (ridge/Lasso/SGL); Bayesian baselines (GRRHS, RHS, GIGG, etc.) never read `splits.inner`, so they do not indirectly peek at outer-test data.

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
python scripts/plot_check.py outputs/runs/grrhs_svi_B-<timestamp>
```
Outputs include:
- `scatter_pred_vs_truth.png`
- `residual_hist.png`
- `prediction_over_index.png`
- `coefficients_sorted.png`
- `posterior_trace_beta0.png`, `posterior_hist_beta0.png`
- `posterior_trace_tau.png`, `posterior_hist_tau.png`, etc., depending on available draws.

### 6.2 Custom Plots
Import `grrhs.viz.plots` directly for advanced plotting (e.g., multiple coefficients, overlay comparisons).

---

## 7. Reporting & Aggregation

### 7.1 make_report CLI
Summarize one or more runs into JSON:
```bash
python -m grrhs.cli.make_report \
  --run outputs/runs/sim_s1_snr1_grrhs-<timestamp> \
  --run outputs/runs/real_crisisdata_grrhs-<timestamp>
```
Creates per-run summaries and a consolidated `summary_index.json` in `outputs/reports/`. Each summary contains metrics, dataset stats, posterior metadata, and convergence results.

Integrate into scripts/notebooks to compare models across the four benchmark suites, the ablations, and any real datasets you plug in.

### 7.2 Tables & Plots
- Format metrics for publication via `grrhs.viz.tables` (extend as needed).
- Combine with `grrhs.viz.plots` or custom plotting to build comparative figures.

### 7.3 Sweep comparison summaries
- Every `run_sweep` invocation now emits `sweep_comparison_<timestamp>.json/.csv/.md` alongside `sweep_summary_<timestamp>.json` in the sweep output directory.
- The CSV/Markdown tables list each variation (model) with its aggregated metrics so you can paste them straight into spreadsheets or docs.
- The JSON payload also includes `metric_extrema`, recording which variation achieved the min/max value for every metric; use it to programmatically pick winners or trigger alerts.
- Metrics that a model cannot provide (e.g., MLPD for deterministic baselines) are rendered as `N/A`, so “missing” can’t be confused with “zero”.
- Re-run `run_sweep` with `--dry-run` to preview planned jobs without executing; comparison files are written only for completed sweeps.
- Convex baselines now populate the `MLPD` column via a Gaussian-residual proxy (residual variance measured on the inner training folds, then applied to the outer test residuals). Each fold’s `metrics.json` flags this with `MLPD_source: "gaussian_residual_proxy"` so you can disclose the approximation in the paper.

---

## 8. Posterior Diagnostics & Convergence

- `grrhs/diagnostics/postprocess.py`: shrinkage diagnostics (,  budgets, EDF) from posterior draws.
- `grrhs/diagnostics/convergence.py`: `split_rhat`, `effective_sample_size`, `summarize_convergence` utilities.

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
  --run-dir outputs/sweeps/sim_s1/grrhs_snr1-<timestamp> \
  --burn-in 1000 --max-lag 120 \
  --strong-count 4 --weak-count 4 \
  --groups-to-plot 10 \
  --coverage-levels 0.5 0.7 0.8 0.9 0.95 \
  --dest figures/sim_s1_grrhs --dpi 150
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

### 8.2 GRRHS vs RHS Group-Level Comparison

Use `scripts/plot_group_level_comparison.py` to regenerate the synthetic ground-truth coefficients, align them with each fold’s standardized coefficients, and build the GRRHS-vs-RHS figures that highlight structural gains:

```bash
python scripts/plot_group_level_comparison.py \
  --grrhs-dir outputs/sweeps/sim_s3/snr1p0_grrhs-<timestamp> \
  --rhs-dir  outputs/sweeps/sim_s3/snr1p0_rhs-<timestamp> \
  --title "sim_s3 mixed signal (SNR=1.0)" \
  --output-dir outputs/figures/sim_s3_snr1_grrhs_vs_rhs
```

Outputs:
- `sim_s3_group_mse.png` – per-group MSE bars (GRRHS vs RHS) with the strong/medium/weak/null tags shown on the x-axis.
- `sim_s3_group_scales.png` – posterior \(E[\log \phi_g]\) with 90% intervals, contrasted against the RHS global \(E[\log \tau]\).
- `sim_s3_group_combined.png` – side-by-side panel (MSE + scales) for easy drop-in to the main text.
- `sim_s3_group_calibration.png` – scatter of true group ‖β‖ vs the estimated ‖β̂‖ for both models, color-coded by group type.
- `sim_s3_group_stacked.png` – stacked bars of strong/medium/weak/null signal mass comparing ground truth vs GRRHS vs RHS.
- `sim_s3_triptych.png` – 3-in-1 figure (left: per-group MSE, right: log φ_g panel, bottom: MeanEffectiveNonzeros vs SNR with the current SNR highlighted).
- `group_comparison_summary.json` – machine-readable dump (group tags, per-group statistics, figure paths) for tables or supplements.

The script infers group tags from the sim_s3 blueprint, so the captions automatically read “strong dense group”, “medium”, “sparse/weak”, or “null” without any manual labeling.

### 8.3 Coefficient-Level Recovery

To inspect every coefficient (top-k, full scatter, stacked mass), run:

```bash
python scripts/plot_coefficient_recovery.py \
  --grrhs-dir outputs/sweeps/sim_s3/snr3p0_grrhs-<timestamp> \
  --rhs-dir  outputs/sweeps/sim_s3/snr3p0_rhs-<timestamp> \
  --output-dir outputs/figures/sim_s3_snr3_coefficients \
  --title "sim_s3 (SNR=3.0)" \
  --top-k 40
```

Outputs:
- `coeff_scatter_all_std.png` / `coeff_scatter_topk_std.png`: standardized true vs estimated β_j (all coefficients / top-k)。
- `coeff_scatter_all_raw.png` / `coeff_scatter_topk_raw.png`: 同样的散点但映射回原始系数尺度。
- `coeff_bar_topk.png`: grouped bars (truth/GRRHS/RHS) for the top-k coefficients.
- `coeff_mass_stacked.png`: coefficient-level |β| mass share by group tag (truth vs GRRHS vs RHS).
- `coefficients_summary.csv`: per-coefficient table with group tags, true/estimated values, and |β| ranking.

### 8.2 Randomized Sweep Selector

Use `scripts/random_sweep_selector.py` to randomly subsample `configs/sweeps/mixed_signal_grid.yaml`, execute the corresponding sweeps, and report the best (lowest) RMSE achieved by `grrhs_gibbs`:

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
- After all runs finish, the script parses the summaries, locates the GRRHS run with the smallest RMSE, and prints the winning run directory and summary path.
- Adjust `--samples`, `--subset-size`, and `--seed` to explore different random subsets or increase coverage.

---

## 9. Testing & Validation

### 9.1 Full Test Suite
```bash
python -m pytest
```
Runs unit tests for data generation, inference (SVI/Gibbs), GIG sampler, convergence utilities, visualization scaffolding, and overall smoke tests.

### 9.2 Full Validation Checklist (RHS-aware)
Run the automated harness (covers the doctor-level checklist: sanity, degeneracy, sensitivity, negative controls, interpretability, failure notes):
```bash
python scripts/run_validation_checklist.py              # full battery
python scripts/run_validation_checklist.py --minimum    # 14-item publishable core
python scripts/run_validation_checklist.py --minimum --fast --output outputs/checklist.json
```
Outputs list each scenario with `status` (pass/warn), metrics, and notes. Core coverage:
- **Sanity (SC-1/2/3):** pure noise shrinkage; no-group collapse to RHS; single strong signal protected without dragging neighbours.
- **Degeneracy (D-1/2/3/4):** dense-weak → RHS, high-noise collapse, ridge-like behaviour when λ_j are constrained, slab extremes c→∞/small.
- **Sensitivity (S-1/2/3/4):** smooth performance vs τ/ϕ/λ/c sweeps; stable group ordering and controlled false positives.
- **Negative controls (NC-1/2):** dense-weak regime (GRRHS≈RHS≈Ridge); mild degradation under 20% mis-specified groups.
- **Interpretability (E-1/2/3):** ϕ_g rank vs true group strength; ordering stability across seeds; κ separates strong/weak.
- **Failure modes:** documents p≫n + high correlation and near-equal/overlapping groups where GRRHS should not beat RHS but must remain stable.

If you need manual inspection beyond the harness: run `scripts/plot_check.py` on checklist-generated runs, check `convergence.json` for R-hat/ESS, and confirm `posterior_samples.npz` is present for Bayesian models.

---

## 10. Extending the Toolkit

### 10.1 New Model/Baseline
- Implement under `grrhs/models/`.
- Register via `@register("name")` in `grrhs/experiments/registry.py`.
- Populate posterior attributes (e.g., `coef_samples_`) if you want convergence diagnostics and posterior plots.

### 10.2 Additional Datasets
- Add YAML configs (synthetic) or loader adapters (real data).
- Modify CLI or scripts to include new benchmark scenarios or real datasets in loops/sweeps.

### 10.3 Diagnostics/Visualization Extensions
- Add new diagnostics to `grrhs/diagnostics/` (e.g., running means, autocorrelation plots).
- Extend `scripts/plot_check.py` or create new scripts for custom visual analytics (e.g., comparing multiple runs on a single figure).

### 10.4 Reporting Enhancements
- Enhance `grrhs/cli/make_report.py` to produce Markdown/HTML, embed plots, or compute aggregated statistics across runs.

### 10.5 Validation Checklist
- 入口: `python scripts/run_validation_checklist.py [--minimum] [--fast] [--output path]`. `--minimum` 是 14 项核心，`--fast` 只缩短迭代/burn-in，阈值不变。
- 关键参数（fast 数在括号内）：
  - **SC-1 Null / Pure Noise**: `tau0=0.0015`, `eta=0.6`, iters/burn-in `4800/2200` (`3000/1500`); collapse_ok 当 `beta_abs<0.08` 且 `phi_spread<0.35`，仅在 `tau_median>0.5` 且未 collapse 时 WARN。
  - **SC-2 No Group Structure**: grouped vs RHS, `eta=0.5`, iters/burn-in `700/250` (`400/250`); WARN 需 `phi_spread>0.45` 且 `max|Pr(phi_g>phi_h)-0.5|>0.35`，或 `rmse_gap>0.25`。
  - **SC-3 Single Strong Signal**: `c=1.5`, `tau0=0.15`, `eta=0.5`, iters/burn-in `900/300` (`450/300`); warn 若 active beta 未保留或 `beta_abs_inactive>0.25`。
  - **D-1 GRRHS -> RHS**: `eta=0.6`, iters/burn-in `800/250` (`450/250`); warn if `phi_spread>0.25` 或 `rmse_gap>0.15`。
  - **D-2 High-Noise / Small-n**: `tau0=0.0025`, `eta=0.45`, iters/burn-in `3600/1400` (`2400/1000`); warn if (`tau_median>0.65` 且 `beta_abs>0.35`) 或 `beta_abs>0.45`，并附一次严格复核（iters/burn-in `5200/2000` 或 `3600/1600`）记录 `strict_tau_median/strict_beta_abs_mean/strict_rmse` 用于区分 fast 偏差与真实问题。
  - **D-3 Local Shrinkage Collapse**: iters/burn-in `750/240` (`420/240`); lambda 无需 collapse，仅在 ridge-like `kappa` spread `>0.35` 时 WARN。
  - **D-4 Slab Extremes**: `c=0.5` vs `c=50.0`, iters/burn-in `650/220` (`380/220`); 仅在极端 kappa 排序反转时 WARN（默认只记录）。
  - **S-1 tau sensitivity**: `tau0` x {0.3,1,3,10} with `eta=0.6`, iters/burn-in `520/180` (`320/180`); warn if 相邻 RMSE 跳变 `>=0.35`。
  - **S-2 phi_g sensitivity**: `eta` in {0.3,1,3}, iters/burn-in `520/170` (`320/170`); warn if 最小 Spearman(rank) `<0.6`。
  - **S-3 lambda_j sensitivity**: iters/burn-in `780/230` (`500/200`); warn only if active/inactive κ gap `< -0.02`（更长迭代+更宽容，避免 fast 伪影）。
  - **S-4 Slab c sensitivity**: `c` in {0.5,1,2,5}, iters/burn-in `520/170` (`320/170`); 监控 kappa 单调性，连续 c 间 `kappa_mean` 下跌 `>0.1` 才 WARN（`r_mean` 仅记录）。
  - **NC-1 Dense-and-Weak**: `eta=0.4`, iters/burn-in `1000/300` (`650/300`); warn if `rmse_gap>0.35` 或 ridge gap `>0.35`。
  - **NC-2 Misspecified groups**: `eta=0.6`, iters/burn-in `520/170` (`320/170`)，并有严格复核（`900/300` 或 `520/220` fast）对同一错分组：仅当严格复核 `rmse_gap>0.9` 或 `tau_gap>0.75` 时 WARN；若 fast 有大 gap 但严格版收敛，则标记为 fast 伪影。
  - **E-1 / E-2 / E-3**: 共用 `520/170` (`320/170`); E-1 用 Spearman/Top-k/AUC，E-2 以“识别稳”为主：`order_corr_min>=0.4` 或 Top-k 命中/胜率≥0.7 即 PASS（绝对排序可不稳），E-3 检查 kappa 强>弱。
  - **FailureModes**: 恒 INFO；记录已知困难区（近等强组、高相关且 p>>n 等）。
- 采样器 `thin=2`，未列出参数沿用 `grrhs.models.grrhs_gibbs` 默认。

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
| Run S1 @ SNR=1 (GRRHS) | `python -m grrhs.cli.run_experiment --config configs/base.yaml configs/experiments/sim_s1.yaml configs/overrides/snr_1p0.yaml configs/methods/grrhs_regression.yaml --name sim_s1_snr1_grrhs` |
| Run S2 sweep | `python -m grrhs.cli.run_sweep --base-config configs/base.yaml --sweep-config configs/sweeps/sim_s2.yaml --jobs 6` |
| Run Exp4 misspec sweep | `python -m grrhs.cli.run_sweep --base-config configs/base.yaml --sweep-config configs/sweeps/exp4_group_misspec.yaml --jobs 2` |
| Generate plots | `python scripts/plot_check.py <run_dir>` |
| GRRHS vs RHS per-group plots | `python scripts/plot_group_level_comparison.py --grrhs-dir <grrhs_variant_dir> --rhs-dir <rhs_variant_dir>` |
| Coefficient recovery plots | `python scripts/plot_coefficient_recovery.py --grrhs-dir <grrhs_variant_dir> --rhs-dir <rhs_variant_dir>` |
| Summarize runs | `python -m grrhs.cli.make_report --run <run_dir> [--run <run_dir>]` |
| Run tests | `python -m pytest` |
| Clean outputs (PowerShell) | `Remove-Item outputs\runs -Recurse -Force` |

---

## 13. Contribution Guidelines

- Keep documentation in sync with code (README, configs, CLI help).
- Run `python -m pytest` before submitting changes.
- Add targeted tests for new models/diagnostics.
- Use issues/PRs to coordinate new features or bug fixes.

By following the steps above you can benchmark all supported models across the regression suites, targeted ablations, and any real datasets you plug in, inspecting metrics, posterior behavior, and convergence diagnostics end-to-end.












