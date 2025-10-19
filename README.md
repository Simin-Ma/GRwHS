# GRwHS Experimentation Toolkit

Comprehensive infrastructure for benchmarking generalized regularized horseshoe (GRwHS) models across synthetic scenarios (A-D) and real datasets. The toolkit helps you generate data, train multiple model families, evaluate metrics, track posterior convergence, and produce reproducible reports and plots.

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

## 2. Repository Layout

```
configs/          YAML experiment templates (base + scenarios A-D)
data/             synthetic generators, preprocessing, splits, loaders
grwhs/            main package namespace
  cli/            command-line entry points
  diagnostics/    posterior & convergence diagnostics
  experiments/    experiment orchestration (runner, sweeps, aggregator)
  inference/      linear algebra helpers, samplers, GIG sampling
  metrics/        regression/selection/uncertainty metrics
  models/         GRwHS implementations (SVI, Gibbs) & baselines
  utils/          config parsing, logging, IO
  viz/            plotting/table utilities
outputs/          run artifacts (created automatically)
scripts/          convenience scripts (plot & posterior checks)
tests/            pytest suite
```

Important modules:
- `grwhs/cli/run_experiment.py`: merges configs, runs a single experiment
- `grwhs/cli/make_report.py`: builds JSON summaries across runs
- `grwhs/experiments/runner.py`: generates data, fits models, saves metrics & posterior samples
- `grwhs/models/grwhs_svi_numpyro.py`, `grwhs/models/grwhs_gibbs.py`: GRwHS model implementations
- `grwhs/diagnostics/convergence.py`: split R-hat & ESS computations
- `scripts/plot_check.py`: generates standard and posterior plots for a run

---

## 3. Configurations & Scenarios

### 3.1 Base Config (`configs/base.yaml`)
Defines defaults for experiment orchestration, including:
- `data`: synthetic settings (n, p, groups, correlation, signal)
- `model`: default GRwHS hyperparameters
- `inference.svi` / `inference.gibbs`: iteration counts, seeds, learning rates
- `experiments.save_posterior`: enabled (true) to persist posterior draws & convergence stats
- `logging`, `checkpointing`, `viz`: global behavior

### 3.2 Scenario Templates (`configs/scenario_A.yaml` - `scenario_D.yaml`)
Each scenario overrides the base configuration:
- A: Sparse + weak correlation
- B: Mixed strengths + block correlation
- C: Dense weak signals + strong correlation
- D: High-dimensional scalability (p >> n)

### 3.3 CLI Overrides
Extra tweaks without editing YAML:
```bash
python -m grwhs.cli.run_experiment \
  --config configs/base.yaml configs/scenario_A.yaml \
  --name custom_run \
  --override data.n=64 data.p=32 experiments.repeats=3 \
             model.name=grwhs_svi inference.svi.steps=500
```
Supports dotted paths (`section.subsection.key=value`).

### 3.4 Binary Classification Mode
Logistic likelihoods are now fully supported via a dedicated Gibbs sampler that augments the Horseshoe hierarchy with Pólya–Gamma variables. Every configuration (`base.yaml`, scenarios A–D) exposes a `model_variants.classification` block that switches `model.name` to `grwhs_gibbs_logistic` and mirrors the regression hyper-parameters. Optional overrides can live under `inference_variants.classification.gibbs` (e.g., different burn-in or slice tuning).

To run a scenario in classification mode:

1. Set `task: classification` (in YAML or `--override task=classification`).
2. Keep `data.classification` tuned for your desired logit scale/bias and leave the sparsity block unchanged.
3. Disable response centering (`standardization.y_center=false`) so that binary labels remain in `{0, 1}`.
4. (Optional) Adjust logistic-specific inference controls under `inference_variants.classification`.

Example excerpt:
```yaml
task: classification
standardization:
  X: unit_variance
  y_center: false
model_variants:
  classification:
    name: grwhs_gibbs_logistic
    iters: 12000
inference_variants:
  classification:
    gibbs:
      burn_in: 6000
      slice_w: 0.15
      slice_m: 15
data:
  classification:
    scale: 0.9
    bias: 0.1
experiments:
  classification_threshold: 0.45
```

Behind the scenes the sampler draws `ω ~ PG(1, xᵢᵀβ)` using the [`polyagamma`](https://pypi.org/project/polyagamma/) package (now listed as a dependency). Posterior predictive probabilities are exposed through `predict_proba`, while `predict` returns hard labels for convenience. Classification metrics (`ClassAccuracy`, `ClassLogLoss`, `ClassBrier`, `ClassAUROC`, etc.) remain available alongside sparsity diagnostics (`AUC-PR`, `F1`) for synthetic truth comparisons.

### 3.5 Real Data Support
Set `data.type=loader` and implement adapter in `data/loaders.py` (mapping to `(X, y, groups)` tuple). Provide file paths under `data.loader.*`. You can mix real data config with overrides or scenario templates.

---

## 4. Benchmark Workflow

### 4.1 Synthetic Benchmarks (Scenarios A-D)
Use the Gibbs sampler by default for all GRwHS comparisons. SVI should only be selected for very large problems (e.g., `p > 1500` or `n`/`p` beyond the Gibbs defaults).

**Step 1 – Sweep all models on a scenario**

```bash
# Scenario A comparison (GRwHS-Gibbs + convex baselines)
python -m grwhs.cli.run_sweep \
  --base-config configs/scenario_A.yaml \
  --sweep-config configs/sweeps/all_models_base.yaml \
  --outdir outputs/sweeps/scenario_A
```

The sweep uses `configs/scenario_A.yaml` for shared settings and the variations defined in `configs/sweeps/all_models_base.yaml` (GRwHS-Gibbs plus ridge/lasso/elastic-net/group-lasso/sparse-group-lasso/horseshoe baselines). Repeat with `configs/scenario_B.yaml`, etc., by changing `--base-config` (and optionally `--outdir`).

**Scenario B quick commands**

```powershell
# 1. Full multi-model sweep (re-uses Scenario B core settings)
python -m grwhs.cli.run_sweep `
  --base-config configs/scenario_B.yaml `
  --sweep-config configs/sweeps/all_models_base.yaml `
  --outdir outputs/sweeps/scenario_B/models_full

# 2. Gibbs-only noise-scale scan (short 8k chains)
python -m grwhs.cli.run_sweep `
  --base-config configs/base.yaml `
  --sweep-config configs/sweeps/scenario_B_gibbs_s0_coarse.yaml

# 3. Gibbs-only long-chain refinement (20k effective samples per run)
python -m grwhs.cli.run_sweep `
  --base-config configs/base.yaml `
  --sweep-config configs/sweeps/scenario_B_gibbs_s0_long.yaml

# 4. Standard logistic regression baseline on Scenario B
python -m grwhs.cli.run_experiment `
  --config configs/scenario_B.yaml `
  --name scenarioB_logreg `
  --override task=classification `
             standardization.y_center=false `
             model.name=logistic_regression
```

Each command writes metrics under `outputs/sweeps/scenario_B/...` (for sweeps) or `outputs/runs/...` (for single baselines). Re-run the applicable `make_report` command afterwards to refresh the markdown/JSON summaries.

**Step 2 – Optional SVI runs for large-scale cases**

Only if Scenario D (or a custom config) exceeds the practical Gibbs limit, run SVI explicitly:

```bash
python -m grwhs.cli.run_experiment \
  --config configs/scenario_D.yaml \
  --name grwhs_svi_D \
  --override model.name=grwhs_svi \
             inference.svi.steps=3000 \
             inference.svi.lr=8e-3 \
             inference.svi.batch_size=512
```

Keep the output directory pattern (`--name`) aligned with sweep naming so that reports group correctly.

### 4.2 Real Dataset Benchmarks
1. Implement loader returning `(X, y, group_indices)` in `data/loaders.py`.
2. Create YAML config, e.g. `configs/real_dataset.yaml`:
   ```yaml
   defaults: "configs/base.yaml"
   data:
     type: "loader"
     loader:
       path_X: "data/my_dataset_X.npy"
       path_y: "data/my_dataset_y.npy"
       group_map: "data/my_dataset_groups.json"
   experiments:
     repeats: 3
     save_posterior: true
   ```
3. Run suite for each model:
   ```bash
   for model in ridge lasso elastic_net grwhs_gibbs grwhs_svi; do
     python -m grwhs.cli.run_experiment \
       --config configs/real_dataset.yaml \
       --name ${model}_real \
       --override model.name=${model}
   done
   ```

### 4.3 Automation via Scripts or Sweeps
- Use `grwhs/experiments/sweeps.py` to build custom grid/combinations.
- Script loops (e.g., bash/PowerShell or Python) to iterate over models, scenarios, seeds.

### 4.4 Capturing Results in Reports
After each sweep or standalone run, update the consolidated report artifacts:

```bash
# Collect metrics for the freshest Scenario A sweep
python -m grwhs.cli.make_report \
  --run outputs/sweeps/scenario_A/* \
  --outdir outputs/reports/scenario_A

# Combine multiple scenarios into one summary
python -m grwhs.cli.make_report \
  --run outputs/sweeps/scenario_A/* \
         outputs/sweeps/scenario_B/* \
  --outdir outputs/reports/combined
```

The report command writes a `summary_index.json` plus per-run JSON summaries. Include the resulting report directory (e.g., `outputs/reports/scenario_A`) when drafting comparison tables or figures.

---

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
  --run outputs/runs/grwhs_svi_B-<timestamp> \
  --run outputs/runs/grwhs_gibbs_B-<timestamp>
```
Creates per-run summaries and a consolidated `summary_index.json` in `outputs/reports/`. Each summary contains metrics, dataset stats, posterior metadata, and convergence results.

Integrate into scripts/notebooks to compare models across scenarios and real datasets.

### 7.2 Tables & Plots
- Format metrics for publication via `grwhs.viz.tables` (extend as needed).
- Combine with `grwhs.viz.plots` or custom plotting to build comparative figures.

---

## 8. Posterior Diagnostics & Convergence

- `grwhs/diagnostics/postprocess.py`: shrinkage diagnostics (κ, ω budgets, EDF) from posterior draws.
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

Use `scripts/plot_diagnostics.py` to produce the five reviewer-facing panels (trace, autocorrelation, β densities, group-level ϕ_g shrinkage, coverage–width calibration) with configurable options:

```bash
python scripts/plot_diagnostics.py \
  --run-dir outputs/sweeps/scenario_A/grwhs_gibbs-<timestamp> \
  --burn-in 1000 --max-lag 120 \
  --strong-count 4 --weak-count 4 \
  --groups-to-plot 10 \
  --coverage-levels 0.5 0.7 0.8 0.9 0.95 \
  --dest figures/scenario_A_gibbs --dpi 150
```

- `--run-dir` points to the target run (expects `posterior_samples.npz`, `dataset.npz`, and metadata).
- `--burn-in` (or `--burn-in-frac`) trims early samples; the burn-in split is marked on trace plots.
- `--strong-count` / `--weak-count` select how many strong vs. weak coefficients (with truth overlays) are shown.
- `--groups-to-plot` limits the number of ϕ_g violins, sorted by posterior median to highlight selective shrinkage.
- `--coverage-levels` sets the interval grid for the coverage–width calibration curve; the nominal target is annotated automatically.
- Figures default to `<run>/figures/`, but `--dest` can redirect outputs to a publication assets folder.
- The CLI also produces `posterior_reconstruction.png`, overlaying observed responses (gray crosses), posterior mean reconstructions (black dots), and the true signal trace (red) for quick visual assessment of recovery quality.
- Additional hierarchy-focused outputs include:
  * `group_shrinkage_landscape.png` – ϕ_g means/credible intervals with signal groups highlighted.
  * `group_coefficient_heatmap.png` – per-group |β_j| posterior means with φ_g bars and truth boxes.
  * `group_vs_individual_scatter.png` – co-variation of group-level ϕ_g and within-group λ_j medians.

The script reads group structure, seeds, and posterior arrays directly from the stored artifacts, so the same command works for any scenario/model without hard-coded indices.

### 8.2 Randomized Sweep Selector

Use `scripts/random_sweep_selector.py` to randomly subsample `configs/sweeps/mixed_signal_grid.yaml`, execute the corresponding sweeps, and report the best (lowest) RMSE achieved by `grwhs_gibbs`:

```bash
python scripts/random_sweep_selector.py \
  --base-config configs/scenario_A.yaml \
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
1. Execute SVI & Gibbs runs for each scenario (and real data, if available).
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

### 10.2 Additional Scenarios or Datasets
- Add YAML configs (synthetic) or loader adapters (real data).
- Modify CLI or scripts to include new scenario/dataset combinations in benchmark loops.

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
| Run scenario A (SVI) | `python -m grwhs.cli.run_experiment --config configs/base.yaml configs/scenario_A.yaml --name grwhs_svi_A --override model.name=grwhs_svi inference.svi.steps=3000` |
| Run scenario B (Gibbs) | `python -m grwhs.cli.run_experiment --config configs/base.yaml configs/scenario_B.yaml --name grwhs_gibbs_B --override model.name=grwhs_gibbs` |
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

By following the steps above you can benchmark all supported models across scenarios A-D and any real datasets you plug in, inspecting metrics, posterior behavior, and convergence diagnostics end-to-end.
