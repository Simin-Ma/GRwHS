# Thesis Real-Data Figure Plan

## Formal CV sweeps

- COVID formal sweep: `configs/sweeps/real_covid19_trust_experts_methods_thesis.yaml`
- NHANES formal sweep: `configs/sweeps/real_nhanes_2003_2004_ggt_methods_thesis.yaml`
- Shared 5-fold override: `configs/overrides/real_data_thesis_cv5.yaml`

These thesis sweeps restore:

- `5` outer folds
- `5` inner folds
- posterior saving for Bayesian runs
- convergence diagnostics enabled

## COVID figures

### Main comparison figure

```bash
python scripts/plot_real_sweep_overview.py ^
  --comparison-csv outputs/sweeps/real_covid19_trust_experts_thesis/sweep_comparison_<timestamp>.csv ^
  --out outputs/reports/covid_thesis/covid_sweep_overview.png ^
  --title "COVID-19 Trust in Experts: 5-fold CV Model Comparison"
```

### Bayesian diagnostics

For each Bayesian method (`GR-RHS`, `RHS`, `GIGG`), first build a run-like directory from a representative fold:

```bash
python scripts/build_covid_run_like.py ^
  --run-dir outputs/sweeps/real_covid19_trust_experts_thesis/trust_experts_grrhs-<timestamp> ^
  --repeat 1 ^
  --fold 1 ^
  --dest outputs/reports/covid_thesis/grrhs_fold1_run_like
```

Then generate diagnostics:

```bash
python scripts/plot_diagnostics.py ^
  --run-dir outputs/reports/covid_thesis/grrhs_fold1_run_like ^
  --dest outputs/reports/covid_thesis/grrhs_fold1_diagnostics
```

Recommended thesis panels from COVID:

- sweep overview
- GR-RHS diagnostics
- RHS diagnostics
- GIGG diagnostics

## NHANES figures

### Sweep overview

```bash
python scripts/plot_real_sweep_overview.py ^
  --comparison-csv outputs/sweeps/real_nhanes_2003_2004_ggt_thesis/sweep_comparison_<timestamp>.csv ^
  --out outputs/reports/nhanes_thesis/nhanes_sweep_overview.png ^
  --title "NHANES 2003-2004: 5-fold CV Model Comparison"
```

### Effect summary and paper figures

```bash
python scripts/summarize_nhanes_effects.py ^
  --sweep-dir outputs/sweeps/real_nhanes_2003_2004_ggt_thesis ^
  --out-dir outputs/reports/nhanes_thesis/effects ^
  --reference-model RHS
```

```bash
python scripts/plot_nhanes_group_ci.py ^
  --summary-csv outputs/reports/nhanes_thesis/effects/nhanes_group_ci_summary.csv ^
  --out outputs/reports/nhanes_thesis/effects/nhanes_group_ci_barplot.png
```

```bash
python scripts/plot_nhanes_group_ci_main.py ^
  --summary-csv outputs/reports/nhanes_thesis/effects/nhanes_group_ci_summary.csv ^
  --out outputs/reports/nhanes_thesis/effects/nhanes_group_ci_main.png
```

```bash
python scripts/plot_nhanes_grrhs_rhs_forest.py ^
  --exposure-csv outputs/reports/nhanes_thesis/effects/nhanes_exposure_effects.csv ^
  --out outputs/reports/nhanes_thesis/effects/nhanes_grrhs_rhs_forest.png
```

```bash
python scripts/plot_nhanes_all_models_forest.py ^
  --exposure-csv outputs/reports/nhanes_thesis/effects/nhanes_exposure_effects.csv ^
  --out outputs/reports/nhanes_thesis/effects/nhanes_all_models_forest.png
```

```bash
python scripts/plot_nhanes_group_structure.py ^
  --sweep-summary outputs/sweeps/real_nhanes_2003_2004_ggt_thesis/sweep_summary_<timestamp>.json ^
  --out-dir outputs/reports/nhanes_thesis/group_structure
```

Recommended thesis panels from NHANES:

- sweep overview
- GR-RHS vs RHS forest
- all-model forest
- group CI main
- group CI supplementary bar plot
- predictive-complexity tradeoff
- group signal concentration
- coefficient heatmap by exposure group
- group predictive contribution
- GR-RHS group-level posterior shrinkage supplement
