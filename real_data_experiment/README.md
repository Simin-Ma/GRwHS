# real_data_experiment

Real-data comparison package for `GR-RHS` and the main baselines on the two runner-ready datasets in `data/real/`.

## What It Does

- runs repeated holdout comparisons on the same split across methods
- reuses the existing fit runtime, convergence gate, history output, paired summary, and paper-table workflow
- keeps real-data metrics separate from synthetic `beta_true` metrics

## Convergence-First Rule

- `run-real-data` always forces the Bayesian convergence gate on and uses the legacy `until converged` retry mode (`enforce_bayes_convergence = true`, `max_convergence_retries = -1`).
- A real-data replicate only has formal discussion value when the row is `status = ok` and `converged = True`.
- Cross-method claims should use the common-converged paired outputs: `raw_results_paired.csv`, `summary_paired.csv`, `summary_paired_deltas.csv`, and the `paper_tables/` artifacts.
- `summary.csv` is still useful, but it is a marginal converged-only diagnostic summary, not the final paired evidence table.
- Smoke runs can debug dataset wiring and runtime issues, but unconverged runs do not count as completed evidence.

## Datasets

- `nhanes_2003_2004`
  Uses the 35 grouped exposure variables as the shrinkage design matrix and keeps the 9 covariates in a train-only nuisance model.
- `covid19_trust_experts`
  Uses the 101 grouped features directly, with no extra covariate residualization path.
- `gse40279_age_gene_groups_smoke`
  Uses 2,000 high-variance CpGs from GSE40279 and groups them into disjoint single-gene proxy groups for a methylation-age smoke benchmark.
- `gse40279_age_gene_groups_micro`
  Uses a 200-feature subset from the same preprocessing pipeline as a lightweight runner-verification benchmark.

## Evaluation Scope

- predictive metrics: `rmse_test`, `mae_test`, `r2_test`, `lpd_test`
- operational metrics: runtime, convergence, retry count
- sparsity and group diagnostics: selected coefficient count, selected group count, top group, group entropy
- stability summaries: pairwise group-selection Jaccard and per-group selection frequency

For `NHANES`, the runner residualizes both `X` and `y` against covariates on the training split only, fits methods on the residual problem, then adds the nuisance prediction back when scoring the test set. This keeps the grouped-exposure comparison fair while preserving held-out metrics on the original response scale.

`response_standardization` supports `none`, `train_center`, and `train_center_scale`. Evaluation always maps predictions back to the original response scale before computing held-out metrics.

## CLI

```bash
python -m real_data_experiment.src.run_real_data list-datasets
python -m real_data_experiment.src.run_real_data describe-dataset --dataset-id nhanes_2003_2004
python -m real_data_experiment.src.run_real_data run-real-data
python -m real_data_experiment.src.run_real_data run-real-data --datasets nhanes_2003_2004 --methods GR_RHS RHS OLS
python -m real_data_experiment.src.run_real_data --config real_data_experiment/config/gse40279_micro_runner_check.yaml run-real-data
python -m real_data_experiment.src.run_real_data --config real_data_experiment/config/gse40279_smoke_runner_check.yaml run-real-data
python -m real_data_experiment.src.run_real_data build-tables --results-dir outputs/history/real_data_experiment/main
```

## Main Outputs

- `raw_results.csv`
- `summary.csv`
- `raw_results_paired.csv`
- `summary_paired.csv`
- `summary_paired_deltas.csv`
- `selection_stability.csv`
- `group_selection_frequency.csv`
- `paper_tables/paper_table_main.{csv,md,tex}`
- `paper_tables/paper_table_appendix_full.{csv,md,tex}`

Each run is written into its own timestamped directory under the configured history root, and the root keeps `latest_run.json`, `latest_run.txt`, and `session_index.jsonl`.
