# EXP3 Run Summary

- Timestamp: `20260420_173746`
- Run directory: `ab_runs\exp3_core30_lasso_r1_20260420_173742\results\exp3_linear_benchmark\runs\20260420_173746`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3_core30_lasso_r1_20260420_173742\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3_core30_lasso_r1_20260420_173742\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3_core30_lasso_r1_20260420_173742\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3_core30_lasso_r1_20260420_173742\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3_core30_lasso_r1_20260420_173742\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3_core30_lasso_r1_20260420_173742\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\exp3_core30_lasso_r1_20260420_173742\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LASSO_CV | 30 | 30 | 1 | 0.45 | 0.14 | 1.2 | 1 | 1 | 1 | 0.0211681 | 0.108026 | 0.0344836 | -2.22029 | nan | nan | nan | nan | 1 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['LASSO_CV']
  NOTE: ['GR_RHS', 'RHS'] absent from summary (likely did not converge -- check logs)
  signal=boundary: GR_RHS not in results (did not converge)
    All: LASSO_CV: 0.02626
  signal=concentrated: GR_RHS not in results (did not converge)
    All: LASSO_CV: 0.06317
  signal=distributed: GR_RHS not in results (did not converge)
    All: LASSO_CV: 0.00991
```
