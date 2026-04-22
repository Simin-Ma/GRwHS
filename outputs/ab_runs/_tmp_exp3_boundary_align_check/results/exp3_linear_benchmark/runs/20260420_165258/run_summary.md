# EXP3 Run Summary

- Timestamp: `20260420_165258`
- Run directory: `ab_runs\_tmp_exp3_boundary_align_check\results\exp3_linear_benchmark\runs\20260420_165258`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\_tmp_exp3_boundary_align_check\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\_tmp_exp3_boundary_align_check\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\_tmp_exp3_boundary_align_check\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\_tmp_exp3_boundary_align_check\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\_tmp_exp3_boundary_align_check\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\_tmp_exp3_boundary_align_check\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\_tmp_exp3_boundary_align_check\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLS | 1 | 1 | 1 | 0.3 | 0.1 | 0.5 | 1 | 1 | 1 | 0.0377491 | 0.0158342 | 0.0289831 | -3.63634 | nan | nan | nan | nan | 1 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['OLS']
  NOTE: ['GR_RHS', 'RHS'] absent from summary (likely did not converge -- check logs)
  signal=boundary: GR_RHS not in results (did not converge)
    All: OLS: 0.02898
```
