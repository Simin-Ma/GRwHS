# EXP3 Run Summary

- Timestamp: `20260420_191504`
- Run directory: `ab_runs\exp3_core30_grrhs_r1_explore_20260420_190152\results\exp3_linear_benchmark\runs\20260420_191504`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3_core30_grrhs_r1_explore_20260420_190152\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3_core30_grrhs_r1_explore_20260420_190152\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3_core30_grrhs_r1_explore_20260420_190152\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3_core30_grrhs_r1_explore_20260420_190152\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3_core30_grrhs_r1_explore_20260420_190152\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3_core30_grrhs_r1_explore_20260420_190152\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\exp3_core30_grrhs_r1_explore_20260420_190152\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GR_RHS | 30 | 28 | 0.933333 | 0.45 | 0.14 | 1.2 | 1 | 0.933333 | 0.933333 | 0.0167305 | 0.0893901 | 0.0244418 | -2.15968 | 0.988571 | 0.628408 | 0.260528 | 0.411951 | 0.933333 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['GR_RHS']
  NOTE: ['RHS'] absent from summary (likely did not converge -- check logs)
  signal=boundary: GR_RHS MSE=0.00943 (rank 1/1, +0.0% vs best GR_RHS)
    All: GR_RHS: 0.00943
  signal=concentrated: GR_RHS MSE=0.04514 (rank 1/1, +0.0% vs best GR_RHS)
    All: GR_RHS: 0.04514
  signal=distributed: GR_RHS MSE=0.00875 (rank 1/1, +0.0% vs best GR_RHS)
    All: GR_RHS: 0.00875
```
