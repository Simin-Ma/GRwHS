# EXP3 Run Summary

- Timestamp: `20260420_193833`
- Run directory: `ab_runs\exp3_processpool_guard_smoke_20260420_193513\results\exp3_linear_benchmark\runs\20260420_193833`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3_processpool_guard_smoke_20260420_193513\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3_processpool_guard_smoke_20260420_193513\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3_processpool_guard_smoke_20260420_193513\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3_processpool_guard_smoke_20260420_193513\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3_processpool_guard_smoke_20260420_193513\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3_processpool_guard_smoke_20260420_193513\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\exp3_processpool_guard_smoke_20260420_193513\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RHS | 30 | 17 | 0.566667 | 0.45 | 0.14 | 1.2 | 1 | 0.566667 | 0.566667 | 0.0332566 | 0.0628589 | 0.0378388 | -2.28861 | 0.991765 | 0.832305 | nan | nan | 0.566667 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['RHS']
  NOTE: ['GR_RHS'] absent from summary (likely did not converge -- check logs)
  signal=boundary: GR_RHS not in results (did not converge)
    All: RHS: 0.01432
  signal=concentrated: GR_RHS not in results (did not converge)
    All: RHS: 0.06070
  signal=distributed: GR_RHS not in results (did not converge)
    All: RHS: 0.00152
```
