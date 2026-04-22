# EXP3 Run Summary

- Timestamp: `20260421_182230`
- Run directory: `ab_runs\exp3_gigg_mode_compare_20260421_171226\stable\results\exp3_linear_benchmark\runs\20260421_182230`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\stable\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\stable\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\stable\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\stable\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\stable\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\stable\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\stable\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GIGG_MMLE | 48 | 34 | 0.708333 | 0.425 | 0.15 | 1.25 | 3 | 2.125 | 2.125 | 0.404225 | 0.514878 | 0.381409 | -2.71336 | 0.263125 | 0.34208 | nan | nan | 2.125 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['GIGG_MMLE']
  NOTE: ['GR_RHS', 'RHS'] absent from summary (likely did not converge -- check logs)
  signal=concentrated: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.74626
  signal=distributed: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.01656
```
