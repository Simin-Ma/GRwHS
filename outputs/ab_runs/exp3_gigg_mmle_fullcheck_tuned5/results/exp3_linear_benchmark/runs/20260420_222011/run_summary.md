# EXP3 Run Summary

- Timestamp: `20260420_222011`
- Run directory: `ab_runs\exp3_gigg_mmle_fullcheck_tuned5\results\exp3_linear_benchmark\runs\20260420_222011`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3_gigg_mmle_fullcheck_tuned5\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3_gigg_mmle_fullcheck_tuned5\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3_gigg_mmle_fullcheck_tuned5\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3_gigg_mmle_fullcheck_tuned5\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3_gigg_mmle_fullcheck_tuned5\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3_gigg_mmle_fullcheck_tuned5\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\exp3_gigg_mmle_fullcheck_tuned5\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GIGG_MMLE | 30 | 20 | 0.666667 | 0.45 | 0.14 | 1.2 | 1 | 0.666667 | 0.666667 | 0.127635 | 0.266689 | 0.151897 | -2.25613 | 0.29 | 0.158304 | nan | nan | 0.666667 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['GIGG_MMLE']
  NOTE: ['GR_RHS', 'RHS'] absent from summary (likely did not converge -- check logs)
  signal=boundary: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.03912
  signal=concentrated: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.34276
  signal=distributed: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.01432
```
