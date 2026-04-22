# EXP3 Run Summary

- Timestamp: `20260420_200059`
- Run directory: `ab_runs\exp3_gigg_mmle_core30_r1_20260420_195523\results\exp3_linear_benchmark\runs\20260420_200059`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3_gigg_mmle_core30_r1_20260420_195523\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3_gigg_mmle_core30_r1_20260420_195523\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3_gigg_mmle_core30_r1_20260420_195523\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3_gigg_mmle_core30_r1_20260420_195523\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3_gigg_mmle_core30_r1_20260420_195523\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3_gigg_mmle_core30_r1_20260420_195523\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\exp3_gigg_mmle_core30_r1_20260420_195523\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GIGG_MMLE | 30 | 18 | 0.6 | 0.45 | 0.14 | 1.2 | 1 | 0.6 | 0.6 | 0.00371964 | 0.151344 | 0.0243131 | -2.08505 | 0.284444 | 0.0713751 | nan | nan | 0.6 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['GIGG_MMLE']
  NOTE: ['GR_RHS', 'RHS'] absent from summary (likely did not converge -- check logs)
  signal=boundary: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.03935
  signal=concentrated: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.02341
  signal=distributed: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.01434
```
