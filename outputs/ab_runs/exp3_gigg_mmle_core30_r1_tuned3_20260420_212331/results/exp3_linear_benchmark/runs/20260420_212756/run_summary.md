# EXP3 Run Summary

- Timestamp: `20260420_212756`
- Run directory: `ab_runs\exp3_gigg_mmle_core30_r1_tuned3_20260420_212331\results\exp3_linear_benchmark\runs\20260420_212756`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3_gigg_mmle_core30_r1_tuned3_20260420_212331\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3_gigg_mmle_core30_r1_tuned3_20260420_212331\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3_gigg_mmle_core30_r1_tuned3_20260420_212331\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3_gigg_mmle_core30_r1_tuned3_20260420_212331\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3_gigg_mmle_core30_r1_tuned3_20260420_212331\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3_gigg_mmle_core30_r1_tuned3_20260420_212331\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\exp3_gigg_mmle_core30_r1_tuned3_20260420_212331\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GIGG_MMLE | 30 | 19 | 0.633333 | 0.45 | 0.14 | 1.2 | 1 | 0.633333 | 0.633333 | 0.00140952 | 0.163943 | 0.0218828 | -1.98274 | 0.270526 | 0.0685485 | nan | nan | 0.633333 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['GIGG_MMLE']
  NOTE: ['GR_RHS', 'RHS'] absent from summary (likely did not converge -- check logs)
  signal=boundary: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.03845
  signal=concentrated: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.01387
  signal=distributed: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.01446
```
