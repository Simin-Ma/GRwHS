# EXP3 Run Summary

- Timestamp: `20260420_192741`
- Run directory: `ab_runs\exp3_grrhs_core30_r1_20260420_192529\results\exp3_linear_benchmark\runs\20260420_192741`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3_grrhs_core30_r1_20260420_192529\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3_grrhs_core30_r1_20260420_192529\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3_grrhs_core30_r1_20260420_192529\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3_grrhs_core30_r1_20260420_192529\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3_grrhs_core30_r1_20260420_192529\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3_grrhs_core30_r1_20260420_192529\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\exp3_grrhs_core30_r1_20260420_192529\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GR_RHS | 30 | 30 | 1 | 0.45 | 0.14 | 1.2 | 1 | 1 | 1 | 0.0164899 | 0.0858262 | 0.02524 | -2.16062 | 0.984 | 0.626973 | 0.249616 | 0.406318 | 1 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['GR_RHS']
  NOTE: ['RHS'] absent from summary (likely did not converge -- check logs)
  signal=boundary: GR_RHS MSE=0.01909 (rank 1/1, +0.0% vs best GR_RHS)
    All: GR_RHS: 0.01909
  signal=concentrated: GR_RHS MSE=0.04510 (rank 1/1, +0.0% vs best GR_RHS)
    All: GR_RHS: 0.04510
  signal=distributed: GR_RHS MSE=0.00846 (rank 1/1, +0.0% vs best GR_RHS)
    All: GR_RHS: 0.00846
```
