# EXP3 Run Summary

- Timestamp: `20260421_171547`
- Run directory: `ab_runs\exp3_gigg_mode_compare_20260421_171226\paper_ref\results\exp3_linear_benchmark\runs\20260421_171547`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\paper_ref\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\paper_ref\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\paper_ref\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\paper_ref\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\paper_ref\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\paper_ref\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\exp3_gigg_mode_compare_20260421_171226\paper_ref\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GIGG_MMLE | 48 | 5 | 0.104167 | 0.425 | 0.15 | 1.25 | 3 | 0.3125 | 0.3125 | 0.00258537 | 0.243225 | 0.0177185 | -1.96809 | 0.292 | 0.0297183 | nan | nan | 0.3125 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['GIGG_MMLE']
  NOTE: ['GR_RHS', 'RHS'] absent from summary (likely did not converge -- check logs)
  signal=concentrated: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.03060
  signal=distributed: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.01450
```
