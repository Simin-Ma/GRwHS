# EXP3A Run Summary

- Timestamp: `20260421_192524`
- Run directory: `ab_runs\exp3ab_dirs_smoke_20260421_192521\results\exp3a_main_benchmark\runs\20260421_192524`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3ab_dirs_smoke_20260421_192521\figures\exp3a_main_benchmark\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3ab_dirs_smoke_20260421_192521\figures\exp3a_main_benchmark\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3ab_dirs_smoke_20260421_192521\figures\exp3a_main_benchmark\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3ab_dirs_smoke_20260421_192521\results\exp3a_main_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3ab_dirs_smoke_20260421_192521\results\exp3a_main_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3ab_dirs_smoke_20260421_192521\results\exp3a_main_benchmark\summary.csv`
- `table`: `ab_runs\exp3ab_dirs_smoke_20260421_192521\tables\exp3a_main_benchmark\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLS | 12 | 12 | 1 | 0.55 | 0.1 | 1 | 1 | 1 | 1 | 0.416078 | 0.605622 | 0.544926 | -3.26901 | nan | nan | nan | nan | 1 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['OLS']
  NOTE: ['GR_RHS', 'RHS'] absent from summary (likely did not converge -- check logs)
  signal=concentrated: GR_RHS not in results (did not converge)
    All: OLS: 0.97667
  signal=distributed: GR_RHS not in results (did not converge)
    All: OLS: 0.11318
```
