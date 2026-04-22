# EXP3 Run Summary

- Timestamp: `20260421_183132`
- Run directory: `ab_runs\exp3_rwgt_rb_smoke_20260421_183128\results\exp3_linear_benchmark\runs\20260421_183132`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3_rwgt_rb_smoke_20260421_183128\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3_rwgt_rb_smoke_20260421_183128\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3_rwgt_rb_smoke_20260421_183128\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3_rwgt_rb_smoke_20260421_183128\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3_rwgt_rb_smoke_20260421_183128\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3_rwgt_rb_smoke_20260421_183128\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\exp3_rwgt_rb_smoke_20260421_183128\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLS | 24 | 24 | 1 | 0.4875 | 0.1 | 1.25 | 1 | 1 | 1 | 0.287826 | 0.270774 | 0.29915 | -3.21634 | nan | nan | nan | nan | 1 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['OLS']
  NOTE: ['GR_RHS', 'RHS'] absent from summary (likely did not converge -- check logs)
  signal=boundary: GR_RHS not in results (did not converge)
    All: OLS: 0.05654
  signal=concentrated: GR_RHS not in results (did not converge)
    All: OLS: 0.66645
  signal=distributed: GR_RHS not in results (did not converge)
    All: OLS: 0.09359
```
