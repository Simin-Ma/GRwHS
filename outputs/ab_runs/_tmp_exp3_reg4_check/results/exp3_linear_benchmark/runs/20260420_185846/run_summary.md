# EXP3 Run Summary

- Timestamp: `20260420_185846`
- Run directory: `ab_runs\_tmp_exp3_reg4_check\results\exp3_linear_benchmark\runs\20260420_185846`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\_tmp_exp3_reg4_check\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\_tmp_exp3_reg4_check\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\_tmp_exp3_reg4_check\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\_tmp_exp3_reg4_check\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\_tmp_exp3_reg4_check\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\_tmp_exp3_reg4_check\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\_tmp_exp3_reg4_check\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RHS | 1 | 1 | 1 | 0.3 | 0.1 | 1 | 1 | 1 | 1 | 0.0324185 | 0.0897443 | 0.0553488 | -2.91168 | 0.98 | 0.932977 | nan | nan | 1 |
| OLS | 1 | 1 | 1 | 0.3 | 0.1 | 1 | 1 | 1 | 1 | 0.212506 | 0.193951 | 0.205084 | -4.29111 | nan | nan | nan | nan | 1 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['OLS', 'RHS']
  NOTE: ['GR_RHS'] absent from summary (likely did not converge -- check logs)
  signal=concentrated: GR_RHS not in results (did not converge)
    All: RHS: 0.05535    OLS: 0.20508
```
