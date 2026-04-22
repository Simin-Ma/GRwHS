# EXP3 Run Summary

- Timestamp: `20260420_194220`
- Run directory: `ab_runs\exp3_rhs_lasso_ols_core30_r1_fixed_20260420_194021\results\exp3_linear_benchmark\runs\20260420_194220`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3_rhs_lasso_ols_core30_r1_fixed_20260420_194021\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3_rhs_lasso_ols_core30_r1_fixed_20260420_194021\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3_rhs_lasso_ols_core30_r1_fixed_20260420_194021\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3_rhs_lasso_ols_core30_r1_fixed_20260420_194021\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3_rhs_lasso_ols_core30_r1_fixed_20260420_194021\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3_rhs_lasso_ols_core30_r1_fixed_20260420_194021\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\exp3_rhs_lasso_ols_core30_r1_fixed_20260420_194021\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LASSO_CV | 30 | 30 | 1 | 0.45 | 0.14 | 1.2 | 1 | 1 | 1 | 0.0211681 | 0.108026 | 0.0344836 | -2.22029 | nan | nan | nan | nan | 1 |
| RHS | 30 | 28 | 0.933333 | 0.45 | 0.14 | 1.2 | 1 | 0.933333 | 0.933333 | 0.0269568 | 0.0896129 | 0.0360198 | -2.23962 | 0.977143 | 0.71867 | nan | nan | 0.933333 |
| OLS | 30 | 30 | 1 | 0.45 | 0.14 | 1.2 | 1 | 1 | 1 | 0.327603 | 0.341148 | 0.322923 | -3.4476 | nan | nan | nan | nan | 1 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['LASSO_CV', 'OLS', 'RHS']
  NOTE: ['GR_RHS'] absent from summary (likely did not converge -- check logs)
  signal=boundary: GR_RHS not in results (did not converge)
    All: LASSO_CV: 0.02626    RHS: 0.03057    OLS: 0.06460
  signal=concentrated: GR_RHS not in results (did not converge)
    All: RHS: 0.06302    LASSO_CV: 0.06317    OLS: 0.67637
  signal=distributed: GR_RHS not in results (did not converge)
    All: RHS: 0.00690    LASSO_CV: 0.00991    OLS: 0.09864
```
