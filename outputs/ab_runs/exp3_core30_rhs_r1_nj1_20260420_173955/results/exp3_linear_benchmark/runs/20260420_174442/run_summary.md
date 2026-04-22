# EXP3 Run Summary

- Timestamp: `20260420_174442`
- Run directory: `ab_runs\exp3_core30_rhs_r1_nj1_20260420_173955\results\exp3_linear_benchmark\runs\20260420_174442`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3_core30_rhs_r1_nj1_20260420_173955\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3_core30_rhs_r1_nj1_20260420_173955\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3_core30_rhs_r1_nj1_20260420_173955\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3_core30_rhs_r1_nj1_20260420_173955\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3_core30_rhs_r1_nj1_20260420_173955\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3_core30_rhs_r1_nj1_20260420_173955\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\exp3_core30_rhs_r1_nj1_20260420_173955\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RHS | 30 | 25 | 0.833333 | 0.45 | 0.14 | 1.2 | 1 | 0.833333 | 0.833333 | 0.024982 | 0.0751411 | 0.0340697 | -2.21618 | 0.9808 | 0.714507 | nan | nan | 0.833333 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['RHS']
  NOTE: ['GR_RHS'] absent from summary (likely did not converge -- check logs)
  signal=boundary: GR_RHS not in results (did not converge)
    All: RHS: 0.03009
  signal=concentrated: GR_RHS not in results (did not converge)
    All: RHS: 0.05965
  signal=distributed: GR_RHS not in results (did not converge)
    All: RHS: 0.00188
```
