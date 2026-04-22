# EXP3 Run Summary

- Timestamp: `20260421_153358`
- Run directory: `ab_runs\exp3_gigg_mmle_patch_eval_20260421_151749\laptop_r3_retry_default\results\exp3_linear_benchmark\runs\20260421_153358`

## Output Files
- `fig3a_mse_by_signal`: `ab_runs\exp3_gigg_mmle_patch_eval_20260421_151749\laptop_r3_retry_default\figures\fig3a_mse_by_signal.png`
- `fig3b_lpd_by_signal`: `ab_runs\exp3_gigg_mmle_patch_eval_20260421_151749\laptop_r3_retry_default\figures\fig3b_lpd_by_signal.png`
- `fig3c_null_signal_scatter`: `ab_runs\exp3_gigg_mmle_patch_eval_20260421_151749\laptop_r3_retry_default\figures\fig3c_null_signal_scatter.png`
- `meta`: `ab_runs\exp3_gigg_mmle_patch_eval_20260421_151749\laptop_r3_retry_default\results\exp3_linear_benchmark\exp3_meta.json`
- `raw`: `ab_runs\exp3_gigg_mmle_patch_eval_20260421_151749\laptop_r3_retry_default\results\exp3_linear_benchmark\raw_results.csv`
- `summary`: `ab_runs\exp3_gigg_mmle_patch_eval_20260421_151749\laptop_r3_retry_default\results\exp3_linear_benchmark\summary.csv`
- `table`: `ab_runs\exp3_gigg_mmle_patch_eval_20260421_151749\laptop_r3_retry_default\tables\table_linear_benchmark.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | rho_within | rho_between | target_snr | n_reps_total | n_reps_ok | n_reps_converged | mse_null | mse_signal | mse_overall | lpd_test | coverage_95 | avg_ci_length | kappa_null_mean | kappa_signal_mean | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GIGG_MMLE | 90 | 63 | 0.7 | 0.45 | 0.14 | 1.2 | 3 | 2.1 | 2.1 | 0.0645185 | 0.204026 | 0.0938382 | -2.14332 | 0.292381 | 0.149484 | nan | nan | 2.1 |

## Analyzer Findings
### Finding 1
```text
  Methods present in summary: ['GIGG_MMLE']
  NOTE: ['GR_RHS', 'RHS'] absent from summary (likely did not converge -- check logs)
  signal=boundary: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.03568
  signal=concentrated: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.22715
  signal=distributed: GR_RHS not in results (did not converge)
    All: GIGG_MMLE: 0.01183
```
