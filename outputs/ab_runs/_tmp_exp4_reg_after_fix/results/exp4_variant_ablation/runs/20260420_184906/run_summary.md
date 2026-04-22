# EXP4 Run Summary

- Timestamp: `20260420_184906`
- Run directory: `ab_runs\_tmp_exp4_reg_after_fix\results\exp4_variant_ablation\runs\20260420_184906`

## Output Files
- `fig4a_tau_scatter`: `ab_runs\_tmp_exp4_reg_after_fix\figures\fig4a_tau_scatter.png`
- `fig4b_mse_normalized`: `ab_runs\_tmp_exp4_reg_after_fix\figures\fig4b_mse_normalized.png`
- `meta`: `ab_runs\_tmp_exp4_reg_after_fix\results\exp4_variant_ablation\exp4_meta.json`
- `raw`: `ab_runs\_tmp_exp4_reg_after_fix\results\exp4_variant_ablation\raw_results.csv`
- `summary`: `ab_runs\_tmp_exp4_reg_after_fix\results\exp4_variant_ablation\summary.csv`
- `table`: `ab_runs\_tmp_exp4_reg_after_fix\tables\table_variant_ablation.csv`

## Compact Summary Table
| variant | p0_true | mse_null | mse_signal | mse_overall | tau0_oracle | tau_post_mean | tau_ratio_to_oracle | kappa_null_mean | kappa_signal_mean | n_effective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibrated | 5 | 0.000242165 | 0.00377706 | 0.000595655 | 0.0111111 | 0.192637 | 17.3373 | 0.273885 | 0.780268 | 1 |

## Analyzer Findings
### Finding 1
```text
  p0 = true number of active groups
  p0=5: calibrated MSE=0.00060  oracle MSE=nan  tau_post/tau_oracle=17.337
```
