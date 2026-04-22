# EXP4 Run Summary

- Timestamp: `20260420_194939`
- Run directory: `ab_runs\guard_audit_exp4_smoke\results\exp4_variant_ablation\runs\20260420_194939`

## Output Files
- `fig4a_tau_scatter`: `ab_runs\guard_audit_exp4_smoke\figures\fig4a_tau_scatter.png`
- `fig4b_mse_normalized`: `ab_runs\guard_audit_exp4_smoke\figures\fig4b_mse_normalized.png`
- `meta`: `ab_runs\guard_audit_exp4_smoke\results\exp4_variant_ablation\exp4_meta.json`
- `raw`: `ab_runs\guard_audit_exp4_smoke\results\exp4_variant_ablation\raw_results.csv`
- `summary`: `ab_runs\guard_audit_exp4_smoke\results\exp4_variant_ablation\summary.csv`
- `table`: `ab_runs\guard_audit_exp4_smoke\tables\table_variant_ablation.csv`

## Compact Summary Table
| variant | p0_true | mse_null | mse_signal | mse_overall | tau0_oracle | tau_post_mean | tau_ratio_to_oracle | kappa_null_mean | kappa_signal_mean | n_effective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_10x | 5 | 0.00027194 | 0.00351443 | 0.000596189 | 0.0111111 | 0.241578 | 21.742 | 0.25507 | 0.771419 | 1 |
| calibrated | 5 | 0.000219998 | 0.00406008 | 0.000604006 | 0.0111111 | 0.194134 | 17.4721 | 0.266249 | 0.767996 | 1 |

## Analyzer Findings
### Finding 1
```text
  p0 = true number of active groups
  p0=5: calibrated MSE=0.00060  oracle MSE=nan  tau_post/tau_oracle=17.472
```
