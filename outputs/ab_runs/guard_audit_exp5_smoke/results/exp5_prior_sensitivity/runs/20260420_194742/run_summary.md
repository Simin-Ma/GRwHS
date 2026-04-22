# EXP5 Run Summary

- Timestamp: `20260420_194742`
- Run directory: `ab_runs\guard_audit_exp5_smoke\results\exp5_prior_sensitivity\runs\20260420_194742`

## Output Files
- `fig5_prior_sensitivity`: `ab_runs\guard_audit_exp5_smoke\figures\fig5_prior_sensitivity.png`
- `fig5b_kappa_separation`: `ab_runs\guard_audit_exp5_smoke\figures\fig5b_kappa_separation.png`
- `meta`: `ab_runs\guard_audit_exp5_smoke\results\exp5_prior_sensitivity\exp5_meta.json`
- `raw`: `ab_runs\guard_audit_exp5_smoke\results\exp5_prior_sensitivity\raw_results.csv`
- `summary`: `ab_runs\guard_audit_exp5_smoke\results\exp5_prior_sensitivity\summary.csv`
- `table`: `ab_runs\guard_audit_exp5_smoke\tables\table_prior_sensitivity.csv`

## Compact Summary Table
| alpha_kappa | beta_kappa | setting_id | mse_null | mse_signal | group_auroc | kappa_null_mean | kappa_signal_mean | kappa_null_prob_gt_0_1 | n_effective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2 | 2 | 0.00462341 | 0.0146681 | 1 | 0.381995 | 0.706442 | 0.8275 | 1 |

## Analyzer Findings
### Finding 1
```text
  Prior grid: (alpha_kappa, beta_kappa)
  Setting 2 (1 prior configs):
    AUROC range:      [1.000, 1.000]  (stable)
    MSE_signal range: [0.01467, 0.01467]  CV=0.00  (robust)
```
