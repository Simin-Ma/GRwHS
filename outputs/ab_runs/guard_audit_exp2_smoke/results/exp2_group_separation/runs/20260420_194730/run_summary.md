# EXP2 Run Summary

- Timestamp: `20260420_194730`
- Run directory: `ab_runs\guard_audit_exp2_smoke\results\exp2_group_separation\runs\20260420_194730`

## Output Files
- `fig2a_method_comparison`: `ab_runs\guard_audit_exp2_smoke\figures\fig2a_method_comparison.png`
- `fig2b_kappa_by_group`: `ab_runs\guard_audit_exp2_smoke\figures\fig2b_kappa_by_group.png`
- `kappa_realizations`: `ab_runs\guard_audit_exp2_smoke\results\exp2_group_separation\kappa_realizations.csv`
- `meta`: `ab_runs\guard_audit_exp2_smoke\results\exp2_group_separation\exp2_meta.json`
- `raw`: `ab_runs\guard_audit_exp2_smoke\results\exp2_group_separation\raw_results.csv`
- `summary`: `ab_runs\guard_audit_exp2_smoke\results\exp2_group_separation\summary.csv`
- `table`: `ab_runs\guard_audit_exp2_smoke\tables\table_group_separation.csv`

## Compact Summary Table
| method | n_rows | n_converged | converged_rate | null_group_mse | null_group_mse_std | signal_group_mse | signal_group_mse_std | mse_overall | group_auroc | group_auroc_std | lpd_test | lpd_test_std | n_effective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RHS | 1 | 1 | 1 | 0.108086 | nan | 0.115399 | nan | 0.00605712 | 1 | nan | -1.43846 | nan | 1 |

## Analyzer Findings
### Finding 1
```text
  MSE ranking (lower=better):
    RHS: 0.0061
  AUROC ranking (higher=better):
    RHS: 1.000
  GR_RHS -- MSE: not found in results   AUROC: not found   LPD: not found
```
