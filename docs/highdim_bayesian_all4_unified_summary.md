# Unified High-Dimensional Bayesian Summary

This note unifies the current formal high-dimensional Bayesian benchmark status
for all four Bayesian methods used in this project:

- `GR_RHS`
- `RHS`
- `GIGG_MMLE`
- `GHS_plus`

Source runs:

- main roster `r=1`: [run_summary.json](/d:/FilesP/GR-RHS/tmp/highdim_bayes_isolated_formal_r1/run_summary.json)
- main roster `r=2`: [run_summary.json](/d:/FilesP/GR-RHS/tmp/highdim_bayes_isolated_formal_r2/run_summary.json)
- `GHS_plus` supplement `r=1`: [run_summary.json](/d:/FilesP/GR-RHS/tmp/highdim_ghs_plus_isolated_formal_r1_v2/run_summary.json)
- `GHS_plus` supplement `r=2`: [run_summary.json](/d:/FilesP/GR-RHS/tmp/highdim_ghs_plus_isolated_formal_r2_v2/run_summary.json)

Common protocol:

- settings:
  - `hd_setting_1_classical_anchor`
  - `hd_setting_2_single_mode`
  - `hd_setting_3_multimode_showcase`
- fresh process per case via `--isolated`
- `4` chains
- formal convergence target: `R-hat <= 1.01`

## Bottom line

Under the current formal high-dimensional protocol, all four Bayesian methods
now have repeat-level convergence-qualified evidence on all three settings.

- `GR_RHS`: stable and most accurate overall, but slowest
- `RHS`: stable, but more retry-prone and operationally less smooth
- `GIGG_MMLE`: stable and much faster than HMC-based competitors
- `GHS_plus`: stable and fastest overall, but clearly weakest on coefficient MSE

## Repeat-level convergence status

| Method | Repeat 1 | Repeat 2 | High-dimensional status |
|---|---:|---:|---|
| `GR_RHS` | 3/3 converged | 3/3 converged | stable |
| `RHS` | 3/3 converged | 3/3 converged | stable, but retry-prone |
| `GIGG_MMLE` | 3/3 converged | 3/3 converged | stable |
| `GHS_plus` | 3/3 converged | 3/3 converged | stable |

## Two-repeat aggregate by setting and method

| Setting | Method | n repeats | n converged | Mean R-hat max | Mean wall seconds | Mean MSE overall | Mean attempts |
|---|---|---:|---:|---:|---:|---:|---:|
| hd_setting_1_classical_anchor | GR_RHS | 2 | 2 | 1.004576 | 537.947 | 0.002110 | 1.0 |
| hd_setting_1_classical_anchor | RHS | 2 | 2 | 1.006985 | 349.582 | 0.002073 | 1.5 |
| hd_setting_1_classical_anchor | GIGG_MMLE | 2 | 2 | 1.005626 | 153.824 | 0.002151 | 1.0 |
| hd_setting_1_classical_anchor | GHS_plus | 2 | 2 | 1.006932 | 6.823 | 0.018662 | 1.0 |
| hd_setting_2_single_mode | GR_RHS | 2 | 2 | 1.004262 | 410.351 | 0.002164 | 1.0 |
| hd_setting_2_single_mode | RHS | 2 | 2 | 1.008838 | 405.962 | 0.003678 | 1.5 |
| hd_setting_2_single_mode | GIGG_MMLE | 2 | 2 | 1.006750 | 223.326 | 0.004514 | 1.0 |
| hd_setting_2_single_mode | GHS_plus | 2 | 2 | 1.008886 | 7.697 | 0.028266 | 1.0 |
| hd_setting_3_multimode_showcase | GR_RHS | 2 | 2 | 1.004638 | 394.984 | 0.002152 | 1.0 |
| hd_setting_3_multimode_showcase | RHS | 2 | 2 | 1.007070 | 522.943 | 0.002556 | 2.0 |
| hd_setting_3_multimode_showcase | GIGG_MMLE | 2 | 2 | 1.006059 | 223.088 | 0.003645 | 1.0 |
| hd_setting_3_multimode_showcase | GHS_plus | 2 | 2 | 1.008029 | 7.835 | 0.027651 | 1.0 |

## Practical ranking

If the goal is coefficient recovery under the current `p=500` high-dimensional
synthetic design:

1. `GR_RHS` is the strongest accuracy-first choice.
2. `RHS` is still competitive, but less operationally stable than `GR_RHS`.
3. `GIGG_MMLE` is the main speed-first Bayesian method.
4. `GHS_plus` is now convergence-stable and extremely fast, but not
   competitive on coefficient MSE relative to the other three.

If the goal is only to satisfy high-dimensional posterior stability under the
formal benchmark protocol, then all four Bayesian methods currently pass.

## Interpretation

The important project milestone is no longer just that `GR_RHS` can be made to
work in high dimension. The stronger statement is now available:

- the full Bayesian comparison set is high-dimensionally viable under a common
  convergence standard;
- the methods differ mainly in the accuracy-speed tradeoff, not in whether they
  fail basic posterior stability;
- `GHS_plus` should still be treated as a supplement line in the paper's
  high-dimensional benchmark narrative, because its predictive and coefficient
  error profile is much weaker than the main three-method roster even though
  its convergence behavior is now clean.
