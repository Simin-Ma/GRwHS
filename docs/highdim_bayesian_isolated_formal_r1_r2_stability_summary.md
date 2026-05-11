# High-Dimensional Bayesian Benchmark Stability Summary

This note combines the first two formal isolated-process high-dimensional
Bayesian benchmark repeats:

- [formal r1 summary](/d:/FilesP/GR-RHS/docs/highdim_bayesian_isolated_formal_r1_summary.md)
- [formal r1 run summary](/d:/FilesP/GR-RHS/tmp/highdim_bayes_isolated_formal_r1/run_summary.json)
- [formal r2 run summary](/d:/FilesP/GR-RHS/tmp/highdim_bayes_isolated_formal_r2/run_summary.json)
- [GHS_plus supplement r1 run summary](/d:/FilesP/GR-RHS/tmp/highdim_ghs_plus_isolated_formal_r1_v2/run_summary.json)

Both runs use the same protocol:

- `settings = {hd_setting_1_classical_anchor, hd_setting_2_single_mode, hd_setting_3_multimode_showcase}`
- `main methods = {GR_RHS, RHS, GIGG_MMLE}`
- `supplement method = {GHS_plus}` with a validated `r=1` isolated run
- fresh process per case via `--isolated`
- `timeout_seconds = 1800`
- convergence target `R-hat <= 1.01`

## Main stability takeaways

- Across `r=1` and `r=2`, all three Bayesian methods converged on all three high-dimensional settings.
- The `GHS_plus` supplement line also converged on all three settings in the current validated isolated `r=1` run.
- `GR_RHS` was the most accurate method on average across the three settings.
- `GIGG_MMLE` remained the fastest method by a large margin across both repeats.
- `RHS` remained viable, but it was the least operationally stable of the three Bayesian methods because it more often needed a second attempt.
- `GHS_plus` was the fastest validated Bayesian route overall, but with substantially weaker coefficient MSE than the main three-method roster.

## Per-repeat summary

| Repeat | Setting | Method | Converged | R-hat max | ESS min | Wall seconds | Runtime seconds | MSE overall | Attempts |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | hd\_setting\_1\_classical\_anchor | GR\_RHS | True | 1.004044 | 463.627 | 661.531 | 661.170 | 0.002396 | 1 |
| 1 | hd\_setting\_1\_classical\_anchor | RHS | True | 1.009225 | 878.429 | 488.492 | 317.833 | 0.001689 | 2 |
| 1 | hd\_setting\_1\_classical\_anchor | GIGG\_MMLE | True | 1.005020 | 806.217 | 157.138 | 62.286 | 0.001757 | 1 |
| 1 | hd\_setting\_2\_single\_mode | GR\_RHS | True | 1.003858 | 1099.422 | 381.080 | 380.714 | 0.001881 | 1 |
| 1 | hd\_setting\_2\_single\_mode | RHS | True | 1.008434 | 619.485 | 239.286 | 228.049 | 0.003146 | 1 |
| 1 | hd\_setting\_2\_single\_mode | GIGG\_MMLE | True | 1.006509 | 729.507 | 221.592 | 70.323 | 0.005573 | 1 |
| 1 | hd\_setting\_3\_multimode\_showcase | GR\_RHS | True | 1.004701 | 904.312 | 419.184 | 418.806 | 0.002189 | 1 |
| 1 | hd\_setting\_3\_multimode\_showcase | RHS | True | 1.005745 | 673.000 | 531.398 | 269.575 | 0.002508 | 2 |
| 1 | hd\_setting\_3\_multimode\_showcase | GIGG\_MMLE | True | 1.006151 | 730.722 | 219.883 | 68.964 | 0.003848 | 1 |
| 2 | hd\_setting\_1\_classical\_anchor | GR\_RHS | True | 1.005108 | 653.926 | 414.362 | 413.991 | 0.001825 | 1 |
| 2 | hd\_setting\_1\_classical\_anchor | RHS | True | 1.004744 | 978.421 | 210.672 | 202.275 | 0.002458 | 1 |
| 2 | hd\_setting\_1\_classical\_anchor | GIGG\_MMLE | True | 1.006233 | 833.341 | 150.510 | 59.168 | 0.002544 | 1 |
| 2 | hd\_setting\_2\_single\_mode | GR\_RHS | True | 1.004667 | 912.450 | 439.622 | 439.213 | 0.002446 | 1 |
| 2 | hd\_setting\_2\_single\_mode | RHS | True | 1.009243 | 757.203 | 572.637 | 322.611 | 0.004210 | 2 |
| 2 | hd\_setting\_2\_single\_mode | GIGG\_MMLE | True | 1.006990 | 737.103 | 225.059 | 70.356 | 0.003456 | 1 |
| 2 | hd\_setting\_3\_multimode\_showcase | GR\_RHS | True | 1.004575 | 882.115 | 370.784 | 370.413 | 0.002115 | 1 |
| 2 | hd\_setting\_3\_multimode\_showcase | RHS | True | 1.008394 | 593.946 | 514.488 | 271.139 | 0.002605 | 2 |
| 2 | hd\_setting\_3\_multimode\_showcase | GIGG\_MMLE | True | 1.005967 | 721.466 | 226.293 | 70.287 | 0.003442 | 1 |

## Two-repeat aggregate summary

| Setting | Method | n repeats | n converged | Mean R-hat max | Mean wall seconds | Mean MSE overall | Mean attempts |
|---|---|---:|---:|---:|---:|---:|---:|
| hd\_setting\_1\_classical\_anchor | GR\_RHS | 2 | 2 | 1.004576 | 537.947 | 0.002110 | 1.0 |
| hd\_setting\_1\_classical\_anchor | RHS | 2 | 2 | 1.006985 | 349.582 | 0.002073 | 1.5 |
| hd\_setting\_1\_classical\_anchor | GIGG\_MMLE | 2 | 2 | 1.005626 | 153.824 | 0.002151 | 1.0 |
| hd\_setting\_2\_single\_mode | GR\_RHS | 2 | 2 | 1.004262 | 410.351 | 0.002164 | 1.0 |
| hd\_setting\_2\_single\_mode | RHS | 2 | 2 | 1.008838 | 405.962 | 0.003678 | 1.5 |
| hd\_setting\_2\_single\_mode | GIGG\_MMLE | 2 | 2 | 1.006750 | 223.326 | 0.004514 | 1.0 |
| hd\_setting\_3\_multimode\_showcase | GR\_RHS | 2 | 2 | 1.004638 | 394.984 | 0.002152 | 1.0 |
| hd\_setting\_3\_multimode\_showcase | RHS | 2 | 2 | 1.007070 | 522.943 | 0.002556 | 2.0 |
| hd\_setting\_3\_multimode\_showcase | GIGG\_MMLE | 2 | 2 | 1.006059 | 223.088 | 0.003645 | 1.0 |

## Interpretation

After two formal isolated repeats, the main qualitative pattern is stable:

- `GR_RHS` is the strongest accuracy-oriented method.
- `GIGG_MMLE` is the strongest speed-oriented method.
- `RHS` converges under the isolated protocol, but is less operationally stable because it more often requires retry.

The main limitation is still sample size at the benchmark level: two repeats are
enough to show an early stability pattern, but not enough to support strong
claims about repeat-level variance or ranking certainty.

## GHS_plus supplement note

`GHS_plus` is currently tracked as a high-dimensional supplement line rather
than as part of the original two-repeat main roster. Under the updated Gibbs
budget used by the isolated validation route, it converged on all three
settings in both `r=1` and `r=2`:

| Setting | Method | Repeat | Converged | R-hat max | ESS min | Wall seconds | Runtime seconds | MSE overall | Attempts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hd\_setting\_1\_classical\_anchor | GHS\_plus | 1 | True | 1.005638 | 1537.912 | 6.807 | 4.953 | 0.018678 | 1 |
| hd\_setting\_2\_single\_mode | GHS\_plus | 1 | True | 1.008115 | 1862.861 | 7.750 | 5.633 | 0.028391 | 1 |
| hd\_setting\_3\_multimode\_showcase | GHS\_plus | 1 | True | 1.007130 | 1908.216 | 7.838 | 5.802 | 0.028634 | 1 |
| hd\_setting\_1\_classical\_anchor | GHS\_plus | 2 | True | 1.008226 | 1483.787 | 6.838 | 4.951 | 0.018646 | 1 |
| hd\_setting\_2\_single\_mode | GHS\_plus | 2 | True | 1.009657 | 1575.482 | 7.643 | 5.625 | 0.028141 | 1 |
| hd\_setting\_3\_multimode\_showcase | GHS\_plus | 2 | True | 1.008929 | 1984.908 | 7.831 | 5.819 | 0.026669 | 1 |

This places `GHS_plus` in a different operational corner of the design space:
it is extremely fast and now convergence-stable on the formal high-dimensional
settings, but its coefficient-recovery error is much worse than `GR_RHS`,
`RHS`, and `GIGG_MMLE` in the current \(p=500\) synthetic benchmark.

With this supplement validation in place, the current high-dimensional
synthetic benchmark evidence supports the following concrete statement: under
the formal isolated-process protocol with four chains and convergence target
`R-hat <= 1.01`, all four Bayesian methods now have repeat-level convergence
evidence on all three high-dimensional settings:
`GR_RHS`, `RHS`, `GIGG_MMLE`, and `GHS_plus`.
