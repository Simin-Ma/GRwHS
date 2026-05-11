# High-Dimensional Bayesian Benchmark Summary

This note summarizes the formal isolated-process high-dimensional Bayesian benchmark run stored in:

- [run_summary.json](/d:/FilesP/GR-RHS/tmp/highdim_bayes_isolated_formal_r1/run_summary.json)
- [GHS_plus supplement run summary](/d:/FilesP/GR-RHS/tmp/highdim_ghs_plus_isolated_formal_r1_v2/run_summary.json)

Protocol used for this summary:

- `repeat = 1`
- `settings = {hd_setting_1_classical_anchor, hd_setting_2_single_mode, hd_setting_3_multimode_showcase}`
- `main methods = {GR_RHS, RHS, GIGG_MMLE}`
- `supplement method = {GHS_plus}`
- each case run in a fresh process via `--isolated`
- `timeout_seconds = 1800`
- convergence target: `R-hat <= 1.01`

## Main takeaways

- All three Bayesian methods produced convergence-qualified results on all three high-dimensional settings in this isolated formal run.
- The high-dimensional `GHS_plus` supplement line also produced convergence-qualified results on all three settings after the Gibbs budget was slightly strengthened.
- `GR_RHS` was the most accurate overall in this run, but also the slowest.
- `GIGG_MMLE` was the fastest by a wide margin, but had the weakest overall MSE.
- `RHS` was intermediate in predictive accuracy and runtime, but required two attempts on some settings.
- `GHS_plus` was much faster than the HMC-based methods, and in this run it satisfied the formal convergence line on all three settings with wall times around 7--8 seconds.

## Benchmark table

| Setting | Method | Converged | R-hat max | ESS min | Wall seconds | Runtime seconds | MSE overall | Attempts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hd\_setting\_1\_classical\_anchor | GR\_RHS | True | 1.004044 | 463.627 | 661.531 | 661.170 | 0.002396 | 1 |
| hd\_setting\_1\_classical\_anchor | RHS | True | 1.009225 | 878.429 | 488.492 | 317.833 | 0.001689 | 2 |
| hd\_setting\_1\_classical\_anchor | GIGG\_MMLE | True | 1.005020 | 806.217 | 157.138 | 62.286 | 0.001757 | 1 |
| hd\_setting\_2\_single\_mode | GR\_RHS | True | 1.003858 | 1099.422 | 381.080 | 380.714 | 0.001881 | 1 |
| hd\_setting\_2\_single\_mode | RHS | True | 1.008434 | 619.485 | 239.286 | 228.049 | 0.003146 | 1 |
| hd\_setting\_2\_single\_mode | GIGG\_MMLE | True | 1.006509 | 729.507 | 221.592 | 70.323 | 0.005573 | 1 |
| hd\_setting\_3\_multimode\_showcase | GR\_RHS | True | 1.004701 | 904.312 | 419.184 | 418.806 | 0.002189 | 1 |
| hd\_setting\_3\_multimode\_showcase | RHS | True | 1.005745 | 673.000 | 531.398 | 269.575 | 0.002508 | 2 |
| hd\_setting\_3\_multimode\_showcase | GIGG\_MMLE | True | 1.006151 | 730.722 | 219.883 | 68.964 | 0.003848 | 1 |

## GHS_plus supplement table

| Setting | Method | Converged | R-hat max | ESS min | Wall seconds | Runtime seconds | MSE overall | Attempts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hd\_setting\_1\_classical\_anchor | GHS\_plus | True | 1.005638 | 1537.912 | 6.807 | 4.953 | 0.018678 | 1 |
| hd\_setting\_2\_single\_mode | GHS\_plus | True | 1.008115 | 1862.861 | 7.750 | 5.633 | 0.028391 | 1 |
| hd\_setting\_3\_multimode\_showcase | GHS\_plus | True | 1.007130 | 1908.216 | 7.838 | 5.802 | 0.028634 | 1 |

## Readout by method

- `GR_RHS`
  - strongest overall MSE profile across the three settings
  - clean convergence margins throughout
  - highest runtime cost

- `RHS`
  - competitive MSE, especially on `hd_setting_1_classical_anchor`
  - slower than `GIGG_MMLE`, faster than `GR_RHS` in runtime-only terms
  - not as stable operationally: some settings needed `attempts_used = 2`

- `GIGG_MMLE`
  - fastest method in this benchmark
  - convergence remained acceptable across all three settings in the isolated run
  - predictive error was consistently worse than `GR_RHS` and usually worse than `RHS`

- `GHS_plus`
  - now explicitly validated on the same three high-dimensional settings as a supplement line
  - very fast under the Gibbs-based high-dimensional route and comfortably within the formal convergence threshold
  - coefficient MSE in this run was materially weaker than the main three-method high-dimensional roster, so it should be interpreted as a stable supplement rather than a main accuracy competitor

## Practical interpretation

For the formal high-dimensional synthetic benchmark line, the isolated-process protocol is the one to trust. In this run it gives a clean story:

- `GR_RHS` is the strongest accuracy-first Bayesian method.
- `GIGG_MMLE` is the speed-first Bayesian method.
- `RHS` remains viable, but its operational stability is weaker than the other two, even when it ultimately converges.
- `GHS_plus` now has a clear place in the story as a high-dimensional supplement method: fast and convergence-stable under the exact Gibbs route, but not competitive with the main three methods on coefficient MSE in these \(p=500\) settings.

## Caveat

This summary is based on a single formal repeat. It is suitable for documenting the current high-dimensional pipeline state and for reporting one formal benchmark pass, but not yet for claiming repeat-level stability across many replications.
