# Light High-Dimensional Bayesian Performance Comparison

This note provides a lightweight performance comparison for the four Bayesian
methods under the current formal high-dimensional benchmark protocol:

- `GR_RHS`
- `RHS`
- `GIGG_MMLE`
- `GHS_plus`

It is derived from:

- [unified all-four summary](/d:/FilesP/GR-RHS/docs/highdim_bayesian_all4_unified_summary.md)

The goal here is not to restate every metric, but to make the practical tradeoff
clear for paper writing and model-selection discussion.

## Common benchmark standard

All comparisons below are based on:

- three `p = 500` high-dimensional settings
- two validated repeats
- four chains
- isolated fresh-process execution
- formal convergence target `R-hat <= 1.01`

So this is a comparison among methods that already satisfy the same basic
posterior-stability requirement.

## Quick read

- If you want the best coefficient recovery: choose `GR_RHS`
- If you want the fastest main-line Bayesian method with acceptable convergence: choose `GIGG_MMLE`
- If you want a classical horseshoe-style Bayesian baseline that now also passes the high-dimensional convergence gate: `RHS`
- If you want the fastest overall converged Bayesian method and are willing to accept clearly worse coefficient MSE: `GHS_plus`

## Aggregate comparison

| Method | Convergence status | Runtime profile | Accuracy profile | Operational note |
|---|---|---|---|---|
| `GR_RHS` | stable in both repeats | slowest | best overall | strongest accuracy-first method |
| `RHS` | stable in both repeats | slow | strong, but weaker than `GR_RHS` | more often needs retry |
| `GIGG_MMLE` | stable in both repeats | fast | moderate | strongest speed-first main-line method |
| `GHS_plus` | stable in both repeats | fastest | weak | convergence is clean, but MSE is much worse |

## Mean wall-clock time across the three settings

These are approximate two-repeat means from the formal runs:

| Method | Typical wall time per setting |
|---|---:|
| `GR_RHS` | about `395s` to `538s` |
| `RHS` | about `350s` to `523s` |
| `GIGG_MMLE` | about `154s` to `223s` |
| `GHS_plus` | about `7s` to `8s` |

## Mean coefficient MSE across the three settings

| Method | Typical mean MSE range |
|---|---:|
| `GR_RHS` | about `0.00211` to `0.00216` |
| `RHS` | about `0.00207` to `0.00368` |
| `GIGG_MMLE` | about `0.00215` to `0.00451` |
| `GHS_plus` | about `0.01866` to `0.02827` |

## What this means in practice

There are really two different selection questions here.

### 1. Which method is the best high-dimensional Bayesian method overall?

For the current synthetic high-dimensional benchmark, that answer is still
`GR_RHS`, because it gives the strongest coefficient recovery while also meeting
the formal convergence requirement.

### 2. Which method is the fastest converged Bayesian method?

That answer depends on whether you mean:

- fastest among the main high-dimensional comparison roster: `GIGG_MMLE`
- fastest among all converged Bayesian methods: `GHS_plus`

The reason this distinction matters is that `GHS_plus` achieves its speed with
much worse coefficient error, so it should not be treated as the main speed
winner in an accuracy-aware benchmark narrative.

## Recommended paper wording

A clean way to describe the current state is:

> Under the formal high-dimensional benchmark protocol, all four Bayesian
> methods achieved repeat-level convergence-qualified posterior inference on the
> three \(p=500\) synthetic settings. GR-RHS delivered the best coefficient
> recovery, GIGG-MMLE was the fastest method among the main high-dimensional
> comparison roster, RHS remained viable but more retry-prone, and Grouped
> Horseshoe+ was also convergence-stable and extremely fast, though with much
> weaker coefficient-recovery accuracy.

## Final takeaway

The main milestone is now complete:

- we no longer have a high-dimensional Bayesian comparison where some methods
  fail the basic convergence gate;
- we now have a clean accuracy-speed spectrum among four methods that all pass
  the same formal stability standard;
- future discussion can focus on performance tradeoffs rather than on whether
  the posterior sampler itself is trustworthy.
