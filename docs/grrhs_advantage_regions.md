# GR-RHS Advantage Regions Log

This note records the main experiment evidence collected so far for where `GR-RHS`
shows clear advantage, where that advantage is partial, and where it does not
currently dominate. It is intended as a running lab log for paper writing and
future experiment design.

## Purpose

This document answers four questions:

1. In which data-generating regimes does `GR-RHS` reliably outperform `GIGG_MMLE`?
2. In which regimes does `GR-RHS` also beat `RHS`, not just `GIGG_MMLE`?
3. Under what convergence and budget settings are those claims supported?
4. Which result files should be treated as the current reference artifacts?

This file is the empirical evidence log. For benchmark design and method-stage
mechanism validation, see:

- [grrhs_benchmark_blueprint.md](/d:/FilesP/GR-RHS/docs/grrhs_benchmark_blueprint.md)
- [grrhs_mechanism_experiment_design.md](/d:/FilesP/GR-RHS/docs/grrhs_mechanism_experiment_design.md)

## Executive Summary

The strongest current conclusion is:

- `within_group_mixed` is the most stable dominance region for `GR-RHS`.
- Under high-budget, convergence-enforced runs, `GR-RHS` consistently beats both
  `RHS` and `GIGG_MMLE` on `mse_overall` and `mse_signal`, while maintaining
  near-perfect coverage.
- This stable dominance region is already verified at:
  `rho_within in {0.6, 0.7, 0.8, 0.9}`, `rho_between = 0.2`, and tested
  `n_train in {200, 300, 400, 500}` depending on the point.
- `GIGG_MMLE` is often much slower and substantially overconfident in these
  regimes, even when it converges.
- A new six-method extension against the full `Exp 3a` set now finds `10` fresh
  `within_group_mixed` points where `GR_RHS` is best on both `mse_overall` and
  `mse_signal` among `GR_RHS`, `RHS`, `GIGG_MMLE`, `GHS_plus`, `OLS`,
  `LASSO_CV`, while keeping `coverage_95 >= 0.95`.
- A new high-budget synthetic-region study now adds three paper-grade
  confirmation regions beyond the original `within_group_mixed` benchmark:
  `paired_decoy (rho_within=0.9, snr=1.0)`,
  `size_imbalance (rho_within=0.9, snr=0.35)`, and
  `size_imbalance (rho_within=0.9, snr=1.0)`.
- In those regions, `GR-RHS` again beats `GIGG_MMLE` on overall MSE while
  preserving much higher posterior coverage; the strongest case is
  `paired_decoy`, where `GIGG_MMLE` needs near-paper `10k+10k` iteration
  budgets just to reach acceptable convergence on most replicates.

Secondary conclusion:

- `half_dense` is a strong region where `GR-RHS` clearly dominates
  `GIGG_MMLE`, but `RHS` can remain competitive or slightly better on overall
  MSE at some points. So this is a strong `GR-RHS > GIGG_MMLE` region, but not
  yet a clean universal dominance region.

Classic-benchmark conclusion:

- Under a strict convergence-first six-method comparison, the cleanest
  classical-paper benchmark family is no longer the original tuned equal-group
  candidate, but a broader `equal-size + multimode active groups` region.
- The strongest current classical candidate that still survives after adding
  `OLS` and `LASSO_CV` is:
  `group_sizes=[10,10,10,10,10]`, `n=500`, `rho_within=0.6`,
  `rho_between=0.2`, with `3` active groups that follow different
  within-group signal modes.
- In that setting, all Bayesian methods converged in all `4/4` replicates, and
  `GR-RHS` remained the best method on `mse_overall`.
- A key negative result is that the earlier `2`-active-group classical
  candidate does **not** survive the full six-method check: after `4` repeated
  runs, `LASSO_CV` becomes the winner there.

Inference-oriented conclusion:

- In weak-identification settings, `GR-RHS` preserves uncertainty much better
  than `GIGG_MMLE`. There, the main advantage is not minimum MSE, but honest
  posterior uncertainty and coverage.

## Unified Convergence-First Configuration

The main convergence-first scans used the following sampler setup:

```text
chains = 4
warmup = 1000
post_warmup_draws = 1000
adapt_delta = 0.97
max_treedepth = 13
strict_adapt_delta = 0.995
strict_max_treedepth = 15
max_divergence_ratio = 0.005
rhat_threshold = 1.015
ess_threshold = 400
max_convergence_retries = 5
```

For `GIGG_MMLE`, retry budget scaling was enabled in the targeted high-budget
runs via:

```text
allow_budget_retry = True
retry_cap = 3
```

Interpretation:

- Claims in the dominance region below are based on `status=ok` and
  `converged=True`.
- When comparing methods directly, the most credible rows are the paired files
  that restrict to common converged replicates.

## Main Result Files

### High-value reference files

- Within-group mixed dominance boundary scan:
  [summary.csv](/d:/FilesP/GR-RHS/outputs/within_group_mixed_boundary_scan/summary.csv:1)
- Within-group mixed common-converged comparison:
  [paired_all3_converged.csv](/d:/FilesP/GR-RHS/outputs/within_group_mixed_boundary_scan/paired_all3_converged.csv:1)
- Six-method targeted extension, first batch:
  [dominance_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_sixway_region_scan_strong_r1/dominance_summary.csv:1)
- Six-method targeted extension, top-up batch:
  [dominance_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_sixway_region_scan_topup_r1/dominance_summary.csv:1)
- Six-method combined stable points:
  [stable_sixway_dominance_points.csv](/d:/FilesP/GR-RHS/outputs/grrhs_sixway_region_scan_combined/stable_sixway_dominance_points.csv:1)
- Earlier high-budget targeted scan:
  [summary.csv](/d:/FilesP/GR-RHS/outputs/custom_grrhs_convergence_first_budget/summary.csv:1)
- Earlier paired-converged comparison:
  [paired_converged_summary.csv](/d:/FilesP/GR-RHS/outputs/custom_grrhs_convergence_first_budget/paired_converged_summary.csv:1)
- First exploratory probe:
  [summary.csv](/d:/FilesP/GR-RHS/outputs/custom_grrhs_dominance_explore/summary.csv:1)
- Follow-up frontier probe:
  [summary.csv](/d:/FilesP/GR-RHS/outputs/custom_grrhs_frontier_followup/summary.csv:1)
- Final paper-grade synthetic region summary:
  [grrhs_paper_repro_final_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_final_summary.csv:1)
- Final `GR_RHS` vs `GIGG_MMLE` comparison table:
  [grrhs_paper_repro_final_gr_vs_gigg.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_final_gr_vs_gigg.csv:1)
- Classical-candidate six-method `4`-rep summary:
  [summary_all_methods.csv](/d:/FilesP/GR-RHS/outputs/grrhs_classic_candidates_paper/summary_all_methods.csv:1)
- Classical-candidate paper main table:
  [paper_table_main.md](/d:/FilesP/GR-RHS/outputs/grrhs_classic_candidates_paper/paper_tables/paper_table_main.md:1)
- Classical-candidate paper appendix table:
  [paper_table_appendix_full.md](/d:/FilesP/GR-RHS/outputs/grrhs_classic_candidates_paper/paper_tables/paper_table_appendix_full.md:1)
- Paired-decoy high-budget run:
  [confirm_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_paired_decoy/confirm_summary.csv:1)
- Size-imbalance high-budget run, weak signal:
  [confirm_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_size_imbalance_snr035/confirm_summary.csv:1)
- Size-imbalance weak-signal top-up run:
  [confirm_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_size_imbalance_snr035_topup/confirm_summary.csv:1)
- Size-imbalance high-budget run, moderate signal:
  [confirm_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_size_imbalance_snr10/confirm_summary.csv:1)

### Older supporting evidence

- Weak-ID pilot budget summary:
  [summary.csv](/d:/FilesP/GR-RHS/outputs/pilot_within_group_weakid_budget/results/exp3_weakid_pilot_budget_hi/summary.csv:1)
- Focused within-group mixed benchmark:
  [table_linear_benchmark.csv](/d:/FilesP/GR-RHS/outputs/focused_exp3_checks_v2/tables/exp3_within_group_mixed_focus/table_linear_benchmark.csv:1)
- Low-dimensional probe where both methods converge:
  [summary.csv](/d:/FilesP/GR-RHS/outputs/randomcoef_probe_grrhs_nuts_vs_gigg/lowdim_mixed/summary.csv:1)

## New Paper-Grade Synthetic Confirmation Regions

These runs were designed after the exploratory custom-region search to test
whether the newly discovered `GR-RHS` advantage regions survive under
substantially higher budgets, `20+` repeated fits, and convergence enforcement.

Common setup for these confirmation runs:

```text
confirm_repeats = 24
n_train = 220
n_test = 500
warmup = 500
post_warmup_draws = 500
ess_threshold = 150
max_convergence_retries = 2
GIGG iter_floor = iter_cap = 10000
```

Interpretation:

- these are no longer exploratory low-budget scans
- all summary values below come from the final merged high-budget artifacts
- the `size_imbalance, snr=0.35` case includes an additional top-up batch so
  that both `GR_RHS` and `GIGG_MMLE` exceed `20` converged replicates

### Final cross-region table

Source:
 [grrhs_paper_repro_final_gr_vs_gigg.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_final_gr_vs_gigg.csv:1)

| scenario | rho_within | snr | GR converged | GIGG converged | GR MSE | GIGG MSE | GIGG / GR MSE | GR coverage | GIGG coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `paired_decoy` | 0.9 | 1.0 | 24 | 23 | 0.00849 | 0.03756 | 4.43 | 0.9711 | 0.1944 |
| `size_imbalance` | 0.9 | 0.35 | 30 | 25 | 0.02047 | 0.02281 | 1.11 | 0.9627 | 0.2133 |
| `size_imbalance` | 0.9 | 1.0 | 24 | 24 | 0.00825 | 0.01361 | 1.65 | 0.9673 | 0.1657 |

Main takeaways:

- `GR-RHS` wins on overall MSE in all three newly confirmed regions.
- The coverage gap is much larger and more stable than the MSE gap.
- `GIGG_MMLE` remains dramatically more overconfident than `GR-RHS`, even after
  pushing it to near-paper iteration budgets.
- `GIGG_MMLE` also remains much slower: roughly `2.4x` to `4.1x` the runtime of
  `GR-RHS` across these three regions.

### Interpreting the GIGG gap carefully

These results should not be described too casually as a normal
"`GR-RHS` slightly beats `GIGG_MMLE`" pattern.

In many of the strongest `GR-RHS` regions, the empirical picture looks more
like structural or mechanism-level failure of `GIGG_MMLE` than a small
head-to-head disadvantage:

- `GIGG_MMLE` often attains extremely small `mse_null`
- but it does so together with much larger `mse_signal`
- and with drastically sub-nominal posterior coverage

So the most accurate interpretation is:

- these are regions where `GR-RHS` is much more robust to grouped structural
  stress
- and where `GIGG_MMLE` often becomes too concentrated or overconfident rather
  than merely "a bit worse"

This distinction matters for writing:

- the comparison is still valid as a performance comparison
- but the scientific explanation should emphasize robustness and mismatch, not
  just average superiority

### Region A: Paired-Decoy

Definition:

- one null group is coupled to an active group through an extra latent factor
- this creates a decoy group that is structurally similar to signal
- the scientific question is whether the method can keep the null group from
  being overactivated while still preserving signal

Source:
 [confirm_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_paired_decoy/confirm_summary.csv:1)

Key numbers:

- `GR_RHS`: `n_converged=24`, `mse_overall=0.00849`, `mse_signal=0.01865`,
  `mse_null=0.00340`, `coverage_95=0.9711`, `group_auroc=1.0000`,
  `runtime=11.93s`
- `RHS`: `24`, `0.00859`, `0.01847`, `0.00365`, `0.9514`, `1.0000`, `6.21s`
- `GIGG_MMLE`: `23`, `0.03756`, `0.11199`, `0.00034`, `0.1944`, `0.9903`,
  `40.72s`

Interpretation:

- This is the cleanest new `GR-RHS > GIGG_MMLE` story.
- `GIGG_MMLE` shrinks null coefficients extremely aggressively, which helps
  `mse_null`, but it pays for this with very large `mse_signal` and extremely
  poor coverage.
- `GR-RHS` remains accurate on both signal and null coordinates and keeps
  honest uncertainty.
- `RHS` is competitive on point estimation here, but `GR-RHS` is still slightly
  better on both `mse_overall` and coverage.

### Region B: Size-Imbalance, Weak Signal

Definition:

- group sizes are highly unequal
- smaller groups carry stronger signal while larger groups are mostly null
- the weak-signal case makes it easy to over-shrink the small active groups

Sources:
 [confirm_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_size_imbalance_snr035/confirm_summary.csv:1)
 [confirm_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_size_imbalance_snr035_topup/confirm_summary.csv:1)
 [grrhs_paper_repro_final_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_final_summary.csv:1)

Final merged numbers:

- `GR_RHS`: `n_converged=30`, `mse_overall=0.02047`, `mse_signal=0.06258`,
  `mse_null=0.00898`, `coverage_95=0.9627`, `group_auroc=0.9630`,
  `runtime=17.05s`
- `RHS`: `32`, `0.02266`, `0.06973`, `0.00982`, `0.9501`, `0.9757`, `4.58s`
- `GIGG_MMLE`: `25`, `0.02281`, `0.08092`, `0.00697`, `0.2133`, `0.9911`,
  `63.29s`

Interpretation:

- This is a moderate but real MSE win for `GR-RHS`, not just a coverage story.
- The biggest effect remains uncertainty calibration: `GR-RHS` is near-nominal,
  while `GIGG_MMLE` remains severely overconfident.
- `RHS` is again a serious competitor, but `GR-RHS` still improves both MSE and
  coverage in the merged high-budget summary.

### Region C: Size-Imbalance, Moderate Signal

Source:
 [confirm_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_size_imbalance_snr10/confirm_summary.csv:1)

Key numbers:

- `GR_RHS`: `n_converged=24`, `mse_overall=0.00825`, `mse_signal=0.02554`,
  `mse_null=0.00353`, `coverage_95=0.9673`, `group_auroc=1.0000`,
  `runtime=16.30s`
- `RHS`: `24`, `0.00923`, `0.02914`, `0.00379`, `0.9484`, `0.9954`, `6.29s`
- `GIGG_MMLE`: `24`, `0.01361`, `0.05979`, `0.00101`, `0.1657`, `1.0000`,
  `67.13s`

Interpretation:

- This is the most balanced size-imbalance result: all methods converge, and
  `GR-RHS` still wins on `mse_overall`, `mse_signal`, and coverage.
- `GIGG_MMLE` again shows the same pattern as in the other regions:
  exceptionally small null error, but much worse signal recovery and
  dramatically worse interval calibration.

## Updated Scientific Summary

The enlarged empirical conclusion after adding the new paper-grade confirmation
regions is:

- `within_group_mixed` remains the strongest full dominance region for
  `GR-RHS`, especially when the comparison requires beating both `RHS` and
  `GIGG_MMLE`
- beyond that benchmark, there is now strong high-budget evidence that
  `GR-RHS` also has a broader empirical advantage against `GIGG_MMLE` in
  `paired_decoy` and `size_imbalance`
- in those regions, the most stable pattern is lower overall error together
  with much higher posterior coverage

For benchmark-level interpretation and mechanism-first framing, see the two
linked design documents above.

## Stable Dominance Region

Definition used here:

- all three methods `GR_RHS`, `RHS`, `GIGG_MMLE` share converged replicates
- `GR_RHS` beats both competitors on `mse_overall`
- `GR_RHS` beats both competitors on `mse_signal`
- `GR_RHS` coverage remains at least `0.95`

Under that definition, the current verified stable dominance region is:

| scenario | rho_within | n_train | GR_RHS vs RHS | GR_RHS vs GIGG | stable |
|---|---:|---:|---|---|---|
| `wgmixed_rw060_n400` | 0.6 | 400 | wins | wins | yes |
| `wgmixed_rw070_n200` | 0.7 | 200 | wins | wins | yes |
| `wgmixed_rw070_n400` | 0.7 | 400 | wins | wins | yes |
| `wgmixed_rw080_n200` | 0.8 | 200 | wins | wins | yes |
| `wgmixed_rw080_n300` | 0.8 | 300 | wins | wins | yes |
| `wgmixed_rw080_n400` | 0.8 | 400 | wins | wins | yes |
| `wgmixed_rw080_n500` | 0.8 | 500 | wins | wins | yes |
| `wgmixed_rw090_n400` | 0.9 | 400 | wins | wins | yes |

Source:
 [paired_all3_converged.csv](/d:/FilesP/GR-RHS/outputs/within_group_mixed_boundary_scan/paired_all3_converged.csv:1)

### Concrete numbers

#### `wgmixed_rw060_n400`

- `GR_RHS`: `mse_overall=0.0271`, `mse_signal=0.0536`, `coverage_95=0.99`, `runtime=21.6s`
- `RHS`: `0.0288`, `0.0603`, `0.98`, `8.5s`
- `GIGG_MMLE`: `0.1071`, `0.1100`, `0.10`, `136.8s`

#### `wgmixed_rw070_n200`

- `GR_RHS`: `0.0566`, `0.1053`, `1.00`, `22.8s`
- `RHS`: `0.0578`, `0.1148`, `1.00`, `4.5s`
- `GIGG_MMLE`: `0.5502`, `0.6237`, `0.16`, `71.0s`

#### `wgmixed_rw080_n400`

- `GR_RHS`: `0.0644`, `0.1290`, `1.00`, `29.1s`
- `RHS`: `0.0666`, `0.1398`, `1.00`, `7.2s`
- `GIGG_MMLE`: `0.3361`, `0.4268`, `0.06`, `101.8s`

#### `wgmixed_rw080_n500`

- `GR_RHS`: `0.0344`, `0.0617`, `1.00`, `27.5s`
- `RHS`: `0.0364`, `0.0703`, `1.00`, `9.1s`
- `GIGG_MMLE`: `0.1640`, `0.1576`, `0.03`, `88.9s`

#### `wgmixed_rw090_n400`

- `GR_RHS`: `0.0454`, `0.0958`, `1.00`, `24.0s`
- `RHS`: `0.0465`, `0.1082`, `1.00`, `10.2s`
- `GIGG_MMLE`: `0.4555`, `0.4905`, `0.09`, `79.6s`

## Interpretation of the Within-Group Mixed Region

For benchmark and mechanism interpretation, see:

- [grrhs_benchmark_blueprint.md](/d:/FilesP/GR-RHS/docs/grrhs_benchmark_blueprint.md)
- [grrhs_mechanism_experiment_design.md](/d:/FilesP/GR-RHS/docs/grrhs_mechanism_experiment_design.md)

The empirical takeaway recorded here is narrower:

- `within_group_mixed` is the current best verified dominance region for
  `GR-RHS`
- this is no longer a one-point effect; it is a connected region across
  multiple correlation and sample-size settings under convergence filtering

## Diagnostic Follow-Up On Why GIGG_MMLE Fails Here

After observing that several of the strongest `GR-RHS` regions looked less like
a normal small advantage and more like severe `GIGG_MMLE` underperformance, a
targeted diagnostic follow-up was run in this workspace.

Main motivation:

- in the live calibration probe, several `GIGG_MMLE` fits showed many groups
  with `mmle_q_max = 4.0`
- this raised the question of whether the hard `q <= 4` ceiling was the main
  reason `GIGG_MMLE` was failing in these grouped stress settings

Relevant live probe artifact:

- [summary_live.csv](/d:/FilesP/GR-RHS/outputs/gigg_mmle_calibration_live_test/results/g10x5_snr10_rb20/summary_live.csv:1)
- [raw_results_live.csv](/d:/FilesP/GR-RHS/outputs/gigg_mmle_calibration_live_test/results/g10x5_snr10_rb20/raw_results_live.csv:1)

### What the live calibration probe showed

The probe strengthened the view that these are often mismatch regions rather
than ordinary mild losses:

- in `distributed` settings, `GIGG_MMLE` could achieve moderate point-estimate
  MSE while still collapsing coverage to roughly `0.15` to `0.34`
- in `concentrated` settings, both MSE and coverage could degrade badly
- some `distributed` rows showed repeated `q` values pinned at `4.0`

### Implementation note: the `q <= 4` cap is real

The current `GIGGRegression` implementation uses:

- `b_max = 4.0`
- `q_constraint_mode = "hard"`

and the MMLE update explicitly applies:

```text
q_new[gid] = min(max(est, b_floor), b_max)
```

Sources:

- [gigg_regression.py](/d:/FilesP/GR-RHS/simulation_project/src/core/models/gigg_regression.py:678)
- [gigg_regression.py](/d:/FilesP/GR-RHS/simulation_project/src/core/models/gigg_regression.py:705)
- [gigg_regression.py](/d:/FilesP/GR-RHS/simulation_project/src/core/models/gigg_regression.py:1342)
- [fit_gigg.py](/d:/FilesP/GR-RHS/simulation_project/src/experiments/methods/fit_gigg.py:23)

So boundary pile-up at `q = 4` is a real implementation-level phenomenon, not
just a reporting artifact.

### Quick test 1: changing `q_constraint_mode`

A small pilot was run on representative `Exp3`-style points comparing:

- `q_constraint_mode = hard`
- `q_constraint_mode = soft`
- `q_constraint_mode = none`

Artifact:

- [gigg_q_constraint_quicktest.csv](/d:/FilesP/GR-RHS/outputs/q_constraint_quicktest/gigg_q_constraint_quicktest.csv:1)

Main observations:

- `hard` and `none` produced essentially identical results on the tested cases
- this happens because the MMLE update still truncates at `b_max` before the
  later stabilization step
- `soft` changed the behavior, but not in a clearly beneficial way; it often
  drove `q` to very small values and did not materially repair calibration or
  convergence

Examples from the pilot:

- `concentrated, rho_within=0.9`:
  `hard` and `none` both gave `mse_overall=0.00090`, `coverage_95=0.36`,
  and `3` groups at `q >= 3.99`
- the same point under `soft` gave `mse_overall=0.01225`,
  `coverage_95=0.38`, and `q_max=0.033`
- `distributed, rho_within=0.95` stayed poor in all three modes:
  `hard`/`none` gave `mse_overall=7.2679`, `coverage_95=0.46`,
  while `soft` gave `7.2645`, `0.48`

Interpretation:

- simply changing `q_constraint_mode` is not enough
- in the current implementation, `q_constraint_mode="none"` does not remove the
  MMLE ceiling effect
- and the hard boundary at `4.0` is therefore not isolated by that test alone

### Quick test 2: increasing `b_max`

To test the boundary explanation more directly, a second pilot increased
`b_max` from `4` to `10`, `20`, and `100`.

Artifact:

- [gigg_bmax_quicktest.csv](/d:/FilesP/GR-RHS/outputs/q_constraint_quicktest/gigg_bmax_quicktest.csv:1)

Main observations:

- larger `b_max` values did allow `q` to move far beyond `4`
- but this did not materially improve the tested failure cases
- runtime usually increased and ESS often became worse

Representative results:

- `concentrated, rho_within=0.9`:
  `b_max=4` gave `mse_overall=0.05587`, `coverage_95=0.10`, `q_max=4.0`
- the same point at `b_max=100` gave `mse_overall=0.05518`,
  `coverage_95=0.10`, `q_max=25.64`
- `concentrated, rho_within=0.95`:
  `b_max=4` gave `mse_overall=0.02406`, `coverage_95=0.20`
- at `b_max=100`, this moved only to `0.02453`, `0.22`

Interpretation:

- the `q = 4` pile-up is a meaningful symptom of how `GIGG_MMLE` behaves in
  these regimes
- but removing or relaxing that ceiling did not, in these quick tests, repair
  the underlying failure pattern
- the evidence therefore points to a broader mechanism mismatch, not to a
  single cap-induced pathology

### Practical conclusion for the advantage-region story

- the visible `q = 4` boundary pile-up is a useful diagnostic clue
- but the current evidence does not support reducing the empirical gap to that
  single implementation detail
- for broader mechanism explanation, see the benchmark and mechanism design
  documents

## Six-Method Extension Against Exp 3a

The original dominance log focused on the three Bayesian/group-aware methods
that mattered most for the scientific mechanism claim:

- `GR_RHS`
- `RHS`
- `GIGG_MMLE`

To align the argument with the full `Exp 3a` benchmark roster, I ran a new
targeted `within_group_mixed + G10x5` extension against all six methods:

- `GR_RHS`, `RHS`, `GIGG_MMLE`, `GHS_plus`, `OLS`, `LASSO_CV`

Protocol used for this six-way extension:

- signal: `within_group_mixed`
- group config: `G10x5`
- methods: full `Exp 3a` six-method set
- target SNR fixed at `1.0`
- convergence gate: `status=ok && converged=True`
- `GR_RHS` retry budget increased to `max_convergence_retries=5`
- targeted seed: `20260425`
- each row below currently has `n_common_min_converged = 1`

Important interpretation note:

- These are targeted confirmation points, not a replacement for the higher-value
  three-method boundary map.
- Their role is narrower: show that the already-identified `within_group_mixed`
  advantage region is not only a `GR_RHS > RHS/GIGG_MMLE` story, but also
  survives when `GHS_plus`, `OLS`, and `LASSO_CV` are added back into the
  leaderboard.

Definition used for a six-way dominance point:

- all six methods converge for that setting
- `GR_RHS` has the smallest `mse_overall`
- `GR_RHS` has the smallest `mse_signal`
- `GR_RHS` keeps `coverage_95 >= 0.95`

Under that definition, the following `10` new settings qualify:

| env_id | rho_within | rho_between | n_train | GR_RHS mse_overall | GR_RHS mse_signal | coverage_95 | six-way stable |
|---|---:|---:|---:|---:|---:|---:|---|
| `RW080_RB015_SNR10_N250` | 0.80 | 0.15 | 250 | 0.0507 | 0.1081 | 0.98 | yes |
| `RW080_RB015_SNR10_N300` | 0.80 | 0.15 | 300 | 0.0615 | 0.0439 | 1.00 | yes |
| `RW080_RB015_SNR10_N350` | 0.80 | 0.15 | 350 | 0.0602 | 0.1210 | 0.98 | yes |
| `RW080_RB020_SNR10_N200` | 0.80 | 0.20 | 200 | 0.0992 | 0.2352 | 0.98 | yes |
| `RW080_RB020_SNR10_N300` | 0.80 | 0.20 | 300 | 0.0554 | 0.1094 | 0.98 | yes |
| `RW085_RB020_SNR10_N300` | 0.85 | 0.20 | 300 | 0.1171 | 0.1561 | 0.96 | yes |
| `RW085_RB020_SNR10_N350` | 0.85 | 0.20 | 350 | 0.0640 | 0.1462 | 1.00 | yes |
| `RW088_RB020_SNR10_N300` | 0.88 | 0.20 | 300 | 0.1671 | 0.2699 | 0.98 | yes |
| `RW090_RB020_SNR10_N300` | 0.90 | 0.20 | 300 | 0.0761 | 0.1695 | 1.00 | yes |
| `RW090_RB020_SNR10_N350` | 0.90 | 0.20 | 350 | 0.0607 | 0.1094 | 0.98 | yes |

Source:
 [stable_sixway_dominance_points.csv](/d:/FilesP/GR-RHS/outputs/grrhs_sixway_region_scan_combined/stable_sixway_dominance_points.csv:1)

### What The Six-Method Extension Adds

- It shows the dominance region persists after reintroducing `GHS_plus`, not
  just `RHS` and `GIGG_MMLE`.
- It shows `OLS` is never competitive in this region.
- It shows `LASSO_CV` can still be a serious local competitor, but there are now
  at least `10` concrete `within_group_mixed` points where `GR_RHS` outruns it
  on both overall and signal error.

### Where The Boundary Still Looks Real

The six-way extension also produced near-miss settings:

- `RW080_RB020_SNR10_N250`: `RHS` slightly wins `mse_overall`
- `RW082_RB020_SNR10_N300`: `LASSO_CV` wins `mse_overall`, `RHS` wins `mse_signal`
- `RW085_RB015_SNR10_N300`: `GHS_plus` edges `GR_RHS` on `mse_signal`
- `RW085_RB020_SNR10_N250`: `LASSO_CV` wins both MSE targets

Interpretation:

- the six-way dominance region is real, but it is not universal
- the cleanest currently verified core lies around
  `rho_within in [0.80, 0.90]`, `rho_between in {0.15, 0.20}`,
  `n_train in [200, 350]`, `target_snr = 1.0`

## Classical Benchmark Candidates With Full Six-Method Recheck

To see whether the newly discovered `GR-RHS` advantage patterns can survive in
more classical benchmark language, I ran a separate `4`-rep six-method check
under the same convergence-first rule:

- Bayesian methods are only discussed when `status=ok` and `converged=True`
- methods: `GR_RHS`, `RHS`, `GHS_plus`, `GIGG_MMLE`, `LASSO_CV`, `OLS`
- outputs:
  [summary_all_methods.csv](/d:/FilesP/GR-RHS/outputs/grrhs_classic_candidates_paper/summary_all_methods.csv:1)
  and
  [paper_table_main.md](/d:/FilesP/GR-RHS/outputs/grrhs_classic_candidates_paper/paper_tables/paper_table_main.md:1)

Two classical candidate families were checked:

1. `Classic Multimode 2 Active Groups`
   `group_sizes=[10,10,10,10,10]`, `n=500`, `rho_within=0.8`,
   `rho_between=0.2`
2. `Classic Multimode 3 Active Groups`
   `group_sizes=[10,10,10,10,10]`, `n=500`, `rho_within=0.6`,
   `rho_between=0.2`

### Main result

Only the `3`-active-group version survives as a true six-method `GR-RHS`
advantage point.

| candidate | GR-RHS | RHS | GHS+ | Lasso-CV | GIGG-MMLE | OLS | winner |
|---|---:|---:|---:|---:|---:|---:|---|
| `Classic Multimode 2 Active Groups` | 0.00416 | 0.00446 | 0.00425 | 0.00392 | 0.00494 | 0.01459 | `Lasso-CV` |
| `Classic Multimode 3 Active Groups` | 0.00552 | 0.00748 | 0.00655 | 0.00619 | 0.00784 | 0.01378 | `GR-RHS` |

Interpretation:

- the `2`-active-group equal-size candidate looked promising before the
  classical recheck, but once `LASSO_CV` is included and averaged over `4`
  repeated runs, it becomes a `Lasso-CV` win
- the `3`-active-group multimode candidate remains a valid classical benchmark
  region for `GR-RHS`
- in that surviving classical point, the ordering is:
  `GR_RHS < LASSO_CV < GHS_plus < RHS < GIGG_MMLE < OLS`

### Why the surviving classical point is still scientifically useful

The `Classic Multimode 3 Active Groups` candidate is important because it is:

- based on standard equal-size groups
- based on standard paper-like correlation levels
- still compatible with a clear grouped-regression narrative
- not merely a `GIGG_MMLE` collapse point

At the same time, it shows a more realistic limitation of the `GR-RHS` story:

- when the classical setting becomes too close to ordinary coordinate sparsity,
  `LASSO_CV` can be very hard to beat
- the best classical benchmark family for `GR-RHS` therefore seems to be
  `multi-mode heterogeneous active groups`, not generic sparse grouped signals

## Why Lasso-CV Can Look So Strong

The strong `LASSO_CV` performance in the failed classical `2`-active-group
candidate is not mysterious once the signal geometry is unpacked.

The main reasons are:

- there are only `2` active groups, so the global problem is still fairly
  sparse
- sample size is large (`n=500`), so cross-validated penalty selection is
  stable
- equal group sizes remove one of the main places where a group-layer model can
  differentiate itself
- the underlying signal is still fairly close to coordinate-wise sparse
  recovery, even though it has within-group heterogeneity

So this is not evidence that `LASSO_CV` suddenly understands grouped structure.
It is evidence that, in some classical equal-size sparse-ish designs, the
problem remains easy enough for a strong coordinate-sparse baseline to compete
extremely well.

Practical reading:

- if the active-group count is small and the groups are all the same size,
  `LASSO_CV` can erase a large fraction of the practical advantage of `GR-RHS`
- to make the group-layer mechanism matter, the benchmark needs either:
  more active groups, stronger multimode heterogeneity, larger groups, or
  unequal group sizes

## Group-Size Family Follow-Up

To understand whether `LASSO_CV` is mainly a problem of the equal-size
`10x5` geometry, I re-ran the same multimode logic on several group-size
families under the same convergence-first rule. These were targeted
`2`-rep diagnostics, not new paper tables, but they clarify the mechanism.

Summary:

| family | group sizes | winner | Lasso / GR | GIGG / GR |
|---|---|---|---:|---:|
| `equal_10x5_3act` | `[10,10,10,10,10]` | `GR_RHS` | 1.22 | 1.50 |
| `small_5x10_3act` | `[5]*10` | `LASSO_CV` | 0.84 | 1.54 |
| `large_25x2_2act` | `[25,25]` | `GR_RHS` | 1.37 | 1.73 |
| `unequal_CLlike_3act` | `[30,10,5,3,2]` | `GR_RHS` | 1.31 | 2.01 |

Interpretation:

- `LASSO_CV` is strongest when the groups are very small and numerous
  (`[5]*10`), because the task starts to look like ordinary fine-grained sparse
  recovery
- `GR-RHS` looks much better when groups are wider (`[25,25]`) or unequal
  (`[30,10,5,3,2]`)
- this supports the view that `GR-RHS` is most useful when the benchmark
  actually requires a meaningful group-layer decision, rather than only a
  coefficient-wise sparsity decision

Working practical rule:

- if the goal is to stress-test `GR-RHS` against `LASSO_CV`, avoid using only
  many tiny equal-size groups
- if the goal is to reveal group-layer finite-slab benefits, use either
  moderate-to-large groups or heterogeneous unequal group sizes

## High-Budget Targeted Scan Before Boundary Mapping

Before the boundary scan, a smaller high-budget probe already showed the same
pattern:

Source:
 [summary.csv](/d:/FilesP/GR-RHS/outputs/custom_grrhs_convergence_first_budget/summary.csv:1)

Most important row:

- `wg_mixed_rw08_rb02_n400`
- `GR_RHS`: `mse_overall=0.0613`, `mse_signal=0.1229`, `coverage_95=0.9933`
- `RHS`: `0.0667`, `0.1392`, `0.9933`
- `GIGG_MMLE`: `0.2837`, `0.3292`, `0.1067`

This run motivated the more systematic `within_group_mixed_boundary_scan`.

## Earlier Evidence Before Full Convergence-First Budget

### Focused benchmark evidence

Source:
 [table_linear_benchmark.csv](/d:/FilesP/GR-RHS/outputs/focused_exp3_checks_v2/tables/exp3_within_group_mixed_focus/table_linear_benchmark.csv:1)

For `RW08_SNR10`:

- `GR_RHS`: `mse_overall=0.0967`, `mse_signal=0.1217`, `coverage_95=0.9867`
- `RHS`: `0.0999`, `0.1414`, `0.9867`
- `GIGG_MMLE`: `1.2959`, `1.2184`, `0.44`

This was the first strong sign that `within_group_mixed` was the right place to
search for a true advantage region.

### Exploratory probe evidence

Source:
 [summary.csv](/d:/FilesP/GR-RHS/outputs/custom_grrhs_dominance_explore/summary.csv:1)

This exploratory run was useful mostly for identifying:

- which new DGP angles were promising
- where `GR_RHS` was still failing the convergence gate
- where `RHS` rather than `GIGG_MMLE` was the main competitor

It should not be treated as the final evidence for claims because it used a
smaller budget and many `GR_RHS` rows were filtered out by convergence.

## Partial Advantage Region: Half-Dense

Source:
 [summary.csv](/d:/FilesP/GR-RHS/outputs/custom_grrhs_convergence_first_budget/summary.csv:1)
 and
 [paired_converged_summary.csv](/d:/FilesP/GR-RHS/outputs/custom_grrhs_convergence_first_budget/paired_converged_summary.csv:1)

### Main pattern

- `GR-RHS` consistently beats `GIGG_MMLE`
- `GR-RHS` maintains much better coverage than `GIGG_MMLE`
- `RHS` remains a strong competitor, and can be slightly better on overall MSE
  at some points

### Example: `half_dense_rw08_rb02_n200`

- `GR_RHS`: `mse_overall=0.0610`, `mse_signal=0.1883`, `coverage_95=0.9667`
- `RHS`: `0.0588`, `0.2507`, `0.96`
- `GIGG_MMLE`: `0.1373`, `0.2058`, `0.24`

Interpretation:

- This is a robust `GR-RHS > GIGG_MMLE` region
- It is not yet a clean universal dominance region against `RHS`

## Weak-Identification Region

Source:
 [summary.csv](/d:/FilesP/GR-RHS/outputs/pilot_within_group_weakid_budget/results/exp3_weakid_pilot_budget_hi/summary.csv:1)

### Key row: `RW95_SNR10`

- `GIGG_MMLE`: `mse_overall=0.0004565`, `coverage_95=0.24`, `avg_ci_length=0.0074`
- `GR_RHS`: `mse_overall=0.01910`, `coverage_95=0.98`, `avg_ci_length=1.3019`

Interpretation:

- `GIGG_MMLE` can look excellent if judged only by point-estimate MSE
- But it is severely overconfident
- `GR-RHS` is much more trustworthy for inference

This region is best used to argue:

- honest uncertainty
- robust posterior inference
- not necessarily minimum MSE

## Low-Dimensional Probe Where Both Methods Converge Cleanly

Source:
 [summary.csv](/d:/FilesP/GR-RHS/outputs/randomcoef_probe_grrhs_nuts_vs_gigg/lowdim_mixed/summary.csv:1)

Results:

- `GR_RHS`: `runtime=10.39s`, `mse_overall=0.00600`
- `GIGG_MMLE`: `runtime=32.52s`, `mse_overall=0.00659`

Interpretation:

- Even in a regime where both models converge cleanly, `GR-RHS` is not winning
  only because `GIGG_MMLE` fails.
- There is also a speed-accuracy advantage in at least some easy-to-moderate
  settings.

## Regions That Are Not Currently Dominance Regions

### Random-coefficient high-correlation variants

From exploratory and follow-up scans:

- `GR-RHS` did not reliably pass convergence gates
- `RHS` was often the strongest method
- `GIGG_MMLE` was unstable but not the only weak method

Conclusion:

- This is not currently a good showcase region for `GR-RHS`

### Early low-budget within-group-mixed runs

From:
 [summary.csv](/d:/FilesP/GR-RHS/outputs/custom_grrhs_frontier_followup/summary.csv:1)

At lower budgets or lower sample sizes, `GR-RHS` could still fail convergence.
Those runs were useful for locating the search direction, but they should now be
superseded by the high-budget convergence-first files.

## Consolidated GIGG_MMLE Failure Map

This section consolidates all currently observed `GIGG_MMLE` failure modes in
this workspace, together with the main explanations that are now supported by
the available evidence.

The main point is:

- `GIGG_MMLE` does not fail in only one way
- different grouped stress regimes expose different weaknesses
- the repeated pattern is over-concentration, signal loss, poor calibration,
  heavy runtime, and occasional convergence strain

### Failure Type 1: Severe overconfidence despite seemingly usable point estimates

This is the cleanest inference-failure pattern.

Representative cases:

- weak-identification pilot:
  [summary.csv](/d:/FilesP/GR-RHS/outputs/pilot_within_group_weakid_budget/results/exp3_weakid_pilot_budget_hi/summary.csv:1)
- live calibration probe:
  [summary_live.csv](/d:/FilesP/GR-RHS/outputs/gigg_mmle_calibration_live_test/results/g10x5_snr10_rb20/summary_live.csv:1)

Observed pattern:

- in weak-ID, `GIGG_MMLE` can achieve extremely small point-estimate MSE
- but coverage can collapse to around `0.24`
- in the live `distributed` probe, MSE is sometimes not catastrophic, yet
  coverage still drops to roughly `0.15` to `0.34`

Interpretation:

- this is not just "worse estimation"
- it is a posterior uncertainty failure
- `GIGG_MMLE` becomes far too concentrated relative to the actual uncertainty

### Failure Type 2: Joint estimation and calibration collapse in within-group-mixed

This is the main benchmark failure pattern that supports the `GR-RHS`
advantage-region story.

Representative artifacts:

- [paired_all3_converged.csv](/d:/FilesP/GR-RHS/outputs/within_group_mixed_boundary_scan/paired_all3_converged.csv:1)
- [stable_points_all_methods_full.csv](/d:/FilesP/GR-RHS/outputs/grrhs_sixway_region_scan_combined/stable_points_all_methods_full.csv:1)
- [paper_table_main.md](/d:/FilesP/GR-RHS/outputs/grrhs_sixway_region_scan_combined/paper_tables/paper_table_main.md:1)

Observed pattern:

- in the original three-method `within_group_mixed` boundary scan,
  `GIGG_MMLE` repeatedly has much larger `mse_overall` and `mse_signal`
  together with coverage near `0.03` to `0.16`
- in the six-way extension, the `10` stable six-way points again show
  `GIGG_MMLE` ranked `5/6` on both `mse_overall` and `mse_signal` every time
- across those `10` six-way stable points, `GIGG_MMLE` averages:
  `mse_overall=0.520`, `mse_signal=0.512`, `coverage_95=0.18`,
  `runtime=90.8s`
- over the same points, `GR_RHS` averages:
  `mse_overall=0.081`, `mse_signal=0.147`, `coverage_95=0.984`,
  `runtime=15.4s`

Interpretation:

- in this regime, `GIGG_MMLE` is not merely losing by a small margin
- it is systematically misaligned with the grouped, within-group-heterogeneous
  signal structure
- relative to `GR_RHS`, this is roughly a `6.4x` average overall-MSE gap and a
  `5.9x` runtime gap on the six-way stable points

### Failure Type 3: Aggressive null shrinkage that destroys signal recovery

This is the most interpretable mechanism story in the new paper-grade
synthetic confirmation regions.

Representative artifacts:

- [grrhs_paper_repro_final_gr_vs_gigg.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_final_gr_vs_gigg.csv:1)
- [confirm_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_paired_decoy/confirm_summary.csv:1)
- [confirm_summary.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_size_imbalance_snr10/confirm_summary.csv:1)

Observed pattern:

- in `paired_decoy`, `GIGG_MMLE` gets `mse_null=0.00034`, but pays for that
  with `mse_signal=0.11199` and `coverage_95=0.1944`
- in `size_imbalance`, `GIGG_MMLE` again keeps null error small, but remains
  much worse than `GR_RHS` on coverage and often on signal error

Interpretation:

- this is the strongest evidence that the model tends to over-shrink in the
  wrong direction under grouped structural ambiguity
- when null groups can mimic signal, or when active groups are small and easy to
  over-suppress, `GIGG_MMLE` appears to regularize too aggressively

### Failure Type 4: Parameter-boundary pathology is real, but not sufficient

This is the main implementation-level diagnostic story.

Representative artifacts:

- [raw_results_live.csv](/d:/FilesP/GR-RHS/outputs/gigg_mmle_calibration_live_test/results/g10x5_snr10_rb20/raw_results_live.csv:1)
- [gigg_q_constraint_quicktest.csv](/d:/FilesP/GR-RHS/outputs/q_constraint_quicktest/gigg_q_constraint_quicktest.csv:1)
- [gigg_bmax_quicktest.csv](/d:/FilesP/GR-RHS/outputs/q_constraint_quicktest/gigg_bmax_quicktest.csv:1)

Observed pattern:

- some live probe fits show multiple groups pinned at `mmle_q = 4.0`
- the implementation really does use a hard MMLE cap `b_max = 4.0`
- however, the quick tests show:
  changing only `q_constraint_mode` does not repair the problem
- and even when `b_max` is increased far beyond `4`, the bad coverage and bad
  MSE often remain essentially unchanged

Interpretation:

- boundary pile-up at `q = 4` is a real warning sign
- but it is better understood as a symptom of deeper mismatch
- the current evidence does not support the claim that removing the `4.0` cap
  would materially solve `GIGG_MMLE` in these regimes

### Failure Type 5: Heavy runtime and budget sensitivity

This is the most operational failure pattern.

Representative artifacts:

- [grrhs_paper_repro_final_gr_vs_gigg.csv](/d:/FilesP/GR-RHS/outputs/grrhs_paper_repro_final_gr_vs_gigg.csv:1)
- [paper_table_appendix_full.md](/d:/FilesP/GR-RHS/outputs/grrhs_sixway_region_scan_combined/paper_tables/paper_table_appendix_full.md:1)

Observed pattern:

- in the new paper-grade regions, `GIGG_MMLE` is roughly `2.4x` to `4.1x`
  slower than `GR_RHS`
- in the six-way stable points, it averages `90.8s` versus `15.4s` for
  `GR_RHS`
- in `paired_decoy`, near-paper `10k+10k` budgets were needed just to obtain
  acceptable convergence on most replicates
- in the live probe, some `distributed` settings needed repeated attempts and
  still showed failed convergence for one replicate at `rho_within=0.95`

Interpretation:

- even where `GIGG_MMLE` does return usable output, it often does so with a much
  worse cost profile
- this weakens it both scientifically and practically in the discovered
  advantage regions

### Failure Type 6: Not every bad result should be called a GIGG-only failure

It is important to keep the story honest.

Representative artifacts:

- [summary.csv](/d:/FilesP/GR-RHS/outputs/randomcoef_probe_grrhs_nuts_vs_gigg/lowdim_mixed/summary.csv:1)

Counterexamples / caveats:

- in the low-dimensional clean-convergence probe, both methods converge and the
  difference is small: `0.00600` for `GR_RHS` versus `0.00659` for
  `GIGG_MMLE`
- in random-coefficient high-correlation variants, `GIGG_MMLE` was unstable,
  but `GR-RHS` was not uniformly strong there either, and `RHS` was often the
  real winner

Interpretation:

- the strongest current claim is not that `GIGG_MMLE` always fails
- the stronger and more defensible claim is that there is now a clear family of
  grouped structural stress regimes where `GIGG_MMLE` fails systematically

### Working summary of causes

The most plausible current causes, taken together, are:

- `GIGG_MMLE` is poorly matched to grouped designs with strong within-group
  correlation and heterogeneous within-group signal allocation
- it appears to reward aggressive null shrinkage even when that sacrifices
  signal preservation
- it often generates posteriors that are too concentrated, causing severe
  undercoverage
- the MMLE update can show boundary pile-up at `q = 4`, but relaxing that cap
  does not by itself repair the problem
- its computational path is much heavier and more brittle in the hardest
  grouped regions

Practical bottom line:

- when writing the `GR-RHS` story, `GIGG_MMLE` should be discussed as a method
  that can work in easier settings
- but in the discovered advantage regions, the current evidence points to a
  repeatable pattern of mechanism mismatch, overconfidence, and runtime burden

## Current Best Claims

If a short paper-style claim is needed now, the most defensible version is:

> In the `within_group_mixed` regime, `GR-RHS` exhibits a stable dominance
> region under strict convergence filtering. Across the tested settings with
> `rho_within` from `0.6` to `0.9` and moderate sample sizes, `GR-RHS`
> consistently outperforms both `RHS` and `GIGG_MMLE` on overall and signal
> estimation error while maintaining near-nominal coverage.

If a stronger inference-oriented claim is needed:

> In weak-identification settings, the main advantage of `GR-RHS` is not
> minimum point-estimate error but trustworthy posterior uncertainty; in those
> regimes, `GIGG_MMLE` can be sharply overconfident even when its MSE appears
> favorable.

For method-introduction figures and mechanism-first framing, see:

- [grrhs_mechanism_experiment_design.md](/d:/FilesP/GR-RHS/docs/grrhs_mechanism_experiment_design.md)

## Recommended Citation Order for Future Writing

When presenting the evidence, the cleanest order is:

1. `within_group_mixed_boundary_scan`
2. `custom_grrhs_convergence_first_budget`
3. `focused_exp3_checks_v2` within-group mixed table
4. weak-ID pilot budget
5. low-dimensional clean-convergence probe
6. exploratory scans as historical search log only

## Open Follow-Ups

The current document suggests three natural next steps:

- Produce a visual phase map for `within_group_mixed`
- Test whether the dominance region extends further left in sample size
- Check whether moderate `rho_between > 0.2` preserves the same stable region

## Changelog Note

This file summarizes experiments already run in this workspace as of
`2026-04-25`. It should be updated whenever new dominance-region scans or
counterexamples are added.
