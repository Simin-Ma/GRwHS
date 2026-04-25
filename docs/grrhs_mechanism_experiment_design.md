# GR-RHS Mechanism Experiment Design

This document defines a dedicated experiment package for the **method-introduction**
stage of the paper.

Its purpose is narrower than the full benchmark:

- explain what mechanism `GR-RHS` adds beyond `RHS`
- show that the added group layer changes posterior behavior in the intended way
- provide figures that make the mechanism visually legible
- avoid over-claiming from unconverged or purely exploratory runs

This document should be used together with:

- [grrhs_benchmark_blueprint.md](/d:/FilesP/GR-RHS/docs/grrhs_benchmark_blueprint.md)
- [grrhs_validation_experiment_framework.md](/d:/FilesP/GR-RHS/docs/grrhs_validation_experiment_framework.md)
- [grrhs_advantage_regions.md](/d:/FilesP/GR-RHS/docs/grrhs_advantage_regions.md)

## Core Rule

All analysis in this package is **convergence-first**.

Only rows with:

- `status = ok`
- `converged = True`

have interpretive value.

When comparing `GR_RHS` and `RHS` directly, the default artifact should be the
**common-converged paired subset**. Smoke runs can be used to debug design
direction, but they should not be used as headline figures.

## Goal

The mechanism section should answer four questions in order:

1. Does `GR-RHS` actually learn stronger signal-group vs null-group separation?
2. Does that separation matter more when grouped structure is ambiguous?
3. Is the gain really about the `group -> coefficient` hierarchy rather than a generic tuning effect?
4. What kind of grouped geometry makes the mechanism visible?

That leads to a compact four-experiment package.

## Recommended Mechanism Experiments

### M1. Group Separation

Scientific role:

- the cleanest direct proof that the latent group gate `kappa_g` is doing real work

Implementation anchor:

- `GA-V2-A`
- results path:
  `results/group_aware_v2/ga_v2_group_separation`

Why this belongs in the method section:

- it is the most direct answer to "what extra object does `GR-RHS` learn that `RHS` does not?"
- it naturally supports group-level visualizations

Recommended setup:

- `group_sizes = [10,10,10,10,10]`
- `rho_within = 0.8`
- `rho_between = 0.2`
- `n_train = 100`
- `n_test = 30`
- methods: `GR_RHS`, `RHS`

Primary estimands:

- `group_auroc`
- `kappa_gap`
- `kappa_signal_mean`
- `kappa_null_mean`
- `null_group_mse`
- `signal_group_mse`

Expected claim:

- `GR-RHS` separates active and inactive groups more cleanly than `RHS`, even
  when coefficient-level prediction differences are only moderate

### M2. Correlation Stress Under Structural Ambiguity

Scientific role:

- show that the group-aware mechanism matters more when correlation is high and one null group looks structurally similar to signal

Implementation anchor:

- revised `GA-V2-C`
- current preferred design:
  `mixed_decoy`
- results path:
  `results/group_aware_v2/ga_v2_correlation_stress`

Why this belongs in the method section:

- this is the cleanest visual story for "group-aware shrinkage under ambiguity"
- it isolates the mechanism better than a generic six-method leaderboard

Recommended setup:

- `group_sizes = [10,10,10,10,10]`
- `active_groups = [0,1]`
- `within_group_patterns = [mixed_decoy, concentrated]`
- `rho_within in {0.8, 0.9}`
- `rho_between = 0.2`
- `target_snr = 1.0`
- methods: `GR_RHS`, `RHS`

Interpretation rule:

- `mixed_decoy` is the main design
- `concentrated` is the contrast case
- prioritize mechanism metrics over raw `mse_overall` alone

Primary estimands:

- `group_auroc`
- `kappa_gap`
- `null_group_mse`
- `signal_group_mse`
- `mse_overall`

Expected claim:

- when correlation is high and one null group is structurally close to the active group, `GR-RHS` shows more structure-matched shrinkage than `RHS`

### M3. Complexity Unit / Scope Condition

Scientific role:

- clarify what `GR-RHS` is and is not supposed to do
- show that the method changes the effective unit of regularization from coefficient complexity to group complexity

Implementation anchor:

- cleaned `GA-V2-B`
- results path:
  `results/group_aware_v2/ga_v2_complexity_mismatch`

Why this belongs in the method section:

- it helps prevent overclaiming
- it explains why some cells are near ties under a clean definition

Recommended setup:

- keep unified `rho_between`
- compare:
  `few_groups`
  versus
  `many_groups`
- include:
  `concentrated`
  and
  `distributed`

Primary estimands:

- `group_auroc`
- `kappa_gap`
- `mse_signal`
- `mse_null`
- `mse_overall`

Expected claim:

- `GR-RHS` reacts differently when the same coefficient budget is allocated across different numbers of active groups
- but not every cell should be described as a dominance region
- in particular, `many_groups/distributed` may be close to a tie under the cleaned unified-correlation definition

### M4. Mechanism Ablation

Scientific role:

- identify which part of `GR-RHS` is responsible for the gain

Implementation anchor:

- `Exp4`
- results path:
  `results/exp4_variant_ablation`

Minimum ablations to report:

- calibrated `GR_RHS`
- misspecified or weakened `GR_RHS`
- `RHS_oracle`
- if surfaced in code:
  `GR_RHS_no_kappa`
  or
  `GR_RHS_shared_kappa`

Primary estimands:

- `kappa_gap`
- `mse_overall`
- `mse_signal`
- `tau_ratio_to_oracle`
- runtime / convergence diagnostics

Expected claim:

- removing or weakening the group gate should reduce the mechanism advantage
- this is the strongest evidence that the gain is not just a generic prior-scale effect

## Figure Plan

The method section should not rely on tables alone. It needs a small set of
figures that make the hierarchy visible.

### Figure 1. Mechanism Schematic

Purpose:

- explain the conceptual difference between `RHS` and `GR-RHS`

Content:

- left panel: `RHS`
  coefficient-level shrinkage only
- right panel: `GR-RHS`
  group gate `kappa_g` plus coefficient-level local scales

Message:

- `GR-RHS` changes the unit of regularization from coefficients alone to a
  group-aware two-level hierarchy

This can be a conceptual illustration rather than simulation output.

Data source:

- no CSV required

Axes / layout:

- not a statistical plot
- two conceptual panels:
  `RHS`
  versus
  `GR-RHS`

Suggested visual elements:

- `RHS`:
  global shrinkage node -> coefficient-specific local scales -> coefficients
- `GR-RHS`:
  global/group calibration -> group gate `kappa_g` -> coefficient-specific local
  scales -> coefficients

Caption template:

> **Figure 1. Conceptual hierarchy of `RHS` and `GR-RHS`.**
> `RHS` regularizes at the coefficient level, whereas `GR-RHS` adds a
> group-level gate that modulates within-group local shrinkage. The added group
> layer changes the effective unit of regularization from coefficients alone to
> a group-aware two-level hierarchy.

### Figure 2. Group Separation Plot

Experiment:

- `M1 / GA-V2-A`

Recommended visual:

- point-range or boxplot of posterior mean `kappa_g` by true group status
- color active groups and null groups differently
- compare `GR_RHS` against `RHS`

Alternative:

- bar plot of `group_auroc` and `kappa_gap`

Message:

- `GR-RHS` creates stronger group-level separation than `RHS`

Read from:

- primary:
  [summary_paired.csv](/d:/FilesP/GR-RHS/outputs/_tmp_ga_v2_eval/ga_v2a/results/group_aware_v2/ga_v2_group_separation/summary_paired.csv)
- if a formal rerun is produced later, replace with the final
  `results/group_aware_v2/ga_v2_group_separation/summary_paired.csv`

Recommended plotting choice:

- main panel:
  grouped bar chart
- x-axis:
  `method`
- y-axis:
  `kappa_gap`
- secondary overlaid points or side panel:
  `group_auroc`

Alternative if replicate-level raw draws are available:

- use [raw_results.csv](/d:/FilesP/GR-RHS/outputs/_tmp_ga_v2_eval/ga_v2a/results/group_aware_v2/ga_v2_group_separation/raw_results.csv)
- x-axis:
  `method`
- y-axis:
  replicate-level `kappa_gap`
- geometry:
  boxplot + jitter

Minimal annotation:

- print `mse_overall` under each method bar
- report `n_effective`

Caption template:

> **Figure 2. Group-level separation in `GA-V2-A`.**
> Under the same convergence-qualified replicates, `GR-RHS` shows a larger
> signal-vs-null group separation gap than `RHS`, while also improving overall
> estimation error. This is direct evidence that the added group gate learns
> meaningful group-level structure rather than acting as a redundant latent
> layer.

### Figure 3. Correlation-Ambiguity Heatmap

Experiment:

- `M2 / revised GA-V2-C`

Recommended visual:

- heatmap over
  `rho_within`
  by
  `within_group_pattern`
- fill by:
  `kappa_gap`
  or
  paired `mse_overall` delta
- annotate only common-converged cells

Best choice:

- main panel: `GR_RHS - RHS` delta in `mse_overall`
- side panel: `kappa_gap` for `GR_RHS`

Message:

- the mechanism becomes more useful in high-correlation ambiguous-group designs,
  especially under `mixed_decoy`

Read from:

- primary:
  [summary_paired.csv](/d:/FilesP/GR-RHS/outputs/ga_v2c_mixed_decoy_smoke/results/group_aware_v2/ga_v2_correlation_stress/summary_paired.csv)
- coverage of common-converged cells:
  [paired_stats.csv](/d:/FilesP/GR-RHS/outputs/ga_v2c_mixed_decoy_smoke/results/group_aware_v2/ga_v2_correlation_stress/paired_stats.csv)

Construct two derived tables:

1. `mse_delta = mse_overall(GR_RHS) - mse_overall(RHS)`
2. `kappa_gap_gr = kappa_gap(GR_RHS)`

Panel A:

- x-axis:
  `rho_within`
- y-axis:
  `within_group_pattern`
- fill:
  `mse_delta`
- color convention:
  negative = `GR-RHS` better
  positive = `RHS` better

Panel B:

- x-axis:
  `rho_within`
- y-axis:
  `within_group_pattern`
- fill:
  `kappa_gap_gr`

Cell annotations:

- `n_effective` from `summary_paired.csv`
- or `n_common_replicates` from `paired_stats.csv`

Caption template:

> **Figure 3. Correlation stress under structural ambiguity in revised `GA-V2-C`.**
> Cells show only common-converged paired summaries. The left panel reports the
> paired overall-MSE difference between `GR-RHS` and `RHS`; the right panel
> reports the group-separation gap learned by `GR-RHS`. The `mixed_decoy`
> design makes the group-aware advantage more visible than the simpler
> concentrated baseline, especially as within-group correlation increases.

### Figure 4. Decoy Group Shrinkage Profile

Experiment:

- `M2 / mixed_decoy`

Recommended visual:

- per-group shrinkage profile for one representative common-converged replicate
- x-axis: group id
- y-axis: posterior mean `kappa_g`
- mark:
  active groups
  decoy null group
  other null groups

Message:

- `GR-RHS` better suppresses the decoy null group while retaining active-group support

This is likely the single most persuasive mechanism figure after the schematic.

Read from:

- replicate selection:
  [raw_results.csv](/d:/FilesP/GR-RHS/outputs/ga_v2c_mixed_decoy_smoke/results/group_aware_v2/ga_v2_correlation_stress/raw_results.csv)

Important note:

- `raw_results.csv` currently stores summary diagnostics per replicate, not the
  full per-group `kappa_g` vector
- to draw this figure properly, use one representative converged run and extract
  per-group posterior summaries from the fit object or extend the experiment to
  save per-group `kappa_g` means for each replicate

Recommended derived data to save for plotting:

- `replicate_id`
- `method`
- `group_id`
- `kappa_group_mean`
- `is_active_group`
- `is_decoy_group`

Plot specification:

- x-axis:
  `group_id`
- y-axis:
  posterior mean `kappa_group_mean`
- color:
  group role
  (`active`, `decoy null`, `other null`)
- facet:
  `method`

If full per-group `kappa` export is not available yet:

- use this as a required follow-up output rather than dropping the figure
- this figure is important enough to justify a small output-format extension

Caption template:

> **Figure 4. Posterior group shrinkage in a representative `mixed_decoy` replicate.**
> The decoy null group is structurally similar to the primary active group, but
> `GR-RHS` maintains clearer separation between active and null groups than
> `RHS`. This illustrates the practical effect of the group-level gate under
> ambiguous grouped correlation.

### Figure 5. Complexity Unit Comparison

Experiment:

- `M3 / GA-V2-B`

Recommended visual:

- two-panel plot:
  `few_groups`
  versus
  `many_groups`
- within each panel, compare `GR_RHS` and `RHS` on:
  `kappa_gap`
  and
  `mse_overall`

Message:

- the method responds to group complexity, not just total coefficient count
- but this is a scope-condition figure, not a universal win figure

Read from:

- cleaned unified-correlation review:
  [summary_paired.csv](/d:/FilesP/GR-RHS/outputs/ga_v2b_uniform_rho_smoke/results/group_aware_v2/ga_v2_complexity_mismatch/summary_paired.csv)
- if a later formal common-converged run is produced, replace with that final
  path

Panel A:

- x-axis:
  `complexity_pattern`
  (`few_groups`, `many_groups`)
- y-axis:
  `kappa_gap`
- hue:
  `method`

Panel B:

- x-axis:
  `complexity_pattern`
- y-axis:
  `mse_overall`
- hue:
  `method`

Facet both panels by:

- `within_group_pattern`

Key annotation:

- explicitly note the `many_groups/distributed` cell when it is near a tie

Caption template:

> **Figure 5. Complexity allocation under cleaned `GA-V2-B`.**
> Holding total active coefficient count fixed, `GR-RHS` and `RHS` react
> differently to how signal is distributed across groups. The figure should be
> read as a scope-condition result: it shows sensitivity to group complexity,
> but not every allocation pattern yields a large `GR-RHS` win.

### Figure 6. Ablation Summary

Experiment:

- `M4 / Exp4`

Recommended visual:

- coefficient plot or bar plot of deltas versus baseline `GR_RHS`
- rows:
  `GR_RHS`
  `RHS_oracle`
  `GR_RHS_no_kappa`
  `GR_RHS_shared_kappa`
  misspecified `GR_RHS`
- columns:
  `kappa_gap`
  `mse_overall`
  `mse_signal`

Message:

- the group-layer gate is functionally responsible for a meaningful part of the advantage

Read from:

- final preferred source:
  `results/exp4_variant_ablation/summary.csv`
- if paired or converged-only variants exist, use the common-converged summary
  instead of raw marginal summaries

Recommended visual:

- coefficient / dot-and-whisker plot of deltas relative to baseline `GR_RHS`

Rows:

- baseline `GR_RHS`
- weakened / misspecified `GR_RHS`
- `RHS_oracle`
- `GR_RHS_no_kappa`
- `GR_RHS_shared_kappa`

Columns or facets:

- `kappa_gap`
- `mse_overall`
- `mse_signal`

Y-axis:

- variant name

X-axis:

- metric value or delta versus baseline

Caption template:

> **Figure 6. Mechanism ablation for `GR-RHS`.**
> Weakening or removing the group-level gate reduces the separation and
> estimation advantages associated with the full model. This supports the claim
> that the observed gains arise from the group-aware hierarchy rather than from
> generic prior tuning alone.

## Minimal Figure Set For The Paper

If the method section can only afford `3` figures, use:

1. Figure 1: mechanism schematic
2. Figure 2: `GA-V2-A` group separation figure
3. Figure 4: `GA-V2-C mixed_decoy` decoy-group shrinkage profile

If the method section can afford `4` figures, add:

4. Figure 6: ablation summary

## Table Plan For The Method Section

Use one compact mechanism table rather than a full leaderboard.

Recommended columns:

- experiment
- main scientific question
- primary metric
- `GR_RHS`
- `RHS`
- short takeaway

Recommended rows:

- `GA-V2-A`
- revised `GA-V2-C`
- cleaned `GA-V2-B`
- `Exp4` ablation

Suggested data sources:

- `GA-V2-A`:
  `results/group_aware_v2/ga_v2_group_separation/summary_paired.csv`
- revised `GA-V2-C`:
  `results/group_aware_v2/ga_v2_correlation_stress/summary_paired.csv`
- cleaned `GA-V2-B`:
  `results/group_aware_v2/ga_v2_complexity_mismatch/summary_paired.csv`
- `Exp4` ablation:
  `results/exp4_variant_ablation/summary.csv`

## Writing Guidance

This package should support wording like:

> `GR-RHS` augments `RHS` with a group-level gate that changes the effective
> unit of shrinkage from coefficients alone to a group-aware hierarchy. In
> convergence-qualified mechanism experiments, this produces stronger
> signal-group vs null-group separation and more structure-matched shrinkage,
> especially when groups are correlated and null groups can mimic signal.

Avoid wording like:

- `GR-RHS` wins every stress test
- `GA-V2-B` proves universal superiority
- unconverged or separately filtered summaries as if they were final evidence

## Practical Execution Order

Recommended order for producing method-section evidence:

1. `GA-V2-A`
2. revised `GA-V2-C`
3. `Exp4` ablation
4. cleaned `GA-V2-B`

Reason:

- this sequence moves from the cleanest positive mechanism story to the most
  nuanced scope-condition story

## Deliverables Checklist

- convergence-qualified paired summary for each mechanism experiment
- one representative per-group shrinkage figure
- one compact mechanism table
- one schematic comparing `RHS` and `GR-RHS`
- one short paragraph stating scope conditions and non-goals
