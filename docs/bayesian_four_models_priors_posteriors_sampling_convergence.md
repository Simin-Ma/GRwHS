# Bayesian Model Notes (Unified Runtime)

This note summarizes the Bayesian backends now used by the consolidated simulation pipeline.

## Runtime Modules

- GR-RHS NUTS: `simulation_project/src/core/models/grrhs_nuts.py`
- RHS baseline: `simulation_project/src/core/models/baselines/models.py`
- GIGG: `simulation_project/src/core/models/gigg_regression.py`
- Grouped Horseshoe+: `simulation_project/src/core/models/baselines/grouped_horseshoe.py`

## Diagnostics

Convergence diagnostics are provided by:

- `simulation_project/src/core/diagnostics/convergence.py`

and consumed by `simulation_project/src/utils.py`.

## Repository Policy

All old multi-stack experiment paths are removed; only the unified simulation path is supported.
