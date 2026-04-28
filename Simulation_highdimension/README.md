# Simulation_highdimension

Dedicated high-dimensional benchmark suite for GR-RHS.

This package is intentionally small and reuses the mature fitting, evaluation,
pairing, table-building, and history-output utilities from `simulation_second`.
The purpose of this folder is to keep the high-dimensional benchmark settings
separate from the low-dimensional main benchmark while preserving the same
convergence-first comparison protocol.

## Design

The default suite contains three `p > n` settings:

| ID | Role | Design |
|---|---|---|
| `hd_setting_1_classical_anchor` | Neutral high-dimensional anchor | `n=200`, `p=500`, `50 x 10` groups, classical reference family |
| `hd_setting_2_single_mode` | Transition setting | `n=200`, `p=500`, `50 x 10` groups, single-mode heterogeneous family |
| `hd_setting_3_multimode_showcase` | Main high-dimensional GR-RHS stress point | `n=200`, `p=500`, `50 x 10` groups, multimode heterogeneous family |

All settings use target `R2 = 0.7`, `rho_between = 0.2`, and explicit
active/null group structure. This makes the suite a high-dimensional analogue
of the low-dimensional `simulation_second` benchmark rather than a separate
unstructured sparse-regression toy.

## Commands

```bash
python -m Simulation_highdimension.src.run_highdimension list-settings
python -m Simulation_highdimension.src.run_highdimension dump-manifest
python -m Simulation_highdimension.src.run_highdimension sample-setting --setting-id hd_setting_3_multimode_showcase
python -m Simulation_highdimension.src.run_highdimension run-benchmark --settings hd_setting_3_multimode_showcase --repeats 3 --methods GR_RHS RHS LASSO_CV --no-build-tables
```

If installed in editable mode, the console script is also available:

```bash
run-simulation-highdimension list-settings
```

## Notes

- `OLS` is not included in the default high-dimensional roster because `p > n`
  makes ordinary least squares an ill-posed baseline.
- The default repeats are deliberately modest. Bayesian high-dimensional runs
  with strict convergence gates can be expensive.
- For exploratory checks, start with `sample-setting` or a small method subset.
