# simulation_mechanism

Mechanism-first simulation package for the `GR-RHS` method section.

This package is separate from both `simulation_project/` and
`simulation_second/`. It is built around the experiment plan in
`docs/grrhs_mechanism_experiment_design.md` and focuses on four mechanism
experiments:

- `M1`: group separation
- `M2`: correlation stress under structural ambiguity
- `M3`: complexity unit / scope condition
- `M4`: mechanism ablation

What it adds beyond the older experiment runners:

- one unified mechanism suite with stable setting ids
- convergence-qualified paired summaries
- per-group `kappa_g` export for mechanism figures
- compact mechanism table and figure-ready CSVs

Convergence policy:

- all Bayesian methods in `simulation_mechanism` are always run with Bayesian
  convergence enforcement enabled
- the package forces the legacy runtime's `until converged` retry mode for
  Bayesian methods, including smoke runs and test-driven mechanism runs

Examples:

```bash
python -m simulation_mechanism.src.run_mechanism list-settings
python -m simulation_mechanism.src.run_mechanism dump-manifest --save-path outputs/simulation_mechanism/manifest.json
python -m simulation_mechanism.src.run_mechanism sample-setting --setting-id m2_mixed_decoy_rw080
python -m simulation_mechanism.src.run_mechanism run-mechanism --repeats 3 --save-dir outputs/simulation_mechanism/demo_run
python -m simulation_mechanism.src.run_mechanism build-tables --results-dir outputs/simulation_mechanism/demo_run
python -m simulation_mechanism.src.run_mechanism build-figures --results-dir outputs/simulation_mechanism/demo_run
```
