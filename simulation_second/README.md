# simulation_second

Second-generation benchmark framework for the `GR-RHS` blueprint.

This package is intentionally separate from `simulation_project/` so the new
benchmark design can evolve without inheriting the old `Exp1-5` naming scheme
or experiment assumptions.

What is implemented here:

- YAML-backed benchmark spec in `simulation_second/config/benchmark.yaml`
- blueprint-level benchmark settings and random signal-family generation
- grouped Gaussian train/test dataset generation
- batch runner for six-way method fitting and evaluation
- `raw_results.csv`, `summary.csv`, `summary_paired.csv`,
  `summary_paired_deltas.csv`
- paper-table builder producing markdown, CSV, and LaTeX tables

The package stays separate from `simulation_project/`, but it reuses the
legacy fitting and evaluation kernels where that is the fastest way to get a
complete second-generation benchmark pipeline working.

Examples:

```bash
python -m simulation_second.src.run_blueprint list-settings
python -m simulation_second.src.run_blueprint dump-manifest --save-path outputs/simulation_second/manifest.json
python -m simulation_second.src.run_blueprint sample-suite --repeats 2 --save-dir outputs/simulation_second/samples
python -m simulation_second.src.run_blueprint run-benchmark --settings setting_5_multimode_equal --repeats 3 --save-dir outputs/simulation_second/demo_run
python -m simulation_second.src.run_blueprint build-tables --results-dir outputs/simulation_second/demo_run
```
