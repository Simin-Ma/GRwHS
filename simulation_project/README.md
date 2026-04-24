# simulation_project

Single full simulation framework for GR-RHS Exp1-Exp5, with two additional optional Exp3 extensions:
- `Exp3c`: paper-aligned high-dimensional random-coefficient stress run
- `Exp3d`: legacy boundary-stress run retained for backward compatibility

## Entrypoints

```bash
python -m simulation_project.src.run_experiment --help
python scripts/run_simulation.py --help
```

Sweep runner:

```bash
python -m simulation_project.src.run_sweep --list
python scripts/run_sweep.py --list
```

## Unified Protocol

- Single full paper-analysis configuration only.
- No `laptop/full` split.
- No `--profile` CLI option.
- No `--preset` CLI option.
- Default pipeline order: `Exp1 -> Exp2 -> Exp3a -> Exp3b -> Exp4 -> Exp5 -> analysis`.
- `Exp3c/Exp3d` run only when explicitly selected.
- `Exp3a` and `Exp3c` are the paper-aligned fixed-coefficient and random-coefficient benchmark lines.
- `Exp3d` is not part of that redesign; its historical result path remains `exp3d_within_group_mixed`.
- Scientific conclusion gate: Exp2-Exp5 Bayesian rows must satisfy `converged=True` and `status=ok`.

Default repeats when `--repeats` is omitted:

- `exp1=500`
- `exp2=100`
- `exp3a=100`
- `exp3b=100`
- `exp4=10`
- `exp5=20`

Optional explicit runs:

- `exp3c=30`
- `exp3d=100`

## Typical Commands

Run full pipeline:

```bash
python -m simulation_project.src.run_experiment --experiment all --n-jobs 2 --method-jobs 2 --all-parallel-jobs 2
```

Run one experiment:

```bash
python -m simulation_project.src.run_experiment --experiment 3c
```

Run the legacy boundary-stress extension:

```bash
python -m simulation_project.src.run_experiment --experiment 3d
```

Run analysis only:

```bash
python -m simulation_project.src.run_experiment --experiment analysis
```

## Output Layout

Default CLI output is sessionized:

`outputs/simulation_project/sessions/<timestamp>_cli_<experiment>/`

Useful pointers:

- `outputs/simulation_project/latest_session.txt`
- `outputs/simulation_project/latest_session.json`
- `outputs/simulation_project/session_index.jsonl`

Analysis artifacts include:

- `results/analysis_report.txt`
- `results/analysis_report.json`
- `results/diagnostics_runtime_table.csv`

## Bayesian Methods

- `GR_RHS`: project-default staged Gibbs implementation for Gaussian fits
- `RHS`: single Stan/HMC regularized horseshoe baseline aligned with `rstanarm::hs()`
- `GIGG_*`: Gibbs samplers aligned to the `gigg-master` reference path
- `GHS_plus`: Xu et al. (2016) hierarchical Bayesian grouped horseshoe with paper-aligned Gaussian Gibbs defaults

## Runtime Notes

- On Windows, process pools are available from spawn-safe script entrypoints.
- In interactive or non-spawn-safe launch contexts, the runtime falls back away from process pools automatically.
- Use `--skip-analysis` to skip run-level and global analysis work during long runs.
- Use `--no-archive-artifacts` to skip copying duplicate artifact snapshots into `runs/.../artifacts`.
