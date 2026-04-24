# simulation_project

Single-default simulation framework for GR-RHS Exp1-Exp5 (with optional Exp3c/Exp3d stress runs).

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

- No `laptop/full` split.
- No `--profile` CLI option.
- Default pipeline order: `Exp1 -> Exp2 -> Exp3a -> Exp3b -> Exp4 -> Exp5 -> analysis`.
- `Exp3c/Exp3d` are optional stress experiments and run only when explicitly selected.
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

## Experiments

1. `run_exp1_kappa_profile_regimes`
2. `run_exp2_group_separation`
3. `run_exp3a_main_benchmark`
4. `run_exp3b_boundary_stress`
5. `run_exp3c_highdim_stress` (optional)
6. `run_exp3d_within_group_mixed` (optional)
7. `run_exp4_variant_ablation`
8. `run_exp5_prior_sensitivity`

`run_exp3_linear_benchmark` is still available as a combined Exp3 entry.

## Typical Commands

Run full default pipeline:

```bash
python -m simulation_project.src.run_experiment --experiment all --n-jobs 2 --method-jobs 2 --all-parallel-jobs 2
```

Run the laptop-friendly paper preset:

```bash
python -m simulation_project.src.run_experiment --experiment all --preset paper_laptop
```

Run one experiment:

```bash
python -m simulation_project.src.run_experiment --experiment 3c --preset paper_laptop
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

## Optional Quick Script

`scripts/run_laptop_best_2h.py` is retained as an optional quick-check utility only.
It is not the default scientific protocol.

## Paper-Laptop Design

Main-text recommendation on one laptop:

- `exp1`
- `exp2`
- `exp3a`
- `exp4`

Appendix recommendation:

- `exp3b`
- `exp5`
- `exp3c/exp3d` as spot checks

The `paper_laptop` preset keeps Exp2 at the full default repeats while reducing repeats for selected other experiments, keeps the default Bayesian convergence gate,
restricts the heaviest Exp3 methods to anchor settings, lowers GIGG/GHS+ per-fit budgets,
skips analysis by default, and disables duplicate artifact archiving.

## Runtime Notes

- On Windows, process pools are available from spawn-safe script entrypoints.
- In interactive or non-spawn-safe launch contexts, the runtime falls back away from process pools automatically.
- Use `--skip-analysis` to skip run-level and global analysis work during long runs.
- Use `--no-archive-artifacts` to skip copying duplicate artifact snapshots into `runs/.../artifacts`.
