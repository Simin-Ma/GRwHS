# Fair Benchmark Protocol (Unified)

This document now tracks the fairness contract for the **single active simulation pipeline**.

## Active Scope

All benchmark runs should use the unified experiment CLI, preferably
`python -m simulation_project.src.run_experiment` (or the thin wrapper
`scripts/run_simulation.py`, which delegates to the same entrypoint).

## Core Fairness Rules

1. Same generated dataset for all compared methods within a replicate.
2. Same preprocessing and split logic across compared methods.
3. Convergence diagnostics are recorded and can be enforced before inclusion.
4. Tables must report effective run counts when filtering unconverged fits.


See `docs/simulation_cli_guide.md` for run commands.
