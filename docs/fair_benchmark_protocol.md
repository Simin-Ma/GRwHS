# Fair Benchmark Protocol (Unified)

This document now tracks the fairness contract for the **single active simulation pipeline**.

## Active Scope

All benchmark runs should use `simulation_project/src/run_experiment.py` only.

## Core Fairness Rules

1. Same generated dataset for all compared methods within a replicate.
2. Same preprocessing and split logic across compared methods.
3. Convergence diagnostics are recorded and can be enforced before inclusion.
4. Tables must report effective run counts when filtering unconverged fits.

## What is Removed

- legacy sweep YAML orchestration
- scene-based legacy stacks (`sim_s*`, `scene*` trees)
- old `grrhs` CLI benchmark path

## Where to Run

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2
```
