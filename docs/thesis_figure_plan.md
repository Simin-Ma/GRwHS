# Thesis Figure Plan (Unified Pipeline)

This figure plan assumes all results are generated from `simulation_project`.

## Sources

- experiment outputs: `simulation_project/results/`
- figures: `simulation_project/figures/`
- tables: `simulation_project/tables/`

## Suggested Figure Blocks

1. Theory/profile checks (`exp1`, `exp2`)
2. Linear benchmark core comparison (`exp3`)
3. Tau calibration/ablation (`exp4`)
4. Prior sensitivity (`exp5`)

## Reproducible Runner

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir simulation_project --n-jobs 2
```
