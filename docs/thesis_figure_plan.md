# Thesis Figure Plan (Unified Pipeline)

This figure plan assumes all results are generated from `simulation_project`.

## Sources

- experiment outputs: `outputs/simulation_project/results/`
- figures: `outputs/simulation_project/figures/`
- tables: `outputs/simulation_project/tables/`

## Suggested Figure Blocks

1. Theory/profile checks (`exp1`, `exp2`)
2. Linear benchmark core comparison (`exp3`)
3. Tau calibration/ablation (`exp4`)
4. Prior sensitivity (`exp5`)

## Reproducible Runner

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir outputs/simulation_project --n-jobs 2
```


