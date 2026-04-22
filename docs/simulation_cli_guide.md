# Simulation CLI Guide (By Experiment)

This guide covers the active simulation pipeline under `simulation_project`.

## 1. Unified Entry Points

```bash
python -m simulation_project.src.run_experiment --help
python scripts/run_simulation.py --help
```

Common CLI arguments:
- `--experiment {all,1,2,3,3a,3b,4,5,analysis}`
- `--workspace simulation_project` (resolves to `outputs/simulation_project`)
- `--save-dir <path>` (optional explicit override)
- `--seed 20260415`
- `--repeats <int>`
- `--n-jobs <int>`
- `--profile {full,laptop}`
- `--no-enforce-bayes-convergence`
- `--max-convergence-retries <int>`
- `--until-bayes-converged`
- `--sampler {nuts,collapsed,gibbs}`

Notes:
- `scripts/run_simulation.py` is a thin wrapper over `python -m simulation_project.src.run_experiment`.
- If `--save-dir` is omitted, each run gets an isolated session directory:
  `outputs/simulation_project/sessions/<timestamp>_cli_<experiment>/`.
- Advanced parameters (`methods`, `prior_grid`, `p0_list`, and so on) are exposed in Python function calls.
- For Exp2-Exp5, Bayesian minimum chains are profile-dependent by default:
  - `profile=laptop`: `2`
  - `profile=full`: `4`
- On Windows, process-pool parallelism is disabled by default for stability in interactive launch contexts.
  If needed, enable it explicitly from a spawn-safe script entrypoint with
  `SIM_ALLOW_WINDOWS_PROCESS_POOL=1`.

Default repeats when `--repeats` is omitted:
- `profile=full`: `exp1=500`, `exp2=30`, `exp3=20`, `exp4=30`, `exp5=30`
- `profile=laptop`: `exp1=200`, `exp2=10`, `exp3=5`, `exp4=15`, `exp5=15`

## 2. Convergence And Diagnostics

For Exp2 to Exp5, `raw_results.csv` includes:
- `rhat_max`
- `bulk_ess_min`
- `divergence_ratio`
- `runtime_seconds`
- `error`

Recommended checks:
- First check `converged`.
- Then check `bulk_ess_min` (especially important for Exp5).
- Large `divergence_ratio` usually indicates geometry issues; increase `adapt_delta` and/or `warmup`.

## 3. Recommended Commands By Experiment

The commands below are recommended presets. They are explicit choices and are not required to match the CLI defaults.

### Exp1 (`kappa_profile_regimes`)

Exp1 uses profile-grid posterior computation, not MCMC. `--sampler` has no effect.

```bash
python -m simulation_project.src.run_experiment --experiment 1 --save-dir outputs/simulation_project --repeats 1 --n-jobs 1 --no-enforce-bayes-convergence
python -m simulation_project.src.run_experiment --experiment 1 --save-dir outputs/simulation_project --profile laptop --repeats 200 --n-jobs 8 --no-enforce-bayes-convergence
python -m simulation_project.src.run_experiment --experiment 1 --save-dir outputs/simulation_project --profile full --repeats 500 --n-jobs 8 --no-enforce-bayes-convergence
```

### Exp2 (`group_separation`)

Use `nuts` with convergence enforcement.

```bash
python -m simulation_project.src.run_experiment --experiment 2 --save-dir outputs/simulation_project --profile laptop --repeats 10 --n-jobs 6 --max-convergence-retries 1 --sampler nuts
python -m simulation_project.src.run_experiment --experiment 2 --save-dir outputs/simulation_project --profile full --repeats 30 --n-jobs 6 --max-convergence-retries 2 --sampler nuts
```

### Exp3a (`main_benchmark`)

Exp3a is the primary benchmark layer (concentrated + distributed).

```bash
python -m simulation_project.src.run_experiment --experiment 3a --save-dir outputs/simulation_project --profile laptop --repeats 5 --n-jobs 8 --max-convergence-retries 1 --sampler nuts
python -m simulation_project.src.run_experiment --experiment 3a --save-dir outputs/simulation_project --profile full --repeats 20 --n-jobs 8 --max-convergence-retries 2 --sampler nuts
```

### Exp3b (`boundary_stress`)

Exp3b is the boundary-only stress layer.

```bash
python -m simulation_project.src.run_experiment --experiment 3b --save-dir outputs/simulation_project --profile laptop --repeats 5 --n-jobs 8 --max-convergence-retries 1 --sampler nuts
python -m simulation_project.src.run_experiment --experiment 3b --save-dir outputs/simulation_project --profile full --repeats 20 --n-jobs 8 --max-convergence-retries 2 --sampler nuts
```

Current defaults for Exp3a/Exp3b (`exp3_design="core30"`):
- `group_configs=["G10x5","CL","CS"]`
- env points (enforced `rho_within > rho_between`):
  - `E0: (rho_within=0.3, rho_between=0.1, target_snr=1.0)`
  - `RW_PLUS: (0.8, 0.1, 1.0)`
- no SNR axis in core30 (target_snr fixed to 1.0)
- Exp3a signal types: `["concentrated", "distributed"]`
- Exp3b signal types: `["boundary"]`
- boundary calibration: `xi = 1.2 * xi_crit(u0=0.5, rho_profile)`,
  with `rho_profile = rho_within / sqrt(sigma2_boundary)` and `sigma2_boundary=1.0`
- Bayesian minimum chains in Exp3:
  - default `2` for `profile="laptop"` (exploration-friendly)
  - default `4` for `profile="full"`
  - override in Python with `bayes_min_chains=<k>`
- `tau_target="groups"` for GR-RHS

Legacy full-factor mode is still available from Python via `exp3_design="legacy_factorial"` (without SNR axis).

### Exp4 (`variant_ablation`)

Exp4 now uses a compact mechanism-ablation design aligned with Exp3:
- DGP scale: `n=100`, `p=50`, `group_sizes=[10,10,10,10,10]`
- default sparsity levels: `p0 in {5, 30}` (`p0` = number of active coefficients)
- default variants: `calibrated`, `fixed_10x`, `RHS_oracle`
- optional full-ablation variant: `oracle` (Python call with `include_oracle=True`)
- default convergence retries: `3` (retry-until-quality within bounded budget)

Evaluation note:
- Primary decision metric is relative MSE against `RHS_oracle` (`mse_rel_rhs_oracle`, `<1` is better).
- `tau_ratio_to_oracle` is retained as a diagnostic signal, not a strict pass/fail target.

`gibbs` is the default/recommended sampler for Exp4.

```bash
python -m simulation_project.src.run_experiment --experiment 4 --save-dir outputs/simulation_project --profile laptop --repeats 15 --n-jobs 6 --max-convergence-retries 3 --sampler gibbs
python -m simulation_project.src.run_experiment --experiment 4 --save-dir outputs/simulation_project --profile full --repeats 30 --n-jobs 6 --max-convergence-retries 3 --sampler gibbs
```

Optional full-ablation Python call:

```python
from simulation_project.src.run_experiment import run_exp4_variant_ablation

run_exp4_variant_ablation(
    profile="full",
    repeats=30,
    p0_list=[5, 30],
    include_oracle=True,
    max_convergence_retries=3,
    sampler_backend="gibbs",
    n_jobs=6,
)
```

### Exp5 (`prior_sensitivity`)

Exp5 is also aligned to the Exp3 scale:
- `n=100`, `rho_within=0.3`
- scenarios: `G10x5` and `CL` (both with `p=50`)

Recommended commands:

```bash
# quick quality check
python -m simulation_project.src.run_experiment --experiment 5 --save-dir outputs/simulation_project --profile laptop --repeats 1 --n-jobs 1 --max-convergence-retries 2 --sampler nuts

# development run
python -m simulation_project.src.run_experiment --experiment 5 --save-dir outputs/simulation_project --profile laptop --repeats 15 --n-jobs 2 --max-convergence-retries 1 --sampler nuts

# full-quality run
python -m simulation_project.src.run_experiment --experiment 5 --save-dir outputs/simulation_project --profile full --repeats 30 --n-jobs 2 --max-convergence-retries 2 --sampler nuts
```

## 4. One-Click Full Pipeline

```bash
python -m simulation_project.src.run_experiment --experiment all --save-dir outputs/simulation_project --profile full --n-jobs 2
```

## 5. Quick Convergence Check Script

```powershell
@'
from pathlib import Path
import pandas as pd

base = Path("outputs/simulation_project/results")
experiments = [
    "exp1_kappa_profile_regimes",
    "exp2_group_separation",
    "exp3a_main_benchmark",
    "exp3b_boundary_stress",
    "exp4_variant_ablation",
    "exp5_prior_sensitivity",
]

for exp in experiments:
    csv_path = base / exp / "raw_results.csv"
    if not csv_path.exists():
        print(f"{exp}: missing raw_results.csv")
        continue
    df = pd.read_csv(csv_path)
    if "converged" not in df.columns:
        print(f"{exp}: missing converged column")
        continue
    conv = df["converged"].fillna(False).astype(bool)
    msg = f"{exp}: {conv.sum()}/{len(df)} converged ({conv.mean():.1%})"
    if "bulk_ess_min" in df.columns:
        msg += f", ess_median={df['bulk_ess_min'].median():.1f}"
    print(msg)
'@ | python -
```

## 6. Sweep Runner (Exp1 To Exp5)

List and run sweeps:

```bash
python -m simulation_project.src.run_sweep --list
python scripts/run_sweep.py --list
```

Default config:
- `simulation_project/config/sweeps.yaml`

Typical commands:

```bash
# list sweep names
python -m simulation_project.src.run_sweep --list

# dry-run expansion and validation only
python -m simulation_project.src.run_sweep --sweep exp1_to_exp5 --dry-run

# run one full sweep session
python -m simulation_project.src.run_sweep --sweep exp1_to_exp5

# override sweep params without editing YAML
python -m simulation_project.src.run_sweep --sweep exp1_to_exp5 --set profile=full --set n_jobs=2 --set max_convergence_retries=2
```

Output layout:
- `outputs/simulation_project/sweeps/<sweep_name>/<session_id>/manifest.json`
- `outputs/simulation_project/sweeps/<sweep_name>/<session_id>/runs.csv`
- `outputs/simulation_project/sweeps/<sweep_name>/<session_id>/runs/run_XXX/`

## 7. Per-Run Timestamped Artifacts

After each completed experiment (`exp1` to `exp5`), the runner now creates a
timestamped run archive under that experiment's result directory:

- `session_root = outputs/simulation_project/sessions/<timestamp>_cli_<experiment>`

- `<session_root>/results/<exp_dir>/runs/<YYYYMMDD_HHMMSS>/run_manifest.json`
- `<session_root>/results/<exp_dir>/runs/<YYYYMMDD_HHMMSS>/run_summary_table.csv`
- `<session_root>/results/<exp_dir>/runs/<YYYYMMDD_HHMMSS>/run_summary.md`
- `<session_root>/results/<exp_dir>/runs/<YYYYMMDD_HHMMSS>/run_analysis.json`
- `<session_root>/results/<exp_dir>/runs/<YYYYMMDD_HHMMSS>/artifacts/...`

The experiment directory also keeps:

- `<session_root>/results/<exp_dir>/latest_run.json`

This guarantees every run is reproducibly archived with its timestamp, compact
table summary, markdown summary, and analyzer findings.

Figure outputs are now versioned as well:

- latest figure: `<session_root>/figures/<figure_name>.png`
- immutable history snapshot: `<session_root>/figures/history/<figure_name>_<YYYYMMDD_HHMMSS_microseconds>.png`

So each newly generated figure is preserved with its own timestamp.


