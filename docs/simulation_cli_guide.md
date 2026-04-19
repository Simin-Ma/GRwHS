# Simulation CLI Guide (By Experiment)

本指南只覆盖当前有效实验体系：`simulation_project`。

## 1. 统一入口

```bash
python -m simulation_project.src.run_experiment --help
python scripts/run_simulation.py --help
```

通用参数：
- `--experiment {all,1,2,3,4,5}`
- `--save-dir simulation_project`
- `--seed 20260415`
- `--repeats <int>`
- `--n-jobs <int>`
- `--profile {full,laptop}`
- `--no-enforce-bayes-convergence`
- `--max-convergence-retries <int>`
- `--until-bayes-converged`
- `--sampler {nuts,collapsed,gibbs}`
- Default note: Bayesian methods use at least `4` chains by default (method-level floor).

说明：
- `scripts/run_simulation.py` 是统一封装，行为与 `python -m ...run_experiment` 一致。
- 高级参数（`methods`、`prior_grid`、`p0_list` 等）通过 Python 函数调用传入。

## 2. 收敛与诊断约定

从当前版本开始，`raw_results.csv`（Exp2-Exp5）都会写出：
- `rhat_max`
- `bulk_ess_min`
- `divergence_ratio`
- `runtime_seconds`
- `error`

建议判定：
- 优先看 `converged`。
- 再看 `bulk_ess_min`（Exp5 特别关键）。
- `divergence_ratio` 高通常意味着几何问题，优先提高 `adapt_delta`/`warmup`。

## 3. 按 Exp 的推荐命令

### Exp1 (`kappa_profile_regimes`)

Exp1 是网格后验，不走 MCMC，`--sampler` 对它无效。

```bash
python -m simulation_project.src.run_experiment --experiment 1 --save-dir simulation_project --repeats 1 --n-jobs 1 --no-enforce-bayes-convergence
python -m simulation_project.src.run_experiment --experiment 1 --save-dir simulation_project --repeats 400 --n-jobs 8 --no-enforce-bayes-convergence
```

### Exp2 (`group_separation`)

推荐 `nuts`，并开启收敛约束。

```bash
python -m simulation_project.src.run_experiment --experiment 2 --save-dir simulation_project --profile laptop --repeats 30 --n-jobs 6 --max-convergence-retries 2 --sampler nuts
python -m simulation_project.src.run_experiment --experiment 2 --save-dir simulation_project --profile full --repeats 100 --n-jobs 6 --until-bayes-converged --sampler nuts
```

### Exp3 (`linear_benchmark`)

维度和组合都多，推荐有限重试。
当前代码默认 GIGG 使用 `btrick=False`，Exp3 建议 `--max-convergence-retries 1`。

```bash
python -m simulation_project.src.run_experiment --experiment 3 --save-dir simulation_project --profile laptop --repeats 20 --n-jobs 8 --max-convergence-retries 1 --sampler nuts
python -m simulation_project.src.run_experiment --experiment 3 --save-dir simulation_project --profile full --repeats 100 --n-jobs 8 --max-convergence-retries 1 --sampler nuts
```

### Exp4 (`variant_ablation`)

Exp4 current design is a compact mechanism ablation aligned with Exp3 scale:
- DGP scale: `n=100`, `p=50`, `group_sizes=[10]*5`
- `p0 in {5, 15, 30}`
- variants: `oracle`, `calibrated`, `fixed_10x`, `RHS_oracle`

Exp4 is sensitive to GR_RHS variants; `collapsed` remains the recommended sampler. To avoid ambiguity around `--until-bayes-converged`, use explicit retry budget in commands.

```bash
python -m simulation_project.src.run_experiment --experiment 4 --save-dir simulation_project --profile laptop --repeats 20 --n-jobs 6 --max-convergence-retries 2 --sampler collapsed
python -m simulation_project.src.run_experiment --experiment 4 --save-dir simulation_project --profile full --repeats 50 --n-jobs 6 --max-convergence-retries 2 --sampler collapsed
```

### Exp5 (`prior_sensitivity`) - 当前高质量推荐

Exp5 current design is also aligned with Exp3 scale:
- `n=100`, `rho_within=0.3`
- scenarios: `G10x5` and `CL` (both `p=50`)

当前代码已为 Exp5 加入“质量优先”采样配置（不需要手动改代码）：
- `laptop`: `chains=2, warmup=800, draws=800, ess_threshold=200`
- `full`: `chains=4, warmup=1500, draws=1500, ess_threshold=400`
- 单链时 R-hat 自动跳过，避免伪失败。
- 未显式设置 `--max-convergence-retries` 时，默认按 profile 取值：`full=2`，`laptop=1`。
- 若需要“直到收敛”为止的模式，请显式设置：`--max-convergence-retries -1`。
- 重试不会放宽 `R-hat/ESS/divergence` 阈值，只会增加 warmup/draw 与迭代预算。

命令建议：

```bash
# 快速质量检查（建议先跑）
python -m simulation_project.src.run_experiment --experiment 5 --save-dir simulation_project --profile laptop --repeats 1 --n-jobs 1 --max-convergence-retries 2 --sampler nuts

# 开发版
python -m simulation_project.src.run_experiment --experiment 5 --save-dir simulation_project --profile laptop --repeats 10 --n-jobs 2 --max-convergence-retries 2 --sampler nuts

# 正式版（高质量）
python -m simulation_project.src.run_experiment --experiment 5 --save-dir simulation_project --profile full --repeats 30 --n-jobs 2 --max-convergence-retries 2 --sampler nuts
```

备注：
- Exp5 现在更推荐 `nuts`（当前实现下 `collapsed` 在该实验上成本偏高）。
- Exp5 使用 group-level 语义：`p0_signal_groups` + `tau_target="groups"`（已内置）。

## 4. 一键全量（按推荐顺序）

```bash
python -m simulation_project.src.run_experiment --experiment 1 --save-dir simulation_project --repeats 400 --n-jobs 8 --no-enforce-bayes-convergence
python -m simulation_project.src.run_experiment --experiment 2 --save-dir simulation_project --profile full --repeats 100 --n-jobs 6 --until-bayes-converged --sampler nuts
python -m simulation_project.src.run_experiment --experiment 3 --save-dir simulation_project --profile full --repeats 100 --n-jobs 8 --max-convergence-retries 1 --sampler nuts
python -m simulation_project.src.run_experiment --experiment 4 --save-dir simulation_project --profile full --repeats 50 --n-jobs 6 --max-convergence-retries 2 --sampler collapsed
python -m simulation_project.src.run_experiment --experiment 5 --save-dir simulation_project --profile full --repeats 30 --n-jobs 2 --max-convergence-retries 2 --sampler nuts
```

## 5. 快速检查各 Exp 收敛率

```powershell
@'
from pathlib import Path
import pandas as pd

base = Path("simulation_project/results")
exps = [
    "exp1_kappa_profile_regimes",
    "exp2_group_separation",
    "exp3_linear_benchmark",
    "exp4_variant_ablation",
    "exp5_prior_sensitivity",
]

for e in exps:
    f = base / e / "raw_results.csv"
    if not f.exists():
        print(f"{e}: missing raw_results.csv")
        continue
    df = pd.read_csv(f)
    if "converged" not in df.columns:
        print(f"{e}: no converged column")
        continue
    conv = df["converged"].fillna(False).astype(bool)
    msg = f"{e}: {conv.sum()}/{len(df)} converged ({conv.mean():.1%})"
    if "bulk_ess_min" in df.columns:
        msg += f", ess_median={df['bulk_ess_min'].median():.1f}"
    print(msg)
'@ | python -
```

## 6. Sweep 架构（统一跑 Exp1-Exp5）

现在支持统一 sweep 调度器：

```bash
python -m simulation_project.src.run_sweep --list
python scripts/run_sweep.py --list
```

默认配置文件：
- `simulation_project/config/sweeps.yaml`

常用命令（现在只需要一个 sweep 名）：

```bash
# 查看可用 sweep 名称
python -m simulation_project.src.run_sweep --list

# 先做 dry-run（只展开参数并校验，不执行）
python -m simulation_project.src.run_sweep --sweep exp1_to_exp5 --dry-run

# 真正执行一个 sweep（一次跑完整 Exp1-Exp5）
python -m simulation_project.src.run_sweep --sweep exp1_to_exp5

# 覆盖 sweep 参数（不改 yaml）
python -m simulation_project.src.run_sweep --sweep exp1_to_exp5 --set profile=full --set n_jobs=2 --set max_convergence_retries=2
```

输出结构：
- `simulation_project/sweeps/<sweep_name>/<session_id>/manifest.json`
- `simulation_project/sweeps/<sweep_name>/<session_id>/runs.csv`
- `simulation_project/sweeps/<sweep_name>/<session_id>/runs/run_XXX/`（每组独立输出）
