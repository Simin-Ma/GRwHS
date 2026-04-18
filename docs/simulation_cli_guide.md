# Simulation 命令行运行手册（按 Exp 分类）

本手册只针对当前唯一有效实验体系：`simulation_project`。

## 1. 统一入口

```bash
python -m simulation_project.src.run_experiment --help
python scripts/run_simulation.py --help
```

统一 CLI 参数（所有 Exp 都可用）：

- `--experiment {all,1,2,3,4,5}`
- `--save-dir simulation_project`
- `--seed 20260415`
- `--repeats <int>`
- `--n-jobs <int>`
- `--profile {full,laptop}`
- `--no-enforce-bayes-convergence`
- `--max-convergence-retries <int>`
- `--until-bayes-converged`
- `--sampler {nuts,collapsed,gibbs}`  GR-RHS 后验采样器选择

说明：

- `scripts/run_simulation.py` 是统一入口封装，行为与 `python -m ...run_experiment` 一致。
- `methods`、`signal_types`、`rho_list`、`snr_list`、`prior_grid`、`p0_list` 等高级参数，需要通过 Python 函数调用传入。

## 2. 后验计算模式选择（全局建议）

代码库中有三种后验计算策略（均在 `grrhs_nuts.py`）：

| 策略 | `--sampler` 值 | 采样空间 | 适用场景 |
|------|--------------|---------|---------|
| `GRRHS_NUTS` | `nuts`（默认） | `2p+2G+2`（full）/ `p+G+2`（profile） | 通用，含 Logistic |
| `GRRHS_CollapsedNUTS` | `collapsed` | `p+2G+2`（full）/ **`G+2`（profile）** | GR_RHS 专项 + Gaussian，profile 模式最高效 |
| `GRRHS_Gibbs` | `gibbs` | 同 NUTS，但 β 解析抽样 | 纯 NumPy，无 JAX 依赖；Gaussian only |

**profile 模式**（`lambda_j=1, a_g=1`）对 GR_RHS 专项实验（Exp1/4/5）最合适：
- NUTS 维度从 `2p+2G+2` 降至 `p+G+2`，梯度更稳定，ESS 更高
- Exp2/3 因需要多方法公平比较，用 `full` 模式

**收敛策略优先级：**

```bash
# 正式结果（推荐）：强制收敛，自动重试直到通过
--until-bayes-converged

# 中等成本：最多重试 2 次
--max-convergence-retries 2

# 快速摸底：不强制收敛
--no-enforce-bayes-convergence
```

## 3. 按 Exp 分类的建议与命令

### Exp1: `kappa_profile_regimes`

**最合适模式：不强制收敛**  
**`--sampler`：无效**（Exp1 使用解析网格计算 `kappa_posterior_grid`，不走 MCMC，`--sampler` 参数对此实验没有作用）

- 目标是机制验证（后验 kappa 网格），对单次收敛不敏感，repeats 量大即可
- `--no-enforce-bayes-convergence` 避免在大网格上反复重试拖慢速度

```bash
# 快速测试（1 次，确认能跑通）
python -m simulation_project.src.run_experiment --experiment 1 --save-dir simulation_project --repeats 1 --n-jobs 1 --no-enforce-bayes-convergence

# 开发版
python -m simulation_project.src.run_experiment --experiment 1 --save-dir simulation_project --repeats 50 --n-jobs 6 --no-enforce-bayes-convergence

# 正式推荐
python -m simulation_project.src.run_experiment --experiment 1 --save-dir simulation_project --repeats 400 --n-jobs 8 --no-enforce-bayes-convergence
```

```powershell
# 高级网格控制
@'
from simulation_project.src.run_experiment import run_exp1_kappa_profile_regimes

run_exp1_kappa_profile_regimes(
    save_dir="simulation_project",
    repeats=400,
    n_jobs=8,
    pg_null_list=[10, 20, 50, 100, 200, 500, 1000, 2000],
    tau_null=0.5,
    tail_eps=0.1,
    pg_phase_list=[30, 60, 120, 240, 480],
    tau_phase_list=[0.1, 0.3, 0.5, 1.0],
    xi_multiplier_list=[0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.15, 1.3, 1.5, 2.0],
    u0=0.5,
    sigma2_phase=1.0,
    alpha_kappa=0.5,
    beta_kappa=1.0,
    enforce_bayes_convergence=False,
)
'@ | python -
```

---

### Exp2: `group_separation`

**最合适模式：full + 强制收敛**  
**`--sampler nuts`**（推荐默认）

- 多方法对比（GR_RHS vs RHS vs GIGG_MMLE 等），`--sampler` 只影响 GR_RHS，其他方法不受影响
- `nuts` 最稳定、测试最充分；`collapsed` 也可用（仅对 GR_RHS 提升 ESS，不影响其他方法公平性）
- 这是核心基准实验，收敛质量直接影响结论可信度
- `repeats` 建议：开发 20~40，正式 80~120

```bash
# 快速测试
python -m simulation_project.src.run_experiment --experiment 2 --save-dir simulation_project --profile laptop --repeats 1 --n-jobs 1 --no-enforce-bayes-convergence --sampler nuts

# 开发版
python -m simulation_project.src.run_experiment --experiment 2 --save-dir simulation_project --profile laptop --repeats 30 --n-jobs 6 --no-enforce-bayes-convergence --sampler nuts

# 正式推荐
python -m simulation_project.src.run_experiment --experiment 2 --save-dir simulation_project --profile full --repeats 100 --n-jobs 6 --sampler nuts
```

```powershell
# 指定方法子集 + 有限重试
@'
from simulation_project.src.run_experiment import run_exp2_group_separation

run_exp2_group_separation(
    save_dir="simulation_project",
    repeats=80,
    n_jobs=6,
    profile="full",
    methods=["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus"],
    enforce_bayes_convergence=True,
    max_convergence_retries=2,
    until_bayes_converged=False,
    rho_ref=0.1,
    n_test=200,
    sampler_backend="nuts",
)
'@ | python -
```

---

### Exp3: `linear_benchmark`

**最合适模式：full + 有限重试**  
**`--sampler nuts`**（推荐默认）

- 最重的多因子实验（signal × rho × snr × method），建议先跑子网格
- `--sampler` 只影响 GR_RHS；理由同 Exp2 — `nuts` 最稳定，`collapsed` 可选但收益有限（Exp3 p=50，NUTS 维度差距不如 Exp4 显著）
- `--until-bayes-converged` 成本太高，用 `--max-convergence-retries 2` 即可
- `repeats` 建议：开发 10~30，正式 80~120

```bash
# 快速测试
python -m simulation_project.src.run_experiment --experiment 3 --save-dir simulation_project --profile laptop --repeats 1 --n-jobs 1 --no-enforce-bayes-convergence --sampler nuts

# 开发版（laptop 快速摸底）
python -m simulation_project.src.run_experiment --experiment 3 --save-dir simulation_project --profile laptop --repeats 20 --n-jobs 8 --no-enforce-bayes-convergence --sampler nuts

# 正式推荐
python -m simulation_project.src.run_experiment --experiment 3 --save-dir simulation_project --profile full --repeats 100 --n-jobs 8 --max-convergence-retries 2 --sampler nuts
```

```powershell
# 子网格先行（推荐开发阶段）
@'
from simulation_project.src.run_experiment import run_exp3_linear_benchmark

run_exp3_linear_benchmark(
    save_dir="simulation_project",
    repeats=40,
    n_jobs=8,
    profile="full",
    signal_types=["concentrated", "boundary"],
    rho_list=[0.0, 0.3],
    snr_list=[0.7, 2.0],
    methods=["GR_RHS", "RHS", "GIGG_MMLE", "GHS_plus", "OLS", "LASSO_CV"],
    enforce_bayes_convergence=True,
    max_convergence_retries=2,
    until_bayes_converged=False,
    n_test=200,
    sampler_backend="nuts",
)
'@ | python -
```

---

### Exp4: `variant_ablation`

**最合适模式：full + until-bayes-converged**  
**`--sampler collapsed`** ← 强烈推荐

- 只跑 GR_RHS 变体，全部 Gaussian，`collapsed` 完全适用
- DGP: p=80（4组×20变量），G=4。NUTS 维度对比：
  - `nuts`：p+G+2 = **86 维**
  - `collapsed`（profile 模式）：G+2 = **6 维** ← 降低 93%，梯度更稳定，ESS 更高
- τ 校准对收敛质量很敏感，`collapsed` 的低维 NUTS 在过收缩/欠收缩 variant 上更容易收敛
- `repeats` 建议：开发 15~30，正式 40~60

```bash
# 快速测试
python -m simulation_project.src.run_experiment --experiment 4 --save-dir simulation_project --profile laptop --repeats 1 --n-jobs 1 --no-enforce-bayes-convergence --sampler collapsed

# 开发版
python -m simulation_project.src.run_experiment --experiment 4 --save-dir simulation_project --profile laptop --repeats 20 --n-jobs 6 --max-convergence-retries 2 --sampler collapsed

# 正式推荐
python -m simulation_project.src.run_experiment --experiment 4 --save-dir simulation_project --profile full --repeats 50 --n-jobs 6 --until-bayes-converged --sampler collapsed
```

```powershell
# 指定 p0 档位
@'
from simulation_project.src.run_experiment import run_exp4_variant_ablation

run_exp4_variant_ablation(
    save_dir="simulation_project",
    repeats=50,
    n_jobs=6,
    profile="full",
    p0_list=[2, 8, 20],
    enforce_bayes_convergence=True,
    max_convergence_retries=None,
    until_bayes_converged=True,
    sampler_backend="collapsed",
)
'@ | python -
```

---

### Exp5: `prior_sensitivity`

**最合适模式：full + 有限重试**  
**`--sampler collapsed`** ← 推荐

- 只跑 GR_RHS，全部 Gaussian，`collapsed` 完全适用
- DGP: 两个 scenario（等组 6×20，不等组 50/30/10/5/3），G=6 或 5。NUTS 维度对比：
  - `nuts`：p+G+2 ≈ **128 维**（等组 scenario，p=120）
  - `collapsed`（profile 模式）：G+2 = **8 维** ← 降低 94%
- Paired 设计（同一 DGP 跑多个先验），用 `collapsed` 各先验配置的 ESS 更稳定，先验比较更可信
- `--until-bayes-converged` 成本较高（每个先验配置独立重试），用 `--max-convergence-retries 2` 即可
- `repeats` 建议：开发 10~20，正式 30~50

```bash
# 快速测试
python -m simulation_project.src.run_experiment --experiment 5 --save-dir simulation_project --profile laptop --repeats 1 --n-jobs 1 --no-enforce-bayes-convergence --sampler collapsed

# 开发版
python -m simulation_project.src.run_experiment --experiment 5 --save-dir simulation_project --profile laptop --repeats 15 --n-jobs 6 --max-convergence-retries 2 --sampler collapsed

# 正式推荐
python -m simulation_project.src.run_experiment --experiment 5 --save-dir simulation_project --profile full --repeats 40 --n-jobs 6 --max-convergence-retries 2 --sampler collapsed
```

```powershell
# 自定义 prior_grid
@'
from simulation_project.src.run_experiment import run_exp5_prior_sensitivity

run_exp5_prior_sensitivity(
    save_dir="simulation_project",
    repeats=35,
    n_jobs=6,
    profile="full",
    prior_grid=[(0.5, 1.0), (1.0, 1.0), (0.5, 0.5), (2.0, 5.0), (1.0, 3.0)],
    enforce_bayes_convergence=True,
    max_convergence_retries=2,
    until_bayes_converged=False,
    sampler_backend="collapsed",
)
'@ | python -
```

---

## 4. 一键快速测试（每个 Exp 各跑 1 次确认无报错）

```bash
# Exp1: 无 --sampler（不走 MCMC）
python -m simulation_project.src.run_experiment --experiment 1 --save-dir simulation_project --repeats 1 --n-jobs 1 --no-enforce-bayes-convergence
# Exp2/3: --sampler nuts
python -m simulation_project.src.run_experiment --experiment 2 --save-dir simulation_project --profile laptop --repeats 1 --n-jobs 1 --no-enforce-bayes-convergence --sampler nuts
python -m simulation_project.src.run_experiment --experiment 3 --save-dir simulation_project --profile laptop --repeats 1 --n-jobs 1 --no-enforce-bayes-convergence --sampler nuts
# Exp4/5: --sampler collapsed（推荐）
python -m simulation_project.src.run_experiment --experiment 4 --save-dir simulation_project --profile laptop --repeats 1 --n-jobs 1 --no-enforce-bayes-convergence --sampler collapsed
python -m simulation_project.src.run_experiment --experiment 5 --save-dir simulation_project --profile laptop --repeats 1 --n-jobs 1 --no-enforce-bayes-convergence --sampler collapsed
```

## 5. 一键正式运行（推荐顺序）

```bash
# Exp1: 网格计算，无 sampler 选项
python -m simulation_project.src.run_experiment --experiment 1 --save-dir simulation_project --repeats 400 --n-jobs 8 --no-enforce-bayes-convergence

# Exp2: 多方法基准，nuts（最稳定）
python -m simulation_project.src.run_experiment --experiment 2 --save-dir simulation_project --profile full --repeats 100 --n-jobs 6 --sampler nuts

# Exp3: 多因子基准，nuts，有限重试
python -m simulation_project.src.run_experiment --experiment 3 --save-dir simulation_project --profile full --repeats 100 --n-jobs 8 --max-convergence-retries 2 --sampler nuts

# Exp4: GR_RHS 变体，collapsed（86维→6维），直到收敛
python -m simulation_project.src.run_experiment --experiment 4 --save-dir simulation_project --profile full --repeats 50 --n-jobs 6 --until-bayes-converged --sampler collapsed

# Exp5: 先验敏感性，collapsed（128维→8维），有限重试
python -m simulation_project.src.run_experiment --experiment 5 --save-dir simulation_project --profile full --repeats 40 --n-jobs 6 --max-convergence-retries 2 --sampler collapsed
```

## 6. 输出目录（按 Exp）

- `simulation_project/results/exp1_kappa_profile_regimes/`
- `simulation_project/results/exp2_group_separation/`
- `simulation_project/results/exp3_linear_benchmark/`
- `simulation_project/results/exp4_variant_ablation/`
- `simulation_project/results/exp5_prior_sensitivity/`

图表统一输出：`simulation_project/figures/`  
表格统一输出：`simulation_project/tables/`  
日志统一输出：`simulation_project/logs/`

## 7. 按 Exp 检查收敛结果

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
    print(f"{e}: {conv.sum()}/{len(df)} converged ({conv.mean():.1%})")
'@ | python -
```
