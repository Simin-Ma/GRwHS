# GR-RHS 真实数据集执行方案

## 1. 文档目标

本文档用于把 `GR-RHS` 的真实数据验证方案落到当前仓库可执行的工程路径上。目标不是泛泛补一个“真实数据例子”，而是选择最能体现以下结构优势的数据：

- 存在自然 `group structure`
- 组内相关性强
- 组内信号分配不均匀
- 真实任务能够直接映射到当前 `grouped Gaussian regression` 骨架，或只需有限扩展

本文档优先服务于当前仓库，而不是抽象论文设想。

本方案同样遵循 `convergence-first` 原则：

- Bayesian 方法必须开启 convergence gate，并以 until-converged 模式运行
- 只有 `status = ok` 且 `converged = True` 的结果才有正式讨论价值
- 跨方法主结论应基于 common-converged paired subset，而不是各方法单独过滤后的边际 summary
- smoke 可以用于打通链路和调参，但未收敛结果不算正式实验完成

## 2. 当前仓库可直接复用的骨架

当前工作区已经具备一套真实数据接入骨架，核心入口包括：

- `data/loaders.py`
- `real_data_experiment/src/config.py`
- `real_data_experiment/src/dataset.py`
- `real_data_experiment/src/fitting.py`
- `real_data_experiment/src/evaluation.py`

现有真实数据目录约定也已经成型，参考：

- `data/real/nhanes_2003_2004/processed/runner_ready/`
- `data/real/covid19_trust_experts/processed/runner_ready/`

当前 loader 约定的关键文件格式是：

- `X.npy`
- `y.npy`
- `feature_names.txt`
- `group_map.json`
- 可选的 `C.npy`
- 可选的 `covariate_feature_names.txt`
- 建议保留 `processed/dataset_summary.json`

这意味着本轮最值得优先推进的是：

1. 选择能自然写成 `feature -> one group id` 的成熟数据集
2. 优先选择连续型响应，避免一开始就把任务扩展到 survival / overlapping groups

## 3. 数据集路线总览

| 路线 | 数据集 | 响应变量 | 分组方式 | 与当前架构匹配度 | 工程成本 | 建议 |
|---|---|---|---|---|---|---|
| A1 | `GSE40279` 人类甲基化年龄 | 年龄，连续 | `CpG -> gene` 或 `CpG -> promoter/gene-region` | 很高 | 中等 | 主路线 |
| B1 | `TCGA` RNA-seq + pathway groups | 生存或连续风险分数 | `gene -> pathway` | 当前版本中等偏低 | 高 | 第二阶段 |
| C1 | `ADNI` ROI / imaging genetics | MMSE / ADAS-Cog / diagnosis | `feature -> ROI/network/gene` | 中等 | 高 | 可选扩展 |

## 4. 推荐决策

### 4.1 论文主路线

优先执行：

- 主数据集：`GSE40279`
原因：

- 天然支持连续响应
- 具有高维、强相关、自然 grouped structure
- 比普通 UCI 表格数据更能支持 `GR-RHS` 的结构性叙事
- 能较自然地接入当前 `real_data_experiment` 的 `group_map.json` 设计

### 4.2 为什么不把 TCGA 放在第一阶段

`TCGA` 的学术说服力很强，但当前仓库存在两个实际限制：

1. 当前真实数据 loader 假设 `feature -> single group id`，而 pathway 分组通常是重叠组
2. 最经典的 TCGA group-lasso 真实验证通常是 survival/Cox，而不是当前主干最顺手的 Gaussian regression

所以 TCGA 更适合作为第二阶段扩展，而不是第一阶段主线。

## 5. 主路线一：GSE40279 人类甲基化年龄

## 5.1 数据集定位

`GSE40279` 是 GEO 上非常成熟的人类甲基化年龄数据集，样本量约 `n = 656`，平台为 Illumina 450k array，特征是约 `450k CpG`。这类数据特别适合验证：

- 高维稀疏/收缩
- 组内强相关
- 自然 grouped structure
- 组层与组内层并存的 shrinkage 价值

参考来源：

- GEO: `GSE40279`
- 450k 注释包：`IlluminaHumanMethylation450kanno.ilmn12.hg19`
- `seagull` 论文将甲基化数据作为 sparse-group lasso 真实验证场景

## 5.2 推荐任务定义

- 任务类型：`gaussian`
- 响应变量：`chronological_age`
- 预测特征：甲基化 beta 值
- 主分组方案：`CpG -> nearest gene`
- 敏感性分组方案：`CpG -> promoter/island-region`

主方案优先使用 `nearest gene` 的原因：

- 组是天然的、可解释的
- 更容易构造不重叠分组
- 更贴合当前 `group_map.json` 设计

## 5.3 为什么必须做维度压缩

原始 `450k` 级别特征对当前 Bayesian grouped methods 过重。真实执行中不建议直接把全量矩阵灌入 `runner_ready`。推荐采用双版本数据导出：

- `smoke` 版：`top_k = 2000`
- `main` 版：`top_k = 8000`

如果计算预算允许，可以再增加：

- `extended` 版：`top_k = 12000`

## 5.4 推荐预处理流程

推荐以“先得到稳定可跑的 `runner_ready` 版本”为第一目标。

### 步骤 A：原始输入选择

优先使用 GEO 已提供的 processed beta matrix，而不是从 IDAT 全流程重建。第一版建议直接消费：

- `GSE40279_average_beta.txt.gz`
- 或分块版 `GSE40279_average_beta_GSM*.txt.gz`
- `GSE40279_sample_key.txt.gz`

### 步骤 B：样本与响应整理

- 读取每个样本的年龄作为 `y`
- 保留与 beta matrix 对齐的样本顺序
- 对缺失或异常年龄样本做明确记录

### 步骤 C：特征清理

第一版建议做以下最小清理：

- 删除非数值或全缺失 CpG
- 删除零方差 CpG
- 删除没有可用 gene annotation 的 CpG
- 默认删除 `chrX` / `chrY` 探针，避免把性别差异混入年龄主信号

### 步骤 D：分组映射

对每个 CpG：

1. 从 450k 注释表读取 `gene symbol` 或 nearest-gene 信息
2. 若一个 CpG 对应多个基因，只保留一个稳定规则

推荐规则：

- 先使用注释中的第一个 gene symbol
- 若为空则剔除
- 必须保证每个 feature 最终只映射到一个 `group id`

这是当前仓库能稳定运行的前提。

### 步骤 E：无监督过滤

为控制计算规模，建议按训练无关的无监督准则先做一版固定过滤：

- 全局按方差或 MAD 排序
- `smoke` 保留前 `2000`
- `main` 保留前 `8000`

然后再做组约束：

- 删除过滤后只剩 `1` 个特征的组
- 建议只保留组大小在 `2` 到 `50` 之间的组
- 对超大组，按组内方差截断到前 `30` 或 `50` 个特征

说明：

- 这是“先跑通”的工程方案
- 更严格的无泄漏版本应当把过滤移到每个 split 的训练集内完成
- 但现有 `prepare_split()` 尚未内建特征筛选阶段，所以第一版可以先采用固定无监督过滤，并在文档中明确声明

## 5.5 推荐目录与数据集命名

建议使用以下目录：

```text
data/real/gse40279_methylation_age/
  raw/
  processed/
    analysis_bundle/
    runner_ready_smoke/
    runner_ready_main/
    dataset_summary.json
```

对应建议的 `dataset_id`：

- `gse40279_age_gene_groups_smoke`
- `gse40279_age_gene_groups_main`

## 5.6 推荐 loader 配置形态

建议接入形式与当前 `NHANES` / `trust_experts` 一致：

```yaml
loader:
  path_X: data/real/gse40279_methylation_age/processed/runner_ready_main/X.npy
  path_y: data/real/gse40279_methylation_age/processed/runner_ready_main/y.npy
  path_feature_names: data/real/gse40279_methylation_age/processed/runner_ready_main/feature_names.txt
  path_group_map: data/real/gse40279_methylation_age/processed/runner_ready_main/group_map.json
```

第一版默认：

- `covariate_mode: none`
- `response_standardization: train_center`

如果后续拿到了稳定的 sex / cell-composition covariates，再考虑：

- 增加 `C.npy`
- 设置 `covariate_mode: residualize`

## 7. 第二阶段扩展：TCGA

## 7.1 学术价值

如果论文后续需要更强的“主流 grouped-omics benchmark”说服力，`TCGA` 是最值得扩展的路线之一，尤其是：

- `TCGA-BRCA`
- `TCGA-LUAD`
- `TCGA-KIRC`

## 7.2 当前版本不建议直接作为第一阶段主线

主要原因不是数据不好，而是当前工程抽象还没有完全对齐：

- pathway groups 通常是 overlap 的
- 真实文献常用 survival/Cox
- 当前真实数据骨架更自然支持 `gaussian + disjoint groups`

## 7.3 如果要做 TCGA，建议先补的能力

至少补以下其一：

1. `overlapping group metadata` 支持
2. `survival/Cox` 任务支持

如果短期内不扩展模型任务，则可退而求其次：

- 仅做非重叠的 gene-family / chromosome-block 分组
- 或先做 pathway-level aggregated features，但这会弱化你方法的“组内局部适应”卖点

因此 TCGA 更适合写成第二阶段路线，而非立刻执行的主线。

## 8. 可选扩展：ADNI

`ADNI` 可以作为非组学补充，展示方法在 imaging / ROI grouped design 下也有意义。

但当前不建议优先推进，原因是：

- 访问需要申请
- 数据清洗与预处理较重
- 对当前论文主叙事的边际收益不如甲基化路线高

## 9. 真实数据实验指标建议

当前 `real_data_experiment/src/evaluation.py` 已经在输出一些很适合论文写作的指标。建议主表和附表分别强调不同层面。

### 9.1 主表指标

- `rmse_test`
- `mae_test`
- `r2_test`
- `lpd_test`

### 9.2 结构性附表或附图指标

- `group_selected_count`
- `group_selected_fraction`
- `group_norm_entropy`
- `top_groups_json`
- `kappa_group_mean_json`
- `kappa_group_prob_gt_0_5_json`
- `bridge_ratio_mean`

### 9.3 重复划分建议

推荐：

- `smoke`: `repeats = 2`
- `main`: `repeats = 10`
- `paper`: `repeats = 20`，如果算力允许

所有比较都尽量保持 paired split，正式结论以 common-converged paired subset 为准。

## 10. 分阶段执行计划

## 阶段 0：先确认主路线

结论已经足够明确：

- 先做 `GSE40279`
- `TCGA` 放到第二阶段

## 阶段 1：构造 `GSE40279` 的 `runner_ready_smoke`

交付物：

- `data/real/gse40279_methylation_age/raw/*`
- `data/real/gse40279_methylation_age/processed/runner_ready_smoke/*`
- `data/real/gse40279_methylation_age/processed/dataset_summary.json`
- 预处理脚本

完成标准：

- `load_real_dataset()` 可正常加载
- `load_prepared_real_dataset()` 可正常返回
- `prepare_split()` 可正常切分

## 阶段 2：注册到真实数据配置

交付物：

- 在 `real_data_experiment/src/config.py` 注册一个或多个 dataset spec
- 如需要，补 `real_data_experiment/config/real_data.yaml`

完成标准：

- 新数据集在 manifest 中可见
- split 与 metadata 可保存
- convergence gate 配置继续保持 `enforce_bayes_convergence = true` 与 `max_convergence_retries = -1`

## 阶段 3：核对并加固 runner/CLI 闭环

当前工作区已经有 `real_data_experiment/src/runner.py` 与 `real_data_experiment/src/cli/run_real_data_cli.py`，但真实执行前仍应核对它们是否继续保持闭环与 convergence-first 约束。

因此真实执行前应先检查并必要时补强：

- `real_data_experiment/src/runner.py`
- `real_data_experiment/src/cli/run_real_data_cli.py`

完成标准：

- `run-real-data` 会强制启用 until-converged gate
- 输出结果继续保留 paired summary 与 paper-table 闭环
- 正式比较路径默认落到 common-converged paired summary

## 阶段 4：`smoke` 运行

推荐目标：

- 只跑 `GSE40279 smoke`
- `repeats = 2`
- 降低 sampler 成本

完成标准：

- 能稳定出 split
- 能完成至少 `GR_RHS` 与 `RHS` 的成对比较
- 能输出结果表和 summary csv/json
- Bayesian 方法的正式结果必须满足 `status = ok` 且 `converged = True`
- 至少产出一组 common-converged paired 比较；若没有共同收敛 repeat，则该 smoke 只能算链路验证，不算正式实验完成

## 阶段 5：`main` 运行

推荐目标：

- `GSE40279 main`
- `repeats = 10`

完成标准：

- 主结论使用 common-converged paired summary / paper tables
- 明确报告 `n_converged`、`n_paired` 与 common-rate
- 如果共同收敛覆盖率不足，先诊断收敛问题，再决定是否扩大 repeats 或调整预算

## 阶段 6：论文级整理

输出建议：

- 主文：`GSE40279 main`
- 结构性图表：组选择、`kappa_g`、top groups、bridge ratio

## 11. 主要风险与规避

### 风险 1：维度太高，Bayesian 方法运行过慢

规避：

- 先做 `smoke` / `main` 双版本
- 控制 `top_k`
- 控制组大小上限

### 风险 2：group map 不是天然不重叠

规避：

- 第一阶段只做 `nearest gene`
- 每个 feature 只保留一个 group id

### 风险 3：固定方差过滤存在轻微信息泄漏争议

规避：

- 第一版明确写为工程近似
- 第二版如有需要，把特征过滤移到 split 内部并仅用训练集统计量

### 风险 4：TCGA 很吸引人，但当前仓库抽象还没对齐

规避：

- 不把 TCGA 放在第一阶段里硬做
- 先用甲基化年龄路线建立真实数据主结果

## 12. 外部参考链接

- GEO `GSE40279`: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279
- Illumina 450k annotation package: https://bioconductor.org/packages/IlluminaHumanMethylation450kanno.ilmn12.hg19/
- `seagull` / sparse-group lasso real-data甲基化示例: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03725-w
- TCGA 路径型 group lasso 示例: https://pmc.ncbi.nlm.nih.gov/articles/PMC8667553/
- GDC portal: https://gdc.cancer.gov/access-data/gdc-data-portal
- ADNI data access: https://adni.loni.usc.edu/data-samples/adni-data/
