# GR-RHS 真实数据执行 Prompt 集

## 使用说明

本文档给出的是可直接复制给 Codex 的执行 prompt。建议不要一次性把所有任务塞进一个 prompt，而是按阶段逐个执行。

建议顺序：

1. 先用 Prompt 1 生成 `GSE40279 smoke`
2. 再用 Prompt 2 注册配置并做 split smoke test
3. 再用 Prompt 3 检查并补齐 real-data runner
4. 最后再用 Prompt 4 或 Prompt 5 跑完整实验

如果你想一步推进，也可以直接使用文末的“总控 prompt”。

## Prompt 1：构造 `GSE40279` runner-ready smoke 数据

```text
请基于当前仓库，为 GR-RHS 真实数据实验接入一个新的成熟数据集：GEO 的 GSE40279（human methylation age）。

目标：
1. 下载或整理 GSE40279 的 processed beta matrix 和 sample key
2. 在仓库中创建 data/real/gse40279_methylation_age/
3. 产出一个可被 data/loaders.py 直接加载的 runner-ready smoke 版本
4. 同时保留可复现的预处理脚本和 dataset_summary.json

硬性约束：
1. 必须遵守当前仓库现有 runner_ready 目录约定，参考 data/real/nhanes_2003_2004 和 data/real/covid19_trust_experts
2. 必须输出以下文件：
   - processed/runner_ready_smoke/X.npy
   - processed/runner_ready_smoke/y.npy
   - processed/runner_ready_smoke/feature_names.txt
   - processed/runner_ready_smoke/group_map.json
   - processed/dataset_summary.json
3. 必须保留一个可复现的预处理脚本，优先放在 scripts/ 或 data/real/gse40279_methylation_age/processed/analysis_bundle/
4. 分组必须满足当前 loader 的 single-group-id 约束，也就是每个 feature 只能属于一个 group
5. 第一版按 nearest-gene 规则分组，不要做 overlapping pathways
6. 第一版只做 gaussian 回归，响应变量用 chronological age
7. 必须先做 smoke 版，目标 top_k=2000，不要直接做 full 450k
8. 过滤必须是无监督的，不要使用 y 做 supervised screening
9. 默认删除没有 gene annotation 的 CpG，并默认删除 chrX/chrY 探针
10. 组大小太小时要删组；组太大时要做组内截断，并把规则写进 dataset_summary.json

建议执行细节：
1. 优先使用 GEO 已提供的 processed beta matrix，而不是重建 IDAT 流程
2. 先把原始文件落到 raw/
3. 用注释包或等价注释表将 CpG 映射到基因
4. 全局按方差或 MAD 选前 2000 个特征
5. 删除过滤后只剩 1 个特征的组
6. 生成 feature_names.txt 与 group_map.json
7. 写 dataset_summary.json，记录样本数、特征数、组数、过滤规则、原始来源链接、日期、输入文件名

验证要求：
1. 用当前 data.loaders.load_real_dataset 实测加载
2. 报告 X、y、group 数量和前几个 group size
3. 如果数据下载、注释或格式存在阻塞，不要空泛描述，直接报告卡住的精确文件名、URL 和下一步最小阻塞点

请直接在仓库里完成改动，不只给方案。
```

## Prompt 2：把 `GSE40279 smoke` 注册到真实数据配置并做 split 测试

```text
请把已经准备好的 GSE40279 runner-ready smoke 数据接入当前仓库的 real_data_experiment 骨架，并完成最小 split 级验证。

目标：
1. 在 real_data_experiment/src/config.py 中注册一个新数据集 spec
2. dataset_id 使用 gse40279_age_gene_groups_smoke
3. 保持 task=gaussian
4. 通过 load_prepared_real_dataset() 和 prepare_split() 跑通一次 smoke 检查
5. 如有必要，补一个 real_data.yaml 模板或最小测试脚本

硬性约束：
1. 不要破坏已有 NHANES 和 covid19_trust_experts 配置
2. 新 dataset spec 的 loader 路径必须指向 runner_ready_smoke
3. 第一版 covariate_mode 使用 none
4. response_standardization 使用 train_center
5. repeats 先设成 2
6. test_fraction 可以先设 0.2
7. 必须把 target_label、notes、group_labels 处理清楚

验证要求：
1. 实际调用 load_prepared_real_dataset()
2. 实际调用 prepare_split(... replicate_id=1 ...)
3. 报告 train/test shape、group_count、split_hash
4. 如果当前 real_data_experiment 骨架因为缺 runner/cli 或其他问题不完整，请精确指出缺哪个文件、哪个 import 或哪个调用链

请直接修改仓库并完成验证。
```

## Prompt 3：检查并补齐 `real_data_experiment` 的 runner/CLI 闭环

```text
请检查当前仓库中的 real_data_experiment 是否已经具备完整可运行闭环；如果没有，请补齐一个最小可用版本，风格参考 simulation_second 和 simulation_mechanism。

目标：
1. 检查 real_data_experiment/src/runner.py 是否存在
2. 检查 real_data_experiment/src/cli/run_real_data_cli.py 是否存在
3. 如果缺失，则补齐最小 runner/CLI，使真实数据实验至少能：
   - 列出数据集
   - 运行某个 dataset_id 的 repeats
   - 保存 split
   - 调用 fitting.py 和 evaluation.py
   - 输出基础 results csv/json
4. 保持和现有 output history 风格一致

硬性约束：
1. 不要改动 simulation_second 或 simulation_mechanism 的既有行为
2. 尽量复用现有 config.py、dataset.py、fitting.py、evaluation.py
3. 保存目录风格对齐 outputs/history/real_data_experiment/main
4. 保持 Bayesian convergence gate 的使用逻辑与现有配置一致
5. 如果发现 __init__.py 中引用了不存在的 runner，请一起修复

验证要求：
1. 至少跑一个 dataset_id 的 smoke run
2. repeats=1 或 2 即可
3. 报告生成了哪些结果文件
4. 如果运行时间过长，优先做最小可验证闭环，不要盲目全量跑

请直接实现，不要只停留在分析。
```

## Prompt 4：跑 `GSE40279 smoke` 真实数据实验并输出可读总结

```text
请基于当前仓库，为 gse40279_age_gene_groups_smoke 跑一次最小真实数据实验，并给出结果摘要。

目标：
1. 运行 real_data_experiment 中的 smoke 实验
2. 方法至少包括 GR_RHS 和 RHS
3. repeats 先跑 2
4. 保存 split 与结果文件
5. 输出一个便于论文判断的简洁总结

硬性约束：
1. 不要扩大到 GSE80672 或 TCGA
2. 如果 sampler 太慢，可以降低 smoke 阶段预算，但不要改坏主配置
3. 必须保证 paired split
4. 必须记录 rmse_test、mae_test、r2_test、lpd_test
5. 同时导出 group_selected_count、group_norm_entropy、top_groups_json、kappa_group_mean_json

总结要求：
1. 给出每个 repeat 的核心指标
2. 给出 GR_RHS 相对 RHS 的 paired 差值
3. 指出结果更像“预测优势”还是“结构性差异更明显而预测差异温和”
4. 如果结果不稳定，明确说出是算力问题、收敛问题还是数据集问题

请直接执行并汇总结果。
```

## Prompt 5：构造 `GSE40279 main` 版本并准备论文级真实数据结果

```text
请在已经跑通 GSE40279 smoke 的基础上，构造一个更适合论文主结果的 main 版本，并尽量复用现有预处理脚本与真实数据骨架。

目标：
1. 新建 runner_ready_main
2. 建议 top_k=8000
3. dataset_id 使用 gse40279_age_gene_groups_main
4. 注册到 real_data_experiment 配置
5. 跑一个 main 级别的真实数据实验

硬性约束：
1. 过滤规则必须写入 dataset_summary.json
2. 继续使用 disjoint nearest-gene groups
3. 不要改成 pathway overlaps
4. 保持 Gaussian regression 任务
5. 主表指标与结构性指标都要保留

运行建议：
1. 先做 repeats=2 的 main smoke
2. 如果可行，再扩到 repeats=10
3. 输出一版适合写进 paper 的结果摘要

请直接完成仓库改动、运行和总结。
```

## Prompt 6：增加确认性数据集 `GSE80672`

```text
请在 GSE40279 跑通之后，为 GR-RHS 真实数据实验增加第二个成熟确认性数据集：GSE80672（mouse methylation age）。

目标：
1. 创建 data/real/gse80672_mouse_methylation_age/
2. 构造 runner_ready_smoke
3. 优先使用 processed supplementary files；如果格式复杂，再明确报告阻塞点
4. 使用连续年龄作为 y
5. 采用 nearest-gene 或等价的单组映射规则，保证 group_map.json 满足 single-group-id 约束

硬性约束：
1. 这次不要碰 TCGA
2. 仍然保持 gaussian 任务
3. 仍然先做 smoke 版
4. 必须记录与 GSE40279 的差异：样本数、特征数、组数、预处理难点
5. 如果 raw/processed 文件结构过重，请先产出一个最小可跑子集

完成后：
1. 注册 dataset spec
2. 跑一次 smoke split 检查
3. 如果有余力，跑 repeats=2 的最小实验
```

## Prompt 7：第二阶段扩展到 `TCGA`

```text
请评估并推进 GR-RHS 真实数据实验的第二阶段扩展：TCGA grouped omics benchmark，但要以当前仓库真实能力为准，不要脱离现有抽象空转。

目标：
1. 先检查当前真实数据骨架是否支持 overlapping groups
2. 检查当前方法层是否支持 survival/Cox
3. 如果两者都不支持，请不要直接硬接 TCGA 主线数据，而是先写出最小可执行扩展路线
4. 优先评估 TCGA-BRCA、TCGA-LUAD、TCGA-KIRC

输出要求：
1. 明确列出当前仓库与 TCGA pathway benchmark 的错位点
2. 给出最小改造集：
   - 是先补 overlap-aware groups
   - 还是先补 survival task
   - 还是先退而求其次做 non-overlapping grouped RNA-seq regression
3. 如果能在当前架构下做一个简化版本，请直接实现一个最小原型
4. 如果不适合立刻实现，请产出一份工程设计说明文档

注意：
1. 不要为了追求 TCGA 名气，牺牲当前架构一致性
2. 如果最终判断 TCGA 暂不值得先做，请明确给出结论与理由
```

## Prompt 8：总控 Prompt

```text
请把当前仓库的 GR-RHS 真实数据验证路线，按“先主线可跑、再补强说服力”的原则完整推进。优先路线是：

1. GSE40279 human methylation age
2. GSE80672 mouse methylation age
3. TCGA 作为第二阶段扩展，不要一开始就强行接入

请按以下顺序执行：
1. 检查并利用现有 real_data_experiment 骨架和 data/real runner_ready 约定
2. 接入 GSE40279，先做 runner_ready_smoke
3. 注册 dataset spec，并完成 load_prepared_real_dataset + prepare_split 验证
4. 检查 real_data_experiment 是否缺 runner/cli，缺则补齐最小闭环
5. 跑 GSE40279 smoke 实验
6. 构造 GSE40279 main 版本
7. 如进展顺利，再接入 GSE80672 smoke
8. 每完成一个阶段，都给出当前可交付物、阻塞点和下一步建议

硬性约束：
1. 必须保持当前 loader 的 single-group-id 设计
2. 第一阶段只做 gaussian 任务
3. 过滤必须是无监督的，不能用 y 做 supervised screening
4. 必须优先复用仓库已有目录结构、配置模式和输出风格
5. 不要回退或覆盖用户现有未提交改动
6. 遇到下载、注释或运行阻塞时，要给出精确文件名、路径和错误，而不是泛泛描述

请直接在仓库里实施，不只写方案。
```
