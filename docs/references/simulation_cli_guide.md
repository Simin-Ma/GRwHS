# Simulation CLI Guide

这份文档只保留两个正式论文版 simulation 的运行命令。

## 1. `simulation_second`

用于主 benchmark 正式论文版实验。默认会跑完整默认 suite，并自动生成
`paper_tables`。

正式运行：

```bash
python -m simulation_second.src.run_blueprint run-benchmark
```

`repeat = 1` 测试：

```bash
python -m simulation_second.src.run_blueprint run-benchmark --repeats 1 --save-dir outputs/simulation_second/test_r1
```

默认正式输出目录：

- `outputs/simulation_second/benchmark_main`

## 2. `simulation_mechanism`

用于机制部分正式论文版实验。默认会跑完整默认 suite（`M1-M4`），并自动生成
`paper_tables` 和 `figures`。

正式运行：

```bash
python -m simulation_mechanism.src.run_mechanism run-mechanism
```

`repeat = 1` 测试：

```bash
python -m simulation_mechanism.src.run_mechanism run-mechanism --repeats 1 --save-dir outputs/simulation_mechanism/test_r1
```

默认正式输出目录：

- `outputs/simulation_mechanism/mechanism_main`

## 3. 说明

- 两个 CLI 都会强制使用 Bayesian convergence gate。
- 如果只是检查链路是否通，就先跑各自的 `repeat = 1` 测试。
