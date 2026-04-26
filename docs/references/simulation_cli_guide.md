# Simulation CLI Guide

这份说明只保留论文版 `simulation_second` 和 `simulation_mechanism` 的常用命令。

## 1. `simulation_second`

用于论文 benchmark 主实验。默认输出根目录是
`outputs/history/simulation_second/benchmark_main`，每次运行都会在这个根目录下自动创建
一个独立时间戳子目录，并维护 `latest_run.json`、`latest_run.txt`、
`session_index.jsonl`，因此不会覆盖旧结果。

正式运行：

```bash
python -m simulation_second.src.run_blueprint run-benchmark
```

`repeat = 1` 快速测试：

```bash
python -m simulation_second.src.run_blueprint run-benchmark --repeats 1
```

手动指定输出根目录：

```bash
python -m simulation_second.src.run_blueprint run-benchmark --repeats 1 --save-dir outputs/history/simulation_second/benchmark_main
```

从历史根目录重建表格：

```bash
python -m simulation_second.src.run_blueprint build-tables --results-dir outputs/history/simulation_second/benchmark_main
```

## 2. `simulation_mechanism`

用于论文 mechanism 主实验。默认输出根目录是
`outputs/history/simulation_mechanism/mechanism_main`，每次运行都会在这个根目录下自动创建
一个独立时间戳子目录，并维护 `latest_run.json`、`latest_run.txt`、
`session_index.jsonl`，因此不会覆盖旧结果。

正式运行：

```bash
python -m simulation_mechanism.src.run_mechanism run-mechanism
```

默认主 suite 只包含论文机制主线；`M4` 默认只保留 `p0=5`。
如果要把 `p0=15/30` 的 dense ablation 诊断线一起带上，请显式加：

```bash
python -m simulation_mechanism.src.run_mechanism run-mechanism --include-dense-ablation
```

`repeat = 1` 快速测试：

```bash
python -m simulation_mechanism.src.run_mechanism run-mechanism --repeats 1
```

手动指定输出根目录：

```bash
python -m simulation_mechanism.src.run_mechanism run-mechanism --repeats 1 --save-dir outputs/history/simulation_mechanism/mechanism_main
```

从历史根目录重建表格和图：

```bash
python -m simulation_mechanism.src.run_mechanism build-tables --results-dir outputs/history/simulation_mechanism/mechanism_main
python -m simulation_mechanism.src.run_mechanism build-figures --results-dir outputs/history/simulation_mechanism/mechanism_main
```

## 3. 说明

- 两个 CLI 都会强制启用 Bayesian convergence gate。
- 如果只是检查链路是否通畅，优先先跑各自的 `repeat = 1` 快速测试。
- `build-tables` / `build-figures` 现在既可以直接传某次具体 run 目录，也可以直接传历史根目录，工具会自动解析最近一次结果。
