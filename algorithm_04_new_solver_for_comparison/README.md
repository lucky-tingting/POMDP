# Algorithm 04: New Solver For Comparison

本目录用于放置后续新增的第 4 个求解算法代码。

为了能和现有 `algorithm_01_rolling_route_optimization`、`algorithm_02_deep_qmix_gru_path_planning`、`algorithm_03_baam_qmix_ruma_torch` 公平比较，新算法应尽量继承统一实验设置：

- `N = 20`
- `M = 3`
- `T = 16`
- `seeds = 2026;2027;2028;2029;2030`
- 每架无人机每个时间步最多访问 1 个风险点
- 同一风险点同一时间步最多被 1 架无人机访问
- 距离约束 `22.0`
- 能量代理约束 `capacity = 160.0`, `per_distance = 1.0`, `min_safe = 16.0`

建议后续入口命名：

- `run_algorithm_04_experiment.py`
- `test_algorithm_04.py`

当前已经新增上述入口和测试文件，并按 `step_01_...` 到 `step_06_...` 的操作步骤格式实现代码。

参数锁文件：

- `shared_comparison_settings.json`

后续写算法 04 时，应优先读取或逐项对照这个文件。除非 01/02/03 三个算法也全部用新参数重新运行，否则不要单独修改算法 04 的公共对比参数。

建议后续输出目录：

`../results/algorithm_04_new_solver_for_comparison`

当前导出以下文件，方便单独运行、核对文档伪代码输出，并在需要时接入 `run_all.py` 和 `results/comparison_summary.csv`：

- `algorithm_04_summary.csv`
- `algorithm_04_seed_metrics.csv`
- `algorithm_04_uav_paths.csv`
- `algorithm_04_matching_matrices.csv`
- `algorithm_04_belief_sequence.csv`
- `algorithm_04_rainfall_high_risk.csv`
- `algorithm_04_risk_points.csv`
- `algorithm_04_config.csv`
- `algorithm_04_result.json`

其中 `algorithm_04_matching_matrices.csv` 对应文档第 10 节输出的每个时间步匹配矩阵 `Y_t`，`algorithm_04_belief_sequence.csv` 对应每个风险点的信念序列 `b_t^i`。
