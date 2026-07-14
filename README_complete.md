# MF-MLe-MUAV-UWR-POMDP 四算法实验包

本项目面向多无人机（MUAV）对动态风险点的路径规划与巡检，在统一的 POMDP/信念状态框架下实现四种求解方法，并提供统一参数、统一评价指标和统一结果导出。主入口是 `run_all.py`。

## 1. 四种算法

| 编号 | 目录 | 方法 | 主要用途 |
|---|---|---|---|
| 01 | `algorithm_01_rolling_route_optimization` | Mean-Field + MLE + Rolling Matching | 可解释的滚动优化基线 |
| 02 | `algorithm_02_deep_qmix_gru_path_planning` | Deep QMIX-GRU | 深度多智能体强化学习基线 |
| 03 | `algorithm_03_baam_qmix_ruma_torch` | BAAM-QMIX-RUMA（PyTorch） | 带信念、动作掩码和匹配约束的深度方法 |
| 04 | `algorithm_04_new_solver_for_comparison` | P-PBVI-RUMA | 基于逐点 PBVI 索引和滚动匹配的规划方法 |

统一实验默认使用 20 个风险点、3 架无人机、16 个时间步和随机种子 2026–2030。每架无人机每步最多访问 1 个风险点，同一风险点同一步最多由 1 架无人机访问。

## 2. 项目结构

```text
POMDP_源代码与README/
├── README.md                              # 本说明
├── run_all.py                             # 四算法统一入口
├── experiment_result_index.py             # 结果文件、参数、路径索引整理
├── 00_四算法统一运行与对比.md                # 补充运行说明
├── algorithm_01_rolling_route_optimization/
│   ├── README.md                          # 算法01详细说明
│   ├── mf_mle_muav_uwr_pomdp.py           # 算法01核心模型和仿真
│   ├── run_rolling_route_exports.py        # 算法01统一实验与结果导出入口
│   └── test_mf_mle_muav_uwr_pomdp.py       # 算法01测试
├── algorithm_02_deep_qmix_gru_path_planning/
│   └── deep_qmix_gru_path_planning.py      # 环境、GRU Agent、QMIX Mixer、训练和导出
├── algorithm_03_baam_qmix_ruma_torch/
│   ├── README.md                          # 算法03使用说明
│   ├── requirements_torch.txt              # PyTorch 算法依赖
│   ├── baam_qmix_ruma_torch.py             # BAAM-QMIX-RUMA主体实现
│   ├── run_baam_qmix_ruma_experiment.py    # 算法03标准实验入口（run_all使用）
│   ├── run_baam_qmix_ruma_paper_experiment.py # 真实数据/论文级实验入口
│   ├── step_01_config_and_protocol.py       # 配置与协议
│   ├── step_02_belief_reward.py             # 信念更新与奖励
│   ├── step_03_agent_gru.py                 # 单智能体 GRU 网络
│   ├── step_04_qmix_mixer.py                # QMIX Mixer
│   ├── step_05_action_mask_and_matching.py  # 动作掩码和冲突匹配
│   ├── step_06_trainer.py                   # 经验回放与训练器
│   ├── step_07_online_execution.py          # 在线决策
│   ├── step_08_run_template.py              # 最小运行示例
│   ├── step_10_td_target_and_loss.py        # TD 目标与损失
│   ├── step_11_training_flow.py             # 训练流程封装
│   ├── step_12_online_execution_flow.py     # 在线执行流程封装
│   └── test_*.py                            # 算法03测试
├── algorithm_04_new_solver_for_comparison/
│   ├── README.md                            # 算法04说明
│   ├── shared_comparison_settings.json      # 公平比较参数锁
│   ├── run_algorithm_04_experiment.py       # 算法04入口
│   ├── step_01_config_and_parameters.py     # 参数和数据结构
│   ├── step_02_belief_update.py             # 预测、软观测和硬观测更新
│   ├── step_03_pointwise_pbvi_index.py      # 逐点 PBVI 索引
│   ├── step_04_rolling_matching.py          # 约束滚动匹配
│   ├── step_05_simulation_environment.py    # 仿真环境和奖励
│   ├── step_06_run_p_pbvi_ruma_experiment.py # 多种子实验和结果导出
│   ├── step_07_region_threshold_filter.py   # 区域阈值候选过滤
│   ├── step_08_mle_bs_ruma_simplified.py    # 简化 MLE-BS-RUMA 评分
│   └── test_algorithm_04.py                 # 算法04测试
└── results/                                 # 运行后生成/更新的实验结果
```

`__pycache__` 和 `.pyc` 是 Python 缓存，不属于源代码，分享项目时可以忽略。

## 3. 环境安装

建议使用 Python 3.12。在 PowerShell 中执行：

```powershell
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas torch openpyxl
```

也可使用算法03的依赖文件：

```powershell
python -m pip install -r ".\algorithm_03_baam_qmix_ruma_torch\requirements_torch.txt"
python -m pip install openpyxl
```

## 4. 一键运行全部四个算法

```powershell
cd "D:\programming project\POMDP_源代码与README"
python run_all.py
```

`run_all.py` 会完成以下工作：

1. 创建统一实验参数和随机种子；
2. 调用算法01的 `run_rolling_route_algorithm`；
3. 调用算法02的 `run_deep_qmix_multi_seed`；
4. 调用算法03的 `run_baam_qmix_ruma_experiment`；
5. 调用算法04的 `run_algorithm_04_experiment`；
6. 汇总 Recall、Precision、F1、Reward 等指标；
7. 调用 `experiment_result_index.py` 整理文件清单、参数、风险点和无人机路径。

终端出现以下文字表示四算法均已完成：

```text
MF-MLe-MUAV-UWR-POMDP four-solver package completed.
```

## 5. 单独运行某个算法

### 算法01

```powershell
python ".\algorithm_01_rolling_route_optimization\run_rolling_route_exports.py"
```

### 算法02

```powershell
python ".\algorithm_02_deep_qmix_gru_path_planning\deep_qmix_gru_path_planning.py"
```

### 算法03

```powershell
python ".\algorithm_03_baam_qmix_ruma_torch\run_baam_qmix_ruma_experiment.py"
```

真实数据/论文级入口：

```powershell
python ".\algorithm_03_baam_qmix_ruma_torch\run_baam_qmix_ruma_paper_experiment.py"
```

### 算法04

```powershell
python ".\algorithm_04_new_solver_for_comparison\run_algorithm_04_experiment.py"
```

## 6. 结果文件

统一运行后的总结果位于 `results`：

| 文件 | 含义 |
|---|---|
| `comparison_summary.csv` | 四算法核心指标汇总 |
| `comparison_summary.xlsx` | Excel 版指标和统一参数 |
| `combined_result.json` | 机器可读的统一结果 |
| `experiment_data_index_and_uav_paths.md` | 实验文件和无人机路径说明 |
| `experiment_files_manifest.csv` | 结果文件清单 |
| `experiment_parameter_settings.csv` | 实验参数清单 |
| `uav_final_paths_summary.csv` | 各算法最终路径摘要 |
| `uav_step_monitoring_schedule.csv` | 逐时间步巡检计划 |

四个子目录分别保存各算法的配置、路径、风险点、降雨高风险集合、种子指标和完整结果：

```text
results\algorithm_01_rolling_route_optimization
results\algorithm_02_deep_qmix_gru_path_planning
results\algorithm_03_baam_qmix_ruma_torch
results\algorithm_04_new_solver_for_comparison
```

## 7. 指标解释

- `Recall`：真实高风险点中被成功巡检的比例。
- `Precision`：被巡检点中真实高风险点的比例。
- `F1`：Recall 和 Precision 的调和平均。
- `total_reward`：整个仿真期的累计奖励。
- `mean_reward`：平均每个时间步的奖励。

比较算法时应同时查看安全/覆盖指标和奖励，不能只依据单一数值下结论。

## 8. 测试与排错

算法01测试：

```powershell
python ".\algorithm_01_rolling_route_optimization\test_mf_mle_muav_uwr_pomdp.py"
```

算法04测试：

```powershell
python ".\algorithm_04_new_solver_for_comparison\test_algorithm_04.py"
```

常见问题：

- `ModuleNotFoundError`：先安装“环境安装”中的依赖。
- 找不到 `shared_comparison_settings.json`：该文件必须与算法04代码放在同一目录。
- 找不到真实数据集：论文级算法03需要 `results\hpr\scenario_v0\scenario_dataset_v0.npz`。
- 不要删除 `experiment_result_index.py`，否则 `run_all.py` 无法导入结果索引函数。

## 9. 已验证状态

本代码包已实际执行 `python run_all.py`，四个算法均完成运行，独立结果目录以及 CSV、Excel、JSON 总汇总文件均已生成。
