# MF-MLe-MUAV-UWR-POMDP 四算法实验包

本目录是四种无人机路径规划与风险点巡检算法的统一运行包。主入口为 `run_all.py`，运行后会依次执行四个算法，使用统一参数汇总指标，并把结果保存到 `results`。

## 四个算法

1. `algorithm_01_rolling_route_optimization`：滚动路径优化算法。
2. `algorithm_02_deep_qmix_gru_path_planning`：Deep QMIX-GRU 路径规划算法。
3. `algorithm_03_baam_qmix_ruma_torch`：BAAM-QMIX-RUMA（PyTorch）算法。
4. `algorithm_04_new_solver_for_comparison`：P-PBVI-RUMA 对比算法。

## 运行环境

建议使用 Python 3.12。在 PowerShell 中安装基础依赖：

```powershell
python -m pip install numpy scipy pandas torch openpyxl
```

第三个算法目录中的 `requirements_torch.txt` 记录了其主要依赖，也可以执行：

```powershell
python -m pip install -r ".\algorithm_03_baam_qmix_ruma_torch\requirements_torch.txt"
python -m pip install openpyxl
```

## 一键运行四个算法

先进入本目录：

```powershell
cd "D:\programming project\POMDP_源代码与README"
python run_all.py
```

`run_all.py` 会：

- 统一四个算法的实验参数与随机种子；
- 依次运行算法 01、02、03、04；
- 汇总 Recall、Precision、F1、Reward 等指标；
- 调用 `experiment_result_index.py` 整理实验文件和无人机路径；
- 将所有结果写入 `results`。

## 主要输出

运行成功后重点查看：

```text
results\comparison_summary.csv
results\comparison_summary.xlsx
results\combined_result.json
results\experiment_data_index_and_uav_paths.md
```

每个算法的独立结果分别位于：

```text
results\algorithm_01_rolling_route_optimization
results\algorithm_02_deep_qmix_gru_path_planning
results\algorithm_03_baam_qmix_ruma_torch
results\algorithm_04_new_solver_for_comparison
```

## 根目录关键文件

- `run_all.py`：四算法统一运行主函数，必须保留。
- `experiment_result_index.py`：被主函数直接调用，用于整理结果索引，必须保留。
- `README.md`：本运行说明。
- `00_四算法统一运行与对比.md`：已有的补充说明。

## 已验证状态

本复制包已实际执行 `python run_all.py` 并成功完成四个算法，四类算法结果及总汇总文件均已生成。
