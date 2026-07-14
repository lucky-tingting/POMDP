# Algorithm 01：Rolling Route Optimization

本目录实现四算法实验包中的算法01：基于 Mean-Field 风险传播、ML 软观测、无人机硬观测和滚动二分匹配的多无人机风险点巡检算法。它是一个可解释、无需神经网络训练的规划基线。

## 1. 算法流程

每个时间步执行：

1. 根据上一时刻信念和邻域平均风险进行状态预测；
2. 使用机器学习风险等级进行软观测更新；
3. 根据风险信念、点重要性、漏检惩罚和信息增益计算边际价值；
4. 在距离、能量和一对一访问约束下求解无人机—风险点匹配；
5. 模拟无人机硬观测并进行贝叶斯更新；
6. 计算 TP、FP、FN、TN、Recall、Precision、F1 和奖励；
7. 更新无人机位置、剩余能量和下一时刻真实风险状态。

实现中同时提供三种策略用于比较：

- `rolling_matching`：本文滚动匹配策略；
- `greedy_belief`：按高信念贪心选点；
- `random`：随机可行策略。

## 2. 文件结构

```text
algorithm_01_rolling_route_optimization/
├── README.md
├── mf_mle_muav_uwr_pomdp.py
├── run_rolling_route_exports.py
└── test_mf_mle_muav_uwr_pomdp.py
```

### `mf_mle_muav_uwr_pomdp.py`

核心模型与仿真文件：

- `ModelConfig`：风险点数、无人机数、时域、传感器、距离、能量和奖励参数；
- `RiskPoint`：风险点位置、重要性、邻域和动态风险参数；
- `transition_probabilities` / `predict_beliefs`：Mean-Field 状态转移与信念预测；
- `ml_update` / `belief_update`：ML 软观测和 UAV 硬观测贝叶斯更新；
- `point_marginal_score`：覆盖、漏检、信息增益等价值评分；
- `RollingMatchingPolicy`：滚动匹配策略；
- `GreedyBeliefPolicy`：贪心信念基线；
- `RandomPolicy`：随机基线；
- `solve_assignment`：在可达性、能量和唯一匹配约束下分配任务；
- `reward_value`：计算每步奖励；
- `run_simulation`：运行单策略、单随机种子仿真；
- `run_experiment`：运行三策略、多随机种子对比。

### `run_rolling_route_exports.py`

四算法统一比较使用的算法01入口：

- `run_rolling_route_algorithm`：运行 rolling/greedy/random 多种子实验；
- 生成风险点表、无人机路径、降雨高风险记录、指标汇总和完整 JSON；
- `run_all.py` 会从此文件导入 `run_rolling_route_algorithm`。

因此该文件必须保留。

### `test_mf_mle_muav_uwr_pomdp.py`

算法01单元测试，检查信念更新、距离/能量约束、匹配唯一性、奖励和结果结构。它不参与正式实验，但建议保留以便修改代码后回归验证。

## 3. 独立运行

从项目根目录执行：

```powershell
cd "D:\programming project\POMDP_源代码与README"
python ".\algorithm_01_rolling_route_optimization\run_rolling_route_exports.py"
```

默认配置为：

```text
风险点数 N = 20
无人机数 M = 3
时间步 T = 16
随机种子 = 2026, 2027, 2028, 2029, 2030
策略 = rolling_matching, greedy_belief, random
```

也可以直接运行核心模型自带的演示实验：

```powershell
python ".\algorithm_01_rolling_route_optimization\mf_mle_muav_uwr_pomdp.py"
```

注意：四算法公平比较应使用 `run_all.py` 或 `run_rolling_route_exports.py`，不要用核心文件中的演示参数代替统一参数。

## 4. 在 Python 中调用

```python
from pathlib import Path
import sys

alg_dir = Path("algorithm_01_rolling_route_optimization").resolve()
sys.path.insert(0, str(alg_dir))

from mf_mle_muav_uwr_pomdp import ModelConfig
from run_rolling_route_exports import run_rolling_route_algorithm

cfg = ModelConfig(n_points=20, n_uavs=3, horizon=16, seed=2026)
result = run_rolling_route_algorithm(
    output_dir="results/algorithm_01_rolling_route_optimization",
    cfg=cfg,
    seeds=[2026, 2027, 2028, 2029, 2030],
)

print(result["summary"])
```

## 5. 输出文件

| 文件 | 内容 |
|---|---|
| `rolling_modeling_summary.csv` | 三种策略的多种子平均指标和标准差 |
| `rolling_modeling_risk_points.csv` | 风险点位置、重要性等固定属性 |
| `rolling_modeling_uav_paths.csv` | 每步每架无人机的位置、访问点和能量 |
| `rolling_modeling_rainfall_high_risk.csv` | 每步降雨、高风险集合、选点和 TP/FP/FN |
| `rolling_modeling_experiment_report.md` | 可阅读的实验报告 |
| `rolling_modeling_result.json` | rolling_matching 代表种子的完整轨迹 |

统一运行时，这些文件写入：

```text
results\algorithm_01_rolling_route_optimization
```

## 6. 运行测试

```powershell
python ".\algorithm_01_rolling_route_optimization\test_mf_mle_muav_uwr_pomdp.py"
```

看到 `OK` 表示测试通过。

## 7. 与四算法统一入口的关系

```text
run_all.py
  └── run_rolling_route_exports.py
        └── mf_mle_muav_uwr_pomdp.py
```

- `run_all.py` 提供公共比较参数和输出目录；
- `run_rolling_route_exports.py` 负责算法01多种子执行和标准结果导出；
- `mf_mle_muav_uwr_pomdp.py` 提供算法模型、策略和仿真环境。

三者共同构成算法01在四算法实验中的完整运行链路。
