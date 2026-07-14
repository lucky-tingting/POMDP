# BAAM-QMIX-RUMA PyTorch 算法使用说明

本目录是 `MF_MLe_MUAV_UWR_POMDP_双算法论文级求解包` 中的第三个算法实现：`algorithm_03_baam_qmix_ruma_torch`。它实现的是 PyTorch 版 BAAM-QMIX-RUMA，用于多无人机在城市内涝风险点上的监测调度实验。

核心思想是：

- 使用动作前信念 `pre_action_belief` 表示每个风险点在当前决策前为高风险的概率。
- 每架无人机用局部 GRU Q 网络产生局部动作评分。
- 训练阶段用 QMIX 单调混合网络做集中训练。
- 执行阶段不让无人机各自独立 argmax，而是先用 action mask 排除不可行动作，再用滚动二分匹配生成可执行的无人机-风险点匹配矩阵 `Y_t`。
- 匹配矩阵保证每个时间步“一机最多一点、一点最多一机”。

## 目录结构

```text
algorithm_03_baam_qmix_ruma_torch/
├── README.md                                      # 本说明文件
├── README_PyTorch版BAAM_QMIX_RUMA.md              # 原有简版说明
├── requirements_torch.txt                         # 基础 Python 依赖
├── baam_qmix_ruma_torch.py                        # 对外兼容入口，统一导出各 step 模块
├── run_baam_qmix_ruma_experiment.py               # 可直接运行的完整 benchmark 实验入口
├── step_01_config_and_protocol.py                 # 配置、训练状态、局部观测、环境协议
├── step_02_belief_reward.py                       # 信念更新、信息增益、奖励函数
├── step_03_agent_gru.py                           # PyTorch 局部 GRU Q 网络
├── step_04_qmix_mixer.py                          # QMIX 单调混合网络
├── step_05_action_mask_and_matching.py            # 候选集合、动作掩码、滚动匹配
├── step_06_trainer.py                             # 训练器、target network、保存和加载
├── step_07_online_execution.py                    # 单步在线执行和 episode 执行
├── step_08_run_template.py                        # 构造 trainer 的最小模板入口
├── step_10_td_target_and_loss.py                  # TD target、MSE loss、序列 replay buffer
├── step_11_training_flow.py                       # episode 采集、epsilon 探索、训练循环
├── step_12_online_execution_flow.py               # 在线执行报告：Y_t、轨迹、信念序列
├── test_*.py                                      # 单元测试和一致性测试
├── 09_代码与文档一致性审查.md                     # 代码/文档一致性审查
├── 10_训练阶段TD目标与损失函数实现说明.md          # TD 目标与损失函数说明
└── 11_BAAM_QMIX_RUMA算法设计_全局局部符号完善版.md # 论文级算法设计说明
```

`__pycache__/` 是 Python 自动生成的缓存目录，不需要手动修改。

## 每个代码文件的作用

### `run_baam_qmix_ruma_experiment.py`

这是最重要的运行入口。直接运行它会：

1. 构造一个内部 benchmark 环境 `BAAMBenchmarkEnv`。
2. 使用默认实验参数训练和评估 5 个随机种子。
3. 统计 recall、precision、F1、reward、监测点数量等指标。
4. 导出 CSV、JSON 和 Excel 结果。

默认参数在 `BAAMBenchmarkExperimentConfig` 里：

- `n_points=20`
- `n_uavs=3`
- `horizon=16`
- `seeds=(2026, 2027, 2028, 2029, 2030)`
- `train_episodes=24`
- `top_k_candidates=8`
- `max_flight_distance=22.0`
- `energy_capacity=160.0`
- `energy_per_distance=1.0`
- `min_safe_energy=16.0`

该文件还包含结果导出函数：

- `write_csv(...)`
- `write_excel(...)`
- `run_baam_qmix_ruma_experiment(...)`

如果只是想跑出结果，优先运行这个文件。

### `baam_qmix_ruma_torch.py`

这是一个兼容型总入口。它本身不写新的算法逻辑，而是从各个 `step_*.py` 文件导入类和函数，并通过 `__all__` 统一暴露。

适合在其他代码中这样使用：

```python
from baam_qmix_ruma_torch import BAAMQMixRUMAConfig, BAAMQMIXRUMATrainerTorch
```

直接运行它时，会调用 `step_08_run_template.py` 的 `main()`，只构造 trainer 并打印提示，不会启动完整训练实验。

### `step_01_config_and_protocol.py`

定义算法的基础数据结构：

- `BAAMQMixRUMAConfig`：PyTorch 算法配置，包括风险点数量、无人机数量、网络维度、折扣因子、学习率、奖励权重、传感器灵敏度/特异度等。
- `GlobalTrainingState`：集中训练使用的全局状态，包括真实风险状态、动作前信念、无人机位置、无人机能量和外生状态。
- `UAVLocalObservation`：单架无人机的局部观测结构。
- `BAAMEnvProtocol`：环境必须实现的接口协议，包括 `reset()`、`local_observations()`、`global_state()`、`action_masks()`、`step()` 等。

如果你要接入真实环境或新仿真环境，需要让环境对象满足 `BAAMEnvProtocol`。当前 benchmark 的 `global_state()` 已使用完整训练状态向量 `s_tr=(x,b_bar,q,E,eta)` 的固定维度表示，其中 `eta` 用逐点降雨表示；`local_observations()` 使用“无人机本地状态 + Top-K 候选点槽位”的固定长度向量，候选点槽位包含点编号、位置、动作前信念、重要性、距离和可行 mask。

### `step_02_belief_reward.py`

实现信念更新和奖励计算：

- `predict_belief_np(...)`：根据高风险转移概率预测下一阶段信念。
- `ml_soft_update_np(...)`：用机器学习软观测更新信念。
- `hard_observation_update_np(...)`：用 UAV 实地硬观测更新信念。
- `expected_information_gain_torch(...)`：计算监测某风险点带来的期望信息增益。
- `reward_from_assignment_torch(...)`：根据匹配矩阵计算即时奖励。

奖励由覆盖收益、漏检惩罚、误报惩罚、飞行成本和信息增益组成。

### `step_03_agent_gru.py`

实现局部 Q 网络 `BeliefAwareAgentGRU`。结构是：

```text
local_obs -> Linear/ReLU/LayerNorm -> GRUCell -> Q head -> n_actions
```

其中 `n_actions = n_points + 1`，动作 0 是虚拟动作，表示当前无人机不访问真实风险点；动作 1 到 `n_points` 对应各风险点。

### `step_04_qmix_mixer.py`

实现 QMIX 单调混合网络 `MonotonicQMIXMixerTorch`。

输入：

- 每架无人机已选动作的局部 Q 值 `agent_qs`
- 集中训练阶段的全局状态 `global_state`

输出：

- 全局联合动作价值 `Q_tot`

代码中使用 `torch.abs(...)` 保证 hypernetwork 生成的权重非负，从而维持 QMIX 的单调性约束。

### `step_05_action_mask_and_matching.py`

实现动作候选、可行性约束和滚动匹配：

- `CandidateSetBuilder`：根据信念、重要性、熵和可选监测指标构造候选风险点集合 `C_t`。
- `ActionMaskBuilder`：根据距离、能量、剩余时间、安全矩阵和候选集合生成 action mask。当前 benchmark 的安全矩阵包含返航电量安全约束。
- `RollingMatcher`：把各 UAV 对各风险点的 Q 分数送入二分匹配，生成最终 `assignment_matrix`。
- `assignment_to_action_indices(...)`：把匹配矩阵转换成每架无人机的动作编号。

这里使用 `scipy.optimize.linear_sum_assignment` 做匈牙利匹配，避免多架无人机选择同一个风险点。

### `step_06_trainer.py`

实现训练器 `BAAMQMIXRUMATrainerTorch`。

主要功能：

- 初始化 online agent、target agent、online mixer、target mixer。
- `act(...)`：在线执行阶段，根据局部观测和 action mask 输出匹配动作。
- `train_step(...)`：执行一次 QMIX TD 更新。
- `update_targets()`：同步 target network。
- `save(...)` / `load(...)`：保存和恢复模型 checkpoint。

训练时当前 Q 使用 online 网络，TD target 使用 target 网络和匹配后的下一步动作。

### `step_07_online_execution.py`

封装在线执行：

- `execute_one_decision_step(...)`：执行一个决策步，调用 trainer 产生匹配矩阵，然后调用环境 `step(...)`。
- `online_execution_episode(...)`：运行一个完整在线 episode，返回每步轨迹记录。

### `step_08_run_template.py`

最小运行模板。直接运行：

```powershell
python .\step_08_run_template.py
```

它只会构造默认 `BAAMQMixRUMAConfig` 和 `BAAMQMIXRUMATrainerTorch`，并提示你需要连接环境后才能训练。这个文件适合用作二次开发模板，不是完整实验入口。

### `step_10_td_target_and_loss.py`

实现 TD target 和 loss：

- `matched_target_action_indices(...)`：对 target 网络输出的下一步 Q 值做匹配，而不是每架无人机独立 argmax。
- `gather_agent_qs(...)`：提取已选动作对应的局部 Q。
- `compute_matched_qmix_td_target(...)`：计算 TD target。
- `qmix_td_loss(...)`：MSE 损失。
- `EpisodeSequenceReplayBuffer`：按 episode 存储轨迹，并采样固定长度连续片段。

TD target 形式是：

```text
y_t = r_t + gamma * (1 - done) * Q_tot^-(s_{t+1}, matched_actions_{t+1})
```

### `step_11_training_flow.py`

实现完整训练流程外壳：

- `BAAMTransitionRecord`：保存一个 transition 的所有训练字段。
- `random_feasible_matching(...)`：epsilon 探索时随机选择可行动作，同时保持唯一匹配约束。
- `BAAMQMixRUMATrainingLoop.collect_episode(...)`：采集一个 episode。
- `BAAMQMixRUMATrainingLoop.sample_transition_batch(...)`：从 replay buffer 采样保留时间轴的训练 batch，形状为 `[batch, sequence, ...]`。
- `BAAMQMixRUMATrainingLoop.fit_env(...)`：按 episode 训练环境。

`run_baam_qmix_ruma_experiment.py` 中的 benchmark 实验会调用这里的训练循环。

### `step_12_online_execution_flow.py`

生成在线执行报告：

- `assignment_matrices`：每个时间步的匹配矩阵 `Y_t`
- `trajectories`：每架 UAV 的动作轨迹
- `belief_sequence`：每个时间步后的信念序列
- `trace`：包含每步 reward、info、路径等完整信息

实验 runner 会使用它做训练后的评估。

## 文档和测试文件说明

- `09_代码与文档一致性审查.md`：说明当前代码与算法文档一致的地方，以及仍需说明的限制。
- `10_训练阶段TD目标与损失函数实现说明.md`：解释为什么 TD target 需要经过匹配，而不能独立 argmax。
- `11_BAAM_QMIX_RUMA算法设计_全局局部符号完善版.md`：更完整的论文级算法设计文档，包括符号、信念更新、匹配动作、奖励、训练和执行伪代码。
- `test_baam_qmix_ruma_torch_static.py`：检查源码可编译、关键组件存在、README/审查文档包含必要说明。
- `test_step_10_td_target_and_loss.py`：检查 TD target 匹配约束、MSE loss 和序列 replay buffer。
- `test_full_algorithm_alignment.py`：检查候选集合、随机匹配、训练循环和在线执行报告。
- `test_algorithm_03_experiment_runner.py`：检查默认参数和结果导出能力。

## 安装依赖

建议进入本目录后创建虚拟环境再安装依赖：

```powershell
cd "C:\Users\tingting\Documents\Codex\2026-07-12\https-swj-beijing-gov-cn-swdt\outputs\MF_MLe_MUAV_UWR_POMDP_双算法论文级求解包\algorithm_03_baam_qmix_ruma_torch"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements_torch.txt openpyxl
```

说明：

- `requirements_torch.txt` 当前包含 `torch`、`numpy`、`scipy`、`pandas`。
- 完整实验导出 Excel 时还需要 `openpyxl`，所以安装命令里额外加入了 `openpyxl`。
- 当前已验证环境为 Python 3.12.7、PyTorch 2.13.0 CPU 版、CUDA 不可用。

## 运行代码

### 1. 运行完整 benchmark 实验

在本目录执行：

```powershell
python .\run_baam_qmix_ruma_experiment.py
```

或者使用绝对路径：

```powershell
python "C:\Users\tingting\Documents\Codex\2026-07-12\https-swj-beijing-gov-cn-swdt\outputs\MF_MLe_MUAV_UWR_POMDP_双算法论文级求解包\algorithm_03_baam_qmix_ruma_torch\run_baam_qmix_ruma_experiment.py"
```

正常结束时终端会输出类似：

```text
BAAM-QMIX-RUMA PyTorch experiment completed.
runs=5, N=20, M=3, T=16, recall=..., precision=..., f1=..., reward=...
Results saved to: ...
```

### 2. 运行最小 trainer 构造模板

```powershell
python .\step_08_run_template.py
```

这个命令只验证 trainer 能被构造，不会输出实验结果。

### 3. 从 Python 代码中调用实验

可以在同目录下新建脚本或在 Python 交互环境中调用：

```python
from run_baam_qmix_ruma_experiment import (
    BAAMBenchmarkExperimentConfig,
    run_baam_qmix_ruma_experiment,
)

cfg = BAAMBenchmarkExperimentConfig(
    n_points=8,
    n_uavs=2,
    horizon=3,
    seeds=(31, 32),
    train_episodes=2,
    hidden_dim=16,
    mixer_hidden_dim=8,
)

result = run_baam_qmix_ruma_experiment(cfg, output_dir="quick_results")
print(result["summary_metrics"])
```

这个小配置适合快速检查流程；正式结果建议使用默认配置。

## 结果输出位置

默认输出目录不是当前算法目录，而是在算法包上一级的 `results` 目录：

```text
C:\Users\tingting\Documents\Codex\2026-07-12\https-swj-beijing-gov-cn-swdt\outputs\MF_MLe_MUAV_UWR_POMDP_双算法论文级求解包\results\algorithm_03_baam_qmix_ruma_torch
```

完整实验会生成：

| 文件 | 含义 |
|---|---|
| `baam_qmix_ruma_summary.csv` | 多 seed 平均指标汇总，适合直接放入算法对比总表 |
| `baam_qmix_ruma_seed_metrics.csv` | 每个 seed 的 TP、FP、FN、recall、precision、F1、reward |
| `baam_qmix_ruma_uav_paths.csv` | 每个时间步、每架 UAV 的动作、起终点、距离、剩余能量 |
| `baam_qmix_ruma_rainfall_high_risk.csv` | 每个时间步的降雨、高风险集合、被选点、TP/FP/FN |
| `baam_qmix_ruma_risk_points.csv` | 每个风险点的位置、易涝性、排水能力、重要性、初末状态和信念 |
| `baam_qmix_ruma_config.csv` | 本次实验参数 |
| `baam_qmix_ruma_result.json` | 汇总指标、seed 指标和输出文件索引 |
| `baam_qmix_ruma_results.xlsx` | 把上述表格集中到一个 Excel 文件里 |

当前目录中已有一次默认实验结果。已读到的汇总指标为：

```text
runs=5
seed_list=2026;2027;2028;2029;2030
n_points=20
n_uavs=3
horizon=16
recall=0.243770
precision=0.767670
f1=0.369737
total_reward=103.979875
mean_reward=6.498742
total_monitored=44.2
```

## 如何查看结果

最直接的查看顺序：

1. 先打开 `baam_qmix_ruma_summary.csv` 或 Excel 中的 `summary` sheet，看总体效果。
2. 再看 `baam_qmix_ruma_seed_metrics.csv`，判断不同 seed 的波动。
3. 想看无人机怎么飞，查看 `baam_qmix_ruma_uav_paths.csv`。
4. 想看每个时间步选中了哪些风险点，查看 `baam_qmix_ruma_rainfall_high_risk.csv`。
5. 想看风险点属性和最终信念，查看 `baam_qmix_ruma_risk_points.csv`。

## 验证代码

在本目录运行：

```powershell
python -m unittest test_baam_qmix_ruma_torch_static.py test_step_10_td_target_and_loss.py test_full_algorithm_alignment.py test_algorithm_03_experiment_runner.py
```

测试重点：

- 所有 step 文件能否编译。
- PyTorch GRU、QMIX mixer、trainer、rolling matcher 是否存在。
- TD target 是否使用匹配后的下一步动作，而不是独立 argmax。
- 匹配矩阵是否满足一机一点、一点一机。
- 实验 runner 是否能导出 CSV、JSON 和 Excel。

## 二次开发：接入自己的环境

如果要替换 benchmark 环境，需要写一个环境类，实现 `BAAMEnvProtocol` 中的方法：

```python
class MyEnv:
    n_uavs: int
    n_points: int

    def reset(self): ...
    def local_observations(self): ...
    def global_training_state(self): ...
    def global_state(self): ...
    def action_masks(self): ...
    def pre_action_belief(self): ...
    def point_importance(self): ...
    def uav_positions(self): ...
    def point_locations(self): ...
    def step(self, assignment_matrix): ...
```

然后按以下方式训练：

```python
from baam_qmix_ruma_torch import (
    BAAMQMixRUMAConfig,
    BAAMQMIXRUMATrainerTorch,
    BAAMQMixRUMATrainingLoop,
    online_execution_report,
)

cfg = BAAMQMixRUMAConfig(
    n_points=20,
    n_uavs=3,
    local_obs_dim=5 + 8 * 7,
    global_state_dim=20 + 20 + 3 * 2 + 3 + 20,
    hidden_dim=48,
    mixer_hidden_dim=32,
    device="cpu",
)

env = MyEnv()
trainer = BAAMQMIXRUMATrainerTorch(cfg)
loop = BAAMQMixRUMATrainingLoop(trainer, replay_capacity_episodes=64, sequence_length=4, seed=2026)
history = loop.fit_env(env, episodes=24, horizon=16)
report = online_execution_report(env, trainer, horizon=16)
```

关键要求：

- `local_observations()` 返回形状 `[n_uavs, local_obs_dim]`。
- `global_state()` 返回形状 `[global_state_dim]`。
- `action_masks()` 返回形状 `[n_uavs, n_points + 1]`，第 0 列是虚拟动作。
- `step(assignment_matrix)` 接收形状 `[n_uavs, n_points]` 的 0/1 匹配矩阵。
- `assignment_matrix` 每行最多一个 1，每列最多一个 1。

## 注意事项和限制

1. 当前 benchmark 风险点、降雨、真实高风险状态都是模拟生成，不是真实北京市全量数据。
2. 当前能耗模型是距离线性代理：`energy_cost = energy_per_distance * distance`。
3. 当前 action mask 包含距离、能量、时间、候选集合和返航电量安全约束，但避障、防撞和空域规则没有完整实现。
4. 当前默认 PyTorch 环境是 CPU 版；如果要用 GPU，需要安装匹配 CUDA 的 PyTorch，并确认 `torch.cuda.is_available()` 为 `True`。
5. 如果要做三算法完全公平比较，需要把三算法的 rainfall、高风险状态转移、固定风险点参数抽成共享 scenario 文件，再让每个算法读取同一份 scenario。
