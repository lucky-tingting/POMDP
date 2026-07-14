# PyTorch 版 BAAM-QMIX-RUMA

## 定位

本文件夹是 `MF-MLe-MUAV-UWR-POMDP` 的 PyTorch BAAM-QMIX-RUMA 实现。

它不同于 `algorithm_02_deep_qmix_gru_path_planning`：

- `algorithm_02` 是 NumPy 自包含 Deep QMIX-GRU 原型。
- `algorithm_03` 是 PyTorch BAAM-QMIX-RUMA 实现。

## 当前状态

当前已经具备可运行实验入口：

```powershell
python "C:\Users\tingting\Documents\Codex\2026-07-12\https-swj-beijing-gov-cn-swdt\outputs\MF_MLe_MUAV_UWR_POMDP_双算法论文级求解包\algorithm_03_baam_qmix_ruma_torch\run_baam_qmix_ruma_experiment.py"
```

也已经接入顶层三算法统一入口：

```powershell
python "C:\Users\tingting\Documents\Codex\2026-07-12\https-swj-beijing-gov-cn-swdt\outputs\MF_MLe_MUAV_UWR_POMDP_双算法论文级求解包\run_all.py"
```

算法执行逻辑是：动作前信念输入 PyTorch GRU 局部 Q 网络，QMIX 进行集中训练，执行阶段通过 action mask 和滚动匹配生成 `Y_t`。这保证一机一点、一点一机，不使用各无人机独立选择导致冲突的动作。

## 统一参数

- `N=20`
- `M=3`
- `T=16`
- `seeds=2026;2027;2028;2029;2030`
- 每架无人机每个时间步最多访问 1 个风险点
- `max_flight_distance=22.0`
- `energy_capacity=160.0`
- `energy_per_distance=1.0`
- `min_safe_energy=16.0`
- 一机一点、一点一机

## 代码结构

- `run_baam_qmix_ruma_experiment.py`：独立实验 runner 和结果导出。
- `step_01_config_and_protocol.py`：配置和环境协议。
- `step_02_belief_reward.py`：信念更新、信息增益和奖励函数。
- `step_03_agent_gru.py`：PyTorch GRU 局部 Q 网络。
- `step_04_qmix_mixer.py`：QMIX 单调 mixer。
- `step_05_action_mask_and_matching.py`：动作掩码和滚动二分匹配。
- `step_06_trainer.py`：训练器、优化器、target network。
- `step_10_td_target_and_loss.py`：TD target 和损失函数。
- `step_11_training_flow.py`：episode/序列训练流程。
- `step_12_online_execution_flow.py`：在线执行轨迹输出。

这些文件对应 Dec-POMDP 近似实现、动作前信念、虚拟动作、滚动匹配和 TD 目标训练流程。

## 输出文件

默认输出目录：

`results/algorithm_03_baam_qmix_ruma_torch`

主要输出：

- `baam_qmix_ruma_summary.csv`
- `baam_qmix_ruma_seed_metrics.csv`
- `baam_qmix_ruma_uav_paths.csv`
- `baam_qmix_ruma_rainfall_high_risk.csv`
- `baam_qmix_ruma_risk_points.csv`
- `baam_qmix_ruma_config.csv`
- `baam_qmix_ruma_result.json`
- `baam_qmix_ruma_results.xlsx`

## 当前环境

当前 PyTorch 是 CPU 版：

```text
torch = 2.13.0+cpu
cuda_available = False
```

当前环境已经安装 CPU 版 PyTorch。

因此可以说当前 PyTorch 版已经能在 CPU 上运行并导出仿真实验结果；不能说 CUDA/GPU 训练已经成功。

## 仍需说明的限制

1. 当前 benchmark 环境是 `run_baam_qmix_ruma_experiment.py` 内部实现，尚未抽成三算法共享 scenario 文件。
2. 当前风险点和降雨为模拟生成，不是真实北京市全量风险点和真实降雨数据。
3. 当前能耗是距离线性代理，不是真实电池模型。
4. 返航电量安全已经进入 benchmark action mask；避障、防撞和空域规则尚未完整实现。

## 验证

```powershell
python -m unittest test_baam_qmix_ruma_torch_static.py test_step_10_td_target_and_loss.py test_full_algorithm_alignment.py test_algorithm_03_experiment_runner.py
```

当前验证结果：

```text
Ran 15 tests
OK
```
