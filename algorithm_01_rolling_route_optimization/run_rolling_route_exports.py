from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mf_mle_muav_uwr_pomdp import ModelConfig, run_simulation


Point = Tuple[float, float]


def write_csv(path: str, rows: Sequence[Dict]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key, value in row.items():
                if isinstance(value, (list, tuple, dict)):
                    out[key] = json.dumps(value, ensure_ascii=False)
                else:
                    out[key] = value
            writer.writerow(out)


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def metric_average(policy: str, results: Sequence[Dict]) -> Dict:
    keys = ["recall", "precision", "f1", "total_reward", "mean_reward", "total_monitored"]
    row = {"algorithm": "rolling_route_optimization", "policy": policy, "runs": len(results)}
    for key in keys:
        values = [r["metrics"][key] for r in results]
        row[key] = round(float(np.mean(values)), 6)
        row[f"{key}_std"] = round(float(np.std(values, ddof=0)), 6)
    return row


def risk_point_rows(result: Dict) -> List[Dict]:
    rows = []
    for point in result["points"]:
        rows.append(
            {
                "idx": point["idx"],
                "name": point["name"],
                "x": point["x"],
                "y": point["y"],
                "importance": point["importance"],
                "flood_susceptibility": point["flood_susceptibility"],
                "drainage_capacity": point["drainage_capacity"],
                "fixed_candidate_set": True,
            }
        )
    return rows


def path_rows(result: Dict) -> List[Dict]:
    points = {p["idx"]: p for p in result["points"]}
    n_uavs = result["config"]["n_uavs"]
    positions: List[Point] = [(0.0, 0.0) for _ in range(n_uavs)]
    cumulative: List[List[int]] = [[] for _ in range(n_uavs)]
    rows: List[Dict] = []
    for step in result["history"]:
        assignment = step["assignment"]
        energy_remaining = step.get("uav_energy_remaining", [])
        for m in range(n_uavs):
            idx: Optional[int] = assignment[m] if m < len(assignment) else None
            start = positions[m]
            if idx is None:
                end = start
                route = []
                route_names = []
                segment_distance = 0.0
            else:
                point = points[idx]
                end = (float(point["x"]), float(point["y"]))
                route = [idx]
                route_names = [point["name"]]
                segment_distance = dist(start, end)
                positions[m] = end
                cumulative[m].append(idx)
            rows.append(
                {
                    "time": step["time"],
                    "uav": m,
                    "route": route,
                    "route_names": route_names,
                    "cumulative_path": list(cumulative[m]),
                    "start_x": round(start[0], 6),
                    "start_y": round(start[1], 6),
                    "end_x": round(end[0], 6),
                    "end_y": round(end[1], 6),
                    "distance": round(segment_distance, 6),
                    "energy_remaining": energy_remaining[m] if m < len(energy_remaining) else "",
                    "reward_t": step["reward"],
                    "selected_points_t": step["selected_points"],
                }
            )
    return rows


def rainfall_rows(result: Dict) -> List[Dict]:
    rows = []
    for step in result["history"]:
        rows.append(
            {
                "time": step["time"],
                "mean_rainfall": step["mean_rainfall"],
                "rainfall": step.get("rainfall", []),
                "high_risk_set": step.get("high_risk_set", []),
                "high_risk_count": step["true_high_count"],
                "selected_points": step["selected_points"],
                "tp": step["tp"],
                "fp": step["fp"],
                "fn": step["fn"],
                "reward": step["reward"],
                "mean_pre_action_belief": step["mean_pre_action_belief"],
                "mean_posterior_belief": step["mean_posterior_belief"],
            }
        )
    return rows


def write_report(path: str, cfg: ModelConfig, summary_rows: Sequence[Dict], main_result: Dict) -> None:
    best = max(summary_rows, key=lambda row: row["f1"])
    lines = [
        "# Rolling Route Optimization 实验报告",
        "",
        "本算法是 MF-MLe-MUAV-UWR-POMDP 的论文建模级滚动优化求解器。",
        "",
        "它不是深度强化学习，而是把每个时间步的动作前信念、重要性、漏检惩罚、误监测惩罚、信息价值和飞行成本合成为边权，然后求解多无人机最大权匹配。",
        "",
        "当前版本同步实现了距离-能耗代理约束：候选访问动作需要满足最大单步飞行距离限制，并且执行后的剩余能量不得低于最低安全电量。",
        "",
        "## 动作解释",
        "",
        "当前版本每个时间步每架无人机执行一个监测点；跨多个时间步累积后形成每架无人机完整路径。它适合作为可解释、稳定的论文基准算法。",
        "",
        f"固定候选风险点数 `N={cfg.n_points}`，无人机数 `M={cfg.n_uavs}`，时间步 `T={cfg.horizon}`。",
        "",
        "## 策略对比",
        "",
        "| policy | runs | recall mean | recall std | precision mean | precision std | f1 mean | f1 std | total_reward mean | total_reward std | total_monitored mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['policy']} | {row['runs']} | {row['recall']:.6f} | {row['recall_std']:.6f} | "
            f"{row['precision']:.6f} | {row['precision_std']:.6f} | {row['f1']:.6f} | {row['f1_std']:.6f} | "
            f"{row['total_reward']:.6f} | {row['total_reward_std']:.6f} | {row['total_monitored']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"F1 最好的策略是 `{best['policy']}`。",
            "",
            "## 输出文件",
            "",
            "- `rolling_modeling_uav_paths.csv`: 每个时间步每架无人机的路径记录。",
            "- `rolling_modeling_rainfall_high_risk.csv`: 每个时间步降雨、高风险集合、选点和 TP/FP/FN。",
            "- `rolling_modeling_risk_points.csv`: 固定候选风险点参数。",
            "- `rolling_modeling_summary.csv`: rolling/greedy/random 的平均指标对比。",
            "- `rolling_modeling_result.json`: rolling_matching 单次完整轨迹。",
            "",
            "## 单次 rolling_matching 指标",
            "",
            json.dumps(main_result["metrics"], ensure_ascii=False, indent=2),
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_rolling_route_algorithm(
    output_dir: Optional[str] = None,
    cfg: Optional[ModelConfig] = None,
    seeds: Optional[Sequence[int]] = None,
    policies: Optional[Sequence[str]] = None,
) -> Dict:
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    cfg = cfg or ModelConfig(n_points=20, n_uavs=3, horizon=16, seed=2026)
    policies = list(policies or ["rolling_matching", "greedy_belief", "random"])
    seeds = list(seeds or [2026, 2027, 2028, 2029, 2030])
    all_results: Dict[str, List[Dict]] = {}
    summary_rows: List[Dict] = []
    for policy in policies:
        results = [run_simulation(cfg, policy_name=policy, seed=seed) for seed in seeds]
        all_results[policy] = results
        summary_rows.append(metric_average(policy, results))

    main_result = all_results["rolling_matching"][0]
    write_csv(os.path.join(output_dir, "rolling_modeling_summary.csv"), summary_rows)
    write_csv(os.path.join(output_dir, "rolling_modeling_risk_points.csv"), risk_point_rows(main_result))
    write_csv(os.path.join(output_dir, "rolling_modeling_uav_paths.csv"), path_rows(main_result))
    write_csv(os.path.join(output_dir, "rolling_modeling_rainfall_high_risk.csv"), rainfall_rows(main_result))
    write_report(os.path.join(output_dir, "rolling_modeling_experiment_report.md"), cfg, summary_rows, main_result)
    with open(os.path.join(output_dir, "rolling_modeling_result.json"), "w", encoding="utf-8") as f:
        json.dump(main_result, f, ensure_ascii=False, indent=2)
    return {"summary": summary_rows, "main_result": main_result}


def main() -> None:
    result = run_rolling_route_algorithm()
    print("Rolling route optimization experiment completed.")
    for row in result["summary"]:
        print(
            f"{row['policy']}: recall={row['recall']:.3f}, precision={row['precision']:.3f}, "
            f"f1={row['f1']:.3f}, reward={row['total_reward']:.2f}"
        )


if __name__ == "__main__":
    main()
