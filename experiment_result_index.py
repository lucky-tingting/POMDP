from __future__ import annotations

import ast
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
REPRESENTATIVE_SEED = "2026"

ALGORITHM_SPECS = [
    {
        "algorithm": "rolling_route_optimization",
        "solver_type": "modeling_level_receding_horizon_matching",
        "path_file": "algorithm_01_rolling_route_optimization/rolling_modeling_uav_paths.csv",
        "rainfall_file": "algorithm_01_rolling_route_optimization/rolling_modeling_rainfall_high_risk.csv",
        "risk_file": "algorithm_01_rolling_route_optimization/rolling_modeling_risk_points.csv",
    },
    {
        "algorithm": "deep_qmix_gru_path_planning",
        "solver_type": "deep_marl_qmix_gru_ctde_single_point_actions",
        "path_file": "algorithm_02_deep_qmix_gru_path_planning/deep_qmix_gru_uav_paths.csv",
        "rainfall_file": "algorithm_02_deep_qmix_gru_path_planning/deep_qmix_gru_rainfall_high_risk.csv",
        "risk_file": "algorithm_02_deep_qmix_gru_path_planning/deep_qmix_gru_risk_points.csv",
    },
    {
        "algorithm": "baam_qmix_ruma_torch",
        "solver_type": "pytorch_baam_qmix_ruma_single_point_matching",
        "path_file": "algorithm_03_baam_qmix_ruma_torch/baam_qmix_ruma_uav_paths.csv",
        "rainfall_file": "algorithm_03_baam_qmix_ruma_torch/baam_qmix_ruma_rainfall_high_risk.csv",
        "risk_file": "algorithm_03_baam_qmix_ruma_torch/baam_qmix_ruma_risk_points.csv",
    },
    {
        "algorithm": "p_pbvi_ruma",
        "solver_type": "Pointwise-PBVI Index-based Rolling Multi-UAV Matching",
        "path_file": "algorithm_04_new_solver_for_comparison/algorithm_04_uav_paths.csv",
        "rainfall_file": "algorithm_04_new_solver_for_comparison/algorithm_04_rainfall_high_risk.csv",
        "risk_file": "algorithm_04_new_solver_for_comparison/algorithm_04_risk_points.csv",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_list(value: object) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def representative_rows(rows: Sequence[dict[str, str]], seed: str = REPRESENTATIVE_SEED) -> list[dict[str, str]]:
    if not rows:
        return []
    if "seed" not in rows[0]:
        return list(rows)
    selected = [row for row in rows if str(row.get("seed", "")) == str(seed)]
    return selected or list(rows)


def risk_name_map(risk_rows: Sequence[dict[str, str]]) -> dict[int, str]:
    return {as_int(row.get("idx")): row.get("name", "") for row in risk_rows}


def names_for_points(point_ids: Sequence[object], names: dict[int, str]) -> list[str]:
    return [names.get(as_int(point_id), "") for point_id in point_ids if str(point_id) != ""]


def point_ids_for_path_row(row: dict[str, str]) -> list[int]:
    if "route" in row:
        return [as_int(value) for value in parse_list(row.get("route")) if str(value) != ""]
    point_idx = row.get("point_idx", "")
    return [] if point_idx == "" else [as_int(point_idx)]


def point_names_for_path_row(row: dict[str, str], names: dict[int, str]) -> list[str]:
    if "route_names" in row:
        parsed = parse_list(row.get("route_names"))
        if parsed:
            return [str(value) for value in parsed]
    if row.get("point_name"):
        return [row["point_name"]]
    return names_for_points(point_ids_for_path_row(row), names)


def build_step_schedule(results_dir: Path = RESULTS_DIR, seed: str = REPRESENTATIVE_SEED) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in ALGORITHM_SPECS:
        risk_rows = representative_rows(read_csv(results_dir / spec["risk_file"]), seed)
        names = risk_name_map(risk_rows)
        path_rows = representative_rows(read_csv(results_dir / spec["path_file"]), seed)
        path_rows = sorted(path_rows, key=lambda row: (as_int(row.get("time")), as_int(row.get("uav"))))
        cumulative: dict[int, list[int]] = {}
        cumulative_names: dict[int, list[str]] = {}
        for row in path_rows:
            uav = as_int(row.get("uav"))
            point_ids = point_ids_for_path_row(row)
            point_names = point_names_for_path_row(row, names)
            cumulative.setdefault(uav, []).extend(point_ids)
            cumulative_names.setdefault(uav, []).extend(point_names)
            selected = [as_int(value) for value in parse_list(row.get("selected_points_t"))]
            rows.append(
                {
                    "algorithm": spec["algorithm"],
                    "seed": row.get("seed", seed),
                    "time": as_int(row.get("time")),
                    "uav": uav,
                    "monitor_point_ids_this_step": json.dumps(point_ids, ensure_ascii=False),
                    "monitor_point_names_this_step": json.dumps(point_names, ensure_ascii=False),
                    "start_x": row.get("start_x", ""),
                    "start_y": row.get("start_y", ""),
                    "end_x": row.get("end_x", ""),
                    "end_y": row.get("end_y", ""),
                    "distance_this_step": row.get("distance", ""),
                    "energy_remaining_after_step": row.get("energy_remaining", ""),
                    "reward_t": row.get("reward_t", ""),
                    "all_selected_points_this_step": json.dumps(selected, ensure_ascii=False),
                    "all_selected_point_names_this_step": json.dumps(names_for_points(selected, names), ensure_ascii=False),
                    "cumulative_path_ids_until_this_step": json.dumps(cumulative[uav], ensure_ascii=False),
                    "cumulative_path_names_until_this_step": json.dumps(cumulative_names[uav], ensure_ascii=False),
                }
            )
    return rows


def build_final_paths(schedule_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in schedule_rows:
        grouped.setdefault((str(row["algorithm"]), str(row["seed"]), int(row["uav"])), []).append(row)
    out: list[dict[str, object]] = []
    for (algorithm, seed, uav), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["time"]))
        point_ids: list[int] = []
        point_names: list[str] = []
        distances: list[float] = []
        energies: list[float] = []
        for row in rows:
            point_ids.extend(as_int(value) for value in parse_list(row["monitor_point_ids_this_step"]))
            point_names.extend(str(value) for value in parse_list(row["monitor_point_names_this_step"]))
            distances.append(as_float(row.get("distance_this_step")))
            energies.append(as_float(row.get("energy_remaining_after_step"), 0.0))
        out.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "uav": uav,
                "time_steps": len(rows),
                "visited_count": len(point_ids),
                "visited_point_ids_sequence": json.dumps(point_ids, ensure_ascii=False),
                "visited_point_names_sequence": json.dumps(point_names, ensure_ascii=False),
                "total_distance": round(sum(distances), 6),
                "start_location": f"({rows[0].get('start_x', '')},{rows[0].get('start_y', '')})" if rows else "",
                "end_location": f"({rows[-1].get('end_x', '')},{rows[-1].get('end_y', '')})" if rows else "",
                "final_energy_remaining": round(energies[-1], 6) if energies else "",
                "min_energy_remaining": round(min(energies), 6) if energies else "",
            }
        )
    return out


def build_rainfall_summary(results_dir: Path = RESULTS_DIR, seed: str = REPRESENTATIVE_SEED) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for spec in ALGORITHM_SPECS:
        risk_rows = representative_rows(read_csv(results_dir / spec["risk_file"]), seed)
        names = risk_name_map(risk_rows)
        for row in representative_rows(read_csv(results_dir / spec["rainfall_file"]), seed):
            selected = [as_int(value) for value in parse_list(row.get("selected_points"))]
            out.append(
                {
                    "algorithm": spec["algorithm"],
                    "seed": row.get("seed", seed),
                    "time": as_int(row.get("time")),
                    "mean_rainfall": row.get("mean_rainfall", ""),
                    "rainfall": row.get("rainfall", ""),
                    "high_risk_set": row.get("high_risk_set", ""),
                    "high_risk_count": row.get("high_risk_count", ""),
                    "selected_points": json.dumps(selected, ensure_ascii=False),
                    "selected_names": row.get("selected_names") or json.dumps(names_for_points(selected, names), ensure_ascii=False),
                    "tp": row.get("tp", ""),
                    "fp": row.get("fp", ""),
                    "fn": row.get("fn", ""),
                    "reward": row.get("reward", ""),
                }
            )
    return out


def build_risk_points_combined(results_dir: Path = RESULTS_DIR, seed: str = REPRESENTATIVE_SEED) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for spec in ALGORITHM_SPECS:
        for row in representative_rows(read_csv(results_dir / spec["risk_file"]), seed):
            out.append(
                {
                    "algorithm": spec["algorithm"],
                    "seed": row.get("seed", seed),
                    "idx": row.get("idx", ""),
                    "name": row.get("name", ""),
                    "x": row.get("x", ""),
                    "y": row.get("y", ""),
                    "importance": row.get("importance", ""),
                    "flood_susceptibility": row.get("flood_susceptibility", ""),
                    "drainage_capacity": row.get("drainage_capacity", ""),
                    "fixed_candidate_set": row.get("fixed_candidate_set", "True"),
                    "initial_state": row.get("initial_state", ""),
                    "initial_belief": row.get("initial_belief", ""),
                    "final_state": row.get("final_state", ""),
                    "final_belief": row.get("final_belief", ""),
                }
            )
    return out


def build_parameter_rows(results_dir: Path = RESULTS_DIR) -> list[dict[str, object]]:
    comparison = read_csv(results_dir / "comparison_summary.csv")
    out: list[dict[str, object]] = []
    for row in comparison:
        out.append(
            {
                "algorithm": row.get("algorithm", ""),
                "solver_type": row.get("solver_type", ""),
                "runs": row.get("runs", ""),
                "seed_list": row.get("seed_list", ""),
                "n_points_N": row.get("n_points", ""),
                "n_uavs_M": row.get("n_uavs", ""),
                "horizon_T": row.get("horizon", ""),
                "max_points_per_uav_per_step": row.get("max_points_per_uav_per_step", ""),
                "shared_distance_limit": row.get("shared_distance_limit", ""),
                "shared_energy_capacity": row.get("shared_energy_capacity", ""),
                "shared_energy_per_distance": row.get("shared_energy_per_distance", ""),
                "shared_min_safe_energy": row.get("shared_min_safe_energy", ""),
                "rainfall_model": "storm peak + periodic wave + point-level spatial disturbance: base=8, peak=18, wave=3, local multiplier U(0.82,1.18), noise N(0,1)",
                "risk_point_model": "fixed N grid points, spacing=5.0; susceptibility U(0.70,1.45), drainage U(0.25,0.95), importance U(0.80,1.65)",
                "belief_observation_model": "ML soft observation updates pre-action belief; UAV hard observation updates monitored points only; unmonitored points are not treated as low risk",
                "recall_mean": row.get("recall", ""),
                "recall_std": row.get("recall_std", ""),
                "precision_mean": row.get("precision", ""),
                "precision_std": row.get("precision_std", ""),
                "f1_mean": row.get("f1", ""),
                "f1_std": row.get("f1_std", ""),
                "total_reward_mean": row.get("total_reward", ""),
                "total_reward_std": row.get("total_reward_std", ""),
            }
        )
    return out


def consistency_note_rows() -> list[dict[str, str]]:
    return [
        {"item": "N fixed candidate risk points", "same_across_four_algorithms": "yes", "value_or_note": "20"},
        {"item": "M UAVs", "same_across_four_algorithms": "yes", "value_or_note": "3"},
        {"item": "T time steps", "same_across_four_algorithms": "yes", "value_or_note": "16"},
        {"item": "runs and seed list for aggregate metrics", "same_across_four_algorithms": "yes", "value_or_note": "5 runs; 2026;2027;2028;2029;2030"},
        {"item": "max points per UAV per step", "same_across_four_algorithms": "yes", "value_or_note": "1"},
        {"item": "distance limit", "same_across_four_algorithms": "yes", "value_or_note": "22.0"},
        {"item": "energy proxy", "same_across_four_algorithms": "yes", "value_or_note": "capacity=160.0; per_distance=1.0; min_safe=16.0"},
        {"item": "metric definitions", "same_across_four_algorithms": "yes", "value_or_note": "Recall, Precision, F1, total_reward"},
        {"item": "ML observation and UAV hard observation parameters", "same_across_four_algorithms": "yes", "value_or_note": "same ML matrix, sensitivity=0.88, specificity=0.86"},
        {"item": "reward weights", "same_across_four_algorithms": "yes", "value_or_note": "lambda_cover=18, lambda_miss=8, lambda_fp=3.5, lambda_cost=0.18, lambda_info=3"},
        {"item": "rainfall/risk generation mechanism", "same_across_four_algorithms": "mechanism only", "value_or_note": "same storm/grid-style mechanism; not one shared scenario file"},
        {"item": "risk point numeric values", "same_across_four_algorithms": "no", "value_or_note": "each runner currently samples or stores its own simulated point attributes"},
        {"item": "per-step rainfall/high-risk trajectory", "same_across_four_algorithms": "no", "value_or_note": "current runners do not read one common rainfall/high-risk scenario file"},
        {"item": "algorithm_03 and algorithm_04 exact history/realtime data", "same_across_four_algorithms": "no", "value_or_note": "both use the same public parameters, but their per-step rainfall, high-risk sets, beliefs and selected points are generated by separate runners"},
        {"item": "solver implementation", "same_across_four_algorithms": "no", "value_or_note": "rolling matching vs NumPy QMIX-GRU vs PyTorch BAAM-QMIX-RUMA vs P-PBVI-RUMA"},
        {"item": "training/offline solve mechanism", "same_across_four_algorithms": "no", "value_or_note": "algorithm_02/03 train value networks; algorithm_04 solves pointwise PBVI indices; algorithm_01 is direct rolling optimization"},
    ]


def describe_file(path: Path, relative: str) -> str:
    name = path.name
    descriptions = {
        "comparison_summary.csv": "Unified four-algorithm aggregate comparison with shared public settings and metrics.",
        "comparison_summary.xlsx": "Unified four-algorithm comparison workbook.",
        "experiment_parameter_settings.csv": "Experiment scale, shared parameters, model descriptions and aggregate metrics.",
        "uav_step_monitoring_schedule.csv": "Representative-seed per-step UAV monitoring schedule for all four algorithms.",
        "uav_final_paths_summary.csv": "Representative-seed final UAV path summary for all four algorithms.",
        "rainfall_high_risk_summary.csv": "Representative-seed rainfall, high-risk set, selected points and TP/FP/FN summary.",
        "risk_points_combined.csv": "Representative-seed fixed risk point table for all four algorithms.",
        "four_algorithm_consistency_notes.csv": "What is identical and what differs across the four algorithms.",
        "three_algorithm_consistency_notes.csv": "Legacy three-algorithm consistency notes kept for history; use four_algorithm_consistency_notes.csv for the current four-algorithm comparison.",
        "experiment_data_index_and_uav_paths.md": "Readable index explaining the generated result files and final UAV paths.",
        "experiment_files_manifest.csv": "Complete result file manifest with paths, row counts, fields and usage notes.",
    }
    if name in descriptions:
        return descriptions[name]
    if "algorithm_04" in relative:
        return "Detailed output from algorithm 04 P-PBVI-RUMA."
    if "algorithm_03" in relative:
        return "Detailed output from algorithm 03 BAAM-QMIX-RUMA."
    if "algorithm_02" in relative:
        return "Detailed output from algorithm 02 Deep QMIX-GRU."
    if "algorithm_01" in relative:
        return "Detailed output from algorithm 01 rolling route optimization."
    return "Supplementary result/report/intermediate output from the experiment."


def manifest_rows(results_dir: Path = RESULTS_DIR) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(p for p in results_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(ROOT).as_posix()
        file_type = path.suffix.lstrip(".").lower()
        row_count: object = ""
        columns: object = ""
        if file_type == "csv":
            data = read_csv(path)
            row_count = len(data)
            columns = ", ".join(data[0].keys()) if data else ""
        rows.append(
            {
                "file_name": path.name,
                "relative_path": rel,
                "absolute_path": str(path),
                "file_type": file_type,
                "rows_if_csv": row_count,
                "columns_if_csv": columns,
                "description": describe_file(path, rel),
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return rows


def write_markdown(
    path: Path,
    comparison: Sequence[dict[str, str]],
    params: Sequence[dict[str, object]],
    final_paths: Sequence[dict[str, object]],
    manifest: Sequence[dict[str, object]],
) -> None:
    lines = [
        "# MF-MLe-MUAV-UWR-POMDP 四算法实验数据索引",
        "",
        f"路径、降雨、高风险集合、风险点合并表均使用代表性 seed `{REPRESENTATIVE_SEED}`；多 seed 均值和标准差见 `comparison_summary.csv` 与 `comparison_summary.xlsx`。",
        "",
        "## 实验规模与共享参数",
        "",
        "| 参数 | 值 |",
        "|---|---:|",
    ]
    if params:
        first = params[0]
        lines.extend(
            [
                f"| N 固定候选风险点 | {first['n_points_N']} |",
                f"| M 无人机 | {first['n_uavs_M']} |",
                f"| T 时间步 | {first['horizon_T']} |",
                f"| runs/seeds | {first['runs']} / {first['seed_list']} |",
                f"| 每架 UAV 每步最多访问点数 | {first['max_points_per_uav_per_step']} |",
                f"| 距离约束 | {first['shared_distance_limit']} |",
                f"| 能量约束 | {first['shared_energy_capacity']} / {first['shared_energy_per_distance']} / {first['shared_min_safe_energy']} |",
            ]
        )
    lines.extend(
        [
            "",
            "降雨模型：暴雨峰型 + 周期波动 + 点级空间扰动。固定风险点模型：网格点，`grid_spacing=5.0`，每个点有易涝系数、排水能力、重要性权重。",
            "",
            "## 四算法哪些一样、哪些不一样",
            "",
            "| 项目 | 是否一致 | 说明 |",
            "|---|---|---|",
        ]
    )
    for row in consistency_note_rows():
        lines.append(f"| {row['item']} | {row['same_across_four_algorithms']} | {row['value_or_note']} |")
    lines.extend(
        [
            "",
            "结论：四个算法的公共实验参数一致；但 03 和 04 的历史数据/实时数据不是同一份逐时 scenario。它们的 rainfall、高风险集合、belief 轨迹和实时选点由各自 runner 生成。",
            "",
            "## 文件清单",
            "",
            "| 数据类型 | 文件 | 行数 |",
            "|---|---|---:|",
        ]
    )
    key_files = [
        ("算法对比汇总 CSV", "results/comparison_summary.csv"),
        ("算法对比汇总 Excel", "results/comparison_summary.xlsx"),
        ("实验参数设置", "results/experiment_parameter_settings.csv"),
        ("每步 UAV 监测调度", "results/uav_step_monitoring_schedule.csv"),
        ("最终 UAV 路径", "results/uav_final_paths_summary.csv"),
        ("降雨/高风险过程", "results/rainfall_high_risk_summary.csv"),
        ("固定风险点参数", "results/risk_points_combined.csv"),
        ("全部结果文件清单", "results/experiment_files_manifest.csv"),
    ]
    manifest_by_rel = {str(row["relative_path"]): row for row in manifest}
    for label, rel in key_files:
        row = manifest_by_rel.get(rel, {})
        lines.append(f"| {label} | `{rel}` | {row.get('rows_if_csv', '-')} |")
    lines.extend(
        [
            "",
            "## 四算法结果汇总",
            "",
            "| 算法 | runs | Recall | Precision | F1 | Total reward |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison:
        lines.append(
            f"| {row['algorithm']} | {row['runs']} | {row['recall']} +/- {row['recall_std']} | "
            f"{row['precision']} +/- {row['precision_std']} | {row['f1']} +/- {row['f1_std']} | "
            f"{row['total_reward']} +/- {row['total_reward_std']} |"
        )
    lines.extend(
        [
            "",
            f"## 每架无人机最终路径，代表 seed {REPRESENTATIVE_SEED}",
            "",
            "| 算法 | UAV | visits | total distance | min energy | point ids | point names |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in final_paths:
        lines.append(
            f"| {row['algorithm']} | {row['uav']} | {row['visited_count']} | {row['total_distance']} | "
            f"{row['min_energy_remaining']} | `{row['visited_point_ids_sequence']}` | `{row['visited_point_names_sequence']}` |"
        )
    lines.extend(
        [
            "",
            "每个点编号对应的名称、坐标、易涝系数、排水能力和重要性权重见 `risk_points_combined.csv`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_experiment_result_index(results_dir: Path = RESULTS_DIR, seed: str = REPRESENTATIVE_SEED) -> dict[str, int]:
    comparison = read_csv(results_dir / "comparison_summary.csv")
    params = build_parameter_rows(results_dir)
    schedule = build_step_schedule(results_dir, seed)
    final_paths = build_final_paths(schedule)
    rainfall = build_rainfall_summary(results_dir, seed)
    risk_points = build_risk_points_combined(results_dir, seed)
    notes = consistency_note_rows()

    write_csv(results_dir / "experiment_parameter_settings.csv", params)
    write_csv(results_dir / "uav_step_monitoring_schedule.csv", schedule)
    write_csv(results_dir / "uav_final_paths_summary.csv", final_paths)
    write_csv(results_dir / "rainfall_high_risk_summary.csv", rainfall)
    write_csv(results_dir / "risk_points_combined.csv", risk_points)
    write_csv(results_dir / "four_algorithm_consistency_notes.csv", notes)

    manifest = manifest_rows(results_dir)
    write_csv(results_dir / "experiment_files_manifest.csv", manifest)
    manifest = manifest_rows(results_dir)
    write_csv(results_dir / "experiment_files_manifest.csv", manifest)
    write_markdown(results_dir / "experiment_data_index_and_uav_paths.md", comparison, params, final_paths, manifest)

    return {
        "comparison_rows": len(comparison),
        "parameter_rows": len(params),
        "schedule_rows": len(schedule),
        "final_path_rows": len(final_paths),
        "rainfall_rows": len(rainfall),
        "risk_point_rows": len(risk_points),
        "manifest_rows": len(manifest),
    }


def main() -> None:
    counts = build_experiment_result_index()
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
