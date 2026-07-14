from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from step_01_config_and_parameters import PPBVIRUMAConfig, settings_rows
from step_05_simulation_environment import run_ppbvi_ruma_episode


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "results" / "algorithm_04_new_solver_for_comparison"


def write_csv(path: str | Path, rows: Sequence[Dict]) -> None:
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def summary_metrics(seed_metrics: Sequence[Dict], cfg: PPBVIRUMAConfig) -> Dict[str, object]:
    row: Dict[str, object] = {
        "algorithm": "p_pbvi_ruma",
        "solver_type": "pointwise_pbvi_index_rolling_multi_uav_matching",
        "comparison_role": "algorithm_04_pointwise_pbvi_matching",
        "runs": len(seed_metrics),
        "seed_list": cfg.seed_list,
        "n_points": cfg.n_points,
        "n_uavs": cfg.n_uavs,
        "horizon": cfg.horizon,
        "max_points_per_uav_per_step": cfg.max_points_per_uav_per_step,
        "action_definition": "one risk point per UAV per step; trajectory is accumulated across time steps",
        "shared_distance_limit": cfg.max_route_distance,
        "shared_energy_capacity": cfg.energy_capacity,
        "shared_energy_per_distance": cfg.energy_per_distance,
        "shared_min_safe_energy": cfg.min_safe_energy,
    }
    for key in ["recall", "precision", "f1", "total_reward", "mean_reward", "total_monitored"]:
        mean, std = mean_std(float(metric[key]) for metric in seed_metrics)
        row[key] = round(mean, 6)
        row[f"{key}_std"] = round(std, 6)
    row["fairness_note"] = "Uses the shared algorithm 01/02/03 comparison settings and one-point matching constraints."
    row["path_output"] = "results/algorithm_04_new_solver_for_comparison/algorithm_04_uav_paths.csv"
    row["matching_output"] = "results/algorithm_04_new_solver_for_comparison/algorithm_04_matching_matrices.csv"
    row["belief_output"] = "results/algorithm_04_new_solver_for_comparison/algorithm_04_belief_sequence.csv"
    row["rainfall_output"] = "results/algorithm_04_new_solver_for_comparison/algorithm_04_rainfall_high_risk.csv"
    row["risk_point_output"] = "results/algorithm_04_new_solver_for_comparison/algorithm_04_risk_points.csv"
    return row


def matching_matrix_rows(seed_results: Sequence[Dict], cfg: PPBVIRUMAConfig) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for result in seed_results:
        seed = result["seed"]
        for step in result["history"]:
            assignment = list(step["assignment"])
            matrix = [[1 if point_idx == i else 0 for i in range(cfg.n_points)] for point_idx in assignment]
            matrix_text = json.dumps(matrix, ensure_ascii=False)
            assignment_text = json.dumps(assignment, ensure_ascii=False)
            for uav, point_idx in enumerate(assignment):
                rows.append(
                    {
                        "seed": seed,
                        "time": step["time"],
                        "uav": uav,
                        "assigned_point_idx": "" if point_idx is None else point_idx,
                        "virtual_idle": int(point_idx is None),
                        "assignment_by_uav": assignment_text,
                        "matching_matrix": matrix_text,
                        "reward_t": round(float(step["reward"]), 6),
                    }
                )
    return rows


def belief_sequence_rows(seed_results: Sequence[Dict], cfg: PPBVIRUMAConfig) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for result in seed_results:
        seed = result["seed"]
        for step in result["history"]:
            assignment = list(step["assignment"])
            assigned_set = {point_idx for point_idx in assignment if point_idx is not None}
            assigned_uav_by_point = {point_idx: uav for uav, point_idx in enumerate(assignment) if point_idx is not None}
            for point_idx in range(cfg.n_points):
                rows.append(
                    {
                        "seed": seed,
                        "time": step["time"],
                        "point_idx": point_idx,
                        "ml_level": step["ml_levels"][point_idx],
                        "p01": round(float(step["p01"][point_idx]), 6),
                        "p10": round(float(step["p10"][point_idx]), 6),
                        "predicted_belief": round(float(step["predicted_beliefs"][point_idx]), 6),
                        "pre_action_belief": round(float(step["pre_action_beliefs"][point_idx]), 6),
                        "posterior_belief": round(float(step["posterior_beliefs"][point_idx]), 6),
                        "monitoring_index": round(float(step["monitoring_indices"][point_idx]), 6),
                        "assigned": int(point_idx in assigned_set),
                        "assigned_uav": assigned_uav_by_point.get(point_idx, ""),
                        "reward_t": round(float(step["reward"]), 6),
                    }
                )
    return rows


def run_algorithm_04_experiment(
    cfg: PPBVIRUMAConfig | None = None,
    output_dir: str | Path | None = None,
) -> Dict[str, object]:
    cfg = cfg or PPBVIRUMAConfig()
    output_path = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    seed_results = [run_ppbvi_ruma_episode(cfg, seed) for seed in cfg.seeds]
    seed_metrics = [dict(result["metrics"]) for result in seed_results]
    summary = summary_metrics(seed_metrics, cfg)
    all_paths = [row for result in seed_results for row in result["path_rows"]]
    all_matching = matching_matrix_rows(seed_results, cfg)
    all_beliefs = belief_sequence_rows(seed_results, cfg)
    all_rainfall = [row for result in seed_results for row in result["rainfall_rows"]]
    all_risk_points = [row for result in seed_results for row in result["risk_rows"]]

    write_csv(output_path / "algorithm_04_summary.csv", [summary])
    write_csv(output_path / "algorithm_04_seed_metrics.csv", seed_metrics)
    write_csv(output_path / "algorithm_04_uav_paths.csv", all_paths)
    write_csv(output_path / "algorithm_04_matching_matrices.csv", all_matching)
    write_csv(output_path / "algorithm_04_belief_sequence.csv", all_beliefs)
    write_csv(output_path / "algorithm_04_rainfall_high_risk.csv", all_rainfall)
    write_csv(output_path / "algorithm_04_risk_points.csv", all_risk_points)
    write_csv(output_path / "algorithm_04_config.csv", settings_rows(cfg))

    result = {
        "config": asdict(cfg),
        "summary_metrics": summary,
        "seed_metrics": seed_metrics,
        "output_dir": str(output_path),
        "output_files": {
            "summary": str(output_path / "algorithm_04_summary.csv"),
            "seed_metrics": str(output_path / "algorithm_04_seed_metrics.csv"),
            "uav_paths": str(output_path / "algorithm_04_uav_paths.csv"),
            "matching_matrices": str(output_path / "algorithm_04_matching_matrices.csv"),
            "belief_sequence": str(output_path / "algorithm_04_belief_sequence.csv"),
            "rainfall_high_risk": str(output_path / "algorithm_04_rainfall_high_risk.csv"),
            "risk_points": str(output_path / "algorithm_04_risk_points.csv"),
            "config": str(output_path / "algorithm_04_config.csv"),
        },
    }
    with open(output_path / "algorithm_04_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    result = run_algorithm_04_experiment()
    summary = result["summary_metrics"]
    print("P-PBVI-RUMA algorithm 04 completed.")
    print(
        f"recall={summary['recall']:.3f}, precision={summary['precision']:.3f}, "
        f"f1={summary['f1']:.3f}, reward={summary['total_reward']:.2f}"
    )
    print(f"Results saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
