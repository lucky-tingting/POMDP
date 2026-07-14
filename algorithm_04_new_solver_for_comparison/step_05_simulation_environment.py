from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

from step_01_config_and_parameters import PPBVIRUMAConfig, Point, RiskPoint, distance, energy_cost
from step_02_belief_update import (
    build_neighbors,
    hard_observation_update_all,
    ml_soft_update_all,
    predict_beliefs,
    transition_probabilities,
)
from step_03_pointwise_pbvi_index import PointwisePBVIIndexEstimator, expected_information_gain
from step_04_rolling_matching import solve_rolling_matching


POINT_NAMES = [
    "Yuegezhuang",
    "Liuliqiao",
    "Fengyiqiao",
    "Dahongmen",
    "Xizhimen",
    "Dongzhimen",
    "Fuxingmen",
    "Jianguomen",
    "Chaoyangmen",
    "Chongwenmen",
    "Guangqumen",
    "Guanganmen",
    "Youanmen",
    "Zuoanmen",
    "Yongdingmen",
    "Andingmen",
    "Madian",
    "Sanyuanqiao",
    "Wukesong",
    "Shuangjing",
]


def metric_dict(tp: int, fp: int, fn: int, rewards: Sequence[float], total_monitored: int) -> Dict[str, float]:
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    f1 = 2.0 * recall * precision / (recall + precision) if recall + precision > 0.0 else 0.0
    total_reward = float(sum(rewards))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6),
        "total_reward": round(total_reward, 6),
        "mean_reward": round(total_reward / len(rewards), 6) if rewards else 0.0,
        "total_monitored": total_monitored,
    }


def make_risk_points(cfg: PPBVIRUMAConfig, rng: np.random.Generator) -> list[RiskPoint]:
    side = math.ceil(math.sqrt(cfg.n_points))
    points: list[RiskPoint] = []
    for i in range(cfg.n_points):
        location = ((i % side) * cfg.grid_spacing, (i // side) * cfg.grid_spacing)
        susceptibility = float(rng.uniform(0.70, 1.45))
        drainage = float(rng.uniform(0.25, 0.95))
        importance = float(rng.uniform(0.80, 1.65))
        initial_prob = min(0.80, max(0.15, 0.25 + 0.20 * susceptibility - 0.12 * drainage))
        state = int(rng.random() < initial_prob)
        belief = float(np.clip(initial_prob + rng.normal(0.0, 0.08), 0.05, 0.95))
        name = POINT_NAMES[i] if i < len(POINT_NAMES) else f"Point_{i}"
        points.append(RiskPoint(i, name, location, susceptibility, drainage, importance, state, belief))
    return points


def rainfall_for_step(cfg: PPBVIRUMAConfig, t: int, points: Sequence[RiskPoint], rng: np.random.Generator) -> list[float]:
    phase = math.pi * t / max(1, cfg.horizon - 1)
    base = cfg.rainfall_base + cfg.rainfall_peak * math.sin(phase)
    wave = cfg.rainfall_wave * math.sin(2.0 * math.pi * (t + 1) / max(2, cfg.horizon))
    values = []
    for point in points:
        local_multiplier = rng.uniform(0.82, 1.18)
        susceptibility_boost = 1.0 + 0.10 * (point.flood_susceptibility - 1.0)
        values.append(max(0.0, base * local_multiplier * susceptibility_boost + wave + rng.normal(0.0, 1.0)))
    return values


def sample_ml_levels(
    cfg: PPBVIRUMAConfig,
    true_states: Sequence[int],
    rng: np.random.Generator,
) -> list[int]:
    levels = []
    for state in true_states:
        probs = cfg.ml_observation_matrix[int(state)]
        levels.append(int(rng.choice(np.arange(1, cfg.n_ml_levels + 1), p=np.asarray(probs) / sum(probs))))
    return levels


def sample_hard_observations(
    cfg: PPBVIRUMAConfig,
    true_states: Sequence[int],
    selected_points: Sequence[int],
    rng: np.random.Generator,
) -> dict[int, int]:
    observations: dict[int, int] = {}
    for point in selected_points:
        if true_states[point] == 1:
            observations[point] = int(rng.random() < cfg.uav_sensitivity)
        else:
            observations[point] = int(rng.random() < (1.0 - cfg.uav_specificity))
    return observations


def update_true_states(
    cfg: PPBVIRUMAConfig,
    points: Sequence[RiskPoint],
    rainfall: Sequence[float],
    previous_beliefs: Sequence[float],
    neighbors: Sequence[Sequence[int]],
    rng: np.random.Generator,
) -> tuple[list[int], list[float], list[float]]:
    new_states: list[int] = []
    p01_values: list[float] = []
    p10_values: list[float] = []
    for i, point in enumerate(points):
        neighbor_mean = float(np.mean([previous_beliefs[j] for j in neighbors[i]])) if neighbors[i] else float(np.mean(previous_beliefs))
        p01, p10 = transition_probabilities(cfg, rainfall[i], neighbor_mean, point.drainage_capacity)
        if point.true_state == 1:
            new_state = int(rng.random() >= p10)
        else:
            new_state = int(rng.random() < p01)
        point.true_state = new_state
        new_states.append(new_state)
        p01_values.append(p01)
        p10_values.append(p10)
    return new_states, p01_values, p10_values


def reward_from_assignment(
    cfg: PPBVIRUMAConfig,
    points: Sequence[RiskPoint],
    selected_points: Sequence[int],
    selected_distances: Sequence[float],
    pre_action_beliefs: Sequence[float],
) -> float:
    selected = set(selected_points)
    reward = 0.0
    for point in points:
        if point.idx in selected:
            if point.true_state == 1:
                reward += cfg.lambda_cover * point.importance
            else:
                reward -= cfg.lambda_fp
            reward += cfg.lambda_info * expected_information_gain(
                pre_action_beliefs[point.idx],
                cfg.uav_sensitivity,
                cfg.uav_specificity,
            )
        elif point.true_state == 1:
            reward -= cfg.lambda_miss * point.importance
    reward -= cfg.lambda_cost * sum(selected_distances)
    return float(reward)


def run_ppbvi_ruma_episode(cfg: PPBVIRUMAConfig, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    points = make_risk_points(cfg, rng)
    locations = [point.location for point in points]
    neighbors = build_neighbors(locations, cfg.neighbor_radius)
    uav_positions: list[Point] = [(0.0, 0.0) for _ in range(cfg.n_uavs)]
    energy_remaining = [cfg.energy_capacity for _ in range(cfg.n_uavs)]
    estimator = PointwisePBVIIndexEstimator(cfg)

    history: list[dict] = []
    path_rows: list[dict] = []
    rainfall_rows: list[dict] = []
    rewards: list[float] = []
    tp_total = fp_total = fn_total = total_monitored = 0

    for t in range(cfg.horizon):
        previous_beliefs = [point.belief for point in points]
        rainfall = rainfall_for_step(cfg, t, points, rng)
        true_states, p01_values, p10_values = update_true_states(cfg, points, rainfall, previous_beliefs, neighbors, rng)
        predicted_beliefs, predicted_p01_values, predicted_p10_values = predict_beliefs(
            cfg,
            previous_beliefs,
            rainfall,
            [point.drainage_capacity for point in points],
            neighbors,
        )
        ml_levels = sample_ml_levels(cfg, true_states, rng)
        pre_action_beliefs = ml_soft_update_all(predicted_beliefs, ml_levels, cfg.ml_observation_matrix)
        monitoring_indices = [
            estimator.monitoring_index(belief, point.importance, predicted_p01_values[point.idx], predicted_p10_values[point.idx])
            for belief, point in zip(pre_action_beliefs, points)
        ]
        matching = solve_rolling_matching(
            cfg,
            monitoring_indices,
            uav_positions,
            locations,
            energy_remaining,
        )
        selected_points = [point for point in matching.assignment_by_uav if point is not None]
        selected_set = set(selected_points)
        selected_distances = [
            matching.edge_distances[(m, point)]
            for m, point in enumerate(matching.assignment_by_uav)
            if point is not None
        ]
        hard_obs = sample_hard_observations(cfg, true_states, selected_points, rng)
        posterior = hard_observation_update_all(
            pre_action_beliefs,
            selected_points,
            hard_obs,
            cfg.uav_sensitivity,
            cfg.uav_specificity,
        )
        for point, belief in zip(points, posterior):
            point.belief = belief

        high_risk_set = [i for i, state in enumerate(true_states) if state == 1]
        tp = len([i for i in selected_points if true_states[i] == 1])
        fp = len([i for i in selected_points if true_states[i] == 0])
        fn = len([i for i in high_risk_set if i not in selected_set])
        reward = reward_from_assignment(cfg, points, selected_points, selected_distances, pre_action_beliefs)
        rewards.append(reward)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        total_monitored += len(selected_points)

        for m, point_idx in enumerate(matching.assignment_by_uav):
            start_x, start_y = uav_positions[m]
            if point_idx is None:
                end_x, end_y = start_x, start_y
                travel_distance = 0.0
                point_name = ""
                monitoring_index = ""
                assignment_score = ""
            else:
                end_x, end_y = locations[point_idx]
                travel_distance = matching.edge_distances[(m, point_idx)]
                energy_remaining[m] -= energy_cost(cfg, travel_distance)
                uav_positions[m] = locations[point_idx]
                point_name = points[point_idx].name
                monitoring_index = round(monitoring_indices[point_idx], 6)
                assignment_score = round(matching.edge_scores[(m, point_idx)], 6)
            path_rows.append(
                {
                    "seed": seed,
                    "time": t,
                    "uav": m,
                    "point_idx": "" if point_idx is None else point_idx,
                    "point_name": point_name,
                    "start_x": round(start_x, 6),
                    "start_y": round(start_y, 6),
                    "end_x": round(end_x, 6),
                    "end_y": round(end_y, 6),
                    "distance": round(travel_distance, 6),
                    "energy_remaining": round(energy_remaining[m], 6),
                    "monitoring_index": monitoring_index,
                    "assignment_score": assignment_score,
                    "reward_t": round(reward, 6),
                    "selected_points_t": str(selected_points),
                }
            )
        rainfall_rows.append(
            {
                "seed": seed,
                "time": t,
                "mean_rainfall": round(float(np.mean(rainfall)), 6),
                "rainfall": str([round(v, 6) for v in rainfall]),
                "high_risk_set": str(high_risk_set),
                "high_risk_count": len(high_risk_set),
                "selected_points": str(selected_points),
                "selected_names": str([points[i].name for i in selected_points]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "reward": round(reward, 6),
            }
        )
        history.append(
            {
                "time": t,
                "rainfall": rainfall,
                "ml_levels": ml_levels,
                "p01": p01_values,
                "p10": p10_values,
                "predicted_beliefs": predicted_beliefs,
                "pre_action_beliefs": pre_action_beliefs,
                "posterior_beliefs": posterior,
                "monitoring_indices": monitoring_indices,
                "assignment": matching.assignment_by_uav,
                "reward": reward,
            }
        )

    risk_rows = [
        {
            "seed": seed,
            "idx": point.idx,
            "name": point.name,
            "x": round(point.location[0], 6),
            "y": round(point.location[1], 6),
            "flood_susceptibility": round(point.flood_susceptibility, 6),
            "drainage_capacity": round(point.drainage_capacity, 6),
            "importance": round(point.importance, 6),
            "final_state": point.true_state,
            "final_belief": round(point.belief, 6),
        }
        for point in points
    ]
    metrics = metric_dict(tp_total, fp_total, fn_total, rewards, total_monitored)
    metrics["seed"] = seed
    return {
        "seed": seed,
        "metrics": metrics,
        "history": history,
        "path_rows": path_rows,
        "rainfall_rows": rainfall_rows,
        "risk_rows": risk_rows,
    }
