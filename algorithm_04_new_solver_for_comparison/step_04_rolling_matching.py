from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Sequence

from step_01_config_and_parameters import PPBVIRUMAConfig, Point, distance, energy_cost


@dataclass
class RollingMatchingResult:
    assignment_by_uav: list[Optional[int]]
    assignment_matrix: list[list[int]]
    edge_scores: dict[tuple[int, int], float]
    edge_distances: dict[tuple[int, int], float]
    total_score: float
    candidate_indices: list[int]
    virtual_task_score: float = 0.0


def feasible_edge(
    cfg: PPBVIRUMAConfig,
    uav_position: Point,
    point_location: Point,
    energy_remaining: float,
) -> tuple[bool, float]:
    travel_distance = distance(uav_position, point_location)
    if travel_distance > cfg.max_route_distance + 1e-9:
        return False, travel_distance
    if energy_remaining - energy_cost(cfg, travel_distance) < cfg.min_safe_energy - 1e-9:
        return False, travel_distance
    return True, travel_distance


def solve_rolling_matching(
    cfg: PPBVIRUMAConfig,
    monitoring_indices: Sequence[float],
    uav_positions: Sequence[Point],
    point_locations: Sequence[Point],
    energy_remaining: Sequence[float],
    candidate_indices: Sequence[int] | None = None,
    top_k: int | None = None,
) -> RollingMatchingResult:
    if len(uav_positions) != cfg.n_uavs:
        raise ValueError("uav_positions length must match cfg.n_uavs")
    if len(point_locations) != cfg.n_points:
        raise ValueError("point_locations length must match cfg.n_points")
    if len(monitoring_indices) != cfg.n_points:
        raise ValueError("monitoring_indices length must match cfg.n_points")
    if candidate_indices is None:
        candidates = list(range(cfg.n_points))
    else:
        candidates = [int(i) for i in candidate_indices]
    if not candidates:
        k = min(cfg.n_points, top_k or cfg.candidate_top_k)
        candidates = sorted(range(cfg.n_points), key=lambda i: monitoring_indices[i], reverse=True)[:k]
    candidates = [i for i in candidates if 0 <= i < cfg.n_points]

    edge_scores: dict[tuple[int, int], float] = {}
    edge_distances: dict[tuple[int, int], float] = {}
    choices_by_uav: list[list[Optional[int]]] = []
    for m in range(cfg.n_uavs):
        choices: list[Optional[int]] = [None]
        for i in candidates:
            ok, travel_distance = feasible_edge(cfg, uav_positions[m], point_locations[i], energy_remaining[m])
            if not ok:
                continue
            edge_scores[(m, i)] = float(monitoring_indices[i] - cfg.lambda_cost * travel_distance)
            edge_distances[(m, i)] = travel_distance
            choices.append(i)
        choices_by_uav.append(choices)

    best_assignment: list[Optional[int]] = [None for _ in range(cfg.n_uavs)]
    best_score = 0.0
    for candidate in itertools.product(*choices_by_uav):
        selected = [point for point in candidate if point is not None]
        if len(selected) != len(set(selected)):
            continue
        total = 0.0
        feasible = True
        for m, point in enumerate(candidate):
            if point is None:
                continue
            score = edge_scores.get((m, point))
            if score is None:
                feasible = False
                break
            total += score
        if feasible and total > best_score:
            best_score = total
            best_assignment = list(candidate)

    matrix = [[0 for _ in range(cfg.n_points)] for _ in range(cfg.n_uavs)]
    for m, point in enumerate(best_assignment):
        if point is not None:
            matrix[m][point] = 1
    return RollingMatchingResult(
        assignment_by_uav=best_assignment,
        assignment_matrix=matrix,
        edge_scores=edge_scores,
        edge_distances=edge_distances,
        total_score=best_score,
        candidate_indices=candidates,
        virtual_task_score=0.0,
    )
