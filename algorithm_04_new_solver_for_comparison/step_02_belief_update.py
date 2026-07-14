from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from step_01_config_and_parameters import PPBVIRUMAConfig, Point, distance


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def clip_prob(probability: float) -> float:
    return float(min(0.999999, max(0.000001, probability)))


def bayes_update(prior: float, likelihood_high: float, likelihood_low: float) -> float:
    prior = clip_prob(prior)
    numerator = likelihood_high * prior
    denominator = numerator + likelihood_low * (1.0 - prior)
    if denominator <= 0.0:
        return prior
    return clip_prob(numerator / denominator)


def predict_single_belief(previous_belief: float, p01: float, p10: float) -> float:
    previous_belief = clip_prob(previous_belief)
    return clip_prob(p01 * (1.0 - previous_belief) + (1.0 - p10) * previous_belief)


def build_neighbors(locations: Sequence[Point], radius: float) -> list[list[int]]:
    neighbors: list[list[int]] = []
    for i, loc_i in enumerate(locations):
        current = []
        for j, loc_j in enumerate(locations):
            if i != j and distance(loc_i, loc_j) <= radius:
                current.append(j)
        neighbors.append(current)
    return neighbors


def mean_field_belief(i: int, beliefs: Sequence[float], neighbors: Sequence[Sequence[int]]) -> float:
    if not neighbors[i]:
        return float(np.mean(beliefs))
    return float(np.mean([beliefs[j] for j in neighbors[i]]))


def transition_probabilities(
    cfg: PPBVIRUMAConfig,
    rainfall: float,
    neighbor_mean: float,
    drainage_capacity: float,
) -> Tuple[float, float]:
    h_norm = min(1.5, max(0.0, rainfall / cfg.h_max))
    p01 = sigmoid(cfg.xi0 + cfg.xi1 * h_norm + cfg.xi2 * neighbor_mean - cfg.theta_01)
    p10 = sigmoid(
        cfg.zeta0
        + cfg.zeta1 * (1.0 - min(1.0, h_norm))
        + cfg.zeta2 * drainage_capacity
        - cfg.zeta3 * neighbor_mean
        - cfg.theta_10
    )
    return clip_prob(p01), clip_prob(p10)


def ml_soft_update(
    predicted_belief: float,
    ml_level: int,
    ml_observation_matrix: Sequence[Sequence[float]],
) -> float:
    level_count = len(ml_observation_matrix[0])
    if not 1 <= int(ml_level) <= level_count:
        raise ValueError(f"ml_level must be in 1..{level_count}")
    idx = int(ml_level) - 1
    theta_low = ml_observation_matrix[0][idx]
    theta_high = ml_observation_matrix[1][idx]
    return bayes_update(predicted_belief, theta_high, theta_low)


def hard_observation_update(
    pre_action_belief: float,
    hard_observation: int | None,
    sensitivity: float,
    specificity: float,
) -> float:
    if hard_observation is None:
        return clip_prob(pre_action_belief)
    if hard_observation == 1:
        return bayes_update(pre_action_belief, sensitivity, 1.0 - specificity)
    if hard_observation == 0:
        return bayes_update(pre_action_belief, 1.0 - sensitivity, specificity)
    raise ValueError("hard_observation must be 1, 0, or None")


def predict_beliefs(
    cfg: PPBVIRUMAConfig,
    previous_beliefs: Sequence[float],
    rainfall: Sequence[float],
    drainage_capacity: Sequence[float],
    neighbors: Sequence[Sequence[int]],
) -> tuple[list[float], list[float], list[float]]:
    predicted: list[float] = []
    p01_values: list[float] = []
    p10_values: list[float] = []
    for i, belief in enumerate(previous_beliefs):
        neighbor_mean = mean_field_belief(i, previous_beliefs, neighbors)
        p01, p10 = transition_probabilities(cfg, rainfall[i], neighbor_mean, drainage_capacity[i])
        predicted.append(predict_single_belief(belief, p01, p10))
        p01_values.append(p01)
        p10_values.append(p10)
    return predicted, p01_values, p10_values


def ml_soft_update_all(
    predicted_beliefs: Sequence[float],
    ml_levels: Sequence[int],
    ml_observation_matrix: Sequence[Sequence[float]],
) -> list[float]:
    return [
        ml_soft_update(belief, level, ml_observation_matrix)
        for belief, level in zip(predicted_beliefs, ml_levels)
    ]


def hard_observation_update_all(
    pre_action_beliefs: Sequence[float],
    selected_points: Sequence[int],
    hard_observations: dict[int, int],
    sensitivity: float,
    specificity: float,
) -> list[float]:
    selected = set(selected_points)
    posterior: list[float] = []
    for i, belief in enumerate(pre_action_beliefs):
        posterior.append(
            hard_observation_update(
                belief,
                hard_observations[i] if i in selected else None,
                sensitivity,
                specificity,
            )
        )
    return posterior
