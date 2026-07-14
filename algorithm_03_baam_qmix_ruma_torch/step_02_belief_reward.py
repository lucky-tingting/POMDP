from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from step_01_config_and_protocol import BAAMQMixRUMAConfig


Tensor = torch.Tensor


def clip_probability_np(value: np.ndarray | float) -> np.ndarray | float:
    return np.clip(value, 1e-6, 1.0 - 1e-6)


def bayes_update_np(prior: np.ndarray, likelihood_high: np.ndarray, likelihood_low: np.ndarray) -> np.ndarray:
    prior = clip_probability_np(prior)
    numerator = likelihood_high * prior
    denominator = numerator + likelihood_low * (1.0 - prior)
    return clip_probability_np(numerator / np.maximum(denominator, 1e-12))


def predict_belief_np(previous_belief: np.ndarray, p01: np.ndarray, p10: np.ndarray) -> np.ndarray:
    p11 = 1.0 - p10
    return clip_probability_np(p01 * (1.0 - previous_belief) + p11 * previous_belief)


def ml_soft_update_np(predicted_belief: np.ndarray, ml_levels: Sequence[int], likelihood_matrix: np.ndarray) -> np.ndarray:
    levels = np.asarray(ml_levels, dtype=int)
    theta_0 = likelihood_matrix[0, levels]
    theta_1 = likelihood_matrix[1, levels]
    return bayes_update_np(predicted_belief, theta_1, theta_0)


def hard_observation_update_np(
    cfg: BAAMQMixRUMAConfig,
    pre_action_belief: np.ndarray,
    assignment_matrix: np.ndarray,
    hard_observation: np.ndarray,
) -> np.ndarray:
    monitored = assignment_matrix.sum(axis=0) > 0
    posterior = pre_action_belief.copy()
    positive = monitored & (hard_observation == 1)
    negative = monitored & (hard_observation == 0)
    posterior[positive] = bayes_update_np(
        pre_action_belief[positive],
        np.full(positive.sum(), cfg.uav_sensitivity),
        np.full(positive.sum(), 1.0 - cfg.uav_specificity),
    )
    posterior[negative] = bayes_update_np(
        pre_action_belief[negative],
        np.full(negative.sum(), 1.0 - cfg.uav_sensitivity),
        np.full(negative.sum(), cfg.uav_specificity),
    )
    return clip_probability_np(posterior)


def entropy_torch(probability: Tensor) -> Tensor:
    p = torch.clamp(probability, 1e-6, 1.0 - 1e-6)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def expected_information_gain_torch(pre_action_belief: Tensor, sensitivity: float, specificity: float) -> Tensor:
    belief = torch.clamp(pre_action_belief, 1e-6, 1.0 - 1e-6)
    p_z1 = sensitivity * belief + (1.0 - specificity) * (1.0 - belief)
    p_z0 = 1.0 - p_z1
    post_z1 = sensitivity * belief / torch.clamp(p_z1, min=1e-6)
    post_z0 = (1.0 - sensitivity) * belief / torch.clamp(p_z0, min=1e-6)
    gain = entropy_torch(belief) - p_z1 * entropy_torch(post_z1) - p_z0 * entropy_torch(post_z0)
    return torch.clamp(gain, min=0.0)


def reward_from_assignment_torch(
    cfg: BAAMQMixRUMAConfig,
    pre_action_belief: Tensor,
    point_importance: Tensor,
    assignment_matrix: Tensor,
    distance_matrix: Tensor,
) -> Tensor:
    return reward_components_from_assignment_torch(
        cfg,
        pre_action_belief,
        point_importance,
        assignment_matrix,
        distance_matrix,
    )["total"]


def reward_components_from_assignment_torch(
    cfg: BAAMQMixRUMAConfig,
    pre_action_belief: Tensor,
    point_importance: Tensor,
    assignment_matrix: Tensor,
    distance_matrix: Tensor,
) -> dict[str, Tensor]:
    monitored = torch.clamp(assignment_matrix.sum(dim=0), 0.0, 1.0)
    cover = cfg.lambda_cover * torch.sum(point_importance * pre_action_belief * monitored)
    miss_phi = torch.clamp(pre_action_belief - cfg.miss_threshold, min=0.0)
    miss = cfg.lambda_miss * torch.sum(point_importance * miss_phi * (1.0 - monitored))
    false_positive = cfg.lambda_fp * torch.sum((1.0 - pre_action_belief) * monitored)
    cost = cfg.lambda_cost * torch.sum(assignment_matrix * distance_matrix)
    info_gain = expected_information_gain_torch(pre_action_belief, cfg.uav_sensitivity, cfg.uav_specificity)
    info = cfg.lambda_info * torch.sum(monitored * info_gain)
    total = cover - miss - false_positive - cost + info
    return {
        "cover": cover,
        "miss": miss,
        "false_positive": false_positive,
        "cost": cost,
        "info": info,
        "total": total,
    }
