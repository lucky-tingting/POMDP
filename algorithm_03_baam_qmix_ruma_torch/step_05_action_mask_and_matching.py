from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from step_01_config_and_protocol import BAAMQMixRUMAConfig
from step_02_belief_reward import expected_information_gain_torch


Tensor = torch.Tensor


class CandidateSetBuilder:
    """Construct C_t from belief, importance, entropy, and optional PBVI index."""

    def base_scores(
        self,
        pre_action_belief: np.ndarray,
        point_importance: np.ndarray,
        entropy_value: Optional[np.ndarray] = None,
        monitoring_index: Optional[np.ndarray] = None,
        entropy_weight: float = 0.25,
        monitoring_weight: float = 1.0,
    ) -> np.ndarray:
        belief = np.asarray(pre_action_belief, dtype=float)
        importance = np.asarray(point_importance, dtype=float)
        if belief.shape != importance.shape:
            raise ValueError("pre_action_belief and point_importance must have the same shape")
        if entropy_value is None:
            clipped = np.clip(belief, 1e-6, 1.0 - 1e-6)
            entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
        else:
            entropy = np.asarray(entropy_value, dtype=float)
            if entropy.shape != belief.shape:
                raise ValueError("entropy_value must match pre_action_belief shape")
        scores = importance * belief + entropy_weight * entropy
        if monitoring_index is not None:
            index = np.asarray(monitoring_index, dtype=float)
            if index.shape != belief.shape:
                raise ValueError("monitoring_index must match pre_action_belief shape")
            scores = scores + monitoring_weight * index
        return scores

    def top_k_mask(self, scores: np.ndarray, k: int) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        mask = np.zeros(scores.shape, dtype=bool)
        if k <= 0 or scores.size == 0:
            return mask
        chosen = np.argsort(-scores)[: min(k, scores.size)]
        mask[chosen] = True
        return mask

    def build(
        self,
        pre_action_belief: np.ndarray,
        point_importance: np.ndarray,
        entropy_value: Optional[np.ndarray] = None,
        monitoring_index: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> np.ndarray:
        scores = self.base_scores(
            pre_action_belief=pre_action_belief,
            point_importance=point_importance,
            entropy_value=entropy_value,
            monitoring_index=monitoring_index,
        )
        if score_threshold is None:
            if top_k is None:
                return np.ones(scores.shape, dtype=bool)
            return self.top_k_mask(scores, top_k)
        mask = scores >= float(score_threshold)
        if not np.any(mask):
            fallback_k = len(scores) if top_k is None else top_k
            return self.top_k_mask(scores, fallback_k)
        if top_k is not None and int(mask.sum()) > top_k:
            restricted = np.where(mask, scores, -np.inf)
            return self.top_k_mask(restricted, top_k)
        return mask.astype(bool)


def pairwise_distance(uav_positions: Tensor, point_locations: Tensor) -> Tensor:
    diff = uav_positions[:, None, :] - point_locations[None, :, :]
    return torch.linalg.norm(diff, dim=-1)


def build_assignment_matrix(assignments: Sequence[Optional[int]], n_uavs: int, n_points: int) -> np.ndarray:
    assignment_matrix = np.zeros((n_uavs, n_points), dtype=np.int64)
    for m, point_idx in enumerate(assignments):
        if point_idx is not None:
            assignment_matrix[m, int(point_idx)] = 1
    return assignment_matrix


def assignment_to_action_indices(assignment_matrix: np.ndarray) -> np.ndarray:
    n_uavs, _ = assignment_matrix.shape
    action_indices = np.zeros(n_uavs, dtype=np.int64)
    for m in range(n_uavs):
        selected = np.flatnonzero(assignment_matrix[m] > 0)
        if len(selected) > 0:
            action_indices[m] = int(selected[0]) + 1
    return action_indices


class ActionMaskBuilder:
    """Step 05a: action mask builder for no-visit plus N risk-point actions."""

    def __init__(self, cfg: BAAMQMixRUMAConfig):
        self.cfg = cfg

    def from_reachability(self, reachable_points: np.ndarray) -> np.ndarray:
        mask = np.zeros((self.cfg.n_uavs, self.cfg.n_actions), dtype=bool)
        mask[:, 0] = True
        mask[:, 1:] = reachable_points.astype(bool)
        return mask

    def build_feasibility_mask(
        self,
        distance_matrix: np.ndarray,
        uav_energy: np.ndarray,
        flight_time_matrix: np.ndarray,
        remaining_time: np.ndarray,
        safe_matrix: np.ndarray,
        candidate_mask: np.ndarray,
        max_flight_distance: np.ndarray,
        energy_per_distance: float,
        min_safe_energy: float,
    ) -> np.ndarray:
        """Build A_t^m using distance, energy, time, candidate, and safety constraints."""

        distance_ok = distance_matrix <= max_flight_distance[:, None]
        energy_ok = uav_energy[:, None] - energy_per_distance * distance_matrix >= min_safe_energy
        time_ok = flight_time_matrix <= remaining_time[:, None]
        feasible_points = distance_ok & energy_ok & time_ok & safe_matrix.astype(bool) & candidate_mask[None, :].astype(bool)
        return self.from_reachability(feasible_points)

    def top_k_fallback(
        self,
        pre_action_belief: np.ndarray,
        point_importance: np.ndarray,
        entropy_value: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Fallback C_t = TopK(S_base) when the candidate set is empty."""

        base_score = point_importance * pre_action_belief + 0.25 * entropy_value
        chosen = np.argsort(-base_score)[: max(0, min(k, len(base_score)))]
        candidate_mask = np.zeros(len(base_score), dtype=bool)
        candidate_mask[chosen] = True
        return candidate_mask

    def virtual_action_scores(self, n_uavs: int) -> np.ndarray:
        """Score S_t^{m,0}=0 for the virtual action 0."""

        return np.zeros(n_uavs, dtype=float)


class RollingMatcher:
    """Step 05b: rolling bipartite matching layer for executable Y_t."""

    def __init__(self, cfg: BAAMQMixRUMAConfig):
        self.cfg = cfg

    def compose_pair_scores(
        self,
        q_values: Tensor,
        action_mask: Tensor,
        pre_action_belief: Tensor,
        point_importance: Tensor,
        uav_positions: Tensor,
        point_locations: Tensor,
    ) -> Tensor:
        """Use QMIX local Q as matching score S_t^{m,i}.

        The training reward already includes flight cost, so rolling matching
        does not subtract distance_cost again.
        """

        point_q = q_values[:, 1:]
        info_gain = expected_information_gain_torch(
            pre_action_belief,
            self.cfg.uav_sensitivity,
            self.cfg.uav_specificity,
        )
        _diagnostic_context = pre_action_belief[None, :] + point_importance[None, :] + info_gain[None, :]
        _ = (uav_positions, point_locations, _diagnostic_context)
        scores = self.cfg.q_weight * point_q
        point_mask = action_mask[:, 1:].bool()
        return scores.masked_fill(~point_mask, self.cfg.invalid_score)

    def solve(self, pair_scores: Tensor) -> Tuple[np.ndarray, List[Optional[int]]]:
        scores_np = pair_scores.detach().cpu().numpy()
        n_uavs, n_points = scores_np.shape
        row_ind, col_ind = linear_sum_assignment(-scores_np)
        assignments: List[Optional[int]] = [None] * n_uavs
        for row, col in zip(row_ind, col_ind):
            score = scores_np[row, col]
            if np.isfinite(score) and score > 0.0 and score > self.cfg.invalid_score / 2:
                assignments[int(row)] = int(col)
        return build_assignment_matrix(assignments, n_uavs, n_points), assignments

    def select_assignment(
        self,
        q_values: Tensor,
        action_mask: Tensor,
        pre_action_belief: Tensor,
        point_importance: Tensor,
        uav_positions: Tensor,
        point_locations: Tensor,
    ) -> Tuple[np.ndarray, List[Optional[int]], Tensor]:
        pair_scores = self.compose_pair_scores(
            q_values=q_values,
            action_mask=action_mask,
            pre_action_belief=pre_action_belief,
            point_importance=point_importance,
            uav_positions=uav_positions,
            point_locations=point_locations,
        )
        assignment_matrix, assignments = self.solve(pair_scores)
        return assignment_matrix, assignments, pair_scores
