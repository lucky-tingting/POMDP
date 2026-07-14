from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from step_01_config_and_protocol import BAAMQMixRUMAConfig


Tensor = torch.Tensor


def matched_target_action_indices(
    q_values: Tensor,
    action_masks: Tensor,
    invalid_score: float,
) -> tuple[Tensor, Tensor]:
    """Solve target rolling matching for TD bootstrap actions.

    `q_values` has shape [batch, n_uavs, n_points + 1]. Action 0 is the
    virtual no-visit action. Real point actions are solved as a bipartite
    matching so the TD target keeps the one-UAV-one-point constraint.
    """

    if q_values.ndim != 3:
        raise ValueError("q_values must have shape [batch, n_uavs, n_actions]")
    if action_masks.shape != q_values.shape:
        raise ValueError("action_masks must have the same shape as q_values")

    batch_size, n_uavs, n_actions = q_values.shape
    n_points = n_actions - 1
    if n_points <= 0:
        raise ValueError("q_values must include action 0 and at least one real point action")

    point_scores = q_values[:, :, 1:].masked_fill(~action_masks[:, :, 1:].bool(), invalid_score)
    action_indices = torch.zeros((batch_size, n_uavs), dtype=torch.long, device=q_values.device)
    assignment_matrices = torch.zeros((batch_size, n_uavs, n_points), dtype=torch.long, device=q_values.device)

    scores_np = point_scores.detach().cpu().numpy()
    for b in range(batch_size):
        row_ind, col_ind = linear_sum_assignment(-scores_np[b])
        for row, col in zip(row_ind, col_ind):
            score = scores_np[b, row, col]
            if np.isfinite(score) and score > 0.0 and score > invalid_score / 2.0:
                action_indices[b, row] = int(col) + 1
                assignment_matrices[b, row, col] = 1
    return action_indices, assignment_matrices


def gather_agent_qs(q_values: Tensor, action_indices: Tensor) -> Tensor:
    if q_values.ndim != 3:
        raise ValueError("q_values must have shape [batch, n_uavs, n_actions]")
    if action_indices.shape != q_values.shape[:2]:
        raise ValueError("action_indices must have shape [batch, n_uavs]")
    return q_values.gather(dim=-1, index=action_indices.long().unsqueeze(-1)).squeeze(-1)


def compute_matched_qmix_td_target(
    cfg: BAAMQMixRUMAConfig,
    target_mixer: torch.nn.Module,
    next_q_values: Tensor,
    next_global_state: Tensor,
    next_action_masks: Tensor,
    rewards: Tensor,
    dones: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute y_t with target-network scores followed by target matching."""

    target_action_indices, target_assignment_matrices = matched_target_action_indices(
        q_values=next_q_values,
        action_masks=next_action_masks,
        invalid_score=cfg.invalid_score,
    )
    target_agent_qs = gather_agent_qs(next_q_values, target_action_indices)
    next_q_tot = target_mixer(target_agent_qs, next_global_state)
    td_target = rewards + cfg.gamma * (1.0 - dones) * next_q_tot
    return td_target, target_action_indices, target_assignment_matrices


def qmix_td_loss(current_q_tot: Tensor, td_target: Tensor) -> Tensor:
    return F.mse_loss(current_q_tot, td_target.detach())


@dataclass
class EpisodeSequenceReplayBuffer:
    """Episode/sequence replay buffer for GRU-based training."""

    capacity_episodes: int
    sequence_length: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.capacity_episodes <= 0:
            raise ValueError("capacity_episodes must be positive")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self._episodes: List[List[Any]] = []
        self._rng = np.random.default_rng(self.seed)

    def add_episode(self, episode: Sequence[Any]) -> None:
        rows = list(episode)
        if not rows:
            return
        self._episodes.append(rows)
        if len(self._episodes) > self.capacity_episodes:
            self._episodes = self._episodes[-self.capacity_episodes :]

    def can_sample(self, batch_size: int = 1) -> bool:
        eligible = [episode for episode in self._episodes if len(episode) >= self.sequence_length]
        return len(eligible) > 0 and batch_size > 0

    def sample(self, batch_size: int) -> List[List[Any]]:
        if not self.can_sample(batch_size):
            raise ValueError("not enough episode data to sample sequence fragments")
        eligible = [episode for episode in self._episodes if len(episode) >= self.sequence_length]
        fragments: List[List[Any]] = []
        for _ in range(batch_size):
            episode = eligible[int(self._rng.integers(0, len(eligible)))]
            max_start = len(episode) - self.sequence_length
            start = int(self._rng.integers(0, max_start + 1))
            fragments.append(list(episode[start : start + self.sequence_length]))
        return fragments
