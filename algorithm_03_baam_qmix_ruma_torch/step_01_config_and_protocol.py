from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class BAAMQMixRUMAConfig:
    """Step 01: configuration for BAAM-QMIX-RUMA."""

    n_points: int = 1000
    n_uavs: int = 5
    local_obs_dim: int = 32
    global_state_dim: int = 64
    hidden_dim: int = 128
    mixer_hidden_dim: int = 64
    gamma: float = 0.95
    learning_rate: float = 3e-4
    grad_clip_norm: float = 10.0
    target_update_interval: int = 100
    replay_batch_size: int = 32

    lambda_cover: float = 18.0
    lambda_miss: float = 8.0
    lambda_fp: float = 3.5
    lambda_cost: float = 0.18
    lambda_info: float = 3.0
    miss_threshold: float = 0.25

    q_weight: float = 1.0
    belief_weight: float = 1.0
    importance_weight: float = 1.0
    info_weight: float = 0.2
    invalid_score: float = -1.0e9

    uav_sensitivity: float = 0.88
    uav_specificity: float = 0.86
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def n_actions(self) -> int:
        return self.n_points + 1


@dataclass(frozen=True)
class GlobalTrainingState:
    """Dec-POMDP training state s_tr = (x, b_bar, q, E, eta)."""

    true_risk_state: np.ndarray
    pre_action_belief: np.ndarray
    uav_positions: np.ndarray
    uav_energy: np.ndarray
    exogenous_state: np.ndarray


@dataclass(frozen=True)
class UAVLocalObservation:
    """Local observation upsilon_t^m for one UAV."""

    uav_position: np.ndarray
    uav_energy: float
    candidate_indices: np.ndarray
    candidate_locations: np.ndarray
    candidate_pre_action_belief: np.ndarray
    candidate_importance: np.ndarray
    candidate_distance: np.ndarray
    previous_action: int
    monitoring_index: np.ndarray | None = None


class BAAMEnvProtocol(Protocol):
    """Environment interface used by the PyTorch BAAM-QMIX-RUMA trainer."""

    n_uavs: int
    n_points: int

    def reset(self): ...

    def local_observations(self) -> Sequence[np.ndarray]: ...

    def global_training_state(self) -> GlobalTrainingState: ...

    def global_state(self) -> np.ndarray: ...

    def action_masks(self) -> np.ndarray: ...

    def pre_action_belief(self) -> np.ndarray: ...

    def point_importance(self) -> np.ndarray: ...

    def uav_positions(self) -> np.ndarray: ...

    def point_locations(self) -> np.ndarray: ...

    def step(self, assignment_matrix: np.ndarray): ...
