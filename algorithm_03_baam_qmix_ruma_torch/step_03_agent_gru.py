from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from step_01_config_and_protocol import BAAMQMixRUMAConfig


Tensor = torch.Tensor


class BeliefAwareAgentGRU(nn.Module):
    """Step 03: local recurrent Q network for each UAV."""

    def __init__(self, cfg: BAAMQMixRUMAConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.local_obs_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.n_actions),
        )

    def initial_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> Tensor:
        return torch.zeros(batch_size, self.cfg.hidden_dim, device=device or next(self.parameters()).device)

    def forward(self, local_obs: Tensor, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        encoded = self.encoder(local_obs)
        next_hidden = self.gru(encoded, hidden_state)
        q_values = self.q_head(next_hidden)
        return q_values, next_hidden

    def forward_multi_uav(self, local_obs: Tensor, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, n_uavs, obs_dim = local_obs.shape
        flat_obs = local_obs.reshape(batch_size * n_uavs, obs_dim)
        flat_hidden = hidden_state.reshape(batch_size * n_uavs, self.cfg.hidden_dim)
        flat_q, flat_next_hidden = self.forward(flat_obs, flat_hidden)
        return (
            flat_q.reshape(batch_size, n_uavs, self.cfg.n_actions),
            flat_next_hidden.reshape(batch_size, n_uavs, self.cfg.hidden_dim),
        )
