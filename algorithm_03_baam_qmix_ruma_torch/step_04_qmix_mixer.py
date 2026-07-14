from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from step_01_config_and_protocol import BAAMQMixRUMAConfig


Tensor = torch.Tensor


class MonotonicQMIXMixerTorch(nn.Module):
    """Step 04: monotonic QMIX mixer for centralized training."""

    def __init__(self, cfg: BAAMQMixRUMAConfig):
        super().__init__()
        self.cfg = cfg
        self.hyper_w1 = nn.Sequential(
            nn.Linear(cfg.global_state_dim, cfg.mixer_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.mixer_hidden_dim, cfg.n_uavs * cfg.mixer_hidden_dim),
        )
        self.hyper_b1 = nn.Linear(cfg.global_state_dim, cfg.mixer_hidden_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(cfg.global_state_dim, cfg.mixer_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.mixer_hidden_dim, cfg.mixer_hidden_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(cfg.global_state_dim, cfg.mixer_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.mixer_hidden_dim, 1),
        )

    def forward(self, agent_qs: Tensor, global_state: Tensor) -> Tensor:
        batch_size = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(global_state)).view(batch_size, self.cfg.n_uavs, self.cfg.mixer_hidden_dim)
        b1 = self.hyper_b1(global_state).view(batch_size, 1, self.cfg.mixer_hidden_dim)
        hidden = F.elu(torch.bmm(agent_qs.view(batch_size, 1, self.cfg.n_uavs), w1) + b1)
        w2 = torch.abs(self.hyper_w2(global_state)).view(batch_size, self.cfg.mixer_hidden_dim, 1)
        b2 = self.hyper_b2(global_state).view(batch_size, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(batch_size)
