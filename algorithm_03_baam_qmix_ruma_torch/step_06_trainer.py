from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from step_01_config_and_protocol import BAAMQMixRUMAConfig
from step_03_agent_gru import BeliefAwareAgentGRU
from step_04_qmix_mixer import MonotonicQMIXMixerTorch
from step_05_action_mask_and_matching import RollingMatcher, assignment_to_action_indices
from step_10_td_target_and_loss import compute_matched_qmix_td_target, qmix_td_loss


Tensor = torch.Tensor


@dataclass
class TorchTransitionBatch:
    local_obs: Tensor
    global_state: Tensor
    action_indices: Tensor
    rewards: Tensor
    next_local_obs: Tensor
    next_global_state: Tensor
    next_action_masks: Tensor
    dones: Tensor


class BAAMQMIXRUMATrainerTorch:
    """Step 06: CTDE trainer with GRU agent, QMIX mixer, and matching execution."""

    def __init__(self, cfg: BAAMQMixRUMAConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.agent = BeliefAwareAgentGRU(cfg).to(self.device)
        self.target_agent = copy.deepcopy(self.agent).to(self.device)
        self.mixer = MonotonicQMIXMixerTorch(cfg).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        self.matcher = RollingMatcher(cfg)
        self.optimizer = torch.optim.Adam(
            list(self.agent.parameters()) + list(self.mixer.parameters()),
            lr=cfg.learning_rate,
        )
        self.train_steps = 0

    def act(
        self,
        local_obs: np.ndarray,
        hidden_state: Tensor,
        action_mask: np.ndarray,
        pre_action_belief: np.ndarray,
        point_importance: np.ndarray,
        uav_positions: np.ndarray,
        point_locations: np.ndarray,
    ) -> Dict[str, object]:
        self.agent.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(local_obs, dtype=torch.float32, device=self.device)
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            belief_t = torch.as_tensor(pre_action_belief, dtype=torch.float32, device=self.device)
            importance_t = torch.as_tensor(point_importance, dtype=torch.float32, device=self.device)
            uav_pos_t = torch.as_tensor(uav_positions, dtype=torch.float32, device=self.device)
            point_loc_t = torch.as_tensor(point_locations, dtype=torch.float32, device=self.device)
            q_values, next_hidden = self.agent(obs_t, hidden_state)
            q_values = q_values.masked_fill(~mask_t, self.cfg.invalid_score)
            assignment_matrix, assignments, pair_scores = self.matcher.select_assignment(
                q_values=q_values,
                action_mask=mask_t,
                pre_action_belief=belief_t,
                point_importance=importance_t,
                uav_positions=uav_pos_t,
                point_locations=point_loc_t,
            )
        return {
            "assignment_matrix": assignment_matrix,
            "assignments": assignments,
            "action_indices": assignment_to_action_indices(assignment_matrix),
            "pair_scores": pair_scores.detach().cpu().numpy(),
            "next_hidden": next_hidden,
        }

    def _chosen_agent_qs(self, q_values: Tensor, action_indices: Tensor) -> Tensor:
        return q_values.gather(dim=-1, index=action_indices.unsqueeze(-1)).squeeze(-1)

    def train_step(self, batch: TorchTransitionBatch) -> Dict[str, float]:
        local_obs = batch.local_obs.to(self.device)
        global_state = batch.global_state.to(self.device)
        action_indices = batch.action_indices.to(self.device).long()
        rewards = batch.rewards.to(self.device).float()
        next_local_obs = batch.next_local_obs.to(self.device)
        next_global_state = batch.next_global_state.to(self.device)
        next_action_masks = batch.next_action_masks.to(self.device).bool()
        dones = batch.dones.to(self.device).float()

        if local_obs.ndim == 3:
            local_obs = local_obs.unsqueeze(1)
            global_state = global_state.unsqueeze(1)
            action_indices = action_indices.unsqueeze(1)
            rewards = rewards.unsqueeze(1)
            next_local_obs = next_local_obs.unsqueeze(1)
            next_global_state = next_global_state.unsqueeze(1)
            next_action_masks = next_action_masks.unsqueeze(1)
            dones = dones.unsqueeze(1)
        if local_obs.ndim != 4:
            raise ValueError("local_obs must have shape [batch, sequence, n_uavs, obs_dim]")

        batch_size, sequence_length = local_obs.shape[:2]
        hidden0 = self.agent.initial_hidden(batch_size * self.cfg.n_uavs, self.device)
        hidden = hidden0.view(batch_size, self.cfg.n_uavs, self.cfg.hidden_dim)
        q_tot_rows = []
        for t in range(sequence_length):
            q_values_t, hidden = self.agent.forward_multi_uav(local_obs[:, t], hidden)
            q_tot_rows.append(
                self.mixer(self._chosen_agent_qs(q_values_t, action_indices[:, t]), global_state[:, t])
            )
        q_tot = torch.stack(q_tot_rows, dim=1)

        with torch.no_grad():
            target_hidden0 = self.target_agent.initial_hidden(batch_size * self.cfg.n_uavs, self.device)
            target_hidden = target_hidden0.view(batch_size, self.cfg.n_uavs, self.cfg.hidden_dim)
            td_target_rows = []
            target_action_rows = []
            for t in range(sequence_length):
                next_q_values_t, target_hidden = self.target_agent.forward_multi_uav(next_local_obs[:, t], target_hidden)
                next_q_values_t = next_q_values_t.masked_fill(~next_action_masks[:, t], self.cfg.invalid_score)
                td_target_t, target_action_indices_t, _target_assignment_matrices = compute_matched_qmix_td_target(
                    cfg=self.cfg,
                    target_mixer=self.target_mixer,
                    next_q_values=next_q_values_t,
                    next_global_state=next_global_state[:, t],
                    next_action_masks=next_action_masks[:, t],
                    rewards=rewards[:, t],
                    dones=dones[:, t],
                )
                td_target_rows.append(td_target_t)
                target_action_rows.append(target_action_indices_t)
            td_target = torch.stack(td_target_rows, dim=1)
            target_action_indices = torch.stack(target_action_rows, dim=1)

        loss = qmix_td_loss(q_tot.reshape(-1), td_target.reshape(-1))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent.parameters()) + list(self.mixer.parameters()),
            self.cfg.grad_clip_norm,
        )
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_interval == 0:
            self.update_targets()
        return {
            "loss": float(loss.detach().cpu().item()),
            "mean_q_tot": float(q_tot.detach().mean().cpu().item()),
            "mean_target": float(td_target.detach().mean().cpu().item()),
            "mean_target_real_actions": float((target_action_indices > 0).float().mean().cpu().item()),
        }

    def update_targets(self) -> None:
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.cfg.__dict__,
                "agent": self.agent.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_agent": self.target_agent.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "BAAMQMIXRUMATrainerTorch":
        checkpoint = torch.load(path, map_location=device or "cpu")
        cfg_dict = dict(checkpoint["config"])
        if device is not None:
            cfg_dict["device"] = device
        trainer = cls(BAAMQMixRUMAConfig(**cfg_dict))
        trainer.agent.load_state_dict(checkpoint["agent"])
        trainer.mixer.load_state_dict(checkpoint["mixer"])
        trainer.target_agent.load_state_dict(checkpoint["target_agent"])
        trainer.target_mixer.load_state_dict(checkpoint["target_mixer"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.train_steps = int(checkpoint["train_steps"])
        return trainer
