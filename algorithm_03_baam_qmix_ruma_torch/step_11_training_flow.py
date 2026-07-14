from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from step_01_config_and_protocol import BAAMEnvProtocol
from step_06_trainer import BAAMQMIXRUMATrainerTorch, TorchTransitionBatch
from step_10_td_target_and_loss import EpisodeSequenceReplayBuffer


@dataclass
class BAAMTransitionRecord:
    local_obs: np.ndarray
    global_state: np.ndarray
    action_masks: np.ndarray
    action_indices: np.ndarray
    assignment_matrix: np.ndarray
    reward: float
    next_local_obs: np.ndarray
    next_global_state: np.ndarray
    next_action_masks: np.ndarray
    done: bool
    info: Dict[str, object]


def random_feasible_matching(
    action_mask: np.ndarray,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an epsilon-exploration matching satisfying masks and uniqueness."""

    mask = np.asarray(action_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] < 2:
        raise ValueError("action_mask must have shape [n_uavs, n_points + 1]")
    random = rng or np.random.default_rng(seed)
    n_uavs, n_actions = mask.shape
    n_points = n_actions - 1
    assignment = np.zeros((n_uavs, n_points), dtype=np.int64)
    action_indices = np.zeros(n_uavs, dtype=np.int64)
    used_points: set[int] = set()
    for m in range(n_uavs):
        feasible_points = [i for i in range(n_points) if mask[m, i + 1] and i not in used_points]
        feasible_actions = ([0] if mask[m, 0] else []) + [i + 1 for i in feasible_points]
        if not feasible_actions:
            action = 0
        else:
            action = int(random.choice(np.asarray(feasible_actions, dtype=np.int64)))
        action_indices[m] = action
        if action > 0:
            point_idx = action - 1
            assignment[m, point_idx] = 1
            used_points.add(point_idx)
    return assignment, action_indices


class BAAMQMixRUMATrainingLoop:
    """Algorithm 1 training flow around a BAAMEnvProtocol environment."""

    def __init__(
        self,
        trainer: BAAMQMIXRUMATrainerTorch,
        replay_capacity_episodes: int = 64,
        sequence_length: int = 8,
        seed: int = 0,
    ):
        self.trainer = trainer
        self.replay = EpisodeSequenceReplayBuffer(replay_capacity_episodes, sequence_length, seed=seed)
        self.rng = np.random.default_rng(seed)

    def collect_episode(self, env: BAAMEnvProtocol, horizon: int, epsilon: float) -> List[BAAMTransitionRecord]:
        env.reset()
        hidden = self.trainer.agent.initial_hidden(self.trainer.cfg.n_uavs, self.trainer.device)
        episode: List[BAAMTransitionRecord] = []
        for _ in range(horizon):
            local_obs = np.asarray(env.local_observations(), dtype=np.float32)
            global_state = np.asarray(env.global_state(), dtype=np.float32)
            action_masks = np.asarray(env.action_masks(), dtype=bool)

            if self.rng.random() < epsilon:
                assignment_matrix, action_indices = random_feasible_matching(action_masks, rng=self.rng)
                with torch.no_grad():
                    obs_t = torch.as_tensor(local_obs, dtype=torch.float32, device=self.trainer.device)
                    _q_values, hidden = self.trainer.agent(obs_t, hidden)
            else:
                action = self.trainer.act(
                    local_obs=local_obs,
                    hidden_state=hidden,
                    action_mask=action_masks,
                    pre_action_belief=np.asarray(env.pre_action_belief(), dtype=np.float32),
                    point_importance=np.asarray(env.point_importance(), dtype=np.float32),
                    uav_positions=np.asarray(env.uav_positions(), dtype=np.float32),
                    point_locations=np.asarray(env.point_locations(), dtype=np.float32),
                )
                assignment_matrix = np.asarray(action["assignment_matrix"], dtype=np.int64)
                action_indices = np.asarray(action["action_indices"], dtype=np.int64)
                hidden = action["next_hidden"]

            _next_obs_raw, reward, done, info = env.step(assignment_matrix)
            record = BAAMTransitionRecord(
                local_obs=local_obs,
                global_state=global_state,
                action_masks=action_masks,
                action_indices=action_indices,
                assignment_matrix=assignment_matrix,
                reward=float(reward),
                next_local_obs=np.asarray(env.local_observations(), dtype=np.float32),
                next_global_state=np.asarray(env.global_state(), dtype=np.float32),
                next_action_masks=np.asarray(env.action_masks(), dtype=bool),
                done=bool(done),
                info=dict(info or {}),
            )
            episode.append(record)
            if done:
                break
        self.replay.add_episode(episode)
        return episode

    def sample_transition_batch(self, batch_size: int) -> TorchTransitionBatch:
        fragments = self.replay.sample(batch_size)
        return TorchTransitionBatch(
            local_obs=torch.as_tensor(np.stack([[r.local_obs for r in fragment] for fragment in fragments]), dtype=torch.float32),
            global_state=torch.as_tensor(np.stack([[r.global_state for r in fragment] for fragment in fragments]), dtype=torch.float32),
            action_indices=torch.as_tensor(np.stack([[r.action_indices for r in fragment] for fragment in fragments]), dtype=torch.long),
            rewards=torch.as_tensor([[r.reward for r in fragment] for fragment in fragments], dtype=torch.float32),
            next_local_obs=torch.as_tensor(np.stack([[r.next_local_obs for r in fragment] for fragment in fragments]), dtype=torch.float32),
            next_global_state=torch.as_tensor(np.stack([[r.next_global_state for r in fragment] for fragment in fragments]), dtype=torch.float32),
            next_action_masks=torch.as_tensor(np.stack([[r.next_action_masks for r in fragment] for fragment in fragments]), dtype=torch.bool),
            dones=torch.as_tensor([[r.done for r in fragment] for fragment in fragments], dtype=torch.float32),
        )

    def update_from_replay(self, batch_size: int) -> Dict[str, float]:
        return self.trainer.train_step(self.sample_transition_batch(batch_size))

    def fit_env(
        self,
        env: BAAMEnvProtocol,
        episodes: int,
        horizon: int,
        epsilon_start: float = 0.5,
        epsilon_end: float = 0.05,
        updates_per_episode: int = 1,
    ) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for ep in range(episodes):
            epsilon = max(
                epsilon_end,
                epsilon_start - (epsilon_start - epsilon_end) * ep / max(1, episodes - 1),
            )
            episode = self.collect_episode(env, horizon=horizon, epsilon=epsilon)
            row: Dict[str, float] = {
                "episode": float(ep),
                "epsilon": float(epsilon),
                "episode_reward": float(sum(record.reward for record in episode)),
            }
            if self.replay.can_sample():
                for _ in range(updates_per_episode):
                    row.update(self.update_from_replay(batch_size=1))
            history.append(row)
        return history
