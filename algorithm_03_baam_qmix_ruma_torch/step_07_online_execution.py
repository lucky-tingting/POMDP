from __future__ import annotations

from typing import Dict, List

import numpy as np

from step_01_config_and_protocol import BAAMEnvProtocol
from step_06_trainer import BAAMQMIXRUMATrainerTorch


def execute_one_decision_step(
    env: BAAMEnvProtocol,
    trainer: BAAMQMIXRUMATrainerTorch,
    hidden_state,
) -> Dict[str, object]:
    """Step 07: decentralized scoring plus rolling matching execution."""

    action = trainer.act(
        local_obs=np.asarray(env.local_observations(), dtype=np.float32),
        hidden_state=hidden_state,
        action_mask=np.asarray(env.action_masks(), dtype=bool),
        pre_action_belief=np.asarray(env.pre_action_belief(), dtype=np.float32),
        point_importance=np.asarray(env.point_importance(), dtype=np.float32),
        uav_positions=np.asarray(env.uav_positions(), dtype=np.float32),
        point_locations=np.asarray(env.point_locations(), dtype=np.float32),
    )
    next_obs, reward, done, info = env.step(action["assignment_matrix"])
    return {
        "assignment_matrix": action["assignment_matrix"],
        "assignments": action["assignments"],
        "action_indices": action["action_indices"],
        "pair_scores": action["pair_scores"],
        "next_hidden": action["next_hidden"],
        "next_obs": next_obs,
        "reward": reward,
        "done": done,
        "info": info,
    }


def online_execution_episode(env: BAAMEnvProtocol, trainer: BAAMQMIXRUMATrainerTorch, horizon: int) -> List[Dict[str, object]]:
    """Run one deployment episode after training."""

    env.reset()
    hidden = trainer.agent.initial_hidden(trainer.cfg.n_uavs, trainer.device)
    trace: List[Dict[str, object]] = []
    for _ in range(horizon):
        row = execute_one_decision_step(env, trainer, hidden)
        hidden = row["next_hidden"]
        trace.append(row)
        if row["done"]:
            break
    return trace
