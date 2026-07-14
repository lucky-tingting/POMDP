from __future__ import annotations

from typing import Dict, List

import numpy as np

from step_01_config_and_protocol import BAAMEnvProtocol
from step_06_trainer import BAAMQMIXRUMATrainerTorch
from step_07_online_execution import execute_one_decision_step


def _belief_from_step(env: BAAMEnvProtocol, info: Dict[str, object]) -> List[float]:
    if "posterior_belief" in info:
        return [float(x) for x in info["posterior_belief"]]
    if hasattr(env, "belief_state"):
        return [float(x) for x in np.asarray(env.belief_state(), dtype=float)]
    return [float(x) for x in np.asarray(env.pre_action_belief(), dtype=float)]


def online_execution_report(
    env: BAAMEnvProtocol,
    trainer: BAAMQMIXRUMATrainerTorch,
    horizon: int,
) -> Dict[str, object]:
    """Algorithm 2 online execution output: Y_t, UAV trajectories, beliefs."""

    env.reset()
    hidden = trainer.agent.initial_hidden(trainer.cfg.n_uavs, trainer.device)
    assignment_matrices: List[List[List[int]]] = []
    trajectories: List[List[int]] = [[] for _ in range(trainer.cfg.n_uavs)]
    belief_sequence: List[List[float]] = []
    trace: List[Dict[str, object]] = []
    for _ in range(horizon):
        row = execute_one_decision_step(env, trainer, hidden)
        hidden = row["next_hidden"]
        matrix = np.asarray(row["assignment_matrix"], dtype=np.int64)
        action_indices = np.asarray(row["action_indices"], dtype=np.int64)
        assignment_matrices.append(matrix.tolist())
        for m, action in enumerate(action_indices):
            trajectories[m].append(int(action))
        belief_sequence.append(_belief_from_step(env, dict(row.get("info") or {})))
        trace.append(row)
        if row["done"]:
            break
    return {
        "assignment_matrices": assignment_matrices,
        "trajectories": trajectories,
        "belief_sequence": belief_sequence,
        "trace": trace,
    }
