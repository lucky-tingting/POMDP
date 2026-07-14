from __future__ import annotations

from typing import Sequence

from step_01_config_and_parameters import PPBVIRUMAConfig
from step_03_pointwise_pbvi_index import expected_information_gain, miss_penalty_phi


def simplified_belief_state_score(
    cfg: PPBVIRUMAConfig,
    pre_action_belief: float,
    importance: float,
) -> float:
    belief = min(0.999999, max(0.000001, float(pre_action_belief)))
    return (
        cfg.lambda_cover * importance * belief
        + cfg.lambda_miss * importance * miss_penalty_phi(cfg, belief)
        - cfg.lambda_fp * (1.0 - belief)
        + cfg.lambda_info * expected_information_gain(belief, cfg.uav_sensitivity, cfg.uav_specificity)
    )


def compute_mle_bs_ruma_scores(
    cfg: PPBVIRUMAConfig,
    pre_action_beliefs: Sequence[float],
    importances: Sequence[float],
) -> list[float]:
    if len(pre_action_beliefs) != len(importances):
        raise ValueError("pre_action_beliefs and importances must have the same length")
    return [
        simplified_belief_state_score(cfg, belief, importance)
        for belief, importance in zip(pre_action_beliefs, importances)
    ]
