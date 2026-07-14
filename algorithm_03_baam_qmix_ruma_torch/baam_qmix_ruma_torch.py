from __future__ import annotations

"""Compatibility entry for the step-based BAAM-QMIX-RUMA PyTorch code.

The canonical implementation is split by algorithm operation steps:
step_01_config_and_protocol.py -> step_08_run_template.py.
"""

from step_01_config_and_protocol import BAAMEnvProtocol, BAAMQMixRUMAConfig, GlobalTrainingState, UAVLocalObservation
from step_02_belief_reward import (
    bayes_update_np,
    entropy_torch,
    expected_information_gain_torch,
    hard_observation_update_np,
    ml_soft_update_np,
    predict_belief_np,
    reward_components_from_assignment_torch,
    reward_from_assignment_torch,
)
from step_03_agent_gru import BeliefAwareAgentGRU
from step_04_qmix_mixer import MonotonicQMIXMixerTorch
from step_05_action_mask_and_matching import (
    ActionMaskBuilder,
    CandidateSetBuilder,
    RollingMatcher,
    assignment_to_action_indices,
    build_assignment_matrix,
    pairwise_distance,
)
from step_06_trainer import BAAMQMIXRUMATrainerTorch, TorchTransitionBatch
from step_07_online_execution import execute_one_decision_step, online_execution_episode
from step_08_run_template import main, write_torch_config_template
from step_10_td_target_and_loss import (
    EpisodeSequenceReplayBuffer,
    compute_matched_qmix_td_target,
    gather_agent_qs,
    matched_target_action_indices,
    qmix_td_loss,
)
from step_11_training_flow import BAAMQMixRUMATrainingLoop, BAAMTransitionRecord, random_feasible_matching
from step_12_online_execution_flow import online_execution_report
from run_baam_qmix_ruma_experiment import (
    BAAMDatasetExperimentConfig,
    BAAMScenarioDataset,
    BAAMScenarioDatasetEnv,
    run_baam_qmix_ruma_dataset_experiment,
)


__all__ = [
    "BAAMEnvProtocol",
    "BAAMQMixRUMAConfig",
    "GlobalTrainingState",
    "UAVLocalObservation",
    "bayes_update_np",
    "entropy_torch",
    "expected_information_gain_torch",
    "hard_observation_update_np",
    "ml_soft_update_np",
    "predict_belief_np",
    "reward_components_from_assignment_torch",
    "reward_from_assignment_torch",
    "BeliefAwareAgentGRU",
    "MonotonicQMIXMixerTorch",
    "ActionMaskBuilder",
    "CandidateSetBuilder",
    "RollingMatcher",
    "assignment_to_action_indices",
    "build_assignment_matrix",
    "pairwise_distance",
    "BAAMQMIXRUMATrainerTorch",
    "TorchTransitionBatch",
    "execute_one_decision_step",
    "online_execution_episode",
    "write_torch_config_template",
    "matched_target_action_indices",
    "gather_agent_qs",
    "compute_matched_qmix_td_target",
    "qmix_td_loss",
    "EpisodeSequenceReplayBuffer",
    "BAAMTransitionRecord",
    "random_feasible_matching",
    "BAAMQMixRUMATrainingLoop",
    "online_execution_report",
    "BAAMDatasetExperimentConfig",
    "BAAMScenarioDataset",
    "BAAMScenarioDatasetEnv",
    "run_baam_qmix_ruma_dataset_experiment",
]


if __name__ == "__main__":
    main()
