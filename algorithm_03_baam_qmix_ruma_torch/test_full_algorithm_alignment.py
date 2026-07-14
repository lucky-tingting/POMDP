from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class TinyProtocolEnv:
    def __init__(self, n_uavs=2, n_points=4, local_obs_dim=5, global_state_dim=6):
        self.n_uavs = n_uavs
        self.n_points = n_points
        self.local_obs_dim = local_obs_dim
        self.global_state_dim = global_state_dim
        self.t = 0
        self.last_assignment = None

    def reset(self):
        self.t = 0
        self.last_assignment = None
        return self.local_observations()

    def local_observations(self):
        return np.full((self.n_uavs, self.local_obs_dim), float(self.t), dtype=np.float32)

    def global_state(self):
        return np.full(self.global_state_dim, float(self.t), dtype=np.float32)

    def global_training_state(self):
        return self.global_state()

    def action_masks(self):
        mask = np.ones((self.n_uavs, self.n_points + 1), dtype=bool)
        mask[1, 1] = False
        return mask

    def pre_action_belief(self):
        return np.linspace(0.2, 0.8, self.n_points, dtype=np.float32) + 0.01 * self.t

    def point_importance(self):
        return np.linspace(1.0, 1.5, self.n_points, dtype=np.float32)

    def uav_positions(self):
        return np.zeros((self.n_uavs, 2), dtype=np.float32)

    def point_locations(self):
        return np.stack([np.arange(self.n_points), np.zeros(self.n_points)], axis=1).astype(np.float32)

    def step(self, assignment_matrix):
        self.last_assignment = np.asarray(assignment_matrix, dtype=np.int64)
        self.t += 1
        done = self.t >= 2
        posterior = self.pre_action_belief() + 0.05
        info = {"posterior_belief": posterior.tolist(), "time": self.t}
        return self.local_observations(), 1.25, done, info


class FullAlgorithmAlignmentTests(unittest.TestCase):
    def test_candidate_builder_uses_weighted_score_and_top_k_fallback(self):
        from step_05_action_mask_and_matching import CandidateSetBuilder

        belief = np.array([0.05, 0.90, 0.50, 0.20])
        importance = np.array([5.0, 1.0, 1.2, 1.0])
        entropy = np.array([0.1, 0.2, 0.9, 0.3])
        builder = CandidateSetBuilder()

        mask = builder.build(
            pre_action_belief=belief,
            point_importance=importance,
            entropy_value=entropy,
            top_k=2,
            score_threshold=99.0,
        )

        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertEqual(mask.tolist(), [False, True, True, False])

    def test_random_feasible_matching_respects_masks_and_unique_assignment(self):
        from step_11_training_flow import random_feasible_matching

        action_mask = np.array(
            [
                [True, True, True, False],
                [True, True, False, True],
                [True, False, True, True],
            ],
            dtype=bool,
        )

        assignment, action_indices = random_feasible_matching(action_mask, seed=13)

        self.assertEqual(assignment.shape, (3, 3))
        self.assertEqual(action_indices.shape, (3,))
        self.assertTrue(np.all(assignment.sum(axis=1) <= 1))
        self.assertTrue(np.all(assignment.sum(axis=0) <= 1))
        for m, action in enumerate(action_indices):
            self.assertTrue(action_mask[m, action])

    def test_training_loop_collects_episode_transitions_into_sequence_replay(self):
        from step_01_config_and_protocol import BAAMQMixRUMAConfig
        from step_06_trainer import BAAMQMIXRUMATrainerTorch
        from step_11_training_flow import BAAMQMixRUMATrainingLoop

        cfg = BAAMQMixRUMAConfig(
            n_points=4,
            n_uavs=2,
            local_obs_dim=5,
            global_state_dim=6,
            hidden_dim=8,
            mixer_hidden_dim=4,
            device="cpu",
        )
        trainer = BAAMQMIXRUMATrainerTorch(cfg)
        loop = BAAMQMixRUMATrainingLoop(trainer, replay_capacity_episodes=4, sequence_length=2, seed=5)
        episode = loop.collect_episode(TinyProtocolEnv(), horizon=3, epsilon=1.0)

        self.assertGreaterEqual(len(episode), 2)
        self.assertTrue(loop.replay.can_sample())
        first = episode[0]
        self.assertEqual(first.local_obs.shape, (2, 5))
        self.assertEqual(first.global_state.shape, (6,))
        self.assertEqual(first.action_indices.shape, (2,))
        self.assertEqual(first.next_action_masks.shape, (2, 5))

    def test_sequence_replay_preserves_time_axis_for_gru_training(self):
        from step_01_config_and_protocol import BAAMQMixRUMAConfig
        from step_06_trainer import BAAMQMIXRUMATrainerTorch
        from step_11_training_flow import BAAMQMixRUMATrainingLoop

        cfg = BAAMQMixRUMAConfig(
            n_points=4,
            n_uavs=2,
            local_obs_dim=5,
            global_state_dim=6,
            hidden_dim=8,
            mixer_hidden_dim=4,
            device="cpu",
        )
        trainer = BAAMQMIXRUMATrainerTorch(cfg)
        loop = BAAMQMixRUMATrainingLoop(trainer, replay_capacity_episodes=4, sequence_length=2, seed=6)
        loop.collect_episode(TinyProtocolEnv(), horizon=3, epsilon=1.0)

        batch = loop.sample_transition_batch(batch_size=1)

        self.assertEqual(batch.local_obs.shape, (1, 2, cfg.n_uavs, cfg.local_obs_dim))
        self.assertEqual(batch.global_state.shape, (1, 2, cfg.global_state_dim))
        self.assertEqual(batch.action_indices.shape, (1, 2, cfg.n_uavs))
        metrics = trainer.train_step(batch)
        self.assertIn("loss", metrics)

    def test_online_execution_report_returns_required_outputs(self):
        from step_01_config_and_protocol import BAAMQMixRUMAConfig
        from step_06_trainer import BAAMQMIXRUMATrainerTorch
        from step_12_online_execution_flow import online_execution_report

        cfg = BAAMQMixRUMAConfig(
            n_points=4,
            n_uavs=2,
            local_obs_dim=5,
            global_state_dim=6,
            hidden_dim=8,
            mixer_hidden_dim=4,
            device="cpu",
        )
        trainer = BAAMQMIXRUMATrainerTorch(cfg)
        report = online_execution_report(TinyProtocolEnv(), trainer, horizon=2)

        self.assertIn("assignment_matrices", report)
        self.assertIn("trajectories", report)
        self.assertIn("belief_sequence", report)
        self.assertEqual(len(report["assignment_matrices"]), 2)
        self.assertEqual(len(report["trajectories"]), cfg.n_uavs)
        self.assertEqual(len(report["belief_sequence"]), 2)
        for matrix in report["assignment_matrices"]:
            arr = np.asarray(matrix)
            self.assertTrue(np.all(arr.sum(axis=1) <= 1))
            self.assertTrue(np.all(arr.sum(axis=0) <= 1))


if __name__ == "__main__":
    unittest.main()
