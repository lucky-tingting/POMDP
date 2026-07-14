from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class Step10TDTargetAndLossTests(unittest.TestCase):
    def test_target_matching_keeps_unique_point_assignment(self):
        from step_10_td_target_and_loss import matched_target_action_indices

        q_values = torch.tensor(
            [
                [
                    [0.0, 10.0, 9.0, 0.5],
                    [0.0, 9.5, 1.0, 0.5],
                ]
            ],
            dtype=torch.float32,
        )
        action_mask = torch.ones_like(q_values, dtype=torch.bool)

        action_indices, assignment_matrices = matched_target_action_indices(
            q_values=q_values,
            action_masks=action_mask,
            invalid_score=-1.0e9,
        )

        self.assertEqual(action_indices.shape, (1, 2))
        self.assertEqual(assignment_matrices.shape, (1, 2, 3))
        self.assertEqual(set(action_indices[0].tolist()), {1, 2})
        self.assertNotEqual(action_indices[0].tolist(), [1, 1])
        self.assertTrue(torch.all(assignment_matrices.sum(dim=1) <= 1))
        self.assertTrue(torch.all(assignment_matrices.sum(dim=2) <= 1))

    def test_td_target_uses_target_matching_not_independent_argmax(self):
        from step_01_config_and_protocol import BAAMQMixRUMAConfig
        from step_10_td_target_and_loss import compute_matched_qmix_td_target

        class SumMixer(torch.nn.Module):
            def forward(self, agent_qs, global_state):
                return agent_qs.sum(dim=-1)

        cfg = BAAMQMixRUMAConfig(
            n_points=3,
            n_uavs=2,
            local_obs_dim=4,
            global_state_dim=5,
            hidden_dim=8,
            mixer_hidden_dim=4,
            gamma=0.5,
            device="cpu",
        )
        next_q_values = torch.tensor(
            [
                [
                    [0.0, 10.0, 9.0, 0.5],
                    [0.0, 9.5, 1.0, 0.5],
                ]
            ],
            dtype=torch.float32,
        )
        next_action_masks = torch.ones_like(next_q_values, dtype=torch.bool)
        rewards = torch.tensor([1.0])
        dones = torch.tensor([0.0])
        next_global_state = torch.zeros(1, cfg.global_state_dim)

        td_target, target_action_indices, target_assignment_matrices = compute_matched_qmix_td_target(
            cfg=cfg,
            target_mixer=SumMixer(),
            next_q_values=next_q_values,
            next_global_state=next_global_state,
            next_action_masks=next_action_masks,
            rewards=rewards,
            dones=dones,
        )

        self.assertEqual(set(target_action_indices[0].tolist()), {1, 2})
        self.assertTrue(torch.all(target_assignment_matrices.sum(dim=1) <= 1))
        self.assertAlmostEqual(float(td_target.item()), 10.25, places=6)

    def test_episode_sequence_replay_samples_contiguous_fragments(self):
        from step_10_td_target_and_loss import EpisodeSequenceReplayBuffer

        replay = EpisodeSequenceReplayBuffer(capacity_episodes=2, sequence_length=3, seed=7)
        replay.add_episode([{"t": t} for t in range(5)])

        sampled = replay.sample(batch_size=1)

        self.assertEqual(len(sampled), 1)
        self.assertEqual(len(sampled[0]), 3)
        times = [row["t"] for row in sampled[0]]
        self.assertEqual(times, list(range(times[0], times[0] + 3)))


if __name__ == "__main__":
    unittest.main()
