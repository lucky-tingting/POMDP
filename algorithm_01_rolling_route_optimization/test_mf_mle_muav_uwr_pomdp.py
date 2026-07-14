import math
import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(__file__))


from mf_mle_muav_uwr_pomdp import (
    ModelConfig,
    RollingMatchingPolicy,
    belief_update,
    expected_information_gain,
    run_simulation,
)


class NewPOMDPModelTests(unittest.TestCase):
    def test_unmonitored_point_keeps_ml_updated_belief(self):
        cfg = ModelConfig(n_points=2, n_uavs=1, horizon=1, seed=7)
        predicted = [0.62, 0.62]
        ml_levels = [4, 4]
        actions = [1, 0]
        uav_obs = {0: 1}

        posterior = belief_update(cfg, predicted, ml_levels, actions, uav_obs)

        self.assertGreater(posterior[1], 0.62)
        self.assertLess(posterior[1], 1.0)
        self.assertGreater(posterior[0], posterior[1])

    def test_rolling_matching_respects_assignment_constraints(self):
        cfg = ModelConfig(n_points=5, n_uavs=2, horizon=1, seed=11)
        policy = RollingMatchingPolicy(cfg)
        beliefs = [0.91, 0.87, 0.40, 0.35, 0.20]
        uav_positions = [(0.0, 0.0), (10.0, 0.0)]
        point_locations = [(0.5, 0.0), (9.5, 0.0), (3.0, 3.0), (8.0, 8.0), (20.0, 20.0)]
        importance = [1.0] * 5

        assignment = policy.select_actions(beliefs, uav_positions, point_locations, importance)

        self.assertEqual(len(assignment), 2)
        chosen = [idx for idx in assignment if idx is not None]
        self.assertLessEqual(len(chosen), cfg.n_uavs)
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertTrue(all(0 <= idx < cfg.n_points for idx in chosen))

    def test_information_gain_is_nonnegative_and_larger_near_uncertainty(self):
        cfg = ModelConfig(n_points=1, n_uavs=1, horizon=1)
        uncertain = expected_information_gain(0.5, cfg.uav_sensitivity, cfg.uav_specificity)
        confident = expected_information_gain(0.95, cfg.uav_sensitivity, cfg.uav_specificity)

        self.assertGreaterEqual(uncertain, 0.0)
        self.assertGreaterEqual(confident, 0.0)
        self.assertGreater(uncertain, confident)

    def test_simulation_produces_valid_reproducible_metrics(self):
        cfg = ModelConfig(n_points=12, n_uavs=3, horizon=8, seed=123)
        result_a = run_simulation(cfg, policy_name="rolling_matching", seed=123)
        result_b = run_simulation(cfg, policy_name="rolling_matching", seed=123)

        self.assertEqual(result_a["metrics"], result_b["metrics"])
        self.assertEqual(len(result_a["history"]), cfg.horizon)
        self.assertLessEqual(result_a["metrics"]["total_monitored"], cfg.n_uavs * cfg.horizon)
        for key in ("recall", "precision", "f1"):
            self.assertTrue(0.0 <= result_a["metrics"][key] <= 1.0, key)
            self.assertFalse(math.isnan(result_a["metrics"][key]), key)


if __name__ == "__main__":
    unittest.main()
