from __future__ import annotations

import csv
import importlib
import importlib.util
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def require_module(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise AssertionError(f"expected module {name} to exist")
    return importlib.import_module(name)


class PPBVIRUMATests(unittest.TestCase):
    def test_default_config_matches_shared_comparison_parameter_lock(self):
        step_01 = require_module("step_01_config_and_parameters")

        cfg = step_01.PPBVIRUMAConfig()
        locked = step_01.load_shared_comparison_settings()

        self.assertEqual(cfg.n_points, locked["n_points"])
        self.assertEqual(cfg.n_uavs, locked["n_uavs"])
        self.assertEqual(cfg.horizon, locked["horizon"])
        self.assertEqual(tuple(cfg.seeds), tuple(locked["seeds"]))
        self.assertEqual(cfg.max_points_per_uav_per_step, 1)
        self.assertEqual(cfg.max_uavs_per_point_per_step, 1)
        self.assertEqual(cfg.max_route_distance, locked["max_route_distance"])
        self.assertEqual(cfg.energy_capacity, locked["energy_capacity"])
        self.assertEqual(cfg.energy_per_distance, locked["energy_per_distance"])
        self.assertEqual(cfg.min_safe_energy, locked["min_safe_energy"])

    def test_three_stage_belief_update_keeps_unmonitored_point_at_pre_action_belief(self):
        step_01 = require_module("step_01_config_and_parameters")
        step_02 = require_module("step_02_belief_update")

        cfg = step_01.PPBVIRUMAConfig(n_points=3, n_uavs=1, horizon=2)
        predicted = step_02.predict_single_belief(previous_belief=0.40, p01=0.20, p10=0.10)
        pre_action = step_02.ml_soft_update(predicted, ml_level=4, ml_observation_matrix=cfg.ml_observation_matrix)
        unmonitored = step_02.hard_observation_update(
            pre_action,
            hard_observation=None,
            sensitivity=cfg.uav_sensitivity,
            specificity=cfg.uav_specificity,
        )
        monitored_high = step_02.hard_observation_update(
            pre_action,
            hard_observation=1,
            sensitivity=cfg.uav_sensitivity,
            specificity=cfg.uav_specificity,
        )

        self.assertAlmostEqual(predicted, 0.48)
        self.assertGreater(pre_action, predicted)
        self.assertAlmostEqual(unmonitored, pre_action)
        self.assertGreater(monitored_high, pre_action)

    def test_pointwise_local_reward_has_no_flight_cost_and_pbvi_index_matches_one_step_difference(self):
        step_01 = require_module("step_01_config_and_parameters")
        step_03 = require_module("step_03_pointwise_pbvi_index")

        cfg = step_01.PPBVIRUMAConfig(pbvi_iterations=1)
        belief = 0.72
        importance = 1.25

        monitor_reward = step_03.local_immediate_reward(cfg, belief, action=1, importance=importance)
        skip_reward = step_03.local_immediate_reward(cfg, belief, action=0, importance=importance)
        index = step_03.PointwisePBVIIndexEstimator(cfg).monitoring_index(belief, importance)

        expected_monitor = (
            cfg.lambda_cover * importance * belief
            - cfg.lambda_fp * (1.0 - belief)
            + cfg.lambda_info
            * step_03.expected_information_gain(belief, cfg.uav_sensitivity, cfg.uav_specificity)
        )
        expected_skip = -cfg.lambda_miss * importance * max(0.0, belief - cfg.miss_threshold)

        self.assertAlmostEqual(monitor_reward, expected_monitor)
        self.assertAlmostEqual(skip_reward, expected_skip)
        solution = step_03.PointwisePBVIIndexEstimator(cfg).solve_values(
            importance=importance,
            p01=cfg.local_transition_p01,
            p10=cfg.local_transition_p10,
        )
        estimator = step_03.PointwisePBVIIndexEstimator(cfg)
        interpolated_index = estimator.value_at(belief, solution.q_monitor) - estimator.value_at(belief, solution.q_skip)
        self.assertAlmostEqual(index, interpolated_index)
        self.assertNotIn("distance", step_03.local_immediate_reward.__code__.co_varnames)

    def test_section_5_3_local_belief_operators_follow_hard_transition_and_soft_updates(self):
        step_01 = require_module("step_01_config_and_parameters")
        step_03 = require_module("step_03_pointwise_pbvi_index")

        cfg = step_01.PPBVIRUMAConfig()
        belief = 0.60

        self.assertEqual(step_03.hard_observation_set(action=1), [0, 1])
        self.assertEqual(step_03.hard_observation_set(action=0), [None])

        monitored_high = step_03.hard_belief_operator(cfg, belief, action=1, hard_observation=1)
        unmonitored = step_03.hard_belief_operator(cfg, belief, action=0, hard_observation=None)
        transitioned = step_03.transition_operator(monitored_high, p01=0.20, p10=0.10)
        soft_updated = step_03.soft_belief_operator(
            transitioned,
            ml_level=4,
            ml_observation_matrix=cfg.ml_observation_matrix,
        )

        expected_monitored_high = step_03.bayes_update(belief, cfg.uav_sensitivity, 1.0 - cfg.uav_specificity)
        expected_transitioned = 0.20 * (1.0 - expected_monitored_high) + 0.90 * expected_monitored_high

        self.assertAlmostEqual(monitored_high, expected_monitored_high)
        self.assertAlmostEqual(unmonitored, belief)
        self.assertAlmostEqual(transitioned, expected_transitioned)
        self.assertGreater(soft_updated, transitioned)

    def test_section_5_4_action_value_uses_hard_observation_transition_and_next_ml_expectation(self):
        step_01 = require_module("step_01_config_and_parameters")
        step_03 = require_module("step_03_pointwise_pbvi_index")

        cfg = step_01.PPBVIRUMAConfig(
            n_ml_levels=2,
            ml_observation_matrix=((0.8, 0.2), (0.2, 0.8)),
            gamma=0.5,
            pbvi_iterations=1,
            belief_grid_size=3,
        )
        estimator = step_03.PointwisePBVIIndexEstimator(cfg, belief_grid=[0.1, 0.5, 0.9])
        values = [1.0, 5.0, 9.0]
        belief = 0.50
        p01 = 0.20
        p10 = 0.10
        importance = 1.0

        monitor_q = estimator.action_value(
            belief=belief,
            action=1,
            importance=importance,
            values=values,
            p01=p01,
            p10=p10,
        )

        expected_future = 0.0
        for z in [0, 1]:
            p_z = step_03.hard_observation_probability(cfg, belief, action=1, hard_observation=z)
            b_z = step_03.hard_belief_operator(cfg, belief, action=1, hard_observation=z)
            tilde_b = step_03.transition_operator(b_z, p01=p01, p10=p10)
            inner = 0.0
            for r in [1, 2]:
                p_r = step_03.soft_observation_probability(tilde_b, r, cfg.ml_observation_matrix)
                b_next = step_03.soft_belief_operator(tilde_b, r, cfg.ml_observation_matrix)
                inner += p_r * estimator.nearest_value(b_next, values)
            expected_future += p_z * inner
        expected = step_03.local_immediate_reward(cfg, belief, 1, importance) + cfg.gamma * expected_future

        self.assertAlmostEqual(monitor_q, expected)

    def test_section_6_pbvi_uses_zero_to_one_grid_linear_interpolation_and_convergence(self):
        step_01 = require_module("step_01_config_and_parameters")
        step_03 = require_module("step_03_pointwise_pbvi_index")

        cfg = step_01.PPBVIRUMAConfig(
            belief_grid_size=4,
            pbvi_iterations=40,
            pbvi_tolerance=1e-3,
            gamma=0.0,
        )
        estimator = step_03.PointwisePBVIIndexEstimator(cfg)

        self.assertEqual(estimator.belief_grid, [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertAlmostEqual(estimator.value_at(0.375, [0.0, 2.5, 5.0, 7.5, 10.0]), 3.75)

        solution = estimator.solve_values(importance=1.0, p01=0.2, p10=0.1)

        self.assertLessEqual(solution.iterations, cfg.pbvi_iterations)
        self.assertLess(solution.max_residual, cfg.pbvi_tolerance)
        self.assertEqual(len(solution.values), len(estimator.belief_grid))
        self.assertEqual(len(solution.q_monitor), len(estimator.belief_grid))
        self.assertEqual(len(solution.q_skip), len(estimator.belief_grid))

        threshold = estimator.pointwise_threshold(importance=1.0, p01=0.2, p10=0.1)
        self.assertTrue(threshold is None or 0.0 <= threshold <= 1.0)

    def test_section_7_matching_uses_topk_fallback_virtual_idle_and_positive_net_utility(self):
        step_01 = require_module("step_01_config_and_parameters")
        step_04 = require_module("step_04_rolling_matching")

        cfg = step_01.PPBVIRUMAConfig(
            n_points=4,
            n_uavs=2,
            horizon=1,
            max_route_distance=30.0,
            lambda_cost=1.0,
            candidate_top_k=2,
        )
        result = step_04.solve_rolling_matching(
            cfg=cfg,
            monitoring_indices=[1.0, 9.0, 8.0, -5.0],
            uav_positions=[(0.0, 0.0), (20.0, 0.0)],
            point_locations=[(0.0, 0.0), (1.0, 0.0), (20.0, 0.0), (5.0, 0.0)],
            energy_remaining=[160.0, 160.0],
            candidate_indices=[],
        )

        self.assertEqual(result.candidate_indices, [1, 2])
        self.assertEqual(result.virtual_task_score, 0.0)
        self.assertEqual(result.assignment_by_uav, [1, 2])
        self.assertEqual(result.edge_scores[(0, 1)], 8.0)
        self.assertEqual(result.edge_scores[(1, 2)], 8.0)

        idle_result = step_04.solve_rolling_matching(
            cfg=cfg,
            monitoring_indices=[-1.0, -2.0, -3.0, -4.0],
            uav_positions=[(0.0, 0.0), (20.0, 0.0)],
            point_locations=[(0.0, 0.0), (1.0, 0.0), (20.0, 0.0), (5.0, 0.0)],
            energy_remaining=[160.0, 160.0],
        )

        self.assertEqual(idle_result.assignment_by_uav, [None, None])
        self.assertEqual(idle_result.total_score, 0.0)

    def test_section_8_region_threshold_filter_constructs_candidates_but_matching_decides_actions(self):
        step_01 = require_module("step_01_config_and_parameters")
        step_04 = require_module("step_04_rolling_matching")
        step_07 = require_module("step_07_region_threshold_filter")

        cfg = step_01.PPBVIRUMAConfig(
            n_points=5,
            n_uavs=1,
            horizon=1,
            max_route_distance=50.0,
            lambda_cost=0.0,
            fallback_top_k=2,
        )
        point_regions = [0, 0, 1, 1, 1]
        threshold_result = step_07.region_threshold_candidates(
            pre_action_beliefs=[0.20, 0.70, 0.40, 0.85, 0.10],
            point_regions=point_regions,
            region_thresholds={0: 0.60, 1: 0.80},
        )
        topk_result = step_07.region_topk_candidates(
            monitoring_indices=[1.0, 5.0, 4.0, 2.0, 9.0],
            point_regions=point_regions,
            top_k_per_region=1,
        )
        fallback_result = step_07.construct_candidate_set(
            mode="regional_threshold",
            pre_action_beliefs=[0.10, 0.20, 0.30, 0.40, 0.10],
            monitoring_indices=[1.0, 9.0, 8.0, 2.0, 7.0],
            point_regions=point_regions,
            region_thresholds={0: 0.95, 1: 0.95},
            fallback_top_k=2,
        )

        self.assertEqual(threshold_result.candidate_indices, [1, 3])
        self.assertFalse(threshold_result.fallback_used)
        self.assertEqual(topk_result.candidate_indices, [1, 4])
        self.assertEqual(fallback_result.candidate_indices, [1, 2])
        self.assertTrue(fallback_result.fallback_used)

        matching = step_04.solve_rolling_matching(
            cfg=cfg,
            monitoring_indices=[-5.0, -1.0, 10.0, 9.0, -2.0],
            uav_positions=[(0.0, 0.0)],
            point_locations=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
            energy_remaining=[160.0],
            candidate_indices=threshold_result.candidate_indices,
        )

        self.assertEqual(matching.candidate_indices, [1, 3])
        self.assertEqual(matching.assignment_by_uav, [3])

    def test_section_13_mle_bs_ruma_score_matches_simplified_formula_and_uses_same_matching(self):
        step_01 = require_module("step_01_config_and_parameters")
        step_04 = require_module("step_04_rolling_matching")
        step_08 = require_module("step_08_mle_bs_ruma_simplified")

        cfg = step_01.PPBVIRUMAConfig(n_points=2, n_uavs=1, horizon=1, max_route_distance=10.0)
        belief = 0.70
        importance = 1.20
        score = step_08.simplified_belief_state_score(cfg, belief, importance)
        expected = (
            cfg.lambda_cover * importance * belief
            + cfg.lambda_miss * importance * max(0.0, belief - cfg.miss_threshold)
            - cfg.lambda_fp * (1.0 - belief)
            + cfg.lambda_info * step_08.expected_information_gain(belief, cfg.uav_sensitivity, cfg.uav_specificity)
        )

        self.assertAlmostEqual(score, expected)
        scores = step_08.compute_mle_bs_ruma_scores(cfg, [0.70, 0.20], [1.20, 1.00])
        matching = step_04.solve_rolling_matching(
            cfg=cfg,
            monitoring_indices=scores,
            uav_positions=[(0.0, 0.0)],
            point_locations=[(0.0, 0.0), (5.0, 0.0)],
            energy_remaining=[160.0],
        )

        self.assertEqual(matching.assignment_by_uav, [0])

    def test_rolling_matching_applies_flight_cost_and_enforces_unique_feasible_assignment(self):
        step_01 = require_module("step_01_config_and_parameters")
        step_04 = require_module("step_04_rolling_matching")

        cfg = step_01.PPBVIRUMAConfig(
            n_points=3,
            n_uavs=2,
            horizon=1,
            max_route_distance=10.0,
            energy_capacity=20.0,
            min_safe_energy=5.0,
            lambda_cost=0.5,
        )
        result = step_04.solve_rolling_matching(
            cfg=cfg,
            monitoring_indices=[10.0, 10.0, 100.0],
            uav_positions=[(0.0, 0.0), (0.0, 9.0)],
            point_locations=[(0.0, 1.0), (0.0, 8.0), (100.0, 100.0)],
            energy_remaining=[20.0, 20.0],
        )

        self.assertEqual(result.assignment_by_uav, [0, 1])
        self.assertEqual(result.assignment_matrix, [[1, 0, 0], [0, 1, 0]])
        self.assertLess(result.edge_scores[(0, 0)], 10.0)
        self.assertNotIn((0, 2), result.edge_scores)
        self.assertNotIn((1, 2), result.edge_scores)

    def test_small_experiment_exports_algorithm_04_comparison_files(self):
        step_01 = require_module("step_01_config_and_parameters")
        step_06 = require_module("step_06_run_p_pbvi_ruma_experiment")

        cfg = step_01.PPBVIRUMAConfig(
            n_points=6,
            n_uavs=2,
            horizon=3,
            seeds=(2026, 2027),
            pbvi_iterations=2,
            max_route_distance=14.5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = step_06.run_algorithm_04_experiment(cfg=cfg, output_dir=tmp)

            required = [
                "algorithm_04_summary.csv",
                "algorithm_04_seed_metrics.csv",
                "algorithm_04_uav_paths.csv",
                "algorithm_04_matching_matrices.csv",
                "algorithm_04_belief_sequence.csv",
                "algorithm_04_rainfall_high_risk.csv",
                "algorithm_04_risk_points.csv",
                "algorithm_04_result.json",
            ]
            for name in required:
                self.assertTrue(os.path.exists(os.path.join(tmp, name)), name)

            self.assertEqual(result["summary_metrics"]["runs"], 2)
            self.assertIn("recall", result["summary_metrics"])
            self.assertIn("precision", result["summary_metrics"])
            self.assertIn("f1", result["summary_metrics"])
            with open(os.path.join(tmp, "algorithm_04_uav_paths.csv"), newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertGreater(len(rows), 0)
            self.assertIn("monitoring_index", rows[0])
            self.assertIn("assignment_score", rows[0])

            with open(os.path.join(tmp, "algorithm_04_matching_matrices.csv"), newline="", encoding="utf-8") as f:
                matching_rows = list(csv.DictReader(f))
            self.assertGreater(len(matching_rows), 0)
            self.assertIn("matching_matrix", matching_rows[0])
            self.assertIn("assigned_point_idx", matching_rows[0])

            with open(os.path.join(tmp, "algorithm_04_belief_sequence.csv"), newline="", encoding="utf-8") as f:
                belief_rows = list(csv.DictReader(f))
            self.assertEqual(len(belief_rows), cfg.n_points * cfg.horizon * len(cfg.seeds))
            self.assertIn("predicted_belief", belief_rows[0])
            self.assertIn("pre_action_belief", belief_rows[0])
            self.assertIn("posterior_belief", belief_rows[0])
            self.assertIn("monitoring_index", belief_rows[0])


if __name__ == "__main__":
    unittest.main()
