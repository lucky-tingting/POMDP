from __future__ import annotations

import unittest
from pathlib import Path


class PaperExperimentEntrypointTests(unittest.TestCase):
    def test_paper_config_defaults_are_not_smoke_or_short_validation(self):
        from run_baam_qmix_ruma_paper_experiment import build_paper_config, default_paper_output_dir

        cfg = build_paper_config()

        self.assertEqual(cfg.train_episodes, 10000)
        self.assertEqual(cfg.eval_episodes, 2000)
        self.assertEqual(cfg.seeds, (2026, 2027, 2028, 2029, 2030))
        self.assertTrue(cfg.full_experiment_profile)
        self.assertTrue(cfg.use_reward_curriculum)
        self.assertTrue(cfg.include_static_risk_features)
        self.assertTrue(cfg.include_rainfall_trend_features)
        self.assertTrue(cfg.include_freshness_feature)
        self.assertEqual(Path(cfg.dataset_path).name, "scenario_dataset_v0.npz")
        self.assertIn("论文真实版", str(default_paper_output_dir()))


if __name__ == "__main__":
    unittest.main()
