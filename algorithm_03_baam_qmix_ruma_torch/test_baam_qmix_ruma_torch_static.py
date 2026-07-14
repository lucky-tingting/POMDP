from pathlib import Path
import py_compile
import unittest


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "baam_qmix_ruma_torch.py"
README = ROOT / "README_PyTorch版BAAM_QMIX_RUMA.md"
STEP_FILES = [
    "step_01_config_and_protocol.py",
    "step_02_belief_reward.py",
    "step_03_agent_gru.py",
    "step_04_qmix_mixer.py",
    "step_05_action_mask_and_matching.py",
    "step_06_trainer.py",
    "step_07_online_execution.py",
    "step_08_run_template.py",
    "step_10_td_target_and_loss.py",
    "step_11_training_flow.py",
    "step_12_online_execution_flow.py",
    "09_代码与文档一致性审查.md",
]


class BAAMQMixRUMATorchStaticTests(unittest.TestCase):
    def test_source_file_exists_and_compiles(self):
        self.assertTrue(SOURCE.exists(), SOURCE)
        py_compile.compile(str(SOURCE), doraise=True)
        for name in STEP_FILES:
            path = ROOT / name
            self.assertTrue(path.exists(), path)
            if path.suffix == ".py":
                py_compile.compile(str(path), doraise=True)

    def test_pytorch_qmix_gru_components_are_declared(self):
        text = "\n".join(
            [SOURCE.read_text(encoding="utf-8")]
            + [
                (ROOT / name).read_text(encoding="utf-8")
                for name in STEP_FILES
                if (ROOT / name).suffix == ".py" and (ROOT / name).exists()
            ]
        )
        required_snippets = [
            "import torch",
            "import torch.nn as nn",
            "class BAAMQMixRUMAConfig",
            "class GlobalTrainingState",
            "class UAVLocalObservation",
            "class BeliefAwareAgentGRU(nn.Module)",
            "nn.GRUCell",
            "class MonotonicQMIXMixerTorch(nn.Module)",
            "torch.abs",
            "class RollingMatcher",
            "class CandidateSetBuilder",
            "linear_sum_assignment",
            "class BAAMQMIXRUMATrainerTorch",
            "torch.optim.Adam",
            "masked_fill",
            "matched_target_action_indices",
            "compute_matched_qmix_td_target",
            "EpisodeSequenceReplayBuffer",
            "random_feasible_matching",
            "class BAAMQMixRUMATrainingLoop",
            "online_execution_report",
            "build_assignment_matrix",
            "pre_action_belief",
            "assignment_matrix",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, text, snippet)

    def test_sections_6_to_9_requirements_are_reflected_in_code(self):
        step01 = (ROOT / "step_01_config_and_protocol.py").read_text(encoding="utf-8")
        step05 = (ROOT / "step_05_action_mask_and_matching.py").read_text(encoding="utf-8")
        review = (ROOT / "09_代码与文档一致性审查.md").read_text(encoding="utf-8")

        for snippet in [
            "true_risk_state",
            "pre_action_belief",
            "uav_positions",
            "uav_energy",
            "exogenous_state",
            "previous_action",
            "candidate_indices",
            "monitoring_index",
        ]:
            self.assertIn(snippet, step01, snippet)

        for snippet in [
            "max_flight_distance",
            "energy_per_distance",
            "min_safe_energy",
            "remaining_time",
            "safe_matrix",
            "top_k_fallback",
            "virtual action",
            "virtual_action_scores",
        ]:
            self.assertIn(snippet, step05, snippet)

        self.assertIn("不再额外重复扣除飞行成本", review)
        self.assertIn("训练奖励已经包含飞行成本", review)
        self.assertNotIn("- distance_cost", step05)
        step06 = (ROOT / "step_06_trainer.py").read_text(encoding="utf-8")
        step10 = (ROOT / "step_10_td_target_and_loss.py").read_text(encoding="utf-8")
        self.assertIn("compute_matched_qmix_td_target", step06)
        self.assertIn("linear_sum_assignment", step10)
        self.assertNotIn("next_q_values.max(dim=-1).values", step06)

    def test_step_readme_and_consistency_review_match_document_requirements(self):
        readme_text = README.read_text(encoding="utf-8")
        review_text = (ROOT / "09_代码与文档一致性审查.md").read_text(encoding="utf-8")
        for snippet in [
            "step_01_config_and_protocol.py",
            "step_02_belief_reward.py",
            "step_05_action_mask_and_matching.py",
            "step_06_trainer.py",
            "step_10_td_target_and_loss.py",
            "step_11_training_flow.py",
            "step_12_online_execution_flow.py",
            "动作前信念",
            "滚动匹配",
            "Dec-POMDP",
            "虚拟动作",
        ]:
            self.assertIn(snippet, readme_text + review_text, snippet)
        self.assertIn("不一致", review_text)
        self.assertIn("当前已经安装 torch", review_text)

    def test_readme_explains_positioning_and_current_torch_status(self):
        self.assertTrue(README.exists(), README)
        text = README.read_text(encoding="utf-8")
        self.assertIn("BAAM-QMIX-RUMA", text)
        self.assertIn("PyTorch", text)
        self.assertIn("一机一点", text)
        self.assertIn("一点一机", text)
        self.assertIn("当前环境已经安装 CPU 版 PyTorch", text)

    def test_torch_import_and_trainer_can_be_constructed(self):
        import sys

        sys.path.insert(0, str(ROOT))
        import torch
        from baam_qmix_ruma_torch import BAAMQMixRUMAConfig, BAAMQMIXRUMATrainerTorch

        cfg = BAAMQMixRUMAConfig(
            n_points=20,
            n_uavs=3,
            local_obs_dim=8,
            global_state_dim=12,
            hidden_dim=16,
            mixer_hidden_dim=8,
        )
        trainer = BAAMQMIXRUMATrainerTorch(cfg)
        self.assertTrue(torch.__version__)
        self.assertEqual(cfg.n_actions, 21)
        self.assertEqual(type(trainer).__name__, "BAAMQMIXRUMATrainerTorch")


if __name__ == "__main__":
    unittest.main()
