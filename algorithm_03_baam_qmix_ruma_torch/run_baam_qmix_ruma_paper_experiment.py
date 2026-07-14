from __future__ import annotations

"""Paper-scale BAAM-QMIX-RUMA experiment entrypoint.

Run this file when the goal is the final paper experiment rather than a smoke
test or short engineering validation.
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

from run_baam_qmix_ruma_experiment import (
    BAAMDatasetExperimentConfig,
    PACKAGE_ROOT,
    ROOT,
    _parse_seed_tuple,
    run_baam_qmix_ruma_dataset_experiment,
)


PAPER_TRAIN_EPISODES = 10000
PAPER_EVAL_EPISODES = 2000
PAPER_SEEDS = (2026, 2027, 2028, 2029, 2030)


def default_paper_dataset_path() -> Path:
    return PACKAGE_ROOT / "results" / "hpr" / "scenario_v0" / "scenario_dataset_v0.npz"


def default_paper_output_dir() -> Path:
    return (
        ROOT
        / "LT高优先级问题数据结构目标函数结果分析"
        / "04_BAAM真实数据集实验_论文真实版_10000train_2000test_5seed"
    )


def build_paper_config(
    dataset_path: str | Path | None = None,
    train_episodes: int = PAPER_TRAIN_EPISODES,
    eval_episodes: int = PAPER_EVAL_EPISODES,
    seeds: Tuple[int, ...] = PAPER_SEEDS,
) -> BAAMDatasetExperimentConfig:
    return BAAMDatasetExperimentConfig(
        dataset_path=str(dataset_path or default_paper_dataset_path()),
        train_episodes=int(train_episodes),
        eval_episodes=int(eval_episodes),
        seeds=tuple(int(seed) for seed in seeds),
        eval_interval=100,
        hidden_dim=128,
        mixer_hidden_dim=64,
        sequence_length=8,
        epsilon_start=1.0,
        epsilon_end=0.05,
        updates_per_episode=4,
        target_update_interval=100,
        include_static_risk_features=True,
        include_rainfall_trend_features=True,
        include_freshness_feature=True,
        use_reward_curriculum=True,
        curriculum_core_fraction=0.35,
        curriculum_partial_fraction=0.70,
        full_experiment_profile=True,
    )


def run_paper_experiment(
    output_dir: str | Path | None = None,
    cfg: Optional[BAAMDatasetExperimentConfig] = None,
) -> dict:
    paper_cfg = cfg or build_paper_config()
    output = Path(output_dir) if output_dir else default_paper_output_dir()
    return run_baam_qmix_ruma_dataset_experiment(paper_cfg, output)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the paper-scale BAAM-QMIX-RUMA experiment on the scenario_dataset_v0 train/test split."
    )
    parser.add_argument("--dataset-path", default="", help="Override scenario_dataset_v0.npz path.")
    parser.add_argument("--output-dir", default="", help="Override paper experiment output directory.")
    parser.add_argument("--train-episodes", type=int, default=PAPER_TRAIN_EPISODES, help="Default: 10000.")
    parser.add_argument("--eval-episodes", type=int, default=PAPER_EVAL_EPISODES, help="Default: 2000.")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in PAPER_SEEDS), help="Comma-separated seeds.")
    args = parser.parse_args(argv)

    cfg = build_paper_config(
        dataset_path=args.dataset_path or None,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seeds=_parse_seed_tuple(args.seeds),
    )
    output = Path(args.output_dir) if args.output_dir else default_paper_output_dir()

    print("BAAM-QMIX-RUMA paper-scale experiment started.")
    print(f"dataset_path={cfg.dataset_path}")
    print(f"output_dir={output}")
    print(f"train_episodes={cfg.train_episodes}, eval_episodes={cfg.eval_episodes}, seeds={cfg.seed_list}")
    print("This is the paper-scale run, not the 2-episode smoke test or 50-episode validation.")

    started = time.time()
    result = run_paper_experiment(output, cfg)
    elapsed_seconds = time.time() - started
    summary = result["summary_metrics"]
    print("BAAM-QMIX-RUMA paper-scale experiment completed.")
    print(
        f"elapsed_seconds={elapsed_seconds:.1f}, recall={summary['recall']:.3f}, "
        f"precision={summary['precision']:.3f}, f1={summary['f1']:.3f}, "
        f"reward={summary['total_reward']:.2f}"
    )
    print(f"Results saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
