from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Point = Tuple[float, float]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clip_prob(p: float) -> float:
    return float(min(0.999999, max(0.000001, p)))


def entropy(p: float) -> float:
    p = clip_prob(p)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


@dataclass(frozen=True)
class ModelConfig:
    n_points: int = 16
    n_uavs: int = 3
    horizon: int = 24
    n_ml_levels: int = 5
    seed: int = 42

    grid_spacing: float = 5.0
    neighbor_radius: float = 7.2
    max_flight_distance: Optional[float] = None
    energy_capacity: float = 160.0
    energy_per_distance: float = 1.0
    min_safe_energy: float = 16.0

    h_max: float = 45.0
    xi0: float = 0.0
    xi1: float = 3.0
    xi2: float = 1.2
    theta_01: float = 2.45
    zeta0: float = 0.0
    zeta1: float = 1.25
    zeta2: float = 1.15
    zeta3: float = 1.1
    theta_10: float = 1.15

    uav_sensitivity: float = 0.88
    uav_specificity: float = 0.86
    ml_observation_matrix: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
        (0.48, 0.25, 0.14, 0.08, 0.05),
        (0.05, 0.08, 0.14, 0.25, 0.48),
    )

    lambda_cover: float = 18.0
    lambda_miss: float = 8.0
    lambda_fp: float = 3.5
    lambda_cost: float = 0.18
    lambda_info: float = 3.0
    miss_threshold: float = 0.25
    discount: float = 0.95

    def __post_init__(self) -> None:
        if self.n_points <= 0:
            raise ValueError("n_points must be positive")
        if self.n_uavs <= 0:
            raise ValueError("n_uavs must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if len(self.ml_observation_matrix) != 2:
            raise ValueError("ml_observation_matrix must contain low/high rows")
        if any(len(row) != self.n_ml_levels for row in self.ml_observation_matrix):
            raise ValueError("each ML observation row must match n_ml_levels")
        if self.energy_capacity <= 0.0:
            raise ValueError("energy_capacity must be positive")
        if self.energy_per_distance <= 0.0:
            raise ValueError("energy_per_distance must be positive")
        if self.min_safe_energy < 0.0:
            raise ValueError("min_safe_energy must be non-negative")
        if self.min_safe_energy >= self.energy_capacity:
            raise ValueError("min_safe_energy must be smaller than energy_capacity")


@dataclass
class RiskPoint:
    idx: int
    name: str
    location: Point
    flood_susceptibility: float
    drainage_capacity: float
    importance: float
    true_state: int
    belief: float


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def energy_cost(cfg: ModelConfig, travel_distance: float) -> float:
    return cfg.energy_per_distance * travel_distance


def build_neighbors(locations: Sequence[Point], radius: float) -> List[List[int]]:
    neighbors: List[List[int]] = []
    for i, loc_i in enumerate(locations):
        current = []
        for j, loc_j in enumerate(locations):
            if i != j and distance(loc_i, loc_j) <= radius:
                current.append(j)
        neighbors.append(current)
    return neighbors


def mean_field_belief(i: int, beliefs: Sequence[float], neighbors: Sequence[Sequence[int]]) -> float:
    if not neighbors[i]:
        return float(np.mean(beliefs))
    return float(np.mean([beliefs[j] for j in neighbors[i]]))


def transition_probabilities(
    cfg: ModelConfig,
    drive: float,
    neighbor_mean: float,
    drainage_capacity: float,
) -> Tuple[float, float, float, float]:
    h_norm = min(1.5, max(0.0, drive / cfg.h_max))
    p01 = sigmoid(cfg.xi0 + cfg.xi1 * h_norm + cfg.xi2 * neighbor_mean - cfg.theta_01)
    p10 = sigmoid(
        cfg.zeta0
        + cfg.zeta1 * (1.0 - min(1.0, h_norm))
        + cfg.zeta2 * drainage_capacity
        - cfg.zeta3 * neighbor_mean
        - cfg.theta_10
    )
    p01 = min(0.72, max(0.01, p01))
    p10 = min(0.65, max(0.01, p10))
    return 1.0 - p01, p01, p10, 1.0 - p10


def predict_beliefs(
    cfg: ModelConfig,
    points: Sequence[RiskPoint],
    beliefs: Sequence[float],
    rainfall: Sequence[float],
    neighbors: Sequence[Sequence[int]],
) -> List[float]:
    predicted: List[float] = []
    for i, point in enumerate(points):
        mu = mean_field_belief(i, beliefs, neighbors)
        drive = rainfall[i] * point.flood_susceptibility
        _, p01, _, p11 = transition_probabilities(cfg, drive, mu, point.drainage_capacity)
        predicted.append(clip_prob(p01 * (1.0 - beliefs[i]) + p11 * beliefs[i]))
    return predicted


def bayes_update(prior: float, likelihood_high: float, likelihood_low: float) -> float:
    prior = clip_prob(prior)
    numerator = likelihood_high * prior
    denominator = numerator + likelihood_low * (1.0 - prior)
    if denominator <= 0.0:
        return prior
    return clip_prob(numerator / denominator)


def ml_update(cfg: ModelConfig, predicted_beliefs: Sequence[float], ml_levels: Sequence[int]) -> List[float]:
    updated: List[float] = []
    for prior, level in zip(predicted_beliefs, ml_levels):
        likelihood_low = cfg.ml_observation_matrix[0][level]
        likelihood_high = cfg.ml_observation_matrix[1][level]
        updated.append(bayes_update(prior, likelihood_high, likelihood_low))
    return updated


def belief_update(
    cfg: ModelConfig,
    predicted_beliefs: Sequence[float],
    ml_levels: Sequence[int],
    actions: Sequence[int],
    uav_observations: Dict[int, int],
) -> List[float]:
    action_pre_beliefs = ml_update(cfg, predicted_beliefs, ml_levels)
    posterior: List[float] = []
    for i, b_bar in enumerate(action_pre_beliefs):
        if actions[i] == 1 and i in uav_observations:
            z = uav_observations[i]
            if z == 1:
                likelihood_high = cfg.uav_sensitivity
                likelihood_low = 1.0 - cfg.uav_specificity
            else:
                likelihood_high = 1.0 - cfg.uav_sensitivity
                likelihood_low = cfg.uav_specificity
            posterior.append(bayes_update(b_bar, likelihood_high, likelihood_low))
        else:
            posterior.append(clip_prob(b_bar))
    return posterior


def expected_information_gain(belief: float, sensitivity: float, specificity: float) -> float:
    belief = clip_prob(belief)
    p_z1 = sensitivity * belief + (1.0 - specificity) * (1.0 - belief)
    p_z0 = 1.0 - p_z1
    post_z1 = bayes_update(belief, sensitivity, 1.0 - specificity)
    post_z0 = bayes_update(belief, 1.0 - sensitivity, specificity)
    gain = entropy(belief) - (p_z1 * entropy(post_z1) + p_z0 * entropy(post_z0))
    return max(0.0, gain)


def miss_penalty_phi(belief: float, threshold: float) -> float:
    return max(0.0, belief - threshold)


def point_marginal_score(cfg: ModelConfig, belief: float, importance: float) -> float:
    return (
        cfg.lambda_cover * importance * belief
        + cfg.lambda_miss * importance * miss_penalty_phi(belief, cfg.miss_threshold)
        - cfg.lambda_fp * (1.0 - belief)
        + cfg.lambda_info * expected_information_gain(belief, cfg.uav_sensitivity, cfg.uav_specificity)
    )


class RollingMatchingPolicy:
    """One-step rolling maximum-weight matching from the final model."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def select_actions(
        self,
        beliefs: Sequence[float],
        uav_positions: Sequence[Point],
        point_locations: Sequence[Point],
        importance: Sequence[float],
        uav_energy: Optional[Sequence[float]] = None,
    ) -> List[Optional[int]]:
        scores = [point_marginal_score(self.cfg, b, importance[i]) for i, b in enumerate(beliefs)]
        return solve_assignment(
            self.cfg,
            scores=scores,
            uav_positions=uav_positions,
            point_locations=point_locations,
            uav_energy=uav_energy,
        )


class GreedyBeliefPolicy:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def select_actions(
        self,
        beliefs: Sequence[float],
        uav_positions: Sequence[Point],
        point_locations: Sequence[Point],
        importance: Sequence[float],
        uav_energy: Optional[Sequence[float]] = None,
    ) -> List[Optional[int]]:
        scores = [importance[i] * beliefs[i] for i in range(len(beliefs))]
        return solve_assignment(
            self.cfg,
            scores=scores,
            uav_positions=uav_positions,
            point_locations=point_locations,
            uav_energy=uav_energy,
        )


class RandomPolicy:
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def select_actions(
        self,
        beliefs: Sequence[float],
        uav_positions: Sequence[Point],
        point_locations: Sequence[Point],
        importance: Sequence[float],
        uav_energy: Optional[Sequence[float]] = None,
    ) -> List[Optional[int]]:
        available = list(range(len(beliefs)))
        self.rng.shuffle(available)
        selected = available[: min(self.cfg.n_uavs, len(available))]
        assignment: List[Optional[int]] = [None] * self.cfg.n_uavs
        for m, point_idx in enumerate(selected):
            remaining = None if uav_energy is None else uav_energy[m]
            if is_reachable(self.cfg, uav_positions[m], point_locations[point_idx], remaining):
                assignment[m] = point_idx
        return assignment


def is_reachable(
    cfg: ModelConfig,
    uav_pos: Point,
    point_pos: Point,
    energy_remaining: Optional[float] = None,
) -> bool:
    travel_distance = distance(uav_pos, point_pos)
    if cfg.max_flight_distance is not None and travel_distance > cfg.max_flight_distance:
        return False
    if energy_remaining is not None:
        return energy_remaining - energy_cost(cfg, travel_distance) >= cfg.min_safe_energy
    return True


def solve_assignment(
    cfg: ModelConfig,
    scores: Sequence[float],
    uav_positions: Sequence[Point],
    point_locations: Sequence[Point],
    uav_energy: Optional[Sequence[float]] = None,
) -> List[Optional[int]]:
    best_value = 0.0
    best_assignment: List[Optional[int]] = [None] * len(uav_positions)

    def recurse(m: int, used: set[int], current: List[Optional[int]], value: float) -> None:
        nonlocal best_value, best_assignment
        if m == len(uav_positions):
            if value > best_value:
                best_value = value
                best_assignment = current.copy()
            return

        current.append(None)
        recurse(m + 1, used, current, value)
        current.pop()

        for i, point_pos in enumerate(point_locations):
            if i in used:
                continue
            remaining = None if uav_energy is None else uav_energy[m]
            if not is_reachable(cfg, uav_positions[m], point_pos, remaining):
                continue
            net_value = scores[i] - cfg.lambda_cost * distance(uav_positions[m], point_pos)
            if net_value <= 0.0:
                continue
            used.add(i)
            current.append(i)
            recurse(m + 1, used, current, value + net_value)
            current.pop()
            used.remove(i)

    recurse(0, set(), [], 0.0)
    return best_assignment


def make_demo_points(cfg: ModelConfig, rng: np.random.Generator) -> List[RiskPoint]:
    names = [
        "Yuegezhuang", "Liuliqiao", "Fengyiqiao", "Dahongmen",
        "Xizhimen", "Dongzhimen", "Fuxingmen", "Jianguomen",
        "Chaoyangmen", "Fuchengmen", "Guangqumen", "Guanganmen",
        "Youanmen", "Zuoanmen", "Yongdingmen", "Andingmen",
        "Madian", "Sanyuanqiao", "Wukesong", "Shuangjing",
    ]
    points: List[RiskPoint] = []
    side = math.ceil(math.sqrt(cfg.n_points))
    for i in range(cfg.n_points):
        row = i // side
        col = i % side
        loc = (col * cfg.grid_spacing, row * cfg.grid_spacing)
        susceptibility = float(rng.uniform(0.70, 1.45))
        drainage = float(rng.uniform(0.25, 0.95))
        importance = float(rng.uniform(0.80, 1.65))
        initial_high_prob = min(0.55, 0.10 + 0.18 * susceptibility + 0.08 * (1.0 - drainage))
        true_state = int(rng.random() < initial_high_prob)
        belief = float(rng.uniform(0.55, 0.75) if true_state else rng.uniform(0.18, 0.42))
        points.append(
            RiskPoint(
                idx=i,
                name=names[i] if i < len(names) else f"Point-{i + 1}",
                location=loc,
                flood_susceptibility=susceptibility,
                drainage_capacity=drainage,
                importance=importance,
                true_state=true_state,
                belief=belief,
            )
        )
    return points


def rainfall_for_step(cfg: ModelConfig, t: int, points: Sequence[RiskPoint], rng: np.random.Generator) -> List[float]:
    storm_shape = 8.0 + 18.0 * math.exp(-((t - cfg.horizon * 0.45) ** 2) / max(1.0, 0.08 * cfg.horizon ** 2))
    diurnal = 3.0 * (1.0 + math.sin(2.0 * math.pi * t / max(1, cfg.horizon)))
    base = storm_shape + diurnal
    rainfall = []
    for point in points:
        local = base * rng.uniform(0.82, 1.18) + rng.normal(0.0, 1.0)
        rainfall.append(max(0.0, float(local)))
    return rainfall


def sample_ml_levels(cfg: ModelConfig, true_states: Sequence[int], rng: np.random.Generator) -> List[int]:
    levels: List[int] = []
    for state in true_states:
        probs = np.array(cfg.ml_observation_matrix[state], dtype=float)
        probs = probs / probs.sum()
        levels.append(int(rng.choice(cfg.n_ml_levels, p=probs)))
    return levels


def sample_uav_observations(
    cfg: ModelConfig,
    true_states: Sequence[int],
    selected_points: Iterable[int],
    rng: np.random.Generator,
) -> Dict[int, int]:
    observations: Dict[int, int] = {}
    for idx in selected_points:
        if true_states[idx] == 1:
            observations[idx] = int(rng.random() < cfg.uav_sensitivity)
        else:
            observations[idx] = int(rng.random() < (1.0 - cfg.uav_specificity))
    return observations


def update_true_states(
    cfg: ModelConfig,
    points: Sequence[RiskPoint],
    rainfall: Sequence[float],
    beliefs: Sequence[float],
    neighbors: Sequence[Sequence[int]],
    rng: np.random.Generator,
) -> List[int]:
    states: List[int] = []
    for i, point in enumerate(points):
        mu = mean_field_belief(i, beliefs, neighbors)
        drive = rainfall[i] * point.flood_susceptibility
        _, p01, p10, _ = transition_probabilities(cfg, drive, mu, point.drainage_capacity)
        if point.true_state == 0:
            point.true_state = int(rng.random() < p01)
        else:
            point.true_state = int(not (rng.random() < p10))
        states.append(point.true_state)
    return states


def action_vector(n_points: int, assignment: Sequence[Optional[int]]) -> List[int]:
    actions = [0] * n_points
    for idx in assignment:
        if idx is not None:
            actions[idx] = 1
    return actions


def reward_value(
    cfg: ModelConfig,
    action_pre_beliefs: Sequence[float],
    assignment: Sequence[Optional[int]],
    uav_positions: Sequence[Point],
    point_locations: Sequence[Point],
    importance: Sequence[float],
) -> float:
    actions = action_vector(len(action_pre_beliefs), assignment)
    reward = 0.0
    for i, belief in enumerate(action_pre_beliefs):
        if actions[i] == 1:
            reward += cfg.lambda_cover * importance[i] * belief
            reward -= cfg.lambda_fp * (1.0 - belief)
            reward += cfg.lambda_info * expected_information_gain(
                belief, cfg.uav_sensitivity, cfg.uav_specificity
            )
        else:
            reward -= cfg.lambda_miss * importance[i] * miss_penalty_phi(
                belief, cfg.miss_threshold
            )
    for m, idx in enumerate(assignment):
        if idx is not None:
            reward -= cfg.lambda_cost * distance(uav_positions[m], point_locations[idx])
    return float(reward)


def classification_counts(true_states: Sequence[int], actions: Sequence[int]) -> Tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for state, action in zip(true_states, actions):
        if state == 1 and action == 1:
            tp += 1
        elif state == 0 and action == 1:
            fp += 1
        elif state == 1 and action == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def compute_metrics(tp: int, fp: int, fn: int, tn: int, rewards: Sequence[float]) -> Dict[str, float]:
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "recall": round(float(recall), 6),
        "precision": round(float(precision), 6),
        "f1": round(float(f1), 6),
        "total_reward": round(float(sum(rewards)), 6),
        "mean_reward": round(float(np.mean(rewards)) if rewards else 0.0, 6),
        "total_monitored": int(tp + fp),
    }


def build_policy(cfg: ModelConfig, policy_name: str, rng: np.random.Generator):
    if policy_name == "rolling_matching":
        return RollingMatchingPolicy(cfg)
    if policy_name == "greedy_belief":
        return GreedyBeliefPolicy(cfg)
    if policy_name == "random":
        return RandomPolicy(cfg, rng)
    raise ValueError(f"unknown policy_name: {policy_name}")


def run_simulation(cfg: ModelConfig, policy_name: str = "rolling_matching", seed: Optional[int] = None) -> Dict:
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    points = make_demo_points(cfg, rng)
    locations = [p.location for p in points]
    importance = [p.importance for p in points]
    neighbors = build_neighbors(locations, cfg.neighbor_radius)
    beliefs = [p.belief for p in points]
    true_states = [p.true_state for p in points]
    uav_positions: List[Point] = [(0.0, 0.0) for _ in range(cfg.n_uavs)]
    uav_energy: List[float] = [cfg.energy_capacity for _ in range(cfg.n_uavs)]
    policy = build_policy(cfg, policy_name, rng)

    history: List[Dict] = []
    rewards: List[float] = []
    total_tp = total_fp = total_fn = total_tn = 0

    for t in range(cfg.horizon):
        rainfall = rainfall_for_step(cfg, t, points, rng)
        true_states = update_true_states(cfg, points, rainfall, beliefs, neighbors, rng)
        predicted = predict_beliefs(cfg, points, beliefs, rainfall, neighbors)
        ml_levels = sample_ml_levels(cfg, true_states, rng)
        action_pre_beliefs = ml_update(cfg, predicted, ml_levels)

        assignment = policy.select_actions(action_pre_beliefs, uav_positions, locations, importance, uav_energy)
        actions = action_vector(cfg.n_points, assignment)
        selected_points = [idx for idx in assignment if idx is not None]
        hard_observations = sample_uav_observations(cfg, true_states, selected_points, rng)
        posterior = belief_update(cfg, predicted, ml_levels, actions, hard_observations)

        reward = reward_value(cfg, action_pre_beliefs, assignment, uav_positions, locations, importance)
        rewards.append(reward)
        tp, fp, fn, tn = classification_counts(true_states, actions)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        for m, idx in enumerate(assignment):
            if idx is not None:
                travel_distance = distance(uav_positions[m], locations[idx])
                uav_positions[m] = locations[idx]
                uav_energy[m] = max(0.0, uav_energy[m] - energy_cost(cfg, travel_distance))

        history.append(
            {
                "time": t,
                "mean_rainfall": round(float(np.mean(rainfall)), 4),
                "rainfall": [round(float(x), 6) for x in rainfall],
                "high_risk_set": [i for i, state in enumerate(true_states) if state == 1],
                "true_high_count": int(sum(true_states)),
                "selected_points": selected_points,
                "assignment": assignment,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "reward": round(reward, 6),
                "mean_pre_action_belief": round(float(np.mean(action_pre_beliefs)), 6),
                "mean_posterior_belief": round(float(np.mean(posterior)), 6),
                "uav_energy_remaining": [round(float(x), 6) for x in uav_energy],
            }
        )

        beliefs = posterior
        for i, point in enumerate(points):
            point.belief = posterior[i]

    metrics = compute_metrics(total_tp, total_fp, total_fn, total_tn, rewards)
    return {
        "config": {
            "n_points": cfg.n_points,
            "n_uavs": cfg.n_uavs,
            "horizon": cfg.horizon,
            "seed": cfg.seed if seed is None else seed,
            "policy": policy_name,
            "max_flight_distance": cfg.max_flight_distance,
            "energy_capacity": cfg.energy_capacity,
            "energy_per_distance": cfg.energy_per_distance,
            "min_safe_energy": cfg.min_safe_energy,
        },
        "metrics": metrics,
        "history": history,
        "points": [
            {
                "idx": p.idx,
                "name": p.name,
                "x": p.location[0],
                "y": p.location[1],
                "importance": round(p.importance, 4),
                "flood_susceptibility": round(p.flood_susceptibility, 4),
                "drainage_capacity": round(p.drainage_capacity, 4),
            }
            for p in points
        ],
    }


def write_history_csv(path: str, result: Dict) -> None:
    fieldnames = [
        "time",
        "mean_rainfall",
        "true_high_count",
        "selected_points",
        "assignment",
        "tp",
        "fp",
        "fn",
        "reward",
        "mean_pre_action_belief",
        "mean_posterior_belief",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["history"]:
            out = dict(row)
            out["selected_points"] = json.dumps(out["selected_points"], ensure_ascii=False)
            out["assignment"] = json.dumps(out["assignment"], ensure_ascii=False)
            writer.writerow(out)


def write_summary_csv(path: str, rows: Sequence[Dict]) -> None:
    fieldnames = ["policy", "runs", "recall", "precision", "f1", "total_reward", "total_monitored"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def average_metric_rows(policy_name: str, results: Sequence[Dict]) -> Dict:
    keys = ["recall", "precision", "f1", "total_reward", "total_monitored"]
    row = {"policy": policy_name, "runs": len(results)}
    for key in keys:
        row[key] = round(float(np.mean([r["metrics"][key] for r in results])), 6)
    return row


def plot_policy_comparison(path: str, summary_rows: Sequence[Dict]) -> None:
    import matplotlib.pyplot as plt

    policies = [row["policy"] for row in summary_rows]
    x = np.arange(len(policies))
    width = 0.24

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x - width, [row["recall"] for row in summary_rows], width, label="Recall")
    ax1.bar(x, [row["precision"] for row in summary_rows], width, label="Precision")
    ax1.bar(x + width, [row["f1"] for row in summary_rows], width, label="F1")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, rotation=15)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left")
    ax1.set_title("MF-MLe-MUAV-UWR-POMDP policy comparison")

    ax2 = ax1.twinx()
    ax2.plot(x, [row["total_reward"] for row in summary_rows], "ko--", label="Total reward")
    ax2.set_ylabel("Mean total reward")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(path: str, summary_rows: Sequence[Dict], main_result: Dict) -> None:
    best = max(summary_rows, key=lambda row: row["f1"])
    lines = [
        "# 新版本建模仿真实验报告",
        "",
        "模型：MF-MLe-MUAV-UWR-POMDP",
        "",
        "求解策略：采用滚动时域最大权匹配近似。每个时间步先用机器学习软观测形成动作前信念，再用边际收益减飞行成本作为边权，求解满足“一机一点、一点一机”的二元匹配。",
        "",
        "## 策略平均结果",
        "",
        "| 策略 | 运行次数 | Recall | Precision | F1 | 总奖励 | 总监测次数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['policy']} | {row['runs']} | {row['recall']:.4f} | "
            f"{row['precision']:.4f} | {row['f1']:.4f} | {row['total_reward']:.2f} | "
            f"{row['total_monitored']:.1f} |"
        )
    lines.extend(
        [
            "",
            f"最佳 F1 策略：`{best['policy']}`。",
            "",
            "## rolling_matching 单次轨迹摘要",
            "",
            "| t | 平均降雨 | 真实高风险数 | 选中点 | TP | FP | FN | 奖励 | 平均动作前信念 |",
            "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in main_result["history"]:
        lines.append(
            f"| {row['time']} | {row['mean_rainfall']:.2f} | {row['true_high_count']} | "
            f"{row['selected_points']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['reward']:.2f} | {row['mean_pre_action_belief']:.3f} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_experiment(output_dir: Optional[str] = None) -> Dict:
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    base_cfg = ModelConfig(n_points=16, n_uavs=3, horizon=24, seed=2026)
    policies = ["rolling_matching", "greedy_belief", "random"]
    seeds = [2026, 2027, 2028, 2029, 2030]
    all_results: Dict[str, List[Dict]] = {}
    summary_rows: List[Dict] = []

    for policy in policies:
        policy_results = [run_simulation(base_cfg, policy_name=policy, seed=seed) for seed in seeds]
        all_results[policy] = policy_results
        summary_rows.append(average_metric_rows(policy, policy_results))

    main_result = all_results["rolling_matching"][0]
    write_history_csv(os.path.join(output_dir, "新模型rolling_matching单次轨迹.csv"), main_result)
    write_summary_csv(os.path.join(output_dir, "新模型策略对比汇总.csv"), summary_rows)
    plot_policy_comparison(os.path.join(output_dir, "新模型策略对比.png"), summary_rows)
    write_report(os.path.join(output_dir, "新模型实验报告.md"), summary_rows, main_result)

    with open(os.path.join(output_dir, "新模型rolling_matching单次结果.json"), "w", encoding="utf-8") as f:
        json.dump(main_result, f, ensure_ascii=False, indent=2)

    return {"summary": summary_rows, "main_result": main_result}


def main() -> None:
    output_dir = os.path.dirname(os.path.abspath(__file__))
    result = run_experiment(output_dir)
    print("MF-MLe-MUAV-UWR-POMDP experiment completed.")
    for row in result["summary"]:
        print(
            f"{row['policy']}: recall={row['recall']:.3f}, "
            f"precision={row['precision']:.3f}, f1={row['f1']:.3f}, "
            f"reward={row['total_reward']:.2f}"
        )
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
