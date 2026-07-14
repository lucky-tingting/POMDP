from __future__ import annotations

import csv
import itertools
import json
import math
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
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


def bayes_update(prior: float, likelihood_high: float, likelihood_low: float) -> float:
    prior = clip_prob(prior)
    numerator = likelihood_high * prior
    denominator = numerator + likelihood_low * (1.0 - prior)
    if denominator <= 0.0:
        return prior
    return clip_prob(numerator / denominator)


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def energy_cost(cfg: "PaperQMIXConfig", travel_distance: float) -> float:
    return cfg.energy_per_distance * travel_distance


def expected_information_gain(belief: float, sensitivity: float, specificity: float) -> float:
    belief = clip_prob(belief)
    p_z1 = sensitivity * belief + (1.0 - specificity) * (1.0 - belief)
    p_z0 = 1.0 - p_z1
    post_z1 = bayes_update(belief, sensitivity, 1.0 - specificity)
    post_z0 = bayes_update(belief, 1.0 - sensitivity, specificity)
    return max(0.0, entropy(belief) - p_z1 * entropy(post_z1) - p_z0 * entropy(post_z0))


@dataclass(frozen=True)
class PaperQMIXConfig:
    """Paper-level MF-MLe-MUAV-UWR-POMDP + QMIX-GRU configuration."""

    n_points: int = 20
    n_uavs: int = 3
    horizon: int = 16
    episodes: int = 24
    seed: int = 2026

    grid_spacing: float = 5.0
    neighbor_radius: float = 7.2
    route_length: int = 1
    top_k_routes: int = 8
    max_route_actions: int = 36
    max_route_distance: float = 22.0
    energy_capacity: float = 160.0
    energy_per_distance: float = 1.0
    min_safe_energy: float = 16.0

    h_max: float = 45.0
    rainfall_base: float = 8.0
    rainfall_peak: float = 18.0
    rainfall_wave: float = 3.0

    n_ml_levels: int = 5
    ml_observation_matrix: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
        (0.48, 0.25, 0.14, 0.08, 0.05),
        (0.05, 0.08, 0.14, 0.25, 0.48),
    )
    uav_sensitivity: float = 0.88
    uav_specificity: float = 0.86

    xi0: float = 0.0
    xi1: float = 3.0
    xi2: float = 1.2
    theta_01: float = 2.45
    zeta0: float = 0.0
    zeta1: float = 1.25
    zeta2: float = 1.15
    zeta3: float = 1.1
    theta_10: float = 1.15

    lambda_cover: float = 18.0
    lambda_miss: float = 8.0
    lambda_fp: float = 3.5
    lambda_cost: float = 0.18
    lambda_info: float = 3.0
    miss_threshold: float = 0.25

    hidden_dim: int = 48
    gamma: float = 0.95
    learning_rate: float = 0.015
    epsilon_start: float = 0.50
    epsilon_end: float = 0.05
    replay_capacity: int = 600
    batch_size: int = 16
    target_update_interval: int = 4

    @property
    def route_feature_dim(self) -> int:
        return 4

    @property
    def base_obs_dim(self) -> int:
        return 10

    @property
    def observation_dim(self) -> int:
        return self.base_obs_dim + self.max_route_actions * self.route_feature_dim

    @property
    def global_state_dim(self) -> int:
        return 12

    def __post_init__(self) -> None:
        if self.n_points <= 0:
            raise ValueError("n_points must be positive")
        if self.n_uavs <= 0:
            raise ValueError("n_uavs must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.route_length <= 0:
            raise ValueError("route_length must be positive")
        if self.max_route_actions < 2:
            raise ValueError("max_route_actions must include wait and at least one route")
        if self.energy_capacity <= 0.0:
            raise ValueError("energy_capacity must be positive")
        if self.energy_per_distance <= 0.0:
            raise ValueError("energy_per_distance must be positive")
        if self.min_safe_energy < 0.0:
            raise ValueError("min_safe_energy must be non-negative")
        if self.min_safe_energy >= self.energy_capacity:
            raise ValueError("min_safe_energy must be smaller than energy_capacity")
        if len(self.ml_observation_matrix) != 2:
            raise ValueError("ml_observation_matrix must have low/high rows")
        for row in self.ml_observation_matrix:
            if len(row) != self.n_ml_levels:
                raise ValueError("each ML observation row must match n_ml_levels")


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
    initial_belief: float
    initial_state: int


@dataclass(frozen=True)
class RouteCandidate:
    action_id: int
    route: Tuple[int, ...]
    distance: float
    score: float
    features: Tuple[float, float, float, float]


@dataclass
class Transition:
    obs: List[np.ndarray]
    state: np.ndarray
    hidden: List[np.ndarray]
    actions: List[int]
    reward: float
    next_obs: List[np.ndarray]
    next_state: np.ndarray
    next_hidden: List[np.ndarray]
    masks: List[np.ndarray]
    next_masks: List[np.ndarray]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.items: List[Transition] = []

    def add(self, item: Transition) -> None:
        if len(self.items) >= self.capacity:
            self.items.pop(0)
        self.items.append(item)

    def sample(self, batch_size: int) -> List[Transition]:
        if not self.items:
            return []
        count = min(batch_size, len(self.items))
        indices = self.rng.choice(len(self.items), size=count, replace=False)
        return [self.items[int(i)] for i in indices]

    def __len__(self) -> int:
        return len(self.items)


def metric_dict(tp: int, fp: int, fn: int, rewards: Sequence[float]) -> Dict[str, float]:
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "recall": round(float(recall), 6),
        "precision": round(float(precision), 6),
        "f1": round(float(f1), 6),
        "total_reward": round(float(sum(rewards)), 6),
        "mean_reward": round(float(np.mean(rewards)) if rewards else 0.0, 6),
        "total_monitored": int(tp + fp),
    }


def metric_summary(results: Sequence[Dict]) -> Dict[str, float]:
    keys = ["recall", "precision", "f1", "total_reward", "mean_reward", "total_monitored"]
    summary: Dict[str, float] = {"runs": len(results)}
    for key in keys:
        values = [result["final_metrics"][key] for result in results]
        summary[key] = round(float(np.mean(values)), 6)
        summary[f"{key}_std"] = round(float(np.std(values, ddof=0)), 6)
    return summary


class PaperPOMDPEnv:
    """MF-MLe-MUAV-UWR-POMDP environment with route-set actions."""

    def __init__(self, cfg: PaperQMIXConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.points: List[RiskPoint] = []
        self.neighbors: List[List[int]] = []
        self.uav_positions: List[Point] = []
        self.uav_energy: List[float] = []
        self.time = 0
        self.current_rainfall: List[float] = []
        self.current_ml_levels: List[int] = []
        self.pre_action_beliefs: List[float] = []
        self._decision_ready = False
        self.reset()

    def reset(self, seed_offset: int = 0):
        self.rng = np.random.default_rng(self.cfg.seed + seed_offset * 1009)
        self.points = self._make_points()
        self.neighbors = self._build_neighbors()
        self.uav_positions = [(0.0, 0.0) for _ in range(self.cfg.n_uavs)]
        self.uav_energy = [self.cfg.energy_capacity for _ in range(self.cfg.n_uavs)]
        self.time = 0
        self._prepare_decision()
        return self.observation_bundle()

    def _make_points(self) -> List[RiskPoint]:
        names = [
            "Yuegezhuang", "Liuliqiao", "Fengyiqiao", "Dahongmen",
            "Xizhimen", "Dongzhimen", "Fuxingmen", "Jianguomen",
            "Chaoyangmen", "Fuchengmen", "Guangqumen", "Guanganmen",
            "Youanmen", "Zuoanmen", "Yongdingmen", "Andingmen",
            "Madian", "Sanyuanqiao", "Wukesong", "Shuangjing",
        ]
        side = math.ceil(math.sqrt(self.cfg.n_points))
        points: List[RiskPoint] = []
        for i in range(self.cfg.n_points):
            row = i // side
            col = i % side
            susceptibility = float(self.rng.uniform(0.70, 1.45))
            drainage = float(self.rng.uniform(0.25, 0.95))
            importance = float(self.rng.uniform(0.80, 1.65))
            init_prob = min(0.55, 0.10 + 0.18 * susceptibility + 0.08 * (1.0 - drainage))
            state = int(self.rng.random() < init_prob)
            belief = float(self.rng.uniform(0.55, 0.75) if state else self.rng.uniform(0.18, 0.42))
            points.append(
                RiskPoint(
                    idx=i,
                    name=names[i] if i < len(names) else f"Point-{i + 1}",
                    location=(col * self.cfg.grid_spacing, row * self.cfg.grid_spacing),
                    flood_susceptibility=susceptibility,
                    drainage_capacity=drainage,
                    importance=importance,
                    true_state=state,
                    belief=belief,
                    initial_belief=belief,
                    initial_state=state,
                )
            )
        return points

    def _build_neighbors(self) -> List[List[int]]:
        neighbors: List[List[int]] = []
        for i, p_i in enumerate(self.points):
            current = []
            for j, p_j in enumerate(self.points):
                if i != j and distance(p_i.location, p_j.location) <= self.cfg.neighbor_radius:
                    current.append(j)
            neighbors.append(current)
        return neighbors

    def _rainfall(self) -> List[float]:
        storm = self.cfg.rainfall_base + self.cfg.rainfall_peak * math.exp(
            -((self.time - self.cfg.horizon * 0.45) ** 2) / max(1.0, 0.08 * self.cfg.horizon ** 2)
        )
        wave = self.cfg.rainfall_wave * (1.0 + math.sin(2.0 * math.pi * self.time / max(1, self.cfg.horizon)))
        base = storm + wave
        rainfall = []
        for point in self.points:
            local = base * self.rng.uniform(0.82, 1.18) + self.rng.normal(0.0, 1.0)
            rainfall.append(max(0.0, float(local)))
        return rainfall

    def _mean_field_belief(self, idx: int, beliefs: Sequence[float]) -> float:
        if not self.neighbors[idx]:
            return float(np.mean(beliefs))
        return float(np.mean([beliefs[j] for j in self.neighbors[idx]]))

    def _mean_field_state(self, idx: int) -> float:
        if not self.neighbors[idx]:
            return float(np.mean([p.true_state for p in self.points]))
        return float(np.mean([self.points[j].true_state for j in self.neighbors[idx]]))

    def _transition_probabilities(
        self,
        idx: int,
        rainfall: float,
        neighbor_mean: float,
    ) -> Tuple[float, float]:
        point = self.points[idx]
        h_norm = min(1.5, max(0.0, rainfall * point.flood_susceptibility / self.cfg.h_max))
        p01 = sigmoid(self.cfg.xi0 + self.cfg.xi1 * h_norm + self.cfg.xi2 * neighbor_mean - self.cfg.theta_01)
        p10 = sigmoid(
            self.cfg.zeta0
            + self.cfg.zeta1 * (1.0 - min(1.0, h_norm))
            + self.cfg.zeta2 * point.drainage_capacity
            - self.cfg.zeta3 * neighbor_mean
            - self.cfg.theta_10
        )
        return min(0.72, max(0.01, p01)), min(0.65, max(0.01, p10))

    def _update_true_states(self, rainfall: Sequence[float]) -> None:
        next_states = []
        for i, point in enumerate(self.points):
            p01, p10 = self._transition_probabilities(i, rainfall[i], self._mean_field_state(i))
            if point.true_state == 0:
                next_states.append(int(self.rng.random() < p01))
            else:
                next_states.append(int(not (self.rng.random() < p10)))
        for point, state in zip(self.points, next_states):
            point.true_state = state

    def _predict_beliefs(self, rainfall: Sequence[float]) -> List[float]:
        prev = [p.belief for p in self.points]
        predicted = []
        for i, point in enumerate(self.points):
            mu = self._mean_field_belief(i, prev)
            p01, p10 = self._transition_probabilities(i, rainfall[i], mu)
            predicted.append(clip_prob(p01 * (1.0 - point.belief) + (1.0 - p10) * point.belief))
        return predicted

    def _sample_ml_levels(self) -> List[int]:
        levels = []
        for point in self.points:
            probs = np.asarray(self.cfg.ml_observation_matrix[point.true_state], dtype=float)
            probs = probs / probs.sum()
            levels.append(int(self.rng.choice(self.cfg.n_ml_levels, p=probs)))
        return levels

    def _ml_update(self, predicted: Sequence[float], levels: Sequence[int]) -> List[float]:
        updated = []
        for prior, level in zip(predicted, levels):
            updated.append(
                bayes_update(
                    prior,
                    self.cfg.ml_observation_matrix[1][level],
                    self.cfg.ml_observation_matrix[0][level],
                )
            )
        return updated

    def _prepare_decision(self) -> None:
        self.current_rainfall = self._rainfall()
        self._update_true_states(self.current_rainfall)
        predicted = self._predict_beliefs(self.current_rainfall)
        self.current_ml_levels = self._sample_ml_levels()
        self.pre_action_beliefs = self._ml_update(predicted, self.current_ml_levels)
        self._decision_ready = True

    def point_score(self, idx: int) -> float:
        b = self.pre_action_beliefs[idx]
        point = self.points[idx]
        return (
            self.cfg.lambda_cover * point.importance * b
            + self.cfg.lambda_miss * point.importance * max(0.0, b - self.cfg.miss_threshold)
            - self.cfg.lambda_fp * (1.0 - b)
            + self.cfg.lambda_info * expected_information_gain(b, self.cfg.uav_sensitivity, self.cfg.uav_specificity)
        )

    def route_distance(self, start: Point, route: Sequence[int]) -> float:
        total = 0.0
        current = start
        for idx in route:
            nxt = self.points[idx].location
            total += distance(current, nxt)
            current = nxt
        return float(total)

    def route_candidates_for_uav(self, uav_idx: int) -> List[RouteCandidate]:
        if not self._decision_ready:
            self._prepare_decision()
        scores = [(i, self.point_score(i)) for i in range(self.cfg.n_points)]
        scores.sort(key=lambda item: item[1], reverse=True)
        candidate_points = [i for i, _ in scores[: self.cfg.top_k_routes]]
        raw: List[Tuple[Tuple[int, ...], float, float]] = [(tuple(), 0.0, 0.0)]

        energy_feasible_distance = max(
            0.0,
            (self.uav_energy[uav_idx] - self.cfg.min_safe_energy) / self.cfg.energy_per_distance,
        )
        max_distance = min(self.cfg.max_route_distance, energy_feasible_distance)
        for length in range(1, self.cfg.route_length + 1):
            for route in itertools.permutations(candidate_points, length):
                dist = self.route_distance(self.uav_positions[uav_idx], route)
                if dist > max_distance:
                    continue
                score = sum((self.cfg.gamma ** k) * self.point_score(idx) for k, idx in enumerate(route))
                score -= self.cfg.lambda_cost * dist
                if score > 0.0:
                    raw.append((route, dist, float(score)))

        raw = sorted(raw, key=lambda item: item[2], reverse=True)[: self.cfg.max_route_actions]
        candidates: List[RouteCandidate] = []
        normalizer = max(1.0, max(abs(item[2]) for item in raw))
        for action_id, (route, dist, score) in enumerate(raw):
            max_belief = max([self.pre_action_beliefs[i] for i in route], default=0.0)
            features = (
                float(score / normalizer),
                float(dist / max(1.0, self.cfg.max_route_distance)),
                float(len(route) / max(1, self.cfg.route_length)),
                float(max_belief),
            )
            candidates.append(RouteCandidate(action_id, route, dist, score, features))
        return candidates

    def _base_observation(self, uav_idx: int) -> List[float]:
        beliefs = np.asarray(self.pre_action_beliefs, dtype=float)
        pos = self.uav_positions[uav_idx]
        map_scale = max(1.0, self.cfg.grid_spacing * math.ceil(math.sqrt(self.cfg.n_points)))
        return [
            float(np.mean(beliefs)),
            float(np.max(beliefs)),
            float(np.std(beliefs)),
            float(np.mean(self.current_rainfall) / max(1.0, self.cfg.h_max)),
            float(np.max(self.current_rainfall) / max(1.0, self.cfg.h_max)),
            float(pos[0] / map_scale),
            float(pos[1] / map_scale),
            float(self.uav_energy[uav_idx] / max(1.0, self.cfg.energy_capacity)),
            float(self.time / max(1, self.cfg.horizon)),
            float(np.mean([p.importance for p in self.points])),
        ]

    def local_observation(self, uav_idx: int, candidates: Sequence[RouteCandidate]) -> np.ndarray:
        values = self._base_observation(uav_idx)
        for action_id in range(self.cfg.max_route_actions):
            if action_id < len(candidates):
                values.extend(candidates[action_id].features)
            else:
                values.extend([0.0] * self.cfg.route_feature_dim)
        return np.asarray(values, dtype=float)

    def global_state(self) -> np.ndarray:
        beliefs = np.asarray(self.pre_action_beliefs, dtype=float)
        true_states = np.asarray([p.true_state for p in self.points], dtype=float)
        rainfall = np.asarray(self.current_rainfall, dtype=float)
        return np.asarray(
            [
                float(np.mean(beliefs)),
                float(np.max(beliefs)),
                float(np.std(beliefs)),
                float(np.mean(true_states)),
                float(np.mean(rainfall) / max(1.0, self.cfg.h_max)),
                float(np.max(rainfall) / max(1.0, self.cfg.h_max)),
                float(np.mean([p.importance for p in self.points])),
                float(np.mean([p.flood_susceptibility for p in self.points])),
                float(np.mean([p.drainage_capacity for p in self.points])),
                float(np.mean(self.uav_energy) / max(1.0, self.cfg.energy_capacity)),
                float(min(self.uav_energy) / max(1.0, self.cfg.energy_capacity)),
                float(self.time / max(1, self.cfg.horizon)),
            ],
            dtype=float,
        )

    def observation_bundle(self):
        candidates = [self.route_candidates_for_uav(m) for m in range(self.cfg.n_uavs)]
        observations = [self.local_observation(m, candidates[m]) for m in range(self.cfg.n_uavs)]
        masks = []
        for cands in candidates:
            mask = np.zeros(self.cfg.max_route_actions, dtype=bool)
            mask[: len(cands)] = True
            masks.append(mask)
        return observations, self.global_state(), candidates, masks

    def _resolve_actions(
        self,
        action_ids: Sequence[int],
        candidates: Sequence[Sequence[RouteCandidate]],
    ) -> Tuple[List[int], List[RouteCandidate]]:
        used_points = set()
        resolved_ids: List[int] = []
        resolved_candidates: List[RouteCandidate] = []
        for m, action_id in enumerate(action_ids):
            cands = list(candidates[m])
            chosen: Optional[RouteCandidate] = None
            preferred = int(action_id) if 0 <= int(action_id) < len(cands) else 0
            ordered = [preferred] + [i for i in range(len(cands)) if i != preferred]
            for idx in ordered:
                cand = cands[idx]
                if any(point in used_points for point in cand.route):
                    continue
                chosen = cand
                break
            if chosen is None:
                chosen = cands[0]
            resolved_ids.append(chosen.action_id)
            resolved_candidates.append(chosen)
            used_points.update(chosen.route)
        return resolved_ids, resolved_candidates

    def _hard_update(self, selected_points: Iterable[int]) -> Dict[int, int]:
        observations: Dict[int, int] = {}
        for idx in selected_points:
            if self.points[idx].true_state == 1:
                observations[idx] = int(self.rng.random() < self.cfg.uav_sensitivity)
            else:
                observations[idx] = int(self.rng.random() < (1.0 - self.cfg.uav_specificity))
        return observations

    def _posterior_after_hard_observation(self, hard_obs: Dict[int, int]) -> List[float]:
        posterior = []
        for i, b in enumerate(self.pre_action_beliefs):
            if i not in hard_obs:
                posterior.append(clip_prob(b))
            elif hard_obs[i] == 1:
                posterior.append(bayes_update(b, self.cfg.uav_sensitivity, 1.0 - self.cfg.uav_specificity))
            else:
                posterior.append(bayes_update(b, 1.0 - self.cfg.uav_sensitivity, self.cfg.uav_specificity))
        return posterior

    def _reward(self, selected_points: Sequence[int], routes: Sequence[RouteCandidate]) -> float:
        selected = set(selected_points)
        reward = 0.0
        for i, belief in enumerate(self.pre_action_beliefs):
            point = self.points[i]
            if i in selected:
                reward += self.cfg.lambda_cover * point.importance * belief
                reward -= self.cfg.lambda_fp * (1.0 - belief)
                reward += self.cfg.lambda_info * expected_information_gain(
                    belief, self.cfg.uav_sensitivity, self.cfg.uav_specificity
                )
            else:
                reward -= self.cfg.lambda_miss * point.importance * max(0.0, belief - self.cfg.miss_threshold)
        for cand in routes:
            reward -= self.cfg.lambda_cost * cand.distance
        return float(reward)

    def step(self, action_ids: Sequence[int]):
        if len(action_ids) != self.cfg.n_uavs:
            raise ValueError("one route action is required for each UAV")
        obs, state, candidates, masks = self.observation_bundle()
        resolved_ids, chosen = self._resolve_actions(action_ids, candidates)
        selected_points: List[int] = []
        for cand in chosen:
            for idx in cand.route:
                if idx not in selected_points:
                    selected_points.append(idx)

        hard_obs = self._hard_update(selected_points)
        posterior = self._posterior_after_hard_observation(hard_obs)
        for point, b in zip(self.points, posterior):
            point.belief = b

        reward = self._reward(selected_points, chosen)
        route_rows = []
        for m, cand in enumerate(chosen):
            start = self.uav_positions[m]
            if cand.route:
                self.uav_positions[m] = self.points[cand.route[-1]].location
            self.uav_energy[m] = max(0.0, self.uav_energy[m] - energy_cost(self.cfg, cand.distance))
            route_rows.append(
                {
                    "time": self.time,
                    "uav": m,
                    "action_id": resolved_ids[m],
                    "route": list(cand.route),
                    "route_names": [self.points[i].name for i in cand.route],
                    "start_x": round(start[0], 6),
                    "start_y": round(start[1], 6),
                    "end_x": round(self.uav_positions[m][0], 6),
                    "end_y": round(self.uav_positions[m][1], 6),
                    "distance": round(cand.distance, 6),
                    "energy_remaining": round(self.uav_energy[m], 6),
                }
            )

        true_states = [p.true_state for p in self.points]
        high_risk_set = [i for i, state_i in enumerate(true_states) if state_i == 1]
        tp = sum(1 for i in selected_points if true_states[i] == 1)
        fp = sum(1 for i in selected_points if true_states[i] == 0)
        fn = sum(1 for i in high_risk_set if i not in selected_points)
        info = {
            "time": self.time,
            "chosen_action_ids": resolved_ids,
            "routes": [list(c.route) for c in chosen],
            "route_rows": route_rows,
            "selected_points": selected_points,
            "selected_names": [self.points[i].name for i in selected_points],
            "high_risk_set": high_risk_set,
            "rainfall": list(self.current_rainfall),
            "mean_rainfall": round(float(np.mean(self.current_rainfall)), 6),
            "ml_levels": list(self.current_ml_levels),
            "pre_action_beliefs": list(self.pre_action_beliefs),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "reward": round(float(reward), 6),
        }

        self.time += 1
        done = self.time >= self.cfg.horizon
        if not done:
            self._prepare_decision()
            next_obs, next_state, next_candidates, next_masks = self.observation_bundle()
        else:
            next_obs = [np.zeros(self.cfg.observation_dim, dtype=float) for _ in range(self.cfg.n_uavs)]
            next_state = np.zeros(self.cfg.global_state_dim, dtype=float)
            next_candidates = [[] for _ in range(self.cfg.n_uavs)]
            next_masks = [np.zeros(self.cfg.max_route_actions, dtype=bool) for _ in range(self.cfg.n_uavs)]
        return next_obs, next_state, next_candidates, next_masks, reward, done, info, obs, state, masks


class DeepGRUAgent:
    """Shared recurrent decentralized Q-network.

    The implementation is NumPy-based so this deliverable runs without a
    heavyweight deep-learning dependency. It keeps the paper-level QMIX-GRU
    structure: GRU hidden state, one local Q-vector per UAV, target network,
    and decentralized masked route-action selection.
    """

    def __init__(self, cfg: PaperQMIXConfig, obs_dim: int):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.rng = np.random.default_rng(cfg.seed + 101)
        in_dim = obs_dim + cfg.hidden_dim
        scale = 1.0 / math.sqrt(max(1, in_dim))
        self.W_z = self.rng.normal(0.0, scale, size=(in_dim, cfg.hidden_dim))
        self.b_z = np.zeros(cfg.hidden_dim, dtype=float)
        self.W_r = self.rng.normal(0.0, scale, size=(in_dim, cfg.hidden_dim))
        self.b_r = np.zeros(cfg.hidden_dim, dtype=float)
        self.W_h = self.rng.normal(0.0, scale, size=(in_dim, cfg.hidden_dim))
        self.b_h = np.zeros(cfg.hidden_dim, dtype=float)
        self.W_o = self.rng.normal(0.0, 0.08, size=(cfg.hidden_dim, cfg.max_route_actions))
        self.b_o = np.zeros(cfg.max_route_actions, dtype=float)

    def initial_hidden(self) -> np.ndarray:
        return np.zeros(self.cfg.hidden_dim, dtype=float)

    def _forward_cache(self, obs: np.ndarray, hidden: np.ndarray):
        obs = np.asarray(obs, dtype=float)
        hidden = np.asarray(hidden, dtype=float)
        joined = np.concatenate([obs, hidden])
        z = 1.0 / (1.0 + np.exp(-(joined @ self.W_z + self.b_z)))
        r = 1.0 / (1.0 + np.exp(-(joined @ self.W_r + self.b_r)))
        candidate_joined = np.concatenate([obs, r * hidden])
        h_tilde = np.tanh(candidate_joined @ self.W_h + self.b_h)
        next_hidden = (1.0 - z) * hidden + z * h_tilde
        q_values = next_hidden @ self.W_o + self.b_o
        cache = {
            "obs": obs,
            "hidden": hidden,
            "joined": joined,
            "z": z,
            "r": r,
            "candidate_joined": candidate_joined,
            "h_tilde": h_tilde,
            "next_hidden": next_hidden,
        }
        return q_values.astype(float), next_hidden.astype(float), cache

    def forward(self, obs: np.ndarray, hidden: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q_values, next_hidden, _cache = self._forward_cache(obs, hidden)
        return q_values.astype(float), next_hidden.astype(float)

    def select_actions(
        self,
        observations: Sequence[np.ndarray],
        hidden_states: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        epsilon: float,
    ) -> Tuple[List[int], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        actions: List[int] = []
        next_hidden: List[np.ndarray] = []
        q_values_by_agent: List[np.ndarray] = []
        chosen_hidden: List[np.ndarray] = []
        for obs, h, mask in zip(observations, hidden_states, masks):
            q_values, h_next = self.forward(obs, h)
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                action = 0
            elif self.rng.random() < epsilon:
                action = int(self.rng.choice(valid))
            else:
                masked_q = np.full(self.cfg.max_route_actions, -1.0e9, dtype=float)
                masked_q[valid] = q_values[valid]
                action = int(np.argmax(masked_q))
            actions.append(action)
            next_hidden.append(h_next)
            q_values_by_agent.append(q_values)
            chosen_hidden.append(h_next)
        return actions, next_hidden, q_values_by_agent, chosen_hidden

    def update_recurrent_step(
        self,
        observations: Sequence[np.ndarray],
        previous_hidden_states: Sequence[np.ndarray],
        actions: Sequence[int],
        td_error: float,
        mixer_weights: Sequence[float],
        learning_rate: float,
    ) -> None:
        clipped = float(np.clip(td_error, -8.0, 8.0))
        for m, (obs, h_prev, action) in enumerate(zip(observations, previous_hidden_states, actions)):
            _q_values, h_next, cache = self._forward_cache(obs, h_prev)
            scale = learning_rate * clipped * float(mixer_weights[m])
            old_output_col = self.W_o[:, action].copy()
            self.W_o[:, action] += scale * h_next
            self.b_o[action] += scale

            grad_h = scale * old_output_col
            z = cache["z"]
            r = cache["r"]
            h_tilde = cache["h_tilde"]
            hidden = cache["hidden"]
            joined = cache["joined"]
            candidate_joined = cache["candidate_joined"]

            grad_h_tilde = grad_h * z
            grad_z = grad_h * (h_tilde - hidden)
            grad_u_h = grad_h_tilde * (1.0 - h_tilde * h_tilde)
            grad_candidate_joined = self.W_h @ grad_u_h
            grad_r_hidden = grad_candidate_joined[self.obs_dim :]
            grad_r = grad_r_hidden * hidden
            grad_u_r = grad_r * r * (1.0 - r)
            grad_u_z = grad_z * z * (1.0 - z)

            self.W_h += np.outer(candidate_joined, grad_u_h)
            self.b_h += grad_u_h
            self.W_r += np.outer(joined, grad_u_r)
            self.b_r += grad_u_r
            self.W_z += np.outer(joined, grad_u_z)
            self.b_z += grad_u_z

        self.W_o = np.clip(self.W_o, -5.0, 5.0)
        self.b_o = np.clip(self.b_o, -5.0, 5.0)
        self.W_h = np.clip(self.W_h, -5.0, 5.0)
        self.b_h = np.clip(self.b_h, -5.0, 5.0)
        self.W_r = np.clip(self.W_r, -5.0, 5.0)
        self.b_r = np.clip(self.b_r, -5.0, 5.0)
        self.W_z = np.clip(self.W_z, -5.0, 5.0)
        self.b_z = np.clip(self.b_z, -5.0, 5.0)


class MonotonicQMIXMixer:
    """State-conditioned monotonic mixer Q_tot = f_mix(Q_1,...,Q_M;s)."""

    def __init__(self, cfg: PaperQMIXConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed + 202)
        self.hyper_w = self.rng.normal(0.0, 0.04, size=(cfg.global_state_dim, cfg.n_uavs))
        self.hyper_b = np.ones(cfg.n_uavs, dtype=float) / cfg.n_uavs
        self.bias_w = self.rng.normal(0.0, 0.02, size=cfg.global_state_dim)
        self.bias_b = 0.0
        self.last_weights = np.ones(cfg.n_uavs, dtype=float) / cfg.n_uavs

    def weights_for_state(self, state: Sequence[float]) -> np.ndarray:
        weights = np.abs(np.asarray(state, dtype=float) @ self.hyper_w + self.hyper_b) + 1.0e-6
        self.last_weights = weights
        return weights

    def forward(self, agent_qs: Sequence[float], state: Sequence[float]) -> float:
        weights = self.weights_for_state(state)
        bias = float(np.asarray(state, dtype=float) @ self.bias_w + self.bias_b)
        return float(np.dot(weights, np.asarray(agent_qs, dtype=float)) + bias)

    def update_bias(self, state: Sequence[float], td_error: float, learning_rate: float) -> None:
        clipped = float(np.clip(td_error, -8.0, 8.0))
        state_arr = np.asarray(state, dtype=float)
        self.bias_w += learning_rate * clipped * state_arr
        self.bias_b += learning_rate * clipped
        self.bias_w = np.clip(self.bias_w, -5.0, 5.0)
        self.bias_b = float(np.clip(self.bias_b, -5.0, 5.0))


def hard_update(target, source) -> None:
    for name, value in source.__dict__.items():
        if isinstance(value, np.ndarray):
            setattr(target, name, value.copy())
        elif name not in {"rng"}:
            setattr(target, name, deepcopy(value))


class DeepQMIXGRUTrainer:
    def __init__(self, cfg: PaperQMIXConfig):
        self.cfg = cfg
        self.env = PaperPOMDPEnv(cfg)
        self.agent = DeepGRUAgent(cfg, cfg.observation_dim)
        self.target_agent = deepcopy(self.agent)
        self.mixer = MonotonicQMIXMixer(cfg)
        self.target_mixer = deepcopy(self.mixer)
        self.buffer = ReplayBuffer(cfg.replay_capacity, cfg.seed + 303)

    def _chosen_qs(
        self,
        agent: DeepGRUAgent,
        observations: Sequence[np.ndarray],
        hidden_states: Sequence[np.ndarray],
        actions: Sequence[int],
    ) -> Tuple[List[float], List[np.ndarray]]:
        qs = []
        next_hidden = []
        for obs, h, action in zip(observations, hidden_states, actions):
            q_values, h_next = agent.forward(obs, h)
            qs.append(float(q_values[action]))
            next_hidden.append(h_next)
        return qs, next_hidden

    def _max_next_qs(
        self,
        next_observations: Sequence[np.ndarray],
        next_hidden: Sequence[np.ndarray],
        next_masks: Sequence[np.ndarray],
    ) -> List[float]:
        values = []
        for obs, h, mask in zip(next_observations, next_hidden, next_masks):
            q_values, _ = self.target_agent.forward(obs, h)
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                values.append(0.0)
            else:
                values.append(float(np.max(q_values[valid])))
        return values

    def _learn_from_transition(self, transition: Transition) -> float:
        chosen_qs, chosen_hidden = self._chosen_qs(
            self.agent, transition.obs, transition.hidden, transition.actions
        )
        q_tot = self.mixer.forward(chosen_qs, transition.state)
        if transition.done:
            target = transition.reward
        else:
            next_qs = self._max_next_qs(transition.next_obs, transition.next_hidden, transition.next_masks)
            next_q_tot = self.target_mixer.forward(next_qs, transition.next_state)
            target = transition.reward + self.cfg.gamma * next_q_tot
        td_error = float(np.clip(target - q_tot, -10.0, 10.0))
        mixer_weights = self.mixer.weights_for_state(transition.state)
        self.agent.update_recurrent_step(
            transition.obs,
            transition.hidden,
            transition.actions,
            td_error,
            mixer_weights,
            self.cfg.learning_rate,
        )
        self.mixer.update_bias(transition.state, td_error, self.cfg.learning_rate * 0.2)
        return td_error * td_error

    def train(self) -> Dict:
        episode_rewards: List[float] = []
        losses: List[float] = []
        for ep in range(self.cfg.episodes):
            observations, state, _candidates, masks = self.env.reset(seed_offset=ep)
            hidden = [self.agent.initial_hidden() for _ in range(self.cfg.n_uavs)]
            done = False
            ep_reward = 0.0
            epsilon = max(
                self.cfg.epsilon_end,
                self.cfg.epsilon_start
                - (self.cfg.epsilon_start - self.cfg.epsilon_end) * ep / max(1, self.cfg.episodes - 1),
            )
            while not done:
                prev_hidden = [h.copy() for h in hidden]
                actions, next_hidden_policy, _q_values, _chosen_hidden = self.agent.select_actions(
                    observations, hidden, masks, epsilon=epsilon
                )
                next_obs, next_state, _next_candidates, next_masks, reward, done, _info, _obs0, _state0, _masks0 = self.env.step(actions)
                self.buffer.add(
                    Transition(
                        obs=observations,
                        state=state,
                        hidden=prev_hidden,
                        actions=actions,
                        reward=float(reward),
                        next_obs=next_obs,
                        next_state=next_state,
                        next_hidden=next_hidden_policy,
                        masks=masks,
                        next_masks=next_masks,
                        done=done,
                    )
                )
                for sample in self.buffer.sample(self.cfg.batch_size):
                    losses.append(self._learn_from_transition(sample))
                observations = next_obs
                state = next_state
                masks = next_masks
                hidden = next_hidden_policy
                ep_reward += float(reward)

            episode_rewards.append(float(ep_reward))
            if (ep + 1) % self.cfg.target_update_interval == 0:
                hard_update(self.target_agent, self.agent)
                hard_update(self.target_mixer, self.mixer)

        evaluation = self.evaluate(seed_offset=999)
        return {
            "episode_rewards": [round(x, 6) for x in episode_rewards],
            "loss_tail": [round(float(x), 6) for x in losses[-100:]],
            "final_metrics": evaluation["metrics"],
            "final_history": evaluation["history"],
            "path_rows": evaluation["path_rows"],
            "risk_points": self.risk_point_rows(evaluation["points"]),
            "config": config_rows(self.cfg),
        }

    def evaluate(self, seed_offset: int = 999) -> Dict:
        observations, state, _candidates, masks = self.env.reset(seed_offset=seed_offset)
        hidden = [self.agent.initial_hidden() for _ in range(self.cfg.n_uavs)]
        done = False
        rewards: List[float] = []
        total_tp = total_fp = total_fn = 0
        history: List[Dict] = []
        path_rows: List[Dict] = []
        while not done:
            actions, hidden, _q_values, _chosen_hidden = self.agent.select_actions(
                observations, hidden, masks, epsilon=0.0
            )
            next_obs, next_state, _next_candidates, next_masks, reward, done, info, _obs0, _state0, _masks0 = self.env.step(actions)
            rewards.append(float(reward))
            total_tp += int(info["tp"])
            total_fp += int(info["fp"])
            total_fn += int(info["fn"])
            history.append(
                {
                    "time": info["time"],
                    "mean_rainfall": info["mean_rainfall"],
                    "rainfall": [round(float(x), 6) for x in info["rainfall"]],
                    "high_risk_set": info["high_risk_set"],
                    "high_risk_count": len(info["high_risk_set"]),
                    "selected_points": info["selected_points"],
                    "selected_names": info["selected_names"],
                    "routes": info["routes"],
                    "tp": info["tp"],
                    "fp": info["fp"],
                    "fn": info["fn"],
                    "reward": info["reward"],
                }
            )
            for row in info["route_rows"]:
                row = dict(row)
                row["reward_t"] = info["reward"]
                row["selected_points_t"] = info["selected_points"]
                path_rows.append(row)
            observations = next_obs
            state = next_state
            masks = next_masks
        points_snapshot = deepcopy(self.env.points)
        return {
            "metrics": metric_dict(total_tp, total_fp, total_fn, rewards),
            "history": history,
            "path_rows": path_rows,
            "points": points_snapshot,
        }

    @staticmethod
    def risk_point_rows(points: Sequence[RiskPoint]) -> List[Dict]:
        rows = []
        for point in points:
            rows.append(
                {
                    "idx": point.idx,
                    "name": point.name,
                    "x": round(point.location[0], 6),
                    "y": round(point.location[1], 6),
                    "flood_susceptibility": round(point.flood_susceptibility, 6),
                    "drainage_capacity": round(point.drainage_capacity, 6),
                    "importance": round(point.importance, 6),
                    "initial_state": point.initial_state,
                    "initial_belief": round(point.initial_belief, 6),
                    "final_state": point.true_state,
                    "final_belief": round(point.belief, 6),
                }
            )
        return rows


def config_rows(cfg: PaperQMIXConfig) -> List[Dict[str, object]]:
    rows = []
    for key, value in asdict(cfg).items():
        rows.append({"parameter": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, tuple) else value})
    rows.extend(
        [
            {"parameter": "observation_dim", "value": cfg.observation_dim},
            {"parameter": "global_state_dim", "value": cfg.global_state_dim},
        {"parameter": "action_definition", "value": "one risk point per UAV per time step; route/trajectory is accumulated across time steps"},
        ]
    )
    return rows


def write_csv(path: str, rows: Sequence[Dict]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key, value in row.items():
                if isinstance(value, (list, tuple, dict)):
                    out[key] = json.dumps(value, ensure_ascii=False)
                else:
                    out[key] = value
            writer.writerow(out)


def write_report(path: str, cfg: PaperQMIXConfig, result: Dict) -> None:
    metrics = result["final_metrics"]
    lines = [
        "# Deep QMIX-GRU 单点滚动匹配/轨迹规划实验报告",
        "",
        "本文件对应 MF-MLe-MUAV-UWR-POMDP 的深度 QMIX-GRU + CTDE 求解器。",
        "",
        "## 动作定义",
        "",
        "按照最终符号审查后的建模版，每个时间步每架无人机最多访问 1 个风险点；一个任务周期内的完整路径由多个时间步的连续匹配动作累积形成。",
        "",
        "`a_t = Y_t`，其中 `Y_t^{m,i}=1` 表示无人机 `m` 在时间步 `t` 访问风险点 `i`。跨时间步累计得到 `rho^m=(i_1^m,i_2^m,...,i_T^m)`。",
        "",
        f"本次设置：`M={cfg.n_uavs}`，`N={cfg.n_points}`，`T={cfg.horizon}`，每架无人机每个时间步最多访问 `{cfg.route_length}` 个点。",
        "",
        "## 可行动作约束",
        "",
        f"当前代码实现距离-能耗代理约束：单步最大飞行距离为 `{cfg.max_route_distance}`，初始能量为 `{cfg.energy_capacity}`，单位距离能耗为 `{cfg.energy_per_distance}`，最低安全电量为 `{cfg.min_safe_energy}`。候选动作必须满足执行后剩余能量不低于最低安全电量。",
        "",
        "## 降雨设置",
        "",
        "降雨由暴雨峰型、周期波动和空间扰动组成，并对每个固定风险点分别生成局部降雨强度。",
        "",
        "## 结果指标",
        "",
        f"- Recall: {metrics['recall']:.6f}",
        f"- Precision: {metrics['precision']:.6f}",
        f"- F1: {metrics['f1']:.6f}",
        f"- Total reward: {metrics['total_reward']:.6f}",
        "",
        "## 输出文件",
        "",
            "- `deep_qmix_gru_uav_paths.csv`: 每个时间步每架无人机的单点动作，以及跨时间步累计形成的轨迹。",
        "- `deep_qmix_gru_rainfall_high_risk.csv`: 每个时间步降雨、高风险集合、选点和 TP/FP/FN。",
        "- `deep_qmix_gru_risk_points.csv`: 固定候选风险点的坐标、易涝系数、排水能力、重要性和信念。",
        "- `deep_qmix_gru_training_result.json`: 训练奖励、损失尾部、最终轨迹和指标。",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_deep_qmix_experiment(cfg: Optional[PaperQMIXConfig] = None, output_dir: Optional[str] = None) -> Dict:
    cfg = cfg or PaperQMIXConfig()
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    trainer = DeepQMIXGRUTrainer(cfg)
    result = trainer.train()

    write_csv(os.path.join(output_dir, "deep_qmix_gru_config.csv"), result["config"])
    write_csv(os.path.join(output_dir, "deep_qmix_gru_risk_points.csv"), result["risk_points"])
    write_csv(os.path.join(output_dir, "deep_qmix_gru_uav_paths.csv"), result["path_rows"])
    rain_rows = []
    for row in result["final_history"]:
        rain_rows.append(
            {
                "time": row["time"],
                "mean_rainfall": row["mean_rainfall"],
                "rainfall": row["rainfall"],
                "high_risk_set": row["high_risk_set"],
                "high_risk_count": row["high_risk_count"],
                "selected_points": row["selected_points"],
                "selected_names": row["selected_names"],
                "routes": row["routes"],
                "tp": row["tp"],
                "fp": row["fp"],
                "fn": row["fn"],
                "reward": row["reward"],
            }
        )
    write_csv(os.path.join(output_dir, "deep_qmix_gru_rainfall_high_risk.csv"), rain_rows)
    write_report(os.path.join(output_dir, "deep_qmix_gru_experiment_report.md"), cfg, result)
    with open(os.path.join(output_dir, "deep_qmix_gru_training_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def run_deep_qmix_multi_seed(
    cfg: Optional[PaperQMIXConfig] = None,
    output_dir: Optional[str] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict:
    cfg = cfg or PaperQMIXConfig()
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    seeds = list(seeds or [cfg.seed])

    seed_results: List[Dict] = []
    full_results: List[Dict] = []
    main_result: Optional[Dict] = None
    for run_idx, seed in enumerate(seeds):
        run_cfg = replace(cfg, seed=int(seed))
        run_output_dir = output_dir if run_idx == 0 else os.path.join(output_dir, f"seed_{seed}")
        result = run_deep_qmix_experiment(run_cfg, run_output_dir)
        full_results.append(result)
        if main_result is None:
            main_result = result
        seed_row = {"seed": int(seed), "run_index": run_idx}
        seed_row.update(result["final_metrics"])
        seed_results.append(seed_row)

    summary = metric_summary(full_results)
    write_csv(os.path.join(output_dir, "deep_qmix_gru_seed_metrics.csv"), seed_results)
    with open(os.path.join(output_dir, "deep_qmix_gru_multi_seed_result.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary_metrics": summary,
                "seed_results": seed_results,
                "main_seed": seeds[0],
                "config": config_rows(replace(cfg, seed=int(seeds[0]))),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return {
        "summary_metrics": summary,
        "seed_results": seed_results,
        "main_result": main_result,
        "final_metrics": summary,
    }


def main() -> None:
    cfg = PaperQMIXConfig()
    result = run_deep_qmix_experiment(cfg)
    metrics = result["final_metrics"]
    print("Deep QMIX-GRU path-planning experiment completed.")
    print(
        f"recall={metrics['recall']:.3f}, precision={metrics['precision']:.3f}, "
        f"f1={metrics['f1']:.3f}, reward={metrics['total_reward']:.2f}"
    )


if __name__ == "__main__":
    main()
