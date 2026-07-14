from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from step_01_config_and_parameters import PPBVIRUMAConfig
from step_02_belief_update import bayes_update, clip_prob


def entropy(belief: float) -> float:
    belief = clip_prob(belief)
    return -(belief * math.log(belief) + (1.0 - belief) * math.log(1.0 - belief))


def expected_information_gain(belief: float, sensitivity: float, specificity: float) -> float:
    belief = clip_prob(belief)
    p_z1 = sensitivity * belief + (1.0 - specificity) * (1.0 - belief)
    p_z0 = 1.0 - p_z1
    post_z1 = bayes_update(belief, sensitivity, 1.0 - specificity)
    post_z0 = bayes_update(belief, 1.0 - sensitivity, specificity)
    gain = entropy(belief) - p_z1 * entropy(post_z1) - p_z0 * entropy(post_z0)
    return max(0.0, gain)


def miss_penalty_phi(cfg: PPBVIRUMAConfig, belief: float) -> float:
    return max(0.0, clip_prob(belief) - cfg.miss_threshold)


def local_immediate_reward(
    cfg: PPBVIRUMAConfig,
    belief: float,
    action: int,
    importance: float,
) -> float:
    belief = clip_prob(belief)
    if action == 1:
        info_gain = expected_information_gain(belief, cfg.uav_sensitivity, cfg.uav_specificity)
        return (
            cfg.lambda_cover * importance * belief
            - cfg.lambda_fp * (1.0 - belief)
            + cfg.lambda_info * info_gain
        )
    if action == 0:
        return -cfg.lambda_miss * importance * miss_penalty_phi(cfg, belief)
    raise ValueError("local action must be 0 or 1")


def hard_observation_set(action: int) -> list[int | None]:
    if action == 1:
        return [0, 1]
    if action == 0:
        return [None]
    raise ValueError("local action must be 0 or 1")


def hard_observation_likelihood(
    cfg: PPBVIRUMAConfig,
    state: int,
    action: int,
    hard_observation: int | None,
) -> float:
    if action == 0:
        return 1.0 if hard_observation is None else 0.0
    if action != 1:
        raise ValueError("local action must be 0 or 1")
    if hard_observation == 1:
        return cfg.uav_sensitivity if state == 1 else 1.0 - cfg.uav_specificity
    if hard_observation == 0:
        return 1.0 - cfg.uav_sensitivity if state == 1 else cfg.uav_specificity
    return 0.0


def hard_observation_probability(
    cfg: PPBVIRUMAConfig,
    belief: float,
    action: int,
    hard_observation: int | None,
) -> float:
    belief = clip_prob(belief)
    psi_high = hard_observation_likelihood(cfg, 1, action, hard_observation)
    psi_low = hard_observation_likelihood(cfg, 0, action, hard_observation)
    return psi_high * belief + psi_low * (1.0 - belief)


def hard_belief_operator(
    cfg: PPBVIRUMAConfig,
    belief: float,
    action: int,
    hard_observation: int | None,
) -> float:
    if action == 0:
        if hard_observation is not None:
            raise ValueError("unmonitored action only admits null hard observation")
        return clip_prob(belief)
    if action != 1:
        raise ValueError("local action must be 0 or 1")
    psi_high = hard_observation_likelihood(cfg, 1, action, hard_observation)
    psi_low = hard_observation_likelihood(cfg, 0, action, hard_observation)
    return bayes_update(belief, psi_high, psi_low)


def transition_operator(belief: float, p01: float, p10: float) -> float:
    return clip_prob(p01 * (1.0 - clip_prob(belief)) + (1.0 - p10) * clip_prob(belief))


def soft_observation_probability(
    predicted_belief: float,
    ml_level: int,
    ml_observation_matrix: Sequence[Sequence[float]],
) -> float:
    level_count = len(ml_observation_matrix[0])
    if not 1 <= int(ml_level) <= level_count:
        raise ValueError(f"ml_level must be in 1..{level_count}")
    idx = int(ml_level) - 1
    theta_low = ml_observation_matrix[0][idx]
    theta_high = ml_observation_matrix[1][idx]
    predicted_belief = clip_prob(predicted_belief)
    return theta_high * predicted_belief + theta_low * (1.0 - predicted_belief)


def soft_belief_operator(
    predicted_belief: float,
    ml_level: int,
    ml_observation_matrix: Sequence[Sequence[float]],
) -> float:
    level_count = len(ml_observation_matrix[0])
    if not 1 <= int(ml_level) <= level_count:
        raise ValueError(f"ml_level must be in 1..{level_count}")
    idx = int(ml_level) - 1
    theta_low = ml_observation_matrix[0][idx]
    theta_high = ml_observation_matrix[1][idx]
    return bayes_update(predicted_belief, theta_high, theta_low)


@dataclass
class PBVISolution:
    belief_grid: list[float]
    values: list[float]
    q_monitor: list[float]
    q_skip: list[float]
    iterations: int
    max_residual: float
    converged: bool


@dataclass
class PointwisePBVIIndexEstimator:
    """Finite-grid pointwise PBVI approximation for a single risk point."""

    cfg: PPBVIRUMAConfig
    belief_grid: Sequence[float] | None = None
    _solution_cache: dict[tuple[float, float, float], PBVISolution] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.belief_grid is None:
            intervals = self.cfg.belief_grid_size
            self.belief_grid = [round(i / intervals, 12) for i in range(intervals + 1)]
        else:
            self.belief_grid = [float(value) for value in self.belief_grid]

    def value_at(self, belief: float, values: Sequence[float]) -> float:
        if len(values) != len(self.belief_grid):
            raise ValueError("values length must match belief_grid length")
        belief = min(1.0, max(0.0, float(belief)))
        if belief <= self.belief_grid[0]:
            return float(values[0])
        if belief >= self.belief_grid[-1]:
            return float(values[-1])
        for right_idx in range(1, len(self.belief_grid)):
            left_b = self.belief_grid[right_idx - 1]
            right_b = self.belief_grid[right_idx]
            if left_b <= belief <= right_b:
                width = right_b - left_b
                if width <= 0.0:
                    return float(values[right_idx])
                weight = (belief - left_b) / width
                return float((1.0 - weight) * values[right_idx - 1] + weight * values[right_idx])
        return float(values[-1])

    def nearest_value(self, belief: float, values: Sequence[float]) -> float:
        return self.value_at(belief, values)

    def _expected_future_value(
        self,
        belief: float,
        action: int,
        values: Sequence[float],
        p01: float,
        p10: float,
    ) -> float:
        expected = 0.0
        for hard_observation in hard_observation_set(action):
            p_z = hard_observation_probability(self.cfg, belief, action, hard_observation)
            b_z = hard_belief_operator(self.cfg, belief, action, hard_observation)
            predicted_next = transition_operator(b_z, p01, p10)
            ml_expected = 0.0
            for ml_level in range(1, self.cfg.n_ml_levels + 1):
                p_r = soft_observation_probability(predicted_next, ml_level, self.cfg.ml_observation_matrix)
                next_belief = soft_belief_operator(predicted_next, ml_level, self.cfg.ml_observation_matrix)
                ml_expected += p_r * self.value_at(next_belief, values)
            expected += p_z * ml_expected
        return expected

    def action_value(
        self,
        belief: float,
        action: int,
        importance: float,
        values: Sequence[float],
        p01: float,
        p10: float,
    ) -> float:
        return local_immediate_reward(self.cfg, belief, action, importance) + self.cfg.gamma * self._expected_future_value(
            belief,
            action,
            values,
            p01,
            p10,
        )

    def solve_values(self, importance: float, p01: float, p10: float) -> PBVISolution:
        key = (round(float(importance), 8), round(float(p01), 8), round(float(p10), 8))
        if key in self._solution_cache:
            return self._solution_cache[key]
        values = [0.0 for _ in self.belief_grid]
        q_monitor = [0.0 for _ in self.belief_grid]
        q_skip = [0.0 for _ in self.belief_grid]
        residual = float("inf")
        converged = False
        iterations = 0
        for iteration in range(1, self.cfg.pbvi_iterations + 1):
            new_values: list[float] = []
            q_monitor = []
            q_skip = []
            for belief in self.belief_grid:
                monitor_q = self.action_value(belief, 1, importance, values, p01, p10)
                skip_q = self.action_value(belief, 0, importance, values, p01, p10)
                q_monitor.append(monitor_q)
                q_skip.append(skip_q)
                new_values.append(max(monitor_q, skip_q))
            residual = max(abs(new - old) for new, old in zip(new_values, values))
            values = new_values
            iterations = iteration
            if residual < self.cfg.pbvi_tolerance:
                converged = True
                break
        solution = PBVISolution(
            belief_grid=list(self.belief_grid),
            values=values,
            q_monitor=q_monitor,
            q_skip=q_skip,
            iterations=iterations,
            max_residual=residual,
            converged=converged,
        )
        self._solution_cache[key] = solution
        return solution

    def q_values(
        self,
        belief: float,
        importance: float,
        p01: float | None = None,
        p10: float | None = None,
    ) -> tuple[float, float]:
        p01 = self.cfg.local_transition_p01 if p01 is None else p01
        p10 = self.cfg.local_transition_p10 if p10 is None else p10
        solution = self.solve_values(importance, p01, p10)
        return self.value_at(belief, solution.q_monitor), self.value_at(belief, solution.q_skip)

    def monitoring_index(
        self,
        belief: float,
        importance: float,
        p01: float | None = None,
        p10: float | None = None,
    ) -> float:
        monitor_q, skip_q = self.q_values(belief, importance, p01, p10)
        return monitor_q - skip_q

    def pointwise_threshold(
        self,
        importance: float,
        p01: float | None = None,
        p10: float | None = None,
    ) -> float | None:
        p01 = self.cfg.local_transition_p01 if p01 is None else p01
        p10 = self.cfg.local_transition_p10 if p10 is None else p10
        solution = self.solve_values(importance, p01, p10)
        for belief, monitor_q, skip_q in zip(solution.belief_grid, solution.q_monitor, solution.q_skip):
            if monitor_q >= skip_q:
                return belief
        return None
