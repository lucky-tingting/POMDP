from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple


Point = Tuple[float, float]
ROOT = Path(__file__).resolve().parent
SHARED_SETTINGS_PATH = ROOT / "shared_comparison_settings.json"


def load_shared_comparison_settings(path: str | Path = SHARED_SETTINGS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def energy_cost(cfg: "PPBVIRUMAConfig", travel_distance: float) -> float:
    return cfg.energy_per_distance * travel_distance


@dataclass(frozen=True)
class PPBVIRUMAConfig:
    """Configuration for P-PBVI-RUMA under the shared comparison setting."""

    n_points: int = 20
    n_uavs: int = 3
    horizon: int = 16
    seeds: Tuple[int, ...] = (2026, 2027, 2028, 2029, 2030)

    max_points_per_uav_per_step: int = 1
    max_uavs_per_point_per_step: int = 1
    max_route_distance: float = 22.0
    energy_capacity: float = 160.0
    energy_per_distance: float = 1.0
    min_safe_energy: float = 16.0

    grid_spacing: float = 5.0
    neighbor_radius: float = 7.2
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
    gamma: float = 0.95
    local_transition_p01: float = 0.20
    local_transition_p10: float = 0.10
    pbvi_iterations: int = 3
    pbvi_tolerance: float = 1e-6
    belief_grid_size: int = 51
    candidate_top_k: int = 20
    fallback_top_k: int = 20
    region_count: int = 4
    top_k_per_region: int = 3

    @property
    def seed_list(self) -> str:
        return ";".join(str(seed) for seed in self.seeds)

    def __post_init__(self) -> None:
        if self.n_points <= 0:
            raise ValueError("n_points must be positive")
        if self.n_uavs <= 0:
            raise ValueError("n_uavs must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.max_points_per_uav_per_step != 1:
            raise ValueError("P-PBVI-RUMA comparison uses one point per UAV per step")
        if self.max_uavs_per_point_per_step != 1:
            raise ValueError("P-PBVI-RUMA comparison uses one UAV per point per step")
        if len(self.ml_observation_matrix) != 2:
            raise ValueError("ml_observation_matrix must have low/high state rows")
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
        if self.pbvi_iterations <= 0:
            raise ValueError("pbvi_iterations must be positive")
        if self.pbvi_tolerance <= 0.0:
            raise ValueError("pbvi_tolerance must be positive")
        if self.belief_grid_size < 1:
            raise ValueError("belief_grid_size must be at least 1")
        if self.candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be positive")
        if self.fallback_top_k <= 0:
            raise ValueError("fallback_top_k must be positive")
        if self.region_count <= 0:
            raise ValueError("region_count must be positive")
        if self.top_k_per_region <= 0:
            raise ValueError("top_k_per_region must be positive")


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


def assert_matches_shared_settings(cfg: PPBVIRUMAConfig, settings: dict | None = None) -> None:
    settings = settings or load_shared_comparison_settings()
    checks = {
        "n_points": cfg.n_points,
        "n_uavs": cfg.n_uavs,
        "horizon": cfg.horizon,
        "max_points_per_uav_per_step": cfg.max_points_per_uav_per_step,
        "max_uavs_per_point_per_step": cfg.max_uavs_per_point_per_step,
        "max_route_distance": cfg.max_route_distance,
        "energy_capacity": cfg.energy_capacity,
        "energy_per_distance": cfg.energy_per_distance,
        "min_safe_energy": cfg.min_safe_energy,
    }
    for key, actual in checks.items():
        if actual != settings[key]:
            raise ValueError(f"{key}={actual!r} does not match shared setting {settings[key]!r}")
    if tuple(cfg.seeds) != tuple(settings["seeds"]):
        raise ValueError("seeds do not match shared comparison settings")


def settings_rows(cfg: PPBVIRUMAConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, value in cfg.__dict__.items():
        if isinstance(value, tuple):
            rows.append({"parameter": key, "value": ";".join(map(str, value))})
        else:
            rows.append({"parameter": key, "value": value})
    return rows


def validate_point_vectors(points: Sequence[RiskPoint], cfg: PPBVIRUMAConfig) -> None:
    if len(points) != cfg.n_points:
        raise ValueError("risk point count does not match cfg.n_points")
