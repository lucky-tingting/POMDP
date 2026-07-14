from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CandidateFilterResult:
    candidate_indices: list[int]
    mode: str
    fallback_used: bool = False


def _unique_sorted(indices: Sequence[int]) -> list[int]:
    return sorted(dict.fromkeys(int(i) for i in indices))


def topk_indices(values: Sequence[float], k: int, allowed_indices: Sequence[int] | None = None) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    pool = list(range(len(values))) if allowed_indices is None else [int(i) for i in allowed_indices]
    return sorted(pool, key=lambda i: values[i], reverse=True)[: min(k, len(pool))]


def region_threshold_candidates(
    pre_action_beliefs: Sequence[float],
    point_regions: Sequence[int],
    region_thresholds: Mapping[int, float],
) -> CandidateFilterResult:
    if len(pre_action_beliefs) != len(point_regions):
        raise ValueError("pre_action_beliefs and point_regions must have the same length")
    candidates = []
    for i, belief in enumerate(pre_action_beliefs):
        region = int(point_regions[i])
        if region not in region_thresholds:
            raise ValueError(f"missing threshold for region {region}")
        if float(belief) >= float(region_thresholds[region]):
            candidates.append(i)
    return CandidateFilterResult(_unique_sorted(candidates), mode="regional_threshold")


def region_topk_candidates(
    monitoring_indices: Sequence[float],
    point_regions: Sequence[int],
    top_k_per_region: int,
) -> CandidateFilterResult:
    if len(monitoring_indices) != len(point_regions):
        raise ValueError("monitoring_indices and point_regions must have the same length")
    by_region: dict[int, list[int]] = {}
    for i, region in enumerate(point_regions):
        by_region.setdefault(int(region), []).append(i)
    candidates: list[int] = []
    for region_indices in by_region.values():
        candidates.extend(topk_indices(monitoring_indices, top_k_per_region, region_indices))
    return CandidateFilterResult(_unique_sorted(candidates), mode="regional_topk")


def construct_candidate_set(
    mode: str,
    pre_action_beliefs: Sequence[float],
    monitoring_indices: Sequence[float],
    point_regions: Sequence[int] | None = None,
    region_thresholds: Mapping[int, float] | None = None,
    top_k: int | None = None,
    top_k_per_region: int | None = None,
    fallback_top_k: int = 20,
) -> CandidateFilterResult:
    mode = mode.lower()
    if mode == "all":
        return CandidateFilterResult(list(range(len(monitoring_indices))), mode="all")
    if mode == "topk":
        if top_k is None:
            raise ValueError("top_k is required for topk mode")
        result = CandidateFilterResult(topk_indices(monitoring_indices, top_k), mode="topk")
    elif mode == "regional_threshold":
        if point_regions is None or region_thresholds is None:
            raise ValueError("point_regions and region_thresholds are required for regional_threshold mode")
        result = region_threshold_candidates(pre_action_beliefs, point_regions, region_thresholds)
    elif mode == "regional_topk":
        if point_regions is None or top_k_per_region is None:
            raise ValueError("point_regions and top_k_per_region are required for regional_topk mode")
        result = region_topk_candidates(monitoring_indices, point_regions, top_k_per_region)
    else:
        raise ValueError(f"unknown candidate filter mode: {mode}")

    if result.candidate_indices:
        return result
    return CandidateFilterResult(
        topk_indices(monitoring_indices, fallback_top_k),
        mode=result.mode,
        fallback_used=True,
    )


def assign_grid_regions(n_points: int, region_count: int) -> list[int]:
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if region_count <= 0:
        raise ValueError("region_count must be positive")
    return [i % region_count for i in range(n_points)]
