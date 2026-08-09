"""Greedy LP solver over upper-hull segments for robust MCKP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .hull import Hull
from .utils import EPS


@dataclass
class ItemLPPosition:
    """Continuous position on an item's hull."""

    lower_vertex: int
    upper_vertex: int
    lambda_: float
    cost: float
    value: float


@dataclass
class LPSolution:
    """LP relaxation solution returned by greedy filling."""

    lp_value: float
    capacity: float
    total_cost: float
    positions: List[ItemLPPosition]
    fractional_item: Optional[int]
    fractional_lambda: float


@dataclass
class Segment:
    item: int
    index: int
    slope: float
    length: float


def _position_from_cost(hull: Hull, cost: float) -> ItemLPPosition:
    costs = hull.costs
    values = hull.values
    if cost <= costs[0]:
        return ItemLPPosition(lower_vertex=0, upper_vertex=0, lambda_=0.0, cost=costs[0], value=values[0])
    if cost >= costs[-1]:
        return ItemLPPosition(
            lower_vertex=len(costs) - 1,
            upper_vertex=len(costs) - 1,
            lambda_=0.0,
            cost=costs[-1],
            value=values[-1],
        )

    k = int(np.searchsorted(costs, cost, side="right") - 1)
    k = max(0, min(k, len(costs) - 2))
    c0 = costs[k]
    c1 = costs[k + 1]
    if c1 == c0:
        lam = 0.0
    else:
        lam = (cost - c0) / (c1 - c0)
    lam = float(min(1.0, max(0.0, lam)))
    v0 = values[k]
    v1 = values[k + 1]
    val = v0 + lam * (v1 - v0)
    return ItemLPPosition(lower_vertex=k, upper_vertex=k + 1, lambda_=lam, cost=cost, value=val)


def greedy_lp(hulls: List[Hull], capacity: float) -> LPSolution:
    """Solve the LP relaxation by greedy segment filling.

    The LP is:
        max Σ_i v_i(c_i)  s.t.  Σ_i c_i ≤ C,
    where each v_i(c) is the concave, piecewise-linear upper hull.

    Args:
        hulls: List of Hull objects, one per item.
        capacity: Total allowable cost across items (residual capacity).

    Returns:
        LPSolution with per-item positions and LP objective value.
    """

    n = len(hulls)
    base_cost = 0.0
    base_value = 0.0
    for hull in hulls:
        if hull.costs.size == 0:
            raise ValueError("empty hull")
        base_cost += float(hull.costs[0])
        base_value += float(hull.values[0])

    residual = capacity - base_cost
    if residual < -EPS:
        # infeasible; clamp to baseline
        positions = [
            ItemLPPosition(lower_vertex=0, upper_vertex=0, lambda_=0.0, cost=float(h.costs[0]), value=float(h.values[0]))
            for h in hulls
        ]
        return LPSolution(
            lp_value=base_value,
            capacity=capacity,
            total_cost=base_cost,
            positions=positions,
            fractional_item=None,
            fractional_lambda=0.0,
        )

    segments: List[Segment] = []
    for i, hull in enumerate(hulls):
        for k, (slope, length) in enumerate(zip(hull.slopes, hull.delta_costs)):
            if length <= 0.0:
                continue
            segments.append(Segment(item=i, index=k, slope=float(slope), length=float(length)))

    segments.sort(key=lambda seg: seg.slope, reverse=True)

    extra_costs = np.zeros(n, dtype=float)
    extra_value = 0.0
    fractional_item: Optional[int] = None
    fractional_lambda = 0.0

    for seg in segments:
        if residual <= 0.0:
            break
        take = min(seg.length, residual)
        if take <= 0.0:
            continue
        extra_costs[seg.item] += take
        extra_value += take * seg.slope
        residual -= take
        if take < seg.length:
            fractional_item = seg.item
            fractional_lambda = take / seg.length
            break

    positions: List[ItemLPPosition] = []
    total_cost = base_cost
    total_value = base_value
    for i, hull in enumerate(hulls):
        cost_i = float(hull.costs[0] + extra_costs[i])
        pos = _position_from_cost(hull, cost_i)
        positions.append(pos)
        total_cost += extra_costs[i]
        total_value += pos.value - float(hull.values[0])

    return LPSolution(
        lp_value=total_value,
        capacity=capacity,
        total_cost=total_cost,
        positions=positions,
        fractional_item=fractional_item,
        fractional_lambda=float(fractional_lambda),
    )
