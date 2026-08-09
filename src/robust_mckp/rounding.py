"""Rounding heuristics for discrete recovery from LP solution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .greedy import LPSolution
from .hull import Hull
from .utils import EPS


@dataclass
class DiscreteSolution:
    """Discrete solution in hull-vertex space."""

    vertices: List[int]
    cost: float
    value: float


def _discrete_value_cost(vertices: List[int], hulls: List[Hull]) -> Tuple[float, float]:
    total_cost = 0.0
    total_value = 0.0
    for idx, hull in zip(vertices, hulls):
        total_cost += float(hull.costs[idx])
        total_value += float(hull.values[idx])
    return total_value, total_cost


def _round_positions(lp: LPSolution, hulls: List[Hull]) -> List[int]:
    vertices: List[int] = []
    for pos, hull in zip(lp.positions, hulls):
        if pos.lambda_ >= 1.0:
            vertices.append(pos.upper_vertex)
        else:
            vertices.append(pos.lower_vertex)
    return vertices


def _find_fractional_item(lp: LPSolution) -> Optional[int]:
    for i, pos in enumerate(lp.positions):
        if 0.0 < pos.lambda_ < 1.0:
            return i
    return None


def _repair(
    vertices: List[int],
    hulls: List[Hull],
    capacity: float,
    exclude_item: Optional[int] = None,
) -> Optional[DiscreteSolution]:
    value, cost = _discrete_value_cost(vertices, hulls)
    if cost <= capacity:
        return DiscreteSolution(vertices=vertices, cost=cost, value=value)

    n = len(vertices)
    while cost > capacity:
        best_item = None
        best_ratio = np.inf
        for i in range(n):
            if exclude_item is not None and i == exclude_item:
                continue
            idx = vertices[i]
            if idx <= 0:
                continue
            hull = hulls[i]
            dc = float(hull.costs[idx] - hull.costs[idx - 1])
            dv = float(hull.values[idx] - hull.values[idx - 1])
            if dc <= 0.0:
                continue
            ratio = dv / dc
            if ratio < best_ratio - EPS:
                best_ratio = ratio
                best_item = i
        if best_item is None:
            return None
        # apply downgrade
        i = best_item
        hull = hulls[i]
        idx = vertices[i]
        dc = float(hull.costs[idx] - hull.costs[idx - 1])
        dv = float(hull.values[idx] - hull.values[idx - 1])
        vertices[i] = idx - 1
        cost -= dc
        value -= dv

    return DiscreteSolution(vertices=vertices, cost=cost, value=value)


def _upgrade_completion(vertices: List[int], hulls: List[Hull], capacity: float) -> DiscreteSolution:
    value, cost = _discrete_value_cost(vertices, hulls)
    residual = capacity - cost
    if residual <= 0.0:
        return DiscreteSolution(vertices=vertices, cost=cost, value=value)

    n = len(vertices)
    while residual > 0.0:
        best_item = None
        best_ratio = -np.inf
        best_dc = 0.0
        best_dv = 0.0
        for i in range(n):
            idx = vertices[i]
            hull = hulls[i]
            if idx >= len(hull.costs) - 1:
                continue
            dc = float(hull.costs[idx + 1] - hull.costs[idx])
            dv = float(hull.values[idx + 1] - hull.values[idx])
            if dc <= 0.0 or dc > residual:
                continue
            ratio = dv / dc
            if ratio > best_ratio + EPS:
                best_ratio = ratio
                best_item = i
                best_dc = dc
                best_dv = dv
        if best_item is None:
            break
        vertices[best_item] += 1
        cost += best_dc
        value += best_dv
        residual = capacity - cost

    return DiscreteSolution(vertices=vertices, cost=cost, value=value)


def round_lp_solution(
    lp: LPSolution,
    hulls: List[Hull],
    capacity: float,
    *,
    upgrade_completion: bool = True,
) -> Optional[DiscreteSolution]:
    """Round a fractional LP solution into a discrete selection.

    Implements round-down and round-up + repair, then optional upgrade completion.
    """

    frac_item = _find_fractional_item(lp)
    if frac_item is None:
        vertices = _round_positions(lp, hulls)
        value, cost = _discrete_value_cost(vertices, hulls)
        sol = DiscreteSolution(vertices=vertices, cost=cost, value=value)
        if upgrade_completion:
            return _upgrade_completion(vertices=sol.vertices, hulls=hulls, capacity=capacity)
        return sol

    # round down
    vertices_down = _round_positions(lp, hulls)
    vertices_down[frac_item] = lp.positions[frac_item].lower_vertex
    sol_down = _repair(vertices_down.copy(), hulls, capacity)

    # round up + repair
    vertices_up = _round_positions(lp, hulls)
    vertices_up[frac_item] = lp.positions[frac_item].upper_vertex
    sol_up = _repair(vertices_up.copy(), hulls, capacity, exclude_item=frac_item)

    candidates = [s for s in [sol_down, sol_up] if s is not None]
    if not candidates:
        return None
    best = max(candidates, key=lambda s: s.value)
    if upgrade_completion:
        return _upgrade_completion(vertices=best.vertices, hulls=hulls, capacity=capacity)
    return best
