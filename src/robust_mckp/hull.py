"""Upper-hull filtering for convexified option sets."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import sys
from typing import List, Sequence, Tuple

import numpy as np



@dataclass
class Hull:
    """Upper hull representation for an item.

    Attributes:
        costs: 1D array of hull vertex costs (c values), increasing.
        values: 1D array of hull vertex values (v values), increasing in cost order.
        option_indices: Original option indices for each hull vertex.
        delta_costs: 1D array of segment lengths between vertices.
        slopes: 1D array of segment slopes Δv/Δc.
    """

    costs: np.ndarray
    values: np.ndarray
    option_indices: np.ndarray
    delta_costs: np.ndarray
    slopes: np.ndarray


@dataclass
class Point:
    cost: float
    value: float
    option_index: int


def merge_equal_cost(points: Sequence[Point]) -> List[Point]:
    """Merge points with equal cost by keeping the maximum value.

    Args:
        points: Iterable of Point objects.

    Returns:
        List of points sorted by cost with duplicates merged.
    """

    if not points:
        return []
    pts = sorted(points, key=lambda p: (p.cost, p.value, p.option_index))
    merged: List[Point] = []
    cur = pts[0]
    for p in pts[1:]:
        if p.cost == cur.cost:
            if p.value > cur.value or (p.value == cur.value and p.option_index < cur.option_index):
                cur = p
        else:
            merged.append(cur)
            cur = p
    merged.append(cur)
    return merged


def prune_dominated(points: Sequence[Point]) -> List[Point]:
    """Remove points dominated by lower-cost higher-value points.

    Args:
        points: Points sorted by cost.

    Returns:
        List with strictly increasing values in cost order.
    """

    if not points:
        return []
    filtered: List[Point] = []
    max_val = -np.inf
    for p in points:
        if p.value > max_val:
            filtered.append(p)
            max_val = p.value
    return filtered


def _cross_sign(o: Point, a: Point, b: Point) -> int:
    """Return the exact orientation sign of three binary64 points."""

    ac = a.cost - o.cost
    bv = b.value - o.value
    av = a.value - o.value
    bc = b.cost - o.cost
    left = ac * bv
    right = av * bc
    determinant = left - right
    error_bound = 8.0 * sys.float_info.epsilon * (abs(left) + abs(right))
    if abs(determinant) > error_bound:
        return 1 if determinant > 0.0 else -1

    f = Fraction.from_float
    exact = (f(a.cost) - f(o.cost)) * (f(b.value) - f(o.value)) - (
        f(a.value) - f(o.value)
    ) * (f(b.cost) - f(o.cost))
    return (exact > 0) - (exact < 0)


def upper_hull(points: Sequence[Point]) -> List[Point]:
    """Compute the upper hull (convex envelope) of 2D points.

    Collinear interior points on the hull boundary are removed.

    Args:
        points: Points sorted by cost.

    Returns:
        List of hull vertices ordered by increasing cost.
    """

    if not points:
        return []
    if len(points) <= 2:
        return list(points)

    pts = list(points)
    upper: List[Point] = []
    for p in pts:
        while len(upper) >= 2:
            if _cross_sign(upper[-2], upper[-1], p) >= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    return upper


def build_upper_hull(costs: np.ndarray, values: np.ndarray, option_indices: np.ndarray) -> Hull:
    """Build upper hull with dominance pruning and segment data.

    Args:
        costs: 1D array of costs.
        values: 1D array of values.
        option_indices: 1D array of option indices aligned with costs/values.

    Returns:
        Hull structure with vertices and segment slopes.
    """

    points = [Point(float(c), float(v), int(j)) for c, v, j in zip(costs, values, option_indices)]
    points = merge_equal_cost(points)
    points = prune_dominated(points)
    points = upper_hull(points)

    if not points:
        return Hull(
            costs=np.array([]),
            values=np.array([]),
            option_indices=np.array([], dtype=int),
            delta_costs=np.array([]),
            slopes=np.array([]),
        )
    if len(points) == 1:
        return Hull(
            costs=np.array([points[0].cost], dtype=float),
            values=np.array([points[0].value], dtype=float),
            option_indices=np.array([points[0].option_index], dtype=int),
            delta_costs=np.array([], dtype=float),
            slopes=np.array([], dtype=float),
        )

    costs_arr = np.array([p.cost for p in points], dtype=float)
    values_arr = np.array([p.value for p in points], dtype=float)
    idx_arr = np.array([p.option_index for p in points], dtype=int)
    delta_costs = np.diff(costs_arr)
    delta_values = np.diff(values_arr)
    # Hull segment lengths are structurally positive after exact-cost merging;
    # no tolerance may erase a genuine positive segment. Overflow to +inf is
    # conservative for an LP upper-bound slope.
    with np.errstate(divide="ignore", over="ignore", invalid="raise"):
        slopes = np.divide(delta_values, delta_costs, dtype=float)
    return Hull(costs=costs_arr, values=values_arr, option_indices=idx_arr, delta_costs=delta_costs, slopes=slopes)
