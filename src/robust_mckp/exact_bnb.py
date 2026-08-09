"""Exact fixed-theta MCKP branch-and-bound with hull-greedy LP bounds."""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .certificate import certificate_is_feasible, compute_certificate
from .greedy import ItemLPPosition, LPSolution, greedy_lp
from .hull import Hull, build_upper_hull
from .model import PricingInstance
from .solver import solve as solve_hullround
from .utils import EPS


@dataclass(frozen=True)
class FixedThetaBNBConfig:
    """Configuration for the fixed-theta exact branch-and-bound solver."""

    tolerance: float = 1e-9
    time_limit_seconds: Optional[float] = None
    node_limit: Optional[int] = None
    node_selection: str = "best_bound"
    branch_strategy: str = "lp_fractional_then_range"
    child_order: str = "value_desc"
    branching_rule: str = "fractional_item_then_spread"
    child_ordering: str = "value_desc"
    strong_branching_candidates: int = 5
    strong_branching_depth_limit: int = 3
    strong_branching_max_children: int = 50
    strong_branching_use_fast_bound: bool = True
    incumbent_heuristic: str = "greedy"
    use_local_incumbent_improvement: bool = False
    local_search_max_passes: int = 2
    local_search_max_swaps: int = 1000
    local_search_neighborhood: str = "single"
    local_search_max_pair_evaluations: int = 20000
    use_greedy_incumbent: bool = True
    objective_cutoff: Optional[float] = None
    initial_incumbent_value: Optional[float] = None
    initial_incumbent_selection: Optional[Sequence[int]] = None
    use_cutoff_pruning: bool = True
    use_cache: bool = True
    collect_diagnostics: bool = False
    max_diagnostic_nodes: int = 20000
    diagnostic_sample_rate: int = 1
    profile_timing: bool = False
    use_fast_residual_lp_bound: bool = True
    use_cheap_prebound: bool = True
    use_min_cost_infeasibility_check: bool = True
    use_bound_cache: bool = False
    bound_cache_max_entries: int = 100000


@dataclass
class FixedThetaBNBResult:
    """Certified result for a fixed-theta MCKP solve."""

    status: str
    theta: float
    objective_value: float
    selected_options: Optional[List[int]]
    used_capacity: float
    capacity: float
    fixed_theta_residual: float
    upper_bound: float
    lower_bound: float
    absolute_gap: float
    relative_gap: float
    nodes_explored: int
    nodes_pruned_bound: int
    nodes_pruned_infeasible: int
    nodes_integral: int
    runtime_seconds: float
    config_metadata: Dict[str, object] = field(default_factory=dict)
    validation_flags: Dict[str, bool] = field(default_factory=dict)
    diagnostics: Dict[str, object] = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True)
class FixedThetaLPBoundResult:
    """Root LP upper bound result for a fixed-theta MCKP."""

    theta: float
    capacity: float
    lp_upper_bound: float
    infeasible_capacity: bool
    lp_feasible: bool
    root_lp_status: str = ""
    fractional_item: Optional[int] = None
    runtime_seconds: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class GlobalThetaBNBConfig:
    """Configuration for global theta-enumerated exact robust solving."""

    tolerance: float = 1e-9
    time_limit_seconds: Optional[float] = None
    node_limit: Optional[int] = None
    fixed_theta_time_limit_seconds: Optional[float] = None
    fixed_theta_node_limit: Optional[int] = None
    use_hullround_incumbent: bool = True
    use_fixed_theta_greedy_incumbent: bool = True
    use_caches: bool = True
    use_objective_cutoff: bool = True
    theta_order: str = "increasing"
    branching_rule: str = "fractional_item_then_spread"
    child_ordering: str = "value_desc"
    strong_branching_candidates: int = 5
    strong_branching_depth_limit: int = 3
    strong_branching_max_children: int = 50
    strong_branching_use_fast_bound: bool = True
    incumbent_heuristic: str = "greedy"
    use_local_incumbent_improvement: bool = False
    local_search_max_passes: int = 2
    local_search_max_swaps: int = 1000
    local_search_neighborhood: str = "single"
    local_search_max_pair_evaluations: int = 20000
    use_multistart_incumbent: bool = False
    multistart_theta_count: int = 8
    multistart_use_local_improvement: bool = True
    collect_diagnostics: bool = False
    max_diagnostic_nodes: int = 20000
    diagnostic_sample_rate: int = 1
    profile_timing: bool = False
    use_fast_residual_lp_bound: bool = True
    use_cheap_prebound: bool = True
    use_min_cost_infeasibility_check: bool = True
    use_bound_cache: bool = False
    bound_cache_max_entries: int = 100000


@dataclass
class GlobalThetaRecord:
    """Per-theta record emitted by the global exact solver."""

    theta: float
    status: str
    fixed_theta_capacity: float
    fixed_theta_lp_upper_bound: float
    incumbent_before_theta: float
    bnb_objective: float
    bnb_upper_bound: float
    bnb_gap: float
    nodes_explored: int
    runtime_seconds: float
    pruned_by_bound: bool
    infeasible_capacity: bool
    root_lp_status: str = ""
    incumbent_after_theta: float = float("-inf")
    bnb_lower_bound: float = float("-inf")
    nodes_pruned_bound: int = 0
    nodes_pruned_infeasible: int = 0
    cutoff_used: bool = False
    initial_incumbent_feasible_for_theta: bool = False
    robust_certificate_passed: bool = False
    root_lp_runtime_seconds: float = 0.0
    diagnostics: Dict[str, object] = field(default_factory=dict)


@dataclass
class GlobalThetaBNBResult:
    """Certified or anytime result for global theta-enumerated robust solving."""

    status: str
    objective_value: float
    selected_options: Optional[List[int]]
    selected_theta: Optional[float]
    robust_certificate: float
    lower_bound: float
    upper_bound: float
    absolute_gap: float
    relative_gap: float
    theta_count_total: int
    theta_count_infeasible_capacity: int
    theta_count_pruned_by_bound: int
    theta_count_solved_optimal: int
    theta_count_limited: int
    theta_count_error: int
    total_nodes_explored: int
    total_runtime_seconds: float
    per_theta_records: List[GlobalThetaRecord]
    total_root_lp_time_seconds: float = 0.0
    total_fixed_theta_bnb_time_seconds: float = 0.0
    diagnostics: Dict[str, object] = field(default_factory=dict)
    config_metadata: Dict[str, object] = field(default_factory=dict)
    validation_flags: Dict[str, bool] = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True)
class FixedThetaData:
    """Fixed-theta arrays used by the exact solver."""

    values: List[np.ndarray]
    s_theta: List[np.ndarray]
    costs: List[np.ndarray]
    capacity: float
    s_star_sum: float
    baseline_indices: List[int]
    exact_costs: Tuple[Tuple[int, ...], ...] = field(default_factory=tuple)
    exact_values: Tuple[Tuple[Fraction, ...], ...] = field(default_factory=tuple)
    exact_capacity: Optional[int] = None
    exact_scale_exponent: int = 0


@dataclass(frozen=True)
class CachedHullSegment:
    """Precomputed hull segment used by the fast residual LP bound."""

    item: int
    index: int
    slope: float
    length: float


@dataclass(frozen=True)
class FixedThetaCache:
    """Exact-safe per-theta cache for fixed-theta B&B and LP bounds."""

    theta: float
    data: FixedThetaData
    option_sets: List[List[int]]
    min_cost: List[float]
    per_item_hulls: List[Hull]
    hull_base_cost: List[float] = field(default_factory=list)
    hull_base_value: List[float] = field(default_factory=list)
    max_value: List[float] = field(default_factory=list)
    global_segments: Tuple[CachedHullSegment, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class _Node:
    fixed: Tuple[int, ...]
    used_cost: float
    fixed_value: float
    upper_bound: float
    depth: int
    tie_key: Tuple[int, ...]


@dataclass
class _BoundInfo:
    upper_bound: float
    lp_solution: Optional[LPSolution]
    hulls: List[Hull]
    free_items: List[int]
    infeasible: bool


def _diag_add(diagnostics: Optional[Dict[str, object]], key: str, value: float) -> None:
    if diagnostics is None:
        return
    diagnostics[key] = float(diagnostics.get(key, 0.0)) + float(value)


def _diag_inc(diagnostics: Optional[Dict[str, object]], key: str, value: int = 1) -> None:
    if diagnostics is None:
        return
    diagnostics[key] = int(diagnostics.get(key, 0)) + int(value)


def _diag_append_sample(
    diagnostics: Optional[Dict[str, object]],
    key: str,
    value: object,
    *,
    max_samples: int,
    sample_rate: int,
    sample_index: int,
) -> None:
    if diagnostics is None or max_samples <= 0:
        return
    rate = max(1, int(sample_rate))
    if sample_index % rate != 0:
        return
    samples = diagnostics.setdefault(key, [])
    if isinstance(samples, list) and len(samples) < max_samples:
        samples.append(value)


def _finite_quantiles(values: Sequence[float]) -> Dict[str, float]:
    finite_values = [float(v) for v in values if math.isfinite(float(v))]
    if not finite_values:
        return {}
    arr = np.array(finite_values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def _fraction_to_float_down(value: Fraction) -> float:
    try:
        rounded = float(value)
    except OverflowError:
        return -math.inf if value < 0 else float(np.finfo(float).max)
    if not math.isfinite(rounded):
        return -math.inf if value < 0 else float(np.finfo(float).max)
    if Fraction.from_float(rounded) > value:
        rounded = float(np.nextafter(rounded, -math.inf))
    return rounded


def _fraction_to_float_up(value: Fraction) -> float:
    try:
        rounded = float(value)
    except OverflowError:
        return -float(np.finfo(float).max) if value < 0 else math.inf
    if not math.isfinite(rounded):
        return -float(np.finfo(float).max) if value < 0 else math.inf
    if Fraction.from_float(rounded) < value:
        rounded = float(np.nextafter(rounded, math.inf))
    return rounded


def _require_representable_objective_range(instance: PricingInstance) -> None:
    """Reject inputs whose aggregate objective cannot be represented safely."""

    lower = sum(
        (
            min(Fraction.from_float(float(option.value)) for option in group)
            for group in instance.items
        ),
        Fraction(0),
    )
    upper = sum(
        (
            max(Fraction.from_float(float(option.value)) for option in group)
            for group in instance.items
        ),
        Fraction(0),
    )
    maximum = Fraction.from_float(float(np.finfo(float).max))
    if lower < -maximum or upper > maximum:
        raise ValueError(
            "aggregate objective range exceeds binary64; rescale objective "
            "coefficients before using the certified solver"
        )


def _power_of_two_exponent(denominator: int) -> int:
    exponent = denominator.bit_length() - 1
    if denominator != 1 << exponent:
        raise ValueError("binary64 rational denominator is not a power of two")
    return exponent


def build_fixed_theta_data(instance: PricingInstance, theta: float) -> FixedThetaData:
    """Build fixed-theta values, costs, and capacity for a PricingInstance."""

    values: List[np.ndarray] = []
    s_theta_values: List[np.ndarray] = []
    costs: List[np.ndarray] = []
    baseline_indices: List[int] = []
    exact_theta = Fraction.from_float(float(theta))
    exact_group_costs: List[List[Fraction]] = []
    exact_stars: List[Fraction] = []
    for group in instance.items:
        v = np.array([opt.value for opt in group], dtype=float)
        exact_slacks = [
            Fraction.from_float(float(opt.margin))
            - max(
                Fraction.from_float(abs(float(opt.uncertainty))) - exact_theta,
                Fraction(0),
            )
            for opt in group
        ]
        exact_star = max(exact_slacks)
        baseline_idx = exact_slacks.index(exact_star)
        exact_cost_group = [exact_star - slack for slack in exact_slacks]
        s_theta = np.asarray([float(slack) for slack in exact_slacks], dtype=float)
        values.append(v)
        s_theta_values.append(s_theta)
        costs.append(
            np.asarray(
                [_fraction_to_float_down(cost) for cost in exact_cost_group],
                dtype=float,
            )
        )
        baseline_indices.append(baseline_idx)
        exact_group_costs.append(exact_cost_group)
        exact_stars.append(exact_star)

    exact_capacity_fraction = sum(exact_stars, Fraction(0)) - int(
        instance.gamma
    ) * exact_theta
    scale_exponent = max(
        [_power_of_two_exponent(exact_capacity_fraction.denominator)]
        + [
            _power_of_two_exponent(cost.denominator)
            for group in exact_group_costs
            for cost in group
        ]
    )

    def scaled_integer(value: Fraction) -> int:
        exponent = _power_of_two_exponent(value.denominator)
        return int(value.numerator) << (scale_exponent - exponent)

    exact_costs = tuple(
        tuple(scaled_integer(cost) for cost in group)
        for group in exact_group_costs
    )
    exact_capacity = scaled_integer(exact_capacity_fraction)
    exact_values = tuple(
        tuple(Fraction.from_float(float(option.value)) for option in group)
        for group in instance.items
    )

    # The floating model is an outward relaxation used only for LP bounds and
    # branching.  Exact scaled-integer comparisons decide feasibility.  The
    # padding covers worst-case summation of one nonnegative cost per group.
    maximum_cost_sum = sum(
        (max(group, default=Fraction(0)) for group in exact_group_costs),
        Fraction(0),
    )
    operation_count = max(1, len(exact_group_costs) + 2)
    epsilon = math.ulp(1.0)
    gamma = (operation_count * epsilon) / max(
        1.0 - operation_count * epsilon, epsilon
    )
    maximum_cost_float = _fraction_to_float_up(maximum_cost_sum)
    if math.isfinite(maximum_cost_float):
        padding = float(np.nextafter(gamma * maximum_cost_float, math.inf))
        capacity = _fraction_to_float_up(
            exact_capacity_fraction + Fraction.from_float(padding)
        )
    else:
        capacity = math.inf
    exact_star_sum = sum(exact_stars, Fraction(0))
    try:
        s_star_sum = float(exact_star_sum)
    except OverflowError:
        s_star_sum = -math.inf if exact_star_sum < 0 else math.inf
    return FixedThetaData(
        values=values,
        s_theta=s_theta_values,
        costs=costs,
        capacity=float(capacity),
        s_star_sum=s_star_sum,
        baseline_indices=baseline_indices,
        exact_costs=exact_costs,
        exact_values=exact_values,
        exact_capacity=exact_capacity,
        exact_scale_exponent=scale_exponent,
    )


def _fixed_theta_selection_is_feasible(
    data: FixedThetaData, selections: Sequence[int]
) -> bool:
    if len(selections) != len(data.costs):
        return False
    if not data.exact_costs or data.exact_capacity is None:
        try:
            used = sum(
                (
                    Fraction.from_float(float(data.costs[i][int(option)]))
                    for i, option in enumerate(selections)
                ),
                Fraction(0),
            )
        except (IndexError, TypeError, ValueError):
            return False
        return used <= Fraction.from_float(float(data.capacity))
    try:
        used = sum(
            data.exact_costs[i][int(option)]
            for i, option in enumerate(selections)
        )
    except (IndexError, TypeError, ValueError):
        return False
    return used <= data.exact_capacity


def _fixed_theta_partial_is_feasible(
    data: FixedThetaData,
    fixed: Sequence[int],
    option_sets: Sequence[Sequence[int]],
) -> bool:
    if not data.exact_costs or data.exact_capacity is None:
        used = Fraction(0)
        capacity = Fraction.from_float(float(data.capacity))
        for i, option in enumerate(fixed):
            if int(option) >= 0:
                used += Fraction.from_float(float(data.costs[i][int(option)]))
            else:
                options = option_sets[i]
                if not options:
                    return False
                used += min(
                    Fraction.from_float(float(data.costs[i][int(j)]))
                    for j in options
                )
            if used > capacity:
                return False
        return used <= capacity
    used = 0
    for i, option in enumerate(fixed):
        if int(option) >= 0:
            used += data.exact_costs[i][int(option)]
        else:
            options = option_sets[i]
            if not options:
                return False
            used += min(data.exact_costs[i][int(j)] for j in options)
        if used > data.exact_capacity:
            return False
    return used <= data.exact_capacity


def _fixed_theta_capacity_is_negative(data: FixedThetaData) -> bool:
    if data.exact_capacity is not None:
        return data.exact_capacity < 0
    return float(data.capacity) < 0.0


def _exact_upper_hull_indices(
    costs: Sequence[int], values: Sequence[Fraction], options: Sequence[int]
) -> List[int]:
    """Return the exact upper hull over a subset of original options."""

    best_at_cost: Dict[int, int] = {}
    for option in options:
        option = int(option)
        cost = int(costs[option])
        incumbent = best_at_cost.get(cost)
        if incumbent is None or values[option] > values[incumbent]:
            best_at_cost[cost] = option
    ordered = [best_at_cost[cost] for cost in sorted(best_at_cost)]
    nondominated: List[int] = []
    best_value: Optional[Fraction] = None
    for option in ordered:
        if best_value is None or values[option] > best_value:
            nondominated.append(option)
            best_value = values[option]
    hull: List[int] = []
    for option in nondominated:
        while len(hull) >= 2:
            left, middle = hull[-2], hull[-1]
            left_slope = (values[middle] - values[left]) / (
                int(costs[middle]) - int(costs[left])
            )
            right_slope = (values[option] - values[middle]) / (
                int(costs[option]) - int(costs[middle])
            )
            if left_slope > right_slope:
                break
            hull.pop()
        hull.append(option)
    return hull


def _exact_fixed_theta_lp_value(
    data: FixedThetaData,
    fixed: Sequence[int],
    option_sets: Sequence[Sequence[int]],
) -> Optional[Fraction]:
    """Return the exact LP value for a fixed/partially fixed theta node."""

    if (
        not data.exact_costs
        or not data.exact_values
        or data.exact_capacity is None
    ):
        return None
    used_cost = 0
    value = Fraction(0)
    segments: List[Tuple[Fraction, int]] = []
    for item, selected in enumerate(fixed):
        if int(selected) >= 0:
            option = int(selected)
            used_cost += int(data.exact_costs[item][option])
            value += data.exact_values[item][option]
            continue
        hull = _exact_upper_hull_indices(
            data.exact_costs[item], data.exact_values[item], option_sets[item]
        )
        if not hull:
            return None
        base = hull[0]
        used_cost += int(data.exact_costs[item][base])
        value += data.exact_values[item][base]
        for lower, upper in zip(hull[:-1], hull[1:]):
            length = int(data.exact_costs[item][upper]) - int(
                data.exact_costs[item][lower]
            )
            if length <= 0:
                continue
            change = data.exact_values[item][upper] - data.exact_values[item][lower]
            if change > 0:
                segments.append((change / length, length))
    residual = int(data.exact_capacity) - used_cost
    if residual < 0:
        return None
    for slope, length in sorted(segments, key=lambda row: row[0], reverse=True):
        if residual <= 0:
            break
        take = min(length, residual)
        value += take * slope
        residual -= take
    return value


def _build_fixed_theta_cache(
    instance: PricingInstance,
    theta: float,
    tol: float,
    data: Optional[FixedThetaData] = None,
) -> FixedThetaCache:
    """Build exact-safe cached structures for one fixed theta."""

    fixed_data = data if data is not None else build_fixed_theta_data(instance, theta)
    if fixed_data.exact_costs:
        option_sets = [
            nondominated_option_indices_exact(c, v)
            for v, c in zip(fixed_data.values, fixed_data.exact_costs)
        ]
    else:
        option_sets = [
            nondominated_option_indices(c, v, tol)
            for v, c in zip(fixed_data.values, fixed_data.costs)
        ]
    min_cost = [
        float(min(float(fixed_data.costs[i][j]) for j in opts)) if opts else float("inf")
        for i, opts in enumerate(option_sets)
    ]
    per_item_hulls: List[Hull] = []
    hull_base_cost: List[float] = []
    hull_base_value: List[float] = []
    max_value: List[float] = []
    global_segments: List[CachedHullSegment] = []
    for i, opts in enumerate(option_sets):
        idx = np.array(opts, dtype=int)
        if idx.size == 0:
            hull = build_upper_hull(np.array([], dtype=float), np.array([], dtype=float), idx)
        else:
            hull = build_upper_hull(fixed_data.costs[i][idx], fixed_data.values[i][idx], idx)
        per_item_hulls.append(hull)
        hull_base_cost.append(float(hull.costs[0]) if hull.costs.size else float("inf"))
        hull_base_value.append(float(hull.values[0]) if hull.values.size else float("-inf"))
        max_value.append(float(max(float(fixed_data.values[i][j]) for j in opts)) if opts else float("-inf"))
        for k, (slope, length) in enumerate(zip(hull.slopes, hull.delta_costs)):
            if float(length) > 0.0:
                global_segments.append(CachedHullSegment(item=i, index=k, slope=float(slope), length=float(length)))
    global_segments.sort(key=lambda seg: (-seg.slope, int(seg.item), int(seg.index)))
    return FixedThetaCache(
        theta=float(theta),
        data=fixed_data,
        option_sets=option_sets,
        min_cost=min_cost,
        per_item_hulls=per_item_hulls,
        hull_base_cost=hull_base_cost,
        hull_base_value=hull_base_value,
        max_value=max_value,
        global_segments=tuple(global_segments),
    )


def build_full_theta_candidates(
    instance: PricingInstance,
    tol: float | None = None,
) -> List[float]:
    """Build the exact binary64-distinct theta set from original options.

    ``tol`` is retained for API compatibility but intentionally does not merge
    nearby deviations. Tolerance clustering can remove the only feasible
    breakpoint and is therefore unsafe for an exact finite-union solver.
    """

    del tol
    return sorted(
        {
            0.0,
            *(
                abs(float(opt.uncertainty))
                for group in instance.items
                for opt in group
            ),
        }
    )


def compute_fixed_theta_lp_upper_bound(
    instance: PricingInstance,
    theta: float,
    tol: float = EPS,
    cache: Optional[FixedThetaCache] = None,
) -> FixedThetaLPBoundResult:
    """Compute a valid root LP upper bound for a fixed-theta MCKP."""

    _require_representable_objective_range(instance)
    start = time.perf_counter()
    try:
        theta_cache = cache if cache is not None else _build_fixed_theta_cache(instance, theta, tol)
        data = theta_cache.data
        if _fixed_theta_capacity_is_negative(data):
            return FixedThetaLPBoundResult(
                theta=float(theta),
                capacity=data.capacity,
                lp_upper_bound=float("-inf"),
                infeasible_capacity=True,
                lp_feasible=False,
                root_lp_status="infeasible_capacity",
                runtime_seconds=float(time.perf_counter() - start),
                message="fixed-theta capacity is negative",
            )
        exact_lp_value = _exact_fixed_theta_lp_value(
            data,
            tuple(-1 for _ in range(instance.n_items)),
            theta_cache.option_sets,
        )
        if exact_lp_value is None:
            return FixedThetaLPBoundResult(
                theta=float(theta),
                capacity=data.capacity,
                lp_upper_bound=float("-inf"),
                infeasible_capacity=False,
                lp_feasible=False,
                root_lp_status="infeasible",
                runtime_seconds=float(time.perf_counter() - start),
                message="exact fixed-threshold LP baseline exceeds capacity",
            )
        exact_lp_upper = _fraction_to_float_up(exact_lp_value)
        hulls = theta_cache.per_item_hulls
        if any(h.costs.size == 0 for h in hulls):
            return FixedThetaLPBoundResult(
                theta=float(theta),
                capacity=data.capacity,
                lp_upper_bound=float(exact_lp_upper),
                infeasible_capacity=False,
                lp_feasible=True,
                root_lp_status="safe_value_fallback",
                runtime_seconds=float(time.perf_counter() - start),
                message="empty numerical hull; using trivial value upper bound",
            )
        lp = greedy_lp(hulls, data.capacity)
        if lp.total_cost > data.capacity + max(1.0, abs(data.capacity)) * tol:
            return FixedThetaLPBoundResult(
                theta=float(theta),
                capacity=data.capacity,
                lp_upper_bound=float(exact_lp_upper),
                infeasible_capacity=False,
                lp_feasible=True,
                root_lp_status="safe_value_fallback",
                fractional_item=lp.fractional_item,
                runtime_seconds=float(time.perf_counter() - start),
                message="numerical LP baseline check failed; using trivial value upper bound",
            )
        return FixedThetaLPBoundResult(
            theta=float(theta),
            capacity=data.capacity,
            lp_upper_bound=float(exact_lp_upper),
            infeasible_capacity=False,
            lp_feasible=True,
            root_lp_status="optimal",
            fractional_item=lp.fractional_item,
            runtime_seconds=float(time.perf_counter() - start),
        )
    except Exception as exc:
        return FixedThetaLPBoundResult(
            theta=float(theta),
            capacity=float("nan"),
            lp_upper_bound=float("inf"),
            infeasible_capacity=False,
            lp_feasible=True,
            root_lp_status="error",
            runtime_seconds=float(time.perf_counter() - start),
            message=str(exc),
        )


def _dominates(cost_a: float, value_a: float, cost_b: float, value_b: float, tol: float) -> bool:
    del tol
    no_worse = cost_a <= cost_b and value_a >= value_b
    strict = cost_a < cost_b or value_a > value_b
    return bool(no_worse and strict)


def nondominated_option_indices(costs: np.ndarray, values: np.ndarray, tol: float = EPS) -> List[int]:
    """Return original option indices after only integer-safe dominance pruning.

    This deliberately does not remove below-upper-hull options. An option is
    removed only when another original option for the same item has no larger
    cost and no smaller value, with at least one strict improvement. Exact ties
    in both cost and value are represented by the smallest original index.
    """

    keep: List[int] = []
    del tol
    for j in range(len(values)):
        duplicate_smaller_index = False
        for k in range(j):
            if float(costs[k]) == float(costs[j]) and float(values[k]) == float(values[j]):
                duplicate_smaller_index = True
                break
        if duplicate_smaller_index:
            continue

        dominated = False
        for k in range(len(values)):
            if k == j:
                continue
            if _dominates(float(costs[k]), float(values[k]), float(costs[j]), float(values[j]), 0.0):
                dominated = True
                break
        if not dominated:
            keep.append(j)
    return keep


def nondominated_option_indices_exact(
    costs: Sequence[int], values: np.ndarray
) -> List[int]:
    """Return integer-safe nondominated indices using exact scaled costs.

    The relaxed binary64 costs used by LP bounds can collapse two distinct
    exact costs. They must therefore never decide which original integer
    options survive branch-and-bound preprocessing.
    """

    keep: List[int] = []
    for j in range(len(values)):
        duplicate_smaller_index = any(
            int(costs[k]) == int(costs[j])
            and float(values[k]) == float(values[j])
            for k in range(j)
        )
        if duplicate_smaller_index:
            continue
        dominated = any(
            k != j
            and int(costs[k]) <= int(costs[j])
            and float(values[k]) >= float(values[j])
            and (
                int(costs[k]) < int(costs[j])
                or float(values[k]) > float(values[j])
            )
            for k in range(len(values))
        )
        if not dominated:
            keep.append(j)
    return keep


def validate_fixed_theta_selection(
    values: Sequence[np.ndarray],
    costs: Sequence[np.ndarray],
    capacity: float,
    selections: Sequence[int],
    tol: float = EPS,
) -> Dict[str, bool]:
    """Validate a fixed-theta MCKP selection against original options."""

    valid_length = len(selections) == len(values)
    valid_indices = valid_length and all(0 <= int(j) < len(values[i]) for i, j in enumerate(selections))
    if not valid_indices:
        return {
            "valid_length": bool(valid_length),
            "valid_indices": False,
            "capacity_feasible": False,
            "objective_recomputed": False,
        }
    used = float(sum(float(costs[i][int(j)]) for i, j in enumerate(selections)))
    return {
        "valid_length": True,
        "valid_indices": True,
        "capacity_feasible": used <= float(capacity) + tol,
        "objective_recomputed": True,
    }


def _selection_objective_fraction(
    values: Sequence[np.ndarray], selections: Sequence[int]
) -> Fraction:
    return sum(
        (
            Fraction.from_float(float(values[i][int(j)]))
            for i, j in enumerate(selections)
        ),
        Fraction(0),
    )


def objective_for_selection(values: Sequence[np.ndarray], selections: Sequence[int]) -> float:
    return float(_selection_objective_fraction(values, selections))


def cost_for_selection(costs: Sequence[np.ndarray], selections: Sequence[int]) -> float:
    try:
        return float(
            math.fsum(float(costs[i][int(j)]) for i, j in enumerate(selections))
        )
    except OverflowError:
        return math.inf


def _relative_gap(upper: float, lower: float, tol: float) -> float:
    if not math.isfinite(upper) or not math.isfinite(lower):
        return float("inf")
    gap = max(0.0, upper - lower)
    denom = max(abs(upper), abs(lower), tol)
    return float(gap / denom)


def _make_result(
    *,
    status: str,
    theta: float,
    capacity: float,
    incumbent_value: float,
    incumbent_selection: Optional[List[int]],
    incumbent_cost: float,
    upper_bound: float,
    nodes_explored: int,
    nodes_pruned_bound: int,
    nodes_pruned_infeasible: int,
    nodes_integral: int,
    start_time: float,
    config: FixedThetaBNBConfig,
    values: Sequence[np.ndarray],
    costs: Sequence[np.ndarray],
    diagnostics: Optional[Dict[str, object]] = None,
    message: str = "",
) -> FixedThetaBNBResult:
    runtime = time.perf_counter() - start_time
    if status == "optimal" and incumbent_selection is not None:
        upper_bound = incumbent_value
    lower_bound = incumbent_value if incumbent_selection is not None else float("-inf")
    abs_gap = 0.0 if status == "optimal" and incumbent_selection is not None else max(0.0, upper_bound - lower_bound)
    rel_gap = 0.0 if abs_gap <= config.tolerance else _relative_gap(upper_bound, lower_bound, config.tolerance)
    residual = float(capacity - incumbent_cost) if incumbent_selection is not None else float("nan")
    if diagnostics is not None:
        diagnostics.setdefault("final_incumbent_value", float(incumbent_value))
        diagnostics.setdefault("final_upper_bound", float(upper_bound))
        diagnostics.setdefault("absolute_gap", float(abs_gap))
        diagnostics.setdefault("relative_gap", float(rel_gap))
        diagnostics.setdefault("best_remaining_node_bound", float("-inf"))
        diagnostics.setdefault("profile_timing", bool(config.profile_timing))
    flags = (
        validate_fixed_theta_selection(values, costs, capacity, incumbent_selection, config.tolerance)
        if incumbent_selection is not None
        else {
            "valid_length": False,
            "valid_indices": False,
            "capacity_feasible": False,
            "objective_recomputed": False,
        }
    )
    if incumbent_selection is not None:
        recomputed_obj = objective_for_selection(values, incumbent_selection)
        flags["objective_matches"] = abs(recomputed_obj - incumbent_value) <= config.tolerance
    else:
        flags["objective_matches"] = False

    return FixedThetaBNBResult(
        status=status,
        theta=float(theta),
        objective_value=float(incumbent_value) if incumbent_selection is not None else float("-inf"),
        selected_options=incumbent_selection,
        used_capacity=float(incumbent_cost) if incumbent_selection is not None else float("inf"),
        capacity=float(capacity),
        fixed_theta_residual=residual,
        upper_bound=float(upper_bound),
        lower_bound=float(lower_bound),
        absolute_gap=float(abs_gap),
        relative_gap=float(rel_gap),
        nodes_explored=int(nodes_explored),
        nodes_pruned_bound=int(nodes_pruned_bound),
        nodes_pruned_infeasible=int(nodes_pruned_infeasible),
        nodes_integral=int(nodes_integral),
        runtime_seconds=float(runtime),
        config_metadata={
            "tolerance": config.tolerance,
            "time_limit_seconds": config.time_limit_seconds,
            "node_limit": config.node_limit,
            "node_selection": config.node_selection,
            "branch_strategy": config.branch_strategy,
            "child_order": config.child_order,
            "branching_rule": config.branching_rule,
            "child_ordering": config.child_ordering,
            "strong_branching_candidates": config.strong_branching_candidates,
            "strong_branching_depth_limit": config.strong_branching_depth_limit,
            "strong_branching_max_children": config.strong_branching_max_children,
            "strong_branching_use_fast_bound": config.strong_branching_use_fast_bound,
            "incumbent_heuristic": config.incumbent_heuristic,
            "use_local_incumbent_improvement": config.use_local_incumbent_improvement,
            "local_search_max_passes": config.local_search_max_passes,
            "local_search_max_swaps": config.local_search_max_swaps,
            "local_search_neighborhood": config.local_search_neighborhood,
            "local_search_max_pair_evaluations": config.local_search_max_pair_evaluations,
            "use_greedy_incumbent": config.use_greedy_incumbent,
            "objective_cutoff": config.objective_cutoff,
            "initial_incumbent_value": config.initial_incumbent_value,
            "use_cutoff_pruning": config.use_cutoff_pruning,
            "use_cache": config.use_cache,
            "collect_diagnostics": config.collect_diagnostics,
            "max_diagnostic_nodes": config.max_diagnostic_nodes,
            "diagnostic_sample_rate": config.diagnostic_sample_rate,
            "profile_timing": config.profile_timing,
            "use_fast_residual_lp_bound": config.use_fast_residual_lp_bound,
            "use_cheap_prebound": config.use_cheap_prebound,
            "use_min_cost_infeasibility_check": config.use_min_cost_infeasibility_check,
            "use_bound_cache": config.use_bound_cache,
            "bound_cache_max_entries": config.bound_cache_max_entries,
        },
        validation_flags=flags,
        diagnostics=diagnostics or {},
        message=message,
    )


def _min_cost_indices(values: Sequence[np.ndarray], costs: Sequence[np.ndarray], option_sets: Sequence[List[int]], tol: float) -> List[int]:
    choices: List[int] = []
    for i, opts in enumerate(option_sets):
        best = min(opts, key=lambda j: (float(costs[i][j]), -float(values[i][j]), int(j)))
        choices.append(int(best))
    return choices


def _greedy_incumbent(
    values: Sequence[np.ndarray],
    costs: Sequence[np.ndarray],
    option_sets: Sequence[List[int]],
    capacity: float,
    tol: float,
) -> Tuple[Optional[List[int]], float, float]:
    """Construct a deterministic feasible incumbent over original options."""

    selections = _min_cost_indices(values, costs, option_sets, tol)
    used = cost_for_selection(costs, selections)
    if used > capacity + tol:
        return None, float("-inf"), float("inf")

    value = objective_for_selection(values, selections)
    while True:
        residual = capacity - used
        best_move: Optional[Tuple[float, float, float, int, int]] = None
        for i, opts in enumerate(option_sets):
            cur = selections[i]
            cur_cost = float(costs[i][cur])
            cur_value = float(values[i][cur])
            for j in opts:
                if j == cur:
                    continue
                dc = float(costs[i][j]) - cur_cost
                dv = float(values[i][j]) - cur_value
                if dc < -tol:
                    # A lower-cost improving move should have been selected by dominance/min-cost,
                    # but allow it for robustness.
                    ratio = float("inf") if dv > tol else 0.0
                elif dc <= tol:
                    ratio = float("inf") if dv > tol else 0.0
                else:
                    if dc > residual + tol:
                        continue
                    ratio = dv / dc
                if dv <= tol:
                    continue
                candidate = (ratio, dv, -dc, -i, -int(j))
                if best_move is None or candidate > best_move:
                    best_move = candidate
        if best_move is None:
            break
        _, dv, neg_dc, neg_i, neg_j = best_move
        i = -neg_i
        j = -neg_j
        dc = -neg_dc
        selections[i] = int(j)
        used += float(dc)
        value += float(dv)
    return (
        selections,
        objective_for_selection(values, selections),
        cost_for_selection(costs, selections),
    )


def _local_improve_incumbent(
    values: Sequence[np.ndarray],
    costs: Sequence[np.ndarray],
    option_sets: Sequence[List[int]],
    capacity: float,
    selections: Sequence[int],
    tol: float,
    *,
    max_passes: int = 2,
    max_swaps: int = 1000,
    neighborhood: str = "single",
    max_pair_evaluations: int = 20000,
) -> Tuple[List[int], float, float, int, int]:
    """Deterministic local improvement for a fixed-theta incumbent.

    The optional two-item neighborhood is exact-safe because it only replaces
    the incumbent by a strictly higher-value selection that is explicitly
    checked against the same fixed-theta capacity. Global robust incumbents are
    still validated separately against the original robust certificate.
    """

    improved = list(map(int, selections))
    used = cost_for_selection(costs, improved)
    value = objective_for_selection(values, improved)
    if used > capacity + tol:
        return improved, value, used, 0, 0
    swaps = 0
    pair_evaluations = 0
    allow_two_item = str(neighborhood).lower() in {"single_two", "two_item", "two-item", "pair"}
    for _pass in range(max(0, int(max_passes))):
        best_score: Optional[Tuple[float, ...]] = None
        best_move: Optional[Tuple[object, ...]] = None
        for i, opts in enumerate(option_sets):
            cur = improved[i]
            cur_cost = float(costs[i][cur])
            cur_value = float(values[i][cur])
            for j in opts:
                if int(j) == cur:
                    continue
                new_cost = used - cur_cost + float(costs[i][j])
                if new_cost > capacity + tol:
                    continue
                dv = float(values[i][j]) - cur_value
                if dv <= tol:
                    continue
                dc = float(costs[i][j]) - cur_cost
                # Prefer larger value gain, then lower cost increase, then stable item/option order.
                score = (float(dv), -max(0.0, float(dc)), -abs(float(dc)), -float(i), -float(j))
                if best_score is None or score > best_score:
                    best_score = score
                    best_move = ("single", int(i), int(j))
        if allow_two_item and pair_evaluations < max(0, int(max_pair_evaluations)):
            n_items = len(option_sets)
            stop_pairs = False
            for i in range(n_items):
                cur_i = improved[i]
                cur_cost_i = float(costs[i][cur_i])
                cur_value_i = float(values[i][cur_i])
                for k in range(i + 1, n_items):
                    cur_k = improved[k]
                    cur_cost_k = float(costs[k][cur_k])
                    cur_value_k = float(values[k][cur_k])
                    for j in option_sets[i]:
                        if int(j) == cur_i:
                            continue
                        for ell in option_sets[k]:
                            if int(ell) == cur_k:
                                continue
                            pair_evaluations += 1
                            new_cost = (
                                used
                                - cur_cost_i
                                - cur_cost_k
                                + float(costs[i][j])
                                + float(costs[k][ell])
                            )
                            if new_cost > capacity + tol:
                                if pair_evaluations >= max(0, int(max_pair_evaluations)):
                                    stop_pairs = True
                                    break
                                continue
                            dv = (
                                float(values[i][j])
                                - cur_value_i
                                + float(values[k][ell])
                                - cur_value_k
                            )
                            if dv <= tol:
                                if pair_evaluations >= max(0, int(max_pair_evaluations)):
                                    stop_pairs = True
                                    break
                                continue
                            dc = new_cost - used
                            score = (
                                float(dv),
                                -max(0.0, float(dc)),
                                -abs(float(dc)),
                                -float(i),
                                -float(j),
                                -float(k),
                                -float(ell),
                            )
                            if best_score is None or score > best_score:
                                best_score = score
                                best_move = ("pair", int(i), int(j), int(k), int(ell))
                            if pair_evaluations >= max(0, int(max_pair_evaluations)):
                                stop_pairs = True
                                break
                        if stop_pairs:
                            break
                    if stop_pairs:
                        break
                if stop_pairs:
                    break
        if best_move is None or swaps >= max(0, int(max_swaps)):
            break
        move_type = str(best_move[0])
        if move_type == "pair":
            i = int(best_move[1])
            j = int(best_move[2])
            k = int(best_move[3])
            ell = int(best_move[4])
            used = (
                used
                - float(costs[i][improved[i]])
                - float(costs[k][improved[k]])
                + float(costs[i][j])
                + float(costs[k][ell])
            )
            value = (
                value
                - float(values[i][improved[i]])
                - float(values[k][improved[k]])
                + float(values[i][j])
                + float(values[k][ell])
            )
            improved[i] = int(j)
            improved[k] = int(ell)
        else:
            i = int(best_move[1])
            j = int(best_move[2])
            used = used - float(costs[i][improved[i]]) + float(costs[i][j])
            value = value - float(values[i][improved[i]]) + float(values[i][j])
            improved[i] = int(j)
        swaps += 1
    return (
        improved,
        objective_for_selection(values, improved),
        cost_for_selection(costs, improved),
        int(swaps),
        int(pair_evaluations),
    )


def _build_free_hulls(
    values: Sequence[np.ndarray],
    costs: Sequence[np.ndarray],
    option_sets: Sequence[List[int]],
    free_items: Sequence[int],
) -> List[Hull]:
    hulls: List[Hull] = []
    for i in free_items:
        idx = np.array(option_sets[i], dtype=int)
        hulls.append(build_upper_hull(costs[i][idx], values[i][idx], idx))
    return hulls


def _position_from_hull_cost(hull: Hull, cost: float) -> ItemLPPosition:
    costs = hull.costs
    values = hull.values
    if cost <= costs[0]:
        return ItemLPPosition(lower_vertex=0, upper_vertex=0, lambda_=0.0, cost=float(costs[0]), value=float(values[0]))
    if cost >= costs[-1]:
        last = len(costs) - 1
        return ItemLPPosition(lower_vertex=last, upper_vertex=last, lambda_=0.0, cost=float(costs[-1]), value=float(values[-1]))
    k = int(np.searchsorted(costs, cost, side="right") - 1)
    k = max(0, min(k, len(costs) - 2))
    c0 = float(costs[k])
    c1 = float(costs[k + 1])
    lam = 0.0 if c1 == c0 else (float(cost) - c0) / (c1 - c0)
    lam = float(min(1.0, max(0.0, lam)))
    v0 = float(values[k])
    v1 = float(values[k + 1])
    return ItemLPPosition(
        lower_vertex=k,
        upper_vertex=k + 1,
        lambda_=lam,
        cost=float(cost),
        value=float(v0 + lam * (v1 - v0)),
    )


def _compute_bound_reference(
    node: _Node,
    values: Sequence[np.ndarray],
    costs: Sequence[np.ndarray],
    option_sets: Sequence[List[int]],
    suffix_min_cost: Sequence[float],
    capacity: float,
    tol: float,
    per_item_hulls: Optional[Sequence[Hull]] = None,
    diagnostics: Optional[Dict[str, object]] = None,
    profile_timing: bool = False,
) -> _BoundInfo:
    ref_start = time.perf_counter() if profile_timing else 0.0
    fixed = node.fixed
    free_items = [i for i, j in enumerate(fixed) if j < 0]
    remaining_capacity = capacity - node.used_cost
    if remaining_capacity < -tol:
        return _BoundInfo(float("-inf"), None, [], free_items, True)
    min_remaining = sum(float(suffix_min_cost[i]) for i in free_items)
    if remaining_capacity + tol < min_remaining:
        return _BoundInfo(float("-inf"), None, [], free_items, True)
    if not free_items:
        return _BoundInfo(float(node.fixed_value), None, [], free_items, False)

    if per_item_hulls is not None:
        hulls = [per_item_hulls[i] for i in free_items]
    else:
        hull_start = time.perf_counter() if profile_timing else 0.0
        hulls = _build_free_hulls(values, costs, option_sets, free_items)
        if profile_timing:
            _diag_add(diagnostics, "time_hull_build_total", time.perf_counter() - hull_start)
    if any(h.costs.size == 0 for h in hulls):
        return _BoundInfo(float("-inf"), None, hulls, free_items, True)
    greedy_start = time.perf_counter() if profile_timing else 0.0
    lp = greedy_lp(hulls, remaining_capacity)
    if profile_timing:
        _diag_add(diagnostics, "time_greedy_lp_total", time.perf_counter() - greedy_start)
    if lp.total_cost > remaining_capacity + max(1.0, abs(remaining_capacity)) * tol:
        result = _BoundInfo(float("-inf"), lp, hulls, free_items, True)
    else:
        result = _BoundInfo(float(node.fixed_value + lp.lp_value), lp, hulls, free_items, False)
    if profile_timing:
        _diag_add(diagnostics, "time_reference_bound", time.perf_counter() - ref_start)
    return result


def _compute_bound_fast(
    node: _Node,
    theta_cache: FixedThetaCache,
    capacity: float,
    tol: float,
    *,
    incumbent_cutoff: float = float("-inf"),
    need_solution: bool = True,
    use_cheap_prebound: bool = True,
    use_min_cost_infeasibility_check: bool = True,
    diagnostics: Optional[Dict[str, object]] = None,
    profile_timing: bool = False,
) -> _BoundInfo:
    """Compute the node LP bound using precomputed per-theta hull segments.

    This is the same upper-hull LP relaxation as the reference path. The
    integer search remains over original non-dominated options; this routine
    only accelerates valid LP upper-bound evaluation.
    """

    fixed = node.fixed
    free_items = [i for i, j in enumerate(fixed) if j < 0]
    remaining_capacity = capacity - node.used_cost
    if remaining_capacity < -tol:
        return _BoundInfo(float("-inf"), None, [], free_items, True)

    if use_min_cost_infeasibility_check:
        t0 = time.perf_counter() if profile_timing else 0.0
        min_remaining = sum(float(theta_cache.hull_base_cost[i]) for i in free_items)
        if profile_timing:
            _diag_add(diagnostics, "time_min_cost_check", time.perf_counter() - t0)
        if remaining_capacity + tol < min_remaining:
            _diag_inc(diagnostics, "min_cost_infeasibility_prunes")
            return _BoundInfo(float("-inf"), None, [], free_items, True)

    if not free_items:
        return _BoundInfo(float(node.fixed_value), None, [], free_items, False)

    if use_cheap_prebound and math.isfinite(incumbent_cutoff):
        t0 = time.perf_counter() if profile_timing else 0.0
        cheap_bound = node.fixed_value + sum(float(theta_cache.max_value[i]) for i in free_items)
        if profile_timing:
            _diag_add(diagnostics, "time_cheap_prebound", time.perf_counter() - t0)
        if cheap_bound <= incumbent_cutoff + tol:
            _diag_inc(diagnostics, "cheap_prebound_prunes")
            return _BoundInfo(float(cheap_bound), None, [], free_items, False)

    if any(theta_cache.per_item_hulls[i].costs.size == 0 for i in free_items):
        _diag_inc(diagnostics, "reference_fallback_count")
        return _compute_bound_reference(
            node,
            theta_cache.data.values,
            theta_cache.data.costs,
            theta_cache.option_sets,
            theta_cache.min_cost,
            capacity,
            tol,
            theta_cache.per_item_hulls,
            diagnostics,
            profile_timing,
        )

    t0 = time.perf_counter() if profile_timing else 0.0
    base_cost = sum(float(theta_cache.hull_base_cost[i]) for i in free_items)
    base_value = sum(float(theta_cache.hull_base_value[i]) for i in free_items)
    residual = remaining_capacity - base_cost
    if residual < -tol:
        if profile_timing:
            _diag_add(diagnostics, "time_fast_bound", time.perf_counter() - t0)
        return _BoundInfo(float("-inf"), None, [theta_cache.per_item_hulls[i] for i in free_items], True)

    filled = 0.0
    extra_value = 0.0
    fractional_global_item: Optional[int] = None
    fractional_lambda = 0.0
    extra_costs = np.zeros(len(theta_cache.per_item_hulls), dtype=float) if need_solution else None

    for seg in theta_cache.global_segments:
        if residual <= 0.0:
            break
        if fixed[seg.item] >= 0:
            continue
        take = min(float(seg.length), residual)
        if take <= 0.0:
            continue
        if extra_costs is not None:
            extra_costs[seg.item] += take
        extra_value += take * float(seg.slope)
        filled += take
        residual -= take
        if take < float(seg.length):
            fractional_global_item = int(seg.item)
            fractional_lambda = float(take / float(seg.length))
            break

    hulls = [theta_cache.per_item_hulls[i] for i in free_items]
    lp_value = float(base_value + extra_value)
    total_cost = float(base_cost + filled)
    lp_solution: Optional[LPSolution] = None
    if need_solution:
        local_index = {item: idx for idx, item in enumerate(free_items)}
        positions: List[ItemLPPosition] = []
        for item in free_items:
            hull = theta_cache.per_item_hulls[item]
            assert extra_costs is not None
            cost_i = float(hull.costs[0] + extra_costs[item])
            positions.append(_position_from_hull_cost(hull, cost_i))
        lp_solution = LPSolution(
            lp_value=lp_value,
            capacity=float(remaining_capacity),
            total_cost=total_cost,
            positions=positions,
            fractional_item=local_index.get(fractional_global_item) if fractional_global_item is not None else None,
            fractional_lambda=float(fractional_lambda),
        )

    if profile_timing:
        _diag_add(diagnostics, "time_fast_bound", time.perf_counter() - t0)
    _diag_inc(diagnostics, "fast_lp_bounds_computed")
    if need_solution:
        _diag_inc(diagnostics, "exact_lp_bounds_computed")
    if total_cost > remaining_capacity + max(1.0, abs(remaining_capacity)) * tol:
        return _BoundInfo(float("-inf"), lp_solution, hulls, free_items, True)
    return _BoundInfo(float(node.fixed_value + lp_value), lp_solution, hulls, free_items, False)


def _compute_bound(
    node: _Node,
    theta_cache: FixedThetaCache,
    capacity: float,
    tol: float,
    *,
    use_fast: bool,
    incumbent_cutoff: float = float("-inf"),
    need_solution: bool = True,
    use_cheap_prebound: bool = True,
    use_min_cost_infeasibility_check: bool = True,
    diagnostics: Optional[Dict[str, object]] = None,
    profile_timing: bool = False,
) -> _BoundInfo:
    if use_fast:
        return _compute_bound_fast(
            node,
            theta_cache,
            capacity,
            tol,
            incumbent_cutoff=incumbent_cutoff,
            need_solution=need_solution,
            use_cheap_prebound=use_cheap_prebound,
            use_min_cost_infeasibility_check=use_min_cost_infeasibility_check,
            diagnostics=diagnostics,
            profile_timing=profile_timing,
        )
    _diag_inc(diagnostics, "reference_fallback_count")
    _diag_inc(diagnostics, "exact_lp_bounds_computed")
    return _compute_bound_reference(
        node,
        theta_cache.data.values,
        theta_cache.data.costs,
        theta_cache.option_sets,
        theta_cache.min_cost,
        capacity,
        tol,
        theta_cache.per_item_hulls if theta_cache.per_item_hulls else None,
        diagnostics,
        profile_timing,
    )


def _lp_integral_completion(
    node: _Node,
    bound_info: _BoundInfo,
    values: Sequence[np.ndarray],
    costs: Sequence[np.ndarray],
    capacity: float,
    tol: float,
) -> Optional[Tuple[List[int], float, float]]:
    if bound_info.lp_solution is None:
        if not bound_info.free_items:
            selections = list(node.fixed)
            return selections, objective_for_selection(values, selections), cost_for_selection(costs, selections)
        return None
    lp = bound_info.lp_solution
    if any(tol < pos.lambda_ < 1.0 - tol for pos in lp.positions):
        return None
    selections = list(node.fixed)
    for local_i, item in enumerate(bound_info.free_items):
        pos = lp.positions[local_i]
        vertex = pos.upper_vertex if pos.lambda_ >= 1.0 - tol else pos.lower_vertex
        selections[item] = int(bound_info.hulls[local_i].option_indices[vertex])
    used = cost_for_selection(costs, selections)
    if used > capacity + tol:
        return None
    return selections, objective_for_selection(values, selections), used


def _free_items_from_node(node: _Node) -> List[int]:
    return [i for i, j in enumerate(node.fixed) if j < 0]


def _item_cost_spread(theta_cache: FixedThetaCache, item: int) -> float:
    opts = theta_cache.option_sets[item]
    costs = theta_cache.data.costs[item]
    return float(np.max(costs[opts]) - np.min(costs[opts])) if opts else 0.0


def _item_value_spread(theta_cache: FixedThetaCache, item: int) -> float:
    opts = theta_cache.option_sets[item]
    values = theta_cache.data.values[item]
    return float(np.max(values[opts]) - np.min(values[opts])) if opts else 0.0


def _item_hull_jump(theta_cache: FixedThetaCache, item: int) -> float:
    hull = theta_cache.per_item_hulls[item]
    if hull.values.size <= 1:
        return 0.0
    return float(np.max(np.diff(hull.values)))


def _fractional_branch_item(bound_info: _BoundInfo, tol: float) -> Optional[int]:
    lp = bound_info.lp_solution
    if lp is None or lp.fractional_item is None:
        return None
    pos = lp.positions[lp.fractional_item]
    if tol < pos.lambda_ < 1.0 - tol:
        return int(bound_info.free_items[lp.fractional_item])
    return None


def _branch_item_score(theta_cache: FixedThetaCache, item: int, rule: str) -> Tuple[float, float, float, float, int]:
    value_spread = _item_value_spread(theta_cache, item)
    cost_spread = _item_cost_spread(theta_cache, item)
    hull_jump = _item_hull_jump(theta_cache, item)
    option_count = float(len(theta_cache.option_sets[item]))
    if rule == "largest_hull_jump":
        primary = hull_jump
    elif rule == "largest_cost_spread":
        primary = cost_spread
    elif rule == "largest_value_spread":
        primary = value_spread
    elif rule == "max_option_count_or_entropy":
        primary = option_count
    else:
        primary = value_spread
    return (float(primary), float(value_spread), float(cost_spread), float(option_count), -int(item))


def _rank_branch_candidates(
    node: _Node,
    theta_cache: FixedThetaCache,
    rule: str,
) -> List[int]:
    free_items = _free_items_from_node(node)
    return sorted(free_items, key=lambda item: _branch_item_score(theta_cache, item, rule), reverse=True)


def _strong_branching_item(
    node: _Node,
    theta_cache: FixedThetaCache,
    config: FixedThetaBNBConfig,
    diagnostics: Dict[str, object],
    child_bound_evaluator: Callable[[int, int], float],
    *,
    base_rule: str,
) -> Optional[int]:
    if node.depth > int(config.strong_branching_depth_limit):
        return None
    candidates = _rank_branch_candidates(node, theta_cache, base_rule)[: max(1, int(config.strong_branching_candidates))]
    if not candidates:
        return None
    start = time.perf_counter()
    best_item: Optional[int] = None
    best_score: Optional[Tuple[float, float, float, int]] = None
    evaluated = 0
    reductions: List[float] = []
    max_children_total = max(1, int(config.strong_branching_max_children))
    for item in candidates:
        child_bounds: List[float] = []
        child_opts = sorted(theta_cache.option_sets[item], key=lambda j: _child_order_key(theta_cache.data.values, theta_cache.data.costs, item, j, "value_desc"))
        for opt in child_opts:
            if evaluated >= max_children_total:
                break
            child_bounds.append(float(child_bound_evaluator(item, int(opt))))
            evaluated += 1
        if not child_bounds:
            continue
        finite_bounds = [b for b in child_bounds if math.isfinite(b)]
        worst_child = max(finite_bounds, default=float("-inf"))
        avg_child = float(np.mean(finite_bounds)) if finite_bounds else float("-inf")
        reduction = max(0.0, float(node.upper_bound) - worst_child) if math.isfinite(node.upper_bound) and math.isfinite(worst_child) else 0.0
        reductions.append(reduction)
        # Minimize worst child upper bound, then average child bound; stable tie by item.
        score = (-worst_child, -avg_child, _branch_item_score(theta_cache, item, base_rule)[0], -int(item))
        if best_score is None or score > best_score:
            best_score = score
            best_item = int(item)
        if evaluated >= max_children_total:
            break
    _diag_inc(diagnostics, "strong_branching_count")
    _diag_inc(diagnostics, "strong_branching_candidates_evaluated", evaluated)
    diagnostics["strong_branching_time"] = float(diagnostics.get("strong_branching_time", 0.0)) + (time.perf_counter() - start)
    if reductions:
        diagnostics["average_bound_reduction"] = float(diagnostics.get("average_bound_reduction", 0.0)) + float(np.mean(reductions))
    return best_item


def _choose_branch_item(
    node: _Node,
    bound_info: _BoundInfo,
    theta_cache: FixedThetaCache,
    config: FixedThetaBNBConfig,
    diagnostics: Dict[str, object],
    tol: float,
    child_bound_evaluator: Optional[Callable[[int, int], float]] = None,
) -> int:
    rule = config.branching_rule or "fractional_item_then_spread"
    if rule == "lp_fractional_then_range":
        rule = "fractional_item_then_spread"
    diagnostics["branching_rule_used"] = rule
    fractional_item = _fractional_branch_item(bound_info, tol)
    if rule == "fractional_item_then_spread" and fractional_item is not None:
        diagnostics["selected_item_score"] = float(_branch_item_score(theta_cache, fractional_item, "largest_value_spread")[0])
        return int(fractional_item)
    if rule == "tight_capacity_hybrid":
        if child_bound_evaluator is not None:
            strong = _strong_branching_item(node, theta_cache, config, diagnostics, child_bound_evaluator, base_rule="largest_cost_spread")
            if strong is not None:
                diagnostics["selected_item_score"] = float(_branch_item_score(theta_cache, strong, "largest_cost_spread")[0])
                return int(strong)
        if fractional_item is not None and len(theta_cache.option_sets[fractional_item]) >= 3:
            diagnostics["selected_item_score"] = float(_branch_item_score(theta_cache, fractional_item, "largest_cost_spread")[0])
            return int(fractional_item)
        ranked = _rank_branch_candidates(node, theta_cache, "largest_cost_spread")
    elif rule == "strong_branching_lite":
        if child_bound_evaluator is not None:
            strong = _strong_branching_item(node, theta_cache, config, diagnostics, child_bound_evaluator, base_rule="largest_cost_spread")
            if strong is not None:
                diagnostics["selected_item_score"] = float(_branch_item_score(theta_cache, strong, "largest_cost_spread")[0])
                return int(strong)
        ranked = _rank_branch_candidates(node, theta_cache, "largest_cost_spread")
    elif rule in {"largest_hull_jump", "largest_cost_spread", "largest_value_spread", "max_option_count_or_entropy"}:
        ranked = _rank_branch_candidates(node, theta_cache, rule)
    else:
        ranked = _rank_branch_candidates(node, theta_cache, "largest_value_spread")
    if not ranked:
        raise ValueError("no free item available for branching")
    diagnostics["selected_item_score"] = float(_branch_item_score(theta_cache, ranked[0], rule if rule else "largest_value_spread")[0])
    return int(ranked[0])


def _child_order_key(
    values: Sequence[np.ndarray],
    costs: Sequence[np.ndarray],
    item: int,
    option: int,
    mode: str = "value_desc",
) -> Tuple[float, float, int]:
    cost = float(costs[item][option])
    value = float(values[item][option])
    if mode == "cost_asc":
        return (cost, -value, int(option))
    if mode == "density_desc":
        density = value / max(cost, EPS)
        return (-density, -value, cost, int(option))
    if mode == "bound_promise":
        return (-value, cost, int(option))
    return (-value, cost, int(option))


def solve_fixed_theta_bnb(
    instance: PricingInstance,
    theta: float,
    config: Optional[FixedThetaBNBConfig] = None,
    fixed_theta_cache: Optional[FixedThetaCache] = None,
) -> FixedThetaBNBResult:
    """Solve a fixed-theta MCKP exactly with branch-and-bound.

    The node upper bound uses upper-hull LP relaxations. Integer branching uses
    original fixed-theta options after only safe per-item dominance pruning; in
    particular, below-upper-hull options are kept in the search.
    """

    _require_representable_objective_range(instance)
    cfg = config or FixedThetaBNBConfig()
    tol = float(cfg.tolerance)
    start = time.perf_counter()
    profile_timing = bool(cfg.profile_timing or cfg.collect_diagnostics)
    diagnostics: Dict[str, object] = {
        "initial_incumbent_feasible_for_theta": False,
        "external_cutoff_prunes": 0,
        "nodes_created": 0,
        "nodes_live_peak": 0,
        "nodes_pruned_cutoff": 0,
        "nodes_limited_by_time": 0,
        "nodes_limited_by_node": 0,
        "branch_fractional_item_count": 0,
        "branch_fallback_count": 0,
        "fractional_branch_count": 0,
        "fallback_branch_count": 0,
        "branching_nodes": 0,
        "total_child_count": 0,
        "max_branching_arity": 0,
        "root_lp_upper_bound": float("nan"),
        "initial_incumbent_value": float("-inf"),
        "final_incumbent_value": float("-inf"),
        "final_upper_bound": float("nan"),
        "best_remaining_node_bound": float("-inf"),
        "time_fixed_theta_data": 0.0,
        "time_root_lp": 0.0,
        "time_node_lp_total": 0.0,
        "time_hull_build_total": 0.0,
        "time_greedy_lp_total": 0.0,
        "time_branching_total": 0.0,
        "time_bound_seconds": 0.0,
        "time_certificate_total": 0.0,
        "time_branching_seconds": 0.0,
        "time_child_generation_seconds": 0.0,
        "time_child_generation_total": 0.0,
        "cheap_prebound_prunes": 0,
        "min_cost_infeasibility_prunes": 0,
        "exact_lp_bounds_computed": 0,
        "fast_lp_bounds_computed": 0,
        "reference_fallback_count": 0,
        "bound_cache_hits": 0,
        "bound_cache_misses": 0,
        "time_cheap_prebound": 0.0,
        "time_min_cost_check": 0.0,
        "time_fast_bound": 0.0,
        "time_reference_bound": 0.0,
        "branching_rule_used": cfg.branching_rule,
        "child_ordering_mode": cfg.child_ordering or cfg.child_order,
        "strong_branching_count": 0,
        "strong_branching_time": 0.0,
        "strong_branching_candidates_evaluated": 0,
        "average_bound_reduction": 0.0,
        "selected_item_score": float("nan"),
        "local_incumbent_swaps": 0,
        "local_incumbent_pair_evaluations": 0,
        "local_incumbent_improved": False,
    }
    try:
        data_start = time.perf_counter()
        theta_cache = fixed_theta_cache if fixed_theta_cache is not None else _build_fixed_theta_cache(instance, theta, tol)
        diagnostics["time_fixed_theta_data"] = float(time.perf_counter() - data_start)
        data = theta_cache.data
        values, costs, capacity = data.values, data.costs, data.capacity
        if _fixed_theta_capacity_is_negative(data):
            return _make_result(
                status="infeasible",
                theta=theta,
                capacity=capacity,
                incumbent_value=float("-inf"),
                incumbent_selection=None,
                incumbent_cost=float("inf"),
                upper_bound=float("-inf"),
                nodes_explored=0,
                nodes_pruned_bound=0,
                nodes_pruned_infeasible=1,
                nodes_integral=0,
                start_time=start,
                config=cfg,
                values=values,
                costs=costs,
                diagnostics=diagnostics,
                message="fixed-theta capacity is negative",
            )

        option_sets = theta_cache.option_sets
        if any(len(opts) == 0 for opts in option_sets):
            return _make_result(
                status="infeasible",
                theta=theta,
                capacity=capacity,
                incumbent_value=float("-inf"),
                incumbent_selection=None,
                incumbent_cost=float("inf"),
                upper_bound=float("-inf"),
                nodes_explored=0,
                nodes_pruned_bound=0,
                nodes_pruned_infeasible=1,
                nodes_integral=0,
                start_time=start,
                config=cfg,
                values=values,
                costs=costs,
                diagnostics=diagnostics,
                message="some item has no available options",
            )

        min_cost = theta_cache.min_cost
        per_item_hulls = theta_cache.per_item_hulls if cfg.use_cache else None
        if not _fixed_theta_partial_is_feasible(
            data, tuple(-1 for _ in range(instance.n_items)), option_sets
        ):
            return _make_result(
                status="infeasible",
                theta=theta,
                capacity=capacity,
                incumbent_value=float("-inf"),
                incumbent_selection=None,
                incumbent_cost=float("inf"),
                upper_bound=float("-inf"),
                nodes_explored=0,
                nodes_pruned_bound=0,
                nodes_pruned_infeasible=1,
                nodes_integral=0,
                start_time=start,
                config=cfg,
                values=values,
                costs=costs,
                diagnostics=diagnostics,
                message="minimum possible fixed-theta cost exceeds capacity",
            )

        incumbent_selection: Optional[List[int]] = None
        incumbent_value = float("-inf")
        incumbent_exact_value: Optional[Fraction] = None
        incumbent_cost = float("inf")
        if cfg.initial_incumbent_selection is not None:
            candidate_selection = list(map(int, cfg.initial_incumbent_selection))
            flags = validate_fixed_theta_selection(values, costs, capacity, candidate_selection, tol)
            if (
                all(
                    flags.get(k, False)
                    for k in ["valid_length", "valid_indices", "capacity_feasible"]
                )
                and _fixed_theta_selection_is_feasible(data, candidate_selection)
            ):
                candidate_value = objective_for_selection(values, candidate_selection)
                if cfg.initial_incumbent_value is None or abs(candidate_value - float(cfg.initial_incumbent_value)) <= max(1.0, abs(candidate_value)) * 1e-7:
                    incumbent_selection = candidate_selection
                    incumbent_value = candidate_value
                    incumbent_exact_value = _selection_objective_fraction(
                        values, candidate_selection
                    )
                    incumbent_cost = cost_for_selection(costs, candidate_selection)
                    diagnostics["initial_incumbent_feasible_for_theta"] = True
        if cfg.use_greedy_incumbent:
            sel, val, used = _greedy_incumbent(values, costs, option_sets, capacity, tol)
            candidate_exact = (
                _selection_objective_fraction(values, sel)
                if sel is not None
                else None
            )
            if (
                sel is not None
                and _fixed_theta_selection_is_feasible(data, sel)
                and candidate_exact is not None
                and (
                    incumbent_exact_value is None
                    or candidate_exact > incumbent_exact_value
                )
            ):
                incumbent_selection = sel
                incumbent_value = val
                incumbent_exact_value = candidate_exact
                incumbent_cost = used
        if cfg.use_local_incumbent_improvement and incumbent_selection is not None:
            improved, improved_value, improved_cost, swaps, pair_evaluations = _local_improve_incumbent(
                values,
                costs,
                option_sets,
                capacity,
                incumbent_selection,
                tol,
                max_passes=cfg.local_search_max_passes,
                max_swaps=cfg.local_search_max_swaps,
                neighborhood=cfg.local_search_neighborhood,
                max_pair_evaluations=cfg.local_search_max_pair_evaluations,
            )
            diagnostics["local_incumbent_swaps"] = int(diagnostics["local_incumbent_swaps"]) + int(swaps)
            diagnostics["local_incumbent_pair_evaluations"] = int(diagnostics["local_incumbent_pair_evaluations"]) + int(pair_evaluations)
            improved_exact = _selection_objective_fraction(values, improved)
            if (
                _fixed_theta_selection_is_feasible(data, improved)
                and (
                    incumbent_exact_value is None
                    or improved_exact > incumbent_exact_value
                )
            ):
                incumbent_selection = improved
                incumbent_value = improved_value
                incumbent_exact_value = improved_exact
                incumbent_cost = improved_cost
                diagnostics["local_incumbent_improved"] = True
        diagnostics["initial_incumbent_value"] = float(incumbent_value)
        cutoff_value = float(cfg.objective_cutoff) if cfg.objective_cutoff is not None and cfg.use_cutoff_pruning else float("-inf")

        def prune_threshold() -> float:
            return max(incumbent_value, cutoff_value)

        def note_bound_prune(bound: float) -> None:
            if math.isfinite(cutoff_value) and bound <= cutoff_value + tol and bound > incumbent_value + tol:
                _diag_inc(diagnostics, "external_cutoff_prunes")
                _diag_inc(diagnostics, "nodes_pruned_cutoff")

        bound_cache: Dict[Tuple[int, ...], Tuple[float, bool]] = {}

        def compute_node_bound(node: _Node, *, need_solution: bool) -> _BoundInfo:
            if cfg.use_bound_cache and not need_solution:
                cached = bound_cache.get(node.fixed)
                if cached is not None:
                    _diag_inc(diagnostics, "bound_cache_hits")
                    free_items = [i for i, j in enumerate(node.fixed) if j < 0]
                    return _BoundInfo(float(cached[0]), None, [], free_items, bool(cached[1]))
                _diag_inc(diagnostics, "bound_cache_misses")
            exact_partial_feasible = _fixed_theta_partial_is_feasible(
                data, node.fixed, option_sets
            )
            if not exact_partial_feasible:
                return _BoundInfo(
                    float("-inf"), None, [], _free_items_from_node(node), True
                )
            bound = _compute_bound(
                node,
                theta_cache,
                capacity,
                tol,
                use_fast=cfg.use_fast_residual_lp_bound,
                incumbent_cutoff=prune_threshold(),
                need_solution=need_solution,
                use_cheap_prebound=cfg.use_cheap_prebound,
                use_min_cost_infeasibility_check=cfg.use_min_cost_infeasibility_check,
                diagnostics=diagnostics if profile_timing or cfg.collect_diagnostics else None,
                profile_timing=profile_timing,
            )
            exact_lp_value: Optional[Fraction] = None
            if data.exact_costs and data.exact_values and data.exact_capacity is not None:
                exact_lp_value = _exact_fixed_theta_lp_value(
                    data, node.fixed, option_sets
                )
                if exact_lp_value is None:
                    return _BoundInfo(
                        float("-inf"), None, [], _free_items_from_node(node), True
                    )
                bound.upper_bound = _fraction_to_float_up(exact_lp_value)
            if bound.infeasible:
                # The floating LP path is an outward relaxation, but retain a
                # fail-closed exact fallback against any residual summation
                # edge case: branch under the trivial value upper bound rather
                # than pruning an exactly feasible node.
                free_items = _free_items_from_node(node)
                fallback_upper = (
                    _fraction_to_float_up(exact_lp_value)
                    if exact_lp_value is not None
                    else float(
                        node.fixed_value
                        + math.fsum(theta_cache.max_value[i] for i in free_items)
                    )
                )
                bound = _BoundInfo(
                    fallback_upper, None, [], free_items, False
                )
            if cfg.use_bound_cache and not need_solution and len(bound_cache) < cfg.bound_cache_max_entries:
                bound_cache[node.fixed] = (float(bound.upper_bound), bool(bound.infeasible))
            return bound

        root_fixed = tuple(-1 for _ in range(instance.n_items))
        root = _Node(root_fixed, 0.0, 0.0, float("inf"), 0, tuple())
        t0 = time.perf_counter()
        root_bound = compute_node_bound(root, need_solution=False)
        root_elapsed = time.perf_counter() - t0
        diagnostics["time_root_lp"] = float(root_elapsed)
        diagnostics["time_bound_seconds"] = float(diagnostics["time_bound_seconds"]) + root_elapsed
        diagnostics["root_lp_upper_bound"] = float(root_bound.upper_bound)
        if root_bound.infeasible:
            return _make_result(
                status="infeasible",
                theta=theta,
                capacity=capacity,
                incumbent_value=float("-inf"),
                incumbent_selection=None,
                incumbent_cost=float("inf"),
                upper_bound=float("-inf"),
                nodes_explored=0,
                nodes_pruned_bound=0,
                nodes_pruned_infeasible=1,
                nodes_integral=0,
                start_time=start,
                config=cfg,
                values=values,
                costs=costs,
                diagnostics=diagnostics,
                message="root node is infeasible",
            )
        if root_bound.upper_bound < prune_threshold():
            note_bound_prune(root_bound.upper_bound)
            status = "optimal" if incumbent_selection is not None else "cutoff_pruned"
            upper = incumbent_value if status == "optimal" else max(prune_threshold(), incumbent_value)
            return _make_result(
                status=status,
                theta=theta,
                capacity=capacity,
                incumbent_value=incumbent_value,
                incumbent_selection=incumbent_selection,
                incumbent_cost=incumbent_cost,
                upper_bound=upper,
                nodes_explored=0,
                nodes_pruned_bound=1,
                nodes_pruned_infeasible=0,
                nodes_integral=0,
                start_time=start,
                config=cfg,
                values=values,
                costs=costs,
                diagnostics=diagnostics,
                message="root pruned by incumbent/cutoff",
            )

        live: List[Tuple[float, Tuple[int, ...], int, _Node]] = []
        counter = itertools.count()
        root = _Node(root_fixed, 0.0, 0.0, root_bound.upper_bound, 0, tuple())
        heapq.heappush(live, (-root.upper_bound, root.tie_key, next(counter), root))
        diagnostics["nodes_created"] = 1
        diagnostics["nodes_live_peak"] = 1

        nodes_explored = 0
        nodes_pruned_bound = 0
        nodes_pruned_infeasible = 0
        nodes_integral = 0
        status = "optimal"
        message = ""

        while live:
            if cfg.time_limit_seconds is not None and time.perf_counter() - start >= cfg.time_limit_seconds:
                status = "time_limit"
                message = "time limit reached"
                diagnostics["nodes_limited_by_time"] = len(live)
                break
            if cfg.node_limit is not None and nodes_explored >= cfg.node_limit:
                status = "node_limit"
                message = "node limit reached"
                diagnostics["nodes_limited_by_node"] = len(live)
                break

            _, _, _, node = heapq.heappop(live)
            if node.upper_bound < prune_threshold():
                note_bound_prune(node.upper_bound)
                nodes_pruned_bound += 1
                continue

            nodes_explored += 1
            _diag_append_sample(
                diagnostics if cfg.collect_diagnostics else None,
                "node_depth_samples",
                int(node.depth),
                max_samples=cfg.max_diagnostic_nodes,
                sample_rate=cfg.diagnostic_sample_rate,
                sample_index=nodes_explored,
            )
            t0 = time.perf_counter()
            bound_info = compute_node_bound(node, need_solution=True)
            bound_elapsed = time.perf_counter() - t0
            diagnostics["time_node_lp_total"] = float(diagnostics["time_node_lp_total"]) + bound_elapsed
            diagnostics["time_bound_seconds"] = float(diagnostics["time_bound_seconds"]) + bound_elapsed
            if bound_info.infeasible:
                nodes_pruned_infeasible += 1
                continue
            if cfg.collect_diagnostics and incumbent_selection is not None:
                _diag_append_sample(
                    diagnostics,
                    "node_bound_gap_samples",
                    float(max(0.0, bound_info.upper_bound - incumbent_value)),
                    max_samples=cfg.max_diagnostic_nodes,
                    sample_rate=cfg.diagnostic_sample_rate,
                    sample_index=nodes_explored,
                )
            if bound_info.upper_bound < prune_threshold():
                note_bound_prune(bound_info.upper_bound)
                nodes_pruned_bound += 1
                continue

            integral = _lp_integral_completion(node, bound_info, values, costs, capacity, tol)
            if integral is not None:
                sel, val, used = integral
                exact_integral_feasible = _fixed_theta_selection_is_feasible(
                    data, sel
                )
                if exact_integral_feasible:
                    nodes_integral += 1
                    _diag_inc(diagnostics, "nodes_integral_lp")
                if exact_integral_feasible and cfg.use_local_incumbent_improvement:
                    improved, improved_value, improved_cost, swaps, pair_evaluations = _local_improve_incumbent(
                        values,
                        costs,
                        option_sets,
                        capacity,
                        sel,
                        tol,
                        max_passes=cfg.local_search_max_passes,
                        max_swaps=cfg.local_search_max_swaps,
                        neighborhood=cfg.local_search_neighborhood,
                        max_pair_evaluations=cfg.local_search_max_pair_evaluations,
                    )
                    diagnostics["local_incumbent_swaps"] = int(diagnostics["local_incumbent_swaps"]) + int(swaps)
                    diagnostics["local_incumbent_pair_evaluations"] = int(diagnostics["local_incumbent_pair_evaluations"]) + int(pair_evaluations)
                    if (
                        _fixed_theta_selection_is_feasible(data, improved)
                        and _selection_objective_fraction(values, improved)
                        > _selection_objective_fraction(values, sel)
                    ):
                        sel, val, used = improved, improved_value, improved_cost
                        diagnostics["local_incumbent_improved"] = True
                candidate_exact = _selection_objective_fraction(values, sel)
                if exact_integral_feasible and (
                    incumbent_exact_value is None
                    or candidate_exact > incumbent_exact_value
                ):
                    incumbent_selection = sel
                    incumbent_value = val
                    incumbent_exact_value = candidate_exact
                    incumbent_cost = used
                if exact_integral_feasible or not bound_info.free_items:
                    if not exact_integral_feasible:
                        nodes_pruned_infeasible += 1
                    continue

            branch_start = time.perf_counter()
            def branch_child_bound(item: int, opt: int) -> float:
                fixed_list = list(node.fixed)
                fixed_list[item] = int(opt)
                used_cost = node.used_cost + float(costs[item][opt])
                fixed_value = node.fixed_value + float(values[item][opt])
                fixed_tuple = tuple(fixed_list)
                child = _Node(
                    fixed=fixed_tuple,
                    used_cost=used_cost,
                    fixed_value=fixed_value,
                    upper_bound=float("inf"),
                    depth=node.depth + 1,
                    tie_key=node.tie_key + (item, int(opt)),
                )
                if not _fixed_theta_partial_is_feasible(
                    data, fixed_tuple, option_sets
                ):
                    return float("-inf")
                bound = compute_node_bound(child, need_solution=False)
                return float(bound.upper_bound)

            branch_item = _choose_branch_item(
                node,
                bound_info,
                theta_cache,
                cfg,
                diagnostics,
                tol,
                child_bound_evaluator=branch_child_bound,
            )
            branch_elapsed = time.perf_counter() - branch_start
            diagnostics["time_branching_total"] = float(diagnostics["time_branching_total"]) + branch_elapsed
            diagnostics["time_branching_seconds"] = float(diagnostics["time_branching_seconds"]) + branch_elapsed
            _diag_append_sample(
                diagnostics if cfg.collect_diagnostics else None,
                "branch_item_samples",
                int(branch_item),
                max_samples=cfg.max_diagnostic_nodes,
                sample_rate=cfg.diagnostic_sample_rate,
                sample_index=int(diagnostics["branching_nodes"]),
            )
            if bound_info.lp_solution is not None and bound_info.lp_solution.fractional_item is not None:
                frac_item = int(bound_info.free_items[bound_info.lp_solution.fractional_item])
                if frac_item == branch_item:
                    _diag_inc(diagnostics, "fractional_branch_count")
                    _diag_inc(diagnostics, "branch_fractional_item_count")
                else:
                    _diag_inc(diagnostics, "fallback_branch_count")
                    _diag_inc(diagnostics, "branch_fallback_count")
            else:
                _diag_inc(diagnostics, "fallback_branch_count")
                _diag_inc(diagnostics, "branch_fallback_count")
            _diag_inc(diagnostics, "branching_nodes")
            child_ordering = cfg.child_ordering or cfg.child_order
            child_options = sorted(option_sets[branch_item], key=lambda j: _child_order_key(values, costs, branch_item, j, child_ordering))
            _diag_inc(diagnostics, "total_child_count", len(child_options))
            diagnostics["max_branching_arity"] = max(int(diagnostics["max_branching_arity"]), len(child_options))
            child_start = time.perf_counter()
            for opt in child_options:
                _diag_inc(diagnostics, "nodes_created")
                fixed_list = list(node.fixed)
                fixed_list[branch_item] = int(opt)
                used_cost = node.used_cost + float(costs[branch_item][opt])
                fixed_value = node.fixed_value + float(values[branch_item][opt])
                fixed_tuple = tuple(fixed_list)
                free_items = [i for i, j in enumerate(fixed_tuple) if j < 0]
                if not _fixed_theta_partial_is_feasible(
                    data, fixed_tuple, option_sets
                ):
                    nodes_pruned_infeasible += 1
                    continue
                child = _Node(
                    fixed=fixed_tuple,
                    used_cost=used_cost,
                    fixed_value=fixed_value,
                    upper_bound=float("inf"),
                    depth=node.depth + 1,
                    tie_key=node.tie_key + (branch_item, int(opt)),
                )
                t0 = time.perf_counter()
                child_bound = compute_node_bound(child, need_solution=False)
                child_bound_elapsed = time.perf_counter() - t0
                diagnostics["time_node_lp_total"] = float(diagnostics["time_node_lp_total"]) + child_bound_elapsed
                diagnostics["time_bound_seconds"] = float(diagnostics["time_bound_seconds"]) + child_bound_elapsed
                if child_bound.infeasible:
                    nodes_pruned_infeasible += 1
                    continue
                if child_bound.upper_bound < prune_threshold():
                    note_bound_prune(child_bound.upper_bound)
                    nodes_pruned_bound += 1
                    continue
                child = _Node(
                    fixed=child.fixed,
                    used_cost=child.used_cost,
                    fixed_value=child.fixed_value,
                    upper_bound=child_bound.upper_bound,
                    depth=child.depth,
                    tie_key=child.tie_key,
                )
                heapq.heappush(live, (-child.upper_bound, child.tie_key, next(counter), child))
                diagnostics["nodes_live_peak"] = max(int(diagnostics["nodes_live_peak"]), len(live))
            child_elapsed = time.perf_counter() - child_start
            diagnostics["time_child_generation_seconds"] = float(diagnostics["time_child_generation_seconds"]) + child_elapsed
            diagnostics["time_child_generation_total"] = float(diagnostics["time_child_generation_total"]) + child_elapsed

        if status == "optimal":
            if int(diagnostics["external_cutoff_prunes"]) > 0 and cutoff_value > incumbent_value + tol:
                status = "cutoff_pruned"
                upper_bound = max(cutoff_value, incumbent_value)
            else:
                upper_bound = incumbent_value if incumbent_selection is not None else float("-inf")
        else:
            live_bound = max([-entry[0] for entry in live], default=float("-inf"))
            upper_bound = max(incumbent_value, live_bound)
            diagnostics["best_remaining_node_bound"] = float(live_bound)
        if incumbent_selection is None and status == "optimal":
            status = "infeasible"
            message = message or "all nodes fathomed without feasible incumbent"
        if cfg.collect_diagnostics:
            diagnostic_gaps = [float(v) for v in diagnostics.get("node_bound_gap_samples", []) if math.isfinite(float(v))]
            diagnostics["node_upper_bound_gap_count"] = len(diagnostic_gaps)
            diagnostics["node_upper_bound_gap_mean"] = float(np.mean(diagnostic_gaps)) if diagnostic_gaps else float("nan")
            diagnostics["node_upper_bound_gap_max"] = float(np.max(diagnostic_gaps)) if diagnostic_gaps else float("nan")
            diagnostics["node_bound_gap_quantiles"] = _finite_quantiles(diagnostic_gaps)
            diagnostics["node_depth_quantiles"] = _finite_quantiles([float(v) for v in diagnostics.get("node_depth_samples", [])])
            live_bounds = [-entry[0] for entry in live]
            diagnostics["live_bound_quantiles"] = _finite_quantiles(live_bounds)
        branching_nodes = int(diagnostics["branching_nodes"])
        diagnostics["average_branching_arity"] = (
            float(int(diagnostics["total_child_count"]) / branching_nodes) if branching_nodes else 0.0
        )
        diagnostics["final_incumbent_value"] = float(incumbent_value)
        diagnostics["final_upper_bound"] = float(upper_bound)
        diagnostics["absolute_gap"] = float(max(0.0, upper_bound - incumbent_value)) if math.isfinite(incumbent_value) else float("inf")
        diagnostics["relative_gap"] = _relative_gap(upper_bound, incumbent_value, tol) if math.isfinite(incumbent_value) else float("inf")
        diagnostics["nodes_explored"] = int(nodes_explored)
        diagnostics["nodes_pruned_bound"] = int(nodes_pruned_bound)
        diagnostics["nodes_pruned_infeasible"] = int(nodes_pruned_infeasible)
        diagnostics["nodes_integral_lp"] = int(nodes_integral)
        diagnostics["profile_timing"] = bool(cfg.profile_timing)

        return _make_result(
            status=status,
            theta=theta,
            capacity=capacity,
            incumbent_value=incumbent_value,
            incumbent_selection=incumbent_selection,
            incumbent_cost=incumbent_cost,
            upper_bound=upper_bound,
            nodes_explored=nodes_explored,
            nodes_pruned_bound=nodes_pruned_bound,
            nodes_pruned_infeasible=nodes_pruned_infeasible,
            nodes_integral=nodes_integral,
            start_time=start,
            config=cfg,
            values=values,
            costs=costs,
            diagnostics=diagnostics,
            message=message,
        )
    except Exception as exc:  # pragma: no cover - exercised only on unexpected failures
        elapsed_capacity = float("nan")
        return FixedThetaBNBResult(
            status="error",
            theta=float(theta),
            objective_value=float("-inf"),
            selected_options=None,
            used_capacity=float("inf"),
            capacity=elapsed_capacity,
            fixed_theta_residual=float("nan"),
            upper_bound=float("inf"),
            lower_bound=float("-inf"),
            absolute_gap=float("inf"),
            relative_gap=float("inf"),
            nodes_explored=0,
            nodes_pruned_bound=0,
            nodes_pruned_infeasible=0,
            nodes_integral=0,
            runtime_seconds=float(time.perf_counter() - start),
            config_metadata={"error": type(exc).__name__},
            validation_flags={},
            diagnostics=diagnostics,
            message=str(exc),
        )


def brute_force_fixed_theta(instance: PricingInstance, theta: float, tol: float = EPS) -> FixedThetaBNBResult:
    """Brute-force fixed-theta MCKP solver for tiny test instances only."""

    _require_representable_objective_range(instance)
    start = time.perf_counter()
    data = build_fixed_theta_data(instance, theta)
    best_sel: Optional[List[int]] = None
    best_obj = float("-inf")
    best_exact_obj: Optional[Fraction] = None
    best_cost = float("inf")
    if not _fixed_theta_capacity_is_negative(data):
        ranges = [range(len(group)) for group in instance.items]
        for combo in itertools.product(*ranges):
            used = cost_for_selection(data.costs, combo)
            if not _fixed_theta_selection_is_feasible(data, combo):
                continue
            obj = objective_for_selection(data.values, combo)
            exact_obj = _selection_objective_fraction(data.values, combo)
            if (
                best_exact_obj is None
                or exact_obj > best_exact_obj
                or (
                    exact_obj == best_exact_obj
                    and list(combo) < (best_sel or [math.inf])
                )
            ):
                best_obj = obj
                best_exact_obj = exact_obj
                best_sel = list(map(int, combo))
                best_cost = used
    status = "optimal" if best_sel is not None else "infeasible"
    return _make_result(
        status=status,
        theta=theta,
        capacity=data.capacity,
        incumbent_value=best_obj,
        incumbent_selection=best_sel,
        incumbent_cost=best_cost,
        upper_bound=best_obj,
        nodes_explored=0,
        nodes_pruned_bound=0,
        nodes_pruned_infeasible=0,
        nodes_integral=0,
        start_time=start,
        config=FixedThetaBNBConfig(tolerance=tol),
        values=data.values,
        costs=data.costs,
        message="brute force reference",
    )


def _robust_objective(instance: PricingInstance, selections: Sequence[int]) -> float:
    return float(sum(float(instance.items[i][int(j)].value) for i, j in enumerate(selections)))


def _global_result_gap(upper: float, lower: float, tol: float) -> Tuple[float, float]:
    if not math.isfinite(upper) or not math.isfinite(lower):
        return float("inf"), float("inf")
    gap = max(0.0, upper - lower)
    if gap <= tol:
        return 0.0, 0.0
    return gap, _relative_gap(upper, lower, tol)


def _validate_global_selection(instance: PricingInstance, selections: Optional[Sequence[int]], objective_value: float, tol: float) -> Dict[str, bool]:
    if selections is None:
        return {
            "has_incumbent": False,
            "valid_selection_length": False,
            "valid_selection_indices": False,
            "objective_matches": False,
            "robust_certificate_feasible": False,
        }
    valid_length = len(selections) == instance.n_items
    valid_indices = valid_length and all(0 <= int(j) < len(instance.items[i]) for i, j in enumerate(selections))
    cert = compute_certificate(instance, selections) if valid_indices else float("-inf")
    obj = _robust_objective(instance, selections) if valid_indices else float("nan")
    return {
        "has_incumbent": True,
        "valid_selection_length": bool(valid_length),
        "valid_selection_indices": bool(valid_indices),
        "objective_matches": bool(valid_indices and abs(obj - objective_value) <= tol),
        "robust_certificate_feasible": bool(valid_indices and certificate_is_feasible(instance, selections)),
    }


def _global_result(
    *,
    instance: PricingInstance,
    status: str,
    objective_value: float,
    selected_options: Optional[List[int]],
    selected_theta: Optional[float],
    lower_bound: float,
    upper_bound: float,
    start_time: float,
    config: GlobalThetaBNBConfig,
    records: List[GlobalThetaRecord],
    total_nodes: int,
    total_root_lp_time: float,
    total_bnb_time: float,
    diagnostics: Optional[Dict[str, object]] = None,
    message: str = "",
) -> GlobalThetaBNBResult:
    cert = compute_certificate(instance, selected_options) if selected_options is not None else float("-inf")
    abs_gap, rel_gap = _global_result_gap(upper_bound, lower_bound, config.tolerance)
    return GlobalThetaBNBResult(
        status=status,
        objective_value=float(objective_value) if selected_options is not None else float("-inf"),
        selected_options=selected_options,
        selected_theta=selected_theta,
        robust_certificate=float(cert),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        absolute_gap=float(abs_gap),
        relative_gap=float(rel_gap),
        theta_count_total=len(records),
        theta_count_infeasible_capacity=sum(1 for r in records if r.infeasible_capacity),
        theta_count_pruned_by_bound=sum(1 for r in records if r.pruned_by_bound),
        theta_count_solved_optimal=sum(1 for r in records if r.status == "optimal"),
        theta_count_limited=sum(1 for r in records if r.status in {"time_limit", "node_limit", "not_run_time_limit", "not_run_node_limit"}),
        theta_count_error=sum(1 for r in records if r.status == "error"),
        total_nodes_explored=int(total_nodes),
        total_runtime_seconds=float(time.perf_counter() - start_time),
        per_theta_records=records,
        total_root_lp_time_seconds=float(total_root_lp_time),
        total_fixed_theta_bnb_time_seconds=float(total_bnb_time),
        diagnostics=diagnostics or {},
        config_metadata={
            "tolerance": config.tolerance,
            "time_limit_seconds": config.time_limit_seconds,
            "node_limit": config.node_limit,
            "fixed_theta_time_limit_seconds": config.fixed_theta_time_limit_seconds,
            "fixed_theta_node_limit": config.fixed_theta_node_limit,
            "use_hullround_incumbent": config.use_hullround_incumbent,
            "use_fixed_theta_greedy_incumbent": config.use_fixed_theta_greedy_incumbent,
            "use_caches": config.use_caches,
            "use_objective_cutoff": config.use_objective_cutoff,
            "theta_order": config.theta_order,
            "branching_rule": config.branching_rule,
            "child_ordering": config.child_ordering,
            "strong_branching_candidates": config.strong_branching_candidates,
            "strong_branching_depth_limit": config.strong_branching_depth_limit,
            "strong_branching_max_children": config.strong_branching_max_children,
            "strong_branching_use_fast_bound": config.strong_branching_use_fast_bound,
            "incumbent_heuristic": config.incumbent_heuristic,
            "use_local_incumbent_improvement": config.use_local_incumbent_improvement,
            "local_search_max_passes": config.local_search_max_passes,
            "local_search_max_swaps": config.local_search_max_swaps,
            "local_search_neighborhood": config.local_search_neighborhood,
            "local_search_max_pair_evaluations": config.local_search_max_pair_evaluations,
            "use_multistart_incumbent": config.use_multistart_incumbent,
            "multistart_theta_count": config.multistart_theta_count,
            "multistart_use_local_improvement": config.multistart_use_local_improvement,
            "collect_diagnostics": config.collect_diagnostics,
            "max_diagnostic_nodes": config.max_diagnostic_nodes,
            "diagnostic_sample_rate": config.diagnostic_sample_rate,
            "profile_timing": config.profile_timing,
            "use_fast_residual_lp_bound": config.use_fast_residual_lp_bound,
            "use_cheap_prebound": config.use_cheap_prebound,
            "use_min_cost_infeasibility_check": config.use_min_cost_infeasibility_check,
            "use_bound_cache": config.use_bound_cache,
            "bound_cache_max_entries": config.bound_cache_max_entries,
        },
        validation_flags=_validate_global_selection(instance, selected_options, objective_value, config.tolerance),
        message=message,
    )


def solve_global_theta_bnb(
    instance: PricingInstance,
    config: Optional[GlobalThetaBNBConfig] = None,
) -> GlobalThetaBNBResult:
    """Solve the global Gamma-robust MCKP by exact full theta enumeration.

    A finite global gap is reported only after every theta candidate has a
    valid root upper-bound record.  The implementation initializes those
    records before the limited search loop: feasible fixed-theta rows receive
    the hull-greedy root LP bound, and rows with negative fixed-theta capacity
    or an infeasible root receive ``-inf``.
    """

    _require_representable_objective_range(instance)
    cfg = config or GlobalThetaBNBConfig()
    tol = float(cfg.tolerance)
    start = time.perf_counter()
    records: List[GlobalThetaRecord] = []
    total_nodes = 0
    candidates = build_full_theta_candidates(instance, tol=tol)
    cache_by_theta: Dict[float, FixedThetaCache] = {}
    lp_bound_by_theta: Dict[float, FixedThetaLPBoundResult] = {}
    total_root_lp_time = 0.0
    total_bnb_time = 0.0
    diagnostics: Dict[str, object] = {"theta_order": cfg.theta_order}
    objective_arrays = [
        np.asarray([option.value for option in group], dtype=float)
        for group in instance.items
    ]

    incumbent_selection: Optional[List[int]] = None
    incumbent_value = float("-inf")
    incumbent_exact_value: Optional[Fraction] = None
    incumbent_theta: Optional[float] = None

    if cfg.use_hullround_incumbent:
        try:
            hr = solve_hullround(instance, upgrade_completion=True)
            if hr.is_feasible and hr.selections and certificate_is_feasible(instance, hr.selections):
                incumbent_selection = list(map(int, hr.selections))
                incumbent_exact_value = _selection_objective_fraction(
                    objective_arrays, incumbent_selection
                )
                incumbent_value = float(incumbent_exact_value)
                incumbent_theta = float(hr.theta)
        except Exception:
            pass

    theta_upper_bounds: List[float] = []
    interruption_status: Optional[str] = None
    message = ""

    def get_cache(theta_value: float) -> FixedThetaCache:
        if theta_value not in cache_by_theta:
            cache_by_theta[theta_value] = _build_fixed_theta_cache(instance, theta_value, tol)
        return cache_by_theta[theta_value]

    def get_lp_bound(theta_value: float) -> FixedThetaLPBoundResult:
        nonlocal total_root_lp_time
        if theta_value not in lp_bound_by_theta:
            lp = compute_fixed_theta_lp_upper_bound(
                instance,
                theta_value,
                tol,
                cache=get_cache(theta_value) if cfg.use_caches else None,
            )
            lp_bound_by_theta[theta_value] = lp
            total_root_lp_time += float(lp.runtime_seconds)
        return lp_bound_by_theta[theta_value]

    if cfg.theta_order not in {"increasing", "lp_bound_desc", "heuristic_incumbent_desc", "hybrid"}:
        return _global_result(
            instance=instance,
            status="error",
            objective_value=float("-inf"),
            selected_options=None,
            selected_theta=None,
            lower_bound=float("-inf"),
            upper_bound=float("inf"),
            start_time=start,
            config=cfg,
            records=[],
            total_nodes=0,
            total_root_lp_time=0.0,
            total_bnb_time=0.0,
            diagnostics={"error": f"unsupported theta_order={cfg.theta_order}"},
            message=f"unsupported theta_order={cfg.theta_order}",
        )

    # Gap-accounting invariant: before any finite global anytime gap can be
    # reported, every theta has an initialized upper-bound record.  This also
    # makes increasing-order limited runs as auditable as LP-bound-ordered runs.
    ub_init_start = time.perf_counter()
    for theta in candidates:
        get_lp_bound(theta)
    valid_ub_count = 0
    invalid_ub_thetas: List[float] = []
    for theta in candidates:
        lp = lp_bound_by_theta[float(theta)]
        ub = float(lp.lp_upper_bound)
        if math.isfinite(ub) or (math.isinf(ub) and ub < 0.0):
            valid_ub_count += 1
        else:
            invalid_ub_thetas.append(float(theta))
    diagnostics.update(
        {
            "theta_upper_bound_records_initialized": True,
            "theta_upper_bound_record_count": int(len(lp_bound_by_theta)),
            "theta_upper_bound_valid_record_count": int(valid_ub_count),
            "theta_upper_bound_missing_count": int(max(0, len(candidates) - len(lp_bound_by_theta))),
            "theta_upper_bound_invalid_count": int(len(invalid_ub_thetas)),
            "theta_upper_bound_initialization_policy": "root_lp_for_all_theta",
            "theta_upper_bound_initialization_time_seconds": float(time.perf_counter() - ub_init_start),
            "theta_upper_bound_invalid_thetas": invalid_ub_thetas[:20],
        }
    )

    if cfg.use_multistart_incumbent and candidates:
        seed_start = time.perf_counter()
        seed_evaluated = 0
        seed_fixed_feasible = 0
        seed_robust_feasible = 0
        seed_improvements = 0
        seed_pair_evaluations = 0
        ranked_all = sorted(
            candidates,
            key=lambda th: (
                float("-inf") if not math.isfinite(get_lp_bound(th).lp_upper_bound) else get_lp_bound(th).lp_upper_bound,
                -float(th),
            ),
            reverse=True,
        )
        max_root_lp_bound = max(
            [float("-inf")]
            + [float(get_lp_bound(th).lp_upper_bound) for th in ranked_all if math.isfinite(get_lp_bound(th).lp_upper_bound)]
        )
        multistart_skipped_closed = bool(
            math.isfinite(incumbent_value)
            and incumbent_value > max_root_lp_bound
        )
        ranked_for_seed = [] if multistart_skipped_closed else ranked_all[: max(0, int(cfg.multistart_theta_count))]
        for seed_theta in ranked_for_seed:
            lp = get_lp_bound(seed_theta)
            if lp.infeasible_capacity or not lp.lp_feasible:
                continue
            cache = get_cache(seed_theta)
            sel, val, _used = _greedy_incumbent(cache.data.values, cache.data.costs, cache.option_sets, cache.data.capacity, tol)
            seed_evaluated += 1
            if sel is None:
                continue
            seed_fixed_feasible += 1
            if cfg.multistart_use_local_improvement:
                sel, val, _used, swaps, pair_evals = _local_improve_incumbent(
                    cache.data.values,
                    cache.data.costs,
                    cache.option_sets,
                    cache.data.capacity,
                    sel,
                    tol,
                    max_passes=cfg.local_search_max_passes,
                    max_swaps=cfg.local_search_max_swaps,
                    neighborhood=cfg.local_search_neighborhood,
                    max_pair_evaluations=cfg.local_search_max_pair_evaluations,
                )
                seed_pair_evaluations += int(pair_evals)
                if swaps:
                    diagnostics["multistart_local_swaps"] = int(diagnostics.get("multistart_local_swaps", 0)) + int(swaps)
            flags = validate_fixed_theta_selection(cache.data.values, cache.data.costs, cache.data.capacity, sel, tol)
            if not (flags["valid_length"] and flags["valid_indices"] and flags["capacity_feasible"]):
                continue
            cert = compute_certificate(instance, sel)
            if not certificate_is_feasible(instance, sel):
                continue
            seed_robust_feasible += 1
            candidate_exact = _selection_objective_fraction(
                cache.data.values, sel
            )
            if (
                incumbent_exact_value is None
                or candidate_exact > incumbent_exact_value
            ):
                incumbent_selection = list(map(int, sel))
                incumbent_value = float(val)
                incumbent_exact_value = candidate_exact
                incumbent_theta = float(seed_theta)
                seed_improvements += 1
        diagnostics.update(
            {
                "multistart_incumbent_enabled": True,
                "multistart_theta_count_requested": int(cfg.multistart_theta_count),
                "multistart_skipped_existing_incumbent_closes_root_bound": bool(multistart_skipped_closed),
                "multistart_max_root_lp_bound": float(max_root_lp_bound),
                "multistart_theta_count_evaluated": int(seed_evaluated),
                "multistart_fixed_feasible_count": int(seed_fixed_feasible),
                "multistart_robust_feasible_count": int(seed_robust_feasible),
                "multistart_incumbent_improvements": int(seed_improvements),
                "multistart_pair_evaluations": int(seed_pair_evaluations),
                "multistart_time_seconds": float(time.perf_counter() - seed_start),
                "multistart_final_incumbent_value": float(incumbent_value),
            }
        )
    else:
        diagnostics.update(
            {
                "multistart_incumbent_enabled": False,
                "multistart_theta_count_evaluated": 0,
                "multistart_fixed_feasible_count": 0,
                "multistart_robust_feasible_count": 0,
                "multistart_incumbent_improvements": 0,
                "multistart_pair_evaluations": 0,
                "multistart_time_seconds": 0.0,
            }
        )

    ordered_candidates = list(candidates)
    heuristic_order_scores: Dict[float, Tuple[float, float, float]] = {}
    if cfg.theta_order in {"heuristic_incumbent_desc", "hybrid"}:
        heuristic_start = time.perf_counter()
        heuristic_evaluated = 0
        heuristic_fixed_feasible = 0
        heuristic_robust_feasible = 0
        heuristic_incumbent_improvements = 0
        for theta in candidates:
            lp = get_lp_bound(theta)
            lp_score = float("-inf") if not math.isfinite(lp.lp_upper_bound) else float(lp.lp_upper_bound)
            robust_value = float("-inf")
            if not lp.infeasible_capacity and lp.lp_feasible:
                heuristic_evaluated += 1
                cache = get_cache(theta)
                sel, val, _used = _greedy_incumbent(
                    cache.data.values,
                    cache.data.costs,
                    cache.option_sets,
                    cache.data.capacity,
                    tol,
                )
                if sel is not None:
                    flags = validate_fixed_theta_selection(cache.data.values, cache.data.costs, cache.data.capacity, sel, tol)
                    if flags["valid_length"] and flags["valid_indices"] and flags["capacity_feasible"]:
                        heuristic_fixed_feasible += 1
                        cert = compute_certificate(instance, sel)
                        if certificate_is_feasible(instance, sel):
                            heuristic_robust_feasible += 1
                            robust_value = float(val)
                            candidate_exact = _selection_objective_fraction(
                                cache.data.values, sel
                            )
                            if (
                                incumbent_exact_value is None
                                or candidate_exact > incumbent_exact_value
                            ):
                                incumbent_selection = list(map(int, sel))
                                incumbent_value = robust_value
                                incumbent_exact_value = candidate_exact
                                incumbent_theta = float(theta)
                                heuristic_incumbent_improvements += 1
            heuristic_order_scores[float(theta)] = (robust_value, lp_score, -float(theta))
        diagnostics.update(
            {
                "heuristic_theta_order_enabled": True,
                "heuristic_theta_order_evaluated": int(heuristic_evaluated),
                "heuristic_theta_order_fixed_feasible": int(heuristic_fixed_feasible),
                "heuristic_theta_order_robust_feasible": int(heuristic_robust_feasible),
                "heuristic_theta_order_incumbent_improvements": int(heuristic_incumbent_improvements),
                "heuristic_theta_order_time_seconds": float(time.perf_counter() - heuristic_start),
                "heuristic_theta_order_final_incumbent_value": float(incumbent_value),
            }
        )
    else:
        diagnostics.update(
            {
                "heuristic_theta_order_enabled": False,
                "heuristic_theta_order_evaluated": 0,
                "heuristic_theta_order_fixed_feasible": 0,
                "heuristic_theta_order_robust_feasible": 0,
                "heuristic_theta_order_incumbent_improvements": 0,
                "heuristic_theta_order_time_seconds": 0.0,
            }
        )

    if cfg.theta_order in {"lp_bound_desc", "hybrid", "heuristic_incumbent_desc"}:
        for theta in candidates:
            get_lp_bound(theta)
        if cfg.theta_order == "heuristic_incumbent_desc":
            ordered_candidates = sorted(
                candidates,
                key=lambda th: heuristic_order_scores.get(
                    float(th),
                    (
                        float("-inf"),
                        float("-inf") if not math.isfinite(get_lp_bound(th).lp_upper_bound) else get_lp_bound(th).lp_upper_bound,
                        -float(th),
                    ),
                ),
                reverse=True,
            )
        elif cfg.theta_order == "hybrid":
            ordered_candidates = sorted(
                candidates,
                key=lambda th: (
                    float("-inf") if not math.isfinite(get_lp_bound(th).lp_upper_bound) else get_lp_bound(th).lp_upper_bound,
                    heuristic_order_scores.get(float(th), (float("-inf"), float("-inf"), -float(th)))[0],
                    -float(th),
                ),
                reverse=True,
            )
        else:
            ordered_candidates = sorted(
                candidates,
                key=lambda th: (
                    float("-inf") if not math.isfinite(get_lp_bound(th).lp_upper_bound) else get_lp_bound(th).lp_upper_bound,
                    -float(th),
                ),
                reverse=True,
            )

    for idx, theta in enumerate(ordered_candidates):
        elapsed = time.perf_counter() - start
        if cfg.time_limit_seconds is not None and elapsed >= cfg.time_limit_seconds:
            interruption_status = "time_limit"
            message = "global time limit reached"
            for rem_theta in ordered_candidates[idx:]:
                lp_bound = get_lp_bound(rem_theta)
                ub = lp_bound.lp_upper_bound
                theta_upper_bounds.append(ub)
                records.append(
                    GlobalThetaRecord(
                        theta=float(rem_theta),
                        status="not_run_time_limit",
                        fixed_theta_capacity=lp_bound.capacity,
                        fixed_theta_lp_upper_bound=ub,
                        incumbent_before_theta=incumbent_value,
                        bnb_objective=float("-inf"),
                        bnb_upper_bound=ub,
                        bnb_gap=float("inf"),
                        nodes_explored=0,
                        runtime_seconds=0.0,
                        pruned_by_bound=False,
                        infeasible_capacity=lp_bound.infeasible_capacity,
                        root_lp_status=lp_bound.root_lp_status,
                        incumbent_after_theta=incumbent_value,
                        root_lp_runtime_seconds=lp_bound.runtime_seconds,
                    )
                )
            break
        if cfg.node_limit is not None and total_nodes >= cfg.node_limit:
            interruption_status = "node_limit"
            message = "global node limit reached"
            for rem_theta in ordered_candidates[idx:]:
                lp_bound = get_lp_bound(rem_theta)
                ub = lp_bound.lp_upper_bound
                theta_upper_bounds.append(ub)
                records.append(
                    GlobalThetaRecord(
                        theta=float(rem_theta),
                        status="not_run_node_limit",
                        fixed_theta_capacity=lp_bound.capacity,
                        fixed_theta_lp_upper_bound=ub,
                        incumbent_before_theta=incumbent_value,
                        bnb_objective=float("-inf"),
                        bnb_upper_bound=ub,
                        bnb_gap=float("inf"),
                        nodes_explored=0,
                        runtime_seconds=0.0,
                        pruned_by_bound=False,
                        infeasible_capacity=lp_bound.infeasible_capacity,
                        root_lp_status=lp_bound.root_lp_status,
                        incumbent_after_theta=incumbent_value,
                        root_lp_runtime_seconds=lp_bound.runtime_seconds,
                    )
                )
            break

        incumbent_before = incumbent_value
        lp_bound = get_lp_bound(theta)
        if (
            lp_bound.root_lp_status == "error"
            or math.isnan(float(lp_bound.lp_upper_bound))
            or (math.isinf(float(lp_bound.lp_upper_bound)) and float(lp_bound.lp_upper_bound) > 0.0)
        ):
            theta_upper_bounds.append(float("inf"))
            if interruption_status is None:
                interruption_status = "error"
                message = lp_bound.message or "fixed-theta root LP upper-bound initialization failed"
            records.append(
                GlobalThetaRecord(
                    theta=float(theta),
                    status="error",
                    fixed_theta_capacity=lp_bound.capacity,
                    fixed_theta_lp_upper_bound=lp_bound.lp_upper_bound,
                    incumbent_before_theta=incumbent_before,
                    bnb_objective=float("-inf"),
                    bnb_upper_bound=float("inf"),
                    bnb_gap=float("inf"),
                    nodes_explored=0,
                    runtime_seconds=0.0,
                    pruned_by_bound=False,
                    infeasible_capacity=False,
                    root_lp_status=lp_bound.root_lp_status,
                    incumbent_after_theta=incumbent_value,
                    root_lp_runtime_seconds=lp_bound.runtime_seconds,
                    diagnostics={"root_lp_message": lp_bound.message},
                )
            )
            continue
        if lp_bound.infeasible_capacity or not lp_bound.lp_feasible:
            theta_upper_bounds.append(float("-inf"))
            records.append(
                GlobalThetaRecord(
                    theta=float(theta),
                    status="infeasible_capacity" if lp_bound.infeasible_capacity else "infeasible",
                    fixed_theta_capacity=lp_bound.capacity,
                    fixed_theta_lp_upper_bound=lp_bound.lp_upper_bound,
                    incumbent_before_theta=incumbent_before,
                    bnb_objective=float("-inf"),
                    bnb_upper_bound=float("-inf"),
                    bnb_gap=float("inf"),
                    nodes_explored=0,
                    runtime_seconds=0.0,
                    pruned_by_bound=False,
                    infeasible_capacity=lp_bound.infeasible_capacity,
                    root_lp_status=lp_bound.root_lp_status,
                    incumbent_after_theta=incumbent_value,
                    root_lp_runtime_seconds=lp_bound.runtime_seconds,
                )
            )
            continue

        if lp_bound.lp_upper_bound < incumbent_value:
            theta_upper_bounds.append(lp_bound.lp_upper_bound)
            records.append(
                GlobalThetaRecord(
                    theta=float(theta),
                    status="pruned_bound",
                    fixed_theta_capacity=lp_bound.capacity,
                    fixed_theta_lp_upper_bound=lp_bound.lp_upper_bound,
                    incumbent_before_theta=incumbent_before,
                    bnb_objective=float("-inf"),
                    bnb_upper_bound=lp_bound.lp_upper_bound,
                    bnb_gap=0.0,
                    nodes_explored=0,
                    runtime_seconds=0.0,
                    pruned_by_bound=True,
                    infeasible_capacity=False,
                    root_lp_status=lp_bound.root_lp_status,
                    incumbent_after_theta=incumbent_value,
                    root_lp_runtime_seconds=lp_bound.runtime_seconds,
                    robust_certificate_passed=incumbent_selection is not None
                    and certificate_is_feasible(instance, incumbent_selection),
                )
            )
            continue

        fixed_time: Optional[float] = cfg.fixed_theta_time_limit_seconds
        if cfg.time_limit_seconds is not None:
            remaining = max(0.0, cfg.time_limit_seconds - (time.perf_counter() - start))
            fixed_time = remaining if fixed_time is None else min(fixed_time, remaining)
        fixed_nodes: Optional[int] = cfg.fixed_theta_node_limit
        if cfg.node_limit is not None:
            remaining_nodes = max(0, cfg.node_limit - total_nodes)
            fixed_nodes = remaining_nodes if fixed_nodes is None else min(fixed_nodes, remaining_nodes)

        initial_selection = incumbent_selection if incumbent_selection is not None else None
        initial_feasible_for_theta = False
        if initial_selection is not None:
            theta_cache_for_validation = get_cache(theta) if cfg.use_caches else _build_fixed_theta_cache(instance, theta, tol)
            initial_flags = validate_fixed_theta_selection(
                theta_cache_for_validation.data.values,
                theta_cache_for_validation.data.costs,
                theta_cache_for_validation.data.capacity,
                initial_selection,
                tol,
            )
            initial_feasible_for_theta = bool(
                initial_flags["valid_length"] and initial_flags["valid_indices"] and initial_flags["capacity_feasible"]
            )

        bnb_start = time.perf_counter()
        bnb = solve_fixed_theta_bnb(
            instance,
            theta,
            FixedThetaBNBConfig(
                tolerance=tol,
                time_limit_seconds=fixed_time,
                node_limit=fixed_nodes,
                use_greedy_incumbent=cfg.use_fixed_theta_greedy_incumbent,
                objective_cutoff=incumbent_value if cfg.use_objective_cutoff and math.isfinite(incumbent_value) else None,
                initial_incumbent_selection=initial_selection if initial_feasible_for_theta else None,
                initial_incumbent_value=incumbent_value if initial_feasible_for_theta and math.isfinite(incumbent_value) else None,
                use_cutoff_pruning=cfg.use_objective_cutoff,
                use_cache=cfg.use_caches,
                branching_rule=cfg.branching_rule,
                child_ordering=cfg.child_ordering,
                strong_branching_candidates=cfg.strong_branching_candidates,
                strong_branching_depth_limit=cfg.strong_branching_depth_limit,
                strong_branching_max_children=cfg.strong_branching_max_children,
                strong_branching_use_fast_bound=cfg.strong_branching_use_fast_bound,
                incumbent_heuristic=cfg.incumbent_heuristic,
                use_local_incumbent_improvement=cfg.use_local_incumbent_improvement,
                local_search_max_passes=cfg.local_search_max_passes,
                local_search_max_swaps=cfg.local_search_max_swaps,
                local_search_neighborhood=cfg.local_search_neighborhood,
                local_search_max_pair_evaluations=cfg.local_search_max_pair_evaluations,
                collect_diagnostics=cfg.collect_diagnostics,
                max_diagnostic_nodes=cfg.max_diagnostic_nodes,
                diagnostic_sample_rate=cfg.diagnostic_sample_rate,
                profile_timing=cfg.profile_timing,
                use_fast_residual_lp_bound=cfg.use_fast_residual_lp_bound,
                use_cheap_prebound=cfg.use_cheap_prebound,
                use_min_cost_infeasibility_check=cfg.use_min_cost_infeasibility_check,
                use_bound_cache=cfg.use_bound_cache,
                bound_cache_max_entries=cfg.bound_cache_max_entries,
            ),
            fixed_theta_cache=get_cache(theta) if cfg.use_caches else None,
        )
        total_bnb_time += time.perf_counter() - bnb_start
        total_nodes += int(bnb.nodes_explored)
        theta_upper_bounds.append(bnb.upper_bound if math.isfinite(bnb.upper_bound) else lp_bound.lp_upper_bound)

        if bnb.status in {"time_limit", "node_limit"} and interruption_status is None:
            interruption_status = bnb.status
            message = bnb.message

        if bnb.status == "error" and interruption_status is None:
            interruption_status = "error"
            message = bnb.message

        robust_passed = False
        bnb_exact_value = (
            _selection_objective_fraction(objective_arrays, bnb.selected_options)
            if bnb.selected_options is not None
            else None
        )
        if bnb.selected_options is not None and bnb_exact_value is not None and (
            incumbent_exact_value is None
            or bnb_exact_value > incumbent_exact_value
        ):
            cert = compute_certificate(instance, bnb.selected_options)
            if certificate_is_feasible(instance, bnb.selected_options):
                incumbent_selection = list(map(int, bnb.selected_options))
                incumbent_value = float(bnb.objective_value)
                incumbent_exact_value = bnb_exact_value
                incumbent_theta = float(theta)
                robust_passed = True
        elif bnb.selected_options is not None:
            robust_passed = certificate_is_feasible(instance, bnb.selected_options)

        records.append(
            GlobalThetaRecord(
                theta=float(theta),
                status=bnb.status,
                fixed_theta_capacity=bnb.capacity,
                fixed_theta_lp_upper_bound=lp_bound.lp_upper_bound,
                incumbent_before_theta=incumbent_before,
                bnb_objective=bnb.objective_value,
                bnb_upper_bound=bnb.upper_bound,
                bnb_gap=bnb.absolute_gap,
                nodes_explored=bnb.nodes_explored,
                runtime_seconds=bnb.runtime_seconds,
                pruned_by_bound=False,
                infeasible_capacity=False,
                root_lp_status=lp_bound.root_lp_status,
                incumbent_after_theta=incumbent_value,
                bnb_lower_bound=bnb.lower_bound,
                nodes_pruned_bound=bnb.nodes_pruned_bound,
                nodes_pruned_infeasible=bnb.nodes_pruned_infeasible,
                cutoff_used=cfg.use_objective_cutoff and math.isfinite(incumbent_before),
                initial_incumbent_feasible_for_theta=initial_feasible_for_theta,
                robust_certificate_passed=robust_passed,
                root_lp_runtime_seconds=lp_bound.runtime_seconds,
                diagnostics=bnb.diagnostics if cfg.collect_diagnostics or cfg.profile_timing else {},
            )
        )

    missing_ub_records = [float(theta) for theta in candidates if float(theta) not in lp_bound_by_theta]
    invalid_final_ub_records = [
        float(theta)
        for theta in candidates
        if float(theta) in lp_bound_by_theta
        and (
            math.isnan(float(lp_bound_by_theta[float(theta)].lp_upper_bound))
            or (
                math.isinf(float(lp_bound_by_theta[float(theta)].lp_upper_bound))
                and float(lp_bound_by_theta[float(theta)].lp_upper_bound) > 0.0
            )
        )
    ]
    diagnostics["theta_upper_bound_missing_count"] = int(len(missing_ub_records))
    diagnostics["theta_upper_bound_invalid_count"] = int(len(invalid_final_ub_records))
    if missing_ub_records or invalid_final_ub_records:
        theta_upper_bounds.append(float("inf"))
        if interruption_status is None:
            interruption_status = "error"
            message = "finite global gap unavailable because some theta upper-bound records are missing or invalid"

    upper_bound = max(theta_upper_bounds + ([incumbent_value] if math.isfinite(incumbent_value) else []), default=float("-inf"))
    lower_bound = incumbent_value if incumbent_selection is not None else float("-inf")

    if incumbent_selection is not None and upper_bound <= lower_bound + tol:
        status = "optimal"
    elif incumbent_selection is None and all(
        r.status in {"infeasible", "infeasible_capacity", "pruned_bound"} for r in records
    ):
        status = "infeasible"
    elif interruption_status in {"time_limit", "node_limit", "error"}:
        status = interruption_status
    else:
        unresolved = [r for r in records if r.status in {"time_limit", "node_limit", "error", "not_run_time_limit", "not_run_node_limit"}]
        if unresolved:
            status = "time_limit" if any("time" in r.status for r in unresolved) else "node_limit"
        else:
            status = "optimal" if incumbent_selection is not None else "infeasible"

    if cfg.collect_diagnostics or cfg.profile_timing:
        unresolved_records = [
            r
            for r in records
            if r.status in {"time_limit", "node_limit", "error", "not_run_time_limit", "not_run_node_limit"}
            or (math.isfinite(r.bnb_upper_bound) and math.isfinite(incumbent_value) and r.bnb_upper_bound > incumbent_value + tol)
        ]
        unresolved_records.sort(
            key=lambda r: (
                float("-inf") if not math.isfinite(r.bnb_upper_bound) else r.bnb_upper_bound,
                -float(r.theta),
            ),
            reverse=True,
        )
        diagnostics.update(
            {
                "theta_total": len(records),
                "theta_infeasible_capacity": sum(1 for r in records if r.infeasible_capacity),
                "theta_pruned_by_root_lp": sum(1 for r in records if r.pruned_by_bound),
                "theta_solved_optimal": sum(1 for r in records if r.status == "optimal"),
                "theta_limited": sum(1 for r in records if r.status in {"time_limit", "node_limit", "not_run_time_limit", "not_run_node_limit"}),
                "theta_error": sum(1 for r in records if r.status == "error"),
                "theta_order_used": cfg.theta_order,
                "total_nodes_pruned_bound": sum(int(r.nodes_pruned_bound) for r in records),
                "total_nodes_pruned_infeasible": sum(int(r.nodes_pruned_infeasible) for r in records),
                "total_nodes_pruned_cutoff": sum(int(r.diagnostics.get("nodes_pruned_cutoff", 0)) for r in records),
                "total_integral_lp_nodes": sum(int(r.diagnostics.get("nodes_integral_lp", 0)) for r in records),
                "total_fractional_branches": sum(int(r.diagnostics.get("branch_fractional_item_count", 0)) for r in records),
                "total_fallback_branches": sum(int(r.diagnostics.get("branch_fallback_count", 0)) for r in records),
                "total_branching_nodes": sum(int(r.diagnostics.get("branching_nodes", 0)) for r in records),
                "total_child_count": sum(int(r.diagnostics.get("total_child_count", 0)) for r in records),
                "total_root_lp_time": float(total_root_lp_time),
                "total_fixed_theta_bnb_time": float(total_bnb_time),
                "total_time_node_lp": sum(float(r.diagnostics.get("time_node_lp_total", 0.0)) for r in records),
                "total_time_hull_build": sum(float(r.diagnostics.get("time_hull_build_total", 0.0)) for r in records),
                "total_time_greedy_lp": sum(float(r.diagnostics.get("time_greedy_lp_total", 0.0)) for r in records),
                "total_time_fast_bound": sum(float(r.diagnostics.get("time_fast_bound", 0.0)) for r in records),
                "total_time_reference_bound": sum(float(r.diagnostics.get("time_reference_bound", 0.0)) for r in records),
                "total_time_cheap_prebound": sum(float(r.diagnostics.get("time_cheap_prebound", 0.0)) for r in records),
                "total_time_min_cost_check": sum(float(r.diagnostics.get("time_min_cost_check", 0.0)) for r in records),
                "total_time_branching": sum(float(r.diagnostics.get("time_branching_total", 0.0)) for r in records),
                "total_time_child_generation": sum(float(r.diagnostics.get("time_child_generation_total", 0.0)) for r in records),
                "total_cheap_prebound_prunes": sum(int(r.diagnostics.get("cheap_prebound_prunes", 0)) for r in records),
                "total_min_cost_infeasibility_prunes": sum(int(r.diagnostics.get("min_cost_infeasibility_prunes", 0)) for r in records),
                "total_fast_lp_bounds_computed": sum(int(r.diagnostics.get("fast_lp_bounds_computed", 0)) for r in records),
                "total_exact_lp_bounds_computed": sum(int(r.diagnostics.get("exact_lp_bounds_computed", 0)) for r in records),
                "total_reference_fallback_count": sum(int(r.diagnostics.get("reference_fallback_count", 0)) for r in records),
                "total_bound_cache_hits": sum(int(r.diagnostics.get("bound_cache_hits", 0)) for r in records),
                "total_bound_cache_misses": sum(int(r.diagnostics.get("bound_cache_misses", 0)) for r in records),
                "total_strong_branching_count": sum(int(r.diagnostics.get("strong_branching_count", 0)) for r in records),
                "total_strong_branching_time": sum(float(r.diagnostics.get("strong_branching_time", 0.0)) for r in records),
                "total_strong_branching_candidates_evaluated": sum(int(r.diagnostics.get("strong_branching_candidates_evaluated", 0)) for r in records),
                "total_local_incumbent_swaps": sum(int(r.diagnostics.get("local_incumbent_swaps", 0)) for r in records),
                "total_local_incumbent_pair_evaluations": sum(int(r.diagnostics.get("local_incumbent_pair_evaluations", 0)) for r in records),
                "unresolved_theta": [
                    {
                        "theta": float(r.theta),
                        "status": r.status,
                        "upper_bound": float(r.bnb_upper_bound),
                        "lower_bound": float(r.bnb_lower_bound),
                        "relative_gap": float(_relative_gap(r.bnb_upper_bound, r.bnb_lower_bound, tol)),
                        "nodes_explored": int(r.nodes_explored),
                        "runtime_seconds": float(r.runtime_seconds),
                    }
                    for r in unresolved_records[: max(0, min(50, cfg.max_diagnostic_nodes))]
                ],
            }
        )
        branching_nodes = int(diagnostics["total_branching_nodes"])
        diagnostics["average_branching_arity"] = (
            float(int(diagnostics["total_child_count"]) / branching_nodes) if branching_nodes else 0.0
        )
        diagnostics["max_branching_arity"] = max((int(r.diagnostics.get("max_branching_arity", 0)) for r in records), default=0)

    return _global_result(
        instance=instance,
        status=status,
        objective_value=incumbent_value,
        selected_options=incumbent_selection,
        selected_theta=incumbent_theta,
        lower_bound=lower_bound,
        upper_bound=upper_bound if incumbent_selection is not None or math.isfinite(upper_bound) else float("-inf"),
        start_time=start,
        config=cfg,
        records=records,
        total_nodes=total_nodes,
        total_root_lp_time=total_root_lp_time,
        total_bnb_time=total_bnb_time,
        diagnostics=diagnostics,
        message=message,
    )


def brute_force_global_robust(instance: PricingInstance, tol: float = EPS) -> GlobalThetaBNBResult:
    """Brute-force global robust MCKP solver for tiny tests only."""

    _require_representable_objective_range(instance)
    start = time.perf_counter()
    best_sel: Optional[List[int]] = None
    best_obj = float("-inf")
    best_exact_obj: Optional[Fraction] = None
    objective_arrays = [
        np.asarray([option.value for option in group], dtype=float)
        for group in instance.items
    ]
    for combo in itertools.product(*[range(len(group)) for group in instance.items]):
        cert = compute_certificate(instance, combo)
        if not certificate_is_feasible(instance, combo):
            continue
        exact_obj = _selection_objective_fraction(objective_arrays, combo)
        obj = float(exact_obj)
        if (
            best_exact_obj is None
            or exact_obj > best_exact_obj
            or (
                exact_obj == best_exact_obj
                and list(combo) < (best_sel or [math.inf])
            )
        ):
            best_obj = obj
            best_exact_obj = exact_obj
            best_sel = list(map(int, combo))
    records: List[GlobalThetaRecord] = []
    status = "optimal" if best_sel is not None else "infeasible"
    cert = compute_certificate(instance, best_sel) if best_sel is not None else float("-inf")
    return GlobalThetaBNBResult(
        status=status,
        objective_value=best_obj if best_sel is not None else float("-inf"),
        selected_options=best_sel,
        selected_theta=None,
        robust_certificate=cert,
        lower_bound=best_obj if best_sel is not None else float("-inf"),
        upper_bound=best_obj if best_sel is not None else float("-inf"),
        absolute_gap=0.0 if best_sel is not None else float("inf"),
        relative_gap=0.0 if best_sel is not None else float("inf"),
        theta_count_total=0,
        theta_count_infeasible_capacity=0,
        theta_count_pruned_by_bound=0,
        theta_count_solved_optimal=0,
        theta_count_limited=0,
        theta_count_error=0,
        total_nodes_explored=0,
        total_runtime_seconds=float(time.perf_counter() - start),
        per_theta_records=records,
        config_metadata={"mode": "brute_force_global_robust", "tolerance": tol},
        validation_flags=_validate_global_selection(instance, best_sel, best_obj, tol),
        message="brute force global robust reference",
    )
