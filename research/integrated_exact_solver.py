"""Exact robust-MCKP solution by best-first threshold-interval search.

The search uses an interval upper bound on the complete family of fixed-
threshold LP relaxations.  It solves selected singleton threshold MCKPs with
the repository's exact branch-and-bound solver and prunes a whole threshold
interval whenever its LP upper bound cannot improve the global incumbent.

Two interval bounds are supported under an identical search policy:

* ``envelope``: the compressed simultaneous group-envelope oracle;
* ``clique``: the bounded-threshold group-clique LP comparator.

Because every integer threshold problem is bounded by its LP relaxation, both
variants are exact up to the configured scaled tolerance.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from fractions import Fraction

from robust_mckp import PricingInstance
from robust_mckp.certificate import certificate_is_feasible, compute_certificate
from robust_mckp.exact_bnb import (
    FixedThetaBNBConfig,
    FixedThetaBNBResult,
    _selection_objective_fraction,
    build_full_theta_candidates,
    solve_fixed_theta_bnb,
)
from robust_mckp.solver import solve as solve_hullround
from research.compressed_interval_oracle import CompressedThetaIntervalOracle
from research.structural_feasibility_study import bounded_theta_clique_lp


BoundKind = Literal["envelope", "clique"]


@dataclass(frozen=True)
class IntervalExactConfig:
    """Configuration shared by both exact interval-search variants."""

    bound_kind: BoundKind = "envelope"
    tolerance: float = 1e-7
    time_limit_seconds: float = 30.0
    seed_endpoint_midpoint: bool = True
    solve_split_midpoints: bool = False
    use_hullround_incumbent: bool = True


@dataclass
class IntervalExactResult:
    """Certified or anytime result from exact threshold-interval search."""

    status: str
    objective_value: float
    selected_options: Optional[list[int]]
    selected_theta: Optional[float]
    lower_bound: float
    upper_bound: float
    relative_gap: float
    runtime_seconds: float
    interval_bound_evaluations: int
    interval_splits: int
    interval_prunes: int
    theta_integer_solves: int
    theta_optimal_solves: int
    total_nodes: int
    thresholds_total: int
    thresholds_covered_by_prunes: int
    bound_seconds: float
    fixed_theta_seconds: float
    diagnostics: dict[str, object] = field(default_factory=dict)


def solve_interval_exact(
    instance: PricingInstance,
    config: Optional[IntervalExactConfig] = None,
) -> IntervalExactResult:
    """Solve the global robust MCKP by exact threshold-interval search."""

    cfg = config or IntervalExactConfig()
    if cfg.bound_kind not in {"envelope", "clique"}:
        raise ValueError(f"unsupported bound_kind={cfg.bound_kind}")
    start = time.perf_counter()
    thetas = np.asarray(build_full_theta_candidates(instance), dtype=float)
    envelope = (
        CompressedThetaIntervalOracle(instance)
        if cfg.bound_kind == "envelope"
        else None
    )
    if envelope is not None:
        thetas = np.asarray(envelope.thetas, dtype=float)

    interval_bound_evaluations = 0
    interval_splits = 0
    interval_prunes = 0
    thresholds_covered_by_prunes = 0
    bound_seconds = 0.0
    fixed_theta_seconds = 0.0
    total_nodes = 0
    theta_results: dict[int, FixedThetaBNBResult] = {}
    incumbent_value = float("-inf")
    incumbent_exact_value: Optional[Fraction] = None
    incumbent_selection: Optional[list[int]] = None
    incumbent_theta: Optional[float] = None
    error_message = ""
    objective_arrays = [
        np.asarray([option.value for option in group], dtype=float)
        for group in instance.items
    ]

    if cfg.use_hullround_incumbent:
        try:
            heuristic = solve_hullround(instance, upgrade_completion=True)
            if (
                heuristic.is_feasible
                and heuristic.selections
                and certificate_is_feasible(instance, heuristic.selections)
            ):
                incumbent_value = float(heuristic.objective)
                incumbent_selection = list(map(int, heuristic.selections))
                incumbent_exact_value = _selection_objective_fraction(
                    objective_arrays, incumbent_selection
                )
                incumbent_value = float(incumbent_exact_value)
                incumbent_theta = float(heuristic.theta)
        except Exception:
            pass

    def elapsed() -> float:
        return time.perf_counter() - start

    def scaled_tolerance() -> float:
        return float(cfg.tolerance) * max(
            1.0, abs(incumbent_value) if math.isfinite(incumbent_value) else 1.0
        )

    def interval_bound(lo: int, hi: int) -> float:
        nonlocal interval_bound_evaluations, bound_seconds
        bound_start = time.perf_counter()
        if envelope is not None:
            value = float(envelope.bound(lo, hi).upper_bound)
        else:
            value, _seconds, _theta = bounded_theta_clique_lp(
                instance, float(thetas[lo]), float(thetas[hi])
            )
            value = float(value)
        bound_seconds += time.perf_counter() - bound_start
        interval_bound_evaluations += 1
        return value

    def solve_theta(index: int) -> None:
        nonlocal incumbent_value, incumbent_exact_value
        nonlocal incumbent_selection, incumbent_theta
        nonlocal fixed_theta_seconds, total_nodes, error_message
        if index in theta_results or elapsed() >= cfg.time_limit_seconds:
            return
        remaining = max(0.0, cfg.time_limit_seconds - elapsed())
        fixed_start = time.perf_counter()
        result = solve_fixed_theta_bnb(
            instance,
            float(thetas[index]),
            FixedThetaBNBConfig(
                tolerance=cfg.tolerance,
                time_limit_seconds=remaining,
                objective_cutoff=(
                    incumbent_value if math.isfinite(incumbent_value) else None
                ),
                use_cutoff_pruning=True,
                use_cache=True,
                use_greedy_incumbent=True,
            ),
        )
        fixed_theta_seconds += time.perf_counter() - fixed_start
        total_nodes += int(result.nodes_explored)
        theta_results[index] = result
        if result.status == "error":
            error_message = result.message or "fixed-threshold solver error"
        if result.selected_options is not None:
            certificate = compute_certificate(instance, result.selected_options)
            candidate_exact = _selection_objective_fraction(
                objective_arrays, result.selected_options
            )
            if (
                certificate_is_feasible(instance, result.selected_options)
                and (
                    incumbent_exact_value is None
                    or candidate_exact > incumbent_exact_value
                )
            ):
                incumbent_value = float(result.objective_value)
                incumbent_exact_value = candidate_exact
                incumbent_selection = list(map(int, result.selected_options))
                incumbent_theta = float(thetas[index])

    root_bound = interval_bound(0, len(thetas) - 1)
    queue: list[tuple[float, int, int]] = []
    if math.isfinite(root_bound):
        heapq.heappush(queue, (-root_bound, 0, len(thetas) - 1))

    if cfg.seed_endpoint_midpoint and queue:
        for index in sorted({0, len(thetas) - 1, (len(thetas) - 1) // 2}):
            solve_theta(index)

    discarded_upper = float("-inf")
    unresolved_upper = float("-inf")
    status = "time_limit"

    while queue and elapsed() < cfg.time_limit_seconds and not error_message:
        neg_bound, lo, hi = heapq.heappop(queue)
        current_bound = -float(neg_bound)
        if math.isfinite(incumbent_value) and current_bound < incumbent_value:
            discarded_upper = max(discarded_upper, current_bound)
            interval_prunes += 1
            thresholds_covered_by_prunes += hi - lo + 1
            continue

        if lo == hi:
            solve_theta(lo)
            result = theta_results.get(lo)
            if result is None:
                unresolved_upper = max(unresolved_upper, current_bound)
                break
            if result.status in {"time_limit", "node_limit"}:
                unresolved_upper = max(
                    unresolved_upper,
                    current_bound,
                    float(result.upper_bound),
                )
                break
            if result.status == "error":
                error_message = result.message or "fixed-threshold solver error"
                unresolved_upper = float("inf")
                break
            continue

        mid = (lo + hi) // 2
        if cfg.solve_split_midpoints:
            solve_theta(mid)
            if elapsed() >= cfg.time_limit_seconds:
                unresolved_upper = max(unresolved_upper, current_bound)
                break

        child_specs = ((lo, mid), (mid + 1, hi))
        children_complete = True
        for child_lo, child_hi in child_specs:
            if elapsed() >= cfg.time_limit_seconds:
                children_complete = False
                break
            if child_lo == child_hi and child_lo in theta_results:
                child_result = theta_results[child_lo]
                if child_result.status in {"optimal", "infeasible"}:
                    continue
            child_bound = interval_bound(child_lo, child_hi)
            if not math.isfinite(child_bound):
                continue
            if math.isfinite(incumbent_value) and child_bound < incumbent_value:
                discarded_upper = max(discarded_upper, child_bound)
                interval_prunes += 1
                thresholds_covered_by_prunes += child_hi - child_lo + 1
            else:
                heapq.heappush(queue, (-child_bound, child_lo, child_hi))
        if not children_complete:
            unresolved_upper = max(unresolved_upper, current_bound)
            break
        interval_splits += 1

    queue_upper = -queue[0][0] if queue else float("-inf")
    upper_bound = max(
        incumbent_value,
        discarded_upper,
        unresolved_upper,
        queue_upper,
    )
    if error_message:
        status = "error"
        upper_bound = float("inf")
    elif not queue and not math.isfinite(unresolved_upper):
        if incumbent_selection is None:
            status = "infeasible"
            upper_bound = float("-inf")
        else:
            status = "optimal"
    elif (
        incumbent_selection is not None
        and math.isfinite(upper_bound)
        and upper_bound <= incumbent_value + scaled_tolerance()
    ):
        status = "optimal"
    else:
        status = "time_limit"

    if math.isfinite(upper_bound) and math.isfinite(incumbent_value):
        relative_gap = max(0.0, upper_bound - incumbent_value) / max(
            1.0, abs(incumbent_value)
        )
    else:
        relative_gap = 0.0 if status == "infeasible" else float("inf")

    return IntervalExactResult(
        status=status,
        objective_value=(
            float(incumbent_value)
            if incumbent_selection is not None
            else float("-inf")
        ),
        selected_options=incumbent_selection,
        selected_theta=incumbent_theta,
        lower_bound=(
            float(incumbent_value)
            if incumbent_selection is not None
            else float("-inf")
        ),
        upper_bound=float(upper_bound),
        relative_gap=float(relative_gap),
        runtime_seconds=float(elapsed()),
        interval_bound_evaluations=int(interval_bound_evaluations),
        interval_splits=int(interval_splits),
        interval_prunes=int(interval_prunes),
        theta_integer_solves=len(theta_results),
        theta_optimal_solves=sum(
            result.status == "optimal" for result in theta_results.values()
        ),
        total_nodes=int(total_nodes),
        thresholds_total=len(thetas),
        thresholds_covered_by_prunes=int(thresholds_covered_by_prunes),
        bound_seconds=float(bound_seconds),
        fixed_theta_seconds=float(fixed_theta_seconds),
        diagnostics={
            "bound_kind": cfg.bound_kind,
            "root_upper_bound": float(root_bound),
            "discarded_upper_bound": float(discarded_upper),
            "unresolved_upper_bound": float(unresolved_upper),
            "error_message": error_message,
            "theta_statuses": {
                str(index): result.status
                for index, result in sorted(theta_results.items())
            },
        },
    )


__all__ = [
    "BoundKind",
    "IntervalExactConfig",
    "IntervalExactResult",
    "solve_interval_exact",
]
