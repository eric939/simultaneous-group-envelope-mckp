#!/usr/bin/env python3
"""Reproducible go/no-go experiments for a stronger robust-MCKP contribution.

This is deliberately research code rather than production solver code.  It
tests two hypotheses before any manuscript rewrite:

1. Group-aware conflict cuts and the full fixed-theta disjunction materially
   strengthen the compact Bertsimas--Sim LP relaxation on small instances.
2. A group-separable Lagrangian bound can cover an interval of theta values
   tightly enough to avoid initializing an LP relaxation at every breakpoint.

All integer-hull values in the polyhedral study are obtained by complete
enumeration.  All interval bounds are upper bounds: every reported value is an
explicit Lagrangian evaluation, not the optimizer's interpolated value.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import itertools
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import scipy
import scipy.optimize as opt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robust_mckp import GlobalThetaBNBConfig, Option, PricingInstance, solve  # noqa: E402
from robust_mckp.certificate import compute_certificate  # noqa: E402
from robust_mckp.exact_bnb import (  # noqa: E402
    FixedThetaBNBConfig,
    build_fixed_theta_data,
    build_full_theta_candidates,
    compute_fixed_theta_lp_upper_bound,
    solve_fixed_theta_bnb,
    solve_global_theta_bnb,
)
from scripts.benchmark_solvers import solve_full_robust_highs, solve_full_robust_scip  # noqa: E402
from research.benchmark_instances import build_benchmark_instance  # noqa: E402


TOL = 1e-8


@dataclass(frozen=True)
class SupportValues:
    integer_hull: float
    compact_lp: float
    disjunctive_lp: float
    conflict_lp: float


@dataclass(frozen=True)
class IntervalBound:
    upper_bound: float
    lambda_value: float
    evaluations: int
    runtime_seconds: float
    # Populated by certifying minimax oracles. Defaults preserve the original
    # four-field research interface for legacy comparators and test doubles.
    lower_bound: float = float("-inf")
    optimality_gap: float = float("inf")
    certified: bool = False


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flatten(instance: PricingInstance) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    sizes = [len(group) for group in instance.items]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    margins = np.array([option.margin for group in instance.items for option in group], dtype=float)
    deviations = np.array([abs(option.uncertainty) for group in instance.items for option in group], dtype=float)
    return sizes, offsets, margins, deviations


def _group_equalities(sizes: Sequence[int], offsets: Sequence[int], nvars: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(sizes), nvars), dtype=float)
    for i, size in enumerate(sizes):
        matrix[i, offsets[i] : offsets[i] + size] = 1.0
    return matrix, np.ones(len(sizes), dtype=float)


def compact_lp_support(
    instance: PricingInstance,
    direction: np.ndarray,
    conflicts: Sequence[tuple[tuple[int, int], ...]] = (),
) -> tuple[float, Optional[np.ndarray]]:
    """Optimize a support direction over the aggregate compact robust LP."""

    sizes, offsets, margins, deviations = _flatten(instance)
    n = instance.n_items
    n_x = offsets[-1]
    theta_idx = n_x
    pi_offset = n_x + 1
    nvars = n_x + 1 + n
    if direction.size != n_x:
        raise ValueError("direction has incorrect length")

    c = np.zeros(nvars, dtype=float)
    c[:n_x] = -direction
    a_eq, b_eq = _group_equalities(sizes, offsets, nvars)
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    # pi_i >= sum_j d_ij x_ij - theta.
    for i, size in enumerate(sizes):
        row = np.zeros(nvars, dtype=float)
        row[offsets[i] : offsets[i] + size] = deviations[offsets[i] : offsets[i] + size]
        row[theta_idx] = -1.0
        row[pi_offset + i] = -1.0
        rows.append(row)
        rhs.append(0.0)

    # sum margin*x - Gamma*theta - sum pi >= 0.
    row = np.zeros(nvars, dtype=float)
    row[:n_x] = -margins
    row[theta_idx] = float(instance.gamma)
    row[pi_offset:] = 1.0
    rows.append(row)
    rhs.append(0.0)

    # Exhaustively generated minimal group conflicts.
    for conflict in conflicts:
        row = np.zeros(nvars, dtype=float)
        for i, j in conflict:
            row[offsets[i] + j] = 1.0
        rows.append(row)
        rhs.append(float(len(conflict) - 1))

    bounds = [(0.0, 1.0)] * n_x + [(0.0, None)] * (1 + n)
    result = opt.linprog(
        c,
        A_ub=np.vstack(rows),
        b_ub=np.array(rhs, dtype=float),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if result.status == 2:
        return float("-inf"), None
    if not result.success or result.x is None:
        raise RuntimeError(f"compact LP failed: {result.message}")
    return float(-result.fun), np.asarray(result.x[:n_x], dtype=float)


def fixed_theta_lp_support(instance: PricingInstance, theta: float, direction: np.ndarray) -> float:
    data = build_fixed_theta_data(instance, float(theta))
    if data.capacity < -TOL:
        return float("-inf")
    sizes, offsets, _, _ = _flatten(instance)
    n_x = offsets[-1]
    a_eq, b_eq = _group_equalities(sizes, offsets, n_x)
    costs = np.concatenate(data.costs)
    result = opt.linprog(
        -direction,
        A_ub=costs[None, :],
        b_ub=np.array([data.capacity], dtype=float),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, 1.0)] * n_x,
        method="highs",
    )
    if result.status == 2:
        return float("-inf")
    if not result.success:
        raise RuntimeError(f"fixed-theta LP failed at theta={theta}: {result.message}")
    return float(-result.fun)


def disjunctive_lp_support(instance: PricingInstance, direction: np.ndarray) -> float:
    return max(
        fixed_theta_lp_support(instance, theta, direction)
        for theta in build_full_theta_candidates(instance)
    )


def all_assignments(instance: PricingInstance) -> list[tuple[int, ...]]:
    return list(itertools.product(*(range(len(group)) for group in instance.items)))


def feasible_assignments(instance: PricingInstance) -> list[tuple[int, ...]]:
    return [
        selection
        for selection in all_assignments(instance)
        if compute_certificate(instance, selection) >= -TOL
    ]


def integer_support(
    instance: PricingInstance,
    direction: np.ndarray,
    feasible: Optional[Sequence[tuple[int, ...]]] = None,
) -> float:
    sizes, offsets, _, _ = _flatten(instance)
    feasible_rows = list(feasible) if feasible is not None else feasible_assignments(instance)
    if not feasible_rows:
        return float("-inf")
    return max(sum(float(direction[offsets[i] + j]) for i, j in enumerate(selection)) for selection in feasible_rows)


def minimal_group_conflicts(
    instance: PricingInstance,
    feasible: Optional[Sequence[tuple[int, ...]]] = None,
) -> list[tuple[tuple[int, int], ...]]:
    """Enumerate all inclusion-minimal partial assignments with no feasible completion."""

    feasible_rows = list(feasible) if feasible is not None else feasible_assignments(instance)
    if not feasible_rows:
        return []
    choices = [range(-1, len(group)) for group in instance.items]
    conflicts: list[tuple[tuple[int, int], ...]] = []
    for pattern in itertools.product(*choices):
        assigned = tuple((i, j) for i, j in enumerate(pattern) if j >= 0)
        if not assigned:
            continue
        has_extension = any(all(selection[i] == j for i, j in assigned) for selection in feasible_rows)
        if has_extension:
            continue
        minimal = True
        for dropped in range(len(assigned)):
            subset = assigned[:dropped] + assigned[dropped + 1 :]
            if subset and not any(all(selection[i] == j for i, j in subset) for selection in feasible_rows):
                minimal = False
                break
        if minimal:
            conflicts.append(assigned)
    return conflicts


def maximum_completion_certificate(
    instance: PricingInstance,
    partial: Sequence[tuple[int, int]],
) -> float:
    """Exact best robust certificate among completions of a partial assignment.

    Once theta is fixed, maximizing the residual is separable by group, so this
    feasibility-oracle calculation needs no integer program.
    """

    fixed = dict(partial)
    best = float("-inf")
    for theta in build_full_theta_candidates(instance):
        residual = -float(instance.gamma) * float(theta)
        for i, group in enumerate(instance.items):
            if i in fixed:
                option = group[fixed[i]]
                residual += option.margin - max(0.0, abs(option.uncertainty) - theta)
            else:
                residual += max(
                    option.margin - max(0.0, abs(option.uncertainty) - theta)
                    for option in group
                )
        best = max(best, float(residual))
    return best


def shrink_group_conflict(
    instance: PricingInstance,
    assignment: Sequence[int],
    x: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    """Greedily shrink an infeasible full assignment to a minimal conflict."""

    _, offsets, _, _ = _flatten(instance)
    conflict = [(i, int(j)) for i, j in enumerate(assignment)]
    # Removing a low-x literal improves the eventual cut violation most.
    order = sorted(conflict, key=lambda pair: float(x[offsets[pair[0]] + pair[1]]))
    for literal in order:
        candidate = [pair for pair in conflict if pair != literal]
        if candidate and maximum_completion_certificate(instance, candidate) < -TOL:
            conflict = candidate
    return tuple(sorted(conflict))


def _candidate_assignments_from_lp(
    instance: PricingInstance,
    x: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> list[tuple[int, ...]]:
    sizes, offsets, _, _ = _flatten(instance)
    candidates: set[tuple[int, ...]] = set()
    argmax = tuple(
        int(np.argmax(x[offsets[i] : offsets[i] + size]))
        for i, size in enumerate(sizes)
    )
    candidates.add(argmax)
    # Deterministic one-group deviations from the modal assignment.
    for i, size in enumerate(sizes):
        ranking = np.argsort(-x[offsets[i] : offsets[i] + size])
        for alternative in ranking[1: min(3, size)]:
            candidate = list(argmax)
            candidate[i] = int(alternative)
            candidates.add(tuple(candidate))
    # Randomized rounding probes combinations of fractional group choices.
    for _ in range(samples):
        selection: list[int] = []
        for i, size in enumerate(sizes):
            probabilities = np.maximum(x[offsets[i] : offsets[i] + size], 0.0)
            total = float(np.sum(probabilities))
            if total <= TOL:
                probabilities = np.full(size, 1.0 / size)
            else:
                probabilities = probabilities / total
            selection.append(int(rng.choice(size, p=probabilities)))
        candidates.add(tuple(selection))
    return list(candidates)


def cutting_plane_support(
    instance: PricingInstance,
    direction: np.ndarray,
    *,
    seed: int,
    max_rounds: int = 20,
    samples_per_round: int = 80,
) -> dict:
    """Heuristic separation of exact-valid group-conflict inequalities."""

    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    cuts: list[tuple[tuple[int, int], ...]] = []
    cut_set: set[tuple[tuple[int, int], ...]] = set()
    bound, x = compact_lp_support(instance, direction)
    initial_bound = bound
    rounds = 0
    generated = 0
    violated_generated = 0
    for round_index in range(max_rounds):
        if x is None:
            break
        rounds = round_index + 1
        _, offsets, _, _ = _flatten(instance)
        new_cuts: list[tuple[tuple[int, int], ...]] = []
        for assignment in _candidate_assignments_from_lp(
            instance, x, rng, samples_per_round
        ):
            full = tuple((i, j) for i, j in enumerate(assignment))
            if maximum_completion_certificate(instance, full) >= -TOL:
                continue
            conflict = shrink_group_conflict(instance, assignment, x)
            generated += 1
            if conflict in cut_set:
                continue
            # Independent validity recheck before accepting a research cut.
            if maximum_completion_certificate(instance, conflict) >= -TOL:
                raise AssertionError("separator produced an invalid group conflict")
            violation = 1.0 - sum(
                1.0 - float(x[offsets[i] + j]) for i, j in conflict
            )
            if violation > 1e-7:
                cut_set.add(conflict)
                new_cuts.append(conflict)
                violated_generated += 1
        if not new_cuts:
            break
        cuts.extend(new_cuts)
        bound, x = compact_lp_support(instance, direction, cuts)
    return {
        "initial_bound": initial_bound,
        "final_bound": bound,
        "rounds": rounds,
        "cuts": len(cuts),
        "conflicts_generated": generated,
        "violated_conflicts_generated": violated_generated,
        "runtime_seconds": time.perf_counter() - start,
    }


def build_small_instance(seed: int, n: int, m: int, gamma: int) -> PricingInstance:
    rng = np.random.default_rng(seed)
    items: list[list[Option]] = []
    for i in range(n):
        base_margin = float(2.7 + rng.uniform(-0.25, 0.25))
        base_value = float(4.0 + rng.uniform(0.0, 1.0))
        group = [Option(value=base_value, margin=base_margin, uncertainty=0.0)]
        for j in range(1, m):
            resource_loss = float(1.15 * j + rng.uniform(0.1, 0.9))
            deviation = float(0.35 + 0.55 * j + rng.uniform(0.0, 1.1) + 0.013 * i)
            value = float(base_value + 4.2 * j + rng.uniform(0.0, 3.5))
            group.append(
                Option(
                    value=value,
                    margin=base_margin - resource_loss,
                    uncertainty=deviation,
                )
            )
        items.append(group)
    return PricingInstance(items=items, gamma=gamma, name=f"small_s{seed}_n{n}_m{m}_g{gamma}")


def run_polyhedral_study(output_dir: Path, instances: int, directions: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    instance_rows: list[dict] = []
    accepted = 0
    attempted = 0
    while accepted < instances:
        attempted += 1
        n = int(rng.integers(4, 7))
        m = 3
        gamma = int(rng.integers(1, min(3, n) + 1))
        inst_seed = int(rng.integers(1, 2**31 - 1))
        instance = build_small_instance(inst_seed, n, m, gamma)
        feasible = feasible_assignments(instance)
        if len(feasible) in {0, m**n}:
            continue
        conflicts = minimal_group_conflicts(instance, feasible)
        sizes, offsets, _, _ = _flatten(instance)
        n_x = offsets[-1]
        accepted += 1
        instance_rows.append(
            {
                "instance": instance.name,
                "n": n,
                "m": m,
                "gamma": gamma,
                "assignments": m**n,
                "feasible_assignments": len(feasible),
                "minimal_conflicts": len(conflicts),
                "theta_count": len(build_full_theta_candidates(instance)),
            }
        )
        original = np.array([option.value for group in instance.items for option in group], dtype=float)
        for direction_id in range(directions):
            if direction_id == 0:
                direction = original
                direction_kind = "model_objective"
            elif direction_id % 2 == 1:
                # Positive support directions resemble alternative menu values
                # while still probing many faces of the feasible hull.
                direction = rng.lognormal(mean=2.0, sigma=0.8, size=n_x)
                direction_kind = "random_positive"
            else:
                # Signed directions are necessary to probe the complete hull,
                # rather than only its economically monotone faces.
                direction = rng.normal(loc=0.0, scale=10.0, size=n_x)
                direction_kind = "random_signed"
            integer = integer_support(instance, direction, feasible)
            compact, compact_x = compact_lp_support(instance, direction)
            disjunctive = disjunctive_lp_support(instance, direction)
            conflict, conflict_x = compact_lp_support(instance, direction, conflicts)
            scale = max(1.0, abs(integer))
            rows.append(
                {
                    "instance": instance.name,
                    "direction_id": direction_id,
                    "direction_kind": direction_kind,
                    "integer_hull": integer,
                    "compact_lp": compact,
                    "disjunctive_lp": disjunctive,
                    "conflict_lp": conflict,
                    "compact_gap_pct": 100.0 * (compact - integer) / scale,
                    "disjunctive_gap_pct": 100.0 * (disjunctive - integer) / scale,
                    "conflict_gap_pct": 100.0 * (conflict - integer) / scale,
                    "disjunctive_improvement_pct": 100.0 * (compact - disjunctive) / scale,
                    "conflict_improvement_pct": 100.0 * (compact - conflict) / scale,
                    "disjunctive_strict": disjunctive < compact - 1e-7,
                    "conflict_strict": conflict < compact - 1e-7,
                    "conflict_beats_disjunctive": conflict < disjunctive - 1e-7,
                    "disjunctive_beats_conflict": disjunctive < conflict - 1e-7,
                    "compact_fractional_variables": int(np.sum((compact_x > TOL) & (compact_x < 1.0 - TOL))) if compact_x is not None else -1,
                    "conflict_fractional_variables": int(np.sum((conflict_x > TOL) & (conflict_x < 1.0 - TOL))) if conflict_x is not None else -1,
                    "minimal_conflicts": len(conflicts),
                }
            )

    _write_csv(output_dir / "polyhedral_support.csv", rows)
    _write_csv(output_dir / "polyhedral_instances.csv", instance_rows)
    compact_gaps = np.array([row["compact_gap_pct"] for row in rows], dtype=float)
    disj_gaps = np.array([row["disjunctive_gap_pct"] for row in rows], dtype=float)
    conflict_gaps = np.array([row["conflict_gap_pct"] for row in rows], dtype=float)
    nontrivial = compact_gaps > 1e-7
    disj_closure = np.divide(
        compact_gaps - disj_gaps,
        compact_gaps,
        out=np.zeros_like(compact_gaps),
        where=nontrivial,
    )
    conflict_closure = np.divide(
        compact_gaps - conflict_gaps,
        compact_gaps,
        out=np.zeros_like(compact_gaps),
        where=nontrivial,
    )
    summary = {
        "instances": instances,
        "attempted_instances": attempted,
        "directions_per_instance": directions,
        "support_problems": len(rows),
        "compact_gap_median_pct": float(np.median(compact_gaps)),
        "disjunctive_gap_median_pct": float(np.median(disj_gaps)),
        "conflict_gap_median_pct": float(np.median(conflict_gaps)),
        "compact_gap_p90_pct": float(np.quantile(compact_gaps, 0.9)),
        "disjunctive_gap_p90_pct": float(np.quantile(disj_gaps, 0.9)),
        "conflict_gap_p90_pct": float(np.quantile(conflict_gaps, 0.9)),
        "nontrivial_compact_gap_cases": int(np.sum(nontrivial)),
        "nontrivial_compact_gap_rate": float(np.mean(nontrivial)),
        "disjunctive_median_gap_closure_on_nontrivial": float(np.median(disj_closure[nontrivial])) if np.any(nontrivial) else 0.0,
        "conflict_median_gap_closure_on_nontrivial": float(np.median(conflict_closure[nontrivial])) if np.any(nontrivial) else 0.0,
        "disjunctive_exact_hull_rate_on_nontrivial": float(np.mean(disj_gaps[nontrivial] <= 1e-7)) if np.any(nontrivial) else 0.0,
        "conflict_exact_hull_rate_on_nontrivial": float(np.mean(conflict_gaps[nontrivial] <= 1e-7)) if np.any(nontrivial) else 0.0,
        "disjunctive_strict_rate": float(np.mean([row["disjunctive_strict"] for row in rows])),
        "conflict_strict_rate": float(np.mean([row["conflict_strict"] for row in rows])),
        "conflict_beats_disjunctive_rate": float(np.mean([row["conflict_beats_disjunctive"] for row in rows])),
        "disjunctive_beats_conflict_rate": float(np.mean([row["disjunctive_beats_conflict"] for row in rows])),
        "mean_minimal_conflicts": float(np.mean([row["minimal_conflicts"] for row in instance_rows])),
    }
    (output_dir / "polyhedral_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class ThetaIntervalOracle:
    """Vectorized group-separable Lagrangian interval upper bound."""

    def __init__(self, instance: PricingInstance):
        start = time.perf_counter()
        self.instance = instance
        self.thetas = np.array(build_full_theta_candidates(instance), dtype=float)
        capacities: list[float] = []
        group_costs: list[list[np.ndarray]] = [[] for _ in instance.items]
        for theta in self.thetas:
            data = build_fixed_theta_data(instance, float(theta))
            capacities.append(float(data.capacity))
            for i, costs in enumerate(data.costs):
                group_costs[i].append(np.asarray(costs, dtype=float))
        self.capacities = np.array(capacities, dtype=float)
        self.group_costs = [np.vstack(rows) for rows in group_costs]
        self.group_values = [np.array([option.value for option in group], dtype=float) for group in instance.items]
        self.preprocessing_seconds = time.perf_counter() - start
        positive_costs = np.concatenate([cost[cost > TOL] for cost in self.group_costs])
        all_values = np.concatenate(self.group_values)
        value_scale = max(1.0, float(np.ptp(all_values)), float(np.max(np.abs(all_values))))
        cost_scale = float(np.median(positive_costs)) if positive_costs.size else 1.0
        self.lambda_scale = max(1e-6, value_scale / max(cost_scale, 1e-6))

    def values_at_lambda(self, lambda_value: float, lo: int, hi: int) -> np.ndarray:
        sl = slice(lo, hi + 1)
        result = float(lambda_value) * self.capacities[sl].copy()
        for values, costs in zip(self.group_values, self.group_costs):
            result += np.max(values[None, :] - float(lambda_value) * costs[sl, :], axis=1)
        result[self.capacities[sl] < -TOL] = -np.inf
        return result

    def bound(self, lo: int, hi: int, local_search: bool = True) -> IntervalBound:
        start = time.perf_counter()
        evaluations: list[tuple[float, float]] = []

        def evaluate(lambda_value: float) -> float:
            lam = max(0.0, float(lambda_value))
            value = float(np.max(self.values_at_lambda(lam, lo, hi)))
            evaluations.append((lam, value))
            return value

        grid = np.concatenate(
            [
                np.array([0.0]),
                self.lambda_scale * np.geomspace(1e-4, 1e4, 25),
            ]
        )
        grid_values = np.array([evaluate(lam) for lam in grid], dtype=float)
        best_index = int(np.argmin(grid_values))
        if local_search and 0 < best_index < len(grid) - 1:
            lower = float(grid[best_index - 1])
            upper = float(grid[best_index + 1])
            result = opt.minimize_scalar(
                evaluate,
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": 1e-9, "maxiter": 80},
            )
            # evaluate() records every point used by the optimizer.  We never
            # trust result.fun without an explicit evaluation.
            if math.isfinite(float(result.x)):
                evaluate(float(result.x))
        finite = [(lam, value) for lam, value in evaluations if math.isfinite(value)]
        if not finite:
            return IntervalBound(float("-inf"), 0.0, len(evaluations), time.perf_counter() - start)
        best_lambda, best_value = min(finite, key=lambda pair: pair[1])
        # Tiny outward padding protects the upper-bound direction against
        # accumulated floating-point roundoff.
        padded = best_value + 1e-10 * max(1.0, abs(best_value))
        return IntervalBound(float(padded), float(best_lambda), len(evaluations), time.perf_counter() - start)


def full_theta_lp_scan(instance: PricingInstance) -> tuple[float, float, int]:
    start = time.perf_counter()
    values: list[float] = []
    feasible = 0
    for theta in build_full_theta_candidates(instance):
        result = compute_fixed_theta_lp_upper_bound(instance, theta)
        if result.lp_feasible:
            values.append(float(result.lp_upper_bound))
            feasible += 1
    return (max(values) if values else float("-inf"), time.perf_counter() - start, feasible)


def run_interval_random_validation(
    output_dir: Path,
    cases: int = 100,
    intervals_per_case: int = 5,
    seed: int = 20260720,
) -> dict:
    """Cross-check interval bounds against every fixed-theta LP on random cases."""

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for case in range(cases):
        n = int(rng.integers(4, 9))
        gamma = int(rng.integers(1, min(3, n) + 1))
        instance = build_small_instance(int(rng.integers(1, 2**31 - 1)), n, 3, gamma)
        if case % 2:
            random_values = rng.normal(0.0, 10.0, size=instance.n_options)
            cursor = 0
            replaced: list[list[Option]] = []
            for group in instance.items:
                new_group: list[Option] = []
                for option in group:
                    new_group.append(
                        Option(
                            value=float(random_values[cursor]),
                            margin=option.margin,
                            uncertainty=option.uncertainty,
                        )
                    )
                    cursor += 1
                replaced.append(new_group)
            instance = PricingInstance(items=replaced, gamma=gamma, name=f"interval_random_{case}")
        oracle = ThetaIntervalOracle(instance)
        direction = np.array([option.value for group in instance.items for option in group], dtype=float)
        exact_values = np.array(
            [fixed_theta_lp_support(instance, theta, direction) for theta in oracle.thetas],
            dtype=float,
        )
        intervals = [(0, len(oracle.thetas) - 1)]
        for _ in range(intervals_per_case):
            lo = int(rng.integers(0, len(oracle.thetas)))
            hi = int(rng.integers(lo, len(oracle.thetas)))
            intervals.append((lo, hi))
        for interval_id, (lo, hi) in enumerate(intervals):
            bound = oracle.bound(lo, hi)
            exact = float(np.max(exact_values[lo : hi + 1]))
            excess = bound.upper_bound - exact
            if excess < -1e-6:
                raise AssertionError(
                    f"interval bound underestimated fixed-theta LP maximum by {excess}"
                )
            rows.append(
                {
                    "case": case,
                    "interval_id": interval_id,
                    "n": n,
                    "gamma": gamma,
                    "signed_objective": bool(case % 2),
                    "lo": lo,
                    "hi": hi,
                    "interval_size": hi - lo + 1,
                    "upper_bound": bound.upper_bound,
                    "exact_fixed_theta_lp_max": exact,
                    "excess": excess,
                    "excess_pct": 100.0 * excess / max(1.0, abs(exact)),
                }
            )
    _write_csv(output_dir / "interval_random_validation.csv", rows)
    excess_pct = np.array([row["excess_pct"] for row in rows], dtype=float)
    summary = {
        "cases": cases,
        "intervals": len(rows),
        "minimum_excess_pct": float(np.min(excess_pct)),
        "median_excess_pct": float(np.median(excess_pct)),
        "p90_excess_pct": float(np.quantile(excess_pct, 0.9)),
        "maximum_excess_pct": float(np.max(excess_pct)),
        "tight_within_1e_6_rate": float(np.mean(np.abs(excess_pct) <= 1e-6)),
    }
    (output_dir / "interval_random_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def interval_exact_solve(
    instance: PricingInstance,
    oracle: ThetaIntervalOracle,
    time_limit_seconds: float = 10.0,
) -> dict:
    """Anytime interval B&B; exact if the queue empties before the limit."""

    start = time.perf_counter()
    heuristic = solve(instance, upgrade_completion=True)
    incumbent = float(heuristic.objective) if heuristic.is_feasible else float("-inf")
    incumbent_selection = list(heuristic.selections) if heuristic.is_feasible else None
    incumbent_theta = float(heuristic.theta) if heuristic.is_feasible else None
    root = oracle.bound(0, len(oracle.thetas) - 1)
    queue: list[tuple[float, int, int, int]] = [(-root.upper_bound, 0, len(oracle.thetas) - 1, 0)]
    interval_bounds = 1
    interval_pruned = 0
    singleton_pruned = 0
    singleton_solved = 0
    nodes = 0
    interrupted = False
    while queue:
        elapsed = time.perf_counter() - start
        if elapsed >= time_limit_seconds:
            interrupted = True
            break
        neg_bound, lo, hi, depth = heapq.heappop(queue)
        upper = -float(neg_bound)
        nodes += 1
        if upper <= incumbent + 1e-7:
            interval_pruned += 1
            if lo == hi:
                singleton_pruned += 1
            continue
        if lo == hi:
            theta = float(oracle.thetas[lo])
            remaining = max(1e-3, time_limit_seconds - (time.perf_counter() - start))
            result = solve_fixed_theta_bnb(
                instance,
                theta,
                FixedThetaBNBConfig(
                    use_greedy_incumbent=True,
                    use_cutoff_pruning=True,
                    objective_cutoff=incumbent if math.isfinite(incumbent) else None,
                    collect_diagnostics=False,
                    time_limit_seconds=remaining,
                ),
            )
            singleton_solved += 1
            if result.selected_options is not None and result.objective_value > incumbent + TOL:
                certificate = compute_certificate(instance, result.selected_options)
                if certificate < -1e-7:
                    raise AssertionError("fixed-theta leaf returned a non-robust incumbent")
                incumbent = float(result.objective_value)
                incumbent_selection = list(result.selected_options)
                incumbent_theta = theta
            if result.status not in {"optimal", "infeasible", "cutoff_pruned"}:
                unresolved_bound = min(
                    upper,
                    float(result.upper_bound) if math.isfinite(result.upper_bound) else upper,
                )
                heapq.heappush(queue, (-unresolved_bound, lo, hi, depth))
                interrupted = True
                break
            continue
        mid = (lo + hi) // 2
        for child_lo, child_hi in ((lo, mid), (mid + 1, hi)):
            bound = oracle.bound(child_lo, child_hi)
            interval_bounds += 1
            if bound.upper_bound <= incumbent + 1e-7:
                interval_pruned += 1
                if child_lo == child_hi:
                    singleton_pruned += 1
            else:
                heapq.heappush(queue, (-bound.upper_bound, child_lo, child_hi, depth + 1))
    certificate = compute_certificate(instance, incumbent_selection) if incumbent_selection is not None else float("-inf")
    unresolved_upper = max(
        incumbent,
        -float(queue[0][0]) if queue else float("-inf"),
    )
    absolute_gap = max(0.0, unresolved_upper - incumbent) if math.isfinite(incumbent) else float("inf")
    relative_gap = absolute_gap / max(1.0, abs(incumbent)) if math.isfinite(absolute_gap) else float("inf")
    certified = not queue and not interrupted and incumbent_selection is not None and certificate >= -1e-7
    return {
        "status": "optimal" if certified else "time_limit",
        "objective": incumbent,
        "theta": incumbent_theta,
        "certificate": certificate,
        "runtime_seconds": time.perf_counter() - start,
        "root_upper_bound": root.upper_bound,
        "root_lambda": root.lambda_value,
        "interval_nodes_processed": nodes,
        "interval_bounds_computed": interval_bounds,
        "intervals_pruned": interval_pruned,
        "singletons_pruned": singleton_pruned,
        "singletons_solved": singleton_solved,
        "theta_count": len(oracle.thetas),
        "upper_bound": unresolved_upper,
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
        "certified_optimal": certified,
    }


def _gamma_for(n: int, mode: str) -> int:
    if mode == "sqrt":
        return max(1, int(round(math.sqrt(n))))
    if mode == "quarter":
        return max(1, int(round(0.25 * n)))
    raise ValueError(mode)


def run_interval_study(
    output_dir: Path,
    sizes: Sequence[int],
    seeds: Sequence[int],
    families: Sequence[str],
    exact_max_n: int,
    exact_time_limit: float,
) -> dict:
    random_validation = run_interval_random_validation(output_dir)
    rows: list[dict] = []
    singleton_validation_errors: list[float] = []
    for family, n, seed in itertools.product(families, sizes, seeds):
        gamma = _gamma_for(n, "sqrt")
        instance = build_benchmark_instance(family, n, 6, gamma, seed)
        oracle_start = time.perf_counter()
        oracle = ThetaIntervalOracle(instance)
        root = oracle.bound(0, len(oracle.thetas) - 1)
        root_total = time.perf_counter() - oracle_start
        lp_max, lp_scan_seconds, feasible_theta = full_theta_lp_scan(instance)
        hr_start = time.perf_counter()
        hr = solve(instance, upgrade_completion=True)
        hr_seconds = time.perf_counter() - hr_start

        # Validate singleton upper bounds at representative positions.
        representatives = sorted(set([0, len(oracle.thetas) // 4, len(oracle.thetas) // 2, 3 * len(oracle.thetas) // 4, len(oracle.thetas) - 1]))
        for idx in representatives:
            singleton = oracle.bound(idx, idx)
            fixed = compute_fixed_theta_lp_upper_bound(instance, float(oracle.thetas[idx]))
            if fixed.lp_feasible:
                error = singleton.upper_bound - float(fixed.lp_upper_bound)
                singleton_validation_errors.append(error)
                if error < -1e-6:
                    raise AssertionError(f"invalid singleton interval bound: {error}")

        row = {
            "instance": instance.name,
            "family": family,
            "n": n,
            "m": 6,
            "gamma": gamma,
            "theta_count": len(oracle.thetas),
            "feasible_theta_count": feasible_theta,
            "oracle_preprocessing_seconds": oracle.preprocessing_seconds,
            "root_bound_seconds_excluding_preprocessing": root.runtime_seconds,
            "root_bound_total_seconds": root_total,
            "full_theta_lp_scan_seconds": lp_scan_seconds,
            "initialization_speedup": lp_scan_seconds / max(root_total, 1e-12),
            "interval_root_upper_bound": root.upper_bound,
            "full_theta_lp_upper_bound": lp_max,
            "interval_excess_over_full_lp_pct": 100.0 * (root.upper_bound - lp_max) / max(1.0, abs(lp_max)),
            "hullround_objective": hr.objective,
            "hullround_feasible": hr.is_feasible,
            "hullround_seconds": hr_seconds,
            "interval_root_gap_to_hullround_pct": 100.0 * (root.upper_bound - hr.objective) / max(1.0, abs(hr.objective)) if hr.is_feasible else float("nan"),
            "full_lp_gap_to_hullround_pct": 100.0 * (lp_max - hr.objective) / max(1.0, abs(hr.objective)) if hr.is_feasible else float("nan"),
            "root_lambda": root.lambda_value,
            "root_lagrangian_evaluations": root.evaluations,
        }
        if n <= exact_max_n:
            exact = interval_exact_solve(instance, oracle, time_limit_seconds=exact_time_limit)
            row.update({f"interval_exact_{key}": value for key, value in exact.items()})
            row["interval_exact_end_to_end_seconds"] = (
                oracle.preprocessing_seconds + float(exact["runtime_seconds"])
            )

            standard = solve_global_theta_bnb(
                instance,
                GlobalThetaBNBConfig(
                    time_limit_seconds=exact_time_limit,
                    theta_order="lp_bound_desc",
                    use_caches=True,
                    use_objective_cutoff=True,
                    use_fast_residual_lp_bound=True,
                ),
            )
            row.update(
                standard_theta_status=standard.status,
                standard_theta_objective=standard.objective_value,
                standard_theta_runtime_seconds=standard.total_runtime_seconds,
                standard_theta_nodes=standard.total_nodes_explored,
                standard_theta_pruned=standard.theta_count_pruned_by_bound,
            )
            highs = solve_full_robust_highs(instance, time_limit=60.0, threads=1)
            row.update(
                highs_status=highs.get("status"),
                highs_certified=highs.get("certified"),
                highs_objective=highs.get("objective"),
                highs_runtime_seconds=highs.get("runtime_s"),
            )
            scip_result = solve_full_robust_scip(instance, time_limit=60.0, threads=1)
            row.update(
                scip_status=scip_result.get("status"),
                scip_certified=scip_result.get("certified"),
                scip_objective=scip_result.get("objective"),
                scip_runtime_seconds=scip_result.get("runtime_s"),
            )
            if bool(exact["certified_optimal"]) and standard.status == "optimal" and abs(float(exact["objective"]) - standard.objective_value) > 1e-6:
                raise AssertionError(f"interval/standard objective mismatch on {instance.name}")
            reference_objective = (
                standard.objective_value
                if standard.status == "optimal"
                else float(exact["objective"]) if bool(exact["certified_optimal"]) else None
            )
            if bool(highs.get("certified")) and reference_objective is not None and abs(float(highs["objective"]) - reference_objective) > 1e-5:
                raise AssertionError(f"HiGHS/standard objective mismatch on {instance.name}")
            if bool(scip_result.get("certified")) and reference_objective is not None and abs(float(scip_result["objective"]) - reference_objective) > 1e-5:
                raise AssertionError(f"SCIP/standard objective mismatch on {instance.name}")
        rows.append(row)
        _write_csv(output_dir / "interval_benchmark.csv", rows)

    excess = np.array([row["interval_excess_over_full_lp_pct"] for row in rows], dtype=float)
    speedups = np.array([row["initialization_speedup"] for row in rows], dtype=float)
    root_gaps = np.array([row["interval_root_gap_to_hullround_pct"] for row in rows], dtype=float)
    exact_rows = [row for row in rows if "interval_exact_singletons_solved" in row]
    summary = {
        "instances": len(rows),
        "families": list(families),
        "sizes": list(sizes),
        "seeds": list(seeds),
        "median_initialization_speedup": float(np.median(speedups)),
        "p10_initialization_speedup": float(np.quantile(speedups, 0.1)),
        "median_interval_excess_over_full_lp_pct": float(np.median(excess)),
        "p90_interval_excess_over_full_lp_pct": float(np.quantile(excess, 0.9)),
        "median_interval_root_gap_to_hullround_pct": float(np.nanmedian(root_gaps)),
        "max_singleton_underestimate": float(min(singleton_validation_errors, default=0.0)),
        "max_singleton_overestimate": float(max(singleton_validation_errors, default=0.0)),
        "exact_instances": len(exact_rows),
        "exact_time_limit_seconds": exact_time_limit,
        "interval_exact_certification_rate": float(np.mean([bool(row["interval_exact_certified_optimal"]) for row in exact_rows])) if exact_rows else float("nan"),
        "standard_theta_certification_rate": float(np.mean([row["standard_theta_status"] == "optimal" for row in exact_rows])) if exact_rows else float("nan"),
        "median_theta_fraction_solved_exact": float(
            np.median(
                [row["interval_exact_singletons_solved"] / row["theta_count"] for row in exact_rows]
            )
        ) if exact_rows else float("nan"),
        "median_interval_exact_runtime_seconds": float(
            np.median([row["interval_exact_runtime_seconds"] for row in exact_rows])
        ) if exact_rows else float("nan"),
        "median_interval_exact_end_to_end_seconds": float(
            np.median([row["interval_exact_end_to_end_seconds"] for row in exact_rows])
        ) if exact_rows else float("nan"),
        "median_standard_theta_runtime_seconds": float(
            np.median([row["standard_theta_runtime_seconds"] for row in exact_rows])
        ) if exact_rows else float("nan"),
        "median_interval_vs_standard_speedup": float(
            np.median(
                [
                    row["standard_theta_runtime_seconds"]
                    / max(row["interval_exact_end_to_end_seconds"], 1e-12)
                    for row in exact_rows
                ]
            )
        ) if exact_rows else float("nan"),
        "highs_certification_rate": float(np.mean([bool(row["highs_certified"]) for row in exact_rows])) if exact_rows else float("nan"),
        "scip_certification_rate": float(np.mean([bool(row["scip_certified"]) for row in exact_rows])) if exact_rows else float("nan"),
        "random_validation": random_validation,
    }
    (output_dir / "interval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_cut_study(
    output_dir: Path,
    sizes: Sequence[int],
    seeds: Sequence[int],
    families: Sequence[str],
    samples_per_round: int,
) -> dict:
    """Benchmark heuristic group-conflict separation against SCIP optima."""

    rows: list[dict] = []
    for family, n, seed in itertools.product(families, sizes, seeds):
        gamma = _gamma_for(n, "sqrt")
        instance = build_benchmark_instance(family, n, 6, gamma, seed)
        direction = np.array(
            [option.value for group in instance.items for option in group], dtype=float
        )
        compact_start = time.perf_counter()
        compact, _ = compact_lp_support(instance, direction)
        compact_seconds = time.perf_counter() - compact_start
        disj_start = time.perf_counter()
        disjunctive = disjunctive_lp_support(instance, direction)
        disj_seconds = time.perf_counter() - disj_start
        separated = cutting_plane_support(
            instance,
            direction,
            seed=20260719 + 1009 * seed + 17 * n,
            samples_per_round=samples_per_round,
        )
        scip_result = solve_full_robust_scip(instance, time_limit=60.0, threads=1)
        certified = bool(scip_result.get("certified"))
        integer = float(scip_result.get("objective", float("nan")))
        scale = max(1.0, abs(integer)) if certified else 1.0
        compact_gap = (compact - integer) / scale if certified else float("nan")
        disj_gap = (disjunctive - integer) / scale if certified else float("nan")
        cut_gap = (float(separated["final_bound"]) - integer) / scale if certified else float("nan")
        if certified:
            if min(compact, disjunctive, float(separated["final_bound"])) < integer - 1e-5:
                raise AssertionError(f"invalid relaxation bound on {instance.name}")
        row = {
            "instance": instance.name,
            "family": family,
            "n": n,
            "m": 6,
            "gamma": gamma,
            "theta_count": len(build_full_theta_candidates(instance)),
            "integer_objective": integer,
            "scip_certified": certified,
            "scip_status": scip_result.get("status"),
            "scip_runtime_seconds": scip_result.get("runtime_s"),
            "compact_bound": compact,
            "compact_seconds": compact_seconds,
            "disjunctive_bound": disjunctive,
            "disjunctive_seconds": disj_seconds,
            "cut_bound": separated["final_bound"],
            "cut_seconds": separated["runtime_seconds"],
            "cut_rounds": separated["rounds"],
            "cuts_added": separated["cuts"],
            "conflicts_generated": separated["conflicts_generated"],
            "compact_gap_pct": 100.0 * compact_gap,
            "disjunctive_gap_pct": 100.0 * disj_gap,
            "cut_gap_pct": 100.0 * cut_gap,
            "disjunctive_gap_closure": (compact_gap - disj_gap) / compact_gap if certified and compact_gap > 1e-10 else 0.0,
            "cut_gap_closure": (compact_gap - cut_gap) / compact_gap if certified and compact_gap > 1e-10 else 0.0,
        }
        rows.append(row)
        _write_csv(output_dir / "cut_benchmark.csv", rows)

    certified_rows = [row for row in rows if row["scip_certified"]]
    nontrivial = [row for row in certified_rows if row["compact_gap_pct"] > 1e-7]
    summary = {
        "instances": len(rows),
        "scip_certification_rate": len(certified_rows) / len(rows) if rows else 0.0,
        "nontrivial_compact_gap_instances": len(nontrivial),
        "median_compact_gap_pct": float(np.median([row["compact_gap_pct"] for row in certified_rows])) if certified_rows else float("nan"),
        "median_disjunctive_gap_pct": float(np.median([row["disjunctive_gap_pct"] for row in certified_rows])) if certified_rows else float("nan"),
        "median_cut_gap_pct": float(np.median([row["cut_gap_pct"] for row in certified_rows])) if certified_rows else float("nan"),
        "median_disjunctive_gap_closure_nontrivial": float(np.median([row["disjunctive_gap_closure"] for row in nontrivial])) if nontrivial else 0.0,
        "median_cut_gap_closure_nontrivial": float(np.median([row["cut_gap_closure"] for row in nontrivial])) if nontrivial else 0.0,
        "cut_exact_bound_rate_nontrivial": float(np.mean([row["cut_gap_pct"] <= 1e-6 for row in nontrivial])) if nontrivial else 0.0,
        "disjunctive_exact_bound_rate_nontrivial": float(np.mean([row["disjunctive_gap_pct"] <= 1e-6 for row in nontrivial])) if nontrivial else 0.0,
        "median_cuts_added": float(np.median([row["cuts_added"] for row in rows])) if rows else 0.0,
        "median_cut_runtime_seconds": float(np.median([row["cut_seconds"] for row in rows])) if rows else 0.0,
        "median_scip_runtime_seconds": float(np.median([row["scip_runtime_seconds"] for row in certified_rows])) if certified_rows else float("nan"),
        "cut_improves_compact_rate": float(np.mean([row["cut_bound"] < row["compact_bound"] - 1e-7 for row in rows])) if rows else 0.0,
        "cut_beats_disjunctive_rate": float(np.mean([row["cut_bound"] < row["disjunctive_bound"] - 1e-7 for row in rows])) if rows else 0.0,
    }
    (output_dir / "cut_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_environment(output_dir: Path, args: argparse.Namespace) -> None:
    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "command": sys.argv,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["polyhedral", "interval", "cuts", "all"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "novelty_go_no_go_20260719")
    parser.add_argument("--poly-instances", type=int, default=40)
    parser.add_argument("--directions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--sizes", default="30,60,90")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--families", default="dense_frontier,correlated_risk,near_tie,many_breakpoints")
    parser.add_argument("--exact-max-n", type=int, default=30)
    parser.add_argument("--exact-time-limit", type=float, default=10.0)
    parser.add_argument("--cut-sizes", default="30,60")
    parser.add_argument("--cut-samples", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    write_environment(output_dir, args)
    result: dict[str, object] = {}
    if args.command in {"polyhedral", "all"}:
        result["polyhedral"] = run_polyhedral_study(
            output_dir,
            instances=args.poly_instances,
            directions=args.directions,
            seed=args.seed,
        )
    if args.command in {"interval", "all"}:
        sizes = [int(value) for value in args.sizes.split(",") if value]
        seeds = [int(value) for value in args.seeds.split(",") if value]
        families = [value for value in args.families.split(",") if value]
        result["interval"] = run_interval_study(
            output_dir,
            sizes=sizes,
            seeds=seeds,
            families=families,
            exact_max_n=args.exact_max_n,
            exact_time_limit=args.exact_time_limit,
        )
    if args.command in {"cuts", "all"}:
        cut_sizes = [int(value) for value in args.cut_sizes.split(",") if value]
        seeds = [int(value) for value in args.seeds.split(",") if value]
        families = [value for value in args.families.split(",") if value]
        result["cuts"] = run_cut_study(
            output_dir,
            sizes=cut_sizes,
            seeds=seeds,
            families=families,
            samples_per_round=args.cut_samples,
        )
    (output_dir / "study_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
