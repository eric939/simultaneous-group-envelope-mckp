#!/usr/bin/env python3
"""Second-stage structural and computational feasibility study.

This campaign tests whether the unchanged Gamma-robust MCKP still contains a
defensible new polyhedral or algorithmic contribution after the first go/no-go
study.  It provides:

* exact rational facet enumeration through cddlib;
* classification of trivial, minimal-conflict, and non-conflict facets;
* tests of which integer-hull facets cut the full theta-disjunctive LP;
* exact MILP separation of group-conflict inequalities;
* fair plain-SCIP versus statically cut-enhanced SCIP comparisons; and
* adaptive interval compression tests against a complete theta-LP scan.
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
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from math import gcd
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipy.optimize as opt
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robust_mckp import Option, PricingInstance  # noqa: E402
from robust_mckp.certificate import certificate_is_feasible, compute_certificate  # noqa: E402
from robust_mckp.hull import build_upper_hull  # noqa: E402
from robust_mckp.exact_bnb import (  # noqa: E402
    _require_representable_objective_range,
    build_fixed_theta_data,
    build_full_theta_candidates,
    compute_fixed_theta_lp_upper_bound,
)
from research.novelty_go_no_go import (  # noqa: E402
    ThetaIntervalOracle,
    _flatten,
    _gamma_for,
    _write_csv,
    compact_lp_support,
    disjunctive_lp_support,
    feasible_assignments,
    full_theta_lp_scan,
    maximum_completion_certificate,
    minimal_group_conflicts,
    shrink_group_conflict,
)
from research.benchmark_instances import build_benchmark_instance  # noqa: E402


TOL = 1e-8
CDDEXEC = Path(
    os.environ.get("CDDEXEC")
    or shutil.which("cddexec_gmp")
    or "/opt/homebrew/bin/cddexec_gmp"
)


class FixedThetaLPOracle:
    """Reusable linear-time solver for the fixed-threshold MCKP LP.

    It builds each group's upper value--cost hull directly and greedily merges
    their slopes.  Unlike the branch-and-bound cache builder, this LP-only
    path does not perform a redundant pairwise integer-dominance pass.
    """

    def __init__(
        self,
        instance: PricingInstance,
        tol: float = 1e-9,
        *,
        objective_range_validated: bool = False,
    ):
        if not objective_range_validated:
            _require_representable_objective_range(instance)
        self.instance = instance
        self.tol = float(tol)
        self.values = [
            np.asarray([option.value for option in group], dtype=float)
            for group in instance.items
        ]
        self.margins = [
            np.asarray([option.margin for option in group], dtype=float)
            for group in instance.items
        ]
        self.deviations = [
            np.abs(np.asarray([option.uncertainty for option in group], dtype=float))
            for group in instance.items
        ]
        self.option_indices = [np.arange(len(group), dtype=int) for group in instance.items]
        self._value_exponent = max(
            float(value).as_integer_ratio()[1].bit_length() - 1
            for values in self.values
            for value in values
        )
        self._coefficient_exponent = max(
            [
                *(
                    float(value).as_integer_ratio()[1].bit_length() - 1
                    for values in self.margins
                    for value in values
                ),
                *(
                    float(value).as_integer_ratio()[1].bit_length() - 1
                    for values in self.deviations
                    for value in values
                ),
            ]
        )

        def scaled(value: float, exponent: int) -> int:
            numerator, denominator = float(value).as_integer_ratio()
            local_exponent = denominator.bit_length() - 1
            return numerator << (exponent - local_exponent)

        self._exact_value_integers = [
            [scaled(float(value), self._value_exponent) for value in values]
            for values in self.values
        ]
        self._exact_margin_integers = [
            [scaled(float(value), self._coefficient_exponent) for value in values]
            for values in self.margins
        ]
        self._exact_deviation_integers = [
            [scaled(float(value), self._coefficient_exponent) for value in values]
            for values in self.deviations
        ]

    @staticmethod
    def _exact_upper_hull(
        costs: Sequence[int], values: Sequence[int]
    ) -> list[int]:
        """Return exact upper-hull option indices in increasing-cost order."""

        best_at_cost: dict[int, int] = {}
        for index, cost in enumerate(costs):
            incumbent = best_at_cost.get(int(cost))
            if incumbent is None or values[index] > values[incumbent]:
                best_at_cost[int(cost)] = index
        ordered = [best_at_cost[cost] for cost in sorted(best_at_cost)]
        nondominated: list[int] = []
        best_value: int | None = None
        for index in ordered:
            if best_value is None or values[index] > best_value:
                nondominated.append(index)
                best_value = values[index]
        hull: list[int] = []
        for index in nondominated:
            while len(hull) >= 2:
                left, middle = hull[-2], hull[-1]
                left_value = values[middle] - values[left]
                left_cost = int(costs[middle]) - int(costs[left])
                right_value = values[index] - values[middle]
                right_cost = int(costs[index]) - int(costs[middle])
                if left_value * right_cost > right_value * left_cost:
                    break
                hull.pop()
            hull.append(index)
        return hull

    def value(self, theta: float) -> float:
        theta = float(theta)
        theta_numerator, theta_denominator = theta.as_integer_ratio()
        theta_exponent = theta_denominator.bit_length() - 1
        coefficient_exponent = max(
            self._coefficient_exponent, theta_exponent
        )
        coefficient_shift = coefficient_exponent - self._coefficient_exponent
        exact_theta = theta_numerator << (coefficient_exponent - theta_exponent)

        exact_costs: list[list[int]] = []
        exact_stars: list[int] = []
        for margins, deviations in zip(
            self._exact_margin_integers,
            self._exact_deviation_integers,
        ):
            slacks = [
                (margin << coefficient_shift)
                - max((deviation << coefficient_shift) - exact_theta, 0)
                for margin, deviation in zip(margins, deviations)
            ]
            star = max(slacks)
            exact_stars.append(star)
            exact_costs.append([star - slack for slack in slacks])
        exact_capacity = sum(exact_stars) - int(self.instance.gamma) * exact_theta
        if exact_capacity < 0:
            return float("-inf")

        base_cost = 0
        base_value = 0
        segments: list[tuple[int, int]] = []
        for group_costs, group_values in zip(
            exact_costs, self._exact_value_integers
        ):
            hull = self._exact_upper_hull(group_costs, group_values)
            if not hull:
                return float("-inf")
            base_option = int(hull[0])
            base_cost += group_costs[base_option]
            base_value += group_values[base_option]
            for lower, upper in zip(hull[:-1], hull[1:]):
                lower_index = int(lower)
                upper_index = int(upper)
                length = group_costs[upper_index] - group_costs[lower_index]
                if length <= 0:
                    continue
                value_change = group_values[upper_index] - group_values[lower_index]
                if value_change > 0:
                    segments.append((value_change, length))
        residual = exact_capacity - base_cost
        if residual < 0:
            return float("-inf")

        def descending_slope(left: tuple[int, int], right: tuple[int, int]) -> int:
            comparison = left[0] * right[1] - right[0] * left[1]
            return -1 if comparison > 0 else (1 if comparison < 0 else 0)

        if segments:
            try:
                approximate_slopes = np.asarray(
                    [value_change / length for value_change, length in segments],
                    dtype=float,
                )
                order = np.argsort(-approximate_slopes, kind="stable").tolist()
                exactly_sorted = all(
                    descending_slope(segments[left], segments[right]) <= 0
                    for left, right in zip(order[:-1], order[1:])
                )
            except OverflowError:
                order = []
                exactly_sorted = False
            if not exactly_sorted:
                ordered_segments = sorted(
                    segments, key=cmp_to_key(descending_slope)
                )
            else:
                ordered_segments = [segments[index] for index in order]
        else:
            ordered_segments = []

        value = Fraction(base_value)
        for value_change, length in ordered_segments:
            if residual <= 0:
                break
            take = min(length, residual)
            if take == length:
                value += value_change
            else:
                value += Fraction(take * value_change, length)
            residual -= take
        value /= 1 << self._value_exponent
        try:
            rounded = float(value)
        except OverflowError:
            return -math.inf if value < 0 else float(np.finfo(float).max)
        if not math.isfinite(rounded):
            return -math.inf if value < 0 else float(np.finfo(float).max)
        if Fraction.from_float(rounded) > value:
            rounded = float(np.nextafter(rounded, -math.inf))
        return rounded


def _normalize_integer_row(values: Sequence[Fraction | int]) -> tuple[int, ...]:
    fractions = [value if isinstance(value, Fraction) else Fraction(value) for value in values]
    lcm = 1
    for value in fractions:
        lcm = math.lcm(lcm, value.denominator)
    integers = [int(value * lcm) for value in fractions]
    common = 0
    for value in integers:
        common = gcd(common, abs(value))
    if common > 1:
        integers = [value // common for value in integers]
    return tuple(integers)


def exact_robust_certificate(instance: PricingInstance, selection: Sequence[int]) -> int:
    margins = [int(instance.items[i][j].margin) for i, j in enumerate(selection)]
    deviations = sorted(
        [abs(int(instance.items[i][j].uncertainty)) for i, j in enumerate(selection)],
        reverse=True,
    )
    return int(sum(margins) - sum(deviations[: instance.gamma]))


def build_integer_instance(seed: int, n: int, m: int, gamma: int) -> PricingInstance:
    rng = np.random.default_rng(seed)
    items: list[list[Option]] = []
    for i in range(n):
        base = int(rng.integers(3, 7))
        group = [Option(value=0.0, margin=float(base), uncertainty=0.0)]
        for j in range(1, m):
            loss = int(rng.integers(j, 3 * j + 3))
            deviation = int(rng.integers(j, 4 * j + 4))
            group.append(
                Option(
                    value=float(5 * j + int(rng.integers(0, 5))),
                    margin=float(base - loss),
                    uncertainty=float(deviation),
                )
            )
        items.append(group)
    return PricingInstance(items=items, gamma=gamma, name=f"integer_s{seed}_n{n}_m{m}_g{gamma}")


def reduced_vertex(instance: PricingInstance, selection: Sequence[int]) -> tuple[int, ...]:
    row: list[int] = []
    for i, group in enumerate(instance.items):
        row.extend(1 if selection[i] == j else 0 for j in range(1, len(group)))
    return tuple(row)


def _full_dimensional(vertices: Sequence[Sequence[int]]) -> bool:
    if not vertices:
        return False
    matrix = np.asarray(vertices, dtype=float)
    if matrix.shape[0] <= matrix.shape[1]:
        return False
    return int(np.linalg.matrix_rank(matrix[1:] - matrix[0])) == matrix.shape[1]


def cdd_facets(vertices: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    if not CDDEXEC.exists():
        raise RuntimeError(f"cddlib executable not found at {CDDEXEC}")
    dimension = len(vertices[0])
    lines = ["V-representation", "begin", f"{len(vertices)} {dimension + 1} rational"]
    lines.extend("1 " + " ".join(str(int(value)) for value in vertex) for vertex in vertices)
    lines.extend(["end", ""])
    result = subprocess.run(
        [str(CDDEXEC), "--rep"],
        input="\n".join(lines),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cddlib failed: {result.stderr}\n{result.stdout}")
    output = result.stdout.splitlines()
    begin = next(i for i, line in enumerate(output) if line.strip() == "begin")
    header = output[begin + 1].split()
    count = int(header[0])
    rows: list[tuple[int, ...]] = []
    for line in output[begin + 2 : begin + 2 + count]:
        tokens = line.split()
        rows.append(_normalize_integer_row([Fraction(token) for token in tokens]))
    return rows


def conflict_row_reduced(
    instance: PricingInstance,
    conflict: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    dimension = sum(len(group) - 1 for group in instance.items)
    coefficients = [0] * dimension
    constant = len(conflict) - 1
    offset = 0
    for i, group in enumerate(instance.items):
        literal = next((j for item, j in conflict if item == i), None)
        if literal == 0:
            constant -= 1
            for k in range(len(group) - 1):
                coefficients[offset + k] += 1
        elif literal is not None:
            coefficients[offset + literal - 1] -= 1
        offset += len(group) - 1
    return _normalize_integer_row([constant, *coefficients])


def trivial_rows_reduced(instance: PricingInstance) -> set[tuple[int, ...]]:
    dimension = sum(len(group) - 1 for group in instance.items)
    rows: set[tuple[int, ...]] = set()
    offset = 0
    for group in instance.items:
        for k in range(len(group) - 1):
            coeff = [0] * dimension
            coeff[offset + k] = 1
            rows.add(_normalize_integer_row([0, *coeff]))
        coeff = [0] * dimension
        for k in range(len(group) - 1):
            coeff[offset + k] = -1
        rows.add(_normalize_integer_row([1, *coeff]))
        offset += len(group) - 1
    return rows


def facet_direction_full(instance: PricingInstance, facet: Sequence[int]) -> np.ndarray:
    direction: list[float] = []
    cursor = 1
    for group in instance.items:
        direction.append(0.0)
        for _ in range(1, len(group)):
            # To minimize b+a*y, maximize -a*y.
            direction.append(-float(facet[cursor]))
            cursor += 1
    return np.asarray(direction, dtype=float)


def run_facet_corpus(
    output_dir: Path,
    target_instances: int,
    seed: int,
    max_n: int,
) -> dict:
    rng = np.random.default_rng(seed)
    instance_rows: list[dict] = []
    facet_rows: list[dict] = []
    accepted = 0
    attempts = 0
    while accepted < target_instances:
        attempts += 1
        n = int(rng.integers(4, max_n + 1))
        m = 3
        gamma = int(rng.integers(1, min(3, n) + 1))
        inst = build_integer_instance(int(rng.integers(1, 2**31 - 1)), n, m, gamma)
        assignments = list(itertools.product(*(range(len(group)) for group in inst.items)))
        feasible = [selection for selection in assignments if exact_robust_certificate(inst, selection) >= 0]
        vertices = sorted(set(reduced_vertex(inst, selection) for selection in feasible))
        if len(feasible) in {0, len(assignments)} or not _full_dimensional(vertices):
            continue
        try:
            facets = cdd_facets(vertices)
        except (RuntimeError, subprocess.TimeoutExpired):
            continue
        conflicts = minimal_group_conflicts(inst, feasible)
        conflict_rows = {conflict_row_reduced(inst, conflict) for conflict in conflicts}
        trivial = trivial_rows_reduced(inst)
        accepted += 1
        counts = {"trivial": 0, "conflict": 0, "nonconflict": 0}
        cuts_disj = 0
        cuts_compact = 0
        nonconflict_cuts_disj = 0
        for facet_id, facet in enumerate(facets):
            if facet in trivial:
                kind = "trivial"
            elif facet in conflict_rows:
                kind = "conflict"
            else:
                kind = "nonconflict"
            counts[kind] += 1
            direction = facet_direction_full(inst, facet)
            compact_support, _ = compact_lp_support(inst, direction)
            disj_support = disjunctive_lp_support(inst, direction)
            compact_min = float(facet[0]) - compact_support
            disj_min = float(facet[0]) - disj_support
            cuts_c = compact_min < -1e-7
            cuts_d = disj_min < -1e-7
            cuts_compact += int(cuts_c)
            cuts_disj += int(cuts_d)
            nonconflict_cuts_disj += int(cuts_d and kind == "nonconflict")
            coefficients = facet[1:]
            facet_rows.append(
                {
                    "instance": inst.name,
                    "facet_id": facet_id,
                    "kind": kind,
                    "constant": facet[0],
                    "dimension": len(coefficients),
                    "support_size": sum(value != 0 for value in coefficients),
                    "distinct_abs_coefficients": len({abs(value) for value in coefficients if value}),
                    "max_abs_coefficient": max([0, *[abs(value) for value in coefficients]]),
                    "cuts_compact": cuts_c,
                    "cuts_disjunctive": cuts_d,
                    "compact_violation": max(0.0, -compact_min),
                    "disjunctive_violation": max(0.0, -disj_min),
                    "row": " ".join(map(str, facet)),
                }
            )
        instance_rows.append(
            {
                "instance": inst.name,
                "n": n,
                "m": m,
                "gamma": gamma,
                "assignments": len(assignments),
                "feasible_vertices": len(vertices),
                "facets": len(facets),
                "trivial_facets": counts["trivial"],
                "conflict_facets": counts["conflict"],
                "nonconflict_facets": counts["nonconflict"],
                "facets_cutting_compact": cuts_compact,
                "facets_cutting_disjunctive": cuts_disj,
                "nonconflict_facets_cutting_disjunctive": nonconflict_cuts_disj,
                "minimal_conflicts": len(conflicts),
            }
        )
        _write_csv(output_dir / "facet_instances.csv", instance_rows)
        _write_csv(output_dir / "facets.csv", facet_rows)

    nontrivial = [row for row in facet_rows if row["kind"] != "trivial"]
    nonconflict = [row for row in facet_rows if row["kind"] == "nonconflict"]
    disj_cutters = [row for row in facet_rows if row["cuts_disjunctive"]]
    summary = {
        "instances": accepted,
        "attempts": attempts,
        "total_facets": len(facet_rows),
        "nontrivial_facets": len(nontrivial),
        "conflict_facets": sum(row["kind"] == "conflict" for row in facet_rows),
        "nonconflict_facets": len(nonconflict),
        "facets_cutting_compact": sum(bool(row["cuts_compact"]) for row in facet_rows),
        "facets_cutting_disjunctive": len(disj_cutters),
        "nonconflict_facets_cutting_disjunctive": sum(
            row["kind"] == "nonconflict" for row in disj_cutters
        ),
        "fraction_nonconflict_among_nontrivial": len(nonconflict) / len(nontrivial) if nontrivial else 0.0,
        "fraction_disj_cutters_with_coefficient_gt_one": float(
            np.mean([row["max_abs_coefficient"] > 1 for row in disj_cutters])
        ) if disj_cutters else 0.0,
        "median_disj_cutter_support": float(
            np.median([row["support_size"] for row in disj_cutters])
        ) if disj_cutters else 0.0,
    }
    (output_dir / "facet_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@dataclass(frozen=True)
class SeparationResult:
    status: str
    conflict: Optional[tuple[tuple[int, int], ...]]
    objective: float
    runtime_seconds: float
    nodes: int


def exact_conflict_separation(
    instance: PricingInstance,
    x: np.ndarray,
    *,
    time_limit: float = 10.0,
) -> SeparationResult:
    """Solve the most-violated basic group-conflict separation MILP exactly."""

    from pyscipopt import Model, quicksum  # type: ignore

    start = time.perf_counter()
    sizes, offsets, _, _ = _flatten(instance)
    model = Model("robust_group_conflict_separation")
    model.hideOutput()
    model.setRealParam("limits/time", float(time_limit))
    try:
        model.setIntParam("parallel/maxnthreads", 1)
    except Exception:
        pass
    y = {
        (i, j): model.addVar(vtype="B", name=f"y_{i}_{j}")
        for i, group in enumerate(instance.items)
        for j in range(len(group))
    }
    for i, group in enumerate(instance.items):
        model.addCons(quicksum(y[i, j] for j in range(len(group))) <= 1.0)
    for theta in build_full_theta_candidates(instance):
        baseline = -float(instance.gamma) * float(theta)
        deltas: dict[tuple[int, int], float] = {}
        for i, group in enumerate(instance.items):
            contributions = [
                option.margin - max(0.0, abs(option.uncertainty) - theta)
                for option in group
            ]
            best = max(contributions)
            baseline += best
            for j, contribution in enumerate(contributions):
                deltas[i, j] = float(contribution - best)
        model.addCons(
            float(baseline)
            + quicksum(
                deltas[i, j] * y[i, j]
                for i, group in enumerate(instance.items)
                for j in range(len(group))
            )
            <= -1e-7
        )
    model.setObjective(
        quicksum(
            (1.0 - float(x[offsets[i] + j])) * y[i, j]
            for i, group in enumerate(instance.items)
            for j in range(len(group))
        ),
        "minimize",
    )
    model.optimize()
    status = str(model.getStatus()).lower()
    nodes = int(model.getNNodes())
    if model.getNSols() <= 0:
        return SeparationResult(status, None, float("inf"), time.perf_counter() - start, nodes)
    objective = float(model.getObjVal())
    selected = tuple(
        sorted(
            (i, j)
            for i, group in enumerate(instance.items)
            for j in range(len(group))
            if model.getVal(y[i, j]) > 0.5
        )
    )
    if not selected or maximum_completion_certificate(instance, selected) >= -1e-7:
        return SeparationResult("invalid_candidate", None, objective, time.perf_counter() - start, nodes)
    return SeparationResult(status, selected, objective, time.perf_counter() - start, nodes)


def exact_conflict_cut_loop(
    instance: PricingInstance,
    direction: np.ndarray,
    *,
    max_rounds: int = 10,
    separation_time_limit: float = 10.0,
) -> dict:
    start = time.perf_counter()
    cuts: list[tuple[tuple[int, int], ...]] = []
    bound, x = compact_lp_support(instance, direction)
    initial = bound
    separation_seconds = 0.0
    separation_nodes = 0
    statuses: list[str] = []
    for _ in range(max_rounds):
        if x is None:
            break
        separated = exact_conflict_separation(
            instance, x, time_limit=separation_time_limit
        )
        separation_seconds += separated.runtime_seconds
        separation_nodes += separated.nodes
        statuses.append(separated.status)
        if separated.conflict is None or separated.objective >= 1.0 - 1e-7:
            break
        conflict = separated.conflict
        # Greedy minimalization can only strengthen the violation.
        assignment = [next((j for i2, j in conflict if i2 == i), 0) for i in range(instance.n_items)]
        # Only shrink full assignments with the existing helper; partial
        # separator outputs are already valid and need not be minimal.
        if len(conflict) == instance.n_items:
            conflict = shrink_group_conflict(instance, assignment, x)
        if conflict in cuts:
            break
        cuts.append(conflict)
        bound, x = compact_lp_support(instance, direction, cuts)
    return {
        "initial_bound": initial,
        "final_bound": bound,
        "cuts": cuts,
        "cut_count": len(cuts),
        "runtime_seconds": time.perf_counter() - start,
        "separation_seconds": separation_seconds,
        "separation_nodes": separation_nodes,
        "statuses": statuses,
    }


def solve_scip_with_conflicts(
    instance: PricingInstance,
    conflicts: Sequence[tuple[tuple[int, int], ...]],
    *,
    time_limit: float,
) -> dict:
    from pyscipopt import Model, quicksum  # type: ignore

    start = time.perf_counter()
    model = Model("robust_mckp_with_group_conflicts")
    model.hideOutput()
    model.setRealParam("limits/time", float(time_limit))
    for name in ("parallel/maxnthreads", "lp/threads"):
        try:
            model.setIntParam(name, 1)
        except Exception:
            pass
    z = {
        (i, j): model.addVar(vtype="B", name=f"z_{i}_{j}")
        for i, group in enumerate(instance.items)
        for j in range(len(group))
    }
    theta = model.addVar(lb=0.0, vtype="C", name="theta")
    pi = {i: model.addVar(lb=0.0, vtype="C", name=f"pi_{i}") for i in range(instance.n_items)}
    for i, group in enumerate(instance.items):
        model.addCons(quicksum(z[i, j] for j in range(len(group))) == 1.0)
        model.addCons(
            pi[i]
            >= quicksum(abs(group[j].uncertainty) * z[i, j] for j in range(len(group)))
            - theta
        )
    model.addCons(
        quicksum(
            option.margin * z[i, j]
            for i, group in enumerate(instance.items)
            for j, option in enumerate(group)
        )
        - float(instance.gamma) * theta
        - quicksum(pi[i] for i in range(instance.n_items))
        >= 0.0
    )
    for conflict in conflicts:
        model.addCons(quicksum(z[i, j] for i, j in conflict) <= len(conflict) - 1)
    model.setObjective(
        quicksum(
            option.value * z[i, j]
            for i, group in enumerate(instance.items)
            for j, option in enumerate(group)
        ),
        "maximize",
    )
    model.optimize()
    status = str(model.getStatus()).lower()
    objective = float("nan")
    certificate = float("nan")
    selection = None
    if model.getNSols() > 0:
        selection = [
            max(range(len(group)), key=lambda j: float(model.getVal(z[i, j])))
            for i, group in enumerate(instance.items)
        ]
        objective = float(sum(instance.items[i][j].value for i, j in enumerate(selection)))
        certificate = compute_certificate(instance, selection)
    return {
        "status": status,
        "certified": status == "optimal" and selection is not None
        and certificate_is_feasible(instance, selection),
        "objective": objective,
        "certificate": certificate,
        "runtime_seconds": time.perf_counter() - start,
        "nodes": int(model.getNNodes()),
        "dual_bound": float(model.getDualbound()),
        "gap": float(model.getGap()) if model.getNSols() > 0 else float("inf"),
    }


def run_exact_cut_benchmark(
    output_dir: Path,
    sizes: Sequence[int],
    seeds: Sequence[int],
    families: Sequence[str],
    solve_time_limit: float,
    separation_time_limit: float,
) -> dict:
    rows: list[dict] = []
    for family, n, seed in itertools.product(families, sizes, seeds):
        instance = build_benchmark_instance(family, n, 6, _gamma_for(n, "sqrt"), seed)
        direction = np.array(
            [option.value for group in instance.items for option in group], dtype=float
        )
        compact, _ = compact_lp_support(instance, direction)
        disj_start = time.perf_counter()
        disj = disjunctive_lp_support(instance, direction)
        disj_seconds = time.perf_counter() - disj_start
        cut_loop = exact_conflict_cut_loop(
            instance,
            direction,
            max_rounds=10,
            separation_time_limit=separation_time_limit,
        )
        plain = solve_scip_with_conflicts(instance, [], time_limit=solve_time_limit)
        enhanced = solve_scip_with_conflicts(
            instance, cut_loop["cuts"], time_limit=solve_time_limit
        )
        reference = plain["objective"] if plain["certified"] else enhanced["objective"] if enhanced["certified"] else float("nan")
        scale = max(1.0, abs(reference)) if math.isfinite(reference) else 1.0
        if plain["certified"] and enhanced["certified"] and abs(plain["objective"] - enhanced["objective"]) > 1e-6:
            raise AssertionError(f"SCIP objective mismatch after valid cuts on {instance.name}")
        rows.append(
            {
                "instance": instance.name,
                "family": family,
                "n": n,
                "gamma": instance.gamma,
                "theta_count": len(build_full_theta_candidates(instance)),
                "reference_objective": reference,
                "compact_bound": compact,
                "disjunctive_bound": disj,
                "exact_cut_bound": cut_loop["final_bound"],
                "compact_gap_pct": 100.0 * (compact - reference) / scale if math.isfinite(reference) else float("nan"),
                "disjunctive_gap_pct": 100.0 * (disj - reference) / scale if math.isfinite(reference) else float("nan"),
                "exact_cut_gap_pct": 100.0 * (cut_loop["final_bound"] - reference) / scale if math.isfinite(reference) else float("nan"),
                "cut_count": cut_loop["cut_count"],
                "cut_loop_seconds": cut_loop["runtime_seconds"],
                "separation_seconds": cut_loop["separation_seconds"],
                "separation_nodes": cut_loop["separation_nodes"],
                "separation_statuses": ";".join(cut_loop["statuses"]),
                "disjunctive_seconds": disj_seconds,
                "plain_status": plain["status"],
                "plain_certified": plain["certified"],
                "plain_seconds": plain["runtime_seconds"],
                "plain_nodes": plain["nodes"],
                "plain_gap": plain["gap"],
                "enhanced_status": enhanced["status"],
                "enhanced_certified": enhanced["certified"],
                "enhanced_seconds": enhanced["runtime_seconds"],
                "enhanced_nodes": enhanced["nodes"],
                "enhanced_gap": enhanced["gap"],
            }
        )
        _write_csv(output_dir / "exact_cut_benchmark.csv", rows)
    finite = [row for row in rows if math.isfinite(float(row["reference_objective"]))]
    cut_positive = [row for row in rows if int(row["cut_count"]) > 0]
    both_certified = [row for row in rows if row["plain_certified"] and row["enhanced_certified"]]
    summary = {
        "instances": len(rows),
        "reference_available_rate": len(finite) / len(rows) if rows else 0.0,
        "exact_separator_finds_cut_rate": len(cut_positive) / len(rows) if rows else 0.0,
        "median_cut_count": float(np.median([row["cut_count"] for row in rows])) if rows else 0.0,
        "median_compact_gap_pct": float(np.median([row["compact_gap_pct"] for row in finite])) if finite else float("nan"),
        "median_disjunctive_gap_pct": float(np.median([row["disjunctive_gap_pct"] for row in finite])) if finite else float("nan"),
        "median_exact_cut_gap_pct": float(np.median([row["exact_cut_gap_pct"] for row in finite])) if finite else float("nan"),
        "median_plain_seconds": float(np.median([row["plain_seconds"] for row in rows])) if rows else float("nan"),
        "median_enhanced_seconds": float(np.median([row["enhanced_seconds"] for row in rows])) if rows else float("nan"),
        "median_enhanced_over_plain_speed_ratio": float(
            np.median([row["plain_seconds"] / max(row["enhanced_seconds"], 1e-12) for row in both_certified])
        ) if both_certified else float("nan"),
        "plain_certification_rate": float(np.mean([row["plain_certified"] for row in rows])) if rows else 0.0,
        "enhanced_certification_rate": float(np.mean([row["enhanced_certified"] for row in rows])) if rows else 0.0,
        "median_separation_seconds": float(np.median([row["separation_seconds"] for row in rows])) if rows else 0.0,
    }
    (output_dir / "exact_cut_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def adaptive_interval_bound(
    instance: PricingInstance,
    oracle: ThetaIntervalOracle,
    *,
    relative_tolerance: float,
    time_limit: float,
) -> dict:
    start = time.perf_counter()
    lp_cache: dict[int, float] = {}
    fixed_theta_oracle = FixedThetaLPOracle(
        instance,
        objective_range_validated=bool(
            getattr(oracle, "_objective_range_validated", False)
        ),
    )
    interval_bound_evaluations = 0
    multiplier_evaluations = 0
    certified_bound_evaluations = 0
    maximum_oracle_gap = 0.0
    maximum_scaled_oracle_gap = 0.0

    def fixed_lp(index: int) -> float:
        if index not in lp_cache:
            lp_cache[index] = fixed_theta_oracle.value(float(oracle.thetas[index]))
        return lp_cache[index]

    def interval_record(lo: int, hi: int) -> tuple[float, int, int]:
        nonlocal interval_bound_evaluations, multiplier_evaluations
        nonlocal certified_bound_evaluations, maximum_oracle_gap
        nonlocal maximum_scaled_oracle_gap
        bound = oracle.bound(lo, hi)
        interval_bound_evaluations += 1
        multiplier_evaluations += int(bound.evaluations)
        if bound.certified:
            certified_bound_evaluations += 1
            maximum_oracle_gap = max(maximum_oracle_gap, float(bound.optimality_gap))
            maximum_scaled_oracle_gap = max(
                maximum_scaled_oracle_gap,
                float(bound.optimality_gap)
                / max(1.0, abs(float(bound.upper_bound))),
            )
        # Candidate selection is deliberately oracle-independent so that the
        # end-to-end comparator differs only in its interval upper bound.
        for index in sorted({lo, hi, (lo + hi) // 2}):
            fixed_lp(index)
        return bound.upper_bound, lo, hi

    root = interval_record(0, len(oracle.thetas) - 1)
    root_bound = root[0]
    queue: list[tuple[float, int, int]] = [(-root[0], root[1], root[2])]
    lower = max(lp_cache.values(), default=float("-inf"))
    if not math.isfinite(lower):
        # A finite incumbent is required before applying the relative-gap test.
        # Every released timed comparison instance has a feasible endpoint or
        # midpoint; this fallback makes the public routine correct generally.
        for index in range(len(oracle.thetas)):
            candidate = fixed_lp(index)
            if math.isfinite(candidate):
                lower = candidate
                break
        if not math.isfinite(lower):
            return {
                "status": "infeasible",
                "root_upper_bound": root_bound,
                "lower_bound": float("-inf"),
                "upper_bound": float("-inf"),
                "relative_gap": 0.0,
                "theta_lp_evaluations": len(lp_cache),
                "interval_bound_evaluations": interval_bound_evaluations,
                "multiplier_evaluations": multiplier_evaluations,
                "certified_bound_evaluations": certified_bound_evaluations,
                "all_interval_bounds_certified": (
                    certified_bound_evaluations == interval_bound_evaluations
                ),
                "maximum_oracle_optimality_gap": maximum_oracle_gap,
                "maximum_scaled_oracle_optimality_gap": maximum_scaled_oracle_gap,
                "interval_splits": 0,
                "runtime_seconds": time.perf_counter() - start,
            }
    discarded_upper = float("-inf")
    splits = 0
    status = "time_limit"
    while queue and time.perf_counter() - start < time_limit:
        upper = max(lower, discarded_upper, -queue[0][0])
        if upper - lower <= relative_tolerance * max(1.0, abs(lower)):
            status = "tolerance"
            break
        neg_upper, lo, hi = heapq.heappop(queue)
        if lo == hi:
            fixed_lp(lo)
            lower = max(lower, lp_cache[lo])
            continue
        mid = (lo + hi) // 2
        for child_lo, child_hi in ((lo, mid), (mid + 1, hi)):
            if child_lo == child_hi:
                fixed_lp(child_lo)
                lower = max(lower, lp_cache[child_lo])
                continue
            child = interval_record(child_lo, child_hi)
            lower = max(lower, max(lp_cache.values(), default=lower))
            if child[0] > lower + relative_tolerance * max(1.0, abs(lower)):
                heapq.heappush(queue, (-child[0], child_lo, child_hi))
            else:
                discarded_upper = max(discarded_upper, child[0])
        splits += 1
    upper = max(lower, discarded_upper, -queue[0][0] if queue else float("-inf"))
    if not queue:
        status = "exact" if upper <= lower else "tolerance"
    return {
        "status": status,
        "root_upper_bound": root_bound,
        "lower_bound": lower,
        "upper_bound": upper,
        "relative_gap": (upper - lower) / max(1.0, abs(lower)),
        "theta_lp_evaluations": len(lp_cache),
        "interval_bound_evaluations": interval_bound_evaluations,
        "multiplier_evaluations": multiplier_evaluations,
        "certified_bound_evaluations": certified_bound_evaluations,
        "all_interval_bounds_certified": (
            certified_bound_evaluations == interval_bound_evaluations
        ),
        "maximum_oracle_optimality_gap": maximum_oracle_gap,
        "maximum_scaled_oracle_optimality_gap": maximum_scaled_oracle_gap,
        "interval_splits": splits,
        "runtime_seconds": time.perf_counter() - start,
    }


def run_scaling_study(
    output_dir: Path,
    sizes: Sequence[int],
    seeds: Sequence[int],
    families: Sequence[str],
    adaptive_time_limit: float,
) -> dict:
    rows: list[dict] = []
    for family, n, seed in itertools.product(families, sizes, seeds):
        instance = build_benchmark_instance(family, n, 6, _gamma_for(n, "sqrt"), seed)
        oracle_start = time.perf_counter()
        oracle = ThetaIntervalOracle(instance)
        root = oracle.bound(0, len(oracle.thetas) - 1)
        root_total = time.perf_counter() - oracle_start
        adaptive = adaptive_interval_bound(
            instance,
            oracle,
            relative_tolerance=1e-6,
            time_limit=adaptive_time_limit,
        )
        scan_value, scan_seconds, feasible_theta = full_theta_lp_scan(instance)
        if root.upper_bound < scan_value - 1e-5 or adaptive["upper_bound"] < scan_value - 1e-5:
            raise AssertionError(f"invalid interval bound on {instance.name}")
        rows.append(
            {
                "instance": instance.name,
                "family": family,
                "n": n,
                "theta_count": len(oracle.thetas),
                "feasible_theta_count": feasible_theta,
                "oracle_preprocessing_seconds": oracle.preprocessing_seconds,
                "root_interval_total_seconds": root_total,
                "root_interval_upper": root.upper_bound,
                "full_scan_seconds": scan_seconds,
                "full_scan_value": scan_value,
                "root_speedup": scan_seconds / max(root_total, 1e-12),
                "root_excess_pct": 100.0 * (root.upper_bound - scan_value) / max(1.0, abs(scan_value)),
                "adaptive_status": adaptive["status"],
                "adaptive_seconds": adaptive["runtime_seconds"],
                "adaptive_lower": adaptive["lower_bound"],
                "adaptive_upper": adaptive["upper_bound"],
                "adaptive_relative_gap": adaptive["relative_gap"],
                "adaptive_theta_lp_evaluations": adaptive["theta_lp_evaluations"],
                "adaptive_theta_fraction": adaptive["theta_lp_evaluations"] / len(oracle.thetas),
                "adaptive_interval_splits": adaptive["interval_splits"],
                "adaptive_speedup": scan_seconds / max(adaptive["runtime_seconds"] + oracle.preprocessing_seconds, 1e-12),
            }
        )
        _write_csv(output_dir / "scaling_benchmark.csv", rows)
    summary = {
        "instances": len(rows),
        "max_n": max(sizes) if sizes else 0,
        "median_root_speedup": float(np.median([row["root_speedup"] for row in rows])) if rows else float("nan"),
        "median_adaptive_speedup": float(np.median([row["adaptive_speedup"] for row in rows])) if rows else float("nan"),
        "five_x_root_speedup_rate": float(np.mean([row["root_speedup"] >= 5.0 for row in rows])) if rows else 0.0,
        "five_x_adaptive_speedup_rate": float(np.mean([row["adaptive_speedup"] >= 5.0 for row in rows])) if rows else 0.0,
        "adaptive_tolerance_rate": float(np.mean([row["adaptive_relative_gap"] <= 1e-6 for row in rows])) if rows else 0.0,
        "median_adaptive_theta_fraction": float(np.median([row["adaptive_theta_fraction"] for row in rows])) if rows else float("nan"),
        "median_root_excess_pct": float(np.median([row["root_excess_pct"] for row in rows])) if rows else float("nan"),
    }
    (output_dir / "scaling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def bounded_theta_strong_lp(
    instance: PricingInstance,
    lower_theta: float,
    upper_theta: float,
) -> tuple[float, float]:
    """Büsing--Gersing--Koster bounded-z strong LP specialized to menu variables."""

    start = time.perf_counter()
    sizes, offsets, margins, deviations = _flatten(instance)
    n_x = offsets[-1]
    p_offset = n_x
    theta_index = 2 * n_x
    nvars = 2 * n_x + 1
    values = np.array(
        [option.value for group in instance.items for option in group], dtype=float
    )
    c = np.zeros(nvars, dtype=float)
    c[:n_x] = -values
    a_eq = np.zeros((instance.n_items, nvars), dtype=float)
    for i, size in enumerate(sizes):
        a_eq[i, offsets[i] : offsets[i] + size] = 1.0
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    robust = np.zeros(nvars, dtype=float)
    robust[:n_x] = -margins
    robust[p_offset : p_offset + n_x] = 1.0
    robust[theta_index] = float(instance.gamma)
    rows.append(robust)
    rhs.append(0.0)
    for k in range(n_x):
        # p_k + theta >= (d_k-lower)^+ x_k + lower.
        first = np.zeros(nvars, dtype=float)
        first[k] = max(0.0, float(deviations[k]) - float(lower_theta))
        first[p_offset + k] = -1.0
        first[theta_index] = -1.0
        rows.append(first)
        rhs.append(-float(lower_theta))
        # p_k >= (d_k-upper)^+ x_k.
        second = np.zeros(nvars, dtype=float)
        second[k] = max(0.0, float(deviations[k]) - float(upper_theta))
        second[p_offset + k] = -1.0
        rows.append(second)
        rhs.append(0.0)
    bounds = (
        [(0.0, 1.0)] * n_x
        + [(0.0, None)] * n_x
        + [(float(lower_theta), float(upper_theta))]
    )
    result = opt.linprog(
        c,
        A_ub=np.vstack(rows),
        b_ub=np.array(rhs, dtype=float),
        A_eq=a_eq,
        b_eq=np.ones(instance.n_items),
        bounds=bounds,
        method="highs",
    )
    if result.status == 2:
        return float("-inf"), time.perf_counter() - start
    if not result.success:
        raise RuntimeError(f"bounded-theta strong LP failed: {result.message}")
    return float(-result.fun), time.perf_counter() - start


def bounded_theta_clique_lp(
    instance: PricingInstance,
    lower_theta: float,
    upper_theta: float,
    *,
    return_diagnostics: bool = False,
) -> tuple[float, float, float] | tuple[float, float, float, dict]:
    """Bounded-z LP strengthened by each exactly-one menu (GUB clique).

    This is the direct specialization of the clique formulation in
    Buesing--Gersing--Koster to one mutually-exclusive option group per item.
    ``theta_value`` is the LP solution and is used only to select promising
    fixed-theta lower-bound evaluations in the adaptive comparison.
    """

    start = time.perf_counter()
    sizes, offsets, margins, deviations = _flatten(instance)
    n_x = offsets[-1]
    p_offset = n_x
    z_index = n_x + instance.n_items
    nvars = z_index + 1
    values = np.array(
        [option.value for group in instance.items for option in group], dtype=float
    )
    c = np.zeros(nvars, dtype=float)
    c[:n_x] = -values
    eq_rows = np.repeat(np.arange(instance.n_items, dtype=int), sizes)
    eq_cols = np.arange(n_x, dtype=int)
    a_eq = sparse.csr_matrix(
        (np.ones(n_x, dtype=float), (eq_rows, eq_cols)),
        shape=(instance.n_items, nvars),
    )

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    robust_x = -margins + np.maximum(0.0, deviations - float(upper_theta))
    nonzero = np.flatnonzero(robust_x)
    ub_rows.extend([0] * len(nonzero))
    ub_cols.extend(nonzero.tolist())
    ub_data.extend(robust_x[nonzero].tolist())
    ub_rows.extend([0] * instance.n_items)
    ub_cols.extend(range(p_offset, p_offset + instance.n_items))
    ub_data.extend([1.0] * instance.n_items)
    if instance.gamma:
        ub_rows.append(0)
        ub_cols.append(z_index)
        ub_data.append(float(instance.gamma))
    rhs: list[float] = [-float(instance.gamma) * float(lower_theta)]

    clipped = np.maximum(
        0.0,
        np.minimum(deviations, float(upper_theta)) - float(lower_theta),
    )
    for i, size in enumerate(sizes):
        start_i, stop_i = offsets[i], offsets[i] + size
        local = clipped[start_i:stop_i]
        local_nonzero = np.flatnonzero(local)
        ub_rows.extend([i + 1] * len(local_nonzero))
        ub_cols.extend((start_i + local_nonzero).tolist())
        ub_data.extend(local[local_nonzero].tolist())
        ub_rows.extend([i + 1, i + 1])
        ub_cols.extend([p_offset + i, z_index])
        ub_data.extend([-1.0, -1.0])
        rhs.append(0.0)

    a_ub = sparse.csr_matrix(
        (np.asarray(ub_data), (np.asarray(ub_rows), np.asarray(ub_cols))),
        shape=(instance.n_items + 1, nvars),
    )

    bounds = (
        [(0.0, 1.0)] * n_x
        + [(0.0, None)] * instance.n_items
        + [(0.0, float(upper_theta) - float(lower_theta))]
    )
    linprog_arguments = {
        "c": c,
        "A_ub": a_ub,
        "b_ub": np.asarray(rhs, dtype=float),
        "A_eq": a_eq,
        "b_eq": np.ones(instance.n_items),
        "bounds": bounds,
    }
    attempts = []
    for method in ("highs", "highs-ds", "highs-ipm"):
        result = opt.linprog(**linprog_arguments, method=method)
        attempts.append((method, result))
        if result.success or result.status == 2:
            break
    if result.status == 2:
        payload = (float("-inf"), time.perf_counter() - start, float(lower_theta))
        if return_diagnostics:
            return (*payload, {
                "matrix_nnz": int(a_ub.nnz + a_eq.nnz),
                "matrix_storage_bytes": int(
                    a_ub.data.nbytes + a_ub.indices.nbytes + a_ub.indptr.nbytes
                    + a_eq.data.nbytes + a_eq.indices.nbytes + a_eq.indptr.nbytes
                ),
                "solver_iterations": int(getattr(result, "nit", 0) or 0),
            })
        return payload
    if not result.success:
        diagnostics = "; ".join(
            f"{method}: status={candidate.status}, message={candidate.message}"
            for method, candidate in attempts
        )
        raise RuntimeError(f"bounded-theta clique LP failed after certified retries: {diagnostics}")
    payload = (
        float(-result.fun),
        time.perf_counter() - start,
        float(lower_theta) + float(result.x[z_index]),
    )
    if return_diagnostics:
        return (*payload, {
            "matrix_nnz": int(a_ub.nnz + a_eq.nnz),
            "matrix_storage_bytes": int(
                a_ub.data.nbytes + a_ub.indices.nbytes + a_ub.indptr.nbytes
                + a_eq.data.nbytes + a_eq.indices.nbytes + a_eq.indptr.nbytes
            ),
            "solver_iterations": int(getattr(result, "nit", 0) or 0),
        })
    return payload


def adaptive_clique_interval_bound(
    instance: PricingInstance,
    *,
    relative_tolerance: float,
    time_limit: float,
) -> dict:
    """Best-first bounded-z clique interval decomposition baseline."""

    start = time.perf_counter()
    thetas = np.asarray(build_full_theta_candidates(instance), dtype=float)
    lp_cache: dict[int, float] = {}
    fixed_theta_oracle = FixedThetaLPOracle(instance)
    interval_lp_seconds = 0.0
    interval_lp_evaluations = 0
    interval_matrix_nnz = 0
    interval_matrix_storage_bytes = 0
    interval_solver_iterations = 0

    def fixed_lp(index: int) -> float:
        if index not in lp_cache:
            lp_cache[index] = fixed_theta_oracle.value(float(thetas[index]))
        return lp_cache[index]

    def interval_record(lo: int, hi: int) -> tuple[float, int, int]:
        nonlocal interval_lp_seconds, interval_lp_evaluations
        nonlocal interval_matrix_nnz, interval_matrix_storage_bytes
        nonlocal interval_solver_iterations
        bound, seconds, _theta_value, diagnostics = bounded_theta_clique_lp(
            instance,
            float(thetas[lo]),
            float(thetas[hi]),
            return_diagnostics=True,
        )
        interval_lp_seconds += seconds
        interval_lp_evaluations += 1
        interval_matrix_nnz += int(diagnostics["matrix_nnz"])
        interval_matrix_storage_bytes += int(diagnostics["matrix_storage_bytes"])
        interval_solver_iterations += int(diagnostics["solver_iterations"])
        for index in sorted({lo, hi, (lo + hi) // 2}):
            fixed_lp(index)
        return bound, lo, hi

    root = interval_record(0, len(thetas) - 1)
    root_bound = root[0]
    queue: list[tuple[float, int, int]] = [(-root[0], root[1], root[2])]
    lower = max(lp_cache.values(), default=float("-inf"))
    if not math.isfinite(lower):
        # See the analogous fallback in adaptive_interval_bound.  It is outside
        # the timed path of every released instance but prevents an infinite
        # scaling term from satisfying the stopping test on a general input.
        for index in range(len(thetas)):
            candidate = fixed_lp(index)
            if math.isfinite(candidate):
                lower = candidate
                break
        if not math.isfinite(lower):
            return {
                "status": "infeasible",
                "root_upper_bound": root_bound,
                "lower_bound": float("-inf"),
                "upper_bound": float("-inf"),
                "relative_gap": 0.0,
                "theta_lp_evaluations": len(lp_cache),
                "interval_lp_evaluations": interval_lp_evaluations,
                "interval_splits": 0,
                "interval_lp_seconds": interval_lp_seconds,
                "interval_matrix_nnz": interval_matrix_nnz,
                "interval_matrix_storage_bytes": interval_matrix_storage_bytes,
                "interval_solver_iterations": interval_solver_iterations,
                "runtime_seconds": time.perf_counter() - start,
            }
    discarded_upper = float("-inf")
    splits = 0
    status = "time_limit"
    while queue and time.perf_counter() - start < time_limit:
        upper = max(lower, discarded_upper, -queue[0][0])
        if upper - lower <= relative_tolerance * max(1.0, abs(lower)):
            status = "tolerance"
            break
        _neg_upper, lo, hi = heapq.heappop(queue)
        if lo == hi:
            fixed_lp(lo)
            lower = max(lower, lp_cache[lo])
            continue
        mid = (lo + hi) // 2
        for child_lo, child_hi in ((lo, mid), (mid + 1, hi)):
            if child_lo == child_hi:
                fixed_lp(child_lo)
                lower = max(lower, lp_cache[child_lo])
                continue
            child = interval_record(child_lo, child_hi)
            lower = max(lower, max(lp_cache.values(), default=lower))
            if child[0] > lower + relative_tolerance * max(1.0, abs(lower)):
                heapq.heappush(queue, (-child[0], child_lo, child_hi))
            else:
                discarded_upper = max(discarded_upper, child[0])
        splits += 1
    upper = max(lower, discarded_upper, -queue[0][0] if queue else float("-inf"))
    if not queue:
        status = "exact" if upper <= lower else "tolerance"
    return {
        "status": status,
        "root_upper_bound": root_bound,
        "lower_bound": lower,
        "upper_bound": upper,
        "relative_gap": (upper - lower) / max(1.0, abs(lower)),
        "theta_lp_evaluations": len(lp_cache),
        "interval_lp_evaluations": interval_lp_evaluations,
        "interval_splits": splits,
        "interval_lp_seconds": interval_lp_seconds,
        "interval_matrix_nnz": interval_matrix_nnz,
        "interval_matrix_storage_bytes": interval_matrix_storage_bytes,
        "interval_solver_iterations": interval_solver_iterations,
        "runtime_seconds": time.perf_counter() - start,
    }


def run_strong_bound_comparison(
    output_dir: Path,
    sizes: Sequence[int],
    seeds: Sequence[int],
    families: Sequence[str],
) -> dict:
    rows: list[dict] = []
    for family, n, seed in itertools.product(families, sizes, seeds):
        instance = build_benchmark_instance(family, n, 6, _gamma_for(n, "sqrt"), seed)
        thetas = build_full_theta_candidates(instance)
        generic_bound, generic_seconds = bounded_theta_strong_lp(
            instance, min(thetas), max(thetas)
        )
        clique_bound, clique_seconds, _ = bounded_theta_clique_lp(
            instance, min(thetas), max(thetas)
        )
        oracle_start = time.perf_counter()
        oracle = ThetaIntervalOracle(instance)
        lagrangian = oracle.bound(0, len(oracle.thetas) - 1)
        lagrangian_total = time.perf_counter() - oracle_start
        disj, disj_seconds, _ = full_theta_lp_scan(instance)
        if min(generic_bound, clique_bound, lagrangian.upper_bound) < disj - 1e-5:
            raise AssertionError(f"invalid strong interval bound on {instance.name}")
        scale = max(1.0, abs(disj))
        rows.append(
            {
                "instance": instance.name,
                "family": family,
                "n": n,
                "theta_count": len(thetas),
                "disjunctive_bound": disj,
                "disjunctive_seconds": disj_seconds,
                "generic_bounded_theta_bound": generic_bound,
                "generic_bounded_theta_seconds": generic_seconds,
                "generic_excess_pct": 100.0 * (generic_bound - disj) / scale,
                "clique_bounded_theta_bound": clique_bound,
                "clique_bounded_theta_seconds": clique_seconds,
                "clique_excess_pct": 100.0 * (clique_bound - disj) / scale,
                "lagrangian_interval_bound": lagrangian.upper_bound,
                "lagrangian_interval_seconds": lagrangian_total,
                "lagrangian_excess_pct": 100.0 * (lagrangian.upper_bound - disj) / scale,
                "lagrangian_tighter": lagrangian.upper_bound < generic_bound - 1e-7,
                "lagrangian_tighter_than_clique": lagrangian.upper_bound < clique_bound - 1e-7,
                "generic_tighter": generic_bound < lagrangian.upper_bound - 1e-7,
                "clique_tighter_than_generic": clique_bound < generic_bound - 1e-7,
                "lagrangian_vs_generic_time_ratio": generic_seconds / max(lagrangian_total, 1e-12),
            }
        )
        _write_csv(output_dir / "strong_bound_comparison.csv", rows)
    summary = {
        "instances": len(rows),
        "lagrangian_tighter_rate": float(np.mean([row["lagrangian_tighter"] for row in rows])) if rows else 0.0,
        "lagrangian_tighter_than_clique_rate": float(np.mean([row["lagrangian_tighter_than_clique"] for row in rows])) if rows else 0.0,
        "clique_tighter_than_generic_rate": float(np.mean([row["clique_tighter_than_generic"] for row in rows])) if rows else 0.0,
        "generic_tighter_rate": float(np.mean([row["generic_tighter"] for row in rows])) if rows else 0.0,
        "median_generic_excess_pct": float(np.median([row["generic_excess_pct"] for row in rows])) if rows else float("nan"),
        "median_clique_excess_pct": float(np.median([row["clique_excess_pct"] for row in rows])) if rows else float("nan"),
        "median_lagrangian_excess_pct": float(np.median([row["lagrangian_excess_pct"] for row in rows])) if rows else float("nan"),
        "median_generic_seconds": float(np.median([row["generic_bounded_theta_seconds"] for row in rows])) if rows else float("nan"),
        "median_clique_seconds": float(np.median([row["clique_bounded_theta_seconds"] for row in rows])) if rows else float("nan"),
        "median_lagrangian_seconds": float(np.median([row["lagrangian_interval_seconds"] for row in rows])) if rows else float("nan"),
        "median_lagrangian_vs_generic_time_ratio": float(np.median([row["lagrangian_vs_generic_time_ratio"] for row in rows])) if rows else float("nan"),
    }
    (output_dir / "strong_bound_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_clique_interval_comparison(
    output_dir: Path,
    sizes: Sequence[int],
    seeds: Sequence[int],
    families: Sequence[str],
    time_limit: float,
) -> dict:
    """Compare adaptive Lagrangian compression to the prior-art clique LP."""

    rows: list[dict] = []
    for family, n, seed in itertools.product(families, sizes, seeds):
        instance = build_benchmark_instance(family, n, 6, _gamma_for(n, "sqrt"), seed)
        scan_value, scan_seconds, _ = full_theta_lp_scan(instance)

        oracle = ThetaIntervalOracle(instance)
        lagrangian = adaptive_interval_bound(
            instance,
            oracle,
            relative_tolerance=1e-6,
            time_limit=time_limit,
        )
        lagrangian_total = oracle.preprocessing_seconds + lagrangian["runtime_seconds"]
        clique = adaptive_clique_interval_bound(
            instance,
            relative_tolerance=1e-6,
            time_limit=time_limit,
        )
        if lagrangian["upper_bound"] < scan_value - 1e-5:
            raise AssertionError(f"invalid Lagrangian adaptive bound on {instance.name}")
        if clique["upper_bound"] < scan_value - 1e-5:
            raise AssertionError(f"invalid clique adaptive bound on {instance.name}")
        rows.append(
            {
                "instance": instance.name,
                "family": family,
                "n": n,
                "theta_count": len(oracle.thetas),
                "full_scan_value": scan_value,
                "full_scan_seconds": scan_seconds,
                "lagrangian_status": lagrangian["status"],
                "lagrangian_total_seconds": lagrangian_total,
                "lagrangian_relative_gap": lagrangian["relative_gap"],
                "lagrangian_theta_lp_evaluations": lagrangian["theta_lp_evaluations"],
                "lagrangian_splits": lagrangian["interval_splits"],
                "lagrangian_speedup_over_scan": scan_seconds / max(lagrangian_total, 1e-12),
                "clique_status": clique["status"],
                "clique_total_seconds": clique["runtime_seconds"],
                "clique_relative_gap": clique["relative_gap"],
                "clique_theta_lp_evaluations": clique["theta_lp_evaluations"],
                "clique_interval_lp_evaluations": clique["interval_lp_evaluations"],
                "clique_splits": clique["interval_splits"],
                "clique_speedup_over_scan": scan_seconds / max(clique["runtime_seconds"], 1e-12),
                "lagrangian_speedup_over_clique": clique["runtime_seconds"] / max(lagrangian_total, 1e-12),
            }
        )
        _write_csv(output_dir / "clique_interval_comparison.csv", rows)
    summary = {
        "instances": len(rows),
        "lagrangian_tolerance_rate": float(np.mean([row["lagrangian_relative_gap"] <= 1e-6 for row in rows])) if rows else 0.0,
        "clique_tolerance_rate": float(np.mean([row["clique_relative_gap"] <= 1e-6 for row in rows])) if rows else 0.0,
        "median_lagrangian_seconds": float(np.median([row["lagrangian_total_seconds"] for row in rows])) if rows else float("nan"),
        "median_clique_seconds": float(np.median([row["clique_total_seconds"] for row in rows])) if rows else float("nan"),
        "median_lagrangian_speedup_over_clique": float(np.median([row["lagrangian_speedup_over_clique"] for row in rows])) if rows else float("nan"),
        "median_lagrangian_speedup_over_scan": float(np.median([row["lagrangian_speedup_over_scan"] for row in rows])) if rows else float("nan"),
        "median_clique_speedup_over_scan": float(np.median([row["clique_speedup_over_scan"] for row in rows])) if rows else float("nan"),
        "median_lagrangian_splits": float(np.median([row["lagrangian_splits"] for row in rows])) if rows else float("nan"),
        "median_clique_splits": float(np.median([row["clique_splits"] for row in rows])) if rows else float("nan"),
    }
    (output_dir / "clique_interval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def write_environment(output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cddexec": str(CDDEXEC),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
        },
    }
    (output_dir / "environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["facets", "cuts", "scaling", "strong", "clique", "all"]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "structural_feasibility_20260719",
    )
    parser.add_argument("--facet-instances", type=int, default=24)
    parser.add_argument("--facet-max-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--sizes", default="30,60,90")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--families",
        default="dense_frontier,correlated_risk,near_tie,many_breakpoints",
    )
    parser.add_argument("--solve-time-limit", type=float, default=15.0)
    parser.add_argument("--separation-time-limit", type=float, default=10.0)
    parser.add_argument("--scaling-sizes", default="90,180,360")
    parser.add_argument("--scaling-families", default="dense_frontier,many_breakpoints")
    parser.add_argument("--adaptive-time-limit", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    write_environment(output_dir, args)
    result: dict[str, object] = {}
    if args.command in {"facets", "all"}:
        result["facets"] = run_facet_corpus(
            output_dir,
            target_instances=args.facet_instances,
            seed=args.seed,
            max_n=args.facet_max_n,
        )
    if args.command in {"cuts", "all"}:
        sizes = [int(value) for value in args.sizes.split(",") if value]
        seeds = [int(value) for value in args.seeds.split(",") if value]
        families = [value for value in args.families.split(",") if value]
        result["cuts"] = run_exact_cut_benchmark(
            output_dir,
            sizes=sizes,
            seeds=seeds,
            families=families,
            solve_time_limit=args.solve_time_limit,
            separation_time_limit=args.separation_time_limit,
        )
    if args.command in {"scaling", "all"}:
        sizes = [int(value) for value in args.scaling_sizes.split(",") if value]
        seeds = [int(value) for value in args.seeds.split(",") if value]
        families = [value for value in args.scaling_families.split(",") if value]
        result["scaling"] = run_scaling_study(
            output_dir,
            sizes=sizes,
            seeds=seeds,
            families=families,
            adaptive_time_limit=args.adaptive_time_limit,
        )
    if args.command in {"strong", "all"}:
        sizes = [int(value) for value in args.sizes.split(",") if value]
        seeds = [int(value) for value in args.seeds.split(",") if value]
        families = [value for value in args.families.split(",") if value]
        result["strong"] = run_strong_bound_comparison(
            output_dir,
            sizes=sizes,
            seeds=seeds,
            families=families,
        )
    if args.command in {"clique", "all"}:
        sizes = [int(value) for value in args.scaling_sizes.split(",") if value]
        seeds = [int(value) for value in args.seeds.split(",") if value]
        families = [value for value in args.scaling_families.split(",") if value]
        result["clique"] = run_clique_interval_comparison(
            output_dir,
            sizes=sizes,
            seeds=seeds,
            families=families,
            time_limit=args.adaptive_time_limit,
        )
    (output_dir / "study_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
