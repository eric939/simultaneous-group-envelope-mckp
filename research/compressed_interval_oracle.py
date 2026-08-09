#!/usr/bin/env python3
"""Compressed group-envelope oracle for robust-MCKP threshold intervals.

For fixed multiplier ``lambda``, the baseline-slack Lagrangian expression

    lambda * C(theta) + sum_i max_j(v_ij - lambda*c_ij(theta))

simplifies exactly to

    sum_i max_j(v_ij + lambda*s_ij(theta)) - lambda*Gamma*theta.

Within a group and between two option-deviation breakpoints, all unsaturated
options have the same slope ``lambda`` and all saturated options are constant.
The group envelope is therefore the maximum of one line and one constant.  We
accumulate those pieces with range differences, avoiding the dense
``groups x thresholds x options`` tensor used by the original research oracle.

The production ``bound`` method also certifies multiplier minimization.  The
interval objective is convex and piecewise linear in ``lambda``.  Geometric
bracketing followed by deterministic golden-section contraction locates a
minimizer interval, and an explicit Lipschitz constant turns its width into an
objective-gap certificate.  The reported upper bound remains an explicitly
evaluated Lagrangian value.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from robust_mckp import PricingInstance
from robust_mckp.exact_bnb import build_full_theta_candidates
from research.novelty_go_no_go import IntervalBound


TOL = 1e-9


@dataclass(frozen=True)
class OracleProfile:
    groups: int
    options: int
    thresholds: int
    local_segments: int
    preprocessing_seconds: float


class CompressedThetaIntervalOracle:
    """Evaluate the same Lagrangian function in O(B+K log B) time."""

    def __init__(self, instance: PricingInstance):
        start = time.perf_counter()
        self.instance = instance
        self.thetas = np.asarray(build_full_theta_candidates(instance), dtype=float)
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
        max_options = max(len(values) for values in self.values)
        n_groups = len(self.values)
        self._sorted_values = np.full((n_groups, max_options), -np.inf, dtype=float)
        self._sorted_margins = np.zeros((n_groups, max_options), dtype=float)
        self._sorted_deviations = np.zeros((n_groups, max_options), dtype=float)
        self._sorted_eligible = np.zeros((n_groups, max_options), dtype=bool)
        self._group_sizes = np.asarray([len(values) for values in self.values], dtype=int)
        segment_lo: list[int] = []
        segment_hi: list[int] = []
        segment_group: list[int] = []
        segment_last_saturated: list[int] = []
        local_segments = 0
        for group_index, (values, margins, deviations) in enumerate(
            zip(self.values, self.margins, self.deviations)
        ):
            order = np.argsort(deviations, kind="stable")
            sorted_values = values[order]
            sorted_margins = margins[order]
            sorted_deviations = deviations[order]
            size = len(values)
            self._sorted_values[group_index, :size] = sorted_values
            self._sorted_margins[group_index, :size] = sorted_margins
            self._sorted_deviations[group_index, :size] = sorted_deviations
            self._sorted_eligible[group_index, :size] = True
            positions = np.searchsorted(self.thetas, sorted_deviations, side="left")
            positions = np.clip(positions, 0, len(self.thetas) - 1)
            starts = np.unique(np.concatenate([np.array([0], dtype=int), positions]))
            local_segments += len(starts)
            for segment_index, lo_raw in enumerate(starts):
                lo = int(lo_raw)
                hi = (
                    int(starts[segment_index + 1]) - 1
                    if segment_index + 1 < len(starts)
                    else len(self.thetas) - 1
                )
                if lo > hi:
                    continue
                segment_lo.append(lo)
                segment_hi.append(hi)
                segment_group.append(group_index)
                segment_last_saturated.append(
                    int(
                        np.searchsorted(
                            sorted_deviations,
                            float(self.thetas[lo]),
                            side="right",
                        )
                        - 1
                    )
                )
        self._segment_lo = np.asarray(segment_lo, dtype=int)
        self._segment_hi = np.asarray(segment_hi, dtype=int)
        self._segment_group = np.asarray(segment_group, dtype=int)
        self._segment_last_saturated = np.asarray(
            segment_last_saturated, dtype=int
        )

        value_array = np.concatenate(self.values)
        deviation_array = np.concatenate(self.deviations)
        value_scale = max(
            1.0,
            float(np.ptp(value_array)),
            float(np.max(np.abs(value_array))),
        )
        # Only within-menu tradeoffs matter for a group-separable multiplier.
        # Forming all global pairwise differences would be quadratic in the
        # number of options and dominated preprocessing on large instances.
        within_group_differences = np.concatenate(
            [
                np.abs(margins[:, None] - margins[None, :]).ravel()
                for margins in self.margins
            ]
        )
        positive = within_group_differences[within_group_differences > TOL]
        cost_scale = float(np.median(positive)) if positive.size else 0.0
        if cost_scale <= TOL:
            positive_deviations = deviation_array[deviation_array > TOL]
            cost_scale = (
                float(np.median(positive_deviations))
                if positive_deviations.size
                else 1.0
            )
        self.lambda_scale = max(1e-6, value_scale / max(cost_scale, 1e-6))
        self.multiplier_grid = np.concatenate(
            [np.array([0.0]), self.lambda_scale * np.geomspace(1e-4, 1e4, 25)]
        )
        self._cacheable_lambdas = {float(value) for value in self.multiplier_grid}
        self._fixed_value_cache: dict[float, np.ndarray] = {}

        # For every threshold, r_ij(theta) lies between margin-deviation and
        # margin. Thus each threshold Lagrangian function, and their pointwise
        # maximum, has a finite explicit Lipschitz constant in lambda.
        self._group_slope_bounds = np.asarray(
            [
                float(
                    np.max(
                        np.maximum(
                            np.abs(margins),
                            np.abs(margins - deviations),
                        )
                    )
                )
                for margins, deviations in zip(self.margins, self.deviations)
            ],
            dtype=float,
        )

        zero_values = [np.zeros_like(values) for values in self.values]
        self.capacities = self._envelope_values(1.0, zero_values)
        self.preprocessing_seconds = time.perf_counter() - start
        self.profile = OracleProfile(
            groups=instance.n_items,
            options=sum(len(group) for group in instance.items),
            thresholds=len(self.thetas),
            local_segments=local_segments,
            preprocessing_seconds=self.preprocessing_seconds,
        )

    def _envelope_values(
        self,
        lambda_value: float,
        objective_values: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        lam = max(0.0, float(lambda_value))
        values_by_group = self.values if objective_values is None else objective_values
        if lam == 0.0:
            constant = sum(float(np.max(values)) for values in values_by_group)
            return np.full(len(self.thetas), constant, dtype=float)

        zero_objective = objective_values is not None
        base = (
            np.where(self._sorted_eligible, 0.0, -np.inf)
            if zero_objective
            else self._sorted_values
        )
        saturated_scores = base + lam * self._sorted_margins
        prefix_max = np.maximum.accumulate(saturated_scores, axis=1)
        active_scores = base + lam * (
            self._sorted_margins - self._sorted_deviations
        )
        suffix_max = np.maximum.accumulate(active_scores[:, ::-1], axis=1)[:, ::-1]
        constants = np.full(len(self._segment_lo), -np.inf, dtype=float)
        has_constant = self._segment_last_saturated >= 0
        constants[has_constant] = prefix_max[
            self._segment_group[has_constant],
            self._segment_last_saturated[has_constant],
        ]
        first_active = self._segment_last_saturated + 1
        has_line = first_active < self._group_sizes[self._segment_group]
        line_intercepts = np.full(len(self._segment_lo), -np.inf, dtype=float)
        line_intercepts[has_line] = suffix_max[
            self._segment_group[has_line], first_active[has_line]
        ]
        finite_constant = np.isfinite(constants)
        finite_line = np.isfinite(line_intercepts)
        first_line = self._segment_lo.copy()
        both = finite_constant & finite_line
        first_line[both] = np.searchsorted(
            self.thetas,
            (constants[both] - line_intercepts[both]) / lam,
            side="left",
        )
        first_line[finite_constant & ~finite_line] = self._segment_hi[
            finite_constant & ~finite_line
        ] + 1
        first_line = np.minimum(
            np.maximum(first_line, self._segment_lo), self._segment_hi + 1
        )

        intercept_diff = np.zeros(len(self.thetas) + 1, dtype=float)
        slope_diff = np.zeros(len(self.thetas) + 1, dtype=float)
        constant_piece = finite_constant & (self._segment_lo < first_line)
        np.add.at(
            intercept_diff,
            self._segment_lo[constant_piece],
            constants[constant_piece],
        )
        np.add.at(
            intercept_diff,
            first_line[constant_piece],
            -constants[constant_piece],
        )
        line_piece = finite_line & (first_line <= self._segment_hi)
        np.add.at(
            intercept_diff,
            first_line[line_piece],
            line_intercepts[line_piece],
        )
        np.add.at(
            intercept_diff,
            self._segment_hi[line_piece] + 1,
            -line_intercepts[line_piece],
        )
        np.add.at(slope_diff, first_line[line_piece], lam)
        np.add.at(slope_diff, self._segment_hi[line_piece] + 1, -lam)
        intercept = np.cumsum(intercept_diff[:-1])
        slope = np.cumsum(slope_diff[:-1])
        return intercept + slope * self.thetas - lam * float(self.instance.gamma) * self.thetas

    def values_at_lambda(self, lambda_value: float, lo: int, hi: int) -> np.ndarray:
        lam = max(0.0, float(lambda_value))
        if lam in self._cacheable_lambdas:
            if lam not in self._fixed_value_cache:
                full = self._envelope_values(lam)
                full[self.capacities < -TOL] = -np.inf
                self._fixed_value_cache[lam] = full
            return self._fixed_value_cache[lam][lo : hi + 1].copy()
        result = self._envelope_values(lam)[lo : hi + 1].copy()
        result[self.capacities[lo : hi + 1] < -TOL] = -np.inf
        return result

    def _interval_lipschitz(self, lo: int, hi: int) -> float:
        """Return a valid Lipschitz constant for the minimax objective."""

        maximum_theta = float(np.max(np.abs(self.thetas[lo : hi + 1])))
        return float(
            np.sum(self._group_slope_bounds)
            + abs(float(self.instance.gamma)) * maximum_theta
        )

    def _grid_bound(self, lo: int, hi: int) -> IntervalBound:
        """Legacy fixed-grid bound retained solely for kernel ablations."""

        start = time.perf_counter()
        evaluations = [
            (
                float(lambda_value),
                float(
                    np.max(
                        self.values_at_lambda(float(lambda_value), lo, hi)
                    )
                ),
            )
            for lambda_value in self.multiplier_grid
        ]
        finite = [(lam, value) for lam, value in evaluations if math.isfinite(value)]
        if not finite:
            return IntervalBound(
                float("-inf"),
                0.0,
                len(evaluations),
                time.perf_counter() - start,
                lower_bound=float("-inf"),
                optimality_gap=0.0,
                certified=True,
            )
        best_lambda, best_value = min(finite, key=lambda pair: pair[1])
        return IntervalBound(
            float(np.nextafter(best_value, math.inf)),
            float(best_lambda),
            len(evaluations),
            time.perf_counter() - start,
        )

    def bound(
        self,
        lo: int,
        hi: int,
        local_search: bool = True,
        *,
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-9,
        max_evaluations: int = 256,
    ) -> IntervalBound:
        """Return a valid and tolerance-certified minimax interval bound.

        The default method returns an explicitly evaluated Lagrangian upper
        bound together with a lower certificate. In real arithmetic its gap is
        at most ``absolute_tolerance + relative_tolerance * max(1, |UB|)``.
        ``local_search=False`` selects the historical fixed-grid ablation,
        which remains a valid upper bound but has no minimization certificate.
        """

        if lo < 0 or hi >= len(self.thetas) or lo > hi:
            raise ValueError("invalid threshold interval")
        if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
            raise ValueError("oracle tolerances must be nonnegative")
        if max_evaluations < 8:
            raise ValueError("max_evaluations must be at least eight")
        if not local_search:
            return self._grid_bound(lo, hi)

        start = time.perf_counter()
        evaluations: list[tuple[float, float]] = []

        def evaluate(lambda_value: float) -> float:
            lam = max(0.0, float(lambda_value))
            value = float(np.max(self.values_at_lambda(lam, lo, hi)))
            evaluations.append((lam, value))
            return value

        feasible = self.capacities[lo : hi + 1] >= -TOL
        if not bool(np.any(feasible)):
            return IntervalBound(
                float("-inf"),
                0.0,
                0,
                time.perf_counter() - start,
                lower_bound=float("-inf"),
                optimality_gap=0.0,
                certified=True,
            )

        # Convex bracketing. If f(b) >= f(a) for 0 <= a < b, convexity
        # implies that some global minimizer lies in [0,b]. Feasibility of a
        # fixed-threshold LP gives a nonnegative asymptotic slope, so geometric
        # expansion reaches such a pair after finitely many breakpoints.
        zero_value = evaluate(0.0)
        previous_value = zero_value
        upper_lambda = max(float(self.lambda_scale), 1e-12)
        upper_value = evaluate(upper_lambda)
        while upper_value < previous_value and len(evaluations) < max_evaluations - 4:
            previous_value = upper_value
            upper_lambda *= 2.0
            if not math.isfinite(upper_lambda):
                raise RuntimeError("failed to bracket minimax multiplier")
            upper_value = evaluate(upper_lambda)
        if upper_value < previous_value:
            raise RuntimeError(
                "minimax bracketing exhausted max_evaluations before certification"
            )

        lower_lambda = 0.0
        lipschitz = self._interval_lipschitz(lo, hi)
        if lipschitz == 0.0:
            best_lambda, best_value = min(evaluations, key=lambda pair: pair[1])
            padded = float(np.nextafter(best_value, math.inf))
            lower = float(np.nextafter(best_value, -math.inf))
            return IntervalBound(
                padded,
                float(best_lambda),
                len(evaluations),
                time.perf_counter() - start,
                lower_bound=lower,
                optimality_gap=float(padded - lower),
                certified=True,
            )

        # Golden-section contraction preserves at least one minimizer in the
        # closed bracket. The stopping rule certifies objective error, unlike a
        # generic scalar optimizer's step-size termination test.
        inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
        left_point = upper_lambda - inverse_phi * (upper_lambda - lower_lambda)
        right_point = lower_lambda + inverse_phi * (upper_lambda - lower_lambda)
        left_value = evaluate(left_point)
        right_value = evaluate(right_point)
        while len(evaluations) < max_evaluations:
            finite = [pair for pair in evaluations if math.isfinite(pair[1])]
            _best_lambda, best_value = min(finite, key=lambda pair: pair[1])
            target_gap = absolute_tolerance + relative_tolerance * max(
                1.0, abs(best_value)
            )
            certified_gap = lipschitz * (upper_lambda - lower_lambda)
            if certified_gap <= target_gap:
                break
            if left_value <= right_value:
                upper_lambda = right_point
                right_point = left_point
                right_value = left_value
                left_point = upper_lambda - inverse_phi * (
                    upper_lambda - lower_lambda
                )
                if left_point <= lower_lambda or left_point >= upper_lambda:
                    break
                left_value = evaluate(left_point)
            else:
                lower_lambda = left_point
                left_point = right_point
                left_value = right_value
                right_point = lower_lambda + inverse_phi * (
                    upper_lambda - lower_lambda
                )
                if right_point <= lower_lambda or right_point >= upper_lambda:
                    break
                right_value = evaluate(right_point)

        finite = [pair for pair in evaluations if math.isfinite(pair[1])]
        in_bracket = [
            pair
            for pair in finite
            if lower_lambda <= pair[0] <= upper_lambda
        ]
        best_lambda, best_value = min(in_bracket, key=lambda pair: pair[1])
        certified_gap = lipschitz * (upper_lambda - lower_lambda)
        target_gap = absolute_tolerance + relative_tolerance * max(
            1.0, abs(best_value)
        )
        if certified_gap > target_gap:
            raise RuntimeError(
                "minimax contraction exhausted max_evaluations before certification: "
                f"gap={certified_gap:.6g}, target={target_gap:.6g}"
            )

        # In binary64, move the evaluated value one representable number upward
        # and the real-arithmetic lower certificate one number downward so
        # storage alone cannot reverse either enclosure direction.
        # This is a solver-tolerance certificate, not interval arithmetic.
        padded = float(np.nextafter(best_value, math.inf))
        lower = float(np.nextafter(best_value - certified_gap, -math.inf))
        return IntervalBound(
            padded,
            float(best_lambda),
            len(evaluations),
            time.perf_counter() - start,
            lower_bound=lower,
            optimality_gap=float(padded - lower),
            certified=True,
        )


__all__ = ["CompressedThetaIntervalOracle", "OracleProfile"]
