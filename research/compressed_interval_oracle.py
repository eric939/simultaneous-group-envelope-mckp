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
from fractions import Fraction
import sys
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from robust_mckp import PricingInstance
from robust_mckp.exact_bnb import (
    _require_representable_objective_range,
    build_full_theta_candidates,
)
from research.novelty_go_no_go import IntervalBound


TOL = 1e-9


def _half_ulp(value: float) -> float:
    if not math.isfinite(value):
        return math.inf
    return 0.5 * math.ulp(float(value))


def _half_ulp_array(values: np.ndarray) -> np.ndarray:
    """Vectorized half-ULP radii for finite binary64 arrays."""

    magnitudes = np.abs(np.asarray(values, dtype=float))
    return 0.5 * (np.nextafter(magnitudes, math.inf) - magnitudes)


def _binary_denominator_exponent(value: float) -> int:
    """Return q for the exact binary rational ``value = p / 2**q``."""

    _numerator, denominator = float(value).as_integer_ratio()
    return denominator.bit_length() - 1


def _binary_integer_at_exponent(value: float, exponent: int) -> int:
    """Scale a finite binary64 value exactly to denominator ``2**exponent``."""

    numerator, denominator = float(value).as_integer_ratio()
    local_exponent = denominator.bit_length() - 1
    return numerator << (exponent - local_exponent)


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
        _require_representable_objective_range(instance)
        self._objective_range_validated = True
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
        value_exponent = max(
            _binary_denominator_exponent(float(value))
            for values in self.values
            for value in values
        )
        coefficient_exponent = max(
            [
                *(
                    _binary_denominator_exponent(float(value))
                    for values in self.margins
                    for value in values
                ),
                *(
                    _binary_denominator_exponent(float(value))
                    for values in self.deviations
                    for value in values
                ),
                *(
                    _binary_denominator_exponent(float(value))
                    for value in self.thetas
                ),
            ]
        )
        self._exact_value_exponent = value_exponent
        self._exact_coefficient_exponent = coefficient_exponent
        self._exact_value_integers = [
            [
                _binary_integer_at_exponent(float(value), value_exponent)
                for value in values
            ]
            for values in self.values
        ]
        self._exact_margin_integers = [
            [
                _binary_integer_at_exponent(float(value), coefficient_exponent)
                for value in values
            ]
            for values in self.margins
        ]
        self._exact_deviation_integers = [
            [
                _binary_integer_at_exponent(float(value), coefficient_exponent)
                for value in values
            ]
            for values in self.deviations
        ]
        self._exact_theta_integers = [
            _binary_integer_at_exponent(float(value), coefficient_exponent)
            for value in self.thetas
        ]
        self._exact_slope_cache: dict[int, list[list[int]]] = {}
        self._exact_slope_matrix_cache: dict[int, np.ndarray] = {}
        self._exact_threshold_value_cache: dict[tuple[float, int], Fraction] = {}
        self._sorted_values: list[np.ndarray] = []
        self._sorted_margins: list[np.ndarray] = []
        self._sorted_deviations: list[np.ndarray] = []
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
            self._sorted_values.append(sorted_values)
            self._sorted_margins.append(sorted_margins)
            self._sorted_deviations.append(sorted_deviations)
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
        option_count = int(np.sum(self._group_sizes))
        maximum_menu = int(np.max(self._group_sizes))
        padded_count = instance.n_items * maximum_menu
        # A rectangular kernel removes Python-level per-group query overhead
        # when padding remains within a constant factor of the ragged input.
        # The guard preserves O(K) work and storage for highly unequal menus.
        self._use_padded_fast_path = padded_count <= 2 * option_count
        self._use_exact_rectangular_path = bool(
            np.all(self._group_sizes == maximum_menu)
        )
        if self._use_exact_rectangular_path:
            self._exact_value_matrix = np.asarray(
                self._exact_value_integers, dtype=object
            )
            self._exact_margin_matrix = np.asarray(
                self._exact_margin_integers, dtype=object
            )
            self._exact_deviation_matrix = np.asarray(
                self._exact_deviation_integers, dtype=object
            )
        if self._use_padded_fast_path:
            shape = (instance.n_items, maximum_menu)
            self._padded_valid = np.zeros(shape, dtype=bool)
            self._padded_values = np.full(shape, -math.inf, dtype=float)
            self._padded_margins = np.zeros(shape, dtype=float)
            self._padded_deviations = np.zeros(shape, dtype=float)
            for group_index, (values, margins, deviations) in enumerate(
                zip(
                    self._sorted_values,
                    self._sorted_margins,
                    self._sorted_deviations,
                )
            ):
                size = len(values)
                self._padded_valid[group_index, :size] = True
                self._padded_values[group_index, :size] = values
                self._padded_margins[group_index, :size] = margins
                self._padded_deviations[group_index, :size] = deviations
            self._padded_robust_margins = (
                self._padded_margins - self._padded_deviations
            )
            unit_roundoff = 0.5 * sys.float_info.epsilon
            rounding_factor = float(
                np.nextafter(unit_roundoff / (1.0 - unit_roundoff), math.inf)
            )
            minimum_subnormal = float(np.nextafter(0.0, 1.0))
            self._padded_robust_errors = np.where(
                self._padded_valid,
                rounding_factor
                * (np.abs(self._padded_margins) + np.abs(self._padded_deviations))
                + minimum_subnormal,
                0.0,
            )

        value_array = np.concatenate(self.values)
        deviation_array = np.concatenate(self.deviations)
        value_range = float(np.ptp(value_array))
        value_scale = max(
            1.0,
            value_range if math.isfinite(value_range) else 1.0,
            float(np.max(np.abs(value_array))),
        )
        # Only within-menu tradeoffs matter for a group-separable multiplier.
        # Forming all global pairwise differences would be quadratic in the
        # number of options and dominated preprocessing on large instances.
        within_group_differences = np.concatenate(
            [
                np.diff(np.unique(np.sort(margins)))
                for margins in self.margins
                if len(margins) > 1
            ]
        ) if any(len(margins) > 1 for margins in self.margins) else np.array([], dtype=float)
        positive = within_group_differences[within_group_differences > TOL]
        cost_scale = float(np.median(positive)) if positive.size else 0.0
        if cost_scale <= TOL:
            positive_deviations = deviation_array[deviation_array > TOL]
            cost_scale = (
                float(np.median(positive_deviations))
                if positive_deviations.size
                else 1.0
            )
        self.lambda_scale = min(
            float(np.finfo(float).max) / 1e4,
            max(1e-6, value_scale / max(cost_scale, 1e-6)),
        )
        self.multiplier_grid = np.concatenate(
            [np.array([0.0]), self.lambda_scale * np.geomspace(1e-4, 1e4, 25)]
        )
        self._cacheable_lambdas = {float(value) for value in self.multiplier_grid}
        self._fixed_value_cache: dict[float, np.ndarray] = {}
        self._certified_vector_cache: dict[
            float, tuple[np.ndarray, np.ndarray]
        ] = {}

        # For every threshold, r_ij(theta) lies between margin-deviation and
        # margin. Thus each threshold Lagrangian function, and their pointwise
        # maximum, has a finite explicit Lipschitz constant in lambda.
        group_slope_bounds: list[float] = []
        coefficient_denominator = 1 << self._exact_coefficient_exponent
        for margins, deviations in zip(
            self._exact_margin_integers,
            self._exact_deviation_integers,
        ):
            exact_bound_integer = max(
                max(abs(margin), abs(margin - deviation))
                for margin, deviation in zip(margins, deviations)
            )
            exact_bound = Fraction(
                exact_bound_integer, coefficient_denominator
            )
            try:
                rounded = float(exact_bound)
            except OverflowError:
                rounded = math.inf
            if math.isfinite(rounded) and Fraction.from_float(rounded) < exact_bound:
                rounded = float(np.nextafter(rounded, math.inf))
            group_slope_bounds.append(rounded)
        self._group_slope_bounds = np.asarray(group_slope_bounds, dtype=float)

        zero_values = [np.zeros_like(values) for values in self.values]
        self.capacities, capacity_errors = self._envelope_values_with_error(
            1.0, zero_values
        )
        lower = self.capacities - capacity_errors
        upper = self.capacities + capacity_errors
        self.feasible_thresholds = lower >= 0.0
        ambiguous = np.flatnonzero((lower < 0.0) & (upper >= 0.0))
        for index in ambiguous:
            exact_capacity = self._exact_capacity(int(index))
            self.capacities[int(index)] = float(exact_capacity)
            self.feasible_thresholds[int(index)] = exact_capacity >= 0
        self.preprocessing_seconds = time.perf_counter() - start
        self.profile = OracleProfile(
            groups=instance.n_items,
            options=sum(len(group) for group in instance.items),
            thresholds=len(self.thetas),
            local_segments=local_segments,
            preprocessing_seconds=self.preprocessing_seconds,
        )

    def _envelope_values_with_error(
        self,
        lambda_value: float,
        objective_values: Sequence[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return binary64 envelope estimates and rigorous rounding radii.

        Range-difference events use vectorized prefix sums with a conservative
        forward-error enclosure.  The returned radius also covers the
        preceding multiplications, additions, maximum comparisons, and the
        final threshold term.  Ambiguous cases are resolved by exact rational
        evaluation in the caller.
        """

        lam = max(0.0, float(lambda_value))
        values_by_group = self.values if objective_values is None else objective_values
        if lam == 0.0:
            terms = [float(np.max(values)) for values in values_by_group]
            try:
                constant = math.fsum(terms)
            except OverflowError:
                return (
                    np.full(len(self.thetas), math.inf, dtype=float),
                    np.full(len(self.thetas), math.inf, dtype=float),
                )
            radius = math.fsum(_half_ulp(term) for term in terms) + _half_ulp(constant)
            return (
                np.full(len(self.thetas), constant, dtype=float),
                np.full(len(self.thetas), radius, dtype=float),
            )

        zero_objective = objective_values is not None
        if self._use_padded_fast_path:
            valid = self._padded_valid
            if zero_objective:
                base = np.where(valid, 0.0, -math.inf)
            elif objective_values is None:
                base = self._padded_values
            else:
                base = np.full_like(self._padded_values, -math.inf)
                for group_index, values in enumerate(objective_values):
                    base[group_index, : len(values)] = values
            margins = self._padded_margins
            margin_products = lam * margins
            saturated_scores = base + margin_products
            robust_margins = self._padded_robust_margins
            active_products = lam * robust_margins
            active_scores = base + active_products
            prefix_matrix = np.maximum.accumulate(saturated_scores, axis=1)
            suffix_matrix = np.maximum.accumulate(
                active_scores[:, ::-1], axis=1
            )[:, ::-1]
            finite_saturated = np.where(valid, saturated_scores, 0.0)
            finite_active = np.where(valid, active_scores, 0.0)
            unit_roundoff = 0.5 * sys.float_info.epsilon
            rounding_factor = float(
                np.nextafter(unit_roundoff / (1.0 - unit_roundoff), math.inf)
            )
            minimum_subnormal = float(np.nextafter(0.0, 1.0))
            saturated_errors = np.where(
                valid,
                rounding_factor
                * (np.abs(margin_products) + np.abs(finite_saturated))
                + 2.0 * minimum_subnormal,
                0.0,
            )
            robust_errors = self._padded_robust_errors
            active_errors = np.where(
                valid,
                abs(lam) * robust_errors
                + rounding_factor
                * (np.abs(active_products) + np.abs(finite_active))
                + 2.0 * minimum_subnormal,
                0.0,
            )
            group_errors = np.max(
                np.maximum(saturated_errors, active_errors), axis=1
            )
            prefix_by_group = None
            suffix_by_group = None
        else:
            prefix_by_group = []
            suffix_by_group = []
            group_errors = []
            for sorted_values, margins, deviations in zip(
                self._sorted_values,
                self._sorted_margins,
                self._sorted_deviations,
            ):
                base = (
                    np.zeros_like(sorted_values)
                    if zero_objective
                    else sorted_values
                )
                margin_products = lam * margins
                saturated_scores = base + margin_products
                robust_margins = margins - deviations
                active_products = lam * robust_margins
                active_scores = base + active_products
                prefix_by_group.append(np.maximum.accumulate(saturated_scores))
                suffix_by_group.append(
                    np.maximum.accumulate(active_scores[::-1])[::-1]
                )
                saturated_errors = (
                    _half_ulp_array(margin_products)
                    + _half_ulp_array(saturated_scores)
                )
                robust_errors = _half_ulp_array(robust_margins)
                active_errors = (
                    abs(lam) * robust_errors
                    + _half_ulp_array(active_products)
                    + _half_ulp_array(active_scores)
                )
                candidate_errors = np.maximum(saturated_errors, active_errors)
                group_errors.append(
                    float(np.max(candidate_errors, initial=0.0))
                )
        group_error = float(np.nextafter(math.fsum(group_errors), math.inf))

        constants = np.full(len(self._segment_lo), -np.inf, dtype=float)
        has_constant = self._segment_last_saturated >= 0
        if self._use_padded_fast_path:
            segments = np.flatnonzero(has_constant)
            constants[segments] = prefix_matrix[
                self._segment_group[segments],
                self._segment_last_saturated[segments],
            ]
        else:
            for segment in np.flatnonzero(has_constant):
                constants[segment] = prefix_by_group[
                    int(self._segment_group[segment])
                ][int(self._segment_last_saturated[segment])]
        first_active = self._segment_last_saturated + 1
        has_line = first_active < self._group_sizes[self._segment_group]
        line_intercepts = np.full(len(self._segment_lo), -np.inf, dtype=float)
        if self._use_padded_fast_path:
            segments = np.flatnonzero(has_line)
            line_intercepts[segments] = suffix_matrix[
                self._segment_group[segments], first_active[segments]
            ]
        else:
            for segment in np.flatnonzero(has_line):
                line_intercepts[segment] = suffix_by_group[
                    int(self._segment_group[segment])
                ][int(first_active[segment])]
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

        event_size = len(self.thetas) + 1
        constant_piece = finite_constant & (self._segment_lo < first_line)
        constant_segments = np.flatnonzero(constant_piece)
        constant_begin = self._segment_lo[constant_segments]
        constant_end = first_line[constant_segments]
        constant_values = constants[constant_segments]
        line_piece = finite_line & (first_line <= self._segment_hi)
        line_segments = np.flatnonzero(line_piece)
        line_begin = first_line[line_segments]
        line_end = self._segment_hi[line_segments] + 1
        line_values = line_intercepts[line_segments]
        intercept_indices = np.concatenate(
            [constant_begin, constant_end, line_begin, line_end]
        )
        intercept_values = np.concatenate(
            [constant_values, -constant_values, line_values, -line_values]
        )
        intercept_diff = np.bincount(
            intercept_indices,
            weights=intercept_values,
            minlength=event_size,
        )
        intercept_abs = np.bincount(
            intercept_indices,
            weights=np.abs(intercept_values),
            minlength=event_size,
        )
        intercept_count = np.bincount(
            intercept_indices, minlength=event_size
        ).astype(np.int64, copy=False)
        slope_indices = np.concatenate([line_begin, line_end])
        slope_values = np.concatenate(
            [np.full(len(line_begin), lam), np.full(len(line_end), -lam)]
        )
        slope_diff = np.bincount(
            slope_indices, weights=slope_values, minlength=event_size
        )
        slope_abs = np.bincount(
            slope_indices, weights=np.abs(slope_values), minlength=event_size
        )
        slope_count = np.bincount(
            slope_indices, minlength=event_size
        ).astype(np.int64, copy=False)

        intercept = np.cumsum(intercept_diff[:-1])
        slope = np.cumsum(slope_diff[:-1])
        theta_values = self.thetas
        slope_terms = slope * theta_values
        gamma_multiplier = lam * float(self.instance.gamma)
        gamma_terms = gamma_multiplier * theta_values
        estimates = intercept + slope_terms - gamma_terms

        # A standard forward-error bound for the binned range additions and
        # prefix sums.  It is deliberately conservative; ambiguity triggers
        # the exact rational fallbacks used by ``bound`` and feasibility setup.
        epsilon = sys.float_info.epsilon
        intercept_operations = np.cumsum(intercept_count[:-1]) + np.arange(
            1, len(self.thetas) + 1, dtype=np.int64
        )
        slope_operations = np.cumsum(slope_count[:-1]) + np.arange(
            1, len(self.thetas) + 1, dtype=np.int64
        )
        intercept_gamma = (
            intercept_operations * epsilon
            / np.maximum(1.0 - intercept_operations * epsilon, epsilon)
        )
        slope_gamma = (
            slope_operations * epsilon
            / np.maximum(1.0 - slope_operations * epsilon, epsilon)
        )
        intercept_rounding = intercept_gamma * np.cumsum(intercept_abs[:-1])
        slope_rounding = slope_gamma * np.cumsum(slope_abs[:-1])
        arithmetic_scale = np.abs(intercept) + np.abs(slope_terms) + np.abs(gamma_terms)
        errors = (
            group_error
            + intercept_rounding
            + np.abs(theta_values) * slope_rounding
            + np.abs(theta_values) * _half_ulp(gamma_multiplier)
            + _half_ulp_array(slope_terms)
            + _half_ulp_array(gamma_terms)
            + _half_ulp_array(estimates)
            + 4.0 * epsilon * arithmetic_scale
        )
        errors = np.nextafter(errors, math.inf)
        return estimates, errors

    def _envelope_values(
        self,
        lambda_value: float,
        objective_values: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        estimates, _errors = self._envelope_values_with_error(
            lambda_value, objective_values
        )
        return estimates

    def _certified_vectors_at_lambda(
        self, lambda_value: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cache full certified vectors shared by overlapping intervals."""

        lam = max(0.0, float(lambda_value))
        cached = self._certified_vector_cache.get(lam)
        if cached is not None:
            return cached
        result = self._envelope_values_with_error(lam)
        if len(self._certified_vector_cache) >= 512:
            self._certified_vector_cache.pop(next(iter(self._certified_vector_cache)))
        self._certified_vector_cache[lam] = result
        return result

    def _exact_capacity(self, index: int) -> Fraction:
        theta = Fraction.from_float(float(self.thetas[index]))
        total = -int(self.instance.gamma) * theta
        for margins, deviations in zip(self.margins, self.deviations):
            best: Fraction | None = None
            for margin, deviation in zip(margins, deviations):
                exact_margin = Fraction.from_float(float(margin))
                exact_deviation = Fraction.from_float(float(deviation))
                slack = exact_margin - max(exact_deviation - theta, Fraction(0))
                best = slack if best is None or slack > best else best
            if best is None:
                raise ValueError("every exactly-one group must contain an option")
            total += best
        return total

    def _exact_threshold_value(
        self, lambda_value: float, index: int
    ) -> Fraction:
        lam = max(0.0, float(lambda_value))
        cache_key = (lam, int(index))
        cached = self._exact_threshold_value_cache.get(cache_key)
        if cached is not None:
            return cached
        lambda_numerator, lambda_denominator = lam.as_integer_ratio()
        lambda_exponent = lambda_denominator.bit_length() - 1
        product_exponent = self._exact_coefficient_exponent + lambda_exponent
        common_exponent = max(self._exact_value_exponent, product_exponent)
        value_shift = common_exponent - self._exact_value_exponent
        product_shift = common_exponent - product_exponent

        if self._use_exact_rectangular_path:
            slopes_matrix = self._exact_slope_matrix_cache.get(index)
            if slopes_matrix is None:
                theta = self._exact_theta_integers[index]
                slopes_matrix = self._exact_margin_matrix - np.maximum(
                    self._exact_deviation_matrix - theta, 0
                )
                self._exact_slope_matrix_cache[index] = slopes_matrix
            value_multiplier = 1 << value_shift
            product_multiplier = lambda_numerator << product_shift
            scores = (
                self._exact_value_matrix * value_multiplier
                + slopes_matrix * product_multiplier
            )
            total = sum(np.max(scores, axis=1).tolist())
            total -= (
                int(self.instance.gamma)
                * self._exact_theta_integers[index]
                * product_multiplier
            )
            result = Fraction(total, 1 << common_exponent)
            if len(self._exact_threshold_value_cache) >= 2048:
                self._exact_threshold_value_cache.pop(
                    next(iter(self._exact_threshold_value_cache))
                )
            self._exact_threshold_value_cache[cache_key] = result
            return result

        slopes = self._exact_slope_cache.get(index)
        if slopes is None:
            theta = self._exact_theta_integers[index]
            slopes = [
                [
                    margin - max(deviation - theta, 0)
                    for margin, deviation in zip(margins, deviations)
                ]
                for margins, deviations in zip(
                    self._exact_margin_integers,
                    self._exact_deviation_integers,
                )
            ]
            self._exact_slope_cache[index] = slopes

        total = (
            -int(self.instance.gamma)
            * self._exact_theta_integers[index]
            * lambda_numerator
            << product_shift
        )
        for values, group_slopes in zip(self._exact_value_integers, slopes):
            total += max(
                (value << value_shift)
                + ((lambda_numerator * slope) << product_shift)
                for value, slope in zip(values, group_slopes)
            )
        result = Fraction(total, 1 << common_exponent)
        if len(self._exact_threshold_value_cache) >= 2048:
            self._exact_threshold_value_cache.pop(
                next(iter(self._exact_threshold_value_cache))
            )
        self._exact_threshold_value_cache[cache_key] = result
        return result

    def _exact_interval_value(
        self,
        lambda_value: float,
        lo: int,
        hi: int,
        candidate_indices: Sequence[int] | None = None,
    ) -> Fraction | None:
        indices = (
            range(lo, hi + 1)
            if candidate_indices is None
            else candidate_indices
        )
        best: Fraction | None = None
        for index in indices:
            index = int(index)
            if index < lo or index > hi or not bool(self.feasible_thresholds[index]):
                continue
            value = self._exact_threshold_value(lambda_value, index)
            best = value if best is None or value > best else best
        return best

    def _exact_singleton_minimax(
        self, index: int
    ) -> tuple[Fraction, Fraction]:
        """Return the exact singleton dual optimum and one minimizer."""

        theta = Fraction.from_float(float(self.thetas[index]))
        group_lines: list[list[tuple[Fraction, Fraction]]] = []
        candidates = {Fraction(0)}
        for values, margins, deviations in zip(
            self.values, self.margins, self.deviations
        ):
            lines: list[tuple[Fraction, Fraction]] = []
            for objective, margin, deviation in zip(values, margins, deviations):
                intercept = Fraction.from_float(float(objective))
                slope = Fraction.from_float(float(margin)) - max(
                    Fraction.from_float(float(deviation)) - theta, Fraction(0)
                )
                lines.append((intercept, slope))
            for left in range(len(lines)):
                for right in range(left + 1, len(lines)):
                    left_intercept, left_slope = lines[left]
                    right_intercept, right_slope = lines[right]
                    if left_slope == right_slope:
                        continue
                    crossing = (right_intercept - left_intercept) / (
                        left_slope - right_slope
                    )
                    if crossing >= 0:
                        candidates.add(crossing)
            group_lines.append(lines)

        best_value: Fraction | None = None
        best_lambda = Fraction(0)
        for lam in candidates:
            value = -lam * int(self.instance.gamma) * theta
            for lines in group_lines:
                value += max(intercept + lam * slope for intercept, slope in lines)
            if best_value is None or value < best_value:
                best_value = value
                best_lambda = lam
        if best_value is None:
            raise ValueError("singleton interval has no option lines")
        return best_value, best_lambda

    @staticmethod
    def _fraction_enclosure(value: Fraction) -> tuple[float, float]:
        try:
            rounded = float(value)
        except OverflowError:
            if value < 0:
                return (-math.inf, -float(np.finfo(float).max))
            return (float(np.finfo(float).max), math.inf)
        if not math.isfinite(rounded):
            if value < 0:
                return (-math.inf, -float(np.finfo(float).max))
            return (float(np.finfo(float).max), math.inf)
        exact_rounded = Fraction.from_float(rounded)
        if exact_rounded == value:
            return rounded, rounded
        if exact_rounded < value:
            return rounded, float(np.nextafter(rounded, math.inf))
        return float(np.nextafter(rounded, -math.inf)), rounded

    def values_at_lambda(self, lambda_value: float, lo: int, hi: int) -> np.ndarray:
        lam = max(0.0, float(lambda_value))
        if lam in self._cacheable_lambdas:
            if lam not in self._fixed_value_cache:
                full = self._envelope_values(lam)
                full[~self.feasible_thresholds] = -np.inf
                self._fixed_value_cache[lam] = full
            return self._fixed_value_cache[lam][lo : hi + 1].copy()
        result = self._envelope_values(lam)[lo : hi + 1].copy()
        result[~self.feasible_thresholds[lo : hi + 1]] = -np.inf
        return result

    def _interval_lipschitz(self, lo: int, hi: int) -> float:
        """Return a valid Lipschitz constant for the minimax objective."""

        maximum_theta = float(np.max(np.abs(self.thetas[lo : hi + 1])))
        bound = math.fsum(float(value) for value in self._group_slope_bounds)
        bound = math.fsum(
            [bound, abs(float(self.instance.gamma)) * maximum_theta]
        )
        return float(np.nextafter(bound, math.inf))

    def _grid_bound(self, lo: int, hi: int) -> IntervalBound:
        """Legacy fixed-grid bound retained solely for kernel ablations."""

        start = time.perf_counter()
        evaluations = []
        for lambda_value in self.multiplier_grid:
            estimates, errors = self._envelope_values_with_error(float(lambda_value))
            feasible = self.feasible_thresholds[lo : hi + 1]
            if bool(np.any(feasible)):
                value = float(
                    np.max((estimates[lo : hi + 1] + errors[lo : hi + 1])[feasible])
                )
                value = float(np.nextafter(value, math.inf))
            else:
                value = float("-inf")
            evaluations.append((float(lambda_value), value))
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
        """Return a valid minimax interval bound with an explicit gap.

        The default method returns an explicitly evaluated Lagrangian upper
        bound together with a lower certificate. The contraction targets
        ``absolute_tolerance + relative_tolerance * max(1, |UB|)``; if
        binary64 multiplier resolution prevents further contraction, the
        method returns the valid coarser gap instead of failing.
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
        evaluations: list[float] = []
        enclosures: dict[float, tuple[float, float]] = {}
        exact_values: dict[float, Fraction] = {}
        exact_candidate_indices: dict[float, np.ndarray] = {}

        def evaluate(lambda_value: float) -> tuple[float, float]:
            lam = max(0.0, float(lambda_value))
            if lam in enclosures:
                return enclosures[lam]
            estimates, errors = self._certified_vectors_at_lambda(lam)
            estimates = estimates[lo : hi + 1]
            errors = errors[lo : hi + 1]
            feasible_slice = self.feasible_thresholds[lo : hi + 1]
            if not bool(np.any(feasible_slice)):
                enclosure = (float("-inf"), float("-inf"))
                exact_candidate_indices[lam] = np.empty(0, dtype=int)
            elif not bool(
                np.all(np.isfinite(estimates[feasible_slice]))
                and np.all(np.isfinite(errors[feasible_slice]))
            ):
                exact = self._exact_interval_value(lam, lo, hi)
                if exact is None:
                    enclosure = (float("-inf"), float("-inf"))
                else:
                    exact_values[lam] = exact
                    enclosure = self._fraction_enclosure(exact)
                exact_candidate_indices[lam] = (
                    np.flatnonzero(feasible_slice) + lo
                )
            else:
                threshold_lower = estimates - errors
                threshold_upper = estimates + errors
                lower_value = float(np.max(threshold_lower[feasible_slice]))
                upper_value = float(np.max(threshold_upper[feasible_slice]))
                # Only thresholds whose certified upper endpoint reaches the
                # best certified lower endpoint can attain the exact maximum.
                # Retaining this ambiguity set makes exact tie-breaking O(K)
                # per plausible threshold instead of rescanning the full
                # threshold family at every multiplier comparison.
                plausible = feasible_slice & (threshold_upper >= lower_value)
                exact_candidate_indices[lam] = np.flatnonzero(plausible) + lo
                enclosure = (
                    float(np.nextafter(lower_value, -math.inf)),
                    float(np.nextafter(upper_value, math.inf)),
                )
            enclosures[lam] = enclosure
            evaluations.append(lam)
            return enclosure

        def exact_value(lam: float) -> Fraction:
            if lam not in exact_values:
                exact = self._exact_interval_value(
                    lam,
                    lo,
                    hi,
                    exact_candidate_indices.get(lam),
                )
                if exact is None:
                    raise RuntimeError("exact evaluation found no feasible threshold")
                exact_values[lam] = exact
                enclosures[lam] = self._fraction_enclosure(exact)
            return exact_values[lam]

        def compare(left_lambda: float, right_lambda: float) -> int:
            """Compare two values exactly only when float enclosures overlap."""

            left_lower, left_upper = evaluate(left_lambda)
            right_lower, right_upper = evaluate(right_lambda)
            if left_upper < right_lower:
                return -1
            if right_upper < left_lower:
                return 1
            left_exact = exact_value(left_lambda)
            right_exact = exact_value(right_lambda)
            return (left_exact > right_exact) - (left_exact < right_exact)

        feasible = self.feasible_thresholds[lo : hi + 1]
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
        feasible_indices = np.flatnonzero(feasible) + lo
        if len(feasible_indices) == 1:
            exact_value, exact_lambda = self._exact_singleton_minimax(
                int(feasible_indices[0])
            )
            lower, upper = self._fraction_enclosure(exact_value)
            try:
                multiplier = float(exact_lambda)
            except OverflowError:
                multiplier = float(np.finfo(float).max)
            return IntervalBound(
                upper,
                multiplier,
                1,
                time.perf_counter() - start,
                lower_bound=lower,
                optimality_gap=float(upper - lower),
                certified=True,
            )

        # Convex bracketing. If f(b) >= f(a) for 0 <= a < b, convexity
        # implies that some global minimizer lies in [0,b]. Feasibility of a
        # fixed-threshold LP gives a nonnegative asymptotic slope, so geometric
        # expansion reaches such a pair after finitely many breakpoints.
        evaluate(0.0)
        previous_lambda = 0.0
        upper_lambda = max(float(self.lambda_scale), 1e-12)
        evaluate(upper_lambda)
        while compare(upper_lambda, previous_lambda) < 0 and len(evaluations) < max_evaluations - 4:
            previous_lambda = upper_lambda
            upper_lambda *= 2.0
            if not math.isfinite(upper_lambda):
                raise RuntimeError("failed to bracket minimax multiplier")
            evaluate(upper_lambda)
        if compare(upper_lambda, previous_lambda) < 0:
            raise RuntimeError(
                "minimax bracketing exhausted max_evaluations before certification"
            )

        lower_lambda = 0.0
        lipschitz = self._interval_lipschitz(lo, hi)
        if lipschitz == 0.0:
            best_lambda = min(evaluations, key=lambda lam: enclosures[lam][1])
            lower, padded = enclosures[best_lambda]
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
        evaluate(left_point)
        evaluate(right_point)
        while len(evaluations) < max_evaluations:
            finite = [lam for lam in evaluations if math.isfinite(enclosures[lam][1])]
            if not finite:
                return IntervalBound(
                    math.inf,
                    0.0,
                    len(evaluations),
                    time.perf_counter() - start,
                    lower_bound=float(np.finfo(float).max),
                    optimality_gap=math.inf,
                    certified=True,
                )
            _best_lambda = min(finite, key=lambda lam: enclosures[lam][1])
            best_upper = enclosures[_best_lambda][1]
            target_gap = absolute_tolerance + relative_tolerance * max(
                1.0, abs(best_upper)
            )
            point_lower = enclosures[_best_lambda][0]
            certified_gap = best_upper - (
                point_lower - lipschitz * (upper_lambda - lower_lambda)
            )
            if certified_gap <= target_gap:
                break
            if compare(left_point, right_point) <= 0:
                upper_lambda = right_point
                right_point = left_point
                left_point = upper_lambda - inverse_phi * (
                    upper_lambda - lower_lambda
                )
                if left_point <= lower_lambda or left_point >= upper_lambda:
                    break
                evaluate(left_point)
            else:
                lower_lambda = left_point
                left_point = right_point
                right_point = lower_lambda + inverse_phi * (
                    upper_lambda - lower_lambda
                )
                if right_point <= lower_lambda or right_point >= upper_lambda:
                    break
                evaluate(right_point)

        finite = [lam for lam in evaluations if math.isfinite(enclosures[lam][1])]
        in_bracket = [
            lam for lam in finite if lower_lambda <= lam <= upper_lambda
        ]
        best_lambda = min(in_bracket, key=lambda lam: enclosures[lam][1])
        best_lower, best_upper = enclosures[best_lambda]
        lower = float(
            np.nextafter(
                best_lower - lipschitz * (upper_lambda - lower_lambda),
                -math.inf,
            )
        )
        padded = float(np.nextafter(best_upper, math.inf))
        certified_gap = padded - lower
        target_gap = absolute_tolerance + relative_tolerance * max(
            1.0, abs(padded)
        )
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
