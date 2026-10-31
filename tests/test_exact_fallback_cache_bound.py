from __future__ import annotations

from fractions import Fraction

import pytest

from robust_mckp import Option, PricingInstance
from research.compressed_interval_oracle import CompressedThetaIntervalOracle


def _direct_exact_value(instance: PricingInstance, theta: float, lam: float) -> Fraction:
    t = Fraction.from_float(theta)
    multiplier = Fraction.from_float(lam)
    total = -instance.gamma * multiplier * t
    for group in instance.items:
        total += max(
            Fraction.from_float(option.value)
            + multiplier
            * (
                Fraction.from_float(option.margin)
                - max(Fraction.from_float(abs(option.uncertainty)) - t, Fraction(0))
            )
            for option in group
        )
    return total


@pytest.mark.parametrize("sizes", [(12, 12, 12), (7, 11, 15)])
def test_exact_slope_cache_stays_bounded_and_eviction_preserves_values(
    sizes: tuple[int, ...],
) -> None:
    groups = []
    position = 0
    for size in sizes:
        group = []
        for _ in range(size):
            position += 1
            group.append(
                Option(
                    value=(position % 7) * 0.5 - 1.25,
                    margin=(position % 5) * 0.75 - 2.0,
                    uncertainty=position * 0.25,
                )
            )
        groups.append(group)
    instance = PricingInstance(items=groups, gamma=1)
    oracle = CompressedThetaIntervalOracle(instance)
    rectangular = len(set(sizes)) == 1
    assert oracle._use_exact_rectangular_path is rectangular
    cache = (
        oracle._exact_slope_matrix_cache
        if rectangular
        else oracle._exact_slope_cache
    )
    inactive = (
        oracle._exact_slope_cache
        if rectangular
        else oracle._exact_slope_matrix_cache
    )
    assert len(oracle.thetas) > 8

    for index, theta in enumerate(oracle.thetas):
        assert oracle._exact_threshold_value(0.375, index) == _direct_exact_value(
            instance, float(theta), 0.375
        )
        assert len(cache) <= 8
        assert not inactive
        retained_coefficients = sum(
            int(value.size) if rectangular else sum(len(group) for group in value)
            for value in cache.values()
        )
        assert retained_coefficients == len(cache) * instance.n_options
        assert retained_coefficients <= 8 * instance.n_options

    assert list(cache) == list(range(len(oracle.thetas) - 8, len(oracle.thetas)))
    assert 0 not in cache
    previous_keys = list(cache)
    # A new multiplier avoids the separate exact-value memoization, forcing
    # reconstruction of the evicted first slope array.
    assert oracle._exact_threshold_value(1.375, 0) == _direct_exact_value(
        instance, float(oracle.thetas[0]), 1.375
    )
    assert list(cache) == previous_keys[1:] + [0]
    # A slope-cache hit at another new multiplier leaves FIFO order intact.
    assert oracle._exact_threshold_value(3.125, 0) == _direct_exact_value(
        instance, float(oracle.thetas[0]), 3.125
    )
    assert list(cache) == previous_keys[1:] + [0]
