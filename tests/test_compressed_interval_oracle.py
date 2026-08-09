from __future__ import annotations

import hashlib

import numpy as np
import pytest

from robust_mckp import Option, PricingInstance
from research.benchmark_instances import build_benchmark_instance
from research.compressed_interval_oracle import CompressedThetaIntervalOracle
from research.interval_baselines import ThetaIntervalOracle, build_small_instance


@pytest.mark.parametrize(
    ("family", "expected_digest"),
    [
        (
            "dense_frontier",
            "ddc193b8b984ad61846fdab0f857b36c58b5db7cd052543a747996fe5f8fd63e",
        ),
        (
            "correlated_risk",
            "c6d6c6b6ea8c41a6fd2b054a21b3a6dd42d788d1a25211e9d6c6144c31684198",
        ),
        (
            "near_tie",
            "8ccc8bed7fd5e93f5881073299db3e0bb7972335567bfaccda62fec2076c1ef8",
        ),
        (
            "many_breakpoints",
            "b7b6e09529ef77f440358058fd6028fb1928b476cb40d543bf63001f09c04085",
        ),
    ],
)
def test_benchmark_generator_preserves_released_coefficients(
    family: str,
    expected_digest: str,
) -> None:
    instance = build_benchmark_instance(family, n=30, m=6, gamma=5, seed=3)
    coefficients = np.asarray(
        [
            (option.value, option.margin, option.uncertainty)
            for group in instance.items
            for option in group
        ],
        dtype=np.float64,
    )
    assert hashlib.sha256(coefficients.tobytes()).hexdigest() == expected_digest


@pytest.mark.parametrize("seed", [0, 1, 2, 17])
def test_compressed_oracle_matches_dense_lagrangian_values(seed: int) -> None:
    instance = build_small_instance(seed=seed, n=6, m=4, gamma=2)
    dense = ThetaIntervalOracle(instance)
    compressed = CompressedThetaIntervalOracle(instance)
    assert compressed.thetas == pytest.approx(dense.thetas, abs=1e-12)
    assert compressed.capacities == pytest.approx(dense.capacities, abs=1e-9)
    for lambda_value in [0.0, 0.01, 0.3, 1.0, 7.5, 100.0]:
        actual = compressed.values_at_lambda(
            lambda_value, 0, len(compressed.thetas) - 1
        )
        expected = dense.values_at_lambda(lambda_value, 0, len(dense.thetas) - 1)
        assert actual == pytest.approx(expected, abs=2e-8, rel=2e-10)


@pytest.mark.parametrize(
    "family", ["dense_frontier", "correlated_risk", "near_tie", "many_breakpoints"]
)
def test_compressed_oracle_matches_dense_bound(family: str) -> None:
    instance = build_benchmark_instance(family, n=30, m=6, gamma=5, seed=3)
    dense = ThetaIntervalOracle(instance)
    compressed = CompressedThetaIntervalOracle(instance)
    intervals = [
        (0, len(dense.thetas) - 1),
        (0, len(dense.thetas) // 2),
        (len(dense.thetas) // 3, len(dense.thetas) - 1),
    ]
    for lo, hi in intervals:
        actual = compressed.bound(lo, hi)
        expected = dense.bound(lo, hi)
        assert actual.upper_bound == pytest.approx(
            expected.upper_bound, abs=5e-6, rel=2e-10
        )


@pytest.mark.parametrize("seed", range(10))
def test_compressed_oracle_matches_dense_on_signed_irregular_menus(seed: int) -> None:
    rng = np.random.default_rng(seed + 9000)
    items = []
    for _ in range(int(rng.integers(3, 9))):
        group = []
        for _ in range(int(rng.integers(2, 8))):
            group.append(
                Option(
                    value=float(rng.normal(0.0, 12.0)),
                    margin=float(rng.normal(0.0, 4.0)),
                    uncertainty=float(rng.choice([0.0, 0.5, 1.0, 1.0, 2.5, 4.0])),
                )
            )
        items.append(group)
    instance = PricingInstance(
        items=items,
        gamma=int(rng.integers(1, len(items) + 1)),
        name=f"signed_irregular_{seed}",
    )
    dense = ThetaIntervalOracle(instance)
    compressed = CompressedThetaIntervalOracle(instance)
    for lambda_value in rng.lognormal(mean=0.0, sigma=2.0, size=12):
        expected = dense.values_at_lambda(
            float(lambda_value), 0, len(dense.thetas) - 1
        )
        actual = compressed.values_at_lambda(
            float(lambda_value), 0, len(compressed.thetas) - 1
        )
        assert actual == pytest.approx(expected, abs=2e-8, rel=2e-10)


def test_compressed_values_match_direct_uncancelled_definition() -> None:
    instance = build_small_instance(seed=515, n=7, m=5, gamma=3)
    oracle = CompressedThetaIntervalOracle(instance)
    for lambda_value in [0.02, 0.7, 4.5, 31.0]:
        direct = []
        for theta in oracle.thetas:
            value = -lambda_value * instance.gamma * theta
            for group in instance.items:
                value += max(
                    option.value
                    + lambda_value
                    * (
                        option.margin
                        - max(0.0, abs(option.uncertainty) - theta)
                    )
                    for option in group
                )
            direct.append(value)
        actual = oracle.values_at_lambda(
            lambda_value, 0, len(oracle.thetas) - 1
        )
        feasible = oracle.capacities >= -1e-9
        assert actual[feasible] == pytest.approx(
            np.asarray(direct)[feasible], abs=2e-8, rel=2e-10
        )


def test_zero_branch_does_not_coalesce_small_positive_multiplier() -> None:
    instance = build_small_instance(seed=616, n=5, m=4, gamma=2)
    oracle = CompressedThetaIntervalOracle(instance)
    lam = 1e-12
    actual = oracle.values_at_lambda(lam, 0, len(oracle.thetas) - 1)
    direct = np.asarray(
        [
            sum(
                max(
                    option.value
                    + lam
                    * (
                        option.margin
                        - max(0.0, abs(option.uncertainty) - float(theta))
                    )
                    for option in group
                )
                for group in instance.items
            )
            - lam * instance.gamma * float(theta)
            for theta in oracle.thetas
        ]
    )
    feasible = np.isfinite(actual)
    assert actual[feasible] == pytest.approx(direct[feasible], abs=1e-12, rel=0.0)


@pytest.mark.parametrize("spacing", [1e-12, 1e-10, 1e-8])
def test_near_repeated_deviations_preserve_endpoint_convention(spacing: float) -> None:
    instance = PricingInstance(
        items=[
            [
                Option(value=1.0, margin=4.0, uncertainty=0.0),
                Option(value=3.0, margin=2.0, uncertainty=1.0),
                Option(value=2.0, margin=3.0, uncertainty=1.0 + spacing),
            ],
            [
                Option(value=0.0, margin=4.0, uncertainty=0.0),
                Option(value=2.0, margin=2.5, uncertainty=1.0),
            ],
        ],
        gamma=1,
        name=f"near_repeated_{spacing}",
    )
    dense = ThetaIntervalOracle(instance)
    compressed = CompressedThetaIntervalOracle(instance)
    assert len(compressed.thetas) == 3
    for lam in (1e-12, 0.3, 7.0):
        expected = dense.values_at_lambda(lam, 0, len(dense.thetas) - 1)
        actual = compressed.values_at_lambda(lam, 0, len(compressed.thetas) - 1)
        assert actual == pytest.approx(expected, abs=2e-11, rel=2e-12)


def test_fixed_multiplier_grid_bound_is_repeatable_across_intervals() -> None:
    instance = build_small_instance(seed=717, n=6, m=4, gamma=2)
    oracle = CompressedThetaIntervalOracle(instance)
    first = oracle.bound(0, len(oracle.thetas) - 1, local_search=False)
    second = oracle.bound(0, len(oracle.thetas) - 1, local_search=False)
    assert first.upper_bound == second.upper_bound
    assert first.lambda_value == second.lambda_value
    assert first.evaluations == len(oracle.multiplier_grid)


def test_exact_capacity_mask_survives_catastrophic_cancellation() -> None:
    instance = PricingInstance(
        items=[
            [Option(0.0, -1e16, 0.0)],
            [Option(0.0, 1.0, 0.0)],
            [Option(0.0, 1e16, 0.0)],
            [Option(0.0, -0.5, 0.0)],
        ],
        gamma=0,
    )
    oracle = CompressedThetaIntervalOracle(instance)
    assert oracle.feasible_thresholds.tolist() == [True]
    assert oracle.capacities.tolist() == [0.5]
    assert oracle.bound(0, 0).upper_bound >= 0.0


def test_singleton_exact_fallback_avoids_multiplier_resolution_floor() -> None:
    instance = PricingInstance(
        items=[
            [
                Option(0.0, 0.0, 0.0),
                Option(
                    137438953472.0,
                    3.5762786865234375e-7,
                    1610612736.0,
                ),
            ]
        ],
        gamma=1,
    )
    oracle = CompressedThetaIntervalOracle(instance)
    result = oracle.bound(0, len(oracle.thetas) - 1)
    assert result.certified
    assert result.upper_bound == pytest.approx(0.0, abs=0.0, rel=0.0)
    assert result.lower_bound == pytest.approx(0.0, abs=0.0, rel=0.0)


def test_overflowing_aggregate_objective_is_rejected_with_rescaling_guidance() -> None:
    maximum = float(np.finfo(float).max)
    instance = PricingInstance(
        items=[
            [Option(maximum, 0.0, 0.0)],
            [Option(maximum, 0.0, 0.0)],
        ],
        gamma=0,
    )
    with pytest.raises(ValueError, match="rescale objective"):
        CompressedThetaIntervalOracle(instance)


def test_negative_overflowing_aggregate_objective_is_rejected() -> None:
    maximum = float(np.finfo(float).max)
    instance = PricingInstance(
        items=[
            [Option(-maximum, 0.0, 0.0)],
            [Option(-maximum, 0.0, 0.0)],
        ],
        gamma=0,
    )
    with pytest.raises(ValueError, match="rescale objective"):
        CompressedThetaIntervalOracle(instance)


def test_certified_bound_encloses_cancelling_group_envelope() -> None:
    instance = PricingInstance(
        items=[
            [Option(100.0, -1.0, 0.0), Option(0.0, 1.0, 0.0)],
            [Option(1.0, 0.0, 0.0)],
            [Option(-0.5, 0.0, 0.0)],
            [Option(-1e16, 0.0, 0.0)],
            [Option(1e16, 0.0, 0.0)],
        ],
        gamma=0,
    )
    result = CompressedThetaIntervalOracle(instance).bound(0, 0)
    assert result.certified
    assert result.lower_bound <= 50.5 <= result.upper_bound


def test_small_negative_capacity_is_not_tolerance_feasible() -> None:
    instance = PricingInstance(
        items=[[Option(1.0, -5e-10, 0.0)]],
        gamma=0,
    )
    result = CompressedThetaIntervalOracle(instance).bound(0, 0)
    assert result.certified
    assert result.upper_bound == float("-inf")


def test_highly_unequal_menus_use_ragged_certified_path() -> None:
    instance = PricingInstance(
        items=[
            [Option(1.0, 2.0, 0.0)],
            [
                Option(
                    value=float(index) / 7.0,
                    margin=3.0 - float(index) / 11.0,
                    uncertainty=float(index % 6) / 5.0,
                )
                for index in range(20)
            ],
            [Option(-0.5, 1.5, 0.25)],
        ],
        gamma=1,
    )
    dense = ThetaIntervalOracle(instance)
    compressed = CompressedThetaIntervalOracle(instance)
    assert not compressed._use_padded_fast_path
    for lambda_value in (0.0, 0.125, 1.75, 19.0):
        actual = compressed.values_at_lambda(
            lambda_value, 0, len(compressed.thetas) - 1
        )
        expected = dense.values_at_lambda(
            lambda_value, 0, len(dense.thetas) - 1
        )
        assert actual == pytest.approx(expected, abs=2e-10, rel=2e-12)
    result = compressed.bound(0, len(compressed.thetas) - 1)
    assert result.certified
