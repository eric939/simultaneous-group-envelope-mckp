from __future__ import annotations

import math

import pytest

from research.bound_dominance import exact_minimax_epigraph_bound
from research.compressed_interval_oracle import CompressedThetaIntervalOracle
from research.novelty_go_no_go import build_small_instance
from research.structural_feasibility_study import bounded_theta_clique_lp


@pytest.mark.parametrize("seed", range(6))
def test_exact_minimax_epigraph_matches_group_envelope_oracle(seed: int) -> None:
    instance = build_small_instance(seed=1000 + seed, n=5, m=4, gamma=2)
    oracle = CompressedThetaIntervalOracle(instance)
    intervals = [
        (0, len(oracle.thetas) - 1),
        (0, len(oracle.thetas) // 2),
        (len(oracle.thetas) // 3, len(oracle.thetas) - 1),
    ]
    for lo, hi in intervals:
        exact = exact_minimax_epigraph_bound(instance, lo, hi)
        approximate = oracle.bound(lo, hi)
        if exact.status == "infeasible":
            assert approximate.upper_bound == float("-inf")
            assert approximate.certified
        else:
            # The returned upper bound is an explicitly evaluated Lagrangian
            # value, while the bracket certificate controls its excess over
            # the exact minimax optimum.
            assert approximate.certified
            assert approximate.upper_bound >= exact.upper_bound - 2e-7
            assert approximate.lower_bound <= exact.upper_bound + 2e-7
            assert (
                approximate.upper_bound - exact.upper_bound
                <= approximate.optimality_gap + 2e-7
            )
            assert approximate.optimality_gap <= (
                1e-9 + 1e-8 * max(1.0, abs(approximate.upper_bound))
            )


@pytest.mark.parametrize("seed", range(10))
def test_minimax_bound_is_dominated_by_bounded_threshold_clique_lp(seed: int) -> None:
    instance = build_small_instance(seed=2000 + seed, n=5, m=4, gamma=2)
    oracle = CompressedThetaIntervalOracle(instance)
    intervals = [
        (0, len(oracle.thetas) - 1),
        (0, len(oracle.thetas) // 2),
        (len(oracle.thetas) // 2, len(oracle.thetas) - 1),
    ]
    for lo, hi in intervals:
        exact = exact_minimax_epigraph_bound(instance, lo, hi)
        clique, _seconds, _theta = bounded_theta_clique_lp(
            instance,
            float(oracle.thetas[lo]),
            float(oracle.thetas[hi]),
        )
        if math.isfinite(exact.upper_bound):
            assert exact.upper_bound <= clique + 2e-7 * max(1.0, abs(clique))


@pytest.mark.parametrize("seed", range(6))
def test_certified_deployed_bound_inherits_dominance_up_to_its_gap(seed: int) -> None:
    instance = build_small_instance(seed=3000 + seed, n=6, m=4, gamma=2)
    oracle = CompressedThetaIntervalOracle(instance)
    intervals = [
        (0, len(oracle.thetas) - 1),
        (0, len(oracle.thetas) // 2),
        (len(oracle.thetas) // 2, len(oracle.thetas) - 1),
    ]
    for lo, hi in intervals:
        deployed = oracle.bound(lo, hi)
        clique, _seconds, _theta = bounded_theta_clique_lp(
            instance,
            float(oracle.thetas[lo]),
            float(oracle.thetas[hi]),
        )
        if math.isfinite(deployed.upper_bound):
            numerical_slack = 5e-7 * max(1.0, abs(clique))
            assert (
                deployed.upper_bound
                <= clique + deployed.optimality_gap + numerical_slack
            )
