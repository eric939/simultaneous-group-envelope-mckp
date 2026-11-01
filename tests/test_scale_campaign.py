from __future__ import annotations

import pytest

from robust_mckp.exact_bnb import brute_force_global_robust, build_full_theta_candidates
from research.interval_baselines import build_small_instance
from research.scale_campaign import (
    INTEGER_METHODS,
    PROTOCOL,
    direct_lp_scan,
    integer_method,
)
from research.structural_feasibility_study import FixedThetaLPOracle


@pytest.mark.parametrize("method", INTEGER_METHODS)
@pytest.mark.parametrize("seed", range(3))
def test_integer_methods_match_global_brute_force(method: str, seed: int) -> None:
    instance = build_small_instance(seed=5100 + seed, n=5, m=3, gamma=2)
    reference = brute_force_global_robust(instance)
    actual = integer_method(instance, method, time_limit=20.0)
    assert actual["status"] == "optimal"
    assert actual["certified"]
    assert actual["objective"] == pytest.approx(
        reference.objective_value, abs=1e-6, rel=1e-7
    )


@pytest.mark.parametrize("seed", range(3))
def test_direct_lp_scan_is_the_maximum_fixed_threshold_lp(seed: int) -> None:
    instance = build_small_instance(seed=5200 + seed, n=6, m=3, gamma=2)
    oracle = FixedThetaLPOracle(instance)
    expected = max(float(oracle.value(float(t))) for t in build_full_theta_candidates(instance))
    actual = direct_lp_scan(instance, time_limit=20.0)
    assert actual["status"] == "optimal"
    assert actual["value"] == pytest.approx(expected, abs=1e-9, rel=1e-12)
    assert actual["value"] >= brute_force_global_robust(instance).objective_value - 1e-7


def test_direct_lp_scan_reports_time_limit() -> None:
    instance = build_small_instance(seed=5300, n=6, m=3, gamma=2)
    actual = direct_lp_scan(instance, time_limit=0.0)
    assert actual["status"] == "time_limit"
    assert actual["theta_lp_evaluations"] == 0


def test_protocol_is_frozen() -> None:
    assert PROTOCOL["version"] == "v2.1.1-scale-20261101-frozen"
    assert PROTOCOL["integer"]["incumbent"].startswith("every method starts cold")
    assert PROTOCOL["integer"]["time_limit_seconds"] == 120.0
