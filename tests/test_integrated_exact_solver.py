from __future__ import annotations

import pytest

from robust_mckp import Option, PricingInstance
from robust_mckp.exact_bnb import (
    brute_force_global_robust,
    build_full_theta_candidates,
)
from research.integrated_exact_solver import (
    IntervalExactConfig,
    solve_interval_exact,
)
from research.novelty_go_no_go import build_small_instance


@pytest.mark.parametrize("bound_kind", ["envelope", "clique"])
@pytest.mark.parametrize("seed", range(4))
def test_interval_exact_solver_matches_global_brute_force(
    bound_kind: str, seed: int
) -> None:
    instance = build_small_instance(seed=3000 + seed, n=5, m=3, gamma=2)
    reference = brute_force_global_robust(instance)
    actual = solve_interval_exact(
        instance,
        IntervalExactConfig(
            bound_kind=bound_kind,
            tolerance=1e-8,
            time_limit_seconds=20.0,
        ),
    )
    assert actual.status == "optimal"
    assert actual.objective_value == pytest.approx(
        reference.objective_value, abs=1e-7, rel=1e-10
    )
    assert actual.upper_bound == pytest.approx(
        actual.lower_bound, abs=2e-6, rel=1e-9
    )
    assert actual.selected_options is not None


def test_interval_exact_solver_reports_valid_anytime_bounds() -> None:
    instance = build_small_instance(seed=4040, n=7, m=4, gamma=3)
    reference = brute_force_global_robust(instance)
    actual = solve_interval_exact(
        instance,
        IntervalExactConfig(
            bound_kind="envelope",
            tolerance=1e-8,
            time_limit_seconds=1e-6,
        ),
    )
    assert actual.lower_bound <= reference.objective_value + 1e-7
    assert actual.upper_bound >= reference.objective_value - 1e-7


@pytest.mark.parametrize("bound_kind", ["envelope", "clique"])
def test_exact_interval_solver_preserves_close_unique_feasible_breakpoint(
    bound_kind: str,
) -> None:
    delta = 9e-11
    total_margin = 1.0 + delta
    items = [
        [
            Option(
                value=1.0,
                margin=total_margin / 100.0,
                uncertainty=1.0 if index == 0 else 1.0 + delta,
            )
        ]
        for index in range(100)
    ]
    instance = PricingInstance(items=items, gamma=1, name="close_unique_breakpoint")

    assert build_full_theta_candidates(instance) == [0.0, 1.0, 1.0 + delta]
    reference = brute_force_global_robust(instance)
    actual = solve_interval_exact(
        instance,
        IntervalExactConfig(
            bound_kind=bound_kind,
            tolerance=1e-12,
            time_limit_seconds=20.0,
            use_hullround_incumbent=False,
        ),
    )

    assert reference.status == "optimal"
    assert actual.status == "optimal"
    assert actual.objective_value == pytest.approx(100.0)
    assert actual.selected_theta == pytest.approx(1.0 + delta, abs=0.0, rel=0.0)
