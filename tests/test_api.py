from __future__ import annotations

import pytest

from robust_mckp import (
    Option,
    PricingInstance,
    filter_admissible_options,
    from_pricing_data,
    solve,
)
from robust_mckp.certificate import compute_certificate
from robust_mckp.solver import _solve_naive_reference


def test_low_level_api_returns_feasible_solution() -> None:
    instance = PricingInstance(
        items=[
            [Option(value=5.0, margin=2.0, uncertainty=0.2), Option(value=6.0, margin=1.2, uncertainty=0.8)],
            [Option(value=4.0, margin=1.5, uncertainty=0.1), Option(value=7.0, margin=0.6, uncertainty=0.7)],
        ],
        gamma=1,
    )

    solution = solve(instance)

    assert solution.is_feasible
    assert len(solution.selections) == instance.n_items
    assert solution.objective > 0.0
    assert compute_certificate(instance, solution.selections) == pytest.approx(solution.certificate_value)
    instr = solution.metadata["instrumentation"]
    assert instr["selected_theta_delta_v_max"] >= 0.0
    assert instr["selected_theta_delta_v_max_over_lp_upper_bound"] >= 0.0
    assert instr["selected_theta_lp_minus_round_down"] <= instr["selected_theta_delta_v_max"] + 1e-8
    assert instr["selected_theta_lp_minus_round_down_over_lp_upper_bound"] >= -1e-8


def test_from_pricing_data_filters_to_admissible_options() -> None:
    instance = from_pricing_data(
        reference_prices=[100.0],
        weights=[1.0],
        price_menus=[[80.0, 95.0, 100.0, 110.0, 125.0]],
        demands=[[1.0, 0.9, 0.85, 0.8, 0.7]],
        uncertainties=[[0.05, 0.05, 0.05, 0.05, 0.05]],
        margin_target=0.9,
        tolerances=[0.10],
        gamma=0,
    )

    kept_prices = [opt.price for opt in instance.items[0]]

    assert kept_prices == pytest.approx([95.0, 100.0, 110.0])


def test_filter_admissible_options_on_low_level_instance() -> None:
    instance = PricingInstance(
        items=[
            [
                Option(value=1.0, margin=0.5, uncertainty=0.1, price=90.0),
                Option(value=2.0, margin=0.8, uncertainty=0.2, price=100.0),
                Option(value=3.0, margin=1.0, uncertainty=0.3, price=120.0),
            ]
        ],
        gamma=0,
    )

    filtered = filter_admissible_options(instance, reference_prices=[100.0], tolerances=[0.10])

    assert len(filtered.items[0]) == 2
    assert [opt.price for opt in filtered.items[0]] == pytest.approx([90.0, 100.0])


def test_optimized_solver_matches_naive_reference_on_small_instance() -> None:
    instance = from_pricing_data(
        reference_prices=[100.0, 110.0, 95.0],
        weights=[1.0, 0.8, 1.2],
        price_menus=[
            [90.0, 100.0, 110.0],
            [99.0, 110.0, 121.0],
            [85.5, 95.0, 104.5],
        ],
        demands=[
            [1.10, 1.00, 0.90],
            [0.95, 0.85, 0.72],
            [1.20, 1.05, 0.88],
        ],
        uncertainties=[
            [0.05, 0.06, 0.08],
            [0.04, 0.05, 0.07],
            [0.03, 0.05, 0.06],
        ],
        margin_target=0.92,
        tolerances=[0.10, 0.10, 0.10],
        gamma=1,
    )

    fast = solve(instance)
    reference = _solve_naive_reference(instance)

    assert fast.is_feasible
    assert reference.is_feasible
    assert fast.objective == pytest.approx(reference.objective)
    assert fast.certificate_value == pytest.approx(reference.certificate_value)
