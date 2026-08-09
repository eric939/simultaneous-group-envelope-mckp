from __future__ import annotations

import math

import numpy as np
import pytest

from robust_mckp import GlobalThetaBNBConfig, Option, PricingInstance, solve
from robust_mckp.exact_bnb import (
    brute_force_global_robust,
    build_full_theta_candidates,
    compute_fixed_theta_lp_upper_bound,
    solve_global_theta_bnb,
)


def _random_instance(rng: np.random.Generator, n: int, m: int, gamma: int) -> PricingInstance:
    items = []
    for _ in range(n):
        group = []
        for _j in range(m):
            value = float(rng.integers(0, 50))
            margin = float(rng.integers(-8, 15))
            uncertainty = float(rng.integers(-6, 7))
            group.append(Option(value=value, margin=margin, uncertainty=uncertainty))
        # Ensure each item has at least one safe-ish option so not every case is infeasible.
        group[0] = Option(value=float(rng.integers(0, 20)), margin=float(rng.integers(2, 12)), uncertainty=0.0)
        items.append(group)
    return PricingInstance(items=items, gamma=gamma)


def _fixed_theta_encoded_instance(cost_groups, value_groups, capacity: float) -> PricingInstance:
    n = len(cost_groups)
    s_star = float(capacity) / float(n)
    return PricingInstance(
        items=[
            [Option(value=float(v), margin=s_star - float(c), uncertainty=0.0) for c, v in zip(costs, values)]
            for costs, values in zip(cost_groups, value_groups)
        ],
        gamma=0,
    )


def _assert_global_matches_bruteforce(instance: PricingInstance) -> None:
    exact = solve_global_theta_bnb(instance, GlobalThetaBNBConfig(use_hullround_incumbent=True))
    brute = brute_force_global_robust(instance)
    assert exact.status == brute.status
    if brute.status == "optimal":
        assert exact.objective_value == pytest.approx(brute.objective_value, abs=1e-8)
        assert exact.robust_certificate >= -1e-8
        assert exact.absolute_gap == pytest.approx(0.0, abs=1e-8)
        assert exact.validation_flags["robust_certificate_feasible"]


def test_full_theta_candidates_include_zero_abs_duplicates_and_negative_signs() -> None:
    instance = PricingInstance(
        items=[
            [Option(1.0, 1.0, 0.0), Option(2.0, 1.0, -3.0)],
            [Option(3.0, 1.0, 3.0), Option(4.0, 1.0, -5.0)],
        ],
        gamma=1,
    )
    assert build_full_theta_candidates(instance) == pytest.approx([0.0, 3.0, 5.0])


def test_full_theta_candidates_all_zero_single_option() -> None:
    instance = PricingInstance(items=[[Option(1.0, 1.0, 0.0)]], gamma=0)
    assert build_full_theta_candidates(instance) == [0.0]


def test_global_bnb_rejects_negative_certificate_inside_solver_tolerance() -> None:
    instance = PricingInstance(
        items=[[Option(1.0, -5e-10, 0.0)]],
        gamma=0,
    )
    assert solve_global_theta_bnb(instance).status == "infeasible"
    assert brute_force_global_robust(instance).status == "infeasible"


def test_exact_cost_dominance_keeps_feasible_option_when_float_costs_collapse() -> None:
    large = float(2**84)
    instance = PricingInstance(
        items=[
            [
                Option(-1000.0, large, 0.0),
                Option(0.0, 0.0, 0.0),
                Option(100.0, -1.0, 0.0),
            ]
        ],
        gamma=0,
    )
    exact = solve_global_theta_bnb(
        instance, GlobalThetaBNBConfig(use_hullround_incumbent=False)
    )
    brute = brute_force_global_robust(instance)
    assert exact.status == "optimal"
    assert exact.selected_options == [1]
    assert exact.objective_value == pytest.approx(0.0, abs=0.0, rel=0.0)
    assert exact.objective_value == pytest.approx(
        brute.objective_value, abs=0.0, rel=0.0
    )


def test_exact_solver_handles_transformed_cost_overflow_conservatively() -> None:
    maximum = float(np.finfo(float).max)
    instance = PricingInstance(
        items=[
            [
                Option(0.0, maximum, 0.0),
                Option(100.0, -maximum, 0.0),
            ]
        ],
        gamma=0,
    )
    exact = solve_global_theta_bnb(
        instance, GlobalThetaBNBConfig(use_hullround_incumbent=False)
    )
    assert exact.status == "optimal"
    assert exact.selected_options == [0]
    assert exact.objective_value == pytest.approx(0.0, abs=0.0, rel=0.0)


def test_reduced_theta_set_can_falsely_reject_feasible_selection() -> None:
    # With two selected deviations of size 1 and Gamma=1, theta=0 gives
    # (0.75 - 1) + (0.75 - 1) < 0, but theta=1 gives 0.75 + 0.75 - 1 > 0.
    # Any exact implementation that checks only theta=0 would therefore reject
    # a robust-feasible selection.  The full original breakpoint set includes 1.
    instance = PricingInstance(
        items=[
            [Option(1.0, 0.75, 1.0)],
            [Option(1.0, 0.75, 1.0)],
        ],
        gamma=1,
    )
    theta_zero_residual = sum(opt.margin - max(0.0, abs(opt.uncertainty) - 0.0) for [opt] in instance.items)
    theta_one_residual = sum(opt.margin - max(0.0, abs(opt.uncertainty) - 1.0) for [opt] in instance.items) - 1.0
    assert theta_zero_residual < 0.0
    assert theta_one_residual > 0.0
    assert build_full_theta_candidates(instance) == pytest.approx([0.0, 1.0])

    res = solve_global_theta_bnb(instance, GlobalThetaBNBConfig(use_hullround_incumbent=False))
    assert res.status == "optimal"
    assert res.robust_certificate == pytest.approx(0.5)


def test_fixed_theta_lp_upper_bound_is_valid_against_bruteforce() -> None:
    instance = _fixed_theta_encoded_instance(
        cost_groups=[[0.0, 5.0, 10.0], [0.0, 3.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0, 4.0]],
        capacity=8.0,
    )
    bound = compute_fixed_theta_lp_upper_bound(instance, 0.0)
    brute = brute_force_global_robust(instance)
    assert bound.lp_feasible
    assert bound.lp_upper_bound + 1e-8 >= brute.objective_value


def test_global_bnb_matches_bruteforce_random_small_instances() -> None:
    rng = np.random.default_rng(20260507)
    for n in range(1, 8):
        for m in range(1, 5):
            gammas = sorted(set([0, n, int(math.floor(math.sqrt(n)))]))
            for gamma in gammas:
                for _ in range(4):
                    instance = _random_instance(rng, n, m, gamma)
                    _assert_global_matches_bruteforce(instance)


def test_global_bnb_all_zero_uncertainty_gamma_n() -> None:
    instance = PricingInstance(
        items=[
            [Option(1.0, 0.0, 0.0), Option(5.0, -1.0, 0.0)],
            [Option(2.0, 1.0, 0.0), Option(7.0, 0.0, 0.0)],
        ],
        gamma=2,
    )
    _assert_global_matches_bruteforce(instance)


def test_global_bnb_no_robust_feasible_solution() -> None:
    instance = PricingInstance(
        items=[
            [Option(10.0, -5.0, 0.0), Option(20.0, -3.0, 1.0)],
            [Option(3.0, -2.0, 0.0), Option(4.0, -1.0, -1.0)],
        ],
        gamma=1,
    )
    res = solve_global_theta_bnb(instance, GlobalThetaBNBConfig(use_hullround_incumbent=False))
    brute = brute_force_global_robust(instance)
    assert res.status == "infeasible"
    assert brute.status == "infeasible"
    assert res.selected_options is None


def test_global_bnb_exactly_one_feasible_and_binding_certificate() -> None:
    instance = PricingInstance(
        items=[
            [Option(1.0, 2.0, 1.0), Option(10.0, -1.0, 0.0)],
            [Option(2.0, -1.0, 0.0), Option(20.0, -2.0, 0.0)],
        ],
        gamma=1,
    )
    res = solve_global_theta_bnb(instance, GlobalThetaBNBConfig(use_hullround_incumbent=False))
    assert res.status == "optimal"
    assert res.selected_options == [0, 0]
    assert res.robust_certificate == pytest.approx(0.0)


def test_global_bnb_preserves_below_hull_integer_option() -> None:
    instance = _fixed_theta_encoded_instance(
        cost_groups=[[0.0, 5.0, 10.0], [0.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0]],
        capacity=5.0,
    )
    res = solve_global_theta_bnb(instance, GlobalThetaBNBConfig(use_hullround_incumbent=False))
    assert res.status == "optimal"
    assert res.objective_value == pytest.approx(8.0)
    assert res.selected_options == [1, 0]


def test_global_exact_objective_not_below_hullround_when_certified() -> None:
    instance = PricingInstance(
        items=[
            [Option(5.0, 3.0, 0.0), Option(9.0, 1.5, 1.0), Option(10.0, 0.0, 2.0)],
            [Option(4.0, 2.0, 0.0), Option(8.0, 1.0, 1.0), Option(11.0, -1.0, 2.0)],
            [Option(3.0, 2.0, 0.0), Option(7.0, 0.5, 1.0), Option(9.0, -1.0, 2.0)],
        ],
        gamma=1,
    )
    hr = solve(instance)
    exact = solve_global_theta_bnb(instance)
    assert exact.status == "optimal"
    assert exact.robust_certificate >= -1e-8
    if hr.is_feasible:
        assert exact.objective_value + 1e-8 >= hr.objective


def test_global_node_limit_returns_anytime_gap_without_optimal_claim() -> None:
    instance = _fixed_theta_encoded_instance(
        cost_groups=[[0.0, 5.0, 10.0], [0.0, 5.0, 10.0], [0.0, 5.0, 10.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0, 9.0, 19.0], [0.0, 7.0, 18.0]],
        capacity=10.0,
    )
    res = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(node_limit=1, use_hullround_incumbent=False, use_fixed_theta_greedy_incumbent=True),
    )
    assert res.status in {"node_limit", "optimal"}
    if res.status == "node_limit":
        assert res.selected_options is not None
        assert res.upper_bound + 1e-8 >= res.lower_bound
        assert res.absolute_gap >= -1e-8


def test_global_cached_and_uncached_match_bruteforce() -> None:
    rng = np.random.default_rng(99)
    instance = _random_instance(rng, n=5, m=4, gamma=2)
    brute = brute_force_global_robust(instance)
    cached = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(use_caches=True, use_hullround_incumbent=False, theta_order="increasing"),
    )
    uncached = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(use_caches=False, use_hullround_incumbent=False, theta_order="increasing"),
    )
    assert cached.status == brute.status
    assert uncached.status == brute.status
    if brute.status == "optimal":
        assert cached.objective_value == pytest.approx(brute.objective_value)
        assert uncached.objective_value == pytest.approx(brute.objective_value)


def test_global_theta_orderings_are_exact_equivalent() -> None:
    rng = np.random.default_rng(2026)
    instance = _random_instance(rng, n=6, m=4, gamma=2)
    brute = brute_force_global_robust(instance)
    objectives = []
    for order in ["increasing", "lp_bound_desc", "hybrid", "heuristic_incumbent_desc"]:
        res = solve_global_theta_bnb(
            instance,
            GlobalThetaBNBConfig(use_caches=True, use_hullround_incumbent=True, theta_order=order),
        )
        assert res.status == brute.status
        if brute.status == "optimal":
            assert res.objective_value == pytest.approx(brute.objective_value)
            assert res.robust_certificate >= -1e-8
            objectives.append(res.objective_value)
    if objectives:
        assert max(objectives) == pytest.approx(min(objectives))


def test_heuristic_theta_order_seeds_valid_incumbents_without_changing_exactness() -> None:
    rng = np.random.default_rng(20260518)
    instance = _random_instance(rng, n=6, m=4, gamma=2)
    brute = brute_force_global_robust(instance)
    res = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(
            use_caches=True,
            use_hullround_incumbent=False,
            use_objective_cutoff=True,
            theta_order="heuristic_incumbent_desc",
            collect_diagnostics=True,
        ),
    )
    assert res.status == brute.status
    if brute.status == "optimal":
        assert res.objective_value == pytest.approx(brute.objective_value)
        assert res.robust_certificate >= -1e-8
    assert res.diagnostics["heuristic_theta_order_enabled"] is True
    assert res.diagnostics["heuristic_theta_order_evaluated"] >= 1
    assert res.diagnostics["heuristic_theta_order_robust_feasible"] >= 0
    assert res.diagnostics["theta_order_used"] == "heuristic_incumbent_desc"


def test_global_cutoff_does_not_false_certify_fixed_theta_but_certifies_global() -> None:
    instance = _fixed_theta_encoded_instance(
        cost_groups=[[0.0, 5.0, 10.0], [0.0, 5.0, 10.0], [0.0, 5.0, 10.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0, 9.0, 19.0], [0.0, 7.0, 18.0]],
        capacity=10.0,
    )
    res = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(use_caches=True, use_objective_cutoff=True, use_hullround_incumbent=True),
    )
    brute = brute_force_global_robust(instance)
    assert res.status == "optimal"
    assert res.objective_value == pytest.approx(brute.objective_value)
    assert all(r.status != "optimal" or r.bnb_upper_bound <= res.objective_value + 1e-8 for r in res.per_theta_records)


def test_global_limited_ordered_run_returns_valid_gap() -> None:
    rng = np.random.default_rng(123)
    instance = _random_instance(rng, n=7, m=4, gamma=2)
    res = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(
            node_limit=1,
            theta_order="lp_bound_desc",
            use_caches=True,
            use_objective_cutoff=True,
            use_hullround_incumbent=True,
        ),
    )
    assert res.status in {"optimal", "node_limit", "time_limit"}
    if res.status != "optimal":
        assert res.upper_bound + 1e-8 >= res.lower_bound
        assert res.absolute_gap >= -1e-8


def test_global_limited_gap_has_upper_bound_record_for_every_theta() -> None:
    instance = PricingInstance(
        items=[
            [Option(0.0, 3.0, 0.0), Option(8.0, 2.0, 1.0), Option(20.0, 0.5, 2.0)],
            [Option(0.0, 3.0, 0.0), Option(9.0, 1.9, 1.5), Option(19.0, 0.4, 3.0)],
            [Option(0.0, 3.0, 0.0), Option(7.0, 2.1, 2.5), Option(18.0, 0.3, 4.0)],
        ],
        gamma=1,
    )
    candidates = build_full_theta_candidates(instance)
    res = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(
            node_limit=0,
            theta_order="increasing",
            use_hullround_incumbent=False,
            use_fixed_theta_greedy_incumbent=False,
            collect_diagnostics=True,
        ),
    )
    assert len(res.per_theta_records) == len(candidates)
    assert res.diagnostics["theta_upper_bound_records_initialized"] is True
    assert res.diagnostics["theta_upper_bound_record_count"] == len(candidates)
    assert res.diagnostics["theta_upper_bound_missing_count"] == 0
    assert res.diagnostics["theta_upper_bound_invalid_count"] == 0
    for record in res.per_theta_records:
        assert record.root_lp_status in {"optimal", "infeasible_capacity", "empty_hull", "lp_infeasible"}
        if record.root_lp_status == "optimal":
            assert math.isfinite(record.fixed_theta_lp_upper_bound)
            assert math.isfinite(record.bnb_upper_bound)
            assert record.bnb_upper_bound == pytest.approx(record.fixed_theta_lp_upper_bound)
        else:
            assert record.bnb_upper_bound == float("-inf")
    assert res.status in {"node_limit", "infeasible", "optimal"}
    if res.status == "node_limit":
        assert res.upper_bound + 1e-8 >= res.lower_bound or not math.isfinite(res.lower_bound)
        assert res.absolute_gap >= -1e-8


def test_global_multistart_incumbent_matches_bruteforce_and_validates_certificate() -> None:
    rng = np.random.default_rng(20260513)
    instance = _random_instance(rng, n=6, m=4, gamma=2)
    brute = brute_force_global_robust(instance)
    res = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(
            use_hullround_incumbent=True,
            use_multistart_incumbent=True,
            multistart_theta_count=5,
            use_local_incumbent_improvement=True,
            local_search_neighborhood="single_two",
            local_search_max_pair_evaluations=2000,
            collect_diagnostics=True,
            theta_order="lp_bound_desc",
        ),
    )
    assert res.status == brute.status
    if brute.status == "optimal":
        assert res.objective_value == pytest.approx(brute.objective_value, abs=1e-8)
        assert res.robust_certificate >= -1e-8
        assert res.validation_flags["robust_certificate_feasible"]
    assert res.diagnostics["multistart_incumbent_enabled"] is True
    assert res.diagnostics["multistart_theta_count_evaluated"] >= 0


def test_global_diagnostics_do_not_change_solution() -> None:
    rng = np.random.default_rng(777)
    instance = _random_instance(rng, n=5, m=4, gamma=2)
    base = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(
            use_caches=True,
            use_objective_cutoff=True,
            use_hullround_incumbent=True,
            theta_order="lp_bound_desc",
        ),
    )
    diag = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(
            use_caches=True,
            use_objective_cutoff=True,
            use_hullround_incumbent=True,
            theta_order="lp_bound_desc",
            collect_diagnostics=True,
            profile_timing=True,
            max_diagnostic_nodes=100,
        ),
    )
    assert diag.status == base.status
    if base.status == "optimal":
        assert diag.objective_value == pytest.approx(base.objective_value)
        assert diag.selected_options == base.selected_options
    assert diag.diagnostics["theta_total"] == len(diag.per_theta_records)
    assert diag.diagnostics["theta_order_used"] == "lp_bound_desc"
    assert "unresolved_theta" in diag.diagnostics
    assert all(isinstance(record.diagnostics, dict) for record in diag.per_theta_records)


def test_global_limited_diagnostics_return_valid_gap() -> None:
    instance = _fixed_theta_encoded_instance(
        cost_groups=[[0.0, 5.0, 10.0], [0.0, 5.0, 10.0], [0.0, 5.0, 10.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0, 9.0, 19.0], [0.0, 7.0, 18.0]],
        capacity=10.0,
    )
    res = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(
            node_limit=1,
            use_caches=True,
            use_objective_cutoff=True,
            use_hullround_incumbent=False,
            collect_diagnostics=True,
            profile_timing=True,
        ),
    )
    assert res.status in {"optimal", "node_limit", "time_limit"}
    if res.status != "optimal":
        assert res.upper_bound + 1e-8 >= res.lower_bound
        assert res.absolute_gap >= -1e-8
    assert "total_nodes_pruned_bound" in res.diagnostics


def test_global_fast_and_reference_bounds_match_small_instance() -> None:
    rng = np.random.default_rng(2027)
    instance = _random_instance(rng, n=5, m=4, gamma=2)
    ref = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(
            use_caches=True,
            use_objective_cutoff=True,
            use_hullround_incumbent=True,
            theta_order="lp_bound_desc",
            use_fast_residual_lp_bound=False,
            use_cheap_prebound=False,
            use_min_cost_infeasibility_check=True,
        ),
    )
    fast = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(
            use_caches=True,
            use_objective_cutoff=True,
            use_hullround_incumbent=True,
            theta_order="lp_bound_desc",
            use_fast_residual_lp_bound=True,
            use_cheap_prebound=True,
            use_min_cost_infeasibility_check=True,
        ),
    )
    assert fast.status == ref.status
    if ref.status == "optimal":
        assert fast.objective_value == pytest.approx(ref.objective_value)
        assert fast.selected_options == ref.selected_options
    assert fast.upper_bound + 1e-8 >= fast.lower_bound


def test_global_fast_bound_preserves_below_hull_option() -> None:
    instance = _fixed_theta_encoded_instance(
        cost_groups=[[0.0, 5.0, 10.0], [0.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0]],
        capacity=5.0,
    )
    res = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(
            use_hullround_incumbent=False,
            use_fast_residual_lp_bound=True,
            use_cheap_prebound=True,
            use_bound_cache=True,
        ),
    )
    assert res.status == "optimal"
    assert res.objective_value == pytest.approx(8.0)
    assert res.selected_options == [1, 0]


@pytest.mark.parametrize(
    "branching_rule",
    [
        "fractional_item_then_spread",
        "largest_hull_jump",
        "largest_cost_spread",
        "tight_capacity_hybrid",
        "strong_branching_lite",
    ],
)
def test_global_branching_rules_match_bruteforce(branching_rule: str) -> None:
    rng = np.random.default_rng(3030)
    instance = _random_instance(rng, n=5, m=4, gamma=2)
    brute = brute_force_global_robust(instance)
    res = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(
            use_caches=True,
            use_hullround_incumbent=True,
            theta_order="lp_bound_desc",
            branching_rule=branching_rule,
            strong_branching_candidates=2,
            strong_branching_depth_limit=2,
            strong_branching_max_children=10,
            use_local_incumbent_improvement=True,
        ),
    )
    assert res.status == brute.status
    if brute.status == "optimal":
        assert res.objective_value == pytest.approx(brute.objective_value)
        assert res.robust_certificate >= -1e-8


def test_global_branching_limited_run_returns_valid_gap() -> None:
    rng = np.random.default_rng(4040)
    instance = _random_instance(rng, n=7, m=4, gamma=2)
    res = solve_global_theta_bnb(
        instance,
        GlobalThetaBNBConfig(
            node_limit=1,
            use_caches=True,
            theta_order="lp_bound_desc",
            branching_rule="tight_capacity_hybrid",
            collect_diagnostics=True,
            profile_timing=True,
        ),
    )
    assert res.status in {"optimal", "node_limit", "time_limit"}
    if res.status != "optimal":
        assert res.upper_bound + 1e-8 >= res.lower_bound
        assert res.absolute_gap >= -1e-8
    assert "total_strong_branching_count" in res.diagnostics
