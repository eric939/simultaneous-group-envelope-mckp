from __future__ import annotations

import itertools
import math
import signal

import numpy as np
import pytest

from robust_mckp import FixedThetaBNBConfig, Option, PricingInstance, solve_fixed_theta_bnb
from robust_mckp.exact_bnb import (
    _Node,
    _build_fixed_theta_cache,
    _compute_bound,
    brute_force_fixed_theta,
    build_fixed_theta_data,
    cost_for_selection,
    objective_for_selection,
)


def _instance_from_fixed_theta_points(cost_groups, value_groups, capacity: float) -> PricingInstance:
    """Encode a fixed-theta MCKP as a gamma=0, theta=0 PricingInstance."""

    n = len(cost_groups)
    s_star = float(capacity) / float(n)
    items = []
    for costs, values in zip(cost_groups, value_groups):
        group = [Option(value=float(v), margin=s_star - float(c), uncertainty=0.0) for c, v in zip(costs, values)]
        items.append(group)
    return PricingInstance(items=items, gamma=0)


def _assert_matches_bruteforce(instance: PricingInstance, theta: float = 0.0) -> None:
    bnb = solve_fixed_theta_bnb(instance, theta)
    brute = brute_force_fixed_theta(instance, theta)
    assert bnb.status == brute.status
    if brute.status == "optimal":
        assert bnb.objective_value == pytest.approx(brute.objective_value, abs=1e-8)
        assert bnb.used_capacity <= bnb.capacity + 1e-8
        assert bnb.absolute_gap == pytest.approx(0.0, abs=1e-8)
        assert bnb.validation_flags["capacity_feasible"]
        assert bnb.validation_flags["valid_indices"]


def test_bnb_matches_bruteforce_random_tiny_instances() -> None:
    rng = np.random.default_rng(12345)
    for n in range(1, 9):
        for m in range(1, 6):
            for _ in range(8):
                cost_groups = []
                value_groups = []
                for _i in range(n):
                    costs = rng.integers(0, 12, size=m).astype(float)
                    costs[0] = 0.0
                    values = rng.integers(0, 30, size=m).astype(float)
                    cost_groups.append(costs)
                    value_groups.append(values)
                capacity = float(rng.integers(0, max(1, 8 * n)))
                instance = _instance_from_fixed_theta_points(cost_groups, value_groups, capacity)
                _assert_matches_bruteforce(instance)


def test_positive_sub_epsilon_hull_segment_is_not_erased() -> None:
    instance = PricingInstance(
        items=[
            [
                Option(value=0.0, margin=5e-11, uncertainty=0.0),
                Option(value=1.0, margin=0.0, uncertainty=0.0),
            ]
        ],
        gamma=0,
    )
    result = solve_fixed_theta_bnb(
        instance,
        0.0,
        FixedThetaBNBConfig(use_greedy_incumbent=False),
    )
    assert result.status == "optimal"
    assert result.selected_options == [1]
    assert result.objective_value == pytest.approx(1.0, abs=0.0, rel=0.0)


def test_exact_lp_bound_preserves_objective_cancellation() -> None:
    instance = PricingInstance(
        items=[
            [Option(0.0, 0.0, 0.0), Option(1.0, -1e-20, 0.0)],
            [Option(-1e30, 0.0, 0.0), Option(0.0, -1e20, 0.0)],
            [Option(0.0, 1e20, 0.0)],
            [Option(0.0, 1e-20, 0.0)],
        ],
        gamma=0,
    )
    result = solve_fixed_theta_bnb(
        instance,
        0.0,
        FixedThetaBNBConfig(use_greedy_incumbent=False),
    )
    brute = brute_force_fixed_theta(instance, 0.0)
    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(1.0, abs=0.0, rel=0.0)
    assert result.objective_value == pytest.approx(
        brute.objective_value, abs=0.0, rel=0.0
    )


def test_exact_objective_comparison_preserves_sub_ulp_improvement() -> None:
    instance = PricingInstance(
        items=[
            [Option(-1.3468787627424937e29, 0.0, 0.0)],
            [Option(2.1267647932558654e37, 0.0, 0.0)],
            [Option(0.0, 1.0, 0.0), Option(1.0, 0.0, 0.0)],
        ],
        gamma=0,
    )
    incumbent = objective_for_selection(
        [
            np.asarray([option.value for option in group], dtype=float)
            for group in instance.items
        ],
        [0, 0, 0],
    )
    result = solve_fixed_theta_bnb(
        instance,
        0.0,
        FixedThetaBNBConfig(
            tolerance=1e-300,
            use_greedy_incumbent=False,
            initial_incumbent_selection=[0, 0, 0],
            initial_incumbent_value=incumbent,
        ),
    )
    brute = brute_force_fixed_theta(instance, 0.0, tol=1e-300)
    assert result.status == "optimal"
    assert result.selected_options == [0, 0, 1]
    assert brute.selected_options == [0, 0, 1]


def test_negative_fixed_theta_capacity_is_infeasible() -> None:
    instance = PricingInstance(
        items=[[Option(value=1.0, margin=0.0, uncertainty=0.0)]],
        gamma=1,
    )
    res = solve_fixed_theta_bnb(instance, theta=1.0)
    assert res.status == "infeasible"
    assert res.selected_options is None


def test_zero_capacity_and_single_option_items() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0], [0.0], [0.0]],
        value_groups=[[1.0], [2.0], [3.0]],
        capacity=0.0,
    )
    res = solve_fixed_theta_bnb(instance, theta=0.0)
    assert res.status == "optimal"
    assert res.objective_value == pytest.approx(6.0)
    assert res.used_capacity == pytest.approx(0.0)
    assert res.selected_options == [0, 0, 0]


def test_equal_cost_and_dominated_options() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[
            [0.0, 2.0, 2.0, 4.0],
            [0.0, 3.0, 6.0],
        ],
        value_groups=[
            [1.0, 3.0, 7.0, 6.0],  # option 1 is equal-cost dominated by option 2; option 3 is dominated by 2.
            [2.0, 5.0, 4.0],  # option 2 is dominated by option 1.
        ],
        capacity=5.0,
    )
    _assert_matches_bruteforce(instance)


def test_below_hull_option_can_be_integer_optimal() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[
            [0.0, 5.0, 10.0],
            [0.0],
        ],
        value_groups=[
            [0.0, 8.0, 20.0],  # (5, 8) is below the upper hull between (0, 0) and (10, 20).
            [0.0],
        ],
        capacity=5.0,
    )
    res = solve_fixed_theta_bnb(instance, theta=0.0, config=FixedThetaBNBConfig(use_greedy_incumbent=False))
    brute = brute_force_fixed_theta(instance, theta=0.0)
    assert brute.selected_options == [1, 0]
    assert res.status == "optimal"
    assert res.objective_value == pytest.approx(8.0)
    assert res.selected_options == [1, 0]


def test_lp_integral_root_capacity_exactly_binding() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[
            [0.0, 3.0],
            [0.0, 2.0],
        ],
        value_groups=[
            [1.0, 10.0],
            [1.0, 8.0],
        ],
        capacity=5.0,
    )
    res = solve_fixed_theta_bnb(instance, theta=0.0)
    assert res.status == "optimal"
    assert res.objective_value == pytest.approx(18.0)
    assert res.used_capacity == pytest.approx(5.0)


def test_node_limit_returns_valid_anytime_gap() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 5.0, 10.0], [0.0, 5.0, 10.0], [0.0, 5.0, 10.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0, 9.0, 19.0], [0.0, 7.0, 18.0]],
        capacity=10.0,
    )
    res = solve_fixed_theta_bnb(instance, theta=0.0, config=FixedThetaBNBConfig(node_limit=1))
    assert res.status in {"node_limit", "optimal"}
    if res.status == "node_limit":
        assert res.selected_options is not None
        assert res.upper_bound + 1e-8 >= res.lower_bound
        assert res.absolute_gap >= -1e-8


def test_optional_scipy_highs_milp_comparison() -> None:
    def _timeout(_signum, _frame):
        raise TimeoutError("scipy.optimize import timed out")

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(20)
    try:
        scipy_opt = pytest.importorskip("scipy.optimize")
    except TimeoutError:
        pytest.skip("scipy.optimize import timed out in this environment")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    if not hasattr(scipy_opt, "milp"):
        pytest.skip("scipy.optimize.milp is unavailable")

    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 4.0, 7.0], [0.0, 3.0, 6.0], [0.0, 2.0, 5.0]],
        value_groups=[[1.0, 9.0, 12.0], [2.0, 7.0, 11.0], [1.0, 5.0, 8.0]],
        capacity=9.0,
    )
    data = build_fixed_theta_data(instance, 0.0)
    sizes = [len(v) for v in data.values]
    total = sum(sizes)
    c = -np.concatenate(data.values)
    integrality = np.ones(total, dtype=int)
    bounds = scipy_opt.Bounds(lb=np.zeros(total), ub=np.ones(total))
    a_eq = np.zeros((len(sizes), total))
    offset = 0
    for i, size in enumerate(sizes):
        a_eq[i, offset : offset + size] = 1.0
        offset += size
    constraints = [
        scipy_opt.LinearConstraint(a_eq, np.ones(len(sizes)), np.ones(len(sizes))),
        scipy_opt.LinearConstraint(np.concatenate(data.costs)[None, :], -np.inf, np.array([data.capacity])),
    ]
    highs = scipy_opt.milp(c, integrality=integrality, bounds=bounds, constraints=constraints)
    if int(getattr(highs, "status", -1)) != 0:
        pytest.skip(f"HiGHS did not certify this test MILP: {getattr(highs, 'message', '')}")

    bnb = solve_fixed_theta_bnb(instance, 0.0)
    assert bnb.status == "optimal"
    assert bnb.objective_value == pytest.approx(-float(highs.fun), abs=1e-8)


def test_bruteforce_reference_uses_original_options() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 1.0], [0.0, 1.0]],
        value_groups=[[0.0, 2.0], [0.0, 3.0]],
        capacity=1.0,
    )
    brute = brute_force_fixed_theta(instance, 0.0)
    assert brute.status == "optimal"
    assert brute.objective_value == pytest.approx(3.0)
    assert brute.selected_options == [0, 1]

    data = build_fixed_theta_data(instance, 0.0)
    assert objective_for_selection(data.values, brute.selected_options) == pytest.approx(3.0)
    assert cost_for_selection(data.costs, brute.selected_options) <= data.capacity + 1e-8


def test_cached_and_uncached_fixed_theta_bnb_match() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 5.0, 10.0], [0.0, 4.0, 9.0], [0.0, 6.0, 11.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0, 7.0, 18.0], [0.0, 9.0, 17.0]],
        capacity=15.0,
    )
    cached = solve_fixed_theta_bnb(instance, 0.0, FixedThetaBNBConfig(use_cache=True))
    uncached = solve_fixed_theta_bnb(instance, 0.0, FixedThetaBNBConfig(use_cache=False))
    assert cached.status == "optimal"
    assert uncached.status == "optimal"
    assert cached.objective_value == pytest.approx(uncached.objective_value)
    assert cached.selected_options == uncached.selected_options


def test_fixed_theta_objective_cutoff_prunes_without_false_optimality() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0, 9.0, 19.0]],
        capacity=10.0,
    )
    # The true optimum is 20. A cutoff above it may stop the local solve, but
    # it must not be reported as a fixed-theta optimum.
    res = solve_fixed_theta_bnb(
        instance,
        0.0,
        FixedThetaBNBConfig(use_greedy_incumbent=False, objective_cutoff=25.0, use_cutoff_pruning=True),
    )
    assert res.status == "cutoff_pruned"
    assert res.status != "optimal"
    assert res.upper_bound >= 20.0
    assert res.absolute_gap >= -1e-8


def test_fixed_theta_cutoff_keeps_optimum_above_cutoff() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0, 9.0, 19.0]],
        capacity=10.0,
    )
    res = solve_fixed_theta_bnb(
        instance,
        0.0,
        FixedThetaBNBConfig(use_greedy_incumbent=False, objective_cutoff=10.0, use_cutoff_pruning=True),
    )
    brute = brute_force_fixed_theta(instance, 0.0)
    assert res.status == "optimal"
    assert res.objective_value == pytest.approx(brute.objective_value)


def test_below_hull_option_survives_cached_cutoff_mode() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 5.0, 10.0], [0.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0]],
        capacity=5.0,
    )
    res = solve_fixed_theta_bnb(
        instance,
        0.0,
        FixedThetaBNBConfig(use_cache=True, objective_cutoff=1.0, use_cutoff_pruning=True, use_greedy_incumbent=False),
    )
    assert res.status == "optimal"
    assert res.selected_options == [1, 0]


def test_fixed_theta_diagnostics_do_not_change_solution() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 5.0, 10.0], [0.0, 4.0, 9.0], [0.0, 6.0, 11.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0, 7.0, 18.0], [0.0, 9.0, 17.0]],
        capacity=15.0,
    )
    base = solve_fixed_theta_bnb(instance, 0.0, FixedThetaBNBConfig(use_cache=True))
    diag = solve_fixed_theta_bnb(
        instance,
        0.0,
        FixedThetaBNBConfig(
            use_cache=True,
            collect_diagnostics=True,
            profile_timing=True,
            max_diagnostic_nodes=100,
            diagnostic_sample_rate=1,
        ),
    )
    assert diag.status == base.status
    assert diag.objective_value == pytest.approx(base.objective_value)
    assert diag.selected_options == base.selected_options
    assert diag.diagnostics["nodes_created"] >= diag.nodes_explored
    assert diag.diagnostics["nodes_live_peak"] >= 0
    assert diag.diagnostics["time_root_lp"] >= 0.0
    assert diag.diagnostics["time_node_lp_total"] >= 0.0
    assert "node_bound_gap_samples" in diag.diagnostics or diag.nodes_explored == 0


def test_below_hull_option_survives_diagnostic_mode() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 5.0, 10.0], [0.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0]],
        capacity=5.0,
    )
    res = solve_fixed_theta_bnb(
        instance,
        0.0,
        FixedThetaBNBConfig(
            use_cache=True,
            objective_cutoff=1.0,
            use_cutoff_pruning=True,
            use_greedy_incumbent=False,
            collect_diagnostics=True,
            profile_timing=True,
        ),
    )
    assert res.status == "optimal"
    assert res.selected_options == [1, 0]
    assert res.objective_value == pytest.approx(8.0)


def test_fast_node_bound_matches_reference_on_small_nodes() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 4.0, 9.0], [0.0, 3.0, 8.0], [0.0, 5.0, 10.0]],
        value_groups=[[1.0, 8.0, 17.0], [2.0, 7.0, 15.0], [1.0, 9.0, 18.0]],
        capacity=12.0,
    )
    cache = _build_fixed_theta_cache(instance, 0.0, 1e-9)
    nodes = [
        _Node(fixed=(-1, -1, -1), used_cost=0.0, fixed_value=0.0, upper_bound=float("inf"), depth=0, tie_key=()),
        _Node(fixed=(1, -1, -1), used_cost=4.0, fixed_value=8.0, upper_bound=float("inf"), depth=1, tie_key=(0, 1)),
        _Node(fixed=(-1, 1, -1), used_cost=3.0, fixed_value=7.0, upper_bound=float("inf"), depth=1, tie_key=(1, 1)),
    ]
    for node in nodes:
        ref = _compute_bound(
            node,
            cache,
            cache.data.capacity,
            1e-9,
            use_fast=False,
            need_solution=True,
            diagnostics={},
        )
        fast = _compute_bound(
            node,
            cache,
            cache.data.capacity,
            1e-9,
            use_fast=True,
            need_solution=True,
            diagnostics={},
        )
        assert fast.infeasible == ref.infeasible
        assert fast.upper_bound == pytest.approx(ref.upper_bound, abs=1e-8)
        assert (fast.lp_solution is None) == (ref.lp_solution is None)


def test_fast_and_reference_fixed_theta_bnb_match() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 5.0, 10.0], [0.0, 4.0, 9.0], [0.0, 6.0, 11.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0, 7.0, 18.0], [0.0, 9.0, 17.0]],
        capacity=15.0,
    )
    ref = solve_fixed_theta_bnb(instance, 0.0, FixedThetaBNBConfig(use_fast_residual_lp_bound=False))
    fast = solve_fixed_theta_bnb(instance, 0.0, FixedThetaBNBConfig(use_fast_residual_lp_bound=True))
    assert fast.status == ref.status
    assert fast.objective_value == pytest.approx(ref.objective_value)
    assert fast.selected_options == ref.selected_options


def test_bound_cache_matches_no_cache_mode() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 5.0, 10.0], [0.0, 4.0, 9.0], [0.0, 6.0, 11.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0, 7.0, 18.0], [0.0, 9.0, 17.0]],
        capacity=15.0,
    )
    no_cache = solve_fixed_theta_bnb(instance, 0.0, FixedThetaBNBConfig(use_bound_cache=False))
    cached = solve_fixed_theta_bnb(instance, 0.0, FixedThetaBNBConfig(use_bound_cache=True, collect_diagnostics=True))
    assert cached.status == no_cache.status
    assert cached.objective_value == pytest.approx(no_cache.objective_value)
    assert cached.selected_options == no_cache.selected_options


def test_cheap_prebound_and_min_cost_checks_are_safe() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0, 9.0, 19.0]],
        capacity=10.0,
    )
    brute = brute_force_fixed_theta(instance, 0.0)
    res = solve_fixed_theta_bnb(
        instance,
        0.0,
        FixedThetaBNBConfig(
            use_fast_residual_lp_bound=True,
            use_cheap_prebound=True,
            use_min_cost_infeasibility_check=True,
            objective_cutoff=10.0,
            use_cutoff_pruning=True,
        ),
    )
    assert res.status == "optimal"
    assert res.objective_value == pytest.approx(brute.objective_value)

    infeasible = _instance_from_fixed_theta_points(
        cost_groups=[[5.0], [6.0]],
        value_groups=[[1.0], [1.0]],
        capacity=10.0,
    )
    inf_res = solve_fixed_theta_bnb(
        infeasible,
        0.0,
        FixedThetaBNBConfig(use_fast_residual_lp_bound=True, use_min_cost_infeasibility_check=True),
    )
    assert inf_res.status == "infeasible"


@pytest.mark.parametrize(
    "branching_rule",
    [
        "fractional_item_then_spread",
        "largest_hull_jump",
        "largest_cost_spread",
        "largest_value_spread",
        "max_option_count_or_entropy",
        "tight_capacity_hybrid",
        "strong_branching_lite",
    ],
)
def test_branching_rules_match_bruteforce(branching_rule: str) -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 4.0, 9.0], [0.0, 3.0, 8.0], [0.0, 5.0, 10.0]],
        value_groups=[[1.0, 8.0, 17.0], [2.0, 7.0, 15.0], [1.0, 9.0, 18.0]],
        capacity=12.0,
    )
    brute = brute_force_fixed_theta(instance, 0.0)
    res = solve_fixed_theta_bnb(
        instance,
        0.0,
        FixedThetaBNBConfig(
            branching_rule=branching_rule,
            strong_branching_candidates=3,
            strong_branching_depth_limit=2,
            strong_branching_max_children=12,
            collect_diagnostics=True,
        ),
    )
    assert res.status == "optimal"
    assert res.objective_value == pytest.approx(brute.objective_value)
    assert res.validation_flags["capacity_feasible"]


@pytest.mark.parametrize("child_ordering", ["value_desc", "cost_asc", "density_desc", "bound_promise"])
def test_child_ordering_does_not_change_optimum(child_ordering: str) -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 5.0, 10.0], [0.0, 4.0, 9.0], [0.0, 6.0, 11.0]],
        value_groups=[[0.0, 8.0, 20.0], [0.0, 7.0, 18.0], [0.0, 9.0, 17.0]],
        capacity=15.0,
    )
    brute = brute_force_fixed_theta(instance, 0.0)
    res = solve_fixed_theta_bnb(instance, 0.0, FixedThetaBNBConfig(child_ordering=child_ordering))
    assert res.status == "optimal"
    assert res.objective_value == pytest.approx(brute.objective_value)


def test_strong_branching_diagnostics_are_recorded() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 4.0, 9.0], [0.0, 3.0, 8.0], [0.0, 5.0, 10.0]],
        value_groups=[[1.0, 8.0, 17.0], [2.0, 7.0, 15.0], [1.0, 9.0, 18.0]],
        capacity=12.0,
    )
    res = solve_fixed_theta_bnb(
        instance,
        0.0,
        FixedThetaBNBConfig(
            branching_rule="strong_branching_lite",
            strong_branching_candidates=2,
            strong_branching_depth_limit=3,
            strong_branching_max_children=10,
            collect_diagnostics=True,
            profile_timing=True,
        ),
    )
    assert res.status == "optimal"
    assert res.diagnostics["branching_rule_used"] == "strong_branching_lite"
    assert res.diagnostics["strong_branching_count"] >= 0
    assert res.diagnostics["strong_branching_candidates_evaluated"] >= 0


def test_local_incumbent_improvement_is_feasible() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 1.0, 4.0], [0.0, 1.0, 4.0], [0.0, 1.0, 4.0]],
        value_groups=[[0.0, 1.0, 9.0], [0.0, 1.0, 8.0], [0.0, 1.0, 7.0]],
        capacity=8.0,
    )
    res = solve_fixed_theta_bnb(
        instance,
        0.0,
        FixedThetaBNBConfig(use_local_incumbent_improvement=True, local_search_max_passes=3, collect_diagnostics=True),
    )
    assert res.status == "optimal"
    assert res.validation_flags["capacity_feasible"]
    assert res.selected_options is not None


def test_two_item_local_incumbent_neighborhood_is_feasible_and_exact() -> None:
    instance = _instance_from_fixed_theta_points(
        cost_groups=[[0.0, 2.0, 6.0], [0.0, 3.0, 7.0], [0.0, 2.0, 5.0]],
        value_groups=[[0.0, 5.0, 12.0], [0.0, 6.0, 15.0], [0.0, 4.0, 9.0]],
        capacity=10.0,
    )
    brute = brute_force_fixed_theta(instance, 0.0)
    res = solve_fixed_theta_bnb(
        instance,
        0.0,
        FixedThetaBNBConfig(
            use_local_incumbent_improvement=True,
            local_search_neighborhood="single_two",
            local_search_max_pair_evaluations=1000,
            collect_diagnostics=True,
        ),
    )
    assert res.status == "optimal"
    assert res.objective_value == pytest.approx(brute.objective_value)
    assert res.validation_flags["capacity_feasible"]
    assert res.diagnostics["local_incumbent_pair_evaluations"] >= 0
