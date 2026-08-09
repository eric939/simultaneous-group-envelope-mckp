from __future__ import annotations

import numpy as np
import pytest

from research.interval_baselines import (
    ThetaIntervalOracle,
    build_small_instance,
    compact_lp_support,
    cutting_plane_support,
    disjunctive_lp_support,
    feasible_assignments,
    fixed_theta_lp_support,
    integer_support,
    maximum_completion_certificate,
    minimal_group_conflicts,
)
from robust_mckp.exact_bnb import build_full_theta_candidates


def test_relaxation_supports_are_valid_and_ordered() -> None:
    instance = build_small_instance(seed=77, n=5, m=3, gamma=2)
    feasible = feasible_assignments(instance)
    conflicts = minimal_group_conflicts(instance, feasible)
    direction = np.array([option.value for group in instance.items for option in group])
    integer = integer_support(instance, direction, feasible)
    compact, _ = compact_lp_support(instance, direction)
    disjunctive = disjunctive_lp_support(instance, direction)
    conflict, _ = compact_lp_support(instance, direction, conflicts)
    assert compact + 1e-7 >= integer
    assert disjunctive + 1e-7 >= integer
    assert conflict + 1e-7 >= integer
    assert compact + 1e-7 >= disjunctive
    assert compact + 1e-7 >= conflict


def test_minimal_conflicts_have_no_feasible_extension_and_are_minimal() -> None:
    instance = build_small_instance(seed=91, n=5, m=3, gamma=2)
    feasible = feasible_assignments(instance)
    conflicts = minimal_group_conflicts(instance, feasible)
    assert conflicts
    for conflict in conflicts:
        assert not any(all(selection[i] == j for i, j in conflict) for selection in feasible)
        for dropped in range(len(conflict)):
            subset = conflict[:dropped] + conflict[dropped + 1 :]
            assert any(all(selection[i] == j for i, j in subset) for selection in feasible)


def test_singleton_interval_lagrangian_bound_is_valid() -> None:
    instance = build_small_instance(seed=123, n=6, m=3, gamma=2)
    oracle = ThetaIntervalOracle(instance)
    for idx, theta in enumerate(build_full_theta_candidates(instance)):
        fixed = fixed_theta_lp_support(
            instance,
            theta,
            np.array([option.value for group in instance.items for option in group]),
        )
        interval = oracle.bound(idx, idx)
        assert interval.upper_bound + 1e-6 >= fixed


def test_parent_interval_bound_covers_child_fixed_theta_lps() -> None:
    instance = build_small_instance(seed=456, n=5, m=3, gamma=2)
    oracle = ThetaIntervalOracle(instance)
    parent = oracle.bound(0, len(oracle.thetas) - 1)
    direction = np.array([option.value for group in instance.items for option in group])
    exact_max = max(fixed_theta_lp_support(instance, theta, direction) for theta in oracle.thetas)
    assert parent.upper_bound + 1e-6 >= exact_max


def test_completion_oracle_matches_enumeration() -> None:
    instance = build_small_instance(seed=987, n=5, m=3, gamma=2)
    feasible = feasible_assignments(instance)
    for partial in [((0, 1),), ((0, 2), (3, 1)), ((1, 2), (2, 2), (4, 2))]:
        oracle_feasible = maximum_completion_certificate(instance, partial) >= -1e-8
        enumerated_feasible = any(
            all(selection[i] == j for i, j in partial) for selection in feasible
        )
        assert oracle_feasible == enumerated_feasible


def test_heuristic_conflict_cuts_preserve_integer_bound() -> None:
    instance = build_small_instance(seed=1357, n=6, m=3, gamma=2)
    direction = np.array([option.value for group in instance.items for option in group])
    integer = integer_support(instance, direction)
    result = cutting_plane_support(
        instance,
        direction,
        seed=42,
        max_rounds=5,
        samples_per_round=20,
    )
    assert result["final_bound"] + 1e-7 >= integer
    assert result["final_bound"] <= result["initial_bound"] + 1e-7
