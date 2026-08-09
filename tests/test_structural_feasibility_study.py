from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from research.structural_feasibility_study import (
    FixedThetaLPOracle,
    adaptive_clique_interval_bound,
    bounded_theta_clique_lp,
    build_integer_instance,
    bounded_theta_strong_lp,
    cdd_facets,
    conflict_row_reduced,
    exact_conflict_separation,
    exact_robust_certificate,
    reduced_vertex,
    trivial_rows_reduced,
)
from research.interval_baselines import (
    _flatten,
    compact_lp_support,
    feasible_assignments,
    maximum_completion_certificate,
    minimal_group_conflicts,
    fixed_theta_lp_support,
)
from robust_mckp import Option, PricingInstance


def test_cdd_exact_facets_contain_simplex_bounds() -> None:
    instance = build_integer_instance(seed=12, n=4, m=3, gamma=2)
    assignments = list(itertools.product(*(range(len(group)) for group in instance.items)))
    feasible = [row for row in assignments if exact_robust_certificate(instance, row) >= 0]
    vertices = sorted(set(reduced_vertex(instance, row) for row in feasible))
    facets = set(cdd_facets(vertices))
    assert facets
    assert facets.intersection(trivial_rows_reduced(instance))
    for facet in facets:
        for vertex in vertices:
            assert facet[0] + sum(a * y for a, y in zip(facet[1:], vertex)) >= 0


def test_conflict_rows_are_valid_for_all_feasible_vertices() -> None:
    instance = build_integer_instance(seed=99, n=5, m=3, gamma=2)
    feasible = feasible_assignments(instance)
    for conflict in minimal_group_conflicts(instance, feasible):
        row = conflict_row_reduced(instance, conflict)
        for selection in feasible:
            vertex = reduced_vertex(instance, selection)
            assert row[0] + sum(a * y for a, y in zip(row[1:], vertex)) >= 0


def test_exact_separator_returns_a_valid_violated_conflict_or_proves_none() -> None:
    instance = build_integer_instance(seed=202, n=5, m=3, gamma=2)
    direction = np.array([option.value for group in instance.items for option in group])
    _bound, x = compact_lp_support(instance, direction)
    assert x is not None
    result = exact_conflict_separation(instance, x, time_limit=5.0)
    if result.conflict is not None:
        assert maximum_completion_certificate(instance, result.conflict) < -1e-7
        sizes, offsets, _, _ = _flatten(instance)
        violation = 1.0 - sum(1.0 - x[offsets[i] + j] for i, j in result.conflict)
        if result.objective < 1.0 - 1e-7:
            assert violation > 1e-7


def test_bounded_theta_lp_matches_fixed_theta_lp_on_singleton_interval() -> None:
    instance = build_integer_instance(seed=303, n=5, m=3, gamma=2)
    direction = np.array([option.value for group in instance.items for option in group])
    for theta in [0.0, 2.0, 5.0]:
        generic, _ = bounded_theta_strong_lp(instance, theta, theta)
        fixed = fixed_theta_lp_support(instance, theta, direction)
        assert generic == pytest.approx(fixed, abs=1e-7)


@pytest.mark.parametrize("seed", range(8))
def test_fast_fixed_theta_oracle_matches_reference(seed: int) -> None:
    instance = build_integer_instance(seed=500 + seed, n=7, m=4, gamma=2)
    direction = np.array([option.value for group in instance.items for option in group])
    oracle = FixedThetaLPOracle(instance)
    thetas = sorted(
        {0.0, *(abs(float(option.uncertainty)) for group in instance.items for option in group)}
    )
    for theta in thetas:
        expected = fixed_theta_lp_support(instance, theta, direction)
        assert oracle.value(theta) == pytest.approx(expected, abs=2e-7, rel=2e-10)


def test_fixed_theta_oracle_keeps_small_positive_hull_segment() -> None:
    instance = PricingInstance(
        items=[
            [
                Option(0.0, 2.5e-10, 0.0),
                Option(100.0, -2.5e-10, 0.0),
            ]
        ],
        gamma=0,
    )
    assert FixedThetaLPOracle(instance).value(0.0) == pytest.approx(50.0)


def test_fixed_theta_oracle_preserves_objective_cancellation_tail() -> None:
    instance = PricingInstance(
        items=[
            [
                Option(0.0, -24.0, 68719476736.0),
                Option(0.0625, 268435456.0, 0.0078125),
            ],
            [
                Option(0.0, -0.0234375, 128.0),
                Option(-1099511627776.0, -4.0, 0.0),
            ],
            [Option(-3.814697265625e-6, 0.0, 9.5367431640625e-7)],
        ],
        gamma=0,
    )
    expected = 0.062496185302734375
    assert FixedThetaLPOracle(instance).value(0.0) == pytest.approx(
        expected, abs=0.0, rel=0.0
    )


def test_fixed_theta_oracle_uses_exact_cost_hull_topology() -> None:
    instance = PricingInstance(
        items=[
            [
                Option(0.0, -9.536743164060332e-7, 0.0),
                Option(0.0, -1.7053025658263084e-13, 0.0),
                Option(-4.835703278458517e24, 2.90142196707511e25, 0.0),
            ]
        ],
        gamma=0,
    )
    expected = -2.842170943043847e-14
    observed = FixedThetaLPOracle(instance).value(0.0)
    assert observed <= expected
    assert observed == pytest.approx(expected, abs=1e-28, rel=0.0)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_fixed_theta_oracle_rejects_aggregate_objective_overflow(
    sign: float,
) -> None:
    maximum = float(np.finfo(float).max)
    instance = PricingInstance(
        items=[
            [Option(sign * maximum, 0.0, 0.0)],
            [Option(sign * maximum, 0.0, 0.0)],
        ],
        gamma=0,
    )
    with pytest.raises(ValueError, match="rescale objective"):
        FixedThetaLPOracle(instance)


def test_bounded_theta_clique_lp_matches_fixed_theta_and_compact_lp() -> None:
    instance = build_integer_instance(seed=404, n=5, m=3, gamma=2)
    direction = np.array([option.value for group in instance.items for option in group])
    for theta in [0.0, 2.0, 5.0]:
        clique, _seconds, returned_theta = bounded_theta_clique_lp(
            instance, theta, theta
        )
        fixed = fixed_theta_lp_support(instance, theta, direction)
        assert returned_theta == pytest.approx(theta, abs=1e-9)
        assert clique == pytest.approx(fixed, abs=1e-7)

    thetas = [abs(option.uncertainty) for group in instance.items for option in group]
    clique, _seconds, _returned_theta = bounded_theta_clique_lp(
        instance, 0.0, max(thetas)
    )
    compact, _ = compact_lp_support(instance, direction)
    assert clique == pytest.approx(compact, abs=1e-7)


@pytest.mark.parametrize("seed", range(5))
def test_bounded_theta_clique_lp_bounds_every_fixed_lp_in_interval(seed: int) -> None:
    instance = build_integer_instance(seed=430 + seed, n=6, m=4, gamma=2)
    direction = np.array([option.value for group in instance.items for option in group])
    thetas = sorted(
        {0.0, *(abs(float(option.uncertainty)) for group in instance.items for option in group)}
    )
    oracle = FixedThetaLPOracle(instance)

    for lower_index, upper_index in [(0, len(thetas) // 2), (1, len(thetas) - 2)]:
        lower = thetas[lower_index]
        upper = thetas[upper_index]
        bound, _seconds, returned_theta = bounded_theta_clique_lp(
            instance, lower, upper
        )
        fixed_maximum = max(
            oracle.value(theta)
            for theta in thetas
            if lower <= theta <= upper
        )
        assert lower - 1e-9 <= returned_theta <= upper + 1e-9
        assert bound + 2e-7 >= fixed_maximum


def test_bounded_theta_clique_lp_retries_unknown_highs_status(monkeypatch) -> None:
    instance = build_integer_instance(seed=405, n=3, m=2, gamma=1)
    calls: list[str] = []

    def fake_linprog(*, method: str, **_kwargs):
        calls.append(method)
        if method == "highs":
            return SimpleNamespace(
                success=False,
                status=4,
                message="unknown despite feasible primal",
                fun=-999.0,
                x=np.zeros(10),
            )
        return SimpleNamespace(
            success=True,
            status=0,
            message="optimal",
            fun=-17.0,
            x=np.zeros(10),
        )

    monkeypatch.setattr("research.structural_feasibility_study.opt.linprog", fake_linprog)
    bound, _seconds, _theta = bounded_theta_clique_lp(instance, 0.0, 5.0)
    assert calls == ["highs", "highs-ds"]
    assert bound == pytest.approx(17.0)


def test_bounded_theta_clique_lp_passes_sparse_matrices_to_highs(monkeypatch) -> None:
    instance = build_integer_instance(seed=407, n=4, m=3, gamma=2)
    seen: list[tuple[bool, bool]] = []

    def fake_linprog(*, A_ub, A_eq, **_kwargs):
        seen.append((sparse.issparse(A_ub), sparse.issparse(A_eq)))
        nvars = A_ub.shape[1]
        return SimpleNamespace(
            success=True,
            status=0,
            message="optimal",
            fun=-17.0,
            x=np.zeros(nvars),
            nit=3,
        )

    monkeypatch.setattr("research.structural_feasibility_study.opt.linprog", fake_linprog)
    bounded_theta_clique_lp(instance, 0.0, 5.0)
    assert seen == [(True, True)]


def test_adaptive_clique_uses_fixed_lp_directly_for_singletons(monkeypatch) -> None:
    instance = build_integer_instance(seed=406, n=3, m=2, gamma=1)
    interval_calls: list[tuple[float, float]] = []

    monkeypatch.setattr(
        "research.structural_feasibility_study.build_full_theta_candidates",
        lambda _instance: [0.0, 1.0, 2.0, 3.0],
    )

    def fake_interval(_instance, lower: float, upper: float, **_kwargs):
        interval_calls.append((lower, upper))
        value = 101.0 if (lower, upper) == (0.0, 3.0) else 100.5
        return value, 0.0, lower, {
            "matrix_nnz": 0,
            "matrix_storage_bytes": 0,
            "solver_iterations": 0,
        }

    monkeypatch.setattr(
        "research.structural_feasibility_study.bounded_theta_clique_lp",
        fake_interval,
    )
    monkeypatch.setattr(
        "research.structural_feasibility_study.FixedThetaLPOracle.value",
        lambda _self, _theta: 100.0,
    )
    result = adaptive_clique_interval_bound(
        instance,
        relative_tolerance=0.0,
        time_limit=1.0,
    )
    assert result["status"] == "exact"
    assert result["interval_lp_evaluations"] == 3
    assert all(lower < upper for lower, upper in interval_calls)
