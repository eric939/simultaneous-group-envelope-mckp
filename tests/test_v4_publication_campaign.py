from __future__ import annotations

import gzip
import math
import zipfile
import numpy as np
import pytest

from robust_mckp import Option, PricingInstance
from research.compressed_interval_oracle import CompressedThetaIntervalOracle
from research.novelty_go_no_go import IntervalBound, build_small_instance
from research.structural_feasibility_study import (
    adaptive_clique_interval_bound,
    adaptive_interval_bound,
)
from research.v4_publication_campaign import (
    PROTOCOL,
    _dyadic_intervals,
    build_external_knapsack_instance,
    geometric_mean,
    summarize_timing_stability,
    stratified_bootstrap_geomean_ci,
    summarize_paired_rows,
)


def test_geometric_mean_rejects_nonpositive_values() -> None:
    assert geometric_mean([1.0, 4.0, 16.0]) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        geometric_mean([1.0, 0.0])


def test_stratified_bootstrap_is_deterministic_and_contains_point_estimate() -> None:
    values = [2.0, 4.0, 3.0, 6.0, 5.0, 10.0]
    strata = ["a", "a", "b", "b", "c", "c"]
    first = stratified_bootstrap_geomean_ci(
        values, strata, confidence=0.95, draws=2000, seed=17
    )
    second = stratified_bootstrap_geomean_ci(
        values, strata, confidence=0.95, draws=2000, seed=17
    )
    assert first == second
    estimate = geometric_mean(values)
    assert first[0] <= estimate <= first[1]


def test_paired_summary_uses_instances_not_timing_repetitions() -> None:
    rows = [
        {
            "family": "a",
            "n": 720,
            "adaptive_speedup": 2.0,
            "compressed_relative_gap": 1e-8,
            "clique_relative_gap": 2e-8,
            "compressed_root_bound": 9.0,
            "clique_root_bound": 10.0,
            "compressed_theta_fraction": 0.25,
        },
        {
            "family": "b",
            "n": 1440,
            "adaptive_speedup": 8.0,
            "compressed_relative_gap": 1e-8,
            "clique_relative_gap": 2e-8,
            "compressed_root_bound": 10.0,
            "clique_root_bound": 10.0,
            "compressed_theta_fraction": 0.50,
        },
    ]
    summary = summarize_paired_rows(rows, bootstrap_draws=1000, bootstrap_seed=9)
    assert summary["instances"] == 2
    assert summary["geomean_speedup"] == pytest.approx(4.0)
    assert summary["win_rate"] == 1.0
    assert summary["joint_tolerance_rate"] == 1.0
    assert summary["root_dominance_rate"] == 1.0
    assert summary["median_compressed_theta_fraction"] == pytest.approx(0.375)
    assert math.isfinite(summary["sign_test_pvalue"])
    assert summary["bootstrap_strata"] == "family_x_size"


def test_protocol_fixes_primary_gates_before_execution() -> None:
    gates = PROTOCOL["gates"]
    assert gates["validation_max_absolute_error"] <= 2e-6
    assert gates["validation_certificate_max_violation"] <= 2e-6
    assert gates["validation_max_scaled_certificate_gap"] <= 1.1e-8
    assert gates["primary_geomean_speedup"] >= 2.0
    assert gates["primary_bootstrap_ci_lower"] > 1.0
    assert gates["primary_compressed_tolerance_rate"] == 1.0
    assert PROTOCOL["external_knapsack"]["record_doi"] == "10.5281/zenodo.7419028"


def test_external_binary_knapsack_maps_to_two_option_groups(tmp_path) -> None:
    nominal = """NAME test
ROWS
 N OBJ
 L capConstr
COLUMNS
    x0 OBJ -10
    x0 capConstr 4
    x1 OBJ -7
    x1 capConstr 3
RHS
    RHS1 capConstr 8
BOUNDS
 BV BND1 x0
 BV BND1 x1
ENDATA
"""
    robustness = "Gamma:1\nx0:2\nx1:1\n"
    archive_path = tmp_path / "benchmark.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "RobustKnapsack/NominalKnapsacks/knapsack_n=2_seed=9.mps.gz",
            gzip.compress(nominal.encode("utf-8")),
        )
        archive.writestr(
            "RobustKnapsack/RobustnessComponents/knapsack_n=2_seed=9.txt",
            robustness,
        )
    instance = build_external_knapsack_instance(archive_path, n=2, seed=9)
    assert instance.gamma == 1
    assert [len(group) for group in instance.items] == [2, 2]
    assert instance.items[0][1].value == 10.0
    assert instance.items[0][1].margin == 0.0
    assert instance.items[1][1].margin == 1.0
    assert instance.items[0][1].uncertainty == 2.0


def test_common_trace_intervals_are_deterministic_and_nonsingleton() -> None:
    intervals = _dyadic_intervals(9, depth=2)
    assert intervals == [(0, 8), (0, 4), (5, 8), (0, 2), (3, 4), (5, 6), (7, 8)]
    assert all(lo < hi for lo, hi in intervals)


def test_timing_stability_uses_paired_repeat_blocks() -> None:
    raw = []
    for repeat, (compressed, clique) in enumerate(((1.0, 3.0), (1.1, 2.2), (0.9, 2.7))):
        for method, seconds, order in (
            ("compressed", compressed, "first" if repeat % 2 == 0 else "second"),
            ("clique", clique, "second" if repeat % 2 == 0 else "first"),
        ):
            raw.append(
                {
                    "instance": "i",
                    "method": method,
                    "repeat": repeat,
                    "seconds": seconds,
                    "execution_order": order,
                }
            )
    summary = summarize_timing_stability(raw)
    assert summary["repeat_blocks"] == 3
    assert summary["repeat_block_wins"] == 3
    assert summary["minimum_repeat_block_speedup"] == pytest.approx(2.0)


def test_adaptive_certificate_retains_tolerance_pruned_upper_bound(monkeypatch) -> None:
    instance = build_small_instance(seed=7, n=4, m=3, gamma=1)

    class FakeOracle:
        thetas = np.arange(4, dtype=float)

        def bound(self, lo: int, hi: int) -> IntervalBound:
            value = 100.2 if (lo, hi) == (0, 3) else 100.05
            return IntervalBound(value, 0.0, 1, 0.0)

        def values_at_lambda(self, _lam: float, lo: int, hi: int) -> np.ndarray:
            return np.zeros(hi - lo + 1)

    monkeypatch.setattr(
        "research.structural_feasibility_study.FixedThetaLPOracle.value",
        lambda _self, _theta: 100.0,
    )
    result = adaptive_interval_bound(
        instance,
        FakeOracle(),
        relative_tolerance=1e-3,
        time_limit=1.0,
    )
    assert result["status"] == "tolerance"
    assert result["lower_bound"] == pytest.approx(100.0)
    assert result["upper_bound"] == pytest.approx(100.05)
    assert result["relative_gap"] == pytest.approx(5e-4)
    assert result["interval_bound_evaluations"] == 3
    assert result["all_interval_bounds_certified"] is False


def test_adaptive_methods_find_feasible_anchor_beyond_root_candidates() -> None:
    instance = PricingInstance(
        items=[
            [
                Option(value=1.0, margin=2.0, uncertainty=5.0),
                Option(value=0.0, margin=0.1, uncertainty=4.0),
                Option(value=0.0, margin=0.4, uncertainty=5.0),
            ],
            [
                Option(value=0.0, margin=2.9, uncertainty=4.0),
                Option(value=0.0, margin=2.6, uncertainty=5.0),
                Option(value=0.0, margin=2.9, uncertainty=7.0),
            ],
            [
                Option(value=1.0, margin=1.3, uncertainty=0.0),
                Option(value=0.0, margin=-0.9, uncertainty=2.0),
                Option(value=2.0, margin=-1.6, uncertainty=1.0),
            ],
        ],
        gamma=1,
        name="hidden_feasible_anchor",
    )
    oracle = CompressedThetaIntervalOracle(instance)
    root_candidates = {0, len(oracle.thetas) - 1, (len(oracle.thetas) - 1) // 2}
    assert all(oracle.capacities[index] < 0.0 for index in root_candidates)
    assert np.any(oracle.capacities >= 0.0)

    compressed = adaptive_interval_bound(
        instance,
        oracle,
        relative_tolerance=1e-8,
        time_limit=2.0,
    )
    clique = adaptive_clique_interval_bound(
        instance,
        relative_tolerance=1e-8,
        time_limit=2.0,
    )
    assert math.isfinite(compressed["lower_bound"])
    assert math.isfinite(clique["lower_bound"])
    assert compressed["lower_bound"] == pytest.approx(clique["lower_bound"])
    assert compressed["relative_gap"] <= 1e-8
    assert clique["relative_gap"] <= 1e-8
    assert compressed["all_interval_bounds_certified"] is True
    assert compressed["maximum_scaled_oracle_optimality_gap"] <= 1.1e-8


def test_adaptive_methods_report_globally_infeasible_family() -> None:
    instance = PricingInstance(
        items=[
            [
                Option(value=1.0, margin=-2.0, uncertainty=0.0),
                Option(value=2.0, margin=-1.0, uncertainty=3.0),
            ]
        ],
        gamma=1,
        name="infeasible_threshold_family",
    )
    compressed = adaptive_interval_bound(
        instance,
        CompressedThetaIntervalOracle(instance),
        relative_tolerance=1e-8,
        time_limit=2.0,
    )
    clique = adaptive_clique_interval_bound(
        instance,
        relative_tolerance=1e-8,
        time_limit=2.0,
    )
    for result in (compressed, clique):
        assert result["status"] == "infeasible"
        assert result["lower_bound"] == float("-inf")
        assert result["upper_bound"] == float("-inf")
        assert result["relative_gap"] == 0.0
