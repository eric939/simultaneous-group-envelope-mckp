#!/usr/bin/env python3
"""Publication-grade experiments for the v4 group-envelope contribution.

The experimental unit is an instance.  Timing repetitions are used only to
estimate that instance's runtime and are never treated as independent samples.
Method order is balanced within each instance. The complete protocol and
success gates below are serialized with every phase.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robust_mckp import Option, PricingInstance  # noqa: E402
from robust_mckp.exact_bnb import (  # noqa: E402
    build_full_theta_candidates,
    compute_fixed_theta_lp_upper_bound,
)
from research.compressed_interval_oracle import (  # noqa: E402
    CompressedThetaIntervalOracle,
)
from research.bound_dominance import exact_minimax_epigraph_bound  # noqa: E402
from research.novelty_go_no_go import (  # noqa: E402
    ThetaIntervalOracle,
    _write_csv,
    full_theta_lp_scan,
)
from research.structural_feasibility_study import (  # noqa: E402
    adaptive_clique_interval_bound,
    adaptive_interval_bound,
    bounded_theta_clique_lp,
    FixedThetaLPOracle,
)
from research.benchmark_instances import build_benchmark_instance  # noqa: E402
from scripts.run_pathC_semisynthetic_application import (  # noqa: E402
    build_portfolio,
    read_segment_calibration,
)


FAMILIES = ("dense_frontier", "correlated_risk", "near_tie", "many_breakpoints")
RELATIVE_TOLERANCE = 1e-6
APPLICATION_SPEC = {
    "dataset": "UCI Online Retail (id 352)",
    "sizes": [360, 720, 1440],
    "seeds": [61, 62, 63],
    "menu_size": 12,
    "gamma_rule": "floor(sqrt(n))",
    "repeats": 3,
    "time_limit_seconds": 60.0,
}
EXTERNAL_KNAPSACK_SPEC = {
    "source": "Gersing, Buesing, and Koster robust-knapsack benchmark archive",
    "record_doi": "10.5281/zenodo.7419028",
    "archive_url": "https://zenodo.org/api/records/7940637/files/RobustKnapsack.zip/content",
    "archive_sha256": "8571b3e545607415a38a39dc506b21bd891b6a22ce252e42a1622a5a5f451818",
    "archive_bytes": 234397168,
    "sizes_and_seeds": {"1000": [31, 32, 33], "5000": [41, 42, 43], "10000": [51, 52, 53]},
    "repeats": 2,
    "time_limit_seconds": 90.0,
    "transformation": (
        "Each binary item becomes a two-option group. Nominal profits and weights "
        "are retained; the published objective-deviation vector is transparently "
        "transferred to resource deviations for this model-compatible stress test."
    ),
}

# This object is intentionally static.  Every output directory contains an
# exact copy plus its SHA-256 digest before numerical work begins.
PROTOCOL = {
    "version": "v4-publication-20260801-certified-minimax",
    "statistical_unit": "instance",
    "timing_estimator": "median wall time within instance",
    "method_order": "alternating paired blocks within instance",
    "threads": 1,
    "thread_environment": {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    },
    "relative_tolerance": RELATIVE_TOLERANCE,
    "generator_seed_namespace": "v4|family|n|m|Gamma|seed",
    "validation": {"instances": 40, "seed_start": 1000},
    "kernel": {
        "sizes": [90, 180, 360, 720],
        "seeds": [21, 22, 23],
        "families": list(FAMILIES),
        "menu_size": 6,
        "gamma_rule": "floor(sqrt(n))",
        "repeats": 5,
        "lambda_multipliers": [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    },
    "common_trace": {
        "sizes": [360, 720, 1440],
        "seeds": [51, 52],
        "families": list(FAMILIES),
        "menu_size": 6,
        "gamma_rule": "floor(sqrt(n))",
        "repeats": 3,
        "dyadic_depth": 2,
    },
    "primary": {
        "sizes": [360, 720, 1440],
        "seeds": [11, 12, 13, 14, 15],
        "families": list(FAMILIES),
        "menu_size": 6,
        "gamma_rule": "floor(sqrt(n))",
        "repeats": 5,
        "time_limit_seconds": 60.0,
    },
    "robustness": {
        "n": 720,
        "seeds": [31, 32, 33],
        "families": list(FAMILIES),
        "repeats": 3,
        "time_limit_seconds": 60.0,
        "configurations": [
            {"name": "sparse_budget", "menu_size": 6, "gamma_rule": "one"},
            {"name": "wide_menu", "menu_size": 12, "gamma_rule": "sqrt"},
            {"name": "large_budget", "menu_size": 6, "gamma_rule": "twenty_percent"},
        ],
    },
    "stress": {
        "sizes": [2880, 5760],
        "seeds": [41],
        "families": list(FAMILIES),
        "menu_size": 6,
        "gamma_rule": "floor(sqrt(n))",
        "repeats": 2,
        "time_limit_seconds": 90.0,
    },
    "application": APPLICATION_SPEC,
    "external_knapsack": EXTERNAL_KNAPSACK_SPEC,
    "gates": {
        "validation_max_absolute_error": 2e-6,
        "validation_minimum_slack": -2e-6,
        "validation_certificate_max_violation": 2e-6,
        "validation_max_scaled_certificate_gap": 1.1e-8,
        "kernel_geomean_speedup_n_ge_360": 3.0,
        "kernel_identity_max_absolute_error": 2e-6,
        "primary_compressed_tolerance_rate": 1.0,
        "primary_joint_tolerance_rate": 1.0,
        "primary_geomean_speedup": 2.0,
        "primary_bootstrap_ci_lower": 1.5,
        "primary_win_rate": 0.8,
        "primary_root_dominance_rate": 0.95,
        "primary_families_with_median_speedup_gt_one": 4,
        "robustness_compressed_tolerance_rate": 1.0,
        "robustness_each_configuration_median_speedup": 1.0,
    },
}


def geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or np.any(array <= 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires finite positive values")
    return float(np.exp(np.mean(np.log(array))))


def stratified_bootstrap_geomean_ci(
    values: Sequence[float],
    strata: Sequence[str],
    *,
    confidence: float = 0.95,
    draws: int = 10_000,
    seed: int = 20260719,
) -> tuple[float, float]:
    values_array = np.asarray(values, dtype=float)
    strata_array = np.asarray(strata, dtype=object)
    if values_array.size != strata_array.size or values_array.size == 0:
        raise ValueError("values and strata must have the same nonzero length")
    if np.any(values_array <= 0.0):
        raise ValueError("bootstrap geometric mean requires positive values")
    groups = [np.flatnonzero(strata_array == label) for label in sorted(set(strata))]
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        estimates[draw] = geometric_mean(values_array[sampled])
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def summarize_paired_rows(
    rows: Sequence[dict],
    *,
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 20260719,
) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty paired experiment")
    speedups = [float(row["adaptive_speedup"]) for row in rows]
    families = [str(row["family"]) for row in rows]
    cells = [f"{row['family']}|n={int(row['n'])}" for row in rows]
    ci_low, ci_high = stratified_bootstrap_geomean_ci(
        speedups,
        cells,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    wins = sum(value > 1.0 for value in speedups)
    root_dominance = []
    for row in rows:
        clique = float(row["clique_root_bound"])
        compressed = float(row["compressed_root_bound"])
        root_dominance.append(
            compressed <= clique + 1e-7 * max(1.0, abs(clique))
        )
    family_medians = {
        family: float(
            statistics.median(
                float(row["adaptive_speedup"])
                for row in rows
                if str(row["family"]) == family
            )
        )
        for family in sorted(set(families))
    }
    size_geomeans = {
        str(size): geometric_mean(
            float(row["adaptive_speedup"])
            for row in rows
            if int(row["n"]) == size
        )
        for size in sorted({int(row["n"]) for row in rows})
    }
    return {
        "instances": len(rows),
        "bootstrap_strata": "family_x_size",
        "geomean_speedup": geometric_mean(speedups),
        "geomean_speedup_ci95": [ci_low, ci_high],
        "median_speedup": float(np.median(speedups)),
        "win_rate": wins / len(rows),
        "wins": wins,
        "sign_test_pvalue": float(
            sum(math.comb(len(rows), k) for k in range(wins, len(rows) + 1))
            / (2 ** len(rows))
        ),
        "compressed_tolerance_rate": float(
            np.mean(
                [float(row["compressed_relative_gap"]) <= RELATIVE_TOLERANCE for row in rows]
            )
        ),
        "joint_tolerance_rate": float(
            np.mean(
                [
                    float(row["compressed_relative_gap"]) <= RELATIVE_TOLERANCE
                    and float(row["clique_relative_gap"]) <= RELATIVE_TOLERANCE
                    for row in rows
                ]
            )
        ),
        "root_dominance_rate": float(np.mean(root_dominance)),
        "family_median_speedup": family_medians,
        "families_with_median_speedup_gt_one": sum(
            value > 1.0 for value in family_medians.values()
        ),
        "size_geomean_speedup": size_geomeans,
        "median_compressed_theta_fraction": float(
            np.median([float(row["compressed_theta_fraction"]) for row in rows])
        ),
    }


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _protocol_bytes() -> bytes:
    return json.dumps(PROTOCOL, indent=2, sort_keys=True).encode("utf-8")


def initialize_output(output_dir: Path, command: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_bytes = _protocol_bytes()
    (output_dir / "protocol.json").write_bytes(protocol_bytes)
    environment = {
        "command": command,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    _json_dump(output_dir / f"environment_{command}.json", environment)


def _gamma(n: int, rule: str) -> int:
    if rule in {"sqrt", "floor(sqrt(n))"}:
        return max(1, int(math.sqrt(n)))
    if rule == "one":
        return 1
    if rule == "twenty_percent":
        return max(1, int(math.floor(0.2 * n)))
    raise ValueError(f"unknown gamma rule: {rule}")


def _time_call(function: Callable[[], dict]) -> tuple[float, dict]:
    gc.collect()
    start = time.perf_counter()
    result = function()
    return time.perf_counter() - start, result


def _paired_repetitions(
    compressed: Callable[[], dict],
    clique: Callable[[], dict],
    *,
    repeats: int,
    parity: int,
) -> tuple[list[float], list[dict], list[float], list[dict]]:
    compressed_times: list[float] = []
    compressed_results: list[dict] = []
    clique_times: list[float] = []
    clique_results: list[dict] = []
    for repeat in range(repeats):
        order = ("compressed", "clique") if (repeat + parity) % 2 == 0 else ("clique", "compressed")
        for method in order:
            elapsed, result = _time_call(compressed if method == "compressed" else clique)
            if method == "compressed":
                compressed_times.append(elapsed)
                compressed_results.append(result)
            else:
                clique_times.append(elapsed)
                clique_results.append(result)
    return compressed_times, compressed_results, clique_times, clique_results


def summarize_timing_stability(raw_rows: Sequence[dict]) -> dict:
    by_instance_method: dict[tuple[str, str], list[float]] = {}
    by_block: dict[tuple[str, int], dict[str, float]] = {}
    by_method_order: dict[tuple[str, str], list[float]] = {}
    for row in raw_rows:
        instance = str(row["instance"])
        method = str(row["method"])
        seconds = float(row["seconds"])
        by_instance_method.setdefault((instance, method), []).append(seconds)
        by_block.setdefault((instance, int(row["repeat"])), {})[method] = seconds
        by_method_order.setdefault((method, str(row["execution_order"])), []).append(seconds)
    cvs = {"compressed": [], "clique": []}
    for (_instance, method), values in by_instance_method.items():
        if len(values) > 1 and float(np.mean(values)) > 0.0:
            cvs[method].append(float(np.std(values, ddof=1) / np.mean(values)))
    block_speedups = [
        values["clique"] / values["compressed"]
        for values in by_block.values()
        if {"compressed", "clique"}.issubset(values)
    ]
    order_ratios = {}
    for method in ("compressed", "clique"):
        first = by_method_order.get((method, "first"), [])
        second = by_method_order.get((method, "second"), [])
        order_ratios[method] = (
            float(np.median(first) / np.median(second)) if first and second else float("nan")
        )
    return {
        "repeat_blocks": len(block_speedups),
        "repeat_block_wins": int(sum(value > 1.0 for value in block_speedups)),
        "minimum_repeat_block_speedup": float(min(block_speedups)),
        "geomean_repeat_block_speedup": geometric_mean(block_speedups),
        "median_within_instance_cv": {
            method: float(np.median(values)) for method, values in cvs.items()
        },
        "maximum_within_instance_cv": {
            method: float(max(values)) for method, values in cvs.items()
        },
        "median_first_over_second_time": order_ratios,
    }


def _irregular_validation_instance(seed: int) -> PricingInstance:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(8, 21))
    items: list[list[Option]] = []
    deviation_pool = np.asarray([0.0, 0.25, 0.5, 1.0, 1.0, 2.0, 4.0, 8.0])
    for _ in range(n):
        size = int(rng.integers(2, 11))
        group = [
            Option(
                value=float(rng.normal(0.0, 10.0)),
                margin=float(rng.uniform(2.0, 8.0)),
                uncertainty=0.0,
            )
        ]
        for _ in range(1, size):
            group.append(
                Option(
                    value=float(rng.normal(0.0, 15.0)),
                    margin=float(rng.normal(0.0, 5.0)),
                    uncertainty=float(rng.choice(deviation_pool) + rng.choice([0.0, 0.0, rng.uniform(0.0, 0.05)])),
                )
            )
        items.append(group)
    return PricingInstance(
        items=items,
        gamma=int(rng.integers(0, n + 1)),
        name=f"publication_validation_s{seed}_n{n}",
    )


def run_validation(output_dir: Path) -> dict:
    spec = PROTOCOL["validation"]
    rows: list[dict] = []
    for seed in range(int(spec["seed_start"]), int(spec["seed_start"]) + int(spec["instances"])):
        instance = _irregular_validation_instance(seed)
        dense = ThetaIntervalOracle(instance)
        compressed = CompressedThetaIntervalOracle(instance)
        rng = np.random.default_rng(seed + 90_000)
        lambdas = np.concatenate(([0.0], np.geomspace(1e-4, 1e4, 16)))
        value_error = 0.0
        direct_error = 0.0
        for lam in lambdas:
            expected = dense.values_at_lambda(float(lam), 0, len(dense.thetas) - 1)
            actual = compressed.values_at_lambda(float(lam), 0, len(compressed.thetas) - 1)
            finite = np.isfinite(expected) & np.isfinite(actual)
            if np.any(finite):
                value_error = max(value_error, float(np.max(np.abs(expected[finite] - actual[finite]))))
            direct = np.asarray(
                [
                    sum(
                        max(
                            option.value
                            + float(lam)
                            * (option.margin - max(0.0, abs(option.uncertainty) - float(theta)))
                            for option in group
                        )
                        for group in instance.items
                    )
                    - float(lam) * instance.gamma * float(theta)
                    for theta in compressed.thetas
                ],
                dtype=float,
            )
            feasible = np.isfinite(actual)
            if np.any(feasible):
                direct_error = max(direct_error, float(np.max(np.abs(actual[feasible] - direct[feasible]))))
        interval_error = 0.0
        certificate_violation = 0.0
        maximum_scaled_certificate_gap = 0.0
        for _ in range(3):
            lo, hi = sorted(rng.integers(0, len(dense.thetas), size=2).tolist())
            dense_bound = dense.bound(int(lo), int(hi))
            compressed_bound = compressed.bound(int(lo), int(hi))
            interval_error = max(
                interval_error,
                abs(dense_bound.upper_bound - compressed_bound.upper_bound),
            )
            exact_bound = exact_minimax_epigraph_bound(instance, int(lo), int(hi))
            if math.isfinite(exact_bound.upper_bound):
                certificate_violation = max(
                    certificate_violation,
                    exact_bound.upper_bound - compressed_bound.upper_bound,
                    compressed_bound.lower_bound - exact_bound.upper_bound,
                    compressed_bound.upper_bound
                    - exact_bound.upper_bound
                    - compressed_bound.optimality_gap,
                )
                maximum_scaled_certificate_gap = max(
                    maximum_scaled_certificate_gap,
                    compressed_bound.optimality_gap
                    / max(1.0, abs(compressed_bound.upper_bound)),
                )
        compressed_root = compressed.bound(0, len(compressed.thetas) - 1)
        fixed_oracle = FixedThetaLPOracle(instance)
        fixed_lp_error = 0.0
        for theta in compressed.thetas:
            reference = compute_fixed_theta_lp_upper_bound(instance, float(theta))
            expected = float(reference.lp_upper_bound) if reference.lp_feasible else float("-inf")
            actual = fixed_oracle.value(float(theta))
            if math.isfinite(expected) and math.isfinite(actual):
                fixed_lp_error = max(fixed_lp_error, abs(expected - actual))
            elif expected != actual:
                fixed_lp_error = float("inf")
        scan, scan_seconds, feasible_thresholds = full_theta_lp_scan(instance)
        slack = compressed_root.upper_bound - scan
        rows.append(
            {
                "instance": instance.name,
                "seed": seed,
                "n": instance.n_items,
                "options": sum(len(group) for group in instance.items),
                "gamma": instance.gamma,
                "theta_count": len(compressed.thetas),
                "feasible_thresholds": feasible_thresholds,
                "value_identity_error": value_error,
                "direct_formula_error": direct_error,
                "interval_bound_error": interval_error,
                "certificate_violation": max(0.0, certificate_violation),
                "maximum_scaled_certificate_gap": maximum_scaled_certificate_gap,
                "fixed_lp_error": fixed_lp_error,
                "root_validity_slack": slack,
                "full_scan_seconds": scan_seconds,
            }
        )
        _write_csv(output_dir / "validation.csv", rows)
    maximum_error = max(
        max(
            float(row["value_identity_error"]),
            float(row["direct_formula_error"]),
            float(row["interval_bound_error"]),
            float(row["fixed_lp_error"]),
        )
        for row in rows
    )
    minimum_slack = min(float(row["root_validity_slack"]) for row in rows)
    maximum_certificate_violation = max(
        float(row["certificate_violation"]) for row in rows
    )
    maximum_scaled_certificate_gap = max(
        float(row["maximum_scaled_certificate_gap"]) for row in rows
    )
    gates = PROTOCOL["gates"]
    summary = {
        "instances": len(rows),
        "maximum_absolute_error": maximum_error,
        "minimum_validity_slack": minimum_slack,
        "maximum_certificate_violation": maximum_certificate_violation,
        "maximum_scaled_certificate_gap": maximum_scaled_certificate_gap,
        "gates": {
            "identity": maximum_error <= float(gates["validation_max_absolute_error"]),
            "validity": minimum_slack >= float(gates["validation_minimum_slack"]),
            "certificate": maximum_certificate_violation
            <= float(gates["validation_certificate_max_violation"]),
            "certificate_tolerance": maximum_scaled_certificate_gap
            <= float(gates["validation_max_scaled_certificate_gap"]),
        },
    }
    summary["pass"] = all(summary["gates"].values())
    _json_dump(output_dir / "validation_summary.json", summary)
    return summary


def _dense_storage_bytes(oracle: ThetaIntervalOracle) -> int:
    return int(
        oracle.thetas.nbytes
        + oracle.capacities.nbytes
        + sum(array.nbytes for array in oracle.group_costs)
        + sum(array.nbytes for array in oracle.group_values)
    )


def _compressed_storage_bytes(oracle: CompressedThetaIntervalOracle) -> int:
    arrays = [
        oracle.thetas,
        oracle.capacities,
        oracle._group_sizes,
        oracle._segment_lo,
        oracle._segment_hi,
        oracle._segment_group,
        oracle._segment_last_saturated,
        oracle.multiplier_grid,
        *oracle._fixed_value_cache.values(),
        *oracle.values,
        *oracle.margins,
        *oracle.deviations,
        *oracle._sorted_values,
        *oracle._sorted_margins,
        *oracle._sorted_deviations,
    ]
    return int(sum(array.nbytes for array in arrays))


def _kernel_call(instance: PricingInstance, method: str, multipliers: Sequence[float]) -> dict:
    construct_start = time.perf_counter()
    oracle = ThetaIntervalOracle(instance) if method == "dense" else CompressedThetaIntervalOracle(instance)
    construct_seconds = time.perf_counter() - construct_start
    query_start = time.perf_counter()
    traces = [
        oracle.values_at_lambda(float(lam) * float(oracle.lambda_scale), 0, len(oracle.thetas) - 1)
        for lam in multipliers
    ]
    query_seconds = time.perf_counter() - query_start
    storage_bytes = _dense_storage_bytes(oracle) if method == "dense" else _compressed_storage_bytes(oracle)
    return {
        "construct_seconds": construct_seconds,
        "query_seconds": query_seconds,
        "storage_bytes": storage_bytes,
        "traces": traces,
        "lambda_scale": float(oracle.lambda_scale),
        "theta_count": len(oracle.thetas),
    }


def run_kernel(output_dir: Path) -> dict:
    spec = PROTOCOL["kernel"]
    rows: list[dict] = []
    raw: list[dict] = []
    multipliers = [float(value) for value in spec["lambda_multipliers"]]
    for family, n, seed in itertools.product(spec["families"], spec["sizes"], spec["seeds"]):
        instance = build_benchmark_instance(
            str(family), int(n), int(spec["menu_size"]), _gamma(int(n), str(spec["gamma_rule"])), int(seed)
        )
        dense_times, dense_results, compressed_times, compressed_results = [], [], [], []
        for repeat in range(int(spec["repeats"])):
            order = ("dense", "compressed") if (repeat + int(seed)) % 2 == 0 else ("compressed", "dense")
            for order_index, method in enumerate(order):
                elapsed, result = _time_call(lambda method=method: _kernel_call(instance, method, multipliers))
                raw.append(
                    {
                        "instance": instance.name,
                        "family": family,
                        "n": n,
                        "seed": seed,
                        "repeat": repeat,
                        "execution_order": "first" if order_index == 0 else "second",
                        "method": method,
                        "total_seconds": elapsed,
                        "construct_seconds": result["construct_seconds"],
                        "query_seconds": result["query_seconds"],
                    }
                )
                if method == "dense":
                    dense_times.append(elapsed)
                    dense_results.append(result)
                else:
                    compressed_times.append(elapsed)
                    compressed_results.append(result)
        dense_median = float(np.median(dense_times))
        compressed_median = float(np.median(compressed_times))
        dense_query = float(np.median([result["query_seconds"] for result in dense_results]))
        compressed_query = float(np.median([result["query_seconds"] for result in compressed_results]))
        # The multiplier scale is algebraically identical but may differ at roundoff.
        # Compare at the dense scale independently of the timed calls.
        dense_check = ThetaIntervalOracle(instance)
        compressed_check = CompressedThetaIntervalOracle(instance)
        identity_error = 0.0
        for multiplier in multipliers:
            lam = multiplier * dense_check.lambda_scale
            expected = dense_check.values_at_lambda(lam, 0, len(dense_check.thetas) - 1)
            actual = compressed_check.values_at_lambda(lam, 0, len(compressed_check.thetas) - 1)
            finite = np.isfinite(expected) & np.isfinite(actual)
            if np.any(finite):
                identity_error = max(identity_error, float(np.max(np.abs(expected[finite] - actual[finite]))))
        row = {
            "instance": instance.name,
            "family": family,
            "n": n,
            "seed": seed,
            "options": sum(len(group) for group in instance.items),
            "theta_count": len(dense_check.thetas),
            "repeats": spec["repeats"],
            "dense_total_seconds": dense_median,
            "compressed_total_seconds": compressed_median,
            "total_speedup": dense_median / max(compressed_median, 1e-12),
            "dense_query_seconds": dense_query,
            "compressed_query_seconds": compressed_query,
            "query_speedup": dense_query / max(compressed_query, 1e-12),
            "dense_storage_bytes": _dense_storage_bytes(dense_check),
            "compressed_storage_bytes": _compressed_storage_bytes(compressed_check),
            "storage_ratio": _dense_storage_bytes(dense_check) / max(_compressed_storage_bytes(compressed_check), 1),
            "identity_error": identity_error,
        }
        rows.append(row)
        _write_csv(output_dir / "kernel.csv", rows)
        _write_csv(output_dir / "kernel_raw_timings.csv", raw)
    large = [row for row in rows if int(row["n"]) >= 360]
    maximum_error = max(float(row["identity_error"]) for row in rows)
    summary = {
        "instances": len(rows),
        "geomean_total_speedup_n_ge_360": geometric_mean(float(row["total_speedup"]) for row in large),
        "geomean_query_speedup_n_ge_360": geometric_mean(float(row["query_speedup"]) for row in large),
        "median_storage_ratio_n_720": float(np.median([float(row["storage_ratio"]) for row in rows if int(row["n"]) == 720])),
        "maximum_identity_error": maximum_error,
        "size_geomean_total_speedup": {
            str(size): geometric_mean(float(row["total_speedup"]) for row in rows if int(row["n"]) == size)
            for size in sorted({int(row["n"]) for row in rows})
        },
    }
    gates = PROTOCOL["gates"]
    summary["gates"] = {
        "speed": summary["geomean_total_speedup_n_ge_360"] >= float(gates["kernel_geomean_speedup_n_ge_360"]),
        "identity": maximum_error <= float(gates["kernel_identity_max_absolute_error"]),
    }
    summary["pass"] = all(summary["gates"].values())
    _json_dump(output_dir / "kernel_summary.json", summary)
    return summary


def _dyadic_intervals(count: int, depth: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    frontier = [(0, count - 1)]
    for _ in range(depth + 1):
        next_frontier: list[tuple[int, int]] = []
        for lo, hi in frontier:
            if lo >= hi:
                continue
            intervals.append((lo, hi))
            mid = (lo + hi) // 2
            next_frontier.extend([(lo, mid), (mid + 1, hi)])
        frontier = next_frontier
    return intervals


def _common_trace_call(
    instance: PricingInstance,
    method: str,
    intervals: Sequence[tuple[int, int]],
) -> dict:
    bounds: list[float] = []
    certificate_gaps: list[float] = []
    certificates: list[bool] = []
    matrix_nnz = 0
    matrix_storage_bytes = 0
    solver_iterations = 0
    if method == "compressed":
        oracle = CompressedThetaIntervalOracle(instance)
        for lo, hi in intervals:
            result = oracle.bound(lo, hi)
            bounds.append(float(result.upper_bound))
            certificate_gaps.append(float(result.optimality_gap))
            certificates.append(bool(result.certified))
    else:
        thetas = np.asarray(build_full_theta_candidates(instance), dtype=float)
        for lo, hi in intervals:
            bound, _seconds, _theta, diagnostics = bounded_theta_clique_lp(
                instance,
                float(thetas[lo]),
                float(thetas[hi]),
                return_diagnostics=True,
            )
            bounds.append(float(bound))
            matrix_nnz += int(diagnostics["matrix_nnz"])
            matrix_storage_bytes += int(diagnostics["matrix_storage_bytes"])
            solver_iterations += int(diagnostics["solver_iterations"])
    return {
        "bounds": bounds,
        "certificate_gaps": certificate_gaps,
        "certificates": certificates,
        "matrix_nnz": matrix_nnz,
        "matrix_storage_bytes": matrix_storage_bytes,
        "solver_iterations": solver_iterations,
    }


def run_common_trace(output_dir: Path) -> dict:
    spec = PROTOCOL["common_trace"]
    rows: list[dict] = []
    raw: list[dict] = []
    for family, n, seed in itertools.product(
        spec["families"], spec["sizes"], spec["seeds"]
    ):
        instance = build_benchmark_instance(
            str(family),
            int(n),
            int(spec["menu_size"]),
            _gamma(int(n), str(spec["gamma_rule"])),
            int(seed),
        )
        theta_count = len(CompressedThetaIntervalOracle(instance).thetas)
        intervals = _dyadic_intervals(theta_count, int(spec["dyadic_depth"]))
        compressed_times, compressed_results, clique_times, clique_results = _paired_repetitions(
            lambda: _common_trace_call(instance, "compressed", intervals),
            lambda: _common_trace_call(instance, "clique", intervals),
            repeats=int(spec["repeats"]),
            parity=int(seed),
        )
        compressed_median = float(np.median(compressed_times))
        clique_median = float(np.median(clique_times))
        compressed = compressed_results[
            min(range(len(compressed_times)), key=lambda i: abs(compressed_times[i] - compressed_median))
        ]
        clique = clique_results[
            min(range(len(clique_times)), key=lambda i: abs(clique_times[i] - clique_median))
        ]
        dominance = [
            left <= right + 1e-7 * max(1.0, abs(right))
            for left, right in zip(compressed["bounds"], clique["bounds"])
        ]
        rows.append(
            {
                "instance": instance.name,
                "family": family,
                "n": n,
                "seed": seed,
                "theta_count": theta_count,
                "intervals": len(intervals),
                "compressed_seconds": compressed_median,
                "clique_seconds": clique_median,
                "speedup": clique_median / max(compressed_median, 1e-12),
                "compressed_bound_le_clique_rate": float(np.mean(dominance)),
                "all_compressed_bounds_certified": all(compressed["certificates"]),
                "maximum_compressed_certificate_gap": max(
                    compressed["certificate_gaps"], default=0.0
                ),
                "maximum_compressed_scaled_certificate_gap": max(
                    (
                        gap / max(1.0, abs(bound))
                        for gap, bound in zip(
                            compressed["certificate_gaps"], compressed["bounds"]
                        )
                    ),
                    default=0.0,
                ),
                "clique_matrix_nnz": clique["matrix_nnz"],
                "clique_matrix_storage_bytes": clique["matrix_storage_bytes"],
                "clique_solver_iterations": clique["solver_iterations"],
            }
        )
        for method, times in (("compressed", compressed_times), ("clique", clique_times)):
            for repeat, seconds in enumerate(times):
                compressed_first = (repeat + int(seed)) % 2 == 0
                method_first = compressed_first == (method == "compressed")
                raw.append(
                    {
                        "instance": instance.name,
                        "family": family,
                        "n": n,
                        "seed": seed,
                        "method": method,
                        "repeat": repeat,
                        "execution_order": "first" if method_first else "second",
                        "seconds": seconds,
                    }
                )
        _write_csv(output_dir / "common_trace.csv", rows)
        _write_csv(output_dir / "common_trace_raw_timings.csv", raw)
    summary = {
        "instances": len(rows),
        "interval_evaluations": int(sum(int(row["intervals"]) for row in rows)),
        "geomean_speedup": geometric_mean(float(row["speedup"]) for row in rows),
        "wins": int(sum(float(row["speedup"]) > 1.0 for row in rows)),
        "bound_dominance_rate": float(
            np.mean([float(row["compressed_bound_le_clique_rate"]) for row in rows])
        ),
        "all_compressed_bounds_certified": all(
            bool(row["all_compressed_bounds_certified"]) for row in rows
        ),
        "maximum_scaled_certificate_gap": max(
            float(row["maximum_compressed_scaled_certificate_gap"]) for row in rows
        ),
    }
    _json_dump(output_dir / "common_trace_summary.json", summary)
    return summary


def _compressed_adaptive(instance: PricingInstance, time_limit: float) -> dict:
    oracle = CompressedThetaIntervalOracle(instance)
    remaining_time = max(0.0, time_limit - oracle.preprocessing_seconds)
    result = adaptive_interval_bound(
        instance,
        oracle,
        relative_tolerance=RELATIVE_TOLERANCE,
        time_limit=remaining_time,
    )
    result["preprocessing_seconds"] = oracle.preprocessing_seconds
    result["theta_count"] = len(oracle.thetas)
    return result


def _adaptive_instance_row(
    instance: PricingInstance,
    *,
    family: str,
    n: int,
    seed: int,
    repeats: int,
    time_limit: float,
    configuration: str,
) -> tuple[dict, list[dict]]:
    compressed_times, compressed_results, clique_times, clique_results = _paired_repetitions(
        lambda: _compressed_adaptive(instance, time_limit),
        lambda: adaptive_clique_interval_bound(
            instance,
            relative_tolerance=RELATIVE_TOLERANCE,
            time_limit=time_limit,
        ),
        repeats=repeats,
        parity=seed,
    )
    def representative(times: list[float], results: list[dict]) -> dict:
        median = float(np.median(times))
        index = min(range(len(times)), key=lambda i: abs(times[i] - median))
        reference = results[index]
        for result in results:
            if result["status"] != reference["status"]:
                raise AssertionError("repeated runs produced inconsistent statuses")
            for field in ("lower_bound", "upper_bound", "relative_gap", "root_upper_bound"):
                if not math.isclose(
                    float(result[field]),
                    float(reference[field]),
                    rel_tol=1e-10,
                    abs_tol=1e-8,
                ):
                    raise AssertionError(f"repeated runs disagree on {field}")
        return reference

    compressed = representative(compressed_times, compressed_results)
    clique = representative(clique_times, clique_results)
    theta_count = int(compressed["theta_count"])
    compressed_median = float(np.median(compressed_times))
    clique_median = float(np.median(clique_times))
    scale = max(1.0, abs(float(clique["lower_bound"])))
    lower_difference = abs(float(compressed["lower_bound"]) - float(clique["lower_bound"])) / scale
    row = {
        "instance": instance.name,
        "configuration": configuration,
        "family": family,
        "n": n,
        "seed": seed,
        "menu_size": max(len(group) for group in instance.items),
        "gamma": instance.gamma,
        "theta_count": theta_count,
        "repeats": repeats,
        "compressed_seconds": compressed_median,
        "clique_seconds": clique_median,
        "adaptive_speedup": clique_median / max(compressed_median, 1e-12),
        "compressed_status": compressed["status"],
        "clique_status": clique["status"],
        "compressed_relative_gap": compressed["relative_gap"],
        "clique_relative_gap": clique["relative_gap"],
        "compressed_root_bound": compressed["root_upper_bound"],
        "clique_root_bound": clique["root_upper_bound"],
        "compressed_theta_evaluations": compressed["theta_lp_evaluations"],
        "clique_theta_evaluations": clique["theta_lp_evaluations"],
        "compressed_theta_fraction": compressed["theta_lp_evaluations"] / theta_count,
        "clique_theta_fraction": clique["theta_lp_evaluations"] / theta_count,
        "compressed_splits": compressed["interval_splits"],
        "compressed_interval_bound_evaluations": compressed[
            "interval_bound_evaluations"
        ],
        "compressed_multiplier_evaluations": compressed[
            "multiplier_evaluations"
        ],
        "compressed_all_interval_bounds_certified": compressed[
            "all_interval_bounds_certified"
        ],
        "compressed_maximum_oracle_optimality_gap": compressed[
            "maximum_oracle_optimality_gap"
        ],
        "compressed_maximum_scaled_oracle_optimality_gap": compressed[
            "maximum_scaled_oracle_optimality_gap"
        ],
        "clique_splits": clique["interval_splits"],
        "clique_interval_lp_evaluations": clique["interval_lp_evaluations"],
        "clique_interval_matrix_nnz": clique["interval_matrix_nnz"],
        "clique_interval_matrix_storage_bytes": clique["interval_matrix_storage_bytes"],
        "clique_solver_iterations": clique["interval_solver_iterations"],
        "final_lower_bound_relative_difference": lower_difference,
    }
    raw: list[dict] = []
    for method, times, results in (
        ("compressed", compressed_times, compressed_results),
        ("clique", clique_times, clique_results),
    ):
        for repeat, (elapsed, result) in enumerate(zip(times, results)):
            compressed_first = (repeat + seed) % 2 == 0
            execution_order = (
                "first"
                if (method == "compressed") == compressed_first
                else "second"
            )
            raw.append(
                {
                    "instance": instance.name,
                    "configuration": configuration,
                    "family": family,
                    "n": n,
                    "seed": seed,
                    "method": method,
                    "repeat": repeat,
                    "execution_order": execution_order,
                    "seconds": elapsed,
                    "status": result["status"],
                    "relative_gap": result["relative_gap"],
                    "lower_bound": result["lower_bound"],
                    "upper_bound": result["upper_bound"],
                    "root_upper_bound": result["root_upper_bound"],
                    "theta_lp_evaluations": result["theta_lp_evaluations"],
                    "interval_splits": result["interval_splits"],
                }
            )
    return row, raw


def _primary_gates(summary: dict) -> dict:
    gates = PROTOCOL["gates"]
    return {
        "compressed_tolerance": summary["compressed_tolerance_rate"] >= float(gates["primary_compressed_tolerance_rate"]),
        "joint_tolerance": summary["joint_tolerance_rate"] >= float(gates["primary_joint_tolerance_rate"]),
        "geomean_speedup": summary["geomean_speedup"] >= float(gates["primary_geomean_speedup"]),
        "bootstrap_ci_lower": summary["geomean_speedup_ci95"][0] > float(gates["primary_bootstrap_ci_lower"]),
        "win_rate": summary["win_rate"] >= float(gates["primary_win_rate"]),
        "root_dominance": summary["root_dominance_rate"] >= float(gates["primary_root_dominance_rate"]),
        "family_breadth": summary["families_with_median_speedup_gt_one"] >= int(gates["primary_families_with_median_speedup_gt_one"]),
    }


def run_primary(output_dir: Path) -> dict:
    spec = PROTOCOL["primary"]
    rows: list[dict] = []
    raw: list[dict] = []
    for family, n, seed in itertools.product(spec["families"], spec["sizes"], spec["seeds"]):
        instance = build_benchmark_instance(
            str(family), int(n), int(spec["menu_size"]), _gamma(int(n), str(spec["gamma_rule"])), int(seed)
        )
        row, raw_rows = _adaptive_instance_row(
            instance,
            family=str(family),
            n=int(n),
            seed=int(seed),
            repeats=int(spec["repeats"]),
            time_limit=float(spec["time_limit_seconds"]),
            configuration="primary",
        )
        rows.append(row)
        raw.extend(raw_rows)
        _write_csv(output_dir / "primary.csv", rows)
        _write_csv(output_dir / "primary_raw_timings.csv", raw)
    summary = summarize_paired_rows(rows)
    summary["timing_stability"] = summarize_timing_stability(raw)
    summary["maximum_final_lower_bound_relative_difference"] = max(
        float(row["final_lower_bound_relative_difference"]) for row in rows
    )
    summary["all_interval_bounds_certified"] = all(
        bool(row["compressed_all_interval_bounds_certified"]) for row in rows
    )
    summary["maximum_scaled_oracle_optimality_gap"] = max(
        float(row["compressed_maximum_scaled_oracle_optimality_gap"])
        for row in rows
    )
    summary["maximum_oracle_optimality_gap"] = max(
        float(row["compressed_maximum_oracle_optimality_gap"])
        for row in rows
    )
    summary["gates"] = _primary_gates(summary)
    summary["pass"] = all(summary["gates"].values())
    _json_dump(output_dir / "primary_summary.json", summary)
    return summary


def run_robustness(output_dir: Path) -> dict:
    spec = PROTOCOL["robustness"]
    rows: list[dict] = []
    raw: list[dict] = []
    n = int(spec["n"])
    for config, family, seed in itertools.product(spec["configurations"], spec["families"], spec["seeds"]):
        instance = build_benchmark_instance(
            str(family), n, int(config["menu_size"]), _gamma(n, str(config["gamma_rule"])), int(seed)
        )
        row, raw_rows = _adaptive_instance_row(
            instance,
            family=str(family),
            n=n,
            seed=int(seed),
            repeats=int(spec["repeats"]),
            time_limit=float(spec["time_limit_seconds"]),
            configuration=str(config["name"]),
        )
        rows.append(row)
        raw.extend(raw_rows)
        _write_csv(output_dir / "robustness.csv", rows)
        _write_csv(output_dir / "robustness_raw_timings.csv", raw)
    by_configuration = {
        str(config["name"]): summarize_paired_rows(
            [row for row in rows if row["configuration"] == config["name"]],
            bootstrap_seed=20260719 + index,
        )
        for index, config in enumerate(spec["configurations"])
    }
    gates = PROTOCOL["gates"]
    summary = {
        "instances": len(rows),
        "by_configuration": by_configuration,
        "gates": {
            "compressed_tolerance": all(
                result["compressed_tolerance_rate"] >= float(gates["robustness_compressed_tolerance_rate"])
                for result in by_configuration.values()
            ),
            "configuration_breadth": all(
                result["median_speedup"] > float(gates["robustness_each_configuration_median_speedup"])
                for result in by_configuration.values()
            ),
        },
    }
    summary["pass"] = all(summary["gates"].values())
    _json_dump(output_dir / "robustness_summary.json", summary)
    return summary


def run_stress(output_dir: Path) -> dict:
    spec = PROTOCOL["stress"]
    rows: list[dict] = []
    raw: list[dict] = []
    for family, n, seed in itertools.product(spec["families"], spec["sizes"], spec["seeds"]):
        instance = build_benchmark_instance(
            str(family), int(n), int(spec["menu_size"]), _gamma(int(n), str(spec["gamma_rule"])), int(seed)
        )
        row, raw_rows = _adaptive_instance_row(
            instance,
            family=str(family),
            n=int(n),
            seed=int(seed),
            repeats=int(spec["repeats"]),
            time_limit=float(spec["time_limit_seconds"]),
            configuration="stress",
        )
        rows.append(row)
        raw.extend(raw_rows)
        _write_csv(output_dir / "stress.csv", rows)
        _write_csv(output_dir / "stress_raw_timings.csv", raw)
    summary = summarize_paired_rows(rows)
    _json_dump(output_dir / "stress_summary.json", summary)
    return summary


def run_application(output_dir: Path, calibration_dir: Path) -> dict:
    calibration, source_used = read_segment_calibration(calibration_dir)
    if source_used == "synthetic_default":
        raise ValueError("application panel requires a public-data calibration")
    rows: list[dict] = []
    raw: list[dict] = []
    for n, seed in itertools.product(APPLICATION_SPEC["sizes"], APPLICATION_SPEC["seeds"]):
        portfolio = build_portfolio(
            calibration,
            seed=int(seed),
            n=int(n),
            m=int(APPLICATION_SPEC["menu_size"]),
        )
        base = portfolio.to_instance(_gamma(int(n), str(APPLICATION_SPEC["gamma_rule"])))
        instance = PricingInstance(
            items=base.items,
            gamma=base.gamma,
            name=f"uci_online_retail_n{n}_s{seed}",
        )
        row, raw_rows = _adaptive_instance_row(
            instance,
            family="uci_online_retail",
            n=int(n),
            seed=int(seed),
            repeats=int(APPLICATION_SPEC["repeats"]),
            time_limit=float(APPLICATION_SPEC["time_limit_seconds"]),
            configuration="application",
        )
        rows.append(row)
        raw.extend(raw_rows)
        _write_csv(output_dir / "application.csv", rows)
        _write_csv(output_dir / "application_raw_timings.csv", raw)
    summary = summarize_paired_rows(rows, bootstrap_seed=20260721)
    calibration_file = calibration_dir / "segment_calibration.csv"
    summary.update(
        {
            "dataset": APPLICATION_SPEC["dataset"],
            "calibration_source": source_used,
            "calibration_sha256": hashlib.sha256(calibration_file.read_bytes()).hexdigest(),
            "design": APPLICATION_SPEC,
            "timing_stability": summarize_timing_stability(raw),
        }
    )
    _json_dump(output_dir / "application_summary.json", summary)
    return summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_external_knapsack_archive(path: Path) -> Path:
    """Download and verify the published CC-BY benchmark archive if needed."""

    expected = str(EXTERNAL_KNAPSACK_SPEC["archive_sha256"])
    if path.is_file() and _sha256_file(path) == expected:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(str(EXTERNAL_KNAPSACK_SPEC["archive_url"]), temporary)
    actual = _sha256_file(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"external archive checksum mismatch: expected={expected}, actual={actual}"
        )
    temporary.replace(path)
    return path


def build_external_knapsack_instance(
    archive_path: Path,
    *,
    n: int,
    seed: int,
) -> PricingInstance:
    """Map a published binary knapsack into two-option robust-choice groups.

    If item i is not selected, its margin is capacity/n; if it is selected,
    its margin is capacity/n minus its nominal weight. Summing group margins
    therefore gives capacity minus selected weight. The published deviation
    vector is used as the selected option's resource deviation. This is a
    coefficient-transfer benchmark, not a claim to reproduce the archive's
    original uncertain-objective model.
    """

    stem = f"knapsack_n={int(n)}_seed={int(seed)}"
    nominal_member = f"RobustKnapsack/NominalKnapsacks/{stem}.mps.gz"
    robust_member = f"RobustKnapsack/RobustnessComponents/{stem}.txt"
    with zipfile.ZipFile(archive_path) as archive:
        nominal_text = gzip.decompress(archive.read(nominal_member)).decode("utf-8")
        robust_text = archive.read(robust_member).decode("utf-8")

    objectives: dict[str, float] = {}
    weights: dict[str, float] = {}
    capacity: float | None = None
    section = ""
    headers = {"NAME", "ROWS", "COLUMNS", "RHS", "BOUNDS", "ENDATA"}
    for line in nominal_text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] in headers:
            section = tokens[0]
            continue
        if section == "COLUMNS" and tokens[0] != "MARKER":
            variable = tokens[0]
            for cursor in range(1, len(tokens), 2):
                row, coefficient = tokens[cursor], float(tokens[cursor + 1])
                if row == "OBJ":
                    objectives[variable] = -coefficient
                elif row == "capConstr":
                    weights[variable] = coefficient
        elif section == "RHS":
            for cursor in range(1, len(tokens), 2):
                if tokens[cursor] == "capConstr":
                    capacity = float(tokens[cursor + 1])

    robust_lines = [line.strip() for line in robust_text.splitlines() if line.strip()]
    gamma = int(float(robust_lines[0].split(":", 1)[1]))
    deviations = {
        variable: abs(float(value))
        for variable, value in (line.split(":", 1) for line in robust_lines[1:])
    }
    if capacity is None or len(objectives) != n or set(objectives) != set(weights):
        raise ValueError(f"unexpected published knapsack structure for {stem}")
    names = sorted(objectives, key=lambda name: int(name.removeprefix("x")))
    capacity_share = capacity / n
    items = [
        [
            Option(value=0.0, margin=capacity_share, uncertainty=0.0),
            Option(
                value=float(objectives[name]),
                margin=float(capacity_share - weights[name]),
                uncertainty=float(deviations.get(name, 0.0)),
            ),
        ]
        for name in names
    ]
    return PricingInstance(items=items, gamma=gamma, name=f"rwth_{stem}")


def run_external_knapsack(output_dir: Path, archive_path: Path) -> dict:
    archive = ensure_external_knapsack_archive(archive_path)
    rows: list[dict] = []
    raw: list[dict] = []
    for n_text, seeds in EXTERNAL_KNAPSACK_SPEC["sizes_and_seeds"].items():
        n = int(n_text)
        for seed in seeds:
            instance = build_external_knapsack_instance(
                archive, n=n, seed=int(seed)
            )
            row, raw_rows = _adaptive_instance_row(
                instance,
                family="rwth_knapsack",
                n=n,
                seed=int(seed),
                repeats=int(EXTERNAL_KNAPSACK_SPEC["repeats"]),
                time_limit=float(EXTERNAL_KNAPSACK_SPEC["time_limit_seconds"]),
                configuration="external_coefficient_transfer",
            )
            rows.append(row)
            raw.extend(raw_rows)
            _write_csv(output_dir / "external_knapsack.csv", rows)
            _write_csv(output_dir / "external_knapsack_raw_timings.csv", raw)
    summary = summarize_paired_rows(rows, bootstrap_seed=20260721)
    summary.update(
        {
            "source": EXTERNAL_KNAPSACK_SPEC["source"],
            "record_doi": EXTERNAL_KNAPSACK_SPEC["record_doi"],
            "archive_sha256": _sha256_file(archive),
            "transformation": EXTERNAL_KNAPSACK_SPEC["transformation"],
            "design": EXTERNAL_KNAPSACK_SPEC,
            "timing_stability": summarize_timing_stability(raw),
            "all_interval_bounds_certified": all(
                bool(row["compressed_all_interval_bounds_certified"])
                for row in rows
            ),
            "maximum_scaled_oracle_optimality_gap": max(
                float(row["compressed_maximum_scaled_oracle_optimality_gap"])
                for row in rows
            ),
        }
    )
    _json_dump(output_dir / "external_knapsack_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["protocol", "validate", "kernel", "trace", "primary", "robustness", "stress", "application", "external"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "release",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=ROOT / "results" / "pathC" / "calibration",
    )
    parser.add_argument(
        "--external-archive",
        type=Path,
        default=ROOT / "data_cache" / "RobustKnapsack.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    initialize_output(output_dir, args.command)
    if args.command == "protocol":
        result = {"protocol": PROTOCOL}
    elif args.command == "validate":
        result = run_validation(output_dir)
    elif args.command == "kernel":
        result = run_kernel(output_dir)
    elif args.command == "trace":
        result = run_common_trace(output_dir)
    elif args.command == "primary":
        result = run_primary(output_dir)
    elif args.command == "robustness":
        result = run_robustness(output_dir)
    elif args.command == "stress":
        result = run_stress(output_dir)
    elif args.command == "application":
        result = run_application(output_dir, args.calibration_dir.resolve())
    else:
        result = run_external_knapsack(
            output_dir, args.external_archive.expanduser().resolve()
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
