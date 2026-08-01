#!/usr/bin/env python3
"""Controlled end-to-end exact-solver comparison for the envelope oracle."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import platform
import sys
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from robust_mckp.exact_bnb import GlobalThetaBNBConfig, solve_global_theta_bnb
from research.integrated_exact_solver import IntervalExactConfig, solve_interval_exact
from research.structural_feasibility_study import solve_scip_with_conflicts
from research.benchmark_instances import build_benchmark_instance


FAMILIES = ("dense_frontier", "correlated_risk", "near_tie", "many_breakpoints")


def _geomean(values: list[float]) -> float:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    return float(math.exp(np.mean(np.log(positive)))) if positive else float("nan")


def _method_result(instance, method: str, time_limit: float) -> dict:
    if method in {"envelope", "clique"}:
        result = solve_interval_exact(
            instance,
            IntervalExactConfig(
                bound_kind=method,
                tolerance=1e-7,
                time_limit_seconds=time_limit,
            ),
        )
        return {
            "status": result.status,
            "certified": result.status == "optimal",
            "objective": result.objective_value,
            "upper_bound": result.upper_bound,
            "relative_gap": result.relative_gap,
            "seconds": result.runtime_seconds,
            "nodes": result.total_nodes,
            "theta_integer_solves": result.theta_integer_solves,
            "interval_bound_evaluations": result.interval_bound_evaluations,
            "interval_splits": result.interval_splits,
            "thresholds_covered_by_prunes": result.thresholds_covered_by_prunes,
        }
    if method == "enumeration":
        result = solve_global_theta_bnb(
            instance,
            GlobalThetaBNBConfig(
                tolerance=1e-7,
                time_limit_seconds=time_limit,
                theta_order="lp_bound_desc",
                use_objective_cutoff=True,
            ),
        )
        return {
            "status": result.status,
            "certified": result.status == "optimal",
            "objective": result.objective_value,
            "upper_bound": result.upper_bound,
            "relative_gap": result.relative_gap,
            "seconds": result.total_runtime_seconds,
            "nodes": result.total_nodes_explored,
            "theta_integer_solves": result.theta_count_solved_optimal,
            "interval_bound_evaluations": 0,
            "interval_splits": 0,
            "thresholds_covered_by_prunes": result.theta_count_pruned_by_bound,
        }
    if method == "scip":
        result = solve_scip_with_conflicts(instance, [], time_limit=time_limit)
        return {
            "status": result["status"],
            "certified": bool(result["certified"]),
            "objective": float(result["objective"]),
            "upper_bound": float(result["dual_bound"]),
            "relative_gap": float(result["gap"]),
            "seconds": float(result["runtime_seconds"]),
            "nodes": int(result["nodes"]),
            "theta_integer_solves": 0,
            "interval_bound_evaluations": 0,
            "interval_splits": 0,
            "thresholds_covered_by_prunes": 0,
        }
    raise ValueError(method)


def run_campaign(
    output_dir: Path,
    *,
    sizes: list[int],
    seeds: list[int],
    repetitions: int,
    time_limit: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = ("envelope", "clique", "enumeration", "scip")
    raw_rows: list[dict] = []
    summary_rows: list[dict] = []

    for family, n, seed in itertools.product(FAMILIES, sizes, seeds):
        instance = build_benchmark_instance(family, n, 6, max(1, int(math.sqrt(n))), seed)
        by_method: dict[str, list[dict]] = {method: [] for method in methods}
        for repeat in range(repetitions):
            rotation = (repeat + seed + n) % len(methods)
            order = methods[rotation:] + methods[:rotation]
            for order_index, method in enumerate(order):
                result = _method_result(instance, method, time_limit)
                by_method[method].append(result)
                raw_rows.append(
                    {
                        "family": family,
                        "groups": n,
                        "seed": seed,
                        "repeat": repeat,
                        "order_index": order_index,
                        "method": method,
                        **result,
                    }
                )

        row: dict[str, object] = {"family": family, "groups": n, "seed": seed}
        for method in methods:
            runs = by_method[method]
            times = [float(run["seconds"]) for run in runs]
            representative = min(
                runs, key=lambda run: abs(float(run["seconds"]) - median(times))
            )
            row[f"{method}_seconds"] = float(median(times))
            for key in (
                "status",
                "certified",
                "objective",
                "upper_bound",
                "relative_gap",
                "nodes",
                "theta_integer_solves",
                "interval_bound_evaluations",
                "interval_splits",
                "thresholds_covered_by_prunes",
            ):
                row[f"{method}_{key}"] = representative[key]
        row["clique_over_envelope_speedup"] = float(row["clique_seconds"]) / max(
            float(row["envelope_seconds"]), 1e-12
        )
        row["enumeration_over_envelope_speedup"] = float(
            row["enumeration_seconds"]
        ) / max(float(row["envelope_seconds"]), 1e-12)
        row["scip_over_envelope_speedup"] = float(row["scip_seconds"]) / max(
            float(row["envelope_seconds"]), 1e-12
        )
        certified_objectives = [
            float(row[f"{method}_objective"])
            for method in methods
            if bool(row[f"{method}_certified"])
        ]
        row["maximum_certified_objective_difference"] = (
            max(certified_objectives) - min(certified_objectives)
            if certified_objectives
            else float("nan")
        )
        summary_rows.append(row)

        _write_csv(output_dir / "exact_integration_raw_timings.csv", raw_rows)
        _write_csv(output_dir / "exact_integration.csv", summary_rows)

    joint = [
        row
        for row in summary_rows
        if bool(row["envelope_certified"]) and bool(row["clique_certified"])
    ]
    summary = {
        "instances": len(summary_rows),
        "design": {
            "families": list(FAMILIES),
            "sizes": sizes,
            "seeds": seeds,
            "repetitions": repetitions,
            "time_limit_seconds": time_limit,
        },
        "envelope_certification_rate": float(
            np.mean([bool(row["envelope_certified"]) for row in summary_rows])
        ),
        "clique_certification_rate": float(
            np.mean([bool(row["clique_certified"]) for row in summary_rows])
        ),
        "enumeration_certification_rate": float(
            np.mean([bool(row["enumeration_certified"]) for row in summary_rows])
        ),
        "scip_certification_rate": float(
            np.mean([bool(row["scip_certified"]) for row in summary_rows])
        ),
        "joint_envelope_clique_instances": len(joint),
        "geomean_clique_over_envelope_speedup_joint": _geomean(
            [float(row["clique_over_envelope_speedup"]) for row in joint]
        ),
        "geomean_enumeration_over_envelope_speedup": _geomean(
            [float(row["enumeration_over_envelope_speedup"]) for row in summary_rows]
        ),
        "geomean_scip_over_envelope_speedup": _geomean(
            [float(row["scip_over_envelope_speedup"]) for row in summary_rows]
        ),
        "maximum_certified_objective_difference": float(
            max(
                [
                    float(row["maximum_certified_objective_difference"])
                    for row in summary_rows
                    if math.isfinite(float(row["maximum_certified_objective_difference"]))
                ]
                or [0.0]
            )
        ),
    }
    (output_dir / "exact_integration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
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
    (output_dir / "environment_exact_integration.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", default="30,60")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=10.0)
    args = parser.parse_args()
    summary = run_campaign(
        args.output_dir,
        sizes=[int(value) for value in args.sizes.split(",") if value],
        seeds=[int(value) for value in args.seeds.split(",") if value],
        repetitions=args.repetitions,
        time_limit=args.time_limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
