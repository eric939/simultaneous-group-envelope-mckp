#!/usr/bin/env python3
"""Two experiments added in release v2.1.

``lp-direct``: the complete LP-family search with the batched bound and with
the clique-LP bound, against the direct baseline that solves every
fixed-threshold LP with the same LP routine.

``integer``: exact robust MCKP at scale.  The threshold-interval search with
either interval bound and HiGHS fixed-threshold subproblems, direct threshold
enumeration with LP-bound pruning and the same subproblems, and the compact
Bertsimas--Sim MILP solved by HiGHS and by SCIP.

The protocol below is serialized into every output directory before any
numerical work begins.  Every returned decision is re-checked in exact
arithmetic.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robust_mckp import PricingInstance  # noqa: E402
from robust_mckp.certificate import certificate_is_feasible  # noqa: E402
from robust_mckp.exact_bnb import (  # noqa: E402
    _selection_objective_fraction,
    build_full_theta_candidates,
)
from research.benchmark_instances import build_benchmark_instance  # noqa: E402
from research.fixed_theta_highs import (  # noqa: E402
    solve_compact_highs,
    solve_compact_scip,
    solve_fixed_theta_highs,
)
from research.integrated_exact_solver import (  # noqa: E402
    IntervalExactConfig,
    solve_interval_exact,
)
from research.structural_feasibility_study import (  # noqa: E402
    FixedThetaLPOracle,
    adaptive_clique_interval_bound,
)
from research.v4_publication_campaign import (  # noqa: E402
    RELATIVE_TOLERANCE,
    _compressed_adaptive,
    _gamma,
    _write_csv,
    geometric_mean,
)
from scripts.run_pathC_semisynthetic_application import (  # noqa: E402
    build_portfolio,
    read_segment_calibration,
)

FAMILIES = ("dense_frontier", "correlated_risk", "near_tie", "many_breakpoints")
LP_METHODS = ("batched", "clique", "direct")
INTEGER_METHODS = ("envelope", "clique", "direct", "compact_highs", "compact_scip")
CALIBRATION = "results/release/2026-08-09-paper-b-final-r5/uci_calibration"
LOAD_GATE = 2.0

PROTOCOL = {
    "version": "v2.1.1-scale-20261101-frozen",
    "revision_note": "replaces v2.1-scale-20261101-frozen: all integer methods start cold "
                     "(the HullRound heuristic did not check the time limit), each instance "
                     "or run starts only below the load gate, CPU time and search effort "
                     "are recorded; instances and seeds are unchanged",
    "load_gate": {"one_minute_load_average_below": LOAD_GATE, "poll_seconds": 30},
    "statistical_unit": "instance",
    "threads": 1,
    "relative_tolerance_lp_search": RELATIVE_TOLERANCE,
    "lp_direct": {
        "purpose": "complete LP-family search: batched and clique-LP interval bounds "
                   "against solving every fixed-threshold LP with the same LP routine",
        "timing_estimator": "median wall time within instance",
        "method_order": "cyclic rotation of (batched, clique, direct) by repeat + seed",
        "panels": {
            "heldout": {"sizes": [360, 720, 1440], "seeds": [601, 602, 603, 604, 605],
                        "families": list(FAMILIES), "menu_size": 6, "gamma_rule": "sqrt",
                        "repeats": 5, "time_limit_seconds": 60.0},
            "robustness": {"n": 720, "seeds": [31, 32, 33], "families": list(FAMILIES),
                           "repeats": 3, "time_limit_seconds": 60.0,
                           "configurations": [
                               {"name": "sparse_budget", "menu_size": 6, "gamma_rule": "one"},
                               {"name": "wide_menu", "menu_size": 12, "gamma_rule": "sqrt"},
                               {"name": "large_budget", "menu_size": 6,
                                "gamma_rule": "twenty_percent"}]},
            "application": {"sizes": [360, 720, 1440], "seeds": [61, 62, 63], "menu_size": 12,
                            "gamma_rule": "sqrt", "repeats": 3, "time_limit_seconds": 60.0,
                            "calibration": CALIBRATION},
        },
        "agreement": "direct maximum M and each search's final lower bound L satisfy "
                     "L <= M <= L + 1e-6 max(1, |L|) (scaled), up to 1e-9 rounding",
    },
    "integer": {
        "purpose": "exact robust MCKP: threshold-interval search with each bound and HiGHS "
                   "fixed-threshold subproblems, direct enumeration with LP pruning, compact MILP",
        "methods": list(INTEGER_METHODS),
        "structured": {"sizes": [360, 720, 1440], "seeds": [701, 702, 703],
                       "families": list(FAMILIES), "menu_size": 6, "gamma_rule": "sqrt"},
        "application": {"sizes": [360, 720, 1440], "seeds": [61, 62, 63], "menu_size": 12,
                        "gamma_rule": "sqrt", "calibration": CALIBRATION},
        "time_limit_seconds": 120.0,
        "repeats": 1,
        "method_order": "cyclic rotation of the five methods by instance index",
        "mip_relative_gap": 1e-7,
        "search_tolerance": 1e-7,
        "incumbent": "every method starts cold, without a heuristic decision",
        "certified": "status optimal, the decision passes the exact rational check, and the "
                     "wall time is within the limit plus one second (overruns count as time limit)",
        "singleton_intervals": "both threshold searches bound a single-threshold interval by "
                               "its fixed-threshold LP, as the LP-family search does",
    },
}


# ---------------------------------------------------------------------------
# Methods


def direct_lp_scan(instance: PricingInstance, time_limit: float) -> dict:
    """Maximum over every fixed-threshold LP, solved with the search's LP routine."""

    start = time.perf_counter()
    oracle = FixedThetaLPOracle(instance)
    thetas = build_full_theta_candidates(instance)
    best = float("-inf")
    evaluated = 0
    for theta in thetas:
        if time.perf_counter() - start >= time_limit:
            break
        best = max(best, float(oracle.value(float(theta))))
        evaluated += 1
    return {
        "status": "optimal" if evaluated == len(thetas) else "time_limit",
        "value": best,
        "theta_count": len(thetas),
        "theta_lp_evaluations": evaluated,
    }


def direct_integer(instance: PricingInstance, *, time_limit: float, tolerance: float) -> dict:
    """Thresholds in decreasing LP bound; solve each with HiGHS until the bound prunes."""

    start = time.perf_counter()
    elapsed = lambda: time.perf_counter() - start  # noqa: E731
    objective_arrays = [np.asarray([o.value for o in g], dtype=float) for g in instance.items]
    selection, exact_value, incumbent = None, None, float("-inf")
    oracle = FixedThetaLPOracle(instance)
    thetas = build_full_theta_candidates(instance)
    bounds = []
    for theta in thetas:
        if elapsed() >= time_limit:
            return _integer_row("time_limit", selection, incumbent, float("inf"), elapsed(),
                                theta_solves=0, lp_evaluations=len(bounds),
                                theta_count=len(thetas))
        bounds.append((float(oracle.value(float(theta))), float(theta)))
    bounds.sort(key=lambda item: -item[0])
    solves = 0
    upper = incumbent
    status = "optimal"
    for position, (bound, theta) in enumerate(bounds):
        if not math.isfinite(bound):
            break
        if math.isfinite(incumbent) and bound <= incumbent + tolerance * max(1.0, abs(incumbent)):
            break
        remaining = time_limit - elapsed()
        if remaining <= 0.0:
            status, upper = "time_limit", max(upper, bound)
            break
        result = solve_fixed_theta_highs(instance, theta, time_limit=remaining)
        solves += 1
        if result.selected_options is not None and certificate_is_feasible(
            instance, result.selected_options
        ):
            candidate = _selection_objective_fraction(objective_arrays, result.selected_options)
            if exact_value is None or candidate > exact_value:
                exact_value, incumbent = candidate, float(candidate)
                selection = list(map(int, result.selected_options))
        if result.status not in {"optimal", "infeasible"}:
            status = "time_limit"
            upper = max(upper, float(result.upper_bound), incumbent)
            if position + 1 < len(bounds):
                upper = max(upper, bounds[position + 1][0])
            break
    if selection is None and status == "optimal":
        status = "infeasible"
    return _integer_row(status, selection, incumbent, upper if status != "optimal" else incumbent,
                        elapsed(), theta_solves=solves, lp_evaluations=len(bounds),
                        theta_count=len(thetas))


def _integer_row(status, selection, objective, upper, seconds, **extra) -> dict:
    gap = (max(0.0, upper - objective) / max(1.0, abs(objective))
           if selection is not None and math.isfinite(upper) else float("inf"))
    return {"status": status, "certified": status == "optimal" and selection is not None,
            "exactly_feasible": selection is not None, "objective": float(objective),
            "upper_bound": float(upper), "relative_gap": float(gap), "seconds": float(seconds),
            **extra}


def integer_method(instance: PricingInstance, method: str, time_limit: float) -> dict:
    tolerance = float(PROTOCOL["integer"]["search_tolerance"])
    if method in {"envelope", "clique"}:
        r = solve_interval_exact(instance, IntervalExactConfig(
            bound_kind=method, tolerance=tolerance, time_limit_seconds=time_limit,
            fixed_theta_solver="highs", singleton_lp_bound=True,
            use_hullround_incumbent=False))
        return _integer_row(r.status, r.selected_options, r.objective_value, r.upper_bound,
                            r.runtime_seconds, theta_solves=r.theta_integer_solves,
                            interval_bound_evaluations=r.interval_bound_evaluations,
                            bound_seconds=r.bound_seconds,
                            fixed_theta_seconds=r.fixed_theta_seconds,
                            theta_count=r.thresholds_total)
    if method == "direct":
        return direct_integer(instance, time_limit=time_limit, tolerance=tolerance)
    solver = solve_compact_highs if method == "compact_highs" else solve_compact_scip
    r = solver(instance, time_limit=time_limit)
    r.pop("selection", None)
    return r


# ---------------------------------------------------------------------------
# Instances


def _application_instance(calibration, n: int, seed: int, m: int, rule: str) -> PricingInstance:
    portfolio = build_portfolio(calibration, seed=seed, n=n, m=m)
    base = portfolio.to_instance(_gamma(n, rule))
    return PricingInstance(items=base.items, gamma=base.gamma, name=f"uci_online_retail_n{n}_s{seed}")


def lp_instances():
    panels = PROTOCOL["lp_direct"]["panels"]
    h = panels["heldout"]
    for family, n, seed in itertools.product(h["families"], h["sizes"], h["seeds"]):
        yield ("heldout", "heldout", family, n, seed, h["repeats"], h["time_limit_seconds"],
               build_benchmark_instance(family, n, h["menu_size"], _gamma(n, h["gamma_rule"]), seed))
    r = panels["robustness"]
    for config, family, seed in itertools.product(r["configurations"], r["families"], r["seeds"]):
        n = r["n"]
        yield ("robustness", config["name"], family, n, seed, r["repeats"], r["time_limit_seconds"],
               build_benchmark_instance(family, n, config["menu_size"],
                                        _gamma(n, config["gamma_rule"]), seed))
    a = panels["application"]
    calibration, source = read_segment_calibration(ROOT / a["calibration"])
    if source == "synthetic_default":
        raise SystemExit("application panel requires the frozen public-data calibration")
    for n, seed in itertools.product(a["sizes"], a["seeds"]):
        yield ("application", "application", "uci_online_retail", n, seed, a["repeats"],
               a["time_limit_seconds"],
               _application_instance(calibration, n, seed, a["menu_size"], a["gamma_rule"]))


def integer_instances():
    spec = PROTOCOL["integer"]
    s = spec["structured"]
    for family, n, seed in itertools.product(s["families"], s["sizes"], s["seeds"]):
        yield ("structured", family, n, seed,
               build_benchmark_instance(family, n, s["menu_size"], _gamma(n, s["gamma_rule"]), seed))
    a = spec["application"]
    calibration, source = read_segment_calibration(ROOT / a["calibration"])
    if source == "synthetic_default":
        raise SystemExit("application panel requires the frozen public-data calibration")
    for n, seed in itertools.product(a["sizes"], a["seeds"]):
        yield ("application", "uci_online_retail", n, seed,
               _application_instance(calibration, n, seed, a["menu_size"], a["gamma_rule"]))


# ---------------------------------------------------------------------------
# Phases


def _timed(function):
    gc.collect()
    cpu, start = time.process_time(), time.perf_counter()
    result = function()
    return time.perf_counter() - start, time.process_time() - cpu, result


def _wait_for_quiet_machine() -> float:
    """Block until the one-minute load average is below the gate; return it."""
    while (load := os.getloadavg()[0]) >= LOAD_GATE:
        print(f"load {load:.2f} >= {LOAD_GATE}; waiting", flush=True)
        time.sleep(30)
    return load


def run_lp_direct(out: Path, limit: int | None = None) -> None:
    rows, raw = [], []
    for index, (panel, config, family, n, seed, repeats, tl, inst) in enumerate(lp_instances()):
        if limit is not None and index >= limit:
            break
        calls = {
            "batched": lambda: _compressed_adaptive(inst, tl),
            "clique": lambda: adaptive_clique_interval_bound(
                inst, relative_tolerance=RELATIVE_TOLERANCE, time_limit=tl),
            "direct": lambda: direct_lp_scan(inst, tl),
        }
        times = {m: [] for m in LP_METHODS}
        cpus = {m: [] for m in LP_METHODS}
        results = {m: [] for m in LP_METHODS}
        load = _wait_for_quiet_machine()
        for repeat in range(repeats):
            shift = (repeat + seed) % 3
            for method in LP_METHODS[shift:] + LP_METHODS[:shift]:
                seconds, cpu, result = _timed(calls[method])
                times[method].append(seconds)
                cpus[method].append(cpu)
                results[method].append(result)
                raw.append({"panel": panel, "configuration": config, "family": family, "n": n,
                            "seed": seed, "method": method, "repeat": repeat, "seconds": seconds,
                            "cpu_seconds": cpu, "status": result["status"],
                            "load_average": os.getloadavg()[0]})
        med = {m: float(np.median(times[m])) for m in LP_METHODS}
        direct = results["direct"][0]
        batched, clique = results["batched"][0], results["clique"][0]
        row = {"panel": panel, "configuration": config, "family": family, "n": n, "seed": seed,
               "instance": inst.name, "theta_count": direct["theta_count"],
               "batched_seconds": med["batched"], "clique_seconds": med["clique"],
               "direct_seconds": med["direct"],
               **{f"{m}_cpu_seconds": float(np.median(cpus[m])) for m in LP_METHODS},
               "load_average_at_start": load,
               "batched_interval_bounds": batched["interval_bound_evaluations"],
               "batched_multiplier_evaluations": batched["multiplier_evaluations"],
               "clique_interval_bounds": clique["interval_lp_evaluations"],
               "batched_root_upper_bound": batched["root_upper_bound"],
               "clique_root_upper_bound": clique["root_upper_bound"],
               "batched_status": batched["status"], "clique_status": clique["status"],
               "direct_status": direct["status"],
               "direct_value": direct["value"],
               "batched_lower_bound": batched["lower_bound"],
               "clique_lower_bound": clique["lower_bound"],
               "batched_theta_lp_evaluations": batched["theta_lp_evaluations"],
               "clique_theta_lp_evaluations": clique["theta_lp_evaluations"],
               "direct_theta_lp_evaluations": direct["theta_lp_evaluations"]}
        if direct["status"] == "optimal":
            for m in ("batched", "clique"):
                lb = float(row[f"{m}_lower_bound"])
                scale = max(1.0, abs(lb))
                row[f"{m}_agrees_with_direct"] = bool(
                    lb <= direct["value"] + 1e-9 * scale
                    and direct["value"] <= lb + RELATIVE_TOLERANCE * scale + 1e-9 * scale)
        rows.append(row)
        _write_csv(out / "lp_direct.csv", rows)
        _write_csv(out / "lp_direct_raw_timings.csv", raw)
    (out / "lp_direct_summary.json").write_text(json.dumps(summarize_lp(rows), indent=2, sort_keys=True))


def summarize_lp(rows: list[dict]) -> dict:
    summary = {}
    for panel in sorted({r["panel"] for r in rows}):
        groups = sorted({r["configuration"] for r in rows if r["panel"] == panel})
        for group in groups:
            sel = [r for r in rows if r["panel"] == panel and r["configuration"] == group]
            entry = {"instances": len(sel)}
            for m in LP_METHODS:
                entry[f"{m}_completed"] = sum(r[f"{m}_status"] == "optimal" for r in sel)
            done = [r for r in sel if r["direct_status"] == "optimal"]
            for m in ("batched", "clique"):
                entry[f"{m}_vs_direct_geomean_speedup"] = geometric_mean(
                    r["direct_seconds"] / r[f"{m}_seconds"] for r in done) if done else None
                entry[f"{m}_wins_vs_direct"] = sum(r[f"{m}_seconds"] < r["direct_seconds"] for r in sel)
                entry[f"{m}_agrees_with_direct"] = sum(bool(r.get(f"{m}_agrees_with_direct")) for r in done)
            entry["batched_vs_clique_geomean_speedup"] = geometric_mean(
                r["clique_seconds"] / r["batched_seconds"] for r in sel)
            entry["fastest_counts"] = {
                m: sum(min(LP_METHODS, key=lambda k: r[f"{k}_seconds"]) == m for r in sel)
                for m in LP_METHODS}
            entry["median_theta_count"] = float(np.median([r["theta_count"] for r in sel]))
            summary[f"{panel}/{group}"] = entry
    return summary


def run_integer(out: Path, limit: int | None = None) -> None:
    tl = float(PROTOCOL["integer"]["time_limit_seconds"])
    rows = []
    for index, (panel, family, n, seed, inst) in enumerate(integer_instances()):
        if limit is not None and index >= limit:
            break
        shift = index % len(INTEGER_METHODS)
        for method in INTEGER_METHODS[shift:] + INTEGER_METHODS[:shift]:
            load = _wait_for_quiet_machine()
            seconds, cpu, result = _timed(lambda: integer_method(inst, method, tl))
            result["cpu_seconds"] = cpu
            result["load_average_at_start"] = load
            result["within_limit"] = seconds <= tl + 1.0
            result["certified"] = bool(result["certified"]) and result["within_limit"]
            rows.append({"panel": panel, "family": family, "n": n, "seed": seed,
                         "instance": inst.name, "method": method, "wall_seconds": seconds,
                         **result})
            _write_csv(out / "integer.csv", rows)
    (out / "integer_summary.json").write_text(json.dumps(summarize_integer(rows), indent=2, sort_keys=True))


def summarize_integer(rows: list[dict]) -> dict:
    summary = {}
    instances = sorted({(r["panel"], r["instance"]) for r in rows})
    disagreements = []
    for key in instances:
        sel = [r for r in rows if (r["panel"], r["instance"]) == key and r["certified"]]
        if sel:
            ref = sel[0]["objective"]
            for r in sel[1:]:
                if abs(r["objective"] - ref) > 1e-6 * max(1.0, abs(ref)):
                    disagreements.append({"instance": key[1], "method": r["method"]})
    for panel in sorted({r["panel"] for r in rows}):
        for n in sorted({r["n"] for r in rows if r["panel"] == panel}):
            sel = [r for r in rows if r["panel"] == panel and r["n"] == n]
            entry = {}
            for m in INTEGER_METHODS:
                ms = [r for r in sel if r["method"] == m]
                entry[m] = {
                    "runs": len(ms),
                    "certified": sum(r["certified"] for r in ms),
                    "inexact": sum(r["status"] == "inexact" for r in ms),
                    "median_seconds": float(np.median([r["seconds"] for r in ms])) if ms else None,
                    "max_seconds": float(max(r["seconds"] for r in ms)) if ms else None,
                }
            names = sorted({r["instance"] for r in sel})
            for a, b in (("clique", "envelope"), ("direct", "envelope"),
                         ("compact_highs", "envelope"), ("compact_scip", "envelope")):
                ratios = []
                for name in names:
                    ra = next(r for r in sel if r["instance"] == name and r["method"] == a)
                    rb = next(r for r in sel if r["instance"] == name and r["method"] == b)
                    if ra["certified"] and rb["certified"]:
                        ratios.append(ra["seconds"] / rb["seconds"])
                entry[f"{a}_over_envelope_geomean_joint"] = geometric_mean(ratios) if ratios else None
                entry[f"{a}_over_envelope_joint_count"] = len(ratios)
            summary[f"{panel}/n={n}"] = entry
    summary["objective_disagreements"] = disagreements
    return summary


# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.strip()


def initialize(out: Path, phase: str, pilot: bool) -> None:
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    if not pilot:
        if _git("status", "--porcelain", "--untracked-files=no"):
            raise SystemExit("tracked tree is dirty; commit the frozen sources first")
    out.mkdir(parents=True)
    protocol = json.dumps(PROTOCOL, indent=2, sort_keys=True).encode()
    (out / "protocol.json").write_bytes(protocol)
    sources = ["research/scale_campaign.py", "research/fixed_theta_highs.py",
               "research/integrated_exact_solver.py", "research/compressed_interval_oracle.py",
               "research/structural_feasibility_study.py", "research/v4_publication_campaign.py",
               "research/benchmark_instances.py"]
    import pyscipopt  # type: ignore
    environment = {
        "phase": phase, "pilot": pilot, "code_commit": _git("rev-parse", "HEAD"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "pyscipopt": pyscipopt.__version__, "platform": platform.platform(),
        "machine": platform.machine(), "load_average_at_start": os.getloadavg(),
        "thread_environment": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    (out / f"environment_{phase}.json").write_text(json.dumps(environment, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["lp-direct", "integer"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-limit", type=int, default=None,
                        help="pilot: pilot seeds only, first K instances; results are not reported")
    parser.add_argument("--pilot-time-limit", type=float, default=None,
                        help="pilot only: integer time limit override")
    args = parser.parse_args()
    pilot = args.pilot_limit is not None
    if args.pilot_time_limit is not None:
        if not pilot:
            raise SystemExit("--pilot-time-limit requires --pilot-limit")
        PROTOCOL["integer"]["time_limit_seconds"] = float(args.pilot_time_limit)
    if pilot:  # pilots time nothing that is reported, so they ignore the load gate
        global LOAD_GATE
        LOAD_GATE = float("inf")
    if pilot:  # separate seeds, so no reported instance is seen before the protocol is frozen
        panels = PROTOCOL["lp_direct"]["panels"]
        panels["heldout"]["seeds"] = [991]
        panels["robustness"]["seeds"] = [992]
        panels["application"]["seeds"] = [99]
        PROTOCOL["integer"]["structured"]["seeds"] = [993]
        PROTOCOL["integer"]["application"]["seeds"] = [99]
    initialize(args.output_dir, args.phase, pilot)
    if args.phase == "lp-direct":
        run_lp_direct(args.output_dir, args.pilot_limit)
    else:
        run_integer(args.output_dir, args.pilot_limit)
    print("PHASE COMPLETE", args.phase)


if __name__ == "__main__":
    main()
