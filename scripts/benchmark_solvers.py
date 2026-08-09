#!/usr/bin/env python3
"""Minimal open-source MILP baselines used by the research audits."""
from __future__ import annotations

import math
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from robust_mckp import PricingInstance
from robust_mckp.certificate import compute_certificate
from robust_mckp.utils import EPS


def option_arrays(instance: PricingInstance) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    values = [np.array([option.value for option in group], dtype=float) for group in instance.items]
    margins = [np.array([option.margin for option in group], dtype=float) for group in instance.items]
    uncertainties = [np.array([abs(option.uncertainty) for option in group], dtype=float) for group in instance.items]
    return values, margins, uncertainties


def raw_theta_candidates(instance: PricingInstance) -> np.ndarray:
    values = [0.0]
    for group in instance.items:
        values.extend(abs(option.uncertainty) for option in group)
    return np.unique(np.array(values, dtype=float))


def fixed_theta_costs(
    instance: PricingInstance, theta: float
) -> Tuple[List[np.ndarray], List[np.ndarray], float, float]:
    values, margins, uncertainties = option_arrays(instance)
    costs: List[np.ndarray] = []
    baseline_sum = 0.0
    for margin, uncertainty in zip(margins, uncertainties):
        transformed = margin - np.maximum(0.0, uncertainty - theta)
        baseline = float(np.max(transformed))
        baseline_sum += baseline
        costs.append(np.maximum(baseline - transformed, 0.0))
    capacity = baseline_sum - float(instance.gamma) * float(theta)
    return values, costs, baseline_sum, capacity


def _scipy_opt():
    try:
        import scipy.optimize as opt

        return opt
    except Exception:
        return None


def _highs_status(status: int) -> str:
    return {0: "OPTIMAL", 1: "TIME_LIMIT", 2: "INFEASIBLE", 3: "UNBOUNDED", 4: "NUMERIC"}.get(
        int(status), f"STATUS_{status}"
    )


def _optional_float(value: object) -> float:
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def solve_full_robust_highs(
    instance: PricingInstance, *, time_limit: float = 60.0, threads: int = 1
) -> Dict[str, object]:
    opt = _scipy_opt()
    if opt is None or not hasattr(opt, "milp"):
        return {"status": "UNAVAILABLE", "certified": False}

    t0 = time.perf_counter()
    values, margins, uncertainties = option_arrays(instance)
    sizes = [len(value) for value in values]
    n = len(sizes)
    z_total = int(sum(sizes))
    u_offset = z_total
    pi_offset = z_total + n
    theta_idx = z_total + 2 * n
    nvars = z_total + 2 * n + 1

    objective_vector = np.zeros(nvars, dtype=float)
    objective_vector[:z_total] = -np.concatenate(values)
    lower = np.zeros(nvars, dtype=float)
    upper = np.full(nvars, np.inf, dtype=float)
    upper[:z_total] = 1.0
    integrality = np.zeros(nvars, dtype=int)
    integrality[:z_total] = 1
    rows: List[np.ndarray] = []
    lows: List[float] = []
    ups: List[float] = []

    offset = 0
    for i, size in enumerate(sizes):
        row = np.zeros(nvars, dtype=float)
        row[offset : offset + size] = 1.0
        rows.append(row)
        lows.append(1.0)
        ups.append(1.0)
        offset += size
    offset = 0
    for i, (size, uncertainty) in enumerate(zip(sizes, uncertainties)):
        row = np.zeros(nvars, dtype=float)
        row[u_offset + i] = 1.0
        row[offset : offset + size] = -uncertainty
        rows.append(row)
        lows.append(0.0)
        ups.append(0.0)
        offset += size
    for i in range(n):
        row = np.zeros(nvars, dtype=float)
        row[pi_offset + i] = 1.0
        row[u_offset + i] = -1.0
        row[theta_idx] = 1.0
        rows.append(row)
        lows.append(0.0)
        ups.append(np.inf)
    row = np.zeros(nvars, dtype=float)
    row[:z_total] = np.concatenate(margins)
    row[theta_idx] = -float(instance.gamma)
    row[pi_offset : pi_offset + n] = -1.0
    rows.append(row)
    lows.append(0.0)
    ups.append(np.inf)

    constraints = opt.LinearConstraint(np.vstack(rows), np.array(lows), np.array(ups))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*threads")
        result = opt.milp(
            objective_vector,
            integrality=integrality,
            bounds=opt.Bounds(lower, upper),
            constraints=constraints,
            options={"time_limit": float(time_limit), "mip_rel_gap": 1e-8, "threads": int(threads)},
        )
    runtime = time.perf_counter() - t0
    status = int(getattr(result, "status", -999))
    x = getattr(result, "x", None)
    objective = -float(result.fun) if getattr(result, "fun", None) is not None else float("nan")
    selections: Optional[List[int]] = None
    certificate = float("nan")
    if x is not None:
        selections = []
        offset = 0
        for size in sizes:
            selections.append(int(np.argmax(x[offset : offset + size])))
            offset += size
        certificate = compute_certificate(instance, selections)
    return {
        "status": _highs_status(status),
        "certified": status == 0 and selections is not None and certificate >= -1e-7,
        "objective": objective,
        "best_bound": -_optional_float(getattr(result, "mip_dual_bound", None)),
        "runtime_s": runtime,
        "mip_gap": _optional_float(getattr(result, "mip_gap", None)),
        "certificate_value": certificate,
        "selections": selections,
        "n_variables": nvars,
        "n_constraints": len(rows),
        "message": str(getattr(result, "message", "")),
    }


def solve_fixed_theta_highs(
    instance: PricingInstance, theta: float, *, time_limit: float = 20.0
) -> Dict[str, object]:
    opt = _scipy_opt()
    if opt is None or not hasattr(opt, "milp"):
        return {"status": "UNAVAILABLE", "certified": False}
    values, costs, _, capacity = fixed_theta_costs(instance, theta)
    if capacity < -EPS:
        return {"status": "INFEASIBLE_CAPACITY", "certified": True}
    sizes = [len(value) for value in values]
    total = int(sum(sizes))
    a_eq = np.zeros((len(sizes), total), dtype=float)
    offset = 0
    for i, size in enumerate(sizes):
        a_eq[i, offset : offset + size] = 1.0
        offset += size
    constraints = [
        opt.LinearConstraint(a_eq, np.ones(len(sizes)), np.ones(len(sizes))),
        opt.LinearConstraint(np.concatenate(costs)[None, :], -np.inf, np.array([capacity])),
    ]
    t0 = time.perf_counter()
    result = opt.milp(
        -np.concatenate(values),
        integrality=np.ones(total, dtype=int),
        bounds=opt.Bounds(np.zeros(total), np.ones(total)),
        constraints=constraints,
        options={"time_limit": float(time_limit), "mip_rel_gap": 1e-8},
    )
    runtime = time.perf_counter() - t0
    status = int(getattr(result, "status", -999))
    x = getattr(result, "x", None)
    objective = -float(result.fun) if getattr(result, "fun", None) is not None else float("nan")
    selections: Optional[List[int]] = None
    certificate = float("nan")
    cost_used = float("nan")
    if x is not None:
        selections = []
        offset = 0
        for size in sizes:
            selections.append(int(np.argmax(x[offset : offset + size])))
            offset += size
        certificate = compute_certificate(instance, selections)
        cost_used = float(sum(costs[i][j] for i, j in enumerate(selections)))
    return {
        "status": _highs_status(status),
        "certified": status == 0 and selections is not None and certificate >= -1e-7 and cost_used <= capacity + 1e-7,
        "objective": objective,
        "runtime_s": runtime,
        "mip_gap": _optional_float(getattr(result, "mip_gap", None)),
        "certificate_value": certificate,
        "selections": selections,
    }


def solve_theta_enum_highs(
    instance: PricingInstance, *, time_limit_per_theta: float = 5.0, max_thetas: int = 220
) -> Dict[str, object]:
    candidates = raw_theta_candidates(instance)
    if candidates.size > max_thetas:
        return {"status": "SKIPPED_TOO_MANY_THETA", "certified": False, "theta_count": int(candidates.size), "runtime_s": 0.0}
    best_objective = -math.inf
    best_selection: Optional[List[int]] = None
    evaluated = 0
    t0 = time.perf_counter()
    for theta in candidates:
        _, _, _, capacity = fixed_theta_costs(instance, float(theta))
        if capacity < -EPS:
            continue
        evaluated += 1
        result = solve_fixed_theta_highs(instance, float(theta), time_limit=time_limit_per_theta)
        if not bool(result.get("certified")):
            return {
                "status": "NOT_CERTIFIED",
                "certified": False,
                "theta_count": int(candidates.size),
                "theta_evaluated_milp": evaluated,
                "runtime_s": time.perf_counter() - t0,
                "failed_status": result.get("status"),
            }
        objective = float(result["objective"])
        if objective > best_objective + EPS:
            best_objective = objective
            best_selection = list(result["selections"])  # type: ignore[arg-type]
    certificate = compute_certificate(instance, best_selection) if best_selection is not None else float("nan")
    return {
        "status": "OPTIMAL" if best_selection is not None else "NO_FEASIBLE_THETA",
        "certified": best_selection is not None and certificate >= -1e-7,
        "objective": float(best_objective),
        "runtime_s": time.perf_counter() - t0,
        "theta_count": int(candidates.size),
        "theta_evaluated_milp": evaluated,
        "certificate_value": float(certificate),
        "selections": best_selection,
    }


def _status_clean(status: object) -> str:
    return str(status).upper().replace(" ", "_")


def _set_scip_threads(model: object, threads: int) -> None:
    for name in ("parallel/maxnthreads", "lp/threads"):
        try:
            model.setIntParam(name, int(threads))  # type: ignore[attr-defined]
        except Exception:
            pass


def solve_full_robust_scip(
    instance: PricingInstance, *, time_limit: float = 600.0, threads: int = 1
) -> Dict[str, object]:
    try:
        from pyscipopt import Model, quicksum  # type: ignore
    except Exception as exc:
        return {"status": "UNAVAILABLE", "certified": False, "error": str(exc)}
    t0 = time.perf_counter()
    values = [[option.value for option in group] for group in instance.items]
    margins = [[option.margin for option in group] for group in instance.items]
    uncertainties = [[abs(option.uncertainty) for option in group] for group in instance.items]
    model = Model("full_robust_mckp")
    model.hideOutput()
    model.setRealParam("limits/time", float(time_limit))
    _set_scip_threads(model, threads)
    z = {(i, j): model.addVar(vtype="B", name=f"z_{i}_{j}") for i, group in enumerate(instance.items) for j in range(len(group))}
    theta = model.addVar(lb=0.0, vtype="C", name="theta")
    u = {i: model.addVar(lb=0.0, vtype="C", name=f"u_{i}") for i in range(instance.n_items)}
    pi = {i: model.addVar(lb=0.0, vtype="C", name=f"pi_{i}") for i in range(instance.n_items)}
    for i, group in enumerate(instance.items):
        model.addCons(quicksum(z[i, j] for j in range(len(group))) == 1.0)
        model.addCons(u[i] == quicksum(uncertainties[i][j] * z[i, j] for j in range(len(group))))
        model.addCons(pi[i] >= u[i] - theta)
    model.addCons(
        quicksum(margins[i][j] * z[i, j] for i, group in enumerate(instance.items) for j in range(len(group)))
        - float(instance.gamma) * theta - quicksum(pi[i] for i in range(instance.n_items)) >= 0.0
    )
    model.setObjective(
        quicksum(values[i][j] * z[i, j] for i, group in enumerate(instance.items) for j in range(len(group))),
        "maximize",
    )
    model.optimize()
    runtime = time.perf_counter() - t0
    status = _status_clean(model.getStatus())
    selections: Optional[List[int]] = None
    certificate = float("nan")
    objective = float("nan")
    if model.getNSols() > 0:
        selections = [max(range(len(group)), key=lambda j: float(model.getVal(z[i, j]))) for i, group in enumerate(instance.items)]
        certificate = compute_certificate(instance, selections)
        objective = float(sum(instance.items[i][j].value for i, j in enumerate(selections)))
    try:
        bound = float(model.getDualbound())
    except Exception:
        bound = float("nan")
    try:
        gap = float(model.getGap())
    except Exception:
        gap = float("nan")
    return {
        "status": status,
        "certified": status == "OPTIMAL" and selections is not None and certificate >= -1e-7,
        "objective": objective,
        "best_bound": bound,
        "mip_gap": gap,
        "runtime_s": runtime,
        "certificate_value": certificate,
        "selections": selections,
        "n_variables": model.getNVars(),
        "n_constraints": model.getNConss(),
    }
