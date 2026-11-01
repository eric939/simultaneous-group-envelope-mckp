"""Integer subproblems and compact robust models solved by MILP solvers.

These routines let the exact threshold-interval search scale beyond the
pure-Python fixed-threshold branch and bound.  HiGHS is called through
``scipy.optimize.milp`` and SCIP through PySCIPOpt.  Every returned selection
is re-checked in exact arithmetic by the caller; solver tolerances therefore
never decide feasibility.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from robust_mckp import PricingInstance
from robust_mckp.certificate import certificate_is_feasible
from robust_mckp.exact_bnb import (
    FixedThetaBNBResult,
    _fixed_theta_capacity_is_negative,
    build_fixed_theta_data,
)

MIP_RELATIVE_GAP = 1e-7


def _empty_result(status: str, theta: float, seconds: float, upper: float, message: str = "") -> FixedThetaBNBResult:
    return FixedThetaBNBResult(
        status=status,
        theta=float(theta),
        objective_value=float("-inf"),
        selected_options=None,
        used_capacity=float("nan"),
        capacity=float("nan"),
        fixed_theta_residual=float("nan"),
        upper_bound=float(upper),
        lower_bound=float("-inf"),
        absolute_gap=float("inf"),
        relative_gap=float("inf"),
        nodes_explored=0,
        nodes_pruned_bound=0,
        nodes_pruned_infeasible=0,
        nodes_integral=0,
        runtime_seconds=float(seconds),
        message=message,
    )


def _one_per_group_matrix(sizes: list[int], extra_columns: int = 0) -> csr_matrix:
    rows, cols = [], []
    offset = 0
    for i, size in enumerate(sizes):
        rows.extend([i] * size)
        cols.extend(range(offset, offset + size))
        offset += size
    return csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(sizes), offset + extra_columns)
    )


def _split(x: np.ndarray, sizes: list[int]) -> list[int]:
    selection, offset = [], 0
    for size in sizes:
        selection.append(int(np.argmax(x[offset:offset + size])))
        offset += size
    return selection


def solve_fixed_theta_highs(
    instance: PricingInstance, theta: float, *, time_limit: float
) -> FixedThetaBNBResult:
    """Solve the fixed-threshold MCKP at ``theta`` with HiGHS."""

    start = time.perf_counter()
    data = build_fixed_theta_data(instance, theta)
    if _fixed_theta_capacity_is_negative(data):
        return _empty_result("infeasible", theta, time.perf_counter() - start, float("-inf"))
    if time_limit <= 0.0:
        return _empty_result("time_limit", theta, 0.0, float("inf"))
    sizes = [len(values) for values in data.values]
    values = np.concatenate(data.values)
    costs = np.concatenate(data.costs)
    constraints = [
        LinearConstraint(_one_per_group_matrix(sizes), 1.0, 1.0),
        LinearConstraint(csr_matrix(costs.reshape(1, -1)), -np.inf, float(data.capacity)),
    ]
    result = milp(
        -values,
        constraints=constraints,
        integrality=np.ones(values.size),
        bounds=Bounds(0.0, 1.0),
        options={"time_limit": float(time_limit), "mip_rel_gap": MIP_RELATIVE_GAP, "disp": False},
    )
    seconds = time.perf_counter() - start
    dual = getattr(result, "mip_dual_bound", None)
    upper = -float(dual) if dual is not None and math.isfinite(float(dual)) else float("inf")
    nodes = int(getattr(result, "mip_node_count", 0) or 0)
    if result.status == 2:
        return _empty_result("infeasible", theta, seconds, float("-inf"))
    if result.x is None:
        status = "time_limit" if result.status == 1 else "error"
        return _empty_result(status, theta, seconds, upper, str(result.message))
    selection = _split(result.x, sizes)
    objective = float(sum(data.values[i][j] for i, j in enumerate(selection)))
    status = "optimal" if result.status == 0 else ("time_limit" if result.status == 1 else "error")
    if status == "optimal":
        upper = max(objective, upper) if math.isfinite(upper) else objective
    used = float(sum(data.costs[i][j] for i, j in enumerate(selection)))
    gap = max(0.0, upper - objective)
    return FixedThetaBNBResult(
        status=status,
        theta=float(theta),
        objective_value=objective,
        selected_options=selection,
        used_capacity=used,
        capacity=float(data.capacity),
        fixed_theta_residual=float(data.capacity) - used,
        upper_bound=float(upper),
        lower_bound=objective,
        absolute_gap=gap,
        relative_gap=gap / max(1.0, abs(objective)),
        nodes_explored=nodes,
        nodes_pruned_bound=0,
        nodes_pruned_infeasible=0,
        nodes_integral=0,
        runtime_seconds=seconds,
        message=str(result.message),
    )


def _compact_result(instance, status, selection, upper, seconds, nodes) -> dict:
    objective = float("-inf")
    feasible = False
    if selection is not None:
        objective = float(sum(instance.items[i][j].value for i, j in enumerate(selection)))
        feasible = bool(certificate_is_feasible(instance, selection))
    certified = status == "optimal" and feasible
    if status == "optimal" and not feasible:
        status = "inexact"
    gap = (
        max(0.0, upper - objective) / max(1.0, abs(objective))
        if selection is not None and math.isfinite(upper)
        else float("inf")
    )
    return {
        "status": status,
        "certified": certified,
        "exactly_feasible": feasible,
        "objective": objective,
        "upper_bound": float(upper),
        "relative_gap": float(gap),
        "seconds": float(seconds),
        "nodes": int(nodes),
        "selection": selection,
    }


def solve_compact_highs(instance: PricingInstance, *, time_limit: float) -> dict:
    """Compact Bertsimas--Sim MILP with HiGHS; variables z, theta, pi."""

    start = time.perf_counter()
    sizes = [len(group) for group in instance.items]
    k, n = sum(sizes), instance.n_items
    values = np.array([o.value for g in instance.items for o in g], dtype=float)
    margins = np.array([o.margin for g in instance.items for o in g], dtype=float)
    deviations = np.array([abs(o.uncertainty) for g in instance.items for o in g], dtype=float)
    width = k + 1 + n  # z, theta, pi
    objective = np.zeros(width)
    objective[:k] = -values
    group_rows = _one_per_group_matrix(sizes, extra_columns=1 + n)
    rows, cols, data = [], [], []
    offset = 0
    for i, size in enumerate(sizes):  # sum_j d_ij z_ij - theta - pi_i <= 0
        for j in range(size):
            rows.append(i); cols.append(offset + j); data.append(deviations[offset + j])
        rows.append(i); cols.append(k); data.append(-1.0)
        rows.append(i); cols.append(k + 1 + i); data.append(-1.0)
        offset += size
    protection = csr_matrix((data, (rows, cols)), shape=(n, width))
    margin_row = np.zeros(width)  # sum a z - Gamma theta - sum pi >= 0
    margin_row[:k] = margins
    margin_row[k] = -float(instance.gamma)
    margin_row[k + 1:] = -1.0
    lower = np.zeros(width)
    upper = np.concatenate([np.ones(k), np.full(1 + n, np.inf)])
    integrality = np.concatenate([np.ones(k), np.zeros(1 + n)])
    result = milp(
        objective,
        constraints=[
            LinearConstraint(group_rows, 1.0, 1.0),
            LinearConstraint(protection, -np.inf, 0.0),
            LinearConstraint(csr_matrix(margin_row.reshape(1, -1)), 0.0, np.inf),
        ],
        integrality=integrality,
        bounds=Bounds(lower, upper),
        options={"time_limit": float(time_limit), "mip_rel_gap": MIP_RELATIVE_GAP, "disp": False},
    )
    seconds = time.perf_counter() - start
    dual = getattr(result, "mip_dual_bound", None)
    bound = -float(dual) if dual is not None and math.isfinite(float(dual)) else float("inf")
    nodes = int(getattr(result, "mip_node_count", 0) or 0)
    if result.status == 2:
        return _compact_result(instance, "infeasible", None, float("-inf"), seconds, nodes)
    selection = _split(result.x[:k], sizes) if result.x is not None else None
    status = {0: "optimal", 1: "time_limit"}.get(int(result.status), "error")
    return _compact_result(instance, status, selection, bound, seconds, nodes)


def solve_compact_scip(instance: PricingInstance, *, time_limit: float) -> dict:
    """Compact Bertsimas--Sim MILP with SCIP, single-threaded, gap 1e-7."""

    from pyscipopt import Model, quicksum  # type: ignore

    start = time.perf_counter()
    model = Model("robust_mckp_compact")
    model.hideOutput()
    model.setRealParam("limits/time", float(time_limit))
    model.setRealParam("limits/gap", MIP_RELATIVE_GAP)
    for name in ("parallel/maxnthreads", "lp/threads"):
        try:
            model.setIntParam(name, 1)
        except Exception:
            pass
    z = {
        (i, j): model.addVar(vtype="B", name=f"z_{i}_{j}")
        for i, group in enumerate(instance.items)
        for j in range(len(group))
    }
    theta = model.addVar(lb=0.0, vtype="C", name="theta")
    pi = [model.addVar(lb=0.0, vtype="C", name=f"pi_{i}") for i in range(instance.n_items)]
    for i, group in enumerate(instance.items):
        model.addCons(quicksum(z[i, j] for j in range(len(group))) == 1.0)
        model.addCons(
            pi[i] >= quicksum(abs(group[j].uncertainty) * z[i, j] for j in range(len(group))) - theta
        )
    model.addCons(
        quicksum(o.margin * z[i, j] for i, g in enumerate(instance.items) for j, o in enumerate(g))
        - float(instance.gamma) * theta - quicksum(pi) >= 0.0
    )
    model.setObjective(
        quicksum(o.value * z[i, j] for i, g in enumerate(instance.items) for j, o in enumerate(g)),
        "maximize",
    )
    model.optimize()
    seconds = time.perf_counter() - start
    raw = str(model.getStatus()).lower()
    status = {"optimal": "optimal", "gaplimit": "optimal", "timelimit": "time_limit",
              "infeasible": "infeasible"}.get(raw, raw)
    selection = None
    if model.getNSols() > 0:
        selection = [
            max(range(len(g)), key=lambda j: float(model.getVal(z[i, j])))
            for i, g in enumerate(instance.items)
        ]
    bound = float(model.getDualbound())
    return _compact_result(instance, status, selection, bound, seconds, model.getNNodes())


__all__ = ["solve_fixed_theta_highs", "solve_compact_highs", "solve_compact_scip"]
