"""Independent epigraph LP and validation helpers for interval-bound dominance.

For a finite threshold interval, the group-envelope minimax bound is

    inf_{lambda >= 0} max_k D(lambda, theta_k).

Its LP epigraph has a useful dual interpretation: it mixes group-simplex
solutions across thresholds while enforcing one aggregate capacity row.  The
mixture maps directly into the bounded-threshold group-clique LP, proving that
the minimax bound never exceeds the clique-LP bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import scipy.optimize as opt
from scipy import sparse

from robust_mckp import PricingInstance
from robust_mckp.exact_bnb import build_fixed_theta_data, build_full_theta_candidates


@dataclass(frozen=True)
class ExactMinimaxResult:
    """Numerical optimizer solution of the finite minimax epigraph LP."""

    upper_bound: float
    multiplier: float
    feasible_thresholds: int
    status: str


def exact_minimax_epigraph_bound(
    instance: PricingInstance,
    lo: int,
    hi: int,
    *,
    tolerance: float = 1e-9,
) -> ExactMinimaxResult:
    """Solve the finite minimax envelope bound as an independent epigraph LP.

    This routine is intended for theorem validation and small/medium audit
    cases.  The publication algorithm uses the faster one-dimensional oracle.
    """

    thetas = np.asarray(build_full_theta_candidates(instance), dtype=float)
    if lo < 0 or hi >= len(thetas) or lo > hi:
        raise ValueError("invalid threshold interval")

    records = []
    for index in range(lo, hi + 1):
        theta = float(thetas[index])
        data = build_fixed_theta_data(instance, theta)
        exact_theta = Fraction.from_float(theta)
        exact_capacity = -int(instance.gamma) * exact_theta
        for group in instance.items:
            exact_capacity += max(
                Fraction.from_float(float(option.margin))
                - max(
                    Fraction.from_float(abs(float(option.uncertainty))) - exact_theta,
                    Fraction(0),
                )
                for option in group
            )
        if exact_capacity >= 0:
            records.append((index, data))
    if not records:
        return ExactMinimaxResult(float("-inf"), 0.0, 0, "infeasible")

    # Variables are lambda, epigraph t, and alpha_{k,i} for each feasible
    # threshold k and group i.  Alpha variables are free.
    n_groups = instance.n_items
    n_thresholds = len(records)
    alpha_offset = 2
    nvars = alpha_offset + n_thresholds * n_groups

    def alpha_index(k_pos: int, group: int) -> int:
        return alpha_offset + k_pos * n_groups + group

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    rhs: list[float] = []
    row = 0

    for k_pos, (_index, data) in enumerate(records):
        # lambda*C_k + sum_i alpha_{k,i} - t <= 0.
        row_indices.extend([row, row])
        col_indices.extend([0, 1])
        values.extend([float(data.capacity), -1.0])
        for group in range(n_groups):
            row_indices.append(row)
            col_indices.append(alpha_index(k_pos, group))
            values.append(1.0)
        rhs.append(0.0)
        row += 1

        # v_ij - lambda*c_ijk - alpha_ki <= 0.
        for group, group_values in enumerate(data.values):
            for option, objective in enumerate(group_values):
                cost = float(data.costs[group][option])
                if cost:
                    row_indices.append(row)
                    col_indices.append(0)
                    values.append(-cost)
                row_indices.append(row)
                col_indices.append(alpha_index(k_pos, group))
                values.append(-1.0)
                rhs.append(-float(objective))
                row += 1

    a_ub = sparse.csr_matrix(
        (np.asarray(values), (np.asarray(row_indices), np.asarray(col_indices))),
        shape=(row, nvars),
    )
    objective = np.zeros(nvars, dtype=float)
    objective[1] = 1.0
    bounds = [(0.0, None), (None, None)] + [(None, None)] * (nvars - 2)
    result = opt.linprog(
        objective,
        A_ub=a_ub,
        b_ub=np.asarray(rhs, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            f"exact minimax epigraph LP failed: status={result.status}, "
            f"message={result.message}"
        )
    return ExactMinimaxResult(
        upper_bound=float(result.fun),
        multiplier=float(result.x[0]),
        feasible_thresholds=n_thresholds,
        status="optimal",
    )


__all__ = ["ExactMinimaxResult", "exact_minimax_epigraph_bound"]
