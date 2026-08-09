"""Deterministic benchmark instances for the v4 publication campaign."""
from __future__ import annotations

import hashlib
import math

import numpy as np

from robust_mckp import Option, PricingInstance


BENCHMARK_FAMILIES = (
    "dense_frontier",
    "correlated_risk",
    "near_tie",
    "many_breakpoints",
)


def _rng(family: str, n: int, m: int, gamma: int, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"v4|{family}|{n}|{m}|{gamma}|{seed}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16) % (2**32))


def build_benchmark_instance(
    family: str,
    n: int,
    m: int,
    gamma: int,
    seed: int,
) -> PricingInstance:
    """Create the capacity-binding structured instances used by v4."""
    if family not in BENCHMARK_FAMILIES:
        raise ValueError(f"unknown benchmark family: {family}")
    rng = _rng(family, n, m, gamma, seed)
    max_cost = 14.0
    base_margin = {
        "dense_frontier": 4.6,
        "correlated_risk": 4.1,
        "near_tie": 3.9,
        "many_breakpoints": 4.4,
    }[family]
    deviation_grid = np.linspace(0.20, 3.20, 25)
    items: list[list[Option]] = []
    for i in range(n):
        base_value = 8.0 + rng.uniform(0.0, 0.4)
        group = [Option(value=base_value, margin=base_margin, uncertainty=0.0)]
        costs = np.linspace(0.8, max_cost, max(1, m - 1))
        for k, cost in enumerate(costs, start=1):
            if family == "dense_frontier":
                value = base_value + 8.0 * math.sqrt(cost) + rng.normal(0.0, 0.35)
                deviation = float(deviation_grid[(7 * i + 3 * k + seed) % len(deviation_grid)])
            elif family == "correlated_risk":
                value = base_value + 3.15 * cost + rng.normal(0.0, 0.75)
                raw = 0.30 + 0.19 * cost + rng.normal(0.0, 0.08)
                deviation = float(deviation_grid[int(np.argmin(np.abs(deviation_grid - raw)))])
            elif family == "near_tie":
                value = base_value + 2.25 * cost + rng.normal(0.0, 1.65)
                deviation = float(deviation_grid[(5 * i + k + seed) % len(deviation_grid)])
            else:
                value = base_value + 7.5 * math.sqrt(cost) + rng.normal(0.0, 0.45)
                deviation = float(0.15 + 0.003 * (i * max(1, m - 1) + k) + 1e-6 * seed)
            margin = base_margin - cost + rng.normal(0.0, 0.08)
            group.append(
                Option(
                    value=max(0.0, float(value)),
                    margin=float(margin),
                    uncertainty=deviation,
                )
            )
        items.append(group)
    return PricingInstance(
        items=items,
        gamma=int(gamma),
        name=f"{family}_n{n}_m{m}_g{gamma}_s{seed}",
    )
