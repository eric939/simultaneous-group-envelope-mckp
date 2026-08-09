"""Preprocessing and instance construction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .model import Option, PricingInstance
from .utils import EPS


def _as_array_1d(x: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return arr


def from_pricing_data(
    reference_prices: Sequence[float],
    weights: Sequence[float],
    price_menus: Sequence[Sequence[float]],
    demands: Sequence[Sequence[float]],
    uncertainties: Sequence[Sequence[float]],
    margin_target: float,
    tolerances: Sequence[float],
    gamma: int,
) -> PricingInstance:
    """Construct a PricingInstance from raw pricing inputs.

    The per-option values are computed as:
        v[i,j] = ω_i * x_{i,j} * ĝ_i(x_{i,j})
        s[i,j] = ω_i * (x_{i,j} - Δ a_i) * ĝ_i(x_{i,j})
        t[i,j] = ω_i * (x_{i,j} - Δ a_i) * δ_i(x_{i,j})

    Admissible options are filtered by the interval
    [(1-σ_i) a_i, (1+σ_i) a_i].

    Args:
        reference_prices: a_i array, shape (n,).
        weights: ω_i array, shape (n,).
        price_menus: list of price arrays, per item.
        demands: list of nominal demand arrays, per item.
        uncertainties: list of uncertainty arrays, per item.
        margin_target: Δ scalar.
        tolerances: σ_i array, shape (n,).
        gamma: budget Γ.

    Returns:
        PricingInstance with filtered options.
    """

    a = _as_array_1d(reference_prices, "reference_prices")
    w = _as_array_1d(weights, "weights")
    sigma = _as_array_1d(tolerances, "tolerances")

    n = a.size
    if w.size != n or sigma.size != n:
        raise ValueError("reference_prices, weights, tolerances must have same length")
    if len(price_menus) != n or len(demands) != n or len(uncertainties) != n:
        raise ValueError("price_menus, demands, uncertainties must have length n")

    items: List[List[Option]] = []

    for i in range(n):
        x = np.asarray(price_menus[i], dtype=float)
        g_hat = np.asarray(demands[i], dtype=float)
        delta = np.asarray(uncertainties[i], dtype=float)
        if x.shape != g_hat.shape or x.shape != delta.shape:
            raise ValueError(f"price, demand, uncertainty shapes mismatch for item {i}")
        if x.ndim != 1:
            raise ValueError(f"price menu for item {i} must be 1D")

        lower = (1.0 - sigma[i]) * a[i]
        upper = (1.0 + sigma[i]) * a[i]
        mask = (x >= lower - EPS) & (x <= upper + EPS)
        if not np.any(mask):
            raise ValueError(f"no admissible options for item {i}")

        x_f = x[mask]
        g_f = g_hat[mask]
        d_f = delta[mask]

        v = w[i] * x_f * g_f
        margin = w[i] * (x_f - margin_target * a[i]) * g_f
        uncertainty = w[i] * (x_f - margin_target * a[i]) * d_f

        group = [
            Option(value=float(vj), margin=float(sj), uncertainty=float(tj), price=float(xj))
            for vj, sj, tj, xj in zip(v, margin, uncertainty, x_f)
        ]
        items.append(group)

    return PricingInstance(items=items, gamma=int(gamma))


def filter_admissible_options(
    instance: PricingInstance,
    reference_prices: Sequence[float],
    tolerances: Sequence[float],
) -> PricingInstance:
    """Filter options to the admissible price interval for low-level instances.

    This helper is useful when constructing PricingInstance directly while still
    enforcing the admissible menu constraint
        x_i ∈ [(1-σ_i) a_i, (1+σ_i) a_i].

    Args:
        instance: PricingInstance with Option.price populated.
        reference_prices: a_i array, shape (n,).
        tolerances: σ_i array, shape (n,).

    Returns:
        A new PricingInstance with inadmissible options removed.
    """

    a = _as_array_1d(reference_prices, "reference_prices")
    sigma = _as_array_1d(tolerances, "tolerances")
    if a.size != instance.n_items or sigma.size != instance.n_items:
        raise ValueError("reference_prices and tolerances must match number of items")

    filtered_items: List[List[Option]] = []
    for i, group in enumerate(instance.items):
        lower = (1.0 - sigma[i]) * a[i]
        upper = (1.0 + sigma[i]) * a[i]
        admissible: List[Option] = []
        for opt in group:
            if opt.price is None:
                raise ValueError("Option.price must be set to filter admissible options")
            if lower - EPS <= opt.price <= upper + EPS:
                admissible.append(opt)
        if not admissible:
            raise ValueError(f"no admissible options for item {i}")
        filtered_items.append(admissible)

    return PricingInstance(items=filtered_items, gamma=instance.gamma, name=instance.name)

