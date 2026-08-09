"""Robust feasibility certificates over binary64 input coefficients."""
from __future__ import annotations

from fractions import Fraction
import math
from typing import List, Sequence

import numpy as np

from .model import PricingInstance
from .utils import top_gamma


def compute_certificate(instance: PricingInstance, selections: Sequence[int]) -> float:
    """Compute certificate value Z = Σ s_i(x_i) - β(x, Γ).

    Args:
        instance: PricingInstance with original margins and uncertainties.
        selections: Selected option indices per item.

    Returns:
        Certificate value Z.
    """

    if len(selections) != instance.n_items:
        raise ValueError("selections length must match number of items")

    s_vals: List[float] = []
    t_vals: List[float] = []
    for i, idx in enumerate(selections):
        option = instance.items[i][idx]
        s_vals.append(float(option.margin))
        t_vals.append(abs(float(option.uncertainty)))

    beta = top_gamma(np.asarray(t_vals, dtype=float), instance.gamma)
    return float(math.fsum(s_vals) - beta)


def certificate_is_feasible(instance: PricingInstance, selections: Sequence[int]) -> bool:
    """Return the exact sign of the sorted-Gamma certificate.

    Input coefficients are binary64 values. Converting those values to
    :class:`fractions.Fraction` checks the sign of the mathematical certificate
    represented by the input bits. Solver tolerances remain numerical controls;
    they never turn a negative original robust certificate into a feasible one.
    """

    if len(selections) != instance.n_items:
        raise ValueError("selections length must match number of items")

    margins: List[Fraction] = []
    deviations: List[tuple[float, Fraction]] = []
    for i, idx in enumerate(selections):
        option = instance.items[i][idx]
        margin = float(option.margin)
        deviation = abs(float(option.uncertainty))
        margins.append(Fraction.from_float(margin))
        deviations.append((deviation, Fraction.from_float(deviation)))

    deviations.sort(key=lambda pair: pair[0], reverse=True)
    gamma = min(max(int(instance.gamma), 0), len(deviations))
    exact_certificate = sum(margins, Fraction(0)) - sum(
        (fraction for _value, fraction in deviations[:gamma]), Fraction(0)
    )
    return exact_certificate >= 0


def is_feasible(instance: PricingInstance, selections: Sequence[int]) -> bool:
    """Return True if selections satisfy robust constraint."""

    return certificate_is_feasible(instance, selections)

