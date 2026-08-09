"""Utility helpers for robust MCKP."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

EPS = 1e-10


def is_close(a: float, b: float, tol: float = EPS) -> bool:
    """Return True if a and b are within tolerance."""

    return abs(a - b) <= tol


def safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    """Safe division with a default for near-zero denominator."""

    if abs(denom) <= EPS:
        return default
    return numer / denom


def top_gamma(values: np.ndarray, gamma: int) -> float:
    """Sum of the top-Γ values in a 1D array.

    Args:
        values: 1D array of nonnegative values.
        gamma: Number of largest elements to sum.

    Returns:
        Sum of largest gamma values (gamma=0 returns 0).
    """

    if gamma <= 0:
        return 0.0
    if gamma >= values.size:
        return float(values.sum())
    # partial sort for top-gamma
    idx = np.argpartition(values, -gamma)[-gamma:]
    return float(values[idx].sum())


