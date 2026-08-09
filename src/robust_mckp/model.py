"""Data models for robust MCKP pricing optimization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class Option:
    """An option within a pricing menu.

    Attributes:
        value: Objective coefficient v[i,j].
        margin: Nominal margin contribution s[i,j].
        uncertainty: Uncertainty contribution t[i,j].
        price: Optional price level x_{i,j} for reference.
    """

    value: float
    margin: float
    uncertainty: float
    price: Optional[float] = None


@dataclass
class PricingInstance:
    """Robust pricing instance for the multiple-choice knapsack structure.

    Attributes:
        items: List of item groups, each a list of Option objects.
        gamma: Budget parameter Γ (integer in [0, n]).
        name: Optional label for logging/reporting.
    """

    items: List[List[Option]]
    gamma: int
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.gamma < 0:
            raise ValueError("gamma must be nonnegative")
        if not self.items:
            raise ValueError("items must be non-empty")
        for group in self.items:
            if not group:
                raise ValueError("each item group must be non-empty")
        if self.gamma > len(self.items):
            raise ValueError("gamma must be at most the number of items")

    @property
    def n_items(self) -> int:
        """Number of item groups."""

        return len(self.items)

    @property
    def n_options(self) -> int:
        """Total number of options across all groups."""

        return sum(len(group) for group in self.items)


@dataclass
class Solution:
    """Solution container for the robust MCKP solver.

    Attributes:
        selections: Selected option index for each item (list of length n).
        objective: Objective value Σ v[i, j(i)].
        lp_value: Optimal LP relaxation value at the chosen θ.
        gap_to_lp: Relative gap (lp_value - objective) / lp_value.
        certificate_value: Robust margin certificate Z = Σ s_i - β.
        theta: θ associated with the chosen candidate.
        elapsed: Wall-clock time in seconds.
        is_feasible: Whether the returned solution satisfies the robust constraint.
        metadata: Optional extra info for debugging.
    """

    selections: List[int]
    objective: float
    lp_value: float
    gap_to_lp: float
    certificate_value: float
    theta: float
    elapsed: float
    is_feasible: bool
    metadata: Optional[dict] = field(default=None)
