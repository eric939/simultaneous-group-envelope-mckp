"""Core models and certifying solvers for robust multiple-choice knapsack."""

from .model import Option, PricingInstance, Solution
from .preprocessing import filter_admissible_options, from_pricing_data
from .solver import solve
from .exact_bnb import (
    FixedThetaBNBConfig,
    FixedThetaBNBResult,
    GlobalThetaBNBConfig,
    GlobalThetaBNBResult,
    solve_fixed_theta_bnb,
    solve_global_theta_bnb,
)
__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Option",
    "PricingInstance",
    "Solution",
    "from_pricing_data",
    "filter_admissible_options",
    "solve",
    "FixedThetaBNBConfig",
    "FixedThetaBNBResult",
    "GlobalThetaBNBConfig",
    "GlobalThetaBNBResult",
    "solve_fixed_theta_bnb",
    "solve_global_theta_bnb",
]
