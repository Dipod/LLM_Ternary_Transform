"""Phase 0 feasibility package: gradient-free factorized ternary decomposition.

The package is a fail-fast gate harness for one Linear layer (design.md, change
``phase0-ternary-feasibility``). It never reads global state: every tunable value
enters through :mod:`phase0_feasibility.config`.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
