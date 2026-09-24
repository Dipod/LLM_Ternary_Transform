"""Symmetric ``mu=1`` baseline, the contrast expected to fail (design.md D1, proposal).

It runs the identical gradient-free procedure at ``mu = 1`` with no importance
weighting, so the primary result is always reported against the same metric set.
"""

from __future__ import annotations

from torch import Tensor

from phase0_feasibility.decomposition.core import TernaryFactors, decompose_ternary


def symmetric_baseline(
    W: Tensor,
    *,
    tau: float,
    batched: bool,
    b: int | None,
    n_inner: int = 15,
    eps: float = 1e-8,
) -> TernaryFactors:
    """Decompose ``W`` with ``mu = 1`` and no importance weighting.

    Args:
        W: Weight matrix ``[out, in]``.
        tau: Ternary threshold.
        batched: Whether to use the batched compute path.
        b: Block size for the batched path (``None`` selects the D4 default).
        n_inner: Inner alternating iterations per component block.
        eps: Numerical epsilon.

    Returns:
        :class:`TernaryFactors` with ``k = round(min(out, in))``.
    """
    return decompose_ternary(
        W,
        mu=1.0,
        tau=tau,
        batched=batched,
        b=b,
        importance=None,
        n_inner=n_inner,
        eps=eps,
    )
