"""Deterministic cosine-greedy column ordering (design.md D8).

Phase 0 implements its own deterministic construction because the primary
source's Eq. (16) is garbled. The ordering is applied identically to the
``in_features`` axis of the weight matrix and of the calibration activations, so
the layer function is preserved exactly by the permutation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def cosine_greedy_order(W: Tensor, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    """Order the columns of ``W`` by greedy cosine similarity.

    The chain starts at the largest-norm column (ties resolved by lowest index)
    and repeatedly appends the unvisited column whose absolute cosine similarity
    to the current column is largest, so structurally similar input channels end
    up adjacent.

    Args:
        W: Weight matrix ``[out, in]``. Any floating dtype; computed in float32.
        eps: Clamp for zero-norm columns before normalisation.

    Returns:
        ``(perm, inv_perm)``, both int64 tensors of shape ``[in]``. ``perm[j]``
        is the original column index placed at new position ``j``; ``inv_perm``
        is its inverse, so ``perm[inv_perm[k]] == k``.

    Raises:
        ValueError: If ``W`` is not a finite 2-D matrix, or ``eps`` <= 0.
    """
    if W.ndim != 2:
        raise ValueError(f"cosine_greedy_order expects a 2-D matrix, got {tuple(W.shape)}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if not bool(torch.isfinite(W).all()):
        raise ValueError("cosine_greedy_order received non-finite values")

    work = W.detach().to(torch.float32)
    n = int(work.shape[1])
    if n == 0:
        empty = torch.empty((0,), dtype=torch.long)
        return empty, empty.clone()

    norms = torch.linalg.norm(work, dim=0).clamp_min(eps)
    unit = work / norms
    similarity = (unit.t() @ unit).abs()

    order = torch.empty((n,), dtype=torch.long)
    visited = torch.zeros((n,), dtype=torch.bool)
    current = int(torch.argmax(norms))
    for position in range(n):
        order[position] = current
        visited[current] = True
        if position == n - 1:
            break
        scores = similarity[current].clone()
        scores[visited] = -1.0
        current = int(torch.argmax(scores))

    inverse = torch.empty((n,), dtype=torch.long)
    inverse[order] = torch.arange(n, dtype=torch.long)
    return order, inverse
