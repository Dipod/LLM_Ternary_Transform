"""Symmetric adaptive ternary operator (design.md D2, ExTernD Eq. (2)).

The threshold is ``tau * mean(|w|)`` over the vector being ternarized. The
operator is positively scale-invariant, so it is applied directly to residual
targets without a separate normalisation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def T_tau(w: Tensor, tau: float) -> Tensor:
    """Ternarize a 1-D vector with a symmetric adaptive threshold.

    Args:
        w: Input vector of shape ``[L]``.
        tau: Positive threshold multiplier; the cutoff is ``tau * mean(|w|)``.

    Returns:
        Tensor of shape ``[L]`` with values in {-1.0, 0.0, 1.0}: ``sign(w_i)``
        where ``|w_i| > tau * mean(|w|)`` and zero otherwise. If that mask is
        empty, the single largest-magnitude entry is kept (safeguard) so the
        vector is never all zeros unless ``w`` itself is zero.

    Raises:
        ValueError: If ``w`` is not 1-D, is empty, non-finite, or ``tau`` <= 0.
    """
    if w.ndim != 1:
        raise ValueError(f"T_tau expects a 1-D vector, got shape {tuple(w.shape)}")
    if w.numel() == 0:
        raise ValueError("T_tau expects a non-empty vector")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if not bool(torch.isfinite(w).all()):
        raise ValueError("T_tau received non-finite values")

    threshold = tau * w.abs().mean()
    mask = w.abs() > threshold
    out = torch.where(mask, torch.sign(w), torch.zeros_like(w))
    if not bool(mask.any()):
        idx = int(torch.argmax(w.abs()))
        out[idx] = torch.sign(w[idx])
    return out


def T_tau_cols(matrix: Tensor, tau: float) -> Tensor:
    """Apply :func:`T_tau` independently to every column of a 2-D matrix.

    Args:
        matrix: Input matrix of shape ``[rows, cols]``; each column is a vector
            to ternarize.
        tau: Positive threshold multiplier, as in :func:`T_tau`.

    Returns:
        Tensor of shape ``[rows, cols]`` with values in {-1.0, 0.0, 1.0}. The
        all-zero safeguard is applied per column.

    Raises:
        ValueError: If ``matrix`` is not 2-D, is non-finite, or ``tau`` <= 0.
    """
    if matrix.ndim != 2:
        raise ValueError(f"T_tau_cols expects a 2-D matrix, got shape {tuple(matrix.shape)}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("T_tau_cols received non-finite values")
    if matrix.shape[1] == 0:
        return matrix.clone()

    threshold = tau * matrix.abs().mean(dim=0, keepdim=True)
    mask = matrix.abs() > threshold
    out = torch.where(mask, torch.sign(matrix), torch.zeros_like(matrix))
    empty_cols = (~mask.any(dim=0)).nonzero(as_tuple=False).flatten()
    if empty_cols.numel() > 0:
        rows = torch.argmax(matrix.abs()[:, empty_cols], dim=0)
        out[rows, empty_cols] = torch.sign(matrix[rows, empty_cols])
    return out
