"""Gradient-free factorized ternary decomposition (design.md D1-D6).

``A ~= B diag(D) C`` with ``B`` in {-1,0,1}^{m x k}, ``D`` real [k] and ``C`` in
{-1,0,1}^{k x n}, where ``k = round(mu * min(m, n))``. The fit is greedy
sequential deflation with alternating closed-form ternary updates; no autograd
is used anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from phase0_feasibility.decomposition.ternary import T_tau, T_tau_cols


@dataclass(frozen=True, eq=False)
class TernaryFactors:
    """Stored factors of a ternary decomposition.

    Attributes:
        B: Ternary factor, int8, shape ``[m, k]``, values in {-1, 0, 1}.
        D: Real scales, float32, shape ``[k]``, finite.
        C: Ternary factor, int8, shape ``[k, n]``, values in {-1, 0, 1}.
        k: Stored rank; equal to ``mu * min(m, n)`` rounded.
        residuals: Frobenius residual after each deflation step, float32,
            length ``steps + 1`` starting at ``||A||_F``. ``None`` if not
            recorded.
        stall_steps: Indices of steps whose residual reduction was below the
            configured epsilon, or whose optimal scalar was zero.
    """

    B: Tensor
    D: Tensor
    C: Tensor
    k: int
    residuals: Tensor | None = None
    stall_steps: tuple[int, ...] = ()

    def reconstruct(self) -> Tensor:
        """Return ``B @ diag(D) @ C`` as float32, shape ``[m, n]``."""
        dtype = self.D.dtype
        return self.B.to(dtype) @ torch.diag(self.D) @ self.C.to(dtype)

    @property
    def stalled(self) -> bool:
        """True when at least one deflation step stalled."""
        return len(self.stall_steps) > 0


def resolve_block_size(b: int | None, m: int, n: int) -> int:
    """Resolve the batched-path block size (design.md D4).

    Args:
        b: Explicit block size, or ``None`` to use ``min(256, min(m, n) // 8)``.
        m: Row count of the matrix being decomposed.
        n: Column count of the matrix being decomposed.

    Returns:
        A block size in ``[1, min(m, n)]``.
    """
    if b is not None:
        return max(1, min(int(b), min(m, n)))
    return max(1, min(256, min(m, n) // 8))


def normalized_importance(h: Tensor, lam: float) -> Tensor:
    """Return ``h / max(h) + lam`` (design.md D7).

    Args:
        h: Per-input-channel activation second moments, shape ``[n]``, >= 0.
        lam: Ridge added after normalisation; 0.0 keeps importance on by shape.

    Returns:
        Float32 tensor of shape ``[n]``.

    Raises:
        ValueError: If ``h`` is not 1-D, is empty, non-finite or negative.
    """
    if h.ndim != 1:
        raise ValueError(f"importance must be 1-D, got shape {tuple(h.shape)}")
    if h.numel() == 0:
        raise ValueError("importance must be non-empty")
    if not bool(torch.isfinite(h).all()):
        raise ValueError("importance contains non-finite values")
    if bool((h < 0).any()):
        raise ValueError("importance must be non-negative")
    h32 = h.to(torch.float32)
    peak = float(h32.max())
    if peak <= 0.0:
        return torch.ones_like(h32)
    return h32 / peak + lam


def decompose_ternary(
    A: Tensor,
    *,
    mu: float,
    tau: float,
    batched: bool,
    b: int | None,
    importance: Tensor | None,
    n_inner: int = 15,
    eps: float = 1e-8,
) -> TernaryFactors:
    """Decompose a weight matrix into ``B diag(D) C`` with ternary ``B``, ``C``.

    Args:
        A: Weight matrix ``[m, n]`` (``m`` rows, ``n`` columns). Converted to
            float32 at this boundary; the caller's tensor is not modified.
        mu: Rank multiplier; ``k = round(mu * min(m, n))``.
        tau: Ternary threshold passed to :func:`T_tau`.
        batched: If True use the batched component path (design.md D4); if
            False use the sequential rank-1 reference (design.md D3).
        b: Block size for the batched path; ``None`` selects the D4 default.
            Ignored when ``batched`` is False.
        importance: Optional per-input-channel weights ``h~`` of shape ``[n]``
            (see :func:`normalized_importance`). ``None`` disables weighting.
        n_inner: Inner alternating iterations per component block (design: 15).
        eps: Numerical epsilon for least-squares solves and stall detection.

    Returns:
        :class:`TernaryFactors` with ``B`` int8 ``[m, k]``, ``D`` float32
        ``[k]``, ``C`` int8 ``[k, n]``, residual history and stall steps.

    Raises:
        ValueError: If ``A`` is not a finite 2-D matrix, ``mu`` <= 0, ``eps``
            <= 0, ``n_inner`` < 1, or ``importance`` has the wrong shape.
    """
    if A.ndim != 2:
        raise ValueError(f"decompose_ternary expects a 2-D matrix, got {tuple(A.shape)}")
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")
    if n_inner < 1:
        raise ValueError(f"n_inner must be >= 1, got {n_inner}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if not bool(torch.isfinite(A).all()):
        raise ValueError("decompose_ternary received non-finite values in A")

    work = A.detach().to(torch.float32)
    m, n = int(work.shape[0]), int(work.shape[1])
    weights: Tensor | None = None
    if importance is not None:
        if importance.ndim != 1 or int(importance.shape[0]) != n:
            raise ValueError(f"importance must have shape [{n}], got {tuple(importance.shape)}")
        weights = importance.detach().to(torch.float32)
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("importance contains non-finite values")

    k = round(mu * min(m, n))
    if k < 1:
        raise ValueError(f"degenerate rank k={k} for mu={mu} and shape {tuple(work.shape)}")

    if batched:
        return _decompose_batched(work, k, tau, weights, b, n_inner, eps)
    return _decompose_sequential(work, k, tau, weights, n_inner, eps)


def _allocation(m: int, n: int, k: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.zeros((m, k), dtype=torch.int8, device=device),
        torch.zeros((k,), dtype=torch.float32, device=device),
        torch.zeros((k, n), dtype=torch.int8, device=device),
    )


def _record_step(
    residuals: Tensor, index: int, previous: float, current: float, eps: float
) -> bool:
    residuals[index + 1] = current
    if previous <= eps:
        return True
    return (previous - current) / previous < eps


def _decompose_sequential(
    work: Tensor,
    k: int,
    tau: float,
    importance: Tensor | None,
    n_inner: int,
    eps: float,
) -> TernaryFactors:
    m, n = int(work.shape[0]), int(work.shape[1])
    B, D, C = _allocation(m, n, k, work.device)
    residual = work.clone()
    residuals = torch.zeros((k + 1,), dtype=torch.float32)
    residuals[0] = torch.linalg.norm(residual)
    stall_steps: list[int] = []

    for i in range(k):
        pivot_flat = int(torch.argmax(residual.abs()))
        pivot_row = pivot_flat // n
        u = torch.zeros(m, dtype=torch.float32)
        u[pivot_row] = 1.0
        v = torch.zeros(n, dtype=torch.float32)
        for _ in range(n_inner):
            v = T_tau(residual.t() @ u, tau)
            u = T_tau(_weighted(residual, importance) @ v, tau)

        if importance is None:
            numerator = float(u @ (residual @ v))
            denominator = float((u @ u) * (v @ v)) + eps
        else:
            numerator = float(u @ ((residual * importance) @ v))
            denominator = float((u @ u) * (importance @ (v * v))) + eps
        d = numerator / denominator

        residual = residual - d * torch.outer(u, v)
        B[:, i] = u.to(torch.int8)
        C[i, :] = v.to(torch.int8)
        D[i] = d

        previous = float(residuals[i])
        current = float(torch.linalg.norm(residual))
        if _record_step(residuals, i, previous, current, eps) or d == 0.0:
            stall_steps.append(i)

    return TernaryFactors(B, D, C, k, residuals, tuple(stall_steps))


def _decompose_batched(
    work: Tensor,
    k: int,
    tau: float,
    importance: Tensor | None,
    b: int | None,
    n_inner: int,
    eps: float,
) -> TernaryFactors:
    m, n = int(work.shape[0]), int(work.shape[1])
    block = resolve_block_size(b, m, n)
    B, D, C = _allocation(m, n, k, work.device)
    residual = work.clone()
    n_blocks = (k + block - 1) // block
    residuals = torch.zeros((n_blocks + 1,), dtype=torch.float32)
    residuals[0] = torch.linalg.norm(residual)
    stall_steps: list[int] = []

    for s in range(0, k, block):
        width = min(block, k - s)
        U, V = _init_block(residual, width, tau)
        eye = torch.eye(width, dtype=torch.float32, device=work.device)

        for _ in range(n_inner):
            if importance is None:
                utu = U.t() @ U + eps * eye
                V = T_tau_cols(torch.linalg.solve(utu, U.t() @ residual).t(), tau)
                vtv = V.t() @ V + eps * eye
                U = T_tau_cols(torch.linalg.solve(vtv, (residual @ V).t()).t(), tau)
            else:
                weighted = residual * importance
                utu = U.t() @ U + eps * eye
                V = T_tau_cols(torch.linalg.solve(utu, U.t() @ weighted).t(), tau)
                vtv = V.t() @ (importance[:, None] * V) + eps * eye
                U = T_tau_cols(torch.linalg.solve(vtv, (weighted @ V).t()).t(), tau)

        if importance is None:
            gram = (U.t() @ U) * (V.t() @ V)
            rhs = torch.diagonal(U.t() @ (residual @ V))
        else:
            gram = (U.t() @ U) * (V.t() @ (importance[:, None] * V))
            rhs = torch.diagonal(U.t() @ ((residual * importance) @ V))
        d = torch.linalg.solve(gram + eps * eye, rhs)

        residual = residual - U @ torch.diag(d) @ V.t()
        B[:, s : s + width] = U.to(torch.int8)
        C[s : s + width, :] = V.t().to(torch.int8)
        D[s : s + width] = d

        block_index = s // block
        previous = float(residuals[block_index])
        current = float(torch.linalg.norm(residual))
        if _record_step(residuals, block_index, previous, current, eps) or bool((d == 0).any()):
            stall_steps.append(s)

    return TernaryFactors(B, D, C, k, residuals, tuple(stall_steps))


def _weighted(residual: Tensor, importance: Tensor | None) -> Tensor:
    return residual if importance is None else residual * importance


def _init_block(residual: Tensor, width: int, tau: float) -> tuple[Tensor, Tensor]:
    """Initialise a block from the largest-magnitude residual rows (design.md D5)."""
    m = int(residual.shape[0])
    row_norms = torch.linalg.norm(residual, dim=1)
    take = min(width, m)
    rows = torch.topk(row_norms, k=take).indices
    U = torch.zeros((m, width), dtype=torch.float32, device=residual.device)
    U[rows, torch.arange(take, device=residual.device)] = 1.0
    V = T_tau_cols(residual.t() @ U, tau)
    return U, V
