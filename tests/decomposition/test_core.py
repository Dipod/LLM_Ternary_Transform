"""Tests for the gradient-free decomposition core (tasks 2.2, 2.3, 2.4)."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from phase0_feasibility.decomposition.baseline import symmetric_baseline
from phase0_feasibility.decomposition.core import (
    TernaryFactors,
    decompose_ternary,
    normalized_importance,
    resolve_block_size,
)

TAU = 0.7
EPS = 1e-8
INNER = 15
CROSS_CHECK_RTOL = 0.5


def _factors(
    A: Tensor,
    *,
    mu: float = 2.0,
    batched: bool = False,
    b: int | None = None,
    importance: Tensor | None = None,
) -> TernaryFactors:
    return decompose_ternary(
        A,
        mu=mu,
        tau=TAU,
        batched=batched,
        b=b,
        importance=importance,
        n_inner=INNER,
        eps=EPS,
    )


def test_shapes_and_value_domains() -> None:
    """B/D/C shapes equal the closed form and the domains are respected."""
    generator = torch.Generator().manual_seed(42)
    A = torch.randn(12, 9, generator=generator)
    factors = _factors(A)
    assert factors.k == 18
    assert factors.B.shape == (12, 18)
    assert factors.D.shape == (18,)
    assert factors.C.shape == (18, 9)
    assert factors.B.dtype == torch.int8
    assert factors.C.dtype == torch.int8
    allowed = torch.tensor([-1, 0, 1], dtype=torch.int8)
    assert bool(torch.isin(factors.B, allowed).all())
    assert bool(torch.isin(factors.C, allowed).all())
    assert bool(torch.isfinite(factors.D).all())


def test_reconstruction_shape() -> None:
    """Reconstruction closes to the original shape."""
    generator = torch.Generator().manual_seed(7)
    A = torch.randn(10, 6, generator=generator)
    assert _factors(A).reconstruct().shape == A.shape


def test_residual_non_increasing() -> None:
    """Every deflation step is non-increasing (design.md D3/D6)."""
    generator = torch.Generator().manual_seed(1)
    A = torch.randn(16, 12, generator=generator)
    factors = _factors(A)
    assert factors.residuals is not None
    residuals = factors.residuals
    assert residuals.shape == (factors.k + 1,)
    assert bool((residuals[1:] - residuals[:-1] <= 1e-6).all())


def test_stall_is_reported_on_engineered_matrix() -> None:
    """A matrix captured exactly by one ternary row stalls on later steps."""
    A = torch.zeros(12, 9)
    A[0] = torch.tensor([1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0])
    factors = _factors(A, mu=1.0)
    assert factors.k == 9
    assert factors.residuals is not None
    assert float(factors.residuals[1]) <= 1e-6
    assert factors.stalled
    assert len(factors.stall_steps) >= 1


def test_batched_cross_check_against_sequential() -> None:
    """The batched path matches the sequential reference within tolerance."""
    generator = torch.Generator().manual_seed(11)
    A = torch.randn(24, 16, generator=generator)
    sequential = _factors(A, batched=False)
    batched = _factors(A, batched=True, b=None)
    assert sequential.B.shape == batched.B.shape
    assert sequential.C.shape == batched.C.shape
    assert sequential.D.shape == batched.D.shape

    def rel_error(factors: TernaryFactors) -> float:
        return float(torch.linalg.norm(factors.reconstruct() - A) / torch.linalg.norm(A))

    sequential_error = rel_error(sequential)
    batched_error = rel_error(batched)
    scale = max(sequential_error, batched_error, 1e-6)
    assert abs(sequential_error - batched_error) / scale <= CROSS_CHECK_RTOL


def test_importance_path_runs_and_reconstructs() -> None:
    """Importance weighting leaves the factor contract unchanged."""
    generator = torch.Generator().manual_seed(5)
    A = torch.randn(14, 10, generator=generator)
    importance = normalized_importance(torch.arange(1, 11, dtype=torch.float32), 0.0)
    factors = _factors(A, importance=importance)
    assert factors.B.shape == (14, 20)
    assert bool(torch.isfinite(factors.D).all())


def test_normalized_importance_scales_peak_to_one() -> None:
    h = torch.tensor([1.0, 4.0, 2.0])
    assert torch.allclose(normalized_importance(h, 0.1), torch.tensor([0.35, 1.1, 0.6]))


def test_normalized_importance_all_zero_is_ones() -> None:
    assert torch.equal(normalized_importance(torch.zeros(3), 0.0), torch.ones(3))


def test_resolve_block_size_default_and_override() -> None:
    assert resolve_block_size(None, 4864, 896) == 112
    assert resolve_block_size(None, 10, 12) == 1
    assert resolve_block_size(64, 4864, 896) == 64
    assert resolve_block_size(4096, 4864, 896) == 896


def test_symmetric_baseline_uses_mu_one() -> None:
    """The baseline keeps the same layer shape with k = min(out, in)."""
    generator = torch.Generator().manual_seed(9)
    W = torch.randn(20, 14, generator=generator)
    factors = symmetric_baseline(W, tau=TAU, batched=False, b=None)
    assert factors.k == 14
    assert factors.B.shape == (20, 14)
    assert factors.C.shape == (14, 14)


@pytest.mark.parametrize(
    ("A", "mu", "match"),
    [
        (torch.ones(3), 2.0, "2-D"),
        (torch.ones(4, 3), 0.0, "mu"),
    ],
)
def test_invalid_inputs_raise(A: Tensor, mu: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        decompose_ternary(A, mu=mu, tau=TAU, batched=False, b=None, importance=None)


def test_wrong_importance_shape_raises() -> None:
    with pytest.raises(ValueError, match="importance"):
        decompose_ternary(
            torch.ones(4, 3),
            mu=2.0,
            tau=TAU,
            batched=False,
            b=None,
            importance=torch.ones(5),
        )
