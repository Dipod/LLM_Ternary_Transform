"""Tests for deterministic cosine-greedy column ordering (task 2.5)."""

from __future__ import annotations

import pytest
import torch

from phase0_feasibility.decomposition.reordering import cosine_greedy_order


def test_permutation_is_valid() -> None:
    generator = torch.Generator().manual_seed(13)
    W = torch.randn(8, 20, generator=generator)
    perm, inv_perm = cosine_greedy_order(W)
    assert perm.shape == (20,)
    assert torch.equal(torch.sort(perm).values, torch.arange(20))
    assert torch.equal(perm[inv_perm], torch.arange(20))


def test_inverse_permutation_recovers_original_order() -> None:
    generator = torch.Generator().manual_seed(17)
    X = torch.randn(5, 12, generator=generator)
    W = torch.randn(7, 12, generator=generator)
    perm, inv_perm = cosine_greedy_order(W)
    recovered = X[:, perm][:, inv_perm]
    assert torch.equal(recovered, X)


def test_function_preservation() -> None:
    """Reordering both X and W preserves ``X W^T`` up to float tolerance."""
    generator = torch.Generator().manual_seed(19)
    X = torch.randn(6, 15, generator=generator)
    W = torch.randn(9, 15, generator=generator)
    perm, _ = cosine_greedy_order(W)
    reordered = X[:, perm] @ W[:, perm].t()
    assert torch.allclose(reordered, X @ W.t(), rtol=1e-5, atol=1e-5)


def test_deterministic() -> None:
    generator = torch.Generator().manual_seed(23)
    W = torch.randn(6, 14, generator=generator)
    first, _ = cosine_greedy_order(W)
    second, _ = cosine_greedy_order(W)
    assert torch.equal(first, second)


def test_empty_and_single_column() -> None:
    empty_perm, empty_inv = cosine_greedy_order(torch.zeros(3, 0))
    assert empty_perm.numel() == 0
    assert empty_inv.numel() == 0
    perm, inv_perm = cosine_greedy_order(torch.ones(3, 1))
    assert torch.equal(perm, torch.tensor([0]))
    assert torch.equal(inv_perm, torch.tensor([0]))


def test_non_finite_raises() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        cosine_greedy_order(torch.tensor([[1.0, float("inf")]]))
