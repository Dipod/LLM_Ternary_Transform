"""Known-answer tests for the symmetric adaptive ternary operator (task 2.1)."""

from __future__ import annotations

import pytest
import torch

from phase0_feasibility.decomposition.ternary import T_tau, T_tau_cols


def test_known_answer_vector() -> None:
    """Hand-computed case: mean|w| = 1.11, tau=1.0 keeps {2.0, -3.0}."""
    w = torch.tensor([0.1, 2.0, -3.0, 0.4, -0.05])
    expected = torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0])
    assert torch.equal(T_tau(w, 1.0), expected)


def test_known_answer_higher_threshold() -> None:
    """tau=2.5 sets the cutoff at 2.775, so only -3.0 survives."""
    w = torch.tensor([0.1, 2.0, -3.0, 0.4, -0.05])
    expected = torch.tensor([0.0, 0.0, -1.0, 0.0, 0.0])
    assert torch.equal(T_tau(w, 2.5), expected)


def test_all_zero_safeguard_keeps_single_largest_entry() -> None:
    """When the mask is empty the largest-magnitude entry is retained."""
    w = torch.tensor([0.1, -0.2, 0.05])
    expected = torch.tensor([0.0, -1.0, 0.0])
    assert torch.equal(T_tau(w, 10.0), expected)


def test_zero_vector_stays_zero() -> None:
    """An all-zero input cannot produce a nonzero sign."""
    w = torch.zeros(4)
    assert torch.equal(T_tau(w, 1.0), torch.zeros(4))


def test_scale_invariance() -> None:
    """T_tau is positively scale-invariant."""
    w = torch.tensor([0.3, -1.7, 2.4, 0.2])
    assert torch.equal(T_tau(w * 5.0, 1.0), T_tau(w, 1.0))


def test_cols_applies_per_column() -> None:
    """Each column is ternarized with its own adaptive threshold."""
    matrix = torch.tensor([[2.0, 0.5], [-0.1, 0.4], [0.05, -3.0]])
    expected = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, -1.0]])
    assert torch.equal(T_tau_cols(matrix, 1.0), expected)


def test_cols_safeguard_per_column() -> None:
    """A column whose mask is empty keeps its own largest entry."""
    matrix = torch.tensor([[1.0, 5.0], [2.0, 0.1], [0.2, 0.2]])
    out = T_tau_cols(matrix, 10.0)
    assert torch.equal(out, torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]))


@pytest.mark.parametrize("tau", [0.0, -1.0])
def test_invalid_tau_raises(tau: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        T_tau(torch.ones(3), tau)


def test_wrong_ndim_raises() -> None:
    with pytest.raises(ValueError, match="1-D"):
        T_tau(torch.ones(2, 2), 1.0)


def test_non_finite_raises() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        T_tau(torch.tensor([1.0, float("nan")]), 1.0)
