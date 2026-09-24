"""Known-answer tests for the Phase 0 metrics (task 4.1)."""

from __future__ import annotations

import pytest
import torch

from phase0_feasibility.analysis.metrics import (
    effective_bpw,
    energy_preserved,
    factor_sparsity,
    layer_output_error,
    metric_validity_check,
)


def test_layer_output_error_known_answer() -> None:
    """Hand-computed: X W^T = [[1]], X W_hat^T = [[0.5]] -> E_x = 0.5."""
    X = torch.tensor([[1.0, 0.0]])
    W = torch.tensor([[1.0, 0.0]])
    W_hat = torch.tensor([[0.5, 0.0]])
    assert layer_output_error(X, W, W_hat) == pytest.approx(0.5)


def test_perfect_reconstruction_error_is_zero() -> None:
    generator = torch.Generator().manual_seed(31)
    X = torch.randn(8, 5, generator=generator)
    W = torch.randn(7, 5, generator=generator)
    assert layer_output_error(X, W, W) <= 1e-6


def test_destroyed_reconstruction_error_at_least_one() -> None:
    generator = torch.Generator().manual_seed(37)
    X = torch.randn(256, 32, generator=generator)
    W = torch.randn(48, 32, generator=generator)
    destroyed = torch.randn(48, 32, generator=generator)
    assert layer_output_error(X, W, destroyed) >= 1.0


def test_energy_preserved_known_answer() -> None:
    W = torch.tensor([[1.0, 0.0]])
    W_hat = torch.tensor([[0.5, 0.0]])
    assert energy_preserved(W, W_hat) == pytest.approx(0.75)


def test_factor_sparsity_known_answer() -> None:
    B = torch.tensor([[1, 0], [0, -1]], dtype=torch.int8)
    C = torch.tensor([[1, 1, 1]], dtype=torch.int8)
    assert factor_sparsity(B, C) == pytest.approx(2 / 7)


def test_effective_bpw_dense_is_positive_and_finite() -> None:
    value = effective_bpw(4, 2, 4, 2.0, 0.0)
    assert value > 0.0
    assert torch.isfinite(torch.tensor(value))


def test_more_planes_give_larger_bpw_at_equal_sparsity() -> None:
    one_plane = effective_bpw(4, 2, 2, 1.0, 0.25)
    two_planes = effective_bpw(4, 2, 4, 2.0, 0.25)
    assert two_planes > one_plane


def test_effective_bpw_rejects_inconsistent_rank() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        effective_bpw(4, 2, 7, 2.0, 0.1)


def test_effective_bpw_rejects_bad_sparsity() -> None:
    with pytest.raises(ValueError, match="sparsity"):
        effective_bpw(4, 2, 4, 2.0, 1.5)


def test_metric_validity_check_confirms_metric_can_fail() -> None:
    generator = torch.Generator().manual_seed(41)
    X = torch.randn(200, 24, generator=generator)
    W = torch.randn(36, 24, generator=generator)
    good_passes, bad_fails, good_e_x, bad_e_x = metric_validity_check(
        X, W, seed=42, good_tol=1e-4, ceiling_tol=1e-6
    )
    assert good_e_x <= 1e-4
    assert bad_e_x >= 1.0 - 1e-6
    assert good_passes
    assert bad_fails


def test_layer_output_error_rejects_zero_reference() -> None:
    X = torch.zeros(2, 3)
    W = torch.ones(4, 3)
    with pytest.raises(ValueError, match="zero"):
        layer_output_error(X, W, W)
