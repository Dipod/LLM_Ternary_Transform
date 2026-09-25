"""Structural removal tests: the zero-contribution known-answer case (tasks 4.1-4.2)."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from phase1_compression.transform.structural import kept_indices, remove_ffn_dims_


class _Mlp(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.up_proj(x) * torch.nn.functional.silu(self.gate_proj(x)))


class _Layer(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.mlp = _Mlp(hidden, intermediate)


class _Inner(nn.Module):
    def __init__(self, layers: int, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(hidden, intermediate) for _ in range(layers)])


class _Tiny(nn.Module):
    def __init__(self, layers: int = 2, hidden: int = 16, intermediate: int = 32) -> None:
        super().__init__()
        self.model: Any = _Inner(layers, hidden, intermediate)


def test_kept_indices_and_bounds() -> None:
    assert kept_indices(5, [1, 3]) == [0, 2, 4]
    assert kept_indices(3, []) == [0, 1, 2]
    with pytest.raises(ValueError, match="out of range"):
        kept_indices(3, [7])


def test_removing_zero_contribution_dims_preserves_the_output() -> None:
    torch.manual_seed(0)
    model = _Tiny(layers=2, hidden=16, intermediate=32)
    x = torch.randn(4, 16)
    with torch.no_grad():
        # Units 1 and 5 contribute nothing after zeroing, so deleting them must be
        # exactly output-preserving.
        for layer in model.model.layers:
            layer.mlp.gate_proj.weight[1].zero_()
            layer.mlp.up_proj.weight[1].zero_()
            layer.mlp.down_proj.weight[:, 1].zero_()
            layer.mlp.gate_proj.weight[5].zero_()
            layer.mlp.up_proj.weight[5].zero_()
            layer.mlp.down_proj.weight[:, 5].zero_()
    before = [layer.mlp(x) for layer in model.model.layers]
    summary = remove_ffn_dims_(model, {0: [1, 5], 1: [1, 5]})
    after = [layer.mlp(x) for layer in model.model.layers]
    assert summary["dims_removed"] == 4
    assert model.model.layers[0].mlp.gate_proj.weight.shape == (30, 16)
    assert model.model.layers[0].mlp.down_proj.weight.shape == (16, 30)
    for old, new in zip(before, after, strict=True):
        assert torch.allclose(old, new, atol=1e-6)


def test_removing_live_dims_changes_the_output() -> None:
    torch.manual_seed(1)
    model = _Tiny(layers=1, hidden=16, intermediate=32)
    x = torch.randn(4, 16)
    before = model.model.layers[0].mlp(x)
    remove_ffn_dims_(model, {0: [0]})
    after = model.model.layers[0].mlp(x)
    assert not torch.allclose(before, after, atol=1e-3)


def test_params_removed_matches_shapes() -> None:
    model = _Tiny(layers=1, hidden=16, intermediate=32)
    summary = remove_ffn_dims_(model, {0: [0, 1, 2]})
    assert summary["params_removed"] == 3 * (16 * 2 + 16)
    assert summary["layers"] == 1


def test_out_of_range_layer_raises() -> None:
    model = _Tiny(layers=1)
    with pytest.raises(ValueError, match="out of range"):
        remove_ffn_dims_(model, {5: [0]})
