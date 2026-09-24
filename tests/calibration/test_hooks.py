"""Tests for activation capture and the second moment (tasks 3.2, 3.3).

These use a tiny fake model so they run without any download.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from phase0_feasibility.calibration.hooks import (
    activation_second_moment,
    capture_gate_inputs,
)
from phase0_feasibility.calibration.loader import layer_module_path, resolve_dtype

HIDDEN = 7


class _FakeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, HIDDEN)


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _FakeMLP()


class _FakeLayers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])


class _FakeGateModel(nn.Module):
    """Minimal model exposing ``model.layers.N.mlp.gate_proj`` and a chained forward."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeLayers()

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        del attention_mask
        hidden = F.one_hot(input_ids.clamp(0, HIDDEN - 1), num_classes=HIDDEN).to(torch.float32)
        for index in range(len(self.model.layers)):
            # ModuleList indexing is typed as Module; the cast states the concrete module.
            gate = cast(nn.Linear, self.get_submodule(f"model.layers.{index}.mlp.gate_proj"))
            hidden = gate(hidden)
        return hidden


def test_resolve_dtype() -> None:
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype("float32") is torch.float32


def test_layer_module_path() -> None:
    assert layer_module_path(12, "gate_proj") == "model.layers.12.mlp.gate_proj"


def test_capture_filters_padding() -> None:
    model = _FakeGateModel()
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.long)
    activations = capture_gate_inputs(
        model,
        input_ids,
        attention_mask,
        layer_path="model.layers.0.mlp.gate_proj",
        batch_size=1,
    )
    assert activations.shape == (4, HIDDEN)
    assert activations.dtype == torch.float32
    assert not activations.requires_grad


def test_early_stop_matches_full_forward() -> None:
    """The exact early stop leaves the captured tensor unchanged (cross-check gate)."""
    model = _FakeGateModel()
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
    full = capture_gate_inputs(
        model,
        input_ids,
        attention_mask,
        layer_path="model.layers.0.mlp.gate_proj",
        batch_size=1,
        early_stop=False,
    )
    early = capture_gate_inputs(
        model,
        input_ids,
        attention_mask,
        layer_path="model.layers.0.mlp.gate_proj",
        batch_size=1,
        early_stop=True,
    )
    assert torch.equal(full, early)


def test_capture_rejects_bad_batch_size() -> None:
    model = _FakeGateModel()
    ids = torch.zeros((1, 3), dtype=torch.long)
    try:
        capture_gate_inputs(
            model, ids, ids, layer_path="model.layers.0.mlp.gate_proj", batch_size=0
        )
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_activation_second_moment_matches_manual() -> None:
    torch.manual_seed(0)
    activations = torch.randn(5, 3)
    moment = activation_second_moment(activations)
    expected = (activations * activations).mean(dim=0)
    assert torch.allclose(moment, expected)
    assert moment.shape == (3,)
    assert not moment.requires_grad


def test_second_moment_does_not_track_gradients() -> None:
    """The statistic is computed with autograd disabled (task 3.3)."""
    activations = torch.randn(4, 2, requires_grad=True)
    moment = activation_second_moment(activations)
    assert not moment.requires_grad
    assert moment.grad_fn is None
