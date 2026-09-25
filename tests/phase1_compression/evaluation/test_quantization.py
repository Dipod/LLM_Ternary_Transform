"""Integer baseline quantizer and storage tests (tasks 2.4, 2.5)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from phase1_compression.evaluation.accounting import stored_bytes
from phase1_compression.evaluation.quantization import (
    clip_scales,
    fake_quantize_,
    group_count,
    quantized_model_storage,
    reference_model_storage,
)


class _Tiny(nn.Module):
    def __init__(self, hidden: int = 256, vocab: int = 256) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Linear(hidden, hidden, bias=False)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)


def _weight(seed: int = 0, *, with_outlier: bool = True) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(64, 256, generator=generator)
    if with_outlier:
        weight[:, 0] = 25.0
    return weight


def test_group_count() -> None:
    assert group_count(256, 128) == 2
    assert group_count(255, 128) == 2
    with pytest.raises(ValueError, match="non-negative"):
        group_count(-1, 128)


def test_clip_search_is_not_worse_than_absmax() -> None:
    weight = _weight()
    optimal = clip_scales(weight, bits=4, group_size=128, n_clip=24)
    absmax = clip_scales(weight, bits=4, group_size=128, n_clip=1)
    assert optimal.shape == weight.shape
    assert optimal.dtype == weight.dtype
    assert ((optimal - weight) ** 2).sum() <= ((absmax - weight) ** 2).sum()


def test_eight_bit_is_near_identity() -> None:
    weight = _weight(seed=1, with_outlier=False)
    quantized = clip_scales(weight, bits=8, group_size=128, n_clip=24)
    rel_err = float(torch.linalg.norm(quantized - weight) / torch.linalg.norm(weight))
    assert rel_err < 0.01


def test_n_clip_one_is_absmax() -> None:
    # Regression: torch.linspace(0.4, 1.0, 1) yields [0.4], which used to make the
    # "absmax" diagnostic clip the weights by 60% and destroy the model.
    weight = _weight(seed=3, with_outlier=False)
    result = clip_scales(weight, bits=8, group_size=128, n_clip=1)
    qmax = 127
    grouped = weight.reshape(64, 2, 128)
    amax = grouped.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    scale = amax / qmax
    expected = ((grouped / scale).round().clamp(-qmax - 1, qmax) * scale).reshape(64, 256)
    assert torch.allclose(result, expected, atol=1e-6)
    scale_04 = (0.4 * amax) / qmax
    clipped_04 = (grouped / scale_04).round().clamp(-qmax - 1, qmax) * scale_04
    assert ((result - weight) ** 2).sum() < ((clipped_04 - grouped) ** 2).sum()


def test_n_clip_validation() -> None:
    with pytest.raises(ValueError, match="n_clip"):
        clip_scales(torch.randn(4, 128), bits=8, group_size=128, n_clip=0)


def test_non_divisible_group_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        clip_scales(torch.randn(4, 100), bits=4, group_size=128, n_clip=4)


def test_fake_quantize_keeps_lm_head_and_storage_spends_scales() -> None:
    model = _Tiny()
    reference = stored_bytes(reference_model_storage(model))
    report = fake_quantize_(model, bits=4, group_size=128, n_clip=8)
    assert set(report.quantized) == {"q_proj", "mlp"}
    assert any(name.startswith("lm_head") for name in report.skipped)
    compressed = stored_bytes(quantized_model_storage(model, bits=4, group_size=128))
    assert compressed < reference
    assert any(
        t.name.endswith(".scales") for t in quantized_model_storage(model, bits=4, group_size=128)
    )
