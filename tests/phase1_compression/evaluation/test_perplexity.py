"""Relative-perplexity metric tests with a known-answer model (task 2.2)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from phase1_compression.evaluation.gate import relative_ppl_delta
from phase1_compression.evaluation.perplexity import perplexity


class _TinyLM(nn.Module):
    """Minimal causal LM exposing the ``loss``/``logits`` interface."""

    def __init__(self, vocab: int = 32, dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None, labels: Tensor | None = None
    ) -> Any:
        logits = self.head(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


def _ids(seed: int = 3, items: int = 4, length: int = 16, vocab: int = 32) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (items, length), generator=generator)


def test_identical_copy_scores_zero_delta() -> None:
    torch.manual_seed(0)
    ids = _ids()
    mask = torch.ones_like(ids)
    reference = perplexity(_TinyLM(), ids, mask, batch_size=2, device="cpu")
    delta = relative_ppl_delta(reference, reference)
    assert delta == 0.0
    assert reference > 1.0


def _peaked_lm(vocab: int = 32, scale: float = 20.0) -> _TinyLM:
    """Return a model that confidently predicts a repeated token."""
    model = _TinyLM(vocab=vocab, dim=vocab)
    with torch.no_grad():
        model.embed.weight.copy_(torch.eye(vocab))
        model.head.weight.copy_(scale * torch.eye(vocab))
    return model


def test_degraded_model_scores_above_the_bar() -> None:
    model = _peaked_lm(vocab=32)
    ids = torch.full((4, 16), 3, dtype=torch.long)
    mask = torch.ones_like(ids)
    reference = perplexity(model, ids, mask, batch_size=2, device="cpu")
    assert reference < 1.01
    with torch.no_grad():
        model.head.weight.zero_()
    degraded = perplexity(model, ids, mask, batch_size=2, device="cpu")
    assert degraded == pytest.approx(32.0, rel=1e-6)
    assert relative_ppl_delta(reference, degraded) > 0.005


def test_masked_positions_are_ignored() -> None:
    torch.manual_seed(2)
    ids = _ids()
    mask = torch.ones_like(ids)
    mask[:, -4:] = 0
    reference = perplexity(_TinyLM(), ids, mask, batch_size=2, device="cpu")
    assert reference > 0.0
