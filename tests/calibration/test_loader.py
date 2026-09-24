"""Non-network tests for slice construction and the fit/held-out split.

The dense-window tokenizer is duck typed, so windowing and the split are checked
without downloading the model or the dataset (tasks 2.1, 2.3).
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from phase0_feasibility.calibration.loader import build_dense_windows, split_slice


class _FakeTokenizer:
    """Minimal tokenizer: returns ``n_tokens`` consecutive ids as one row."""

    def __init__(self, n_tokens: int) -> None:
        self.n_tokens = n_tokens

    def __call__(
        self,
        text: str,
        truncation: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, Tensor]:
        del text, truncation, return_tensors
        ids = torch.arange(self.n_tokens, dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids}


class _FlatTokenizer:
    """Tokenizer whose ``input_ids`` is 1-D (some tokenizers return this form)."""

    def __call__(
        self,
        text: str,
        truncation: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, Tensor]:
        del text, truncation, return_tensors
        return {"input_ids": torch.arange(20, dtype=torch.long)}


def test_build_dense_windows_is_token_exact_and_unpadded() -> None:
    """128 windows x 512 tokens = 65,536 tokens, mask all ones (spec scenario)."""
    tokenizer = _FakeTokenizer(200_000)
    windows, mask = build_dense_windows(tokenizer, "text", seq_len=512, n_windows=128)
    assert windows.shape == (128, 512)
    assert mask.shape == (128, 512)
    assert windows.dtype == torch.long
    assert int(mask.sum()) == 128 * 512 == 65_536
    assert bool(torch.all(mask == 1))


def test_build_dense_windows_cuts_the_stream_in_order() -> None:
    """The windows are consecutive, non-overlapping slices of the token stream."""
    tokenizer = _FakeTokenizer(1_000)
    windows, _mask = build_dense_windows(tokenizer, "text", seq_len=4, n_windows=3)
    expected = torch.arange(12, dtype=torch.long).reshape(3, 4)
    assert torch.equal(windows, expected)


def test_build_dense_windows_accepts_flat_token_ids() -> None:
    """A 1-D ``input_ids`` output is accepted like the 2-D batched form."""
    windows, _mask = build_dense_windows(_FlatTokenizer(), "text", seq_len=5, n_windows=2)
    assert windows.shape == (2, 5)


def test_build_dense_windows_rejects_short_stream() -> None:
    tokenizer = _FakeTokenizer(100)
    with pytest.raises(ValueError, match="need"):
        build_dense_windows(tokenizer, "text", seq_len=512, n_windows=1)


def test_build_dense_windows_rejects_bad_counts() -> None:
    tokenizer = _FakeTokenizer(1_000)
    with pytest.raises(ValueError, match="seq_len"):
        build_dense_windows(tokenizer, "text", seq_len=0, n_windows=1)
    with pytest.raises(ValueError, match="n_windows"):
        build_dense_windows(tokenizer, "text", seq_len=4, n_windows=0)


def test_split_slice_is_contiguous_and_reproducible() -> None:
    """The split keeps order, uses no RNG and repeats identically (spec scenario)."""
    ids = torch.arange(20, dtype=torch.long).reshape(10, 2)
    mask = torch.ones_like(ids)
    fit_ids, fit_mask, held_ids, held_mask = split_slice(ids, mask, 0.2)
    assert fit_ids.shape[0] == 8
    assert held_ids.shape[0] == 2
    assert torch.equal(fit_ids, ids[:8])
    assert torch.equal(fit_mask, mask[:8])
    assert torch.equal(held_ids, ids[8:])
    assert torch.equal(held_mask, mask[8:])

    again = split_slice(ids, mask, 0.2)
    assert torch.equal(fit_ids, again[0])
    assert torch.equal(held_ids, again[2])


def test_split_slice_uses_round_of_the_remainder() -> None:
    """round((1 - 0.2) * 1024) = round(819.2) = 819 fit items."""
    ids = torch.zeros((1024, 1), dtype=torch.long)
    mask = torch.ones_like(ids)
    fit_ids, _fit_mask, held_ids, _held_mask = split_slice(ids, mask, 0.2)
    assert fit_ids.shape[0] == 819
    assert held_ids.shape[0] == 205


def test_split_slice_rejects_bad_fraction() -> None:
    ids = torch.zeros((4, 1), dtype=torch.long)
    mask = torch.ones_like(ids)
    with pytest.raises(ValueError, match="eval_fraction"):
        split_slice(ids, mask, 1.0)


def test_split_slice_rejects_empty_portion() -> None:
    ids = torch.zeros((1, 1), dtype=torch.long)
    mask = torch.ones_like(ids)
    with pytest.raises(ValueError, match="empty portion"):
        split_slice(ids, mask, 0.2)
