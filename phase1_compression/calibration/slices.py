"""Slice construction, delegated to the Phase 0 builder for comparability.

Spec ``phase1/redundancy-cartography`` R6 requires the cartography to use the
Phase 0 calibration definition (WikiText-2, ``dense_windows``, ``seq_len = 512``,
128 windows, 65,536 tokens, pad positions dropped).
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from phase0_feasibility.calibration.loader import (
    build_dense_windows,
    load_text_stream,
)
from phase1_compression.config import Config

__all__ = ["build_slice", "calibration_slice", "ppl_slice"]


def build_slice(
    tokenizer: Any,
    *,
    dataset_id: str,
    dataset_config: str,
    split: str,
    seq_len: int,
    n_windows: int,
) -> tuple[Tensor, Tensor]:
    """Build dense token windows from a dataset split.

    Args:
        tokenizer: HuggingFace tokenizer (or duck-typed equivalent).
        dataset_id: Dataset identifier.
        dataset_config: Dataset configuration name.
        split: Split name.
        seq_len: Window length in tokens.
        n_windows: Number of complete windows to keep.

    Returns:
        ``(input_ids, attention_mask)``, int64 ``[n_windows, seq_len]`` with no
        padding.
    """
    text = load_text_stream(dataset_id, dataset_config, split)
    return build_dense_windows(tokenizer, text, seq_len=seq_len, n_windows=n_windows)


def calibration_slice(config: Config, tokenizer: Any) -> tuple[Tensor, Tensor]:
    """Build the fixed calibration slice shared with Phase 0.

    Args:
        config: Run configuration.
        tokenizer: HuggingFace tokenizer.

    Returns:
        ``(input_ids, attention_mask)``, int64 ``[128, 512]``.
    """
    return build_slice(
        tokenizer,
        dataset_id=config.dataset_id,
        dataset_config=config.dataset_config,
        split=config.calibration_split,
        seq_len=config.seq_len,
        n_windows=config.dense_windows_count,
    )


def ppl_slice(config: Config, tokenizer: Any) -> tuple[Tensor, Tensor]:
    """Build the fixed perplexity slice from the test split.

    Args:
        config: Run configuration.
        tokenizer: HuggingFace tokenizer.

    Returns:
        ``(input_ids, attention_mask)``, int64 ``[ppl_windows_count, seq_len]``.
    """
    return build_slice(
        tokenizer,
        dataset_id=config.dataset_id,
        dataset_config=config.dataset_config,
        split=config.ppl_split,
        seq_len=config.seq_len,
        n_windows=config.ppl_windows_count,
    )
