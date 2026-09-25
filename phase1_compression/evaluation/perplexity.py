"""Perplexity evaluation on a fixed token slice (spec phase1/compression-gate R1, R5).

The slice is dense (full windows, no padding), so the masking below is a
defensive no-op for it and the token count is exact.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from phase1_compression.evaluation.gate import relative_ppl_delta

__all__ = ["perplexity", "relative_ppl_delta", "sequence_nll"]


def sequence_nll(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    batch_size: int,
    device: str,
) -> tuple[float, int]:
    """Return the summed negative log-likelihood and the predicted-token count.

    Args:
        model: Causal LM in evaluation mode.
        input_ids: Token ids ``[items, seq_len]`` int64.
        attention_mask: Mask ``[items, seq_len]`` int64; zeros are ignored.
        batch_size: Items per forward pass.
        device: Torch device string.

    Returns:
        ``(sum_nll, n_tokens)`` where ``n_tokens`` is the number of predicted
        (shifted, unmasked) positions.

    Raises:
        ValueError: If ``input_ids`` and ``attention_mask`` shapes differ.
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"shape mismatch: ids {tuple(input_ids.shape)} vs mask {tuple(attention_mask.shape)}"
        )
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, input_ids.shape[0], batch_size):
            ids = input_ids[start : start + batch_size].to(device)
            mask = attention_mask[start : start + batch_size].to(device)
            labels = ids.masked_fill(mask == 0, -100)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss
            n_pred = int((labels[:, 1:] != -100).sum())
            total_nll += float(loss) * n_pred
            total_tokens += n_pred
    return total_nll, total_tokens


def perplexity(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    batch_size: int,
    device: str,
) -> float:
    """Return the perplexity of ``model`` on the slice.

    Args:
        model: Causal LM in evaluation mode.
        input_ids: Token ids ``[items, seq_len]`` int64.
        attention_mask: Mask ``[items, seq_len]`` int64.
        batch_size: Items per forward pass.
        device: Torch device string.

    Returns:
        ``exp(mean negative log-likelihood)`` over predicted tokens.

    Raises:
        ValueError: If no predicted token is available.
    """
    total_nll, total_tokens = sequence_nll(
        model, input_ids, attention_mask, batch_size=batch_size, device=device
    )
    if total_tokens == 0:
        raise ValueError("no predicted tokens in the slice")
    return math.exp(total_nll / total_tokens)
