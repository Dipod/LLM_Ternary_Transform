"""KL-divergence probe used by the cartography axes.

The probe compares a modified model against a reference model on a fixed benign
slice; KL in nats is the dense, cheap proxy for output-distribution damage that
the directional-ablation literature uses (design.md D6).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["mean_kl"]


def _shifted_logits(model: nn.Module, ids: Tensor, mask: Tensor) -> Tensor:
    """Return next-token logits ``[items * (seq_len - 1), vocab]``.

    Args:
        model: Causal LM in evaluation mode.
        ids: Token ids ``[items, seq_len]`` on the model device.
        mask: Attention mask ``[items, seq_len]`` on the model device.

    Returns:
        Logits for every shifted position, flattened over items and positions.
    """
    out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits[:, :-1, :]
    return logits.reshape(-1, logits.shape[-1])


def mean_kl(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    batch_size: int,
    device: str,
    chunk_tokens: int = 2048,
) -> float:
    """Return the mean KL divergence ``KL(reference || candidate)`` in nats.

    Args:
        reference_model: Unmodified model.
        candidate_model: Model with the removal applied.
        input_ids: Token ids ``[items, seq_len]`` int64.
        attention_mask: Mask ``[items, seq_len]`` int64.
        batch_size: Items per forward pass.
        device: Torch device string.
        chunk_tokens: Rows of the flattened logit matrix processed at once, which
            bounds the fp32 memory used by ``log_softmax``.

    Returns:
        The mean KL divergence over predicted positions, in nats.

    Raises:
        ValueError: If shapes differ or no predicted position is available.
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"shape mismatch: ids {tuple(input_ids.shape)} vs mask {tuple(attention_mask.shape)}"
        )
    total_kl = 0.0
    total_rows = 0
    reference_model.eval()
    candidate_model.eval()
    with torch.inference_mode():
        for start in range(0, input_ids.shape[0], batch_size):
            ids = input_ids[start : start + batch_size].to(device)
            mask = attention_mask[start : start + batch_size].to(device)
            ref = _shifted_logits(reference_model, ids, mask)
            cand = _shifted_logits(candidate_model, ids, mask)
            rows = ref.shape[0]
            for lo in range(0, rows, chunk_tokens):
                hi = min(lo + chunk_tokens, rows)
                logp_ref = torch.log_softmax(ref[lo:hi].float(), dim=-1)
                logp_cand = torch.log_softmax(cand[lo:hi].float(), dim=-1)
                kl = (logp_ref.exp() * (logp_ref - logp_cand)).sum(dim=-1)
                total_kl += float(kl.sum())
            total_rows += rows
    if total_rows == 0:
        raise ValueError("no predicted positions in the KL slice")
    return total_kl / total_rows
