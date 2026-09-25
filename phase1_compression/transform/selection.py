"""Target-model unit selection for the combined probe (task 4.1).

The cartography fixes the *procedure* and the removal fraction; the unit list is
recomputed on the target model with the same Wanda-style score. Scores are
accumulated inside forward hooks, so no activations are stored and the whole 8B
model can be scored in a single pass over the slice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

__all__ = ["rank_ffn_dims", "select_lowest_fraction"]


def rank_ffn_dims(
    model: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    batch_size: int,
    device: str,
) -> list[tuple[int, int, float]]:
    """Rank FFN intermediate dimensions of the target model, least important first.

    The score is ``sum_i |down_proj[i, j]| * ||X[:, j]|| / sqrt(N)``, the same rule
    the cartography used, where ``X`` is the ``down_proj`` input activation
    accumulated over the slice.

    Args:
        model: Causal LM with ``model.layers[i].mlp.down_proj``.
        input_ids: Token ids ``[items, seq_len]``.
        attention_mask: Mask ``[items, seq_len]``.
        batch_size: Items per forward pass.
        device: Torch device string.

    Returns:
        Tuples ``(layer, unit, score)`` sorted ascending by score.

    Raises:
        ValueError: If no token is available.
    """
    layers = model.model.layers
    sums: dict[int, Tensor] = {}
    handles: list[Any] = []

    def make_hook(layer_index: int, width: int) -> Any:
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            activations = args[0].detach().float()
            partial = activations.pow(2).sum(dim=tuple(range(activations.dim() - 1)))
            if layer_index not in sums:
                sums[layer_index] = torch.zeros(width, dtype=torch.float32, device=partial.device)
            sums[layer_index] += partial

        return hook

    for layer_index, layer in enumerate(layers):
        down = layer.mlp.down_proj
        handles.append(
            down.register_forward_pre_hook(make_hook(layer_index, int(down.weight.shape[1])))
        )
    tokens = 0
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, input_ids.shape[0], batch_size):
                ids = input_ids[start : start + batch_size].to(device)
                mask = attention_mask[start : start + batch_size].to(device)
                model(input_ids=ids, attention_mask=mask)
                tokens += int(mask.sum())
    finally:
        for handle in handles:
            handle.remove()
    if tokens == 0:
        raise ValueError("no tokens in the selection slice")
    scored: list[tuple[int, int, float]] = []
    with torch.no_grad():
        for layer_index, layer in enumerate(layers):
            col_norm = (sums[layer_index] / tokens).clamp_min(0.0).sqrt()
            weight = layer.mlp.down_proj.weight.detach().float()
            score = (weight.abs() * col_norm.unsqueeze(0)).sum(dim=0)
            for unit in range(score.numel()):
                scored.append((layer_index, unit, float(score[unit])))
    scored.sort(key=lambda item: item[2])
    return scored


def select_lowest_fraction(
    ranking: Sequence[tuple[int, int, float]], fraction: float
) -> dict[int, list[int]]:
    """Select the lowest-scoring ``fraction`` of units, grouped by layer.

    Args:
        ranking: Ranking from :func:`rank_ffn_dims`, ascending by score.
        fraction: Fraction of units to remove, in ``[0, 1)``.

    Returns:
        Layer index to removed unit indices.

    Raises:
        ValueError: If ``fraction`` is outside ``[0, 1)`` or the ranking is empty.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")
    if not ranking:
        raise ValueError("ranking is empty")
    count = round(fraction * len(ranking))
    selection: dict[int, list[int]] = {}
    for layer_index, unit, _score in ranking[:count]:
        selection.setdefault(layer_index, []).append(unit)
    return selection


def total_units(selection: Mapping[int, Sequence[int]]) -> int:
    """Return the number of selected units.

    Args:
        selection: Layer index to removed unit indices.

    Returns:
        The total count.
    """
    return sum(len(units) for units in selection.values())
