"""Physical structural removal for the combined probe.

Zeroing a unit does not reduce stored bytes; the probe must delete the
dimensions. Removal is therefore a re-shaping of the MLP matrices: an FFN
intermediate dimension is a row of ``gate_proj``/``up_proj`` and a column of
``down_proj``. Deleting a dimension whose contribution is already zero must leave
the layer output bit-identical, which is the known-answer test of this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

__all__ = ["kept_indices", "remove_ffn_dims_"]


def kept_indices(total: int, removed: Sequence[int]) -> list[int]:
    """Return the ascending indices that survive a removal.

    Args:
        total: Number of dimensions before removal.
        removed: Indices to drop.

    Returns:
        The surviving indices in ascending order.

    Raises:
        ValueError: If an index is out of range.
    """
    bad = [index for index in removed if index < 0 or index >= total]
    if bad:
        raise ValueError(f"removed indices out of range for total={total}: {bad[:5]}")
    drop = set(int(index) for index in removed)
    return [index for index in range(total) if index not in drop]


def _replacement_linear(old: nn.Linear, out_features: int, in_features: int) -> nn.Linear:
    """Create an empty linear layer matching ``old``'s device, dtype and bias.

    Args:
        old: Layer whose configuration is copied.
        out_features: Output size of the new layer.
        in_features: Input size of the new layer.

    Returns:
        A new :class:`torch.nn.Linear` on the same device and dtype.
    """
    new = nn.Linear(in_features, out_features, bias=old.bias is not None)
    return new.to(device=old.weight.device, dtype=old.weight.dtype)


def remove_ffn_dims_(model: Any, selection: Mapping[int, Sequence[int]]) -> dict[str, int]:
    """Physically remove FFN intermediate dimensions, in place.

    For every ``(layer, units)`` pair the rows ``units`` of ``gate_proj`` and
    ``up_proj`` and the columns ``units`` of ``down_proj`` are deleted, so the
    stored parameter count actually drops.

    Args:
        model: Causal LM exposing ``model.layers[i].mlp.{gate_proj,up_proj,down_proj}``.
        selection: Layer index to removed intermediate-dimension indices.

    Returns:
        A summary with ``layers`` touched, ``dims_removed`` and ``params_removed``.

    Raises:
        ValueError: If a selected layer is missing or an index is out of range.
    """
    layers = model.model.layers
    dims_removed = 0
    params_removed = 0
    for layer_index, units in sorted(selection.items()):
        if not 0 <= layer_index < len(layers):
            raise ValueError(f"layer {layer_index} out of range for {len(layers)} layers")
        mlp = layers[layer_index].mlp
        gate: nn.Linear = mlp.gate_proj
        up: nn.Linear = mlp.up_proj
        down: nn.Linear = mlp.down_proj
        total = int(gate.weight.shape[0])
        keep = kept_indices(total, units)
        if len(keep) == total:
            continue
        with torch.no_grad():
            new_gate = _replacement_linear(gate, len(keep), int(gate.weight.shape[1]))
            new_up = _replacement_linear(up, len(keep), int(up.weight.shape[1]))
            new_down = _replacement_linear(down, int(down.weight.shape[0]), len(keep))
            index = torch.tensor(keep, device=gate.weight.device)
            new_gate.weight.copy_(gate.weight.index_select(0, index))
            new_up.weight.copy_(up.weight.index_select(0, index))
            new_down.weight.copy_(down.weight.index_select(1, index))
            if gate.bias is not None:
                new_gate.bias.copy_(gate.bias.index_select(0, index))
            if up.bias is not None:
                new_up.bias.copy_(up.bias.index_select(0, index))
        mlp.gate_proj = new_gate
        mlp.up_proj = new_up
        mlp.down_proj = new_down
        dims_removed += total - len(keep)
        params_removed += (total - len(keep)) * (
            int(gate.weight.shape[1]) * 2 + int(down.weight.shape[0])
        )
    return {
        "layers": len(selection),
        "dims_removed": dims_removed,
        "params_removed": params_removed,
    }
