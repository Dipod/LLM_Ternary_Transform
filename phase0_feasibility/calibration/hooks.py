"""Forward-hook capture of ``gate_proj`` inputs and activation statistics.

The hook captures the input to the target projection (post-RMSNorm) with
gradients globally disabled (design.md D6/D7).
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch import Tensor


class _CaptureDoneError(Exception):
    """Sentinel raised from the hook to stop the forward after the target layer."""


def capture_gate_inputs(
    model: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    layer_path: str,
    batch_size: int,
    early_stop: bool = True,
) -> Tensor:
    """Capture the input activations of a target projection across a slice.

    The model is run in ``eval`` mode under ``torch.no_grad`` in document
    batches; padding positions are dropped using ``attention_mask``.

    With ``early_stop`` (default) the forward pass is aborted as soon as the
    target module runs. This is exact, not approximate: the input of a module at
    layer ``l`` depends only on layers ``< l``, so the captured tensor is
    identical to a full forward; only the unneeded later layers are skipped.

    Args:
        model: Causal LM (HuggingFace boundary type).
        input_ids: Token ids ``[n_documents, seq_len]`` int64.
        attention_mask: Attention mask ``[n_documents, seq_len]`` int64.
        layer_path: Dotted module path, see
            :func:`phase0_feasibility.calibration.loader.layer_module_path`.
        batch_size: Documents per forward pass; must be >= 1.
        early_stop: Stop each forward once the target module has fired.

    Returns:
        Captured activations as float32 ``[tokens, hidden]``, pad positions
        removed, in document order.

    Raises:
        ValueError: If ``batch_size`` < 1, ids/mask shapes disagree, or no
            activation was captured.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"input_ids {tuple(input_ids.shape)} and mask {tuple(attention_mask.shape)} differ"
        )

    target = model.get_submodule(layer_path)
    captured: list[Tensor] = []

    def _pre_hook(_module: Any, args: tuple[Any, ...]) -> None:
        captured.append(args[0].detach())
        if early_stop:
            raise _CaptureDoneError

    handle = target.register_forward_pre_hook(_pre_hook)
    chunks: list[Tensor] = []
    try:
        with torch.no_grad():
            n_documents = int(input_ids.shape[0])
            for start in range(0, n_documents, batch_size):
                stop = min(start + batch_size, n_documents)
                ids = input_ids[start:stop]
                mask = attention_mask[start:stop]
                del captured[:]
                with contextlib.suppress(_CaptureDoneError):
                    model(input_ids=ids, attention_mask=mask)
                if not captured:
                    raise RuntimeError(f"forward hook on {layer_path!r} captured nothing")
                batch_activations = captured[0]
                valid = mask.to(torch.bool)
                chunks.append(batch_activations[valid].detach().to(torch.float32))
    finally:
        handle.remove()

    if not chunks:
        raise ValueError("no activations captured; input slice was empty")
    activations = torch.cat(chunks, dim=0)
    if not bool(torch.isfinite(activations).all()):
        raise ValueError("captured activations contain non-finite values")
    return activations


def activation_second_moment(activations: Tensor) -> Tensor:
    """Per-input-channel activation second moment ``h_j = E[x_j^2]``.

    Computed with gradients disabled; the result never requires grad.

    Args:
        activations: Captured activations ``[tokens, hidden]``.

    Returns:
        Float32 tensor ``[hidden]`` with ``h_j = mean_tokens(x_j^2)``.

    Raises:
        ValueError: If ``activations`` is not 2-D or is empty.
    """
    if activations.ndim != 2:
        raise ValueError(f"expected 2-D activations, got {tuple(activations.shape)}")
    if activations.numel() == 0:
        raise ValueError("activation tensor is empty")
    with torch.no_grad():
        x = activations.detach().to(torch.float32)
        return (x * x).mean(dim=0)
