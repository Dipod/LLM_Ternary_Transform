"""Integer group-quantization baselines and model storage accounting.

The quantizer is the group-128 RTN arm with MSE-optimal per-group clipping that
was sanity-checked in `results/bit_frontier/prototype_clip.py` (8-bit
perplexity-neutral), kept identical here so Stage A comparisons stay comparable
with the recorded Phase 0 numbers. Weights are replaced by their dequantized
values (fake quantization) so the same model object can be evaluated end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from phase1_compression.evaluation.accounting import StoredTensor, reference_storage

# Modules whose weights stay in the reference format: the vocabulary projection is
# bound to the embedding table and quantizing it is not part of the baseline.
DEFAULT_SKIP_SUFFIXES: tuple[str, ...] = ("lm_head",)


@dataclass(frozen=True)
class QuantizeReport:
    """Result of quantizing a model in place.

    Attributes:
        bits: Bit width applied.
        group_size: Elements per quantization group.
        quantized: Names of quantized linear modules.
        skipped: Names of linear modules skipped, with the reason.
    """

    bits: int
    group_size: int
    quantized: tuple[str, ...]
    skipped: tuple[str, ...]


def group_count(numel: int, group_size: int) -> int:
    """Return the number of quantization groups for ``numel`` elements.

    Args:
        numel: Number of elements.
        group_size: Elements per group.

    Returns:
        Ceiling of ``numel / group_size``.

    Raises:
        ValueError: If ``numel`` is negative or ``group_size`` is not positive.
    """
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    return -(-numel // group_size)


def clip_scales(W: Tensor, bits: int, group_size: int, n_clip: int) -> Tensor:
    """Quantize ``W`` with symmetric group scales and an MSE-optimal clip ratio.

    Ported from the recorded Phase 0 baseline: for each row and group the clip
    ratio that minimizes squared error is chosen from a fixed grid, because plain
    absmax group quantization leaves ~13% more weight error.

    Args:
        W: Weight matrix ``[out, in]``, any float dtype.
        bits: Bit width; the positive range is ``2**(bits-1) - 1``.
        group_size: Elements per group along the input dimension.
        n_clip: Number of clip ratios searched. ``1`` means pure absmax (ratio
            1.0); larger values search the grid from 0.4 to 1.0.

    Returns:
        The dequantized weight matrix, same shape and dtype as ``W``.

    Raises:
        ValueError: If ``in`` is not divisible by ``group_size`` or ``bits < 2``.
    """
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    if n_clip < 1:
        raise ValueError(f"n_clip must be >= 1, got {n_clip}")
    out_features, in_features = W.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} not divisible by group_size={group_size} "
            f"for weight {tuple(W.shape)}"
        )
    dtype = W.dtype
    qmax = 2 ** (bits - 1) - 1
    grouped = W.float().reshape(out_features, in_features // group_size, group_size)
    amax = grouped.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    best: Tensor | None = None
    best_err: Tensor | None = None
    # n_clip == 1 selects absmax (ratio 1.0); linspace would return [0.4] there.
    ratios = (
        torch.ones(1, device=W.device, dtype=torch.float32)
        if n_clip == 1
        else torch.linspace(0.4, 1.0, n_clip, device=W.device, dtype=torch.float32)
    )
    for c in ratios:
        scale = (c * amax / qmax).clamp_min(1e-12)
        q = (grouped / scale).round().clamp(-qmax - 1, qmax) * scale
        err = ((grouped - q) ** 2).sum(dim=2, keepdim=True)
        if best is None or best_err is None:
            best, best_err = q.clone(), err.clone()
        else:
            take = err < best_err
            best = torch.where(take, q, best)
            best_err = torch.where(take, err, best_err)
    assert best is not None  # n_clip >= 1 makes the loop body run at least once
    return best.reshape(out_features, in_features).to(dtype)


def iter_linear_modules(
    model: nn.Module, skip_suffixes: tuple[str, ...]
) -> list[tuple[str, nn.Linear]]:
    """Return named linear modules to quantize or account for.

    Args:
        model: The model to walk.
        skip_suffixes: Module-name suffixes excluded from quantization.

    Returns:
        ``(qualified_name, module)`` pairs in definition order.
    """
    selected: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not name.endswith(skip_suffixes):
            selected.append((name, module))
    return selected


def fake_quantize_(
    model: nn.Module,
    *,
    bits: int,
    group_size: int,
    n_clip: int,
    skip_suffixes: tuple[str, ...] = DEFAULT_SKIP_SUFFIXES,
) -> QuantizeReport:
    """Quantize every eligible linear weight in place (fake quantization).

    Args:
        model: Model mutated in place.
        bits: Bit width applied to every eligible weight.
        group_size: Elements per quantization group.
        n_clip: Clip-ratio search resolution.
        skip_suffixes: Module-name suffixes left in the reference format.

    Returns:
        A :class:`QuantizeReport` listing quantized and skipped modules.
    """
    quantized: list[str] = []
    skipped: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith(skip_suffixes):
            skipped.append(f"{name}: excluded by suffix filter")
            continue
        weight = module.weight
        if weight.shape[1] % group_size != 0:
            skipped.append(f"{name}: in_features={weight.shape[1]} not divisible by {group_size}")
            continue
        with torch.no_grad():
            module.weight.copy_(clip_scales(weight.detach(), bits, group_size, n_clip))
        quantized.append(name)
    return QuantizeReport(
        bits=bits,
        group_size=group_size,
        quantized=tuple(quantized),
        skipped=tuple(skipped),
    )


def quantized_model_storage(
    model: nn.Module,
    *,
    bits: int,
    group_size: int,
    skip_suffixes: tuple[str, ...] = DEFAULT_SKIP_SUFFIXES,
) -> list[StoredTensor]:
    """Return the stored arrays of a quantized model under a scheme.

    Args:
        model: Model whose structure describes the storage layout.
        bits: Bit width applied to eligible linear weights.
        group_size: Elements per quantization group.
        skip_suffixes: Module-name suffixes left in the reference format.

    Returns:
        One :class:`StoredTensor` per stored array.
    """
    tensors: list[StoredTensor] = []
    eligible = {name for name, _ in iter_linear_modules(model, skip_suffixes)}
    for name, param in model.named_parameters():
        owner = name[: -len(".weight")] if name.endswith(".weight") else name
        if owner not in eligible or param.dim() != 2 or param.shape[1] % group_size != 0:
            tensors.append(reference_storage(name, param.numel()))
            continue
        groups = group_count(param.numel(), group_size)
        tensors.append(StoredTensor(f"{name}.q{bits}", param.numel(), float(bits)))
        tensors.append(StoredTensor(f"{name}.scales", groups, 16.0))
    return tensors


def reference_model_storage(model: nn.Module) -> list[StoredTensor]:
    """Return the stored arrays of the model in the reference 16-bit format.

    Args:
        model: Model to account for.

    Returns:
        One :class:`StoredTensor` per parameter.
    """
    return [reference_storage(name, param.numel()) for name, param in model.named_parameters()]
