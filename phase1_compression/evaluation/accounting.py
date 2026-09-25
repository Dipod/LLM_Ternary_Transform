"""Stored-byte accounting for compressed models (spec phase1/compression-gate R2).

Every tensor needed to run a model is counted, including scales, codebooks and
indices, so a lower nominal bit width cannot score better by hiding auxiliary
storage. Counts are exact storage bits; the byte total rounds up.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredTensor:
    """One stored array of a compressed model.

    Attributes:
        name: Human-readable identifier, e.g. ``layer3.q_proj.weight``.
        numel: Number of stored elements.
        bits_per_element: Storage bits per element (16 for bf16/fp16 arrays).
    """

    name: str
    numel: int
    bits_per_element: float

    def bits(self) -> float:
        """Return the storage size in bits (``numel * bits_per_element``)."""
        return self.numel * self.bits_per_element


def stored_bytes(tensors: Iterable[StoredTensor]) -> int:
    """Return the total stored size in bytes, rounded up.

    Args:
        tensors: Every stored array of the model.

    Returns:
        Total storage in bytes.

    Raises:
        ValueError: If an array is empty or has non-positive storage bits.
    """
    total_bits = 0.0
    for tensor in tensors:
        if tensor.numel < 0:
            raise ValueError(f"stored tensor {tensor.name!r} has negative numel {tensor.numel}")
        if tensor.bits_per_element <= 0:
            raise ValueError(
                f"stored tensor {tensor.name!r} has non-positive bits {tensor.bits_per_element}"
            )
        total_bits += tensor.bits()
    return math.ceil(total_bits / 8.0)


def reference_storage(name: str, numel: int) -> StoredTensor:
    """Return a 16-bit (bf16/fp16) stored array, the reference format.

    Args:
        name: Identifier of the array.
        numel: Number of elements.

    Returns:
        The 16-bit :class:`StoredTensor`.
    """
    return StoredTensor(name=name, numel=numel, bits_per_element=16.0)


def integer_group_storage(name: str, numel: int, bits: int, group_size: int) -> list[StoredTensor]:
    """Return the storage of a symmetric integer group-quantized array.

    Layout: ``bits`` per element plus one 16-bit scale per group of
    ``group_size`` elements; symmetric quantization needs no zero point.

    Args:
        name: Identifier of the array.
        numel: Number of logical elements.
        bits: Quantized bit width per element.
        group_size: Elements per scale.

    Returns:
        The stored arrays: the quantized payload and its scales.

    Raises:
        ValueError: If ``numel`` is negative or ``group_size`` is not positive.
    """
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    groups = math.ceil(numel / group_size) if numel else 0
    return [
        StoredTensor(name=f"{name}.q{bits}", numel=numel, bits_per_element=float(bits)),
        StoredTensor(name=f"{name}.scales", numel=groups, bits_per_element=16.0),
    ]


def vector_quant_storage(
    name: str, numel: int, dim: int, codebook_size: int, index_bits: int
) -> list[StoredTensor]:
    """Return the storage of a vector-quantized array (codebook plus indices).

    Args:
        name: Identifier of the array.
        numel: Number of logical elements.
        dim: Elements encoded by one index.
        codebook_size: Number of codewords.
        index_bits: Bits per index.

    Returns:
        The stored arrays: codebook, indices and a 16-bit scale per group.

    Raises:
        ValueError: If any argument is non-positive.
    """
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel}")
    if dim <= 0 or codebook_size <= 0 or index_bits <= 0:
        raise ValueError(
            f"dim, codebook_size and index_bits must be positive, got "
            f"{dim}, {codebook_size}, {index_bits}"
        )
    groups = math.ceil(numel / dim) if numel else 0
    return [
        StoredTensor(name=f"{name}.codebook", numel=codebook_size * dim, bits_per_element=16.0),
        StoredTensor(name=f"{name}.indices", numel=groups, bits_per_element=float(index_bits)),
        StoredTensor(name=f"{name}.scales", numel=groups, bits_per_element=16.0),
    ]
